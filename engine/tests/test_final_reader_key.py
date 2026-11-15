"""The sentence part of the pinned and cache keys is normalized (sentence_key): voice and Wispr Flow send
"Make it winter." or "make it winter " and must still hit the pinned "make it winter" of engine/demo_prompts.json,
with no Sonnet call. The content check and Sonnet still get the original text. No network."""

import json

import pytest

from commands.state import DEFAULT_BASE, SceneState
from engine.final_reader import (DEFAULT_DEMO_PROMPTS, CacheFile, Entry, FinalReader, FinalReaderState, Outcome,
                                 fingerprint, load, load_pinned, scene_record, sentence_key)
from commands.composer import compose

DEMO = json.loads(open(DEFAULT_DEMO_PROMPTS).read())
DEMO_ENTRIES = DEMO["entries"] if isinstance(DEMO, dict) else DEMO
WINTER = next(e for e in DEMO_ENTRIES if e["sentence"] == "make it winter"
              and e["prompt_before"] == compose(SceneState(DEFAULT_BASE).snapshot()))


class CountingChat:
    model = "fake-sonnet"

    def __init__(self):
        self.calls = []

    def complete(self, system, messages):
        self.calls.append(json.loads(messages[-1]["content"])["sentence"])
        op = {"op": "add", "label": "winter", "kind": "season", "phrase": "snow on the ground", "strength": 1.0}
        return json.dumps({"blocked": False, "operations": [op], "details": "fresh words"})


@pytest.mark.parametrize("said", ["Make it winter.", "make it winter!", "  Make it   winter  ", '"make it winter"',
                                  "“Make it winter.”", "make it winter"])
def test_spoken_forms_hit_the_pinned_demo_entry(said):
    chat = CountingChat()
    r = load("sonnet", chat=chat, cache_path=None, demo_prompts=DEFAULT_DEMO_PROMPTS)
    out = r(said, SceneState(DEFAULT_BASE).snapshot())
    assert out["prompt"] == WINTER["prompt"]
    assert chat.calls == [] and r.stats()["pinned_hits"] == 1


def test_an_extra_word_is_not_a_pinned_hit_and_sonnet_gets_the_original_text():
    chat = CountingChat()
    r = load("sonnet", chat=chat, cache_path=None, demo_prompts=DEFAULT_DEMO_PROMPTS)
    r("Make it winter please.", SceneState(DEFAULT_BASE).snapshot())
    assert chat.calls == ["Make it winter please."] and r.stats()["pinned_hits"] == 0


def test_every_demo_entry_is_pinned():
    assert len(load_pinned(DEFAULT_DEMO_PROMPTS)) == len(DEMO_ENTRIES)


def test_sentence_key_keeps_words_and_inner_punctuation():
    assert sentence_key("Make it winter.") == "make it winter"
    assert sentence_key("make it winter please") == "make it winter please"
    assert sentence_key("Don't stop the snow-covered road, please!") == "don't stop the snow-covered road, please"
    assert sentence_key("  'add a lion' ;") == "add a lion"


def test_the_disk_cache_is_shared_by_spoken_forms_and_old_raw_keys_still_load(tmp_path):
    scene = SceneState("a wide shot of a quiet harbour").snapshot()
    cache = tmp_path / "cache.json"
    chat = CountingChat()
    r = load("sonnet", chat=chat, cache_path=str(cache), demo_prompts=None)
    r("Make it winter.", scene)
    assert chat.calls == ["Make it winter."]
    assert [e["sentence"] for e in json.loads(cache.read_text())["entries"]] == ["make it winter"]
    # a restart: another spoken form is a cache hit, no call
    chat2 = CountingChat()
    r2 = load("sonnet", chat=chat2, cache_path=str(cache), demo_prompts=None)
    r2("make it winter!", scene)
    assert chat2.calls == [] and r2.stats()["cache_hits"] == 1
    # a file written before the keys were normalized (raw "Make it winter.") is normalized on load
    fp = fingerprint(r.system_prompt, "fake-sonnet")
    op = {"op": "add", "label": "winter", "kind": "season", "phrase": "snow on the ground", "strength": 1.0}
    cache.write_text(json.dumps({"fingerprint": fp, "entries": [
        {"prompt_before": compose(scene), "sentence": "Make it winter.", "outcome": "sonnet",
         "operations": [op], "prompt": "old raw words", "scene_before": scene_record(scene)}]}))
    loaded = CacheFile(cache, fp).load()
    assert list(loaded) == [(compose(scene), "make it winter")]
    chat3 = CountingChat()
    r3 = FinalReader(FinalReaderState.ON, chat=chat3, cache_file=CacheFile(cache, fp))
    assert r3("  make it   winter ", scene)["prompt"] == "old raw words"
    assert chat3.calls == []
    assert isinstance(loaded[(compose(scene), "make it winter")], Entry)
    assert loaded[(compose(scene), "make it winter")].outcome is Outcome.SONNET


@pytest.mark.parametrize("bad", [None, 3, ["make it winter"]])
def test_a_restart_keeps_the_saved_answers_when_one_cache_entry_is_broken(tmp_path, bad):
    scene = SceneState("a wide shot of a quiet harbour").snapshot()
    cache = tmp_path / "cache.json"
    r = load("sonnet", chat=CountingChat(), cache_path=str(cache), demo_prompts=None)
    r("make it winter", scene)
    data = json.loads(cache.read_text())
    data["entries"].insert(0, {**data["entries"][0], "sentence": bad})  # a hand-edited or damaged entry
    cache.write_text(json.dumps(data))
    chat = CountingChat()
    r2 = load("sonnet", chat=chat, cache_path=str(cache), demo_prompts=None)  # must not fail at startup
    r2("make it winter", scene)
    assert chat.calls == [] and r2.stats()["cache_hits"] == 1  # the good saved answer still replays

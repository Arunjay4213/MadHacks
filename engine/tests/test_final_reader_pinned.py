"""Pinned demo prompts (engine/demo_prompts.json): the demo sentences give the same edit and the same words
every time, with no Sonnet call. The file is written to a temporary folder here; no network."""

import copy
import json

from commands.composer import compose
from commands.service import CommandService
from commands.state import DEFAULT_BASE, SceneState
from engine.final_reader import CacheFile, FinalReader, FinalReaderState, fingerprint, load, load_pinned
from engine.reader import FinalReadingService

DRAFT = ["make it winter", "add a red car", "make it night", "add a lion", "sunny"]
OPS = {
    "make it winter": [{"op": "add", "label": "winter", "kind": "season", "phrase": "snow on the ground", "strength": 1.0}],
    "add a red car": [{"op": "add", "label": "car", "kind": "extra", "phrase": "a red car driving forward",
                       "strength": 1.0}],
    "make it night": [{"op": "add", "label": "night", "kind": "time", "phrase": "night time under a full moon",
                       "strength": 1.0}],
    "add a lion": [{"op": "add", "label": "lion", "kind": "extra", "phrase": "a lion standing in the snow",
                    "strength": 1.0}],
    # "sunny" is the keyword table's time change day: it replaces night
    "sunny": [{"op": "add", "label": "day", "kind": "time", "phrase": "bright sunshine", "strength": 1.0}],
}


def pinned_prompt(sentence):
    return f"PINNED words for {sentence}"


def demo_file(tmp_path, sentences=DRAFT, mutate=None):
    """The draft script as a demo prompts file: prompt_before is the composed prompt of the replayed scene."""
    s = SceneState(DEFAULT_BASE)
    entries = []
    for sentence in sentences:
        entries.append({"sentence": sentence, "prompt_before": compose(s.snapshot()),
                        "operations": copy.deepcopy(OPS[sentence]), "prompt": pinned_prompt(sentence)})
        for op in OPS[sentence]:
            s.apply(op)
    if mutate:
        mutate(entries)
    path = tmp_path / "demo_prompts.json"
    path.write_text(json.dumps({"base": DEFAULT_BASE, "entries": entries}, indent=1))
    return path


class CountingChat:
    model = "fake-sonnet"

    def __init__(self):
        self.calls = []

    def complete(self, system, messages):
        self.calls.append(json.loads(messages[-1]["content"])["sentence"])
        return json.dumps({"blocked": False, "understood": True, "new_scene": False,
                           "prompt": "A snowy harbour, fresh words.", "chips": ["winter"]})


def final(text):
    return {"type": "final", "text": text, "t_client": 1}


def test_the_draft_script_hits_every_pinned_entry_with_no_call(tmp_path):
    chat = CountingChat()
    r = load("sonnet", chat=chat, cache_path=None, demo_prompts=str(demo_file(tmp_path)))
    svc = FinalReadingService(CommandService(semantic_reader=lambda *a: None), r)
    svc.handle({"type": "tap", "action": "reset"})  # the script's reset line: no entry needed
    for sentence in DRAFT:
        out = svc.handle(final(sentence))
        assert out["steer"]["prompt"] == pinned_prompt(sentence)
        assert not out["state"]["unmatched"]
    assert chat.calls == []
    assert [c["label"] for c in out["state"]["changes"]] == ["winter", "car", "lion", "day"]
    stats = r.stats()
    assert (stats["pinned"], stats["pinned_hits"], stats["calls"]) == (5, 5, 0)
    # played a second time after a reset: the same words again
    svc.handle({"type": "tap", "action": "reset"})
    assert svc.handle(final("make it winter"))["steer"]["prompt"] == pinned_prompt("make it winter")


def test_pinned_entries_survive_a_fingerprint_change_and_win_over_the_disk_cache(tmp_path):
    demo = demo_file(tmp_path)
    before = compose(SceneState(DEFAULT_BASE).snapshot())
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"fingerprint": "an old prompt", "entries": []}))
    r = load("sonnet", chat=CountingChat(), cache_path=str(cache), demo_prompts=str(demo))
    assert r("make it winter", SceneState(DEFAULT_BASE).snapshot())["prompt"] == pinned_prompt("make it winter")
    # a disk cache entry for the same key, with the right fingerprint, does not replace the pinned one
    disk = CacheFile(cache, fingerprint(r.system_prompt, "fake-sonnet"))
    from engine.final_reader import Entry, Outcome, scene_record
    disk.save({(before, "make it winter"): Entry(Outcome.SONNET, tuple(OPS["make it winter"]), "disk words",
                                                 scene_record(SceneState(DEFAULT_BASE).snapshot()))})
    r2 = load("sonnet", chat=CountingChat(), cache_path=str(cache), demo_prompts=str(demo))
    assert r2("make it winter", SceneState(DEFAULT_BASE).snapshot())["prompt"] == pinned_prompt("make it winter")
    # pinned entries are never written into the disk cache
    r2("make it night", SceneState("a quiet harbour").snapshot())
    saved = [e["prompt"] for e in json.loads(cache.read_text())["entries"]]
    assert len(saved) == 2 and "disk words" in saved and "fresh words" in saved[1]
    assert pinned_prompt("make it winter") not in saved


def test_a_sentence_from_another_scene_is_not_a_pinned_hit(tmp_path):
    chat = CountingChat()
    r = FinalReader(FinalReaderState.ON, chat=chat, pinned=load_pinned(demo_file(tmp_path)))
    other = SceneState("a wide shot of a quiet harbour").snapshot()
    assert r("make it winter", other)["prompt"] != pinned_prompt("make it winter")
    assert chat.calls == ["make it winter"] and r.stats()["pinned_hits"] == 0


def test_bad_entries_are_skipped_and_never_crash(tmp_path):
    def break_some(entries):
        entries[1]["operations"] = [{"op": "add", "kind": "planet", "label": "x", "phrase": "p"}]  # not a kind
        entries[3]["prompt"] = ""
    pinned = load_pinned(demo_file(tmp_path, mutate=break_some))
    # entry 1 is skipped, so the scene the later entries assume never happens: only entry 0 is pinned
    assert [k[1] for k in pinned] == ["make it winter"]
    pinned = load_pinned(demo_file(tmp_path, mutate=lambda e: e[3].update(prompt="")))
    assert [k[1] for k in pinned] == ["make it winter", "add a red car", "make it night"]
    for text in ["not json", json.dumps({"base": DEFAULT_BASE, "entries": 3}), json.dumps([1, 2]), json.dumps(7)]:
        (tmp_path / "bad.json").write_text(text)
        assert load_pinned(tmp_path / "bad.json") == {}
    assert load_pinned(tmp_path / "missing.json") == {} and load_pinned(None) == {}


def test_the_list_form_with_note_keys_loads(tmp_path):
    entries = json.loads(demo_file(tmp_path).read_text())["entries"]
    for e in entries:
        e.update(canonical="notes", model="claude-sonnet-5-5", fingerprint="abc")
    path = tmp_path / "list.json"
    path.write_text(json.dumps(entries))
    assert [k[1] for k in load_pinned(path)] == DRAFT

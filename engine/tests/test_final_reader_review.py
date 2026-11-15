"""Fixes for open review comments on the final reader (issue #145), each with the real sentence it broke on.
A fake chat replays the reply shape Sonnet gave; the scene goes through the real CommandService."""

import json

from commands.composer import compose
from commands.service import CommandService
from commands.state import SceneState
from engine.final_reader import (CACHED_MAX_PHRASE, DEFAULT_DEMO_PROMPTS, FINAL_MAX_PHRASE, FinalReader,
                                 FinalReaderState, load_pinned, prompt_spec)
from engine.reader import FinalReadingService

WINTER = {"op": "add", "label": "winter", "kind": "season", "phrase": "snow on the ground", "strength": 1.0}
# Ishaan's measured desert world as one world phrase, as commands/world.py gives it with PR #156 (about 250 characters)
DESERT_250 = ("a long straight desert highway. Red sandstone mesas and golden sand dunes stretch to the horizon on both "
              "sides, dry scrub along the roadside, heat haze shimmering above the asphalt. Hot afternoon sun in a "
              "cloudless pale blue sky, strong hard shadows.")


class ReplayChat:
    model = "fake-sonnet"

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def complete(self, system, messages):
        user = json.loads(messages[-1]["content"])
        self.calls.append(user)
        return json.dumps(self.replies[user["sentence"]])


def reply(ops, details="soft light"):
    return {"blocked": False, "operations": ops, "details": details}


def final(text):
    return {"type": "final", "text": text, "t_client": 1}


def service(replies, pinned=None):
    chat = ReplayChat(replies)
    reader = FinalReader(FinalReaderState.ON, chat=chat, spec=prompt_spec(), pinned=pinned)
    return chat, reader, FinalReadingService(CommandService(semantic_reader=lambda *a: None), reader)


def test_the_phrase_limit_fits_a_measured_world_sentence():
    assert len(DESERT_250) >= 240 and FINAL_MAX_PHRASE >= 400 and CACHED_MAX_PHRASE > FINAL_MAX_PHRASE


def test_a_pinned_250_character_world_phrase_is_used(tmp_path):
    # PR #156 makes world phrases about 250 characters. The pinned entry must load and be used live, not refused
    # by the 120-character default of the cache check (Greptile on PR #120 :1056 and PR #125 demo_prompts.json:106).
    desert = {"op": "add", "label": "desert", "kind": "world", "phrase": DESERT_250, "strength": 1.0}
    path = tmp_path / "demo.json"
    path.write_text(json.dumps([{"sentence": "make it a desert", "prompt_before": compose(SceneState().snapshot()),
                                 "operations": [desert], "prompt": "PINNED desert"}]))
    pinned = load_pinned(path)
    assert len(pinned) == 1
    chat, reader, svc = service({}, pinned)
    out = svc.handle(final("make it a desert"))
    assert out["steer"]["prompt"] == "PINNED desert" and chat.calls == []
    assert reader.stats()["pinned_hits"] == 1


def test_a_fresh_250_character_world_phrase_is_accepted_and_reused_after_an_undo():
    desert = {"op": "add", "label": "dunes world", "kind": "world", "phrase": DESERT_250, "strength": 1.0}
    chat, reader, svc = service({"make it a desert": reply([desert], "warm sand tones")})
    first = svc.handle(final("make it a desert"))
    assert [c["label"] for c in first["state"]["changes"]] == ["dunes world"]
    calls = len(chat.calls)
    svc.handle({"type": "tap", "action": "undo"})
    again = svc.handle(final("make it a desert"))
    assert len(chat.calls) == calls and reader.stats()["cache_hits"] == 1
    assert again["steer"]["prompt"] == first["steer"]["prompt"]


def test_the_pinned_racing_car_of_the_demo_file_is_used():
    # The 7th meadow entry's phrase has 123 characters plus the lane cue.
    pinned = load_pinned(DEFAULT_DEMO_PROMPTS)
    meadow = list(pinned)[:7]  # the meadow chain, in the file's order
    chat, reader, svc = service({}, pinned)
    for before, sentence in meadow:
        assert svc.handle(final(sentence))["steer"]["prompt"] == pinned[(before, sentence)].prompt, sentence
    assert chat.calls == [] and "add a red car racing on a road" in [s for _, s in meadow]

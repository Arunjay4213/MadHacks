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
    # an unknown label, so canonical_world keeps the long phrase instead of the built-in desert phrase
    desert = {"op": "add", "label": "dunes world", "kind": "world", "phrase": DESERT_250, "strength": 1.0}
    path = tmp_path / "demo.json"
    path.write_text(json.dumps([{"sentence": "make it a desert", "prompt_before": compose(SceneState().snapshot()),
                                 "operations": [desert], "prompt": "PINNED desert"}]))
    pinned = load_pinned(path)
    assert [op["phrase"] for e in pinned.values() for op in e.operations] == [DESERT_250]
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


def test_a_base_edit_with_a_new_place_drops_the_copied_snow():
    # Greptile on PR #120 (final_reader.py:1049): the base edit copied winter, and world_first moved the desert
    # before the copy, so the desert kept its snow.
    sentence = "turn the tree into a lighthouse and make it a desert"
    _, _, svc = service({
        "make it winter": reply([WINTER]),
        sentence: reply([{"op": "edit_base", "base": "a wide shot of a single lighthouse"},
                         {"op": "add", "label": "desert", "kind": "world", "phrase": "a desert", "strength": 1.0}]),
    })
    svc.handle(final("make it winter"))
    out = svc.handle(final(sentence))
    assert [c["label"] for c in out["state"]["changes"]] == ["desert"]
    assert "snow" not in out["steer"]["prompt"]


def test_a_season_said_with_the_place_and_a_base_edit_stays():
    sentence = "turn the tree into a lighthouse and take me to a snowy desert"
    _, _, svc = service({
        "make it winter": reply([WINTER]),
        sentence: reply([{"op": "edit_base", "base": "a wide shot of a single lighthouse"}, WINTER,
                         {"op": "add", "label": "desert", "kind": "world", "phrase": "a desert", "strength": 1.0}]),
    })
    svc.handle(final("make it winter"))
    out = svc.handle(final(sentence))
    assert sorted(c["label"] for c in out["state"]["changes"]) == ["desert", "winter"]


LION = {"op": "add", "label": "lion", "kind": "extra",
        "phrase": "a large male lion standing in the foreground at the left of the frame, facing the camera", "strength": 1.0}


def test_a_pinned_removal_removes_the_change_it_names_not_the_one_with_its_old_id(tmp_path):
    # Greptile on PR #105 (final_reader.py:424): the file adds winter (c1) then a lion (c2) and pins "no more snow"
    # as remove c1. Live, the lion came first (c1), the prompt is the same (compose sorts by kind), and the pinned
    # removal took the lion away. The pinned entry is now used only on its own scene, with ids mapped.
    from commands.composer import compose
    from commands.state import SceneState
    from engine.final_reader import load_pinned
    s = SceneState()
    entries = []
    for sentence, ops in [("make it winter", [WINTER]), ("add a lion", [LION]),
                          ("no more snow", [{"op": "remove", "id": "c1"}])]:
        entries.append({"sentence": sentence, "prompt_before": compose(s.snapshot()), "operations": ops,
                        "prompt": f"PINNED {sentence}"})
        for op in ops:
            s.apply(op)
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(entries))
    chat = ReplayChat({"no more snow": reply([{"op": "remove", "id": "c2"}], "green grass")})
    reader = FinalReader(FinalReaderState.ON, chat=chat, spec=prompt_spec(), pinned=load_pinned(path))
    live = CommandService(semantic_reader=lambda *a: None)
    live.apply_final("add a lion", [LION], None)      # lion is c1
    live.apply_final("make it winter", [WINTER], None)  # winter is c2
    assert compose(live.snapshot()) == entries[2]["prompt_before"]
    res = reader("no more snow", live.snapshot())
    assert res["operations"] == [{"op": "remove", "id": "c2"}]
    out = live.apply_final("no more snow", res["operations"], None)
    assert [c["label"] for c in out["state"]["changes"]] == ["lion"]
    # in the file's own order the pinned entry is used, with its own ids
    same = SceneState()
    same.apply(WINTER)
    same.apply(LION)
    assert reader("no more snow", same.snapshot()) == {"operations": [{"op": "remove", "id": "c1"}],
                                                       "prompt": "PINNED no more snow"}


def test_a_cached_base_edit_does_not_bring_back_a_removed_chip():
    # Greptile on PR #87 (final_reader.py:307): a zero-strength chip is not in the composed prompt, so the key of a
    # cached base edit stayed the same after the chip was removed, and the saved adds brought it back.
    sentence = "turn the tree into a lighthouse"
    chat, reader, svc = service({
        "add a lion": reply([LION]),
        sentence: reply([{"op": "edit_base", "base": "a wide shot of a single lighthouse"}]),
    })
    svc.handle(final("add a lion"))
    lion = svc.state_fields()["changes"][0]["id"]
    svc._svc.apply_final("weaken the lion", [{"op": "set_strength", "id": lion, "strength": 0}], None)
    assert "lighthouse" in svc.handle(final(sentence))["steer"]["prompt"]
    svc.handle({"type": "tap", "action": "undo"})
    svc.handle({"type": "tap", "action": "remove", "id": lion})
    out = svc.handle(final(sentence))
    assert [c["label"] for c in out["state"]["changes"]] == []
    assert sum(1 for c in chat.calls if c["sentence"] == sentence) == 2


def test_a_vehicle_that_stands_or_comes_toward_the_camera_is_not_told_to_drive_away():
    # Greptile on PR #89 (final_reader.py:153).
    from engine.final_reader import FORWARD_MOTION, with_forward_motion
    for base in ["a red car parked beside the oak tree", "a car driving toward the camera",
                 "a wide shot of a taxi stopped at a red light", "an oncoming truck on a mountain road"]:
        assert with_forward_motion(base) == base, base
    assert with_forward_motion("a wide shot of a red car on a road").endswith(FORWARD_MOTION)


def test_a_cache_written_by_another_prompt_build_is_not_reused(tmp_path):
    # Greptile on PR #89 (final_reader.py:143): a prompt saved before the engine changed how it builds prompts
    # came back after a restart, because the fingerprint had only the system prompt and the model.
    from engine.final_reader import PROMPT_BUILD, CacheFile, Entry, Outcome, fingerprint, scene_record
    from commands.state import SceneState
    snap = SceneState().snapshot()
    old = CacheFile(tmp_path / "c.json", fingerprint("system", "m", "an older build"))
    old.save({("p", "make it winter"): Entry(Outcome.SONNET, (WINTER,), "old words", scene_record(snap))})
    assert CacheFile(tmp_path / "c.json", fingerprint("system", "m", "an older build")).load()
    assert CacheFile(tmp_path / "c.json", fingerprint("system", "m")).load() == {}
    assert fingerprint("system", "m") == fingerprint("system", "m", PROMPT_BUILD)


def test_a_slow_retry_keeps_the_usable_first_reply():
    # Greptile on PR #120 (final_reader.py:867): the first reply had valid operations but no usable details, the
    # retry ran past the wait, and the dictionary fallback was cached instead of the first reply's edit.
    import time

    class SlowSecond(ReplayChat):
        def complete(self, system, messages):
            if len(self.calls) == 1:
                time.sleep(0.6)
            return super().complete(system, messages)

    chat = SlowSecond({"make it winter": reply([WINTER], "the camera pans slowly")})
    reader = FinalReader(FinalReaderState.ON, chat=chat, spec=prompt_spec(), timeout_s=0.4)
    from commands.state import SceneState
    res = reader("make it winter", SceneState().snapshot())
    assert res is not None and res["operations"] == [WINTER]
    assert "snow on the ground" in res["prompt"] and reader.stats()["retries"] == 1

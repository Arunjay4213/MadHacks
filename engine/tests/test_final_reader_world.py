"""Places in the Sonnet final reader, as the commands/ owner defines them (PR #88).

"take me to the desert" or "make it a desert" is one singleton change of kind world: it replaces the earlier
place and keeps the car and every other change. Only an explicit "new scene of ..." starts over.
A fake chat client stands in for Sonnet (no network); the wrapper runs against the real CommandService.
"""

import json

import pytest

from commands.service import CommandService
from engine.final_reader import FinalReader, FinalReaderState, operations_instructions
from engine.reader import FinalReadingService


CAR = {"op": "add", "label": "car", "kind": "extra", "phrase": "a red car driving forward", "strength": 1.0}


def world(label, phrase):
    return {"op": "add", "label": label, "kind": "world", "phrase": phrase, "strength": 1.0}


def reply(ops=(), details=""):
    return json.dumps({"blocked": False, "operations": list(ops), "details": details})


class FakeChat:
    model = "fake-sonnet"

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def complete(self, system, messages):
        user = json.loads(messages[-1]["content"])
        self.calls.append((system, user))
        return self.replies.get(user["sentence"], reply())


# The details describe only the place and light: the engine puts the car phrase right after the subject.
SONNET = {
    "add a red car": reply([CAR], "a green meadow with soft grass and an oak tree, bright midday sun"),
    "take me to the desert": reply([world("desert", "a sandy desert with dunes")],
                                   "a long desert highway between sandstone mesas, heat haze, "
                                   "orange sunset sky"),
    "make it a desert": reply([world("desert", "a sandy desert")],
                              "a long desert highway between sandstone mesas, heat haze, "
                              "orange sunset sky"),
    "go to mars": reply([world("mars", "the red surface of mars")],
                        "a dusty road across red rocks and craters, a pale pink sky"),
    "new scene of a beach": reply([{"op": "new_scene", "base": "a wide shot of a sandy beach with gentle waves"}],
                                  "warm sand, turquoise water, a clear sky"),
}


def wrapper():
    chat = FakeChat(SONNET)
    svc = FinalReadingService(CommandService(semantic_reader=lambda *a: None),
                              FinalReader(FinalReaderState.ON, chat=chat))
    return svc, chat


def final(text):
    return {"type": "final", "text": text, "t_client": 1}


def scene(svc):
    return [(c["label"], c["kind"]) for c in svc._svc.snapshot()["changes"]]


def test_the_system_prompt_names_world_places_and_the_keyword_labels():
    system = operations_instructions()
    assert "<world|season|weather|time|extra>" in system and "ONE add of kind world" in system
    assert "desert" in system and "neon city" in system  # the known place labels of commands.world
    assert "day (time): " in system and "sunny" in system.split("day (time): ")[1].split("\n")[0]
    assert "new scene of a beach" in system and "turn the\n  tree into a lighthouse" in system


def test_night_then_sunny_leaves_one_time_change_day():
    # The model may call "sunny" weather; the keyword table says it is the time of day "day", which replaces night.
    chat = FakeChat({"make it night": reply([{"op": "add", "label": "night", "kind": "time",
                                              "phrase": "a dark night sky with stars", "strength": 1.0}], ""),
                     "sunny": reply([{"op": "add", "label": "sunny", "kind": "weather",
                                      "phrase": "bright sunshine and a clear blue sky", "strength": 1.0}], "")})
    svc = FinalReadingService(CommandService(semantic_reader=lambda *a: None),
                              FinalReader(FinalReaderState.ON, chat=chat))
    svc.handle(final("make it night"))
    out = svc.handle(final("sunny"))
    assert scene(svc) == [("day", "time")]
    assert [c["label"] for c in out["state"]["changes"]] == ["day"]
    assert "night" not in svc._svc.committed_prompt and "bright sunshine" in svc._svc.committed_prompt


@pytest.mark.parametrize("sentence", ["take me to the desert", "make it a desert"])
def test_a_place_keeps_the_car_and_adds_one_world_change(sentence):
    svc, _ = wrapper()
    svc.handle(final("add a red car"))
    out = svc.handle(final(sentence))
    assert scene(svc) == [("car", "extra"), ("desert", "world")]
    assert [c["label"] for c in out["state"]["changes"]] == ["car", "desert"]
    assert not out["state"]["unmatched"] and out["state"]["heard"] == sentence
    # the same desert as the keyword table's: his tested sentence, not the model's short phrase
    from commands.world import WORLDS
    desert = next(c for c in svc._svc.snapshot()["changes"] if c["kind"] == "world")
    assert desert["phrase"] == WORLDS["desert"]["phrase"]
    assert "desert highway" in out["steer"]["prompt"] and "red car" in out["steer"]["prompt"].lower()


def test_a_second_place_replaces_the_first():
    svc, _ = wrapper()
    svc.handle(final("add a red car"))
    svc.handle(final("take me to the desert"))
    out = svc.handle(final("go to mars"))
    assert scene(svc) == [("car", "extra"), ("mars", "world")]  # one world at a time, the car stays
    assert "Mars" in svc._svc.committed_prompt and "desert" not in svc._svc.committed_prompt
    assert "red rocks" in out["steer"]["prompt"]


def test_only_an_explicit_new_scene_starts_over():
    svc, _ = wrapper()
    svc.handle(final("add a red car"))
    svc.handle(final("take me to the desert"))
    out = svc.handle(final("new scene of a beach"))
    assert scene(svc) == [] and out["state"]["changes"] == []
    assert svc._svc.snapshot()["base"] == "a wide shot of a sandy beach with gentle waves"


def test_undo_after_a_place_goes_back_to_the_car_in_the_meadow():
    svc, chat = wrapper()
    car = svc.handle(final("add a red car"))
    svc.handle(final("take me to the desert"))
    n = len(chat.calls)
    back = svc.handle(final("undo"))  # the keyword table's undo: no Sonnet call
    assert len(chat.calls) == n
    assert scene(svc) == [("car", "extra")]
    assert back["steer"]["prompt"] == car["steer"]["prompt"]  # the same words as before: the same embedding
    again = svc.handle(final("undo"))
    assert scene(svc) == [] and again["state"]["changes"] == []


def test_with_commands_rewriter_the_details_are_checked_by_his_validator():
    from commands.composer import compose
    from engine.final_reader import prompt_spec
    spec = prompt_spec()
    assert spec.name == "commands.rewriter"
    details = "warm golden light, fine ripples across the ground, soft long shadows, photorealistic, sharp detail"
    chat = FakeChat({"take me to the desert": reply([world("desert", "a desert")], details),
                     "go to mars": reply([world("mars", "mars")], "the car drives away fast")})  # motion: rejected
    svc = FinalReadingService(CommandService(semantic_reader=lambda *a: None),
                              FinalReader(FinalReaderState.ON, chat=chat, spec=spec))
    out = svc.handle(final("take me to the desert"))
    canonical = compose(svc._svc.snapshot())
    assert out["steer"]["prompt"] == f"{canonical}; {details}"
    out = svc.handle(final("go to mars"))  # his validator refuses those details: the move to Mars still happens
    assert scene(svc) == [("mars", "world")]
    assert out["steer"]["prompt"] == compose(svc._svc.snapshot())


def test_one_reply_shape_and_forward_motion_with_commands_rewriter():
    # A vehicle base with commands.rewriter: the car still drives forward, and the model is told one reply shape.
    from commands import rewriter
    from engine.final_reader import FORWARD_MOTION, prompt_spec
    spec = prompt_spec()
    details = "warm golden light, fine ripples across the ground, soft long shadows, photorealistic, sharp detail"
    chat = FakeChat({"make it winter": reply([{"op": "add", "label": "winter", "kind": "season",
                                               "phrase": "snow on the ground", "strength": 1.0}], details)})
    r = FinalReader(FinalReaderState.ON, chat=chat, spec=spec)
    from commands.state import SceneState
    out = r("make it winter", SceneState("a wide shot of a red car on a mountain road").snapshot())
    assert out["prompt"].startswith("a wide shot of a red car on a mountain road, " + FORWARD_MOTION + "; ")
    assert out["prompt"].endswith(details)
    system = chat.calls[0][0]
    assert rewriter.SYSTEM_PROMPT in system
    wrap = system.index('For the "details" field, follow the rules below.')
    assert wrap < system.index(rewriter.SYSTEM_PROMPT)
    assert '{"blocked", "operations", "details"}' in system[wrap:system.index(rewriter.SYSTEM_PROMPT)]


def test_a_positioned_lion_keeps_its_exact_phrase_right_after_the_subject():
    # Picture check: a lion described only in the world text goes missing or hidden, and is gone after "sunny".
    # The positioned phrase from the add operation stays word for word, right after the subject, every time.
    lion_phrase = "a large male lion standing in the foreground at the left of the frame, facing the camera"
    chat = FakeChat({
        "add a lion": reply([{"op": "add", "label": "lion", "kind": "extra", "phrase": lion_phrase, "strength": 1.0}],
                            "a green meadow with soft grass, bright midday sun"),
        "sunny": reply([{"op": "add", "label": "day", "kind": "time", "phrase": "bright sunshine", "strength": 1.0}],
                       "a green meadow under a clear blue sky, warm bright sunshine"),  # Sonnet does not repeat it
    })
    svc = FinalReadingService(CommandService(semantic_reader=lambda *a: None),
                              FinalReader(FinalReaderState.ON, chat=chat))
    subject = "A wide shot of a meadow with a single oak tree, "
    for sentence in ["add a lion", "sunny"]:
        prompt = svc.handle(final(sentence))["steer"]["prompt"]
        assert prompt.startswith(subject + lion_phrase + ", "), prompt
        assert prompt.count("lion") == 1
    assert "size and position words" in chat.calls[0][0]

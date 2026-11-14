"""Places in the Sonnet final reader, as the commands/ owner defines them (PR #88).

"take me to the desert" or "make it a desert" is one singleton change of kind world: it replaces the earlier
place and keeps the car and every other change. Only an explicit "new scene of ..." starts over.
A fake chat client stands in for Sonnet (no network); the wrapper runs against the real CommandService.
The world tests need commands.state kind "world" and are skipped on a commands/ without it.
"""

import json

import pytest

from commands.service import CommandService
from commands.state import KINDS
from engine.final_reader import FinalReader, FinalReaderState, PlaceRule, operations_instructions, place_rule
from engine.reader import FinalReadingService, service_snapshot

HAS_WORLD = "world" in KINDS
needs_world = pytest.mark.skipif(not HAS_WORLD, reason="commands.state has no kind 'world' (PR #88 not merged)")

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


# Each reply follows the stand-in rule that every change of the new scene appears in its details.
SONNET = {
    "add a red car": reply([CAR], "a green meadow with soft grass and an oak tree, bright midday sun"),
    "take me to the desert": reply([world("desert", "a sandy desert with dunes")],
                                   "a long desert highway between sandstone mesas with a red car on it, heat haze, "
                                   "orange sunset sky"),
    "make it a desert": reply([world("desert", "a sandy desert")],
                              "a long desert highway between sandstone mesas with a red car on it, heat haze, "
                              "orange sunset sky"),
    "go to mars": reply([world("mars", "the red surface of mars")],
                        "a dusty road across red rocks and craters with a red car on it, a pale pink sky"),
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
    return [(c["label"], c["kind"]) for c in service_snapshot(svc._svc)["changes"]]


def test_place_rule_follows_the_kinds_of_commands_state():
    system = operations_instructions()
    if HAS_WORLD:
        assert place_rule() is PlaceRule.WORLD
        assert "<world|season|weather|time|extra>" in system and "ONE add of kind world" in system
        assert "desert" in system and "neon city" in system  # his known place labels
    else:
        assert place_rule() is PlaceRule.BASE_EDIT
        assert "world" not in system.split("Operations,")[1].split("Rules")[0]
    assert "new scene of a beach" in system and "turn the\n  tree into a lighthouse" in system


@needs_world
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
    desert = next(c for c in service_snapshot(svc._svc)["changes"] if c["kind"] == "world")
    assert desert["phrase"] == WORLDS["desert"]["phrase"]
    assert "desert highway" in out["steer"]["prompt"] and "red car" in out["steer"]["prompt"].lower()


@needs_world
def test_a_second_place_replaces_the_first():
    svc, _ = wrapper()
    svc.handle(final("add a red car"))
    svc.handle(final("take me to the desert"))
    out = svc.handle(final("go to mars"))
    assert scene(svc) == [("car", "extra"), ("mars", "world")]  # one world at a time, the car stays
    assert "Mars" in svc._svc.committed_prompt and "desert" not in svc._svc.committed_prompt
    assert "red rocks" in out["steer"]["prompt"]


@needs_world
def test_only_an_explicit_new_scene_starts_over():
    svc, _ = wrapper()
    svc.handle(final("add a red car"))
    svc.handle(final("take me to the desert"))
    out = svc.handle(final("new scene of a beach"))
    assert scene(svc) == [] and out["state"]["changes"] == []
    assert service_snapshot(svc._svc)["base"] == "a wide shot of a sandy beach with gentle waves"


@needs_world
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


@needs_world
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
    canonical = compose(service_snapshot(svc._svc))
    assert out["steer"]["prompt"] == f"{canonical}; {details}"
    out = svc.handle(final("go to mars"))  # his validator refuses those details: the move to Mars still happens
    assert scene(svc) == [("mars", "world")]
    assert out["steer"]["prompt"] == compose(service_snapshot(svc._svc))

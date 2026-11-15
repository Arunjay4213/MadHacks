"""The car template (engine/car_template.py) through the final reader and the real CommandService, on CPU.

Sentences are the demo's spoken sentences. A fake chat stands in for Sonnet where the template needs the model.
"""

import json

from commands.composer import compose
from commands.service import CommandService
from commands.state import SceneState
from engine import car_template as ct
from engine.final_reader import DEFAULT_DEMO_PROMPTS, FinalReader, FinalReaderState, load_pinned
from engine.reader import FinalReadingService

NEW_SCENE = "new scene of a red sports car driving along a country road"
HEAD = "A red sports car, seen from directly behind with its rear lights facing the camera, drives fast along "
CAMERA = ("The camera follows directly behind the car in a smooth, low tracking shot at the same speed, keeping the "
          "car centred in the frame while the white dashed centre line rushes toward the viewer.")
STYLE = "Cinematic, highly detailed, realistic, sharp focus, motion blur on the road."
CHAIN = [  # sentence -> world
    (NEW_SCENE, "country"), ("make it winter", "winter"), ("make it a desert", "desert"),
    ("make it night", "neon city"), ("make it autumn", "autumn"), ("make it sunset", "sunset"),
    ("make it rain", "rain"), ("take me to the beach", "ocean coast"),
]


class NoChat:
    model = "fake-sonnet"

    def __init__(self):
        self.calls = []

    def complete(self, system, messages):
        self.calls.append(json.loads(messages[-1]["content"]))
        raise AssertionError("a measured car world must not call the model")


def final(text):
    return {"type": "final", "text": text, "t_client": 1}


def service(chat):
    reader = FinalReader(FinalReaderState.ON, chat=chat)
    return reader, FinalReadingService(CommandService(semantic_reader=lambda *a: None), reader)


def test_the_new_car_scene_is_the_template_word_for_word():
    chat = NoChat()
    reader, svc = service(chat)
    out = svc.handle(final(NEW_SCENE))
    assert out["steer"]["prompt"] == (
        "A red sports car, seen from directly behind with its rear lights facing the camera, drives fast along a "
        "straight two-lane country road, moving forward away from the camera. The camera follows directly behind "
        "the car in a smooth, low tracking shot at the same speed, keeping the car centred in the frame while the "
        "white dashed centre line rushes toward the viewer. Green fields and rows of trees line both sides of the "
        "road under a clear blue summer sky. Bright midday sunshine, crisp shadows. Cinematic, highly detailed, "
        "realistic, sharp focus, motion blur on the road.")
    assert svc._svc.snapshot()["base"] == "a red sports car"
    assert out["state"]["changes"] == []  # the country start world is the bare vehicle scene
    assert chat.calls == [] and reader.stats()["template_hits"] == 1


def test_each_world_swaps_road_scene_and_light_and_keeps_subject_and_camera():
    reader, svc = service(NoChat())
    for sentence, name in CHAIN:
        out = svc.handle(final(sentence))
        world = ct.WORLDS[name]
        assert not out["state"]["unmatched"], sentence
        assert out["steer"]["prompt"] == (f"{HEAD}{world.road}, moving forward away from the camera. {CAMERA} "
                                          f"{world.scene} {world.light} {STYLE}"), sentence
        assert [c["label"] for c in out["state"]["changes"]] == ([] if name == "country" else [name]), sentence
    assert reader.stats()["calls"] == 0 and reader.stats()["template_hits"] == len(CHAIN)
    # an undo goes back to the earlier scene and its same words
    out = svc.handle({"type": "tap", "action": "undo"})
    assert out["steer"]["prompt"] == ct.world_prompt("a red sports car", ct.WORLDS["rain"])


def test_desert_in_the_car_chain_even_when_the_base_names_its_road():
    # The dictionary path (Sonnet timed out) keeps the whole sentence as the base; "make it a desert" still swaps the
    # world to Ishaan's desert template. The engine keeps the base as the bare subject instead.
    plain = CommandService(semantic_reader=lambda *a: None)
    plain.handle(final(NEW_SCENE))
    assert plain.snapshot()["base"] == "a red sports car driving along a country road"
    reader = FinalReader(FinalReaderState.ON, chat=NoChat())
    res = reader("make it a desert", plain.snapshot())
    assert res["operations"][0] == {"op": "new_scene", "base": "a red sports car"}
    assert res["prompt"] == ct.world_prompt("a red sports car", ct.WORLDS["desert"])
    out = plain.handle(final("make it a desert"))
    assert not out["state"]["unmatched"] and [c["label"] for c in out["state"]["changes"]] == ["desert"]
    assert out["steer"]["prompt"] == ct.world_prompt("a red sports car", ct.WORLDS["desert"])


def test_the_pinned_car_chain_is_what_the_live_reader_composes():
    entries = json.loads(open(DEFAULT_DEMO_PROMPTS).read())
    pinned = load_pinned(DEFAULT_DEMO_PROMPTS)
    assert len(pinned) == len(entries)
    keys = list(pinned)  # in file order; the first car chain starts with the new scene from the meadow
    start = keys.index((compose(SceneState().snapshot()), NEW_SCENE))
    car = [k for k in keys[start:] if k[1] in dict(CHAIN)][:len(CHAIN)]
    assert [s for _, s in car] == [s for s, _ in CHAIN]
    live = FinalReader(FinalReaderState.ON, chat=NoChat())
    svc = FinalReadingService(CommandService(semantic_reader=lambda *a: None), live)
    for (before, sentence), (_, name) in zip(car, CHAIN):
        assert before == compose(svc._svc.snapshot())
        out = svc.handle(final(sentence))
        assert out["steer"]["prompt"] == pinned[(before, sentence)].prompt == \
            ct.world_prompt("a red sports car", ct.WORLDS[name]), sentence


def test_the_camera_sentence_names_the_vehicle():
    # Greptile on PR #141: a motorcycle prompt told the camera to follow a car.
    prompt = ct.world_prompt("a black motorcycle", ct.WORLDS["desert"])
    assert "The camera follows directly behind the motorcycle in a smooth" in prompt
    assert "keeping the motorcycle centred" in prompt and " car" not in prompt
    assert ct.camera_sentence("a red sports car") == CAMERA  # Ishaan's sentence, exactly, for a car


def test_weakening_an_object_on_a_car_scene_changes_its_sentence():
    # Greptile on PR #141: the object sentence ignored strength, so a weakened lion read the same.
    snap = SceneState("a red sports car").snapshot()
    snap["changes"] = [{"id": "c1", "label": "lion", "kind": "extra", "phrase": "a large lion at the left", "strength": 0.3}]
    assert ct.scene_objects(snap) == ["a subtle hint of a large lion at the left"]
    snap["changes"][0]["strength"] = 1.3
    assert ct.scene_objects(snap) == ["an intense large lion at the left"]  # not "intense a large lion"


def test_removing_a_world_chip_goes_back_to_the_country_template():
    # Greptile on PR #141: the start world was a chip; removing it left the bare vehicle state with no template.
    _, svc = service(NoChat())
    start = svc.handle(final(NEW_SCENE))["steer"]["prompt"]
    winter = svc.handle(final("make it winter"))
    out = svc.handle({"type": "tap", "action": "remove", "id": winter["state"]["changes"][0]["id"]})
    assert out["steer"]["prompt"] == start == ct.world_prompt("a red sports car", ct.WORLDS["country"])
    svc.handle(final("make it winter"))
    assert svc.handle(final("take me back to the countryside"))["steer"]["prompt"] == start


class HangingChat:
    """Sonnet that does not answer within the engine's wait: every sentence times out."""

    model = "fake-sonnet"

    def __init__(self):
        import threading
        self.gate = threading.Event()

    def complete(self, system, messages):
        self.gate.wait(5)
        raise AssertionError("too late: the engine has already used the dictionary")


def test_after_a_sonnet_timeout_a_dog_on_the_car_scene_keeps_the_car_driving_forward():
    # Issue #237: the dictionary path sent "a red sports car; a dog standing nearby", so the car could reverse or the
    # shot changed. The user still wants the same forward-driving car on the same road, now with a dog. Since #271
    # "add a dog" is read whole by the keyword table (the fast path, no call); the result keeps the car template.
    chat = HangingChat()
    reader = FinalReader(FinalReaderState.ON, chat=chat, timeout_s=0.1)
    svc = FinalReadingService(CommandService(semantic_reader=lambda *a: None), reader)
    start = svc.handle(final(NEW_SCENE))["steer"]["prompt"]
    out = svc.handle(final("add a dog"))
    country = ct.WORLDS["country"]
    assert reader.stats()["calls"] == 0 and [c["label"] for c in out["state"]["changes"]] == ["dog"]
    assert out["steer"]["prompt"] == (f"{HEAD}{country.road}, moving forward away from the camera. {CAMERA} "
                                      f"A dog standing nearby. {country.scene} {country.light} {STYLE}")
    assert ct.check_prompt(out["steer"]["prompt"], "a red sports car", ["a dog standing nearby"])
    assert svc.handle({"type": "tap", "action": "undo"})["steer"]["prompt"] == start
    chat.gate.set()
    reader.stop()


def test_after_a_sonnet_timeout_a_scene_without_a_vehicle_keeps_the_composed_prompt():
    chat = HangingChat()
    reader = FinalReader(FinalReaderState.ON, chat=chat, timeout_s=0.1)
    plain = CommandService(semantic_reader=lambda *a: None)
    svc = FinalReadingService(CommandService(semantic_reader=lambda *a: None), reader)
    assert svc.handle(final("add a dog"))["steer"] == plain.handle(final("add a dog"))["steer"]
    chat.gate.set()
    reader.stop()


def test_undo_and_an_edit_on_a_car_scene_keeps_the_car_template():
    # Greptile on PR #238: an undo with an edit is read by the dictionary, which gave the short composed prompt.
    chat = HangingChat()
    reader = FinalReader(FinalReaderState.ON, chat=chat, timeout_s=0.1)
    svc = FinalReadingService(CommandService(semantic_reader=lambda *a: None), reader)
    svc.handle(final(NEW_SCENE))
    svc.handle(final("make it winter"))
    out = svc.handle(final("undo and add a dog"))
    country = ct.WORLDS["country"]
    assert [c["label"] for c in out["state"]["changes"]] == ["dog"]
    assert out["steer"]["prompt"] == (f"{HEAD}{country.road}, moving forward away from the camera. {CAMERA} "
                                      f"A dog standing nearby. {country.scene} {country.light} {STYLE}")
    chat.gate.set()
    reader.stop()


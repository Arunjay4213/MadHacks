"""A base that names its own place no longer blocks every later place change (Arunjay's live test, 2026-11-15 07:27).

On the meadow, "i want no grass. arid environment" was read as a base edit that wrote the place into the base. Every
later world change then failed with "base names its own world", there was no time to ask again, and the picture did
not change for three sentences. The place now becomes a new scene, and the objects stay."""

import json

from commands.service import CommandService
from engine.reader import FinalReadingService
from engine.tests.test_final_reader import FakeChat, reader

ARID = "a wide shot of an arid meadow with bare, dry cracked earth and a single bare oak tree with no leaves"
LION = {"op": "add", "label": "lion", "kind": "extra",
        "phrase": "a large male lion standing in the centre of the frame, facing the camera", "strength": 1.0}


def answer(ops, details="warm golden light, dry air"):
    return json.dumps({"blocked": False, "operations": ops, "details": details})


REPLIES = {
    "i want no grass. arid environment": answer([{"op": "edit_base", "base": ARID}]),
    "add a lion": answer([LION]),
    "now i want a savannah": answer([{"op": "add", "label": "savannah", "kind": "world",
                                      "phrase": "a wide golden savannah of tall dry grass dotted with acacia trees",
                                      "strength": 1.0}]),
    "i now want a city in the night with a car driving on the road": answer([
        {"op": "add", "label": "city", "kind": "world", "phrase": "a wide city street at night", "strength": 1.0},
        {"op": "add", "label": "night", "kind": "time", "phrase": "at night", "strength": 1.0},
        {"op": "add", "label": "car", "kind": "extra",
         "phrase": "a car driving on the road in the centre of the frame, large and clearly visible", "strength": 1.0}]),
    "take me to the forest": answer([{"op": "add", "label": "forest", "kind": "world", "phrase": "a forest.",
                                      "strength": 1.0}]),
}


def play(*sentences):
    svc = FinalReadingService(CommandService(semantic_reader=False), reader(FakeChat(REPLIES)))
    outs = [svc.handle({"type": "final", "text": t, "t_client": i + 1}) for i, t in enumerate(sentences)]
    return svc, outs


def test_a_new_place_on_a_base_that_names_its_own_place_is_a_new_scene_and_keeps_the_objects():
    svc, outs = play("i want no grass. arid environment", "add a lion", "now i want a savannah",
                     "i now want a city in the night with a car driving on the road", "take me to the forest")
    savannah, city, forest = outs[2], outs[3], outs[4]
    assert not savannah["state"]["unmatched"]
    # The oak scene's edited base keeps its subject; the meadow is replaced by the savannah.
    assert savannah["steer"]["prompt"].startswith("a wide shot of a single bare oak tree with no leaves; a large male lion")
    assert "golden savannah" in savannah["steer"]["prompt"] and "arid meadow" not in savannah["steer"]["prompt"]
    assert [c["label"] for c in savannah["state"]["changes"]] == ["lion", "savannah"]
    assert not city["state"]["unmatched"]
    assert "a wide city street at night" in city["steer"]["prompt"]
    assert "savannah" not in city["steer"]["prompt"] and "lion" in city["steer"]["prompt"]
    assert {c["label"] for c in city["state"]["changes"]} == {"lion", "neon city", "night", "car"}
    # A known world is written in its form for the scene: the car stays, so the forest keeps its road.
    assert not forest["state"]["unmatched"]
    assert "a narrow road through a dense green pine forest" in forest["steer"]["prompt"]
    assert "city" not in forest["steer"]["prompt"]
    # One undo goes back to the city.
    undo = svc.handle({"type": "tap", "action": "undo"})
    assert undo["steer"]["prompt"] == city["steer"]["prompt"]


def test_a_base_that_names_another_place_is_rebuilt_around_the_new_place():
    beach = "a wide shot of a quiet beach with a single palm tree"
    replies = {"new scene of a quiet beach": json.dumps({"blocked": False, "operations": [
                   {"op": "new_scene", "base": beach}], "details": ""}),
               "add a lion": answer([LION]), "now i want a savannah": REPLIES["now i want a savannah"]}
    svc = FinalReadingService(CommandService(semantic_reader=False), reader(FakeChat(replies)))
    for i, t in enumerate(["new scene of a quiet beach", "add a lion", "now i want a savannah"]):
        out = svc.handle({"type": "final", "text": t, "t_client": i + 1})
    assert not out["state"]["unmatched"]
    assert out["steer"]["prompt"].startswith("a wide shot; a large male lion")
    assert "golden savannah" in out["steer"]["prompt"] and "beach" not in out["steer"]["prompt"]
    assert [c["label"] for c in out["state"]["changes"]] == ["savannah", "lion"]


def test_a_mover_added_by_the_dictionary_on_the_car_scene_travels_with_the_car():
    from engine.final_reader import FinalReader, FinalReaderState
    chat = FakeChat({})  # every call fails: the dictionary reads the sentence
    svc = FinalReadingService(CommandService(semantic_reader=False), reader(chat))
    svc.handle({"type": "final", "text": "new scene of a red sports car driving along a country road", "t_client": 1})
    out = svc.handle({"type": "final", "text": "add an airplane", "t_client": 2})
    assert not out["state"]["unmatched"]
    assert "airplane, seen from behind, moving forward away from the camera in the same direction as the car" \
        in out["steer"]["prompt"]

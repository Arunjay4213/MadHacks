"""Pieces of the #272 stopgap that stay with the free-scene design (#271): a mover the dictionary adds on the car scene
travels with the car. The stuck-base sequence itself is in engine/tests/test_final_reader.py and acceptance/sentences.json;
the base rebuild (place_as_new_scene) went with the operations merge, since Sonnet now writes the whole prompt."""

from commands.service import CommandService
from engine.reader import FinalReadingService
from engine.tests.test_final_reader import FakeChat, reader


def test_a_mover_added_by_the_dictionary_on_the_car_scene_travels_with_the_car():
    chat = FakeChat({"add an airplane": RuntimeError("down")})  # the call fails: the dictionary reads the sentence
    svc = FinalReadingService(CommandService(semantic_reader=False), reader(chat))
    svc.handle({"type": "final", "text": "new scene of a red sports car driving along a country road", "t_client": 1})
    out = svc.handle({"type": "final", "text": "add an airplane", "t_client": 2})
    assert not out["state"]["unmatched"]
    assert "airplane, seen from behind, moving forward away from the camera in the same direction as the car" \
        in out["steer"]["prompt"]


def test_stationary_movers_keep_their_stillness():
    from engine.final_reader import with_heading
    assert with_heading("a boat anchored beside the road") == "a boat anchored beside the road"
    assert with_heading("a dog sleeping by the road") == "a dog sleeping by the road"
    assert with_heading("an airplane").endswith("in the same direction as the car")

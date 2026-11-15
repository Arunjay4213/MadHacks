"""Greptile P1 on #273 (comment 4577747209): a person add followed by another edit must not drop the other edit.

"add jensen huang next to the car and make it winter" on a composed scene used to take the engine's person
shortcut (people.whole_sentence), because the place pattern swallowed "and make it winter" as part of the spot.
Only Jensen was added and the winter edit was silently skipped.
"""

from commands.state import SceneState
from engine.people import PEOPLE, for_model, person_add, whole_sentence
from engine.tests.test_people import final, labels, reply, service

JENSEN = PEOPLE[0].description
SENTENCE = "add jensen huang next to the car and make it winter"
WINTER = (f"A wide shot of a snowy meadow in winter with a single oak tree, a large red car parked beside it, "
          f"{JENSEN} standing next to the car, large and clearly visible.")


def test_add_him_and_make_it_winter_applies_both_and_sends_no_name():
    sent = for_model(SENTENCE)
    chat, reader, svc = service({sent: reply(WINTER, ["winter", "red car"])})
    out = svc.handle(final(SENTENCE))
    # One Sonnet call read the whole sentence, so the winter edit is in the prompt.
    assert len(chat.sent) == 1 and reader.stats()["person_hits"] == 0
    assert out["steer"]["prompt"] == WINTER
    assert "winter" in labels(out) and "jensen huang" in labels(out)
    # The model sees his look, never his name.
    assert "jensen" not in chat.sent[0]["sentence"] and JENSEN in chat.sent[0]["sentence"]
    assert "make it winter" in chat.sent[0]["sentence"]
    assert chat.sent[0]["person_chip"] == "jensen huang"


def test_the_shortcut_is_only_for_a_sentence_that_is_just_the_add():
    meadow = SceneState().snapshot()
    for text in (SENTENCE, "put jensen next to the car then make it night", "add jensen beside the tree, also add a dog",
                 "add jensen huang next to the car and a dog"):
        assert whole_sentence(text, meadow) is None, text
    assert whole_sentence("add jensen huang next to the car", meadow)["phrase"].endswith(
        "standing next to the car, facing the camera")
    assert whole_sentence("add jensen huang next to the big red car", meadow)["phrase"].endswith(
        "standing next to the big red car, facing the camera")


def test_his_spot_never_carries_the_next_clause():
    # The engine's own add (used when the model gives nothing usable) places him next to the car, not
    # "next to the car and make it winter".
    phrase = person_add(SENTENCE, SceneState().snapshot())["phrase"]
    assert phrase == f"{JENSEN}, a large figure standing next to the car, facing the camera"

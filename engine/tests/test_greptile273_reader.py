"""Greptile on PR #273, engine/reader.py:106: "undo and add a dog" after two free-scene edits applied neither.

The user said two things: go back one step, then add a dog. Both must show: the chips and the prompt of the step
before, plus the dog. Runs the real CommandService and FinalReader with a fake Sonnet (no network), the path the live
engine takes with --reader commands --final-reader sonnet.
"""

from engine.tests.test_reader_compound import final, labels, reply, wrapper

DRY = "A meadow of dry yellow grass under a clear sky."
KITE = "A meadow of dry yellow grass under a clear sky, a red kite flying above it."
DOG = "A meadow of dry yellow grass under a clear sky, a brown dog running through the grass."

REPLIES = {
    "make the grass dry": reply(DRY, ["dry grass"]),
    "add a red kite over the meadow": reply(KITE, ["dry grass", "red kite"]),
    # Sonnet is asked only for the edit, on the scene after the undo (the dry meadow, without the kite)
    "add a dog": reply(DOG, ["dry grass", "dog"]),
}


def two_free_edits():
    svc, chat = wrapper(REPLIES)
    final(svc, "make the grass dry")
    out = final(svc, "add a red kite over the meadow", 2)
    assert labels(out) == ["dry grass", "red kite"] and svc.committed_prompt == KITE
    return svc, chat


def test_undo_and_add_on_a_free_scene_applies_both():
    svc, chat = two_free_edits()
    out = final(svc, "undo and add a dog", 3)
    assert out["state"]["unmatched"] is False
    assert labels(out) == ["dry grass", "dog"]  # the kite is undone, the dog is added
    assert chat.sentences[-1] == "add a dog"  # the edit, read on the scene the undo went back to
    assert out["steer"]["prompt"] == DOG and svc.committed_prompt == DOG
    # the whole sentence is one undo step: undo goes back to the kite scene the user saw before it
    back = final(svc, "undo", 4)
    assert labels(back) == ["dry grass", "red kite"] and back["steer"]["prompt"] == KITE


def test_undo_and_a_blocked_add_changes_nothing():
    svc, _ = two_free_edits()
    out = final(svc, "undo and add a naked man", 3)
    assert out["steer"] is None and out["state"]["unmatched"] is True
    assert labels(out) == ["dry grass", "red kite"] and svc.committed_prompt == KITE  # the undo is refused too


def test_undo_and_an_add_the_model_blocks_changes_nothing():
    svc, chat = two_free_edits()
    out = final(svc, "undo and add a man with no clothes", 3)  # not in REPLIES: the fake model blocks it
    assert chat.sentences[-1] == "add a man with no clothes"
    assert out["steer"] is None and out["state"]["unmatched"] is True
    assert labels(out) == ["dry grass", "red kite"] and svc.committed_prompt == KITE

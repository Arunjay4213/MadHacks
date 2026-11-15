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


KITE_DOG = "A meadow of dry yellow grass under a clear sky, a red kite flying above it and a brown dog below."


def test_undo_of_a_reset_and_add_on_a_free_scene_applies_both():
    """Greptile on PR #280: after a free edit and a reset, the keyword table thought it read "undo and add a dog"
    whole (its trial undo has no history, so it added the dog to the reset scene); the real undo went back to the
    free scene, which refused the add."""
    svc, chat = wrapper({**REPLIES, "add a dog": reply(KITE_DOG, ["dry grass", "red kite", "dog"])})
    final(svc, "make the grass dry")
    final(svc, "add a red kite over the meadow", 2)
    assert labels(final(svc, "reset", 3)) == []
    out = final(svc, "undo and add a dog", 4)
    assert out["state"]["unmatched"] is False
    assert labels(out) == ["dry grass", "red kite", "dog"]  # back to the kite scene, plus the dog
    assert out["steer"]["prompt"] == KITE_DOG and chat.sentences[-1] == "add a dog"


def test_a_saved_answer_for_the_whole_sentence_does_not_block_the_fix():
    """Greptile on PR #280: an older build asked Sonnet with the whole sentence and saved that answer. It must not
    stop the controls and the edits from both applying."""
    svc, chat = two_free_edits()
    fr = svc._final_reader
    chat.replies["undo and add a dog"] = reply("A stale prompt for the whole sentence.", ["stale"])
    fr.ask("undo and add a dog", svc._svc.snapshot(), svc.committed_prompt)  # saved the way the old path did
    out = final(svc, "undo and add a dog", 3)
    assert labels(out) == ["dry grass", "dog"] and svc.committed_prompt == DOG


MEADOW_DOG = "A green meadow under a clear sky, a brown dog sitting in the grass in the middle of the view."


def test_reset_and_add_a_dog_uses_the_model_prompt_it_waited_for():
    """Greptile on PR #281 (P2): a mixed sentence waited for Sonnet and threw its prompt away. "add a dog" is not a
    fast-path add (#281: the table's object phrase carries no size or place), so after the reset Sonnet writes it,
    and its prompt is what the user sees."""
    svc, chat = wrapper({"add a dog": reply(MEADOW_DOG, ["dog"])})
    final(svc, "make it winter")
    out = final(svc, "reset and add a dog", 2)
    assert chat.sentences == ["add a dog"]
    assert labels(out) == ["dog"] and out["steer"]["prompt"] == MEADOW_DOG


def test_a_mix_the_dictionary_reads_whole_makes_no_call():
    """Greptile on PR #281 (P2): an undo after an edit runs in the dictionary, where it is said, so a model call
    would only be a content check; every word is read by the table, so there is nothing to check."""
    svc, chat = wrapper({})
    final(svc, "make it winter")
    out = final(svc, "add a dog and undo", 2)
    assert chat.sentences == []
    assert out["state"]["unmatched"] is False and labels(out) == ["winter"]

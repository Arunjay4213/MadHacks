"""FinalReadingService (engine/reader.py) on compound sentences and pace words, with real sentences.

The wrapper runs against the real commands.service.CommandService and a real FinalReader with a fake chat
client (no network), the path the live engine takes with --reader commands --final-reader sonnet.
"""

import json

from commands.service import CommandService
from engine.final_reader import FinalReader, FinalReaderState
from engine.reader import FinalReadingService, SentenceKind, sentence_kind

NIGHT = {"op": "add", "label": "night", "kind": "time", "phrase": "at night", "strength": 1.0}
WINTER = {"op": "add", "label": "winter", "kind": "season", "phrase": "snow on the ground", "strength": 1.0}
DETAILS = "a quiet meadow with tall grass and wildflowers, soft light"


def reply(ops=(), details=DETAILS, blocked=False):
    return json.dumps({"blocked": blocked, "operations": list(ops), "details": details})


class FakeChat:
    """Answers `replies[sentence]`; anything else gets the model's block, like Sonnet on an unsafe request."""

    model = "fake-sonnet"

    def __init__(self, replies):
        self.replies = replies
        self.sentences = []

    def complete(self, system, messages):
        sentence = json.loads(messages[-1]["content"])["sentence"]
        self.sentences.append(sentence)
        return self.replies.get(sentence, reply(blocked=True))


def wrapper(replies):
    chat = FakeChat(replies)
    return FinalReadingService(CommandService(), FinalReader(FinalReaderState.ON, chat=chat)), chat


def final(svc, text, t=1):
    return svc.handle({"type": "final", "text": text, "t_client": t})


def labels(out):
    return [c["label"] for c in out["state"]["changes"]]


def assert_blocked(svc, out):
    """No steer, unmatched, and nothing of the request in the scene or the prompt (the words line still shows
    what was heard, as for any blocked sentence)."""
    assert out["steer"] is None
    assert out["state"]["unmatched"] is True
    assert "naked" not in json.dumps(out["state"]["changes"])
    assert "naked" not in (svc.committed_prompt or "")


def test_reset_and_an_unsafe_add_is_blocked():
    svc, _ = wrapper({"make it winter": reply([WINTER])})
    final(svc, "make it winter")
    out = final(svc, "reset and add a naked man", 2)
    assert_blocked(svc, out)
    assert labels(out) == ["winter"]  # nothing applied: the reset is refused with the rest of the sentence


def test_undo_and_an_unsafe_add_is_blocked():
    svc, _ = wrapper({"make it winter": reply([WINTER])})
    final(svc, "make it winter")
    out = final(svc, "undo and add a naked man", 2)
    assert_blocked(svc, out)
    assert labels(out) == ["winter"]


def test_an_edit_and_an_unsafe_add_is_blocked():
    svc, _ = wrapper({"make it winter": reply([WINTER])})
    out = final(svc, "make it winter and add a naked man")
    assert_blocked(svc, out)
    assert labels(out) == []


def test_a_mixed_sentence_the_model_blocks_is_blocked():
    """No word from the word list: only the model's own check catches it, so the mixed sentence must reach it."""
    svc, chat = wrapper({"make it winter": reply([WINTER])})
    final(svc, "make it winter")
    out = final(svc, "reset and add a man with no clothes", 2)
    assert chat.sentences[-1] == "reset and add a man with no clothes"
    assert out["steer"] is None and out["state"]["unmatched"] is True
    assert labels(out) == ["winter"]


def test_a_clean_compound_with_reset_still_works():
    sentence = "reset and make it night"
    svc, chat = wrapper({"make it winter": reply([WINTER]), sentence: reply([NIGHT])})
    final(svc, "make it winter")
    out = final(svc, sentence, 2)
    assert sentence in chat.sentences  # checked like any other sentence
    assert out["state"]["unmatched"] is False
    assert labels(out) == ["night"]  # the reset cleared winter, then night was added
    assert "night" in out["steer"]["prompt"] and "snow" not in out["steer"]["prompt"]


def test_undo_and_reset_alone_make_no_call():
    svc, chat = wrapper({"make it winter": reply([WINTER])})
    final(svc, "make it winter")
    assert labels(final(svc, "undo", 2)) == []
    final(svc, "make it winter", 3)
    assert labels(final(svc, "reset", 4)) == []
    assert labels(final(svc, "undo and undo", 5)) == []
    assert chat.sentences == ["make it winter"]  # the second "make it winter" is a cache hit


def test_sentence_kinds():
    snapshot = CommandService().snapshot()
    assert sentence_kind("undo", snapshot) is SentenceKind.CONTROL
    assert sentence_kind("reset", snapshot) is SentenceKind.CONTROL
    assert sentence_kind("reset and add a naked man", snapshot) is SentenceKind.MIXED
    assert sentence_kind("make it winter and add a naked man", snapshot) is SentenceKind.EDIT
    assert sentence_kind("make it winter", snapshot) is SentenceKind.EDIT


def test_pace_words_set_the_ramp_of_a_final_reader_result():
    """Greptile on PR #87, engine/reader.py:131: "slowly" and "suddenly" must survive the Sonnet path."""
    for sentence, ramp in (("slowly make it night", 12), ("suddenly make it night", 1), ("make it night", 1)):
        svc, chat = wrapper({sentence: reply([NIGHT])})
        out = final(svc, sentence)
        assert chat.sentences == [sentence]
        assert labels(out) == ["night"]
        assert out["steer"]["ramp_chunks"] == ramp, sentence


BEACH = {"op": "new_scene", "base": "a sandy beach with gentle waves"}
DESERT = {"op": "new_scene", "base": "a wide sand desert with dunes"}


def test_start_over_with_a_scene_reaches_the_final_reader():
    """Greptile on PR #105, engine/final_reader.py:262: the keyword table reads "start over with a beach" as a
    plain reset, which would drop the beach; the final reader makes it a new scene."""
    for sentence, op, word in (("start over with a beach", BEACH, "beach"), ("new scene of a desert", DESERT, "desert"),
                               ("reset to a beach", BEACH, "beach"),
                               ("reset and start over with a beach", BEACH, "beach"),
                               ("undo and start over with a beach", BEACH, "beach")):
        svc, chat = wrapper({"make it winter": reply([WINTER]), sentence: reply([op])})
        final(svc, "make it winter")
        out = final(svc, sentence, 2)
        assert chat.sentences[-1] == sentence
        assert out["state"]["unmatched"] is False and labels(out) == []
        assert word in out["steer"]["prompt"] and "snow" not in out["steer"]["prompt"]


def test_start_over_with_unsafe_content_is_blocked():
    svc, _ = wrapper({"make it winter": reply([WINTER])})
    final(svc, "make it winter")
    out = final(svc, "start over with a naked man", 2)
    assert_blocked(svc, out)
    assert labels(out) == ["winter"]


def test_a_bare_start_over_keeps_the_fast_path():
    for sentence in ("start over", "reset", "start over please", "reset everything", "start over from scratch",
                     "reset immediately", "slowly start over", "reset right now", "reset and start over"):
        svc, chat = wrapper({"make it winter": reply([WINTER])})
        final(svc, "make it winter")
        out = final(svc, sentence, 2)
        assert chat.sentences == ["make it winter"], sentence  # no call
        assert labels(out) == [], sentence

import time

import torch

from engine.commands import CommandBook, CommandStage
from steering import Controller, Steerer

TARGET = {"prompt": "p", "ramp_chunks": 2, "provisional": False, "cap": 1.0}


def setup():
    book = CommandBook()
    steerer = Steerer(torch.zeros(1, 2, 4))
    embs = {"p": torch.ones(1, 2, 4), "q": -torch.ones(1, 2, 4)}
    ctl = Controller(steerer, embs.__getitem__)
    return book, steerer, ctl


def test_full_path_stamps_in_order():
    book, steerer, ctl = setup()
    rec = book.heard("control", "make it p", 123, time.monotonic_ns())
    book.decided(rec, TARGET)
    time.sleep(0.002)
    result = book.embedded(rec, lambda: ctl.handle(TARGET))
    assert result["applied"] and rec.stage is CommandStage.EMBEDDED
    book.take_cond(steerer.get_cond, chunk=7)
    assert rec.stage is CommandStage.SCHEDULED and rec.applies_from_chunk == 7
    book.chunk_generated(6)  # an earlier chunk does not count
    assert rec.stage is CommandStage.SCHEDULED
    book.chunk_generated(7)
    book.chunk_shown(7)
    assert rec.stage is CommandStage.SHOWN
    stamps = rec.stamps_ms()
    assert list(stamps) == ["heard", "decided", "embedded", "generated", "shown"]
    assert stamps["heard"] == 0 and stamps["embedded"] >= 2
    assert list(stamps.values()) == sorted(stamps.values())
    fields = book.latest_fields()
    assert fields["t_client"] == 123 and fields["applies_from_chunk"] == 7 and fields["heard"] == "make it p"
    assert fields["unmatched"] is False


def test_get_cond_before_set_target_does_not_count():
    book, steerer, ctl = setup()
    rec = book.heard("control", "x", 1, time.monotonic_ns())
    book.decided(rec, TARGET)
    book.take_cond(steerer.get_cond, chunk=1)  # target not set yet
    assert rec.applies_from_chunk is None
    book.embedded(rec, lambda: ctl.handle(TARGET))
    book.take_cond(steerer.get_cond, chunk=2)
    assert rec.applies_from_chunk == 2


def test_older_command_finishing_late_is_superseded():
    book, steerer, ctl = setup()
    old = book.heard("control", "old", 1, time.monotonic_ns())
    book.decided(old, TARGET)
    new = book.heard("control", "new", 2, time.monotonic_ns())
    q = {**TARGET, "prompt": "q"}
    book.decided(new, q)
    book.embedded(new, lambda: ctl.handle(q))  # new one's embedding was cached, so it lands first
    called = []
    assert book.embedded(old, lambda: called.append(1)) is None
    assert old.stage is CommandStage.SUPERSEDED and not called
    assert book.latest() is new


def test_newer_target_supersedes_an_unscheduled_older_one():
    book, steerer, ctl = setup()
    a = book.heard("control", "a", 1, time.monotonic_ns())
    book.decided(a, TARGET)
    book.embedded(a, lambda: ctl.handle(TARGET))
    b = book.heard("control", "b", 2, time.monotonic_ns())
    q = {**TARGET, "prompt": "q"}
    book.decided(b, q)
    book.embedded(b, lambda: ctl.handle(q))
    assert a.stage is CommandStage.SUPERSEDED and b.stage is CommandStage.EMBEDDED


def test_unchanged_target():
    book, steerer, ctl = setup()
    for _ in range(2):
        rec = book.heard("control", "p", 1, time.monotonic_ns())
        book.decided(rec, TARGET)
        book.embedded(rec, lambda: ctl.handle(TARGET))
    assert rec.stage is CommandStage.UNCHANGED


def test_unmatched_final_is_reported_and_partial_is_not():
    book, *_ = setup()
    p = book.heard("control", "make it", 1, time.monotonic_ns())
    book.no_operation(p, "partial", report=False)
    assert book.latest() is None and p.stage is CommandStage.DROPPED
    f = book.heard("control", "gibberish", 2, time.monotonic_ns())
    book.no_operation(f, "no match", report=True)
    assert book.latest_fields()["unmatched"] is True and book.latest_fields()["heard"] == "gibberish"


def test_reset_drops_in_flight():
    book, steerer, ctl = setup()
    rec = book.heard("control", "x", 1, time.monotonic_ns())
    book.decided(rec, TARGET)
    replaced = []
    book.reset(lambda: replaced.append(True))
    assert rec.stage is CommandStage.DROPPED and replaced == [True]
    assert book.embedded(rec, lambda: ctl.handle(TARGET)) is None


def test_no_command_fields():
    book, *_ = setup()
    assert book.latest_fields() == {"stamps": {}, "t_client": None, "applies_from_chunk": None,
                                    "heard": None, "keyword": None, "unmatched": False}

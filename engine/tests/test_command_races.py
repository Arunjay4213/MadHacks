"""Races between a command arriving and the engine's background threads.

1. A reset tap the reader rejects, or whose prompt fails to encode, must not hold later commands forever.
2. A chunk finishing between a command being heard and the prefetch hold. Before the fix hold() ran before
   book.heard(), so a chunk_done in between saw nothing in flight and moved HELD back to SAVING or READY while
   the command was in flight.
"""

import threading
import time

import pytest

from engine.commands import CommandStage
from engine.engine import Engine, EngineConfig, PrefetchState
from engine.lifecycle import Lifecycle

WINTER = {"prompt": "a meadow in deep winter", "ramp_chunks": 1, "provisional": False, "cap": 1.0}
FRESH = {"prompt": "fresh scene", "ramp_chunks": 1, "provisional": False, "cap": 1.0}
READ_RESET_S = 0.3  # the reader takes this long to read a reset tap (the real one may block about 0.4 s)


def wait(pred, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.005)
    return False


class SlowResetReader:
    """A reset tap takes READ_RESET_S in the reader and steers to a fresh scene."""

    committed_prompt = None

    def handle(self, msg):
        if msg["type"] == "tap" and msg["action"] == "reset":
            time.sleep(READ_RESET_S)
            return {"steer": dict(FRESH), "state": self.state_fields()}
        return {"steer": None, "state": self.state_fields()}

    def state_fields(self):
        return {"changes": [], "heard": None, "keyword": None, "unmatched": False}

    def likely_prompts(self):
        return [FRESH["prompt"], WINTER["prompt"]]


@pytest.fixture
def engine():
    cfg = EngineConfig(backend="fake", width=64, height=48, backend_options=dict(
        gen_ms=40, decode_ms=5, encode_ms=5, frames_per_chunk=3, fps=50))
    e = Engine(cfg, lambda b: None, lambda m: None, reader=SlowResetReader())
    e.start()
    assert wait(lambda: e.lifecycle.state is Lifecycle.GENERATING)
    assert wait(lambda: e.chunk_shown >= 2)
    e.encoder.encode(WINTER["prompt"])  # cached, so a later target is ready at once
    yield e
    e.stop()


def test_a_rejected_reset_tap_does_not_hold_later_commands(engine):
    """The reset's ticket is cancelled when the reader rejects the tap, so a target heard after it still applies."""
    def reject(msg):
        raise ValueError("no")
    engine.reader.handle = reject
    engine.handle_control({"type": "tap", "action": "reset", "t_client": 100}, time.monotonic_ns()).result(5)
    target = engine.handle_target(dict(WINTER), time.monotonic_ns())
    assert wait(lambda: target.stage is CommandStage.SHOWN, timeout=3), f"target stuck at {target.stage}"


def _chunk_done_before(engine, method_name):
    """Run one generation-thread chunk_done (with enough idle time for a turn) right before engine.book.<method>."""
    real = getattr(engine.book, method_name)

    def wrapped(*args, **kwargs):
        engine.prefetch.chunk_done(10_000, engine.book.in_flight, 1.0)
        return real(*args, **kwargs)
    return wrapped


def test_a_chunk_between_hearing_and_holding_does_not_release_the_hold():
    cfg = EngineConfig(backend="fake", width=64, height=48, backend_options=dict(
        gen_ms=40, decode_ms=5, encode_ms=5, frames_per_chunk=3, fps=50))
    gate = threading.Event()

    class BlockingReader(SlowResetReader):
        def handle(self, msg):
            gate.wait(5)  # keep the command in flight while the test looks at the pacer
            return {"steer": None, "state": self.state_fields()}

    e = Engine(cfg, lambda b: None, lambda m: None, reader=BlockingReader())
    e.start()
    try:
        assert wait(lambda: e.lifecycle.state is Lifecycle.GENERATING)
        e.book.heard = _chunk_done_before(e, "heard")
        e.handle_control({"type": "final", "text": "make it winter", "t_client": 1}, time.monotonic_ns())
        assert e.book.in_flight()
        assert e.prefetch.state is PrefetchState.HELD
    finally:
        gate.set()
        e.stop()


def test_a_reset_whose_prompt_fails_to_encode_does_not_hold_later_commands(engine):
    """Greptile on PR #130: encoding the reset's prompt raised after its HEARD ticket existed, and the ticket was
    never cancelled, so every later command waited for it forever."""
    real_encode = engine.encoder.encode

    def encode(prompt):
        if prompt == FRESH["prompt"]:
            raise RuntimeError("CUDA out of memory")
        return real_encode(prompt)

    engine.encoder.encode = encode
    reset = engine.handle_control({"type": "tap", "action": "reset", "t_client": 100}, time.monotonic_ns())
    with pytest.raises(RuntimeError):
        reset.result(5)
    target = engine.handle_target(dict(WINTER), time.monotonic_ns())
    assert wait(lambda: target.stage is CommandStage.SHOWN, timeout=3), f"target stuck at {target.stage}"
    reset_rec = next(r for r in engine.book._records if r.t_client == 100)
    assert reset_rec.stage is CommandStage.DROPPED  # not left in flight, which would hold prefetch forever

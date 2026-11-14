"""The engine with the fake backend, fast settings, no server."""

import threading
import time

import pytest

from contracts.validate import errors
from engine.commands import CommandStage
from engine.engine import Engine, EngineConfig
from engine.framing import unpack_frame
from engine.lifecycle import Lifecycle
from engine.messages import BadMessage


def wait(pred, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def engine():
    frames, states = [], []
    lock = threading.Lock()

    def on_frame(b):
        with lock:
            frames.append(unpack_frame(b)[:2])

    cfg = EngineConfig(backend="fake", width=64, height=48, backend_options=dict(
        gen_ms=20, decode_ms=5, encode_ms=30, frames_per_chunk=3, fps=50))
    e = Engine(cfg, on_frame, states.append)
    e.frames, e.states = frames, states
    e.start()
    assert wait(lambda: e.lifecycle.state is Lifecycle.GENERATING)
    yield e
    e.stop()
    assert e.lifecycle.state is Lifecycle.STOPPED


def test_streams_and_applies_a_command(engine):
    assert wait(lambda: len(engine.frames) > 10)
    assert engine.frames[0] == (0, 0)  # first chunk has 1 frame
    with pytest.raises(BadMessage):
        engine.handle_control({"type": "final", "text": "x"}, time.monotonic_ns())
    engine.handle_control({"type": "final", "text": "make it winter", "t_client": 5}, time.monotonic_ns())
    rec = engine.book.latest()
    assert wait(lambda: rec.stage is CommandStage.SHOWN)
    assert wait(lambda: any(s["t_client"] == 5 and "shown" in s["stamps"] for s in engine.states))
    assert all(not errors(4, s) for s in engine.states)
    chunks = [s["chunk"] for s in engine.states]
    assert chunks == sorted(set(chunks))
    order = list(engine.frames)
    assert order == sorted(order)


def test_reset(engine):
    assert wait(lambda: len(engine.frames) > 5)
    done = engine.request_reset()
    assert done.wait(5)
    assert engine.lifecycle.state is Lifecycle.GENERATING
    n = len(engine.frames)
    assert wait(lambda: len(engine.frames) > n + 5)

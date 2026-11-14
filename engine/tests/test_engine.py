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
    engine.handle_control({"type": "final", "text": "make it winter", "t_client": 5}, time.monotonic_ns()).result(5)
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
    done = engine.reset(time.monotonic_ns()).result(5)
    assert done.wait(5)
    assert engine.lifecycle.state is Lifecycle.GENERATING
    n = len(engine.frames)
    assert wait(lambda: len(engine.frames) > n + 5)


class SeedingReader:
    """A reader whose reset tap returns a steer, like the commands/ reader will."""

    committed_prompt = "committed scene"

    def __init__(self):
        self.seen = []

    def handle(self, msg):
        self.seen.append(msg)
        if msg["type"] == "tap" and msg["action"] == "reset":
            return {"steer": {"prompt": "fresh scene", "ramp_chunks": 4, "provisional": False, "cap": 1.0},
                    "state": self.state_fields()}
        if msg["type"] == "final" and msg["text"] == "reject me":
            raise ValueError("no thanks")
        return {"steer": None, "state": self.state_fields()}

    def state_fields(self):
        return {"changes": [{"id": "c1", "label": "winter", "strength": 0.5}], "heard": "h", "keyword": "winter",
                "unmatched": False}

    def likely_prompts(self):
        return ["likely one", "likely two"]


def test_reader_api_reset_seed_and_errors():
    states, reader = [], SeedingReader()
    cfg = EngineConfig(backend="fake", width=64, height=48, backend_options=dict(
        gen_ms=20, decode_ms=5, encode_ms=10, frames_per_chunk=3, fps=50))
    e = Engine(cfg, lambda b: None, states.append, reader=reader)
    e.start()
    try:
        assert wait(lambda: e.lifecycle.state is Lifecycle.GENERATING)
        assert e.base_prompt == "committed scene"
        assert e.encoder.cache_size >= 3  # starting prompt and both likely prompts
        replies = []
        assert e.handle_control({"type": "final", "text": "reject me", "t_client": 1}, time.monotonic_ns(),
                                replies.append).result(5) is None
        assert replies and "no thanks" in replies[0]["error"]
        assert e.book.latest() is None  # a rejected message is not the most recent command
        done = e.handle_control({"type": "tap", "action": "reset", "t_client": 9}, time.monotonic_ns()).result(5)
        assert done.wait(5)
        rec = e.book.latest()
        assert rec.target["prompt"] == "fresh scene"
        assert wait(lambda: rec.stage is CommandStage.SHOWN)
        assert e.steerer.progress == 1.0  # started directly on the new prompt, no ramp
        assert e.controller.committed_prompt == "fresh scene"  # through Controller.reset
        assert wait(lambda: any(s["t_client"] == 9 and "shown" in s["stamps"] for s in states))
        last = states[-1]
        assert last["keyword"] == "winter" and last["changes"][0]["strength"] == 0.5 and not errors(4, last)
    finally:
        e.stop()



def _fast(**fail):
    return EngineConfig(backend="fake", width=64, height=48, backend_options=dict(
        gen_ms=20, decode_ms=5, encode_ms=10, frames_per_chunk=3, fps=50, **fail))


@pytest.mark.parametrize("fail_in", ["generate", "decode"])
def test_recovers_in_process_from_one_crash(fail_in):
    frames, states_seen = [], []
    e = Engine(_fast(fail_on_chunk=15, fail_in=fail_in), lambda b: frames.append(unpack_frame(b)[:2]), lambda m: None)
    fatal = []
    e.on_fatal = fatal.append
    real = e.lifecycle.transition

    def spy(new, reason=""):
        states_seen.append(new)
        real(new, reason)

    e.lifecycle.transition = spy
    e.start()
    try:
        assert wait(lambda: e.stats()["recovery"]["last"] is not None)
        backend = e.backend
        n = len(frames)
        assert wait(lambda: len(frames) > n + 20)  # video keeps playing after the recovery
        s = e.stats()
        assert s["state"] == "GENERATING" and s["failure"] is None and not fatal
        assert s["recovery"]["count"] == 1 and s["video"] == 1
        assert s["recovery"]["last"]["resumed_ms"] < 10_000
        assert e.backend is backend  # the model was not reloaded
        assert states_seen.count(Lifecycle.RECOVERING) == 1
        assert states_seen[states_seen.index(Lifecycle.RECOVERING) + 1] is Lifecycle.GENERATING
        assert e.steerer.progress == 1.0 and e.controller.committed_prompt == e.base_prompt
        # commands work again
        e.handle_control({"type": "final", "text": "after the crash", "t_client": 3}, time.monotonic_ns()).result(5)
        rec = e.book.latest()
        assert wait(lambda: rec.stage is CommandStage.SHOWN)
    finally:
        e.stop()


def test_repeated_crashes_fail_and_call_on_fatal():
    e = Engine(_fast(fail_on_chunk=5, fail_count=100), lambda b: None, lambda m: None)
    fatal = []
    e.on_fatal = fatal.append
    e.start()
    try:
        assert wait(lambda: e.lifecycle.state is Lifecycle.FAILED)
        assert len(fatal) == 1 and "3 crashes within 60 s" in fatal[0]
        assert e.stats()["recovery"]["count"] == 2
    finally:
        e.stop()
    assert e.lifecycle.state is Lifecycle.STOPPED


def test_unmatched_final_is_the_most_recent_command():
    """Contract 4 echoes the most recent command's t_client, also when the reader found no operation for it."""
    from commands.service import CommandService

    states = []
    e = Engine(_fast(), lambda b: None, states.append, reader=CommandService())
    e.start()
    try:
        assert wait(lambda: e.lifecycle.state is Lifecycle.GENERATING)
        e.handle_control({"type": "final", "text": "make it winter", "t_client": 101}, time.monotonic_ns()).result(5)
        winter = e.book.latest()
        assert wait(lambda: winter.stage is CommandStage.SHOWN)
        e.handle_control({"type": "final", "text": "please sing a song", "t_client": 202}, time.monotonic_ns()).result(5)
        assert wait(lambda: any(s["heard"] == "please sing a song" for s in states))
        n = len(states)
        assert wait(lambda: len(states) > n + 2)
        last = states[-1]
        assert not errors(4, last)
        assert last["heard"] == "please sing a song" and last["unmatched"] is True and last["keyword"] is None
        assert last["t_client"] == 202
        assert last["stamps"] == {"heard": 0}  # no operation: no decided, embedded, generated or shown
        assert last["applies_from_chunk"] is None
        day = e.book.latest()
        assert day.stage is CommandStage.DROPPED and day.t_client == 202
        # A partial with no steer is not a command: it does not replace the most recent one.
        e.handle_control({"type": "partial", "text": "make it", "t_client": 303}, time.monotonic_ns()).result(5)
        assert e.book.latest() is day
        # A tap the reader has no operation for is a command too: contract 4 then carries its (absent) t_client.
        e.handle_control({"type": "tap", "action": "remove", "id": "nope"}, time.monotonic_ns()).result(5)
        tap = e.book.latest()
        assert tap is not day and tap.stage is CommandStage.DROPPED and tap.t_client is None
        assert e.book.latest_fields() == {"stamps": {"heard": 0}, "t_client": None, "applies_from_chunk": None}
    finally:
        e.stop()

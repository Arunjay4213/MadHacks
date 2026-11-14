"""A command sent right after a reset tap applies to the new video, and a command sent right before one is dropped.

The reported bug: a reset tap and then "make it winter" a few ms later. The final was applied to the old video's
steerer, scheduled on an old-video chunk, and then the reset dropped it and restarted the steerer without it, so
contract 4 kept winter's t_client with no applies_from_chunk (or no `shown`) forever, and the new video had no winter.
"""

import time

import pytest

from contracts.validate import errors
from engine.commands import STAMP_NAMES, CommandStage
from engine.engine import Engine, EngineConfig
from engine.lifecycle import Lifecycle

WINTER = {"prompt": "a meadow in deep winter", "ramp_chunks": 2, "provisional": False, "cap": 1.0}
FRESH = {"prompt": "fresh scene", "ramp_chunks": 2, "provisional": False, "cap": 1.0}


def wait(pred, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.005)
    return False


class Reader:
    """Like commands/: a reset tap gives a steer to a fresh scene, and "make it winter" gives a final steer."""

    committed_prompt = None

    def handle(self, msg):
        if msg["type"] == "tap" and msg["action"] == "reset":
            return {"steer": dict(FRESH), "state": self.state_fields()}
        if msg["type"] == "final" and msg["text"] == "make it winter":
            return {"steer": dict(WINTER), "state": self.state_fields()}
        return {"steer": None, "state": self.state_fields()}

    def state_fields(self):
        return {"changes": [], "heard": None, "keyword": None, "unmatched": False}

    def likely_prompts(self):
        return [FRESH["prompt"], WINTER["prompt"]]  # both cached, as the real reader's likely prompts are


@pytest.fixture
def engine():
    states = []
    # Slow chunks: the final is embedded long before the generation thread reaches the chunk boundary where the
    # reset runs, which is exactly the window the bug needs.
    cfg = EngineConfig(backend="fake", width=64, height=48, backend_options=dict(
        gen_ms=150, decode_ms=5, encode_ms=5, frames_per_chunk=3, fps=50))
    e = Engine(cfg, lambda b: None, states.append, reader=Reader())
    e.states = states
    e.start()
    assert wait(lambda: e.lifecycle.state is Lifecycle.GENERATING)
    assert wait(lambda: e.chunk_shown >= 2)
    yield e
    e.stop()


@pytest.mark.parametrize("delay_ms", [0, 5, 20, 50])
def test_final_right_after_reset_applies_to_the_new_video(engine, delay_ms):
    video_before = engine.video
    reset_done = engine.handle_control({"type": "tap", "action": "reset", "t_client": 100}, time.monotonic_ns())
    time.sleep(delay_ms / 1e3)
    engine.handle_control({"type": "final", "text": "make it winter", "t_client": 200}, time.monotonic_ns()).result(5)
    assert reset_done.result(5).wait(5)
    winter = engine.book.latest()
    assert winter.text == "make it winter"
    assert wait(lambda: winter.stage is CommandStage.SHOWN), f"winter stuck at {winter.stage}"
    assert engine.video == video_before + 1
    assert engine.controller.committed_prompt == WINTER["prompt"]  # the new video runs on winter
    assert wait(lambda: engine.states[-1]["t_client"] == 200 and len(engine.states[-1]["stamps"]) == 5)
    last = engine.states[-1]
    assert last["applies_from_chunk"] is not None and tuple(last["stamps"]) == STAMP_NAMES
    stamps = list(last["stamps"].values())
    assert stamps == sorted(stamps) and not errors(4, last)


def test_final_right_before_reset_is_dropped_and_the_reset_wins(engine):
    engine.handle_control({"type": "final", "text": "make it winter", "t_client": 200}, time.monotonic_ns())
    reset_done = engine.handle_control({"type": "tap", "action": "reset", "t_client": 300}, time.monotonic_ns())
    assert reset_done.result(5).wait(5)
    reset = engine.book.latest()
    assert reset.t_client == 300
    winter = next(r for r in engine.book._records if r.text == "make it winter" and r.t_client == 200)
    assert winter.stage in (CommandStage.DROPPED, CommandStage.SUPERSEDED)
    assert wait(lambda: reset.stage is CommandStage.SHOWN)
    assert engine.controller.committed_prompt == FRESH["prompt"]
    assert wait(lambda: engine.states[-1]["t_client"] == 300 and len(engine.states[-1]["stamps"]) == 5)

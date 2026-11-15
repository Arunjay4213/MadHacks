"""--flush-on-cut and --interrupt-on-steer: the engine side of a faster steer cut, on the fake backend.

The latency itself is measured by engine/tools/cut_latency.py (a few minutes); a short run of it checks the
direction (marked tools, so run with pytest -m tools), and the rest checks correctness: what is dropped, what is stopped, the stamps, and the cut's dissolve.
"""

import threading
import time
from io import BytesIO

import types

import numpy as np
import pytest
from PIL import Image

import engine.engine as engine_module
from engine.backends import causal_forcing as cf
from engine.backends.fake import ChunkInterrupted, FakeBackend
from engine.commands import CommandStage
from engine.engine import (DissolveFix, Engine, EngineConfig, FlushOnCut, InFlight, SteerInterrupt,
                           dissolve_length)
from engine.framing import unpack_frame
from engine.lifecycle import Lifecycle
from engine.server import parse_args
from engine.tests.test_engine import wait
from engine.tools import cut_latency

# A chunk of 4 frames lasts 100 ms at 40 fps; a normal chunk is 3 passes of 27 ms (81 ms), about the live ratio.
PASSES = dict(pass_ms=27, denoise_passes=2, cut_passes=2, defer_context=True, decode_ms=9, encode_ms=2,
              frames_per_chunk=4, fps=40)
PROMPTS = ["a meadow in deep winter snow", "a desert of red dunes at sunset", "a meadow at night under a full moon"]
WINTER = {"prompt": "a meadow in deep winter snow", "ramp_chunks": 1, "provisional": False, "cap": 1.0}


class Run:
    """An engine on the fake backend that keeps every frame it sends: (chunk, index, jpeg)."""

    def __init__(self, monkeypatch, flush="off", interrupt="off", backend=None, reanchor="on",
                 reanchor_options=None, **backend_options):
        self.sent: list[tuple[int, int, bytes]] = []
        self.reasons: dict[int, str | None] = {}  # chunk number -> re-anchor reason; a rerun overwrites it
        self.taken: tuple[int, str | None] | None = None  # the chunk that took its cond last
        self.stopped: list[tuple[int, str | None]] = []  # chunks the backend stopped
        cfg = EngineConfig(backend="fake", width=64, height=48, record_dir=None, reanchor=reanchor,
                           reanchor_options=reanchor_options or {"refresh_every_chunks": "none"},
                           backend_options={**PASSES, **backend_options},
                           flush_on_cut=flush, interrupt_on_steer=interrupt)
        self.e = Engine(cfg, self._publish, lambda m: None)
        take_cond = self.e.book.take_cond

        def noting(get_cond, chunk):
            out = take_cond(get_cond, chunk)
            self.taken = (chunk, None if out[2] is None else out[2].reason)
            self.reasons[chunk] = self.taken[1]
            return out

        self.e.book.take_cond = noting
        if backend is not None:
            monkeypatch.setattr(engine_module, "load_backend",
                                lambda name, width, height, steps, device, decode_device, **opts:
                                backend(width, height, steps, **opts))
        self.e.start()
        assert wait(lambda: self.e.lifecycle.state is Lifecycle.GENERATING)
        generate = self.e.backend.generate_chunk

        def watched(c):
            try:
                return generate(c)
            except ChunkInterrupted:
                self.stopped.append(self.taken)
                raise

        self.e.backend.generate_chunk = watched
        assert wait(lambda: self.e.chunk_shown >= 4)
        self.stalls0 = self.e.stats()["stalls"]  # the stream's start may stall once; count only what comes after

    @property
    def new_stalls(self) -> int:
        return self.e.stats()["stalls"] - self.stalls0

    def _publish(self, message: bytes) -> None:
        self.sent.append(unpack_frame(message))

    def steer(self, prompt: str | None = None):
        """One steer cut: a ramp-1 command once the policy's gap is met; a new prompt each time unless given."""
        e = self.e
        self.steers = getattr(self, "steers", 0) + 1
        prompt = prompt or PROMPTS[self.steers % len(PROMPTS)]
        e.encoder.encode(prompt)
        assert wait(lambda: e.policy.chunks_since_anchor >= e.reanchor_config.min_gap_chunks)
        rec = e.handle_target({**WINTER, "prompt": prompt}, time.monotonic_ns())
        assert wait(lambda: rec.stage is CommandStage.SHOWN)
        return rec

    def stop(self):
        self.e.stop()


@pytest.fixture
def runs(monkeypatch):
    made = []

    def make(*args, **kwargs):
        r = Run(monkeypatch, *args, **kwargs)
        made.append(r)
        return r

    yield make
    for r in made:
        r.stop()


# --- options -------------------------------------------------------------------------------------------------------

def test_both_are_off_by_default():
    assert (EngineConfig().flush_on_cut, EngineConfig().interrupt_on_steer) == ("off", "off")
    args = parse_args([])
    assert (args.flush_on_cut, args.interrupt_on_steer) == ("off", "off")
    args = parse_args(["--flush-on-cut", "on", "--interrupt-on-steer", "on"])
    assert (args.flush_on_cut, args.interrupt_on_steer) == ("on", "on")


@pytest.mark.parametrize("field", ["flush_on_cut", "interrupt_on_steer"])
def test_a_bad_value_is_refused(field):
    with pytest.raises(ValueError, match=field):
        Engine(EngineConfig(**{field: "yes"}), lambda b: None, lambda m: None)


def test_interrupt_needs_reanchor(runs):
    r = runs(interrupt="on", reanchor="off")
    assert r.e.steer_interrupt is SteerInterrupt.UNSUPPORTED
    assert r.e.stats()["cut"]["interrupt"] == "unsupported"


def test_interrupt_needs_a_backend_that_can_stop(runs):
    class NoInterrupt(FakeBackend):
        interrupt = None

    r = runs(interrupt="on", backend=NoInterrupt)
    assert r.e.steer_interrupt is SteerInterrupt.UNSUPPORTED


def test_off_drops_and_stops_nothing(runs):
    r = runs()
    assert (r.e.flush_on_cut, r.e.steer_interrupt) == (FlushOnCut.OFF, SteerInterrupt.OFF)
    for _ in range(2):
        r.steer()
    cut = r.e.stats()["cut"]
    assert (cut["flushes"], cut["flushed_frames"], cut["interrupted_chunks"]) == (0, 0, 0)


# --- the fake backend's passes -------------------------------------------------------------------------------------

def fake(**options) -> FakeBackend:
    return FakeBackend(16, 16, 0, **{**PASSES, "pass_ms": 20, **options})


def cond():
    return fake().encode_prompt("x")


@pytest.mark.parametrize("case", ["first chunk", "re-anchor", "refresh"])
def test_fake_never_stops_a_first_chunk_or_a_re_anchor(case):
    b = fake()
    if case != "first chunk":
        b.generate_chunk(cond())
        b.reanchor(1.0 if case == "re-anchor" else 0.5, source="last" if case == "re-anchor" else "clean")
    timer = threading.Timer(0.01, b.interrupt)  # during the chunk's first pass
    timer.start()
    out = b.generate_chunk(cond())
    assert out["number"] == (0 if case == "first chunk" else 1) and b.interrupted_chunks == 0


# --- the dissolve after a flush ------------------------------------------------------------------------------------

def decoder(n: int, start: np.ndarray):
    d = object.__new__(cf._Decoder)
    d.cut_fade_frames, d._fade_from, d._fade_done = n, start, 0
    return d


def test_dissolve_fix_turns_the_decoders_dissolve_into_one_from_the_shown_frame():
    rng = np.random.default_rng(0)
    shown, decoded = (rng.integers(0, 256, (6, 5, 3), dtype=np.uint8) for _ in range(2))
    new = [rng.integers(0, 256, (4, 6, 5, 3), dtype=np.uint8) for _ in range(3)]
    from_decoded, from_shown = decoder(8, decoded), decoder(8, shown)
    fix = DissolveFix(video=0, shown=shown, decoded=decoded, length=8)
    for chunk in new:  # the real decoder's dissolve runs across chunk boundaries; so does the fix
        moved = fix.apply(from_decoded._fade(chunk) if from_decoded._fade_from is not None else chunk)
        want = from_shown._fade(chunk) if from_shown._fade_from is not None else chunk
        assert np.abs(moved.astype(int) - want.astype(int)).max() <= 1  # two roundings instead of one
    assert fix.finished


def test_the_fake_dissolve_is_the_real_decoders():
    rng = np.random.default_rng(1)
    start, frames = rng.integers(0, 256, (6, 5, 3), dtype=np.uint8), rng.integers(0, 256, (4, 6, 5, 3), dtype=np.uint8)
    b = fake(cut_fade_frames=8)
    b._fade_from, b._fade_done = start, 0
    assert np.array_equal(b._fade(frames), decoder(8, start)._fade(frames))


def test_dissolve_length_reads_the_latents_mark():
    class Real:
        cut_fade_frames = 8

    assert "fade" in {f.name for f in cf.dataclasses.fields(cf.ChunkLatents)}
    assert dissolve_length(Real(), types.SimpleNamespace(fade=True)) == 8  # the real backend's ChunkLatents
    assert dissolve_length(Real(), types.SimpleNamespace(fade=False)) == 0
    assert dissolve_length(Real(), {"fade": True}) == 8
    assert dissolve_length(Real(), {"fade": False}) == 0
    assert dissolve_length(object(), {"fade": True}) == 0


class FlatFake(FakeBackend):
    """Flat frames whose grey level follows the frame's number in the video (the old scene) or is 230 from a
    re-anchor chunk on (the new scene), so a test can read back from the JPEGs which old frame a dissolve started
    from."""

    new_scene = False

    def _draw(self, latents):
        n = latents["frames"]
        self.new_scene = self.new_scene or latents["anchor"] is not None
        frames = np.empty((n, self.height, self.width, 3), dtype=np.uint8)
        for i in range(n):
            frames[i] = 230 if self.new_scene else 20 + (3 * self._frame_phase) % 150
            self._frame_phase += 1
        return frames


def grey(jpeg: bytes) -> float:
    return float(np.asarray(Image.open(BytesIO(jpeg)).convert("L"), dtype=np.float32).mean())


def test_after_a_flush_the_cut_dissolves_from_the_frame_last_shown(runs):
    r = runs(flush="on", backend=FlatFake, cut_fade_frames=8)
    rec = r.steer()
    cut = rec.applies_from_chunk
    assert r.reasons[cut] == "steer" and r.e.stats()["cut"]["flushed_frames"] > 0
    assert wait(lambda: sum(1 for c, _, _ in r.sent if c > cut) >= 4)
    first = next(k for k, (c, i, _) in enumerate(r.sent) if c == cut)
    shown = grey(r.sent[first - 1][2])
    levels = [grey(j) for _, _, j in r.sent[first:first + 8]]
    want = [(1 - a) * shown + a * 230 for a in ((k + 1) / 9 for k in range(8))]
    assert max(abs(x - y) for x, y in zip(levels, want)) < 2.5, (shown, levels, want)


# --- flush -----------------------------------------------------------------------------------------------------------

def test_flush_drops_only_old_frames_and_never_the_cut(runs):
    r = runs(flush="on")
    recs = [r.steer() for _ in range(2)]
    assert wait(lambda: max(c for c, _, _ in r.sent) > recs[-1].applies_from_chunk + 2)
    s = r.e.stats()
    assert s["cut"]["flushes"] == 2 and s["cut"]["flushed_frames"] > 0 and r.new_stalls == 0
    chunks = [c for c, _, _ in r.sent]
    assert chunks == sorted(chunks)  # never back to an older chunk
    for rec in recs:
        cut = rec.applies_from_chunk
        assert r.reasons[cut] == "steer"
        assert [i for c, i, _ in r.sent if c == cut] == [0, 1, 2, 3]  # the cut is shown whole
        st = rec.stamps_ms()
        assert st["generated"] <= st["shown"]
    assert s["frames_ahead"] >= 0


def test_flush_keeps_the_tail_of_an_older_video():
    e = Engine(EngineConfig(backend="fake", flush_on_cut="on"), lambda b: None, lambda m: None)
    meta = lambda video, chunk: type("M", (), {"video": video, "chunk": chunk, "frames": 4, "est_frames": 4})()
    frame = np.zeros((2, 2, 3), np.uint8)
    for item in [(0, b"", meta(0, 9), frame), (1, b"", meta(0, 9), frame), (0, b"", meta(1, 10), frame),
                 (1, b"", meta(1, 10), frame)]:
        e._playout_queue.put(item)
    e._gate.add(4)
    e._last_taken = (1, frame)
    shown = e._flush_playout(meta(1, 11))
    assert shown is frame
    left = [(m.video, m.chunk, i) for i, _, m, _ in list(e._playout_queue.queue)]
    assert left == [(0, 9, 0), (0, 9, 1)] and e.flushed_frames == 2 and e._gate.ahead == 2


def test_a_refresh_never_flushes(runs):
    r = runs(flush="on", reanchor_options={"refresh_every_chunks": 10, "min_gap_chunks": 2})
    assert wait(lambda: r.e.stats()["reanchor"]["count"] >= 2)
    assert r.e.stats()["cut"]["flushes"] == 0


# --- interrupt -------------------------------------------------------------------------------------------------------

def test_interrupt_stops_the_old_chunk_and_the_cut_takes_its_number(runs):
    r = runs(interrupt="on")
    recs = [r.steer() for _ in range(4)]
    assert wait(lambda: max(c for c, _, _ in r.sent) > recs[-1].applies_from_chunk + 2)
    s = r.e.stats()
    assert s["cut"]["interrupted_chunks"] >= 1 and r.new_stalls == 0
    counts: dict[int, list[int]] = {}
    for c, i, _ in r.sent:
        counts.setdefault(c, []).append(i)
    numbers = sorted(counts)
    # Every chunk number is shown exactly once and whole: a stopped chunk is never shown and leaves no gap.
    assert numbers == list(range(numbers[0], numbers[-1] + 1))
    assert all(v == [0, 1, 2, 3] for c, v in counts.items() if numbers[0] < c < numbers[-1])
    assert s["chunk_generated"] >= numbers[-1]
    for rec in recs:
        assert r.reasons[rec.applies_from_chunk] == "steer"
        st = rec.stamps_ms()
        assert st["embedded"] <= st["generated"] <= st["shown"]


class Recording:
    """A stand-in backend that records interrupt() calls."""

    def __init__(self):
        self.calls = 0

    def interrupt(self):
        self.calls += 1


@pytest.mark.parametrize("stage, stops", [(InFlight.IDLE, False), (InFlight.VIDEO_START, False),
                                          (InFlight.ANCHOR, False), (InFlight.STOPPING, False),
                                          (InFlight.CONTINUATION, True)])
def test_only_a_continuing_chunk_is_ever_stopped(stage, stops):
    e = Engine(EngineConfig(backend="fake"), lambda b: None, lambda m: None)
    e.backend = Recording()
    e._inflight = stage
    e._stop_in_flight()
    assert e.backend.calls == int(stops)
    assert e._inflight is (InFlight.STOPPING if stops else stage)


def test_a_steer_right_after_a_reset_stops_nothing(runs):
    r = runs(flush="on", interrupt="on")
    e = r.e
    assert wait(lambda: e.chunk_shown >= 10)
    calls = []
    original = e.backend.interrupt
    e.backend.interrupt = lambda: (calls.append(e._inflight), original())
    e.reset(time.monotonic_ns()).result().wait(5)
    rec = e.handle_target(WINTER, time.monotonic_ns())
    assert wait(lambda: rec.stage is CommandStage.SHOWN)
    # The reset restarts the policy's gap, so the steer waits for it; nothing was stopped on the new video's start.
    assert calls == [] and e.stats()["cut"]["interrupted_chunks"] == 0


def test_only_old_scene_chunks_are_stopped_with_refreshes_running(runs):
    r = runs(interrupt="on", reanchor_options={"refresh_every_chunks": 6, "min_gap_chunks": 4})
    e = r.e
    for p in PROMPTS:
        e.encoder.encode(p)
    for k in range(10):  # commands at any time, some while a refresh or a cut is being generated
        time.sleep(0.37 + 0.17 * (k % 4))
        e.handle_target({**WINTER, "prompt": PROMPTS[k % len(PROMPTS)]}, time.monotonic_ns())
    time.sleep(0.5)
    reasons = list(r.reasons.values())
    assert reasons.count("refresh") >= 2 and reasons.count("steer") >= 4
    assert r.stopped and all(reason is None for _, reason in r.stopped)  # never a refresh, a cut or a video start
    assert e.stats()["cut"]["interrupted_chunks"] == len(r.stopped)


# --- latency ---------------------------------------------------------------------------------------------------------

@pytest.mark.tools  # about 30 s: run with pytest -m tools (pytest.ini)
def test_flush_and_interrupt_cut_the_delay():
    """A short run of engine/tools/cut_latency.py at the live pass timings (the full table takes minutes)."""
    today = cut_latency.run(cut_latency.VARIANTS["today"], commands=4)
    fast = cut_latency.run(cut_latency.VARIANTS["all"], commands=4)
    assert not today.problems and not fast.problems, (today.problems, fast.problems)
    assert today.stalls == 0 and fast.stalls == 0
    assert np.median(fast.delays_ms) < np.median(today.delays_ms) - 150


# --- interrupts the real backend drops (PR #187: interrupt() only works once generate_chunk is running) ------------

def test_fake_drops_an_interrupt_made_before_the_chunk_starts():
    b = fake()
    b.generate_chunk(cond())
    b.interrupt()  # between chunks, or after the engine marked the chunk but before the call began
    assert b.generate_chunk(cond())["number"] == 1 and b.interrupted_chunks == 0
    assert b.generate_chunk(cond())["number"] == 2 and b.interrupted_chunks == 0  # nothing stale is left over


def _steer_cut_ready(r: Run, prompt: str) -> None:
    r.e.encoder.encode(prompt)
    assert wait(lambda: r.e.policy.chunks_since_anchor >= r.e.reanchor_config.min_gap_chunks)


def _check_cut_follows(r: Run, rec, kept: int) -> None:
    """The steer cut is the chunk right after `kept`, the command's stamps are in order, and nothing later stops."""
    assert wait(lambda: rec.stage is CommandStage.SHOWN)
    assert rec.applies_from_chunk == kept + 1 and r.reasons[kept + 1] == "steer"
    st = rec.stamps_ms()
    assert st["embedded"] <= st["generated"] <= st["shown"]
    assert wait(lambda: max(c for c, _, _ in r.sent) >= kept + 6)
    counts: dict[int, list[int]] = {}
    for c, i, _ in r.sent:
        counts.setdefault(c, []).append(i)
    assert counts[kept] == [0, 1, 2, 3]  # the old chunk that was not stopped is shown whole
    numbers = sorted(counts)
    assert numbers == list(range(numbers[0], numbers[-1] + 1))
    assert all(v == [0, 1, 2, 3] for c, v in counts.items() if numbers[0] < c < numbers[-1])
    cut = r.e.stats()["cut"]
    assert cut["interrupted_chunks"] == 0 and r.e.backend.interrupted_chunks == 0 and r.stopped == []
    assert r.e._inflight in (InFlight.IDLE, InFlight.CONTINUATION, InFlight.ANCHOR)


def test_an_interrupt_before_the_backend_starts_the_chunk_is_late_and_the_cut_is_next(runs):
    r = runs(interrupt="on")
    e, prompt = r.e, PROMPTS[1]
    _steer_cut_ready(r, prompt)
    generate, interrupt = e.backend.generate_chunk, e.backend.interrupt
    armed, kept, recs = [True], [], []
    interrupted = threading.Event()

    def noted_interrupt():
        interrupt()
        interrupted.set()

    def steer_first(c):
        # The engine has marked this chunk CONTINUATION (in take_cond) but the backend has not started it: the
        # window where the real backend drops interrupt(). Steer now and start the call only once interrupt() has
        # returned: the engine marks STOPPING just before it calls interrupt(), so the state alone is too early.
        if armed[0] and r.taken is not None and r.taken[1] is None:
            armed[0] = False
            kept.append(r.taken[0])
            recs.append(e.handle_target({**WINTER, "prompt": prompt}, time.monotonic_ns()))
            assert interrupted.wait(5) and e._inflight is InFlight.STOPPING
        return generate(c)

    e.backend.interrupt = noted_interrupt

    e.backend.generate_chunk = steer_first
    assert wait(lambda: recs and recs[0].stage is CommandStage.SHOWN)
    assert e.stats()["cut"]["late_interrupts"] == 1
    _check_cut_follows(r, recs[0], kept[0])


def test_a_command_between_chunks_calls_no_interrupt_and_the_cut_is_next(runs):
    r = runs(interrupt="on")
    e, prompt = r.e, PROMPTS[2]
    _steer_cut_ready(r, prompt)
    calls = []
    interrupt = e.backend.interrupt
    e.backend.interrupt = lambda: (calls.append(e._inflight), interrupt())
    take_cond = e.book.take_cond  # Run's wrapper, which notes each chunk's re-anchor reason
    armed, nxt, recs = [True], [], []

    def steer_first(get_cond, chunk):
        if armed[0]:  # the generation thread is between chunks: the last one ended, the next has no cond yet
            armed[0] = False
            assert e._inflight is InFlight.IDLE
            nxt.append(chunk)
            recs.append(e.handle_target({**WINTER, "prompt": prompt}, time.monotonic_ns()))
            assert wait(lambda: recs[0].stage is CommandStage.EMBEDDED)
        return take_cond(get_cond, chunk)

    e.book.take_cond = steer_first
    assert wait(lambda: recs and recs[0].stage is CommandStage.SHOWN)
    assert calls == [] and e.stats()["cut"]["late_interrupts"] == 0
    _check_cut_follows(r, recs[0], nxt[0] - 1)

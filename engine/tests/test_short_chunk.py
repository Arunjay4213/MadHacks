"""A short chunk (fewer than frames_per_chunk frames, as after a rebase) is padded before playout."""

import threading
import time

import numpy as np
import pytest

import engine.engine as engine_module
from engine.backends.fake import FakeBackend
from engine.engine import ChunkShape, Engine, EngineConfig, ShortChunkMode, chunk_shape, pad_short_chunk
from engine.framing import unpack_frame
from engine.lifecycle import Lifecycle

H, W = 6, 8


def _frame(value: int) -> np.ndarray:
    return np.full((H, W, 3), value, dtype=np.uint8)


def test_hold_repeats_the_first_new_frame():
    new = _frame(200)[None]
    out = pad_short_chunk(new, _frame(0), 4, ShortChunkMode.HOLD)
    assert out.shape == (4, H, W, 3) and out.dtype == np.uint8
    assert all((f == 200).all() for f in out)


def test_crossfade_blends_from_the_previous_frame_to_the_new_one():
    new = _frame(250)[None]
    out = pad_short_chunk(new, _frame(10), 4, ShortChunkMode.CROSSFADE)
    assert out.shape == (4, H, W, 3) and out.dtype == np.uint8
    # 10 + 240 * k / 4 for k = 1, 2, 3, then the new frame itself
    assert [int(f[0, 0, 0]) for f in out] == [70, 130, 190, 250]
    # downwards too, per pixel and per channel
    prev = np.zeros((H, W, 3), dtype=np.uint8)
    prev[..., 1] = 255
    newer = np.zeros((1, H, W, 3), dtype=np.uint8)
    newer[..., 0] = 255
    out = pad_short_chunk(newer, prev, 4, ShortChunkMode.CROSSFADE)
    assert [tuple(int(c) for c in f[3, 5]) for f in out] == [(64, 191, 0), (128, 128, 0), (191, 64, 0), (255, 0, 0)]


def test_crossfade_rounds_to_nearest():
    out = pad_short_chunk(_frame(1)[None], _frame(0), 4, ShortChunkMode.CROSSFADE)
    assert [int(f[0, 0, 0]) for f in out] == [0, 1, 1, 1]  # 0.25, 0.5, 0.75, 1


def test_fill_leaves_the_chunk_short():
    new = _frame(200)[None]
    assert pad_short_chunk(new, _frame(0), 4, ShortChunkMode.FILL) is new


def test_two_frame_chunk_pads_before_its_frames():
    new = np.stack([_frame(100), _frame(120)])
    out = pad_short_chunk(new, _frame(40), 4, ShortChunkMode.CROSSFADE)
    assert [int(f[0, 0, 0]) for f in out] == [60, 80, 100, 120]  # 40 + 60 * k / 3, then both new frames


def test_chunk_shape():
    assert chunk_shape(1, 4, starts_video=True) is ChunkShape.VIDEO_START
    assert chunk_shape(4, 4, starts_video=True) is ChunkShape.VIDEO_START
    assert chunk_shape(1, 4, starts_video=False) is ChunkShape.SHORT
    assert chunk_shape(4, 4, starts_video=False) is ChunkShape.FULL


def test_bad_mode_is_refused():
    with pytest.raises(ValueError, match="short_chunk must be one of"):
        Engine(EngineConfig(short_chunk="blend"), lambda b: None, lambda m: None)


def test_fake_short_on_chunks():
    b = FakeBackend(16, 8, 0, gen_ms=0, decode_ms=0, encode_ms=0, short_on_chunks="0,2")
    cond = b.encode_prompt("x")
    counts = [len(b.decode_chunk(b.generate_chunk(cond))) for _ in range(4)]
    assert counts == [1, 4, 1, 4]  # chunk 0 is the video's first chunk (1 frame) either way
    assert FakeBackend(16, 8, 0, short_on_chunks=7).short_on_chunks == {7}
    with pytest.raises(ValueError):
        FakeBackend(16, 8, 0, short_on_chunks="a,b")


# --- the engine end to end with the fake backend -------------------------------------------------------------------


def wait(pred, timeout=15.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


class Run:
    """An engine on the fake backend that records every sent frame and every frame array given to JPEG.

    It also records each time the streamer asked the playout buffer for a frame and found it empty (`dry`: chunk,
    frame index and ms waited). That is a frame not ready in time, while a late send under CPU load is not."""

    def __init__(self, monkeypatch, mode: str, short_on: str, **opts):
        self.sent: list[tuple[int, int, float]] = []
        self.arrays: list[np.ndarray] = []
        self.dry: list[tuple[int, int, float]] = []
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)  # notified on every sent frame
        real = engine_module.encode_jpeg

        def spy(frame, quality):
            self.arrays.append(np.array(frame, copy=True))
            return real(frame, quality)

        monkeypatch.setattr(engine_module, "encode_jpeg", spy)
        options = dict(gen_ms=20, decode_ms=5, encode_ms=5, frames_per_chunk=4, fps=50, short_on_chunks=short_on)
        options.update(opts)
        cfg = EngineConfig(backend="fake", width=64, height=48, short_chunk=mode, record_dir=None,
                           backend_options=options)
        self.engine = Engine(cfg, self._on_frame, lambda m: None)
        playout = self.engine._playout_queue
        real_get = playout.get

        def get(timeout=None):
            empty, t = playout.empty(), time.monotonic()
            item = real_get(timeout=timeout)
            if empty:
                with self.lock:
                    self.dry.append((item[2].chunk, item[0], (time.monotonic() - t) * 1e3))
            return item

        playout.get = get

    def _on_frame(self, b: bytes) -> None:
        chunk, index, _ = unpack_frame(b)
        with self.changed:
            self.sent.append((chunk, index, time.monotonic()))
            self.changed.notify_all()

    def wait_until(self, pred, timeout: float = 60.0) -> bool:
        """Block until `pred()` is true, rechecked on every sent frame (and at least every 0.1 s)."""
        deadline = time.monotonic() + timeout
        with self.changed:
            while not pred():
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self.changed.wait(min(left, 0.1))
            return True

    def frames_of(self, chunk: int) -> list[int]:
        with self.lock:
            return [i for c, i, _ in self.sent if c == chunk]

    def __enter__(self):
        self.engine.start()
        assert wait(lambda: self.engine.lifecycle.state is Lifecycle.GENERATING)
        return self

    def __exit__(self, *exc):
        self.engine.stop()


@pytest.mark.parametrize("mode", ["hold", "crossfade"])
def test_engine_pads_a_short_chunk_to_four_frames(monkeypatch, mode):
    with Run(monkeypatch, mode, "10,20") as run:
        e = run.engine
        assert wait(lambda: e.chunk_shown >= 22)
        assert run.frames_of(10) == [0, 1, 2, 3] and run.frames_of(20) == [0, 1, 2, 3]
        assert run.frames_of(9) == [0, 1, 2, 3] and run.frames_of(0) == [0]  # first chunk of the video: 1 frame
        s = e.stats()["short_chunks"]
        assert s["mode"] == mode and s["count"] == 2 and s["last_chunk"] == 20 and s["last"]["frames"] == 1
        # frame arrays in decode order: chunk 0 has 1, chunks 1..9 have 4, so chunk 10 starts at index 37
        start = 1 + 9 * 4
        prev, padded = run.arrays[start - 1], run.arrays[start:start + 4]
        new = padded[3]
        for k in range(3):
            if mode == "hold":
                expected = new
            else:
                a = prev.astype(np.float32)
                expected = (a + (new.astype(np.float32) - a) * (k + 1) / 4 + 0.5).astype(np.uint8)
            assert np.array_equal(padded[k], expected), k
        assert not np.array_equal(prev, new)


def test_fill_mode_plays_the_short_chunk_unpadded(monkeypatch):
    with Run(monkeypatch, "fill", "10") as run:
        e = run.engine
        assert wait(lambda: e.chunk_shown >= 12)
        assert run.frames_of(10) == [0]
        s = e.stats()["short_chunks"]
        assert s["mode"] == "fill" and s["count"] == 1 and s["last_chunk"] == 10


def test_no_stall_and_frames_keep_pace(monkeypatch):
    # gen + decode (36 ms) close to a chunk's play time (80 ms at 50 fps for 4 frames) with a small lead margin,
    # so the generator does not run far ahead: a short chunk left unpadded runs the playout dry (in fill mode the
    # streamer waits about 37 ms on an empty buffer at the first frame after each short chunk).
    short = (15, 25, 35)
    with Run(monkeypatch, "crossfade", ",".join(map(str, short)), gen_ms=30, decode_ms=6) as run:
        e = run.engine
        e.config.lead_margin = 1.0
        assert run.wait_until(lambda: e.chunk_shown >= 40), f"stuck at chunk {e.chunk_shown}"
        assert e.stats()["short_chunks"]["count"] == 3
        with run.lock:
            dry = [d for d in run.dry if d[2] > 1e3 / 50 / 2]  # waited over half a frame slot: a stall
            window = [t for c, _, t in run.sent if 12 <= c <= 38]
    # no frame of a short chunk or of the chunk after it was late (CPU load may delay others; that is not this test)
    assert [d for d in dry if any(c in (d[0], d[0] - 1) for c in short)] == [], dry
    # every sent frame is one 1/50 s slot: frames keep pace with the clock of the sent frames (10% for CPU load)
    elapsed = window[-1] - window[0]
    assert abs((len(window) - 1) - elapsed * 50) <= 0.1 * elapsed * 50 + 2, (len(window), elapsed)


def test_first_chunk_after_a_reset_is_unchanged(monkeypatch):
    with Run(monkeypatch, "crossfade", "none") as run:
        e = run.engine
        assert wait(lambda: e.chunk_shown >= 5)
        assert e.request_reset().wait(10)
        video = e.video
        assert wait(lambda: e.chunk_shown >= 15)
        assert video == 1
        with run.lock:
            done = {c for c, _, _ in run.sent} - {max(c for c, _, _ in run.sent)}  # the last may be still playing
            one_frame = sorted(done - {c for c, i, _ in run.sent if i > 0})
        # chunk 0 and the first chunk of the new video: 1 frame each, not padded and not cross-faded
        assert len(one_frame) == 2 and one_frame[0] == 0
        assert e.stats()["short_chunks"]["count"] == 0


def test_padding_time_counts_as_decode_time(monkeypatch):
    """_lead_limit budgets the pipeline from gen, decode and JPEG ms, so the padding of a short chunk, which runs
    on the decode thread between decode and JPEG, must be in one of them (here: decode). It was in neither."""
    real_pad = engine_module.pad_short_chunk
    padded = threading.Event()

    def slow_pad(*args):
        time.sleep(0.1)  # the padding's own cost, not a wait for the test
        padded.set()
        return real_pad(*args)

    monkeypatch.setattr(engine_module, "pad_short_chunk", slow_pad)
    with Run(monkeypatch, "crossfade", "6", decode_ms=5) as run:
        e = run.engine
        assert padded.wait(60)
        assert run.wait_until(lambda: e.chunk_shown >= 6), f"stuck at chunk {e.chunk_shown}"  # its stats are in
        with e._stats_lock:
            decode = list(e.decode_ms.values)
        assert max(decode) >= 100  # the padded chunk's decode time includes its 100 ms of padding

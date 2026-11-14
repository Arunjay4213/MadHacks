"""The recorder on its own and inside the engine with the fake backend. Needs ffmpeg for most tests."""

import json
import os
import shutil
import subprocess
import time

import numpy as np
import pytest

from engine.engine import Engine, EngineConfig, encode_jpeg
from engine.recorder import Recorder, RecorderState

from .test_engine import wait

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def probe(path) -> dict:
    out = subprocess.run([FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def jpeg(i: int) -> bytes:
    frame = np.full((48, 64, 3), (i * 7) % 256, dtype=np.uint8)
    return encode_jpeg(frame, 80)


def test_disabled_without_dir_or_ffmpeg(tmp_path):
    off = Recorder(None)
    assert off.state is RecorderState.DISABLED
    no_ffmpeg = Recorder(tmp_path, ffmpeg=None)
    assert no_ffmpeg.state is RecorderState.DISABLED
    no_ffmpeg.start(16)
    no_ffmpeg.offer(0, b"x")  # ignored, never raises
    no_ffmpeg.stop()
    assert no_ffmpeg.stats() == {"state": "DISABLED", "file": None, "frames_written": 0, "frames_dropped": 0,
                                 "sessions": 0, "error": None}
    assert list(tmp_path.iterdir()) == []


@needs_ffmpeg
def test_one_file_per_video_and_finalized_on_stop(tmp_path):
    r = Recorder(tmp_path)
    assert r.state is RecorderState.IDLE
    r.start(16)
    for i in range(20):
        r.offer(0, jpeg(i))
    assert wait(lambda: r.stats()["frames_written"] == 20)
    assert r.state is RecorderState.RECORDING
    for i in range(12):
        r.offer(1, jpeg(i))  # a new video: the first file is finalized, a second one starts
    assert wait(lambda: r.stats()["sessions"] == 2 and r.stats()["frames_written"] == 12)
    r.stop()
    assert r.state is RecorderState.STOPPED
    files = sorted(tmp_path.glob("*.mp4"))
    assert [f.name[-9:] for f in files] == ["-s001.mp4", "-s002.mp4"]
    if FFPROBE:
        first, second = probe(files[0]), probe(files[1])
        assert (first["width"], first["height"], first["avg_frame_rate"]) == (64, 48, "16/1")
        assert int(first["nb_read_frames"]) == 20 and int(second["nb_read_frames"]) == 12


@needs_ffmpeg
def test_full_queue_drops_frames_and_counts_them(tmp_path):
    r = Recorder(tmp_path, queue_frames=2)
    r.start(16)
    data = jpeg(1)
    t0 = time.perf_counter()
    for _ in range(500):
        r.offer(0, data)
    assert time.perf_counter() - t0 < 0.5  # offer never waits for the encoder
    r.stop()
    s = r.stats()
    assert s["frames_dropped"] > 0
    assert s["frames_dropped"] + s["frames_written"] == 500


def test_ffmpeg_failure_is_error_until_the_next_video(tmp_path):
    r = Recorder(tmp_path, ffmpeg=shutil.which("false"))
    r.start(16)
    data = jpeg(1) * 200  # big writes, so the broken pipe shows quickly
    for _ in range(50):
        r.offer(0, data)
        time.sleep(0.01)
    assert wait(lambda: r.state is RecorderState.ERROR)
    assert r.stats()["error"]
    r.offer(1, data)  # a new video tries a new file
    assert wait(lambda: r.stats()["sessions"] >= 2)
    r.stop()
    assert r.state is RecorderState.STOPPED


@needs_ffmpeg
def test_oldest_files_are_deleted_over_the_limit(tmp_path):
    for i, name in enumerate(["20260101T000000Z-s001.mp4", "20260101T000100Z-s001.mp4", "20260101T000200Z-s001.mp4"]):
        p = tmp_path / name
        p.write_bytes(b"0" * 1000)
        t = time.time() - 100 + i
        os.utime(p, (t, t))
    r = Recorder(tmp_path, max_bytes=2500)
    r.start(16)
    r.offer(0, jpeg(0))
    assert wait(lambda: r.state is RecorderState.RECORDING)
    r.stop()
    names = sorted(p.name for p in tmp_path.glob("*.mp4"))
    assert "20260101T000000Z-s001.mp4" not in names
    assert "20260101T000100Z-s001.mp4" in names and len(names) == 3


@needs_ffmpeg
def test_engine_records_each_video(tmp_path):
    cfg = EngineConfig(backend="fake", width=64, height=48, record_dir=str(tmp_path), backend_options=dict(
        gen_ms=20, decode_ms=5, encode_ms=10, frames_per_chunk=3, fps=50))
    e = Engine(cfg, lambda b: None, lambda m: None)
    e.start()
    try:
        assert wait(lambda: e.stats()["recorder"]["frames_written"] > 30)
        assert e.stats()["recorder"]["state"] == "RECORDING"
        assert e.reset(time.monotonic_ns()).result(5).wait(5)
        assert wait(lambda: e.stats()["recorder"]["sessions"] == 2 and e.stats()["recorder"]["frames_written"] > 10)
    finally:
        e.stop()
    assert e.stats()["recorder"]["state"] == "STOPPED"
    files = sorted(tmp_path.glob("*.mp4"))
    assert len(files) == 2
    if FFPROBE:
        for f in files:
            s = probe(f)
            assert (s["width"], s["height"], s["avg_frame_rate"]) == (64, 48, "50/1") and int(s["nb_read_frames"]) > 10

"""Saves every video the engine streams to an mp4 file, without ever slowing the streamer.

The streamer calls `offer(video, jpeg)` for each frame it sends, with the number of the video the frame belongs to.
`offer` only puts the frame on a bounded queue; when the queue is full the frame is dropped and counted.
The recorder thread pipes the JPEGs into an ffmpeg subprocess (mjpeg in, H.264 out), so the encoding runs in
another process and holds no Python lock.
When a frame of a new video arrives (after a reset or a recovery), the current file is finalized and a new one starts.

Files are fragmented mp4, so even a file whose ffmpeg was killed is playable up to its last fragment (2 s).
"""

import enum
import logging
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("engine.recorder")

QUEUE_FRAMES = 64              # about 4 s of video at 16 fps
FINALIZE_TIMEOUT_S = 15.0      # how long ffmpeg gets to write the last fragment after its input closes
DEFAULT_MAX_BYTES = 20 * 10**9


class RecorderState(enum.Enum):
    DISABLED = "DISABLED"      # no --record-dir, or ffmpeg is missing; offer() ignores frames
    IDLE = "IDLE"              # running, no file open (before the first frame)
    RECORDING = "RECORDING"    # writing frames of one video into one file
    FINALIZING = "FINALIZING"  # ffmpeg's input is closed; waiting for it to finish the file
    ERROR = "ERROR"            # ffmpeg failed for the current video; its frames are dropped until the next video
    STOPPED = "STOPPED"        # stop() finished; the last file is finalized


ALLOWED: dict[RecorderState, frozenset[RecorderState]] = {
    RecorderState.DISABLED: frozenset(),
    RecorderState.IDLE: frozenset({RecorderState.RECORDING, RecorderState.ERROR, RecorderState.STOPPED}),
    RecorderState.RECORDING: frozenset({RecorderState.FINALIZING, RecorderState.ERROR}),
    RecorderState.FINALIZING: frozenset({RecorderState.IDLE, RecorderState.ERROR, RecorderState.STOPPED}),
    RecorderState.ERROR: frozenset({RecorderState.RECORDING, RecorderState.ERROR, RecorderState.STOPPED}),
    RecorderState.STOPPED: frozenset(),
}

_STOP = object()  # queue sentinel


class Recorder:
    def __init__(self, record_dir: str | Path | None, max_bytes: int = DEFAULT_MAX_BYTES,
                 ffmpeg: str | None = "auto", queue_frames: int = QUEUE_FRAMES) -> None:
        self._lock = threading.Lock()
        self.dir = Path(record_dir).expanduser() if record_dir else None
        self.max_bytes = int(max_bytes)
        self.ffmpeg = shutil.which("ffmpeg") if ffmpeg == "auto" else ffmpeg
        self._queue: queue.Queue = queue.Queue(maxsize=queue_frames)
        self._thread: threading.Thread | None = None
        self.fps = 16
        self.sessions = 0           # files opened by this process; the number in the file name
        self.file: Path | None = None
        self.frames_written = 0     # into the current file
        self.frames_dropped = 0     # over the whole run: queue full, or the current video is in ERROR
        self.error: str | None = None
        self._proc: subprocess.Popen | None = None
        self._video: int | None = None  # the video the current file (or the ERROR) belongs to
        if self.dir is None:
            self._state = RecorderState.DISABLED
            log.info("recorder: DISABLED (no record dir)")
        elif self.ffmpeg is None:
            self._state = RecorderState.DISABLED
            log.warning("recorder: DISABLED because ffmpeg is not installed; the engine runs without recording")
        else:
            self._state = RecorderState.IDLE
            log.info("recorder: IDLE, recording to %s (limit %.1f GB)", self.dir, self.max_bytes / 1e9)

    # --- state -----------------------------------------------------------------------------

    @property
    def state(self) -> RecorderState:
        with self._lock:
            return self._state

    def _transition(self, new: RecorderState, reason: str = "") -> None:
        with self._lock:
            old = self._state
            if new not in ALLOWED[old]:
                raise RuntimeError(f"recorder: {old.value} -> {new.value} is not allowed")
            self._state = new
        log.info("recorder: %s -> %s%s", old.value, new.value, f" ({reason})" if reason else "")

    def stats(self) -> dict:
        with self._lock:
            return {"state": self._state.value, "file": str(self.file) if self.file else None,
                    "frames_written": self.frames_written, "frames_dropped": self.frames_dropped,
                    "sessions": self.sessions, "error": self.error}

    # --- called by the engine --------------------------------------------------------------

    def start(self, fps: int) -> None:
        """Start the recorder thread. Called once the backend's fps is known."""
        if self.state is RecorderState.DISABLED or self._thread is not None:
            return
        self.fps = int(fps)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def offer(self, video: int, jpeg: bytes) -> None:
        """Streamer thread: never blocks. Drops (and counts) the frame when the queue is full."""
        if self._thread is None:
            return
        try:
            self._queue.put_nowait((video, jpeg))
        except queue.Full:
            with self._lock:
                self.frames_dropped += 1
                dropped = self.frames_dropped
            if dropped % 50 == 1:
                log.warning("recorder: queue full, dropped %d frames so far", dropped)

    def stop(self, timeout: float = FINALIZE_TIMEOUT_S + 5) -> None:
        """Finalize the current file and stop the thread. Safe to call more than once."""
        t = self._thread
        if t is None or not t.is_alive():
            return
        deadline = time.monotonic() + timeout
        while t.is_alive() and time.monotonic() < deadline:
            try:
                self._queue.put(_STOP, timeout=0.2)
                break
            except queue.Full:
                continue
        t.join(max(0.0, deadline - time.monotonic()))
        if t.is_alive():
            log.warning("recorder: did not stop within %.0f s", timeout)

    # --- recorder thread -------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            video, jpeg = item
            try:
                self._frame(video, jpeg)
            except Exception as exc:  # never let the thread die; the engine keeps streaming
                log.exception("recorder: unexpected error")
                self._to_error(f"unexpected: {exc!r}")
        state = self.state
        if state is RecorderState.RECORDING:
            self._finalize(RecorderState.STOPPED, "stop")
        else:
            self._transition(RecorderState.STOPPED, "stop")

    def _frame(self, video: int, jpeg: bytes) -> None:
        state = self.state
        if state is RecorderState.RECORDING and video != self._video:
            self._finalize(RecorderState.IDLE, f"video {self._video} ended")
            state = RecorderState.IDLE
        if state is RecorderState.ERROR and video == self._video:
            with self._lock:
                self.frames_dropped += 1
            return
        if state in (RecorderState.IDLE, RecorderState.ERROR):
            if not self._open(video):
                with self._lock:
                    self.frames_dropped += 1
                return
        try:
            self._proc.stdin.write(jpeg)
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._to_error(f"writing to ffmpeg failed: {exc!r} (ffmpeg exit code {self._proc.poll()})")
            with self._lock:
                self.frames_dropped += 1
            return
        with self._lock:
            self.frames_written += 1

    def _open(self, video: int) -> bool:
        """Start ffmpeg on a new file for `video`. False (and ERROR) if it cannot start."""
        self._prune()
        session = self.sessions + 1
        start = datetime.now(timezone.utc)
        path = self.dir / f"{start:%Y%m%dT%H%M%SZ}-s{session:03d}.mp4"
        self._video = video
        try:
            self._proc = subprocess.Popen(self._command(path), stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            with self._lock:
                self.sessions = session
            self._to_error(f"could not start ffmpeg: {exc!r}")
            return False
        # One update under the lock, so stats() never shows the new session with the old file's frame count.
        with self._lock:
            self.sessions, self.file, self.frames_written, self.error = session, path, 0, None
        self._transition(RecorderState.RECORDING, f"video {video} -> {path.name}")
        return True

    def _command(self, path: Path) -> list[str]:
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error",
               "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(self.fps), "-i", "pipe:0",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
               "-g", str(2 * self.fps), "-threads", "2",
               # A fragment per key frame (2 s), each written to disk at once (-flush_packets 1). Without the flush
               # ffmpeg keeps finished fragments in its output buffer, and a SIGKILL would leave an empty file.
               "-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-flush_packets", "1", "-y", str(path)]
        nice = shutil.which("nice")
        return [nice, "-n", "10", *cmd] if nice else cmd  # encoding yields the CPU to the engine

    def _finalize(self, then: RecorderState, reason: str) -> None:
        self._transition(RecorderState.FINALIZING, reason)
        proc, path = self._proc, self.file
        t0 = time.perf_counter()
        self._close_proc()
        size = path.stat().st_size if path and path.exists() else 0
        log.info("recorder: finalized %s: %d frames (%.1f s of video), %.1f MB, exit code %s, in %.0f ms",
                 path.name if path else None, self.frames_written, self.frames_written / self.fps, size / 1e6,
                 proc.returncode if proc else None, (time.perf_counter() - t0) * 1e3)
        with self._lock:
            self.file = None
        self._transition(then, reason)

    def _close_proc(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(FINALIZE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log.warning("recorder: ffmpeg did not finish in %.0f s; killing it", FINALIZE_TIMEOUT_S)
            proc.kill()
            proc.wait()

    def _to_error(self, error: str) -> None:
        log.error("recorder: %s; frames of video %s are dropped until the next video", error, self._video)
        self._close_proc()
        with self._lock:
            self.error = error
        self._transition(RecorderState.ERROR, error)

    def _prune(self) -> None:
        """Delete the oldest recordings while the folder holds more than max_bytes."""
        files = sorted(self.dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while files and total > self.max_bytes:
            old = files.pop(0)
            size = old.stat().st_size
            old.unlink(missing_ok=True)
            total -= size
            log.info("recorder: deleted %s (%.1f MB) to stay under %.1f GB", old.name, size / 1e6, self.max_bytes / 1e9)

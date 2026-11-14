"""The engine: generation thread, decode thread, streamer thread, and the command flow into the steerer.

    generation thread   cond = steerer.get_cond(); latents = backend.generate_chunk(cond); decode_queue.put(...)
    decode thread       decode_queue -> backend.decode_chunk -> JPEG -> playout_queue  (no other shared state)
    streamer thread     playout_queue -> paced at backend.fps -> publish(frame); on a chunk's first frame,
                        stamp `shown` and publish one contract 4 message

Commands: contract 1 -> reader worker thread: reader.handle -> steer (contract 2) -> encoder service (its own thread)
-> steer worker thread: steering.Controller.handle (which calls steerer.set_target) -> takes effect on the next get_cond.
The frame cache is never rebuilt on a command; only the steerer's target changes.

Re-anchor (--reanchor on, off by default): steering.reanchor.ReanchorPolicy sits next to the Controller. It hears
every applied target (on_target after Controller.handle), is asked once per chunk where cond and progress are taken
(next_chunk), and restarts on a reset (on_reset). An anchor it returns goes to the backend in Engine._generate.

Recovery: an exception in the generation or decode thread does not stop the engine. The generation thread moves
to RECOVERING, drops the queued chunks and frames, calls backend.reset() and restarts the steerer on the starting
prompt (the model stays loaded), then goes back to GENERATING. After `recovery_limit` crashes within
`recovery_window_s` it goes to FAILED and calls `on_fatal`, which in the server exits the process for systemd.
"""

import collections
import enum
import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from steering import Controller, Steerer
from steering.reanchor import Anchor, ReanchorPolicy

from .backends import load_backend
from .commands import CommandBook, CommandRecord
from .encoder import EncoderService
from .framing import pack_frame
from .lifecycle import Lifecycle, LifecycleMachine
from .messages import BadMessage, check
from .reader import load_reader
from .recorder import DEFAULT_MAX_BYTES, Recorder

log = logging.getLogger("engine")


@dataclass
class EngineConfig:
    backend: str = "fake"
    reader: str = "standin"         # standin, or commands (commands.service.CommandService)
    width: int = 832
    height: int = 480
    steps: int = 0                  # 0: the backend's distilled schedule
    device: str = "cuda:0"
    decode_device: str = "cuda:0"
    backend_options: dict = field(default_factory=dict)
    base_prompt: str = "a wide shot of a meadow with a single oak tree"  # used when the reader has no committed prompt
    jpeg_quality: int = 80
    precompute_file: str | None = None
    decode_queue_size: int = 2      # chunks waiting for decode; generation blocks (and logs) when full
    playout_max_frames: int = 64    # hard bound on decoded frames waiting to be sent
    lead_margin: float = 1.3        # see LeadGate
    summary_every: int = 20         # chunks between rolling summary log lines
    reanchor: str = "off"           # "off" or "on": see ReanchorState
    record_dir: str | None = None   # where every video is saved as mp4; None turns the recorder off
    record_max_bytes: int = DEFAULT_MAX_BYTES  # the oldest recordings are deleted above this total
    recovery_limit: int = 3         # this many crashes within recovery_window_s -> FAILED
    recovery_window_s: float = 60.0


class ReanchorState(enum.Enum):
    """The engine's re-anchor feature. Chosen once, when the backend is loaded."""
    OFF = "off"                  # --reanchor off: no policy; the backend call is generate_chunk(cond)
    ON = "on"                    # --reanchor on and the backend can re-anchor: the policy's anchors reach it
    UNSUPPORTED = "unsupported"  # --reanchor on but the backend cannot re-anchor: no policy, as OFF


@dataclass
class ChunkMeta:
    """Travels with a chunk through both queues."""
    chunk: int
    progress: float
    gen_ms: float
    est_frames: int
    anchor: Anchor | None = None  # the re-anchor this chunk was generated with, if it was a re-anchor chunk
    video: int = 0                  # which video (session) the chunk belongs to; rises at every reset and recovery
    decode_ms: float = 0.0
    jpeg_ms: float = 0.0
    frames: int = 0


class LeadGate:
    """Limits how far generation runs ahead of the frames already sent, to keep command delay low.

    `ahead` counts frames generated but not yet sent. Before each chunk, generation waits until `ahead` is at most
    `limit()` frames: enough video to cover one chunk's generation plus decode time, times a margin.
    Without this, generation faster than real time would fill every queue and add their length to the delay.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.ahead = 0

    def add(self, frames: int) -> None:
        with self._cond:
            self.ahead += frames

    def sent(self, frames: int = 1) -> None:
        with self._cond:
            self.ahead -= frames
            self._cond.notify_all()

    def wait(self, limit: Callable[[], int], *until: threading.Event) -> None:
        """Wait until `ahead` is at most `limit()`, or until any of the events is set."""
        with self._cond:
            while self.ahead > limit() and not any(e.is_set() for e in until):
                self._cond.wait(0.1)

    def clear(self) -> None:
        with self._cond:
            self.ahead = 0
            self._cond.notify_all()


class Rolling:
    """Last and windowed average of a timing in ms."""

    def __init__(self, n: int = 20) -> None:
        self.values: collections.deque[float] = collections.deque(maxlen=n)

    def add(self, v: float) -> None:
        self.values.append(v)

    @property
    def last(self) -> float | None:
        return self.values[-1] if self.values else None

    @property
    def avg(self) -> float | None:
        return sum(self.values) / len(self.values) if self.values else None


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 1)


class Engine:
    def __init__(self, config: EngineConfig, publish_frame: Callable[[bytes], None],
                 publish_control: Callable[[dict], None], reader=None) -> None:
        self.config = config
        self.publish_frame = publish_frame
        self.publish_control = publish_control
        self.reader = reader if reader is not None else load_reader(config.reader)
        self.base_prompt: str | None = None
        self.lifecycle = LifecycleMachine()
        self.book = CommandBook()
        self.encoder: EncoderService | None = None
        self.backend = None
        self.steerer: Steerer | None = None
        self.controller: Controller | None = None
        self.base_embedding = None
        self.failure: str | None = None
        if config.reanchor not in ("off", "on"):
            raise ValueError(f"reanchor must be 'off' or 'on', got {config.reanchor!r}")
        self.reanchor_state = ReanchorState.OFF  # set from config.reanchor once the backend is loaded
        self.policy: ReanchorPolicy | None = None  # exists exactly when reanchor_state is ON
        self.on_fatal: Callable[[str], None] | None = None  # called once on FAILED; the server exits the process
        self.recorder = Recorder(config.record_dir, config.record_max_bytes)
        self.video = 0  # the current video's number; only the generation thread changes it

        self._stop = threading.Event()
        self._decode_queue: queue.Queue = queue.Queue(maxsize=config.decode_queue_size)
        self._playout_queue: queue.Queue = queue.Queue(maxsize=config.playout_max_frames)
        self._gate = LeadGate()
        self._reset_lock = threading.Lock()
        self._reset_waiters: list[threading.Event] = []
        self._reset_seed: tuple[dict, CommandRecord] | None = None  # (the reset's steer, its command)
        self._threads: list[threading.Thread] = []
        # A crash in the generation or decode thread: set once, handled by the generation thread in _recover.
        self._crash_lock = threading.Lock()
        self._crash: tuple[str, str, float] | None = None  # (where, error, monotonic time) of the first crash
        self._crashed = threading.Event()
        self._crash_times: collections.deque[float] = collections.deque()
        self._resume_pending: tuple[int, float] | None = None  # (video, crash time) until its first frame is sent
        self.recoveries = 0
        self.last_recovery: dict | None = None
        # reader.handle (may block about 0.4 s) and Controller.handle run here, off the generation thread and the
        # asyncio loop. One reader thread keeps commands in arrival order.
        self._reader_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reader")
        self._steer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="steer")

        self._stats_lock = threading.Lock()
        self.chunk_generated = -1   # last chunk number generated
        self.chunk_shown = -1       # last chunk whose first frame was sent
        self.gen_ms = Rolling()
        self.decode_ms = Rolling()
        self.jpeg_ms = Rolling()
        self.frames_sent = 0
        self.stalls = 0
        self.decode_blocked = 0
        self._sent_times: collections.deque[float] = collections.deque(maxlen=200)
        self.reanchors = 0
        self.last_anchor: Anchor | None = None
        self.last_anchor_chunk: int | None = None
        self.client_counts: Callable[[], dict] = lambda: {}

    # --- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        t = threading.Thread(target=self._main, name="generation", daemon=True)
        self._threads.append(t)
        t.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._gate.clear()
        for t in self._threads:
            t.join(timeout)
            if t.is_alive():
                log.warning("thread %s did not stop within %.0f s", t.name, timeout)
        self._reader_pool.shutdown(wait=False, cancel_futures=True)
        self._steer_pool.shutdown(wait=False, cancel_futures=True)
        if self.encoder:
            self.encoder.stop()
        self.recorder.stop()
        if self.lifecycle.state is not Lifecycle.STOPPED:
            self.lifecycle.transition(Lifecycle.STOPPED, "shutdown")

    def _fail(self, what: str, exc: BaseException) -> None:
        log.error("%s failed", what, exc_info=exc)
        self.failure = f"{what}: {exc!r}"
        if self.lifecycle.state in (Lifecycle.FAILED, Lifecycle.STOPPED):
            return
        self.lifecycle.transition(Lifecycle.FAILED, self.failure)
        self._stop.set()
        self._gate.clear()
        self.recorder.stop()  # finalize the open file before the process may exit
        if self.on_fatal is not None:
            self.on_fatal(self.failure)

    def _main(self) -> None:
        cfg = self.config
        try:
            t0 = time.perf_counter()
            self.backend = load_backend(cfg.backend, width=cfg.width, height=cfg.height, steps=cfg.steps,
                                        device=cfg.device, decode_device=cfg.decode_device,
                                        **cfg.backend_options)
            log.info("backend %s loaded in %.1f s: %dx%d, %d fps, %d frames per chunk", cfg.backend,
                     time.perf_counter() - t0, self.backend.width, self.backend.height, self.backend.fps,
                     self.backend.frames_per_chunk)
            if self._stop.is_set():
                return
            self.reanchor_state = self._reanchor_support()
            self.lifecycle.transition(Lifecycle.WARMING, "backend loaded")
            self.encoder = EncoderService(self.backend.encode_prompt_with_ids)
            self.encoder.start()
            self.recorder.start(self.backend.fps)
            committed = self.reader.committed_prompt
            self.base_prompt = committed if committed else cfg.base_prompt
            log.info("starting prompt %r (from %s)", self.base_prompt, "the reader" if committed else "--base-prompt")
            self.base_embedding = self.encoder.encode(self.base_prompt)
            self._precompute()
            self._new_steerer(self.base_embedding)
            for target, name in ((self._decode_loop, "decode"), (self._stream_loop, "streamer")):
                t = threading.Thread(target=target, name=name, daemon=True)
                self._threads.append(t)
                t.start()
            if self._stop.is_set():
                return
            self.lifecycle.transition(Lifecycle.GENERATING, "warm")
            self._run_generation()
        except Exception as exc:
            self._fail("generation thread", exc)

    def _run_generation(self) -> None:
        """Run the generation loop; after a crash anywhere in generation or decode, recover and run it again."""
        while not self._stop.is_set():
            try:
                self._generation_loop()
            except Exception as exc:
                self._report_crash("generation thread", exc)
            if self._stop.is_set():
                return
            if self._crashed.is_set():
                self._recover()  # raises EngineFailed after too many crashes

    # --- recovery --------------------------------------------------------------------------

    def _report_crash(self, where: str, exc: BaseException) -> None:
        """Any engine thread: record a crash. The generation thread notices it at its next step and recovers."""
        log.error("%s raised; the engine will recover", where, exc_info=exc)
        with self._crash_lock:
            if self._crash is None:
                self._crash = (where, repr(exc), time.monotonic())
        self._crashed.set()

    def _recover(self) -> None:
        """Generation thread: RECOVERING, drop the broken video, backend.reset(), starting prompt, GENERATING."""
        with self._crash_lock:
            where, error, t_crash = self._crash
        self.lifecycle.transition(Lifecycle.RECOVERING, f"{where}: {error}")
        while True:
            now = time.monotonic()
            self._crash_times.append(now)
            while self._crash_times and now - self._crash_times[0] > self.config.recovery_window_s:
                self._crash_times.popleft()
            if len(self._crash_times) >= self.config.recovery_limit:
                raise EngineFailed(f"{len(self._crash_times)} crashes within {self.config.recovery_window_s:.0f} s; "
                                   f"last one in {where}: {error}")
            try:
                self._drop_queued_video()
                with self._crash_lock:  # the decoder is idle and the queues are empty: later crashes are new ones
                    self._crash = None
                    self._crashed.clear()
                self.backend.reset()
                self.book.reset(self._restart_on_base_prompt, None)
                break
            except Exception as exc:
                log.error("recovery attempt failed", exc_info=exc)
                where, error = "recovery", repr(exc)
        self.video += 1
        self.recoveries += 1
        self._resume_pending = (self.video, t_crash)
        ms = (time.monotonic() - t_crash) * 1e3
        self.lifecycle.transition(Lifecycle.GENERATING, f"recovered {ms:.0f} ms after the crash, video {self.video} "
                                                         f"on the starting prompt")

    def _restart_on_base_prompt(self) -> None:
        """Under the CommandBook lock, as a reset does: the steerer on the starting prompt, the policy restarted."""
        self.controller.reset(self.base_prompt)
        if self.policy is not None:
            self.policy.on_reset()

    def _drop_queued_video(self) -> None:
        """Throw away every chunk waiting for decode and every frame waiting to be sent."""
        while True:
            try:
                self._decode_queue.get_nowait()
                self._decode_queue.task_done()
            except queue.Empty:
                break
        self._decode_queue.join()  # a chunk the decoder is working on now
        while True:
            try:
                self._playout_queue.get_nowait()
            except queue.Empty:
                break
        self._gate.clear()

    def _precompute(self) -> None:
        """Encode the --precompute-file prompts and the reader's likely prompts before generation starts."""
        prompts = []
        if self.config.precompute_file:
            path = Path(self.config.precompute_file)
            prompts += [line.strip() for line in path.read_text().splitlines() if line.strip()]
        prompts += self._likely_prompts()
        t0 = time.perf_counter()
        for p in dict.fromkeys(prompts):
            if self._stop.is_set():
                return
            self.encoder.encode(p)
        if prompts:
            log.info("precomputed %d prompts in %.1f s", len(set(prompts)), time.perf_counter() - t0)

    def _likely_prompts(self) -> list[str]:
        try:
            return [p for p in self.reader.likely_prompts() if isinstance(p, str) and p]
        except Exception:
            log.exception("reader.likely_prompts failed")
            return []

    def _new_steerer(self, embedding) -> None:
        """A steerer at `embedding`, and the steering owner's Controller that applies contract 2 to it."""
        self.steerer = Steerer(embedding)
        self.controller = Controller(self.steerer, self.encoder.encode, cache_size=CONTROLLER_CACHE)
        if self.reanchor_state is ReanchorState.ON:
            self.policy = ReanchorPolicy()

    # --- re-anchor -------------------------------------------------------------------------

    def _reanchor_support(self) -> ReanchorState:
        """The re-anchor state for config.reanchor and the loaded backend; logs once when it cannot re-anchor."""
        if self.config.reanchor == "off":
            return ReanchorState.OFF
        if callable(getattr(self.backend, "reanchor", None)):
            log.info("re-anchor on: the policy's anchors go to backend.reanchor(strength)")
            return ReanchorState.ON
        log.warning("re-anchor requested but backend %s cannot re-anchor (no reanchor(strength)); running without it",
                    self.config.backend)
        return ReanchorState.UNSUPPORTED

    def _generate(self, cond, anchor: Anchor | None):
        """The one backend call per chunk. `anchor` is None unless reanchor_state is ON."""
        if anchor is not None:
            self.backend.reanchor(anchor.strength)  # takes effect on this generate_chunk
        return self.backend.generate_chunk(cond)

    def _take_cond(self) -> tuple[object, float, Anchor | None]:
        """Under the CommandBook lock: this chunk's cond and progress, and the policy's anchor for it."""
        cond, progress = self.steerer.get_cond(), self.steerer.progress
        anchor = self.policy.next_chunk(progress, self.steerer.is_moving) if self.policy is not None else None
        return cond, progress, anchor

    def _apply_target(self, target: dict) -> dict:
        """Under the CommandBook lock: Controller.handle, then tell the policy what it did."""
        result = self.controller.handle(target)
        if self.policy is not None:
            self.policy.on_target(target, result)
        return result

    # --- generation ------------------------------------------------------------------------

    def _lead_limit(self) -> int:
        """Frames of video that must remain ahead of the playhead when the next chunk starts generating."""
        fpc = self.backend.frames_per_chunk
        with self._stats_lock:
            g, d, j = self.gen_ms.avg, self.decode_ms.avg, self.jpeg_ms.avg
        if g is None or d is None:
            return 2 * fpc
        pipeline_s = (g + d + (j or 0.0)) / 1e3
        return max(1, math.ceil(pipeline_s * self.backend.fps * self.config.lead_margin))

    def _generation_loop(self) -> None:
        backend = self.backend
        chunk = 0
        last_cond = None
        while not self._stop.is_set():
            if self._crashed.is_set():
                return
            if self._reset_waiters:
                self._do_reset()
                last_cond = None
            self._gate.wait(self._lead_limit, self._stop, self._crashed)
            if self._stop.is_set() or self._crashed.is_set():
                break
            cond, progress, anchor = self.book.take_cond(self._take_cond, chunk)
            if cond is not last_cond:
                log.debug("chunk %d: conditioning changed (progress %.2f)", chunk, progress)
            last_cond = cond
            t0 = time.perf_counter()
            latents = self._generate(cond, anchor)
            gen_ms = (time.perf_counter() - t0) * 1e3
            self.book.chunk_generated(chunk)
            with self._stats_lock:
                self.gen_ms.add(gen_ms)
                self.chunk_generated = chunk
                if anchor is not None:
                    self.reanchors += 1
                    self.last_anchor, self.last_anchor_chunk = anchor, chunk
            if anchor is not None:
                log.info("chunk %d: re-anchor %s at strength %.2f (progress %.2f)", chunk, anchor.reason,
                         anchor.strength, progress)
            est = backend.frames_per_chunk
            self._gate.add(est)
            meta = ChunkMeta(chunk=chunk, progress=progress, gen_ms=gen_ms, est_frames=est, anchor=anchor,
                             video=self.video)
            if not self._put(self._decode_queue, (latents, meta), f"decode is behind, generation blocked at chunk {chunk}",
                             self._crashed):
                break
            chunk += 1

    def _put(self, q: queue.Queue, item, blocked_msg: str, abort: threading.Event | None = None) -> bool:
        """Put with backpressure: block (and log how long) when the queue is full. False if stopping or `abort`."""
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            pass
        t0 = time.perf_counter()
        while not self._stop.is_set() and not (abort is not None and abort.is_set()):
            try:
                q.put(item, timeout=0.1)
                log.info("%s for %.0f ms", blocked_msg, (time.perf_counter() - t0) * 1e3)
                if q is self._decode_queue:
                    with self._stats_lock:
                        self.decode_blocked += 1
                return True
            except queue.Full:
                continue
        return False

    # --- reset -----------------------------------------------------------------------------

    def request_reset(self, seed: tuple[dict, CommandRecord] | None = None) -> threading.Event:
        """Ask for a reset at the next chunk boundary. The returned event is set when it is done.

        `seed` is (steer, command): the new video starts directly on the steer's target (its prompt, and its
        strength pair if it has one), with no ramp. Without it the new video starts on the starting prompt.
        The steer's prompts must already be in the encoder cache, so the reset does not wait for encoding.
        """
        state = self.lifecycle.state
        if state not in (Lifecycle.GENERATING, Lifecycle.RESETTING):
            raise BadMessage(f"cannot reset while {state.value}")
        done = threading.Event()
        with self._reset_lock:
            self._reset_waiters.append(done)
            self._reset_seed = seed
        return done

    def _do_reset(self) -> None:
        with self._reset_lock:
            waiters, self._reset_waiters = self._reset_waiters, []
            seed, self._reset_seed = self._reset_seed, None
        steer, rec = seed if seed else ({"prompt": self.base_prompt}, None)
        self.lifecycle.transition(Lifecycle.RESETTING, f"reset requested, new video on {steer['prompt']!r}")
        try:
            self._reset_now(steer, rec)
        finally:  # also after an error (the engine then recovers), so POST /reset does not hang
            for w in waiters:
                w.set()

    def _reset_now(self, steer: dict, rec: CommandRecord | None) -> None:
        t0 = time.perf_counter()
        self._decode_queue.join()  # the decoder finishes the chunks of the old video first
        self.backend.reset()
        self.video += 1  # the recorder starts a new file at this video's first frame

        def restart() -> None:
            """Controller.reset keeps its prompt cache and jumps the steerer to the new target with no ramp."""
            try:
                self.controller.reset(steer["prompt"], steer.get("strength"), steer.get("without_prompt"))
            except Exception:
                log.exception("controller.reset failed for %r; starting on the starting prompt", steer["prompt"])
                self.controller.reset(self.base_prompt)
            if self.policy is not None:
                self.policy.on_reset()

        self.book.reset(restart, rec)
        self.lifecycle.transition(Lifecycle.GENERATING, f"reset done in {(time.perf_counter() - t0) * 1e3:.0f} ms")

    # --- decode ----------------------------------------------------------------------------

    def _decode_loop(self) -> None:
        backend, quality = self.backend, self.config.jpeg_quality
        try:
            while not self._stop.is_set():
                try:
                    latents, meta = self._decode_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if self._crashed.is_set():
                        continue  # the video is broken; recovery drops it
                    t0 = time.perf_counter()
                    frames = backend.decode_chunk(latents)
                    t1 = time.perf_counter()
                    jpegs = [encode_jpeg(f, quality) for f in frames]
                    t2 = time.perf_counter()
                    meta.decode_ms, meta.jpeg_ms, meta.frames = (t1 - t0) * 1e3, (t2 - t1) * 1e3, len(jpegs)
                    log.info("chunk %d: gen %.0f ms, decode %.0f ms, jpeg %.0f ms, %d frames, decode queue %d",
                             meta.chunk, meta.gen_ms, meta.decode_ms, meta.jpeg_ms, meta.frames,
                             self._decode_queue.qsize())
                    if not jpegs:
                        log.warning("chunk %d decoded to 0 frames", meta.chunk)
                        self._gate.add(-meta.est_frames)
                    for i, jpeg in enumerate(jpegs):
                        if not self._put(self._playout_queue, (i, jpeg, meta), "playout buffer full, decode blocked"):
                            return
                except Exception as exc:
                    self._report_crash(f"decode thread at chunk {meta.chunk}", exc)
                finally:
                    self._decode_queue.task_done()
        except Exception as exc:
            self._fail("decode thread", exc)

    # --- streaming -------------------------------------------------------------------------

    def _stream_loop(self) -> None:
        interval = 1.0 / self.backend.fps
        next_due: float | None = None
        try:
            while not self._stop.is_set():
                try:
                    index, jpeg, meta = self._playout_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                now = time.monotonic()
                if next_due is None:
                    next_due = now
                elif now > next_due + interval / 2:
                    late_ms = (now - next_due) * 1e3
                    self.stalls += 1
                    log.info("stream stalled %.0f ms before chunk %d frame %d: the frame was not ready in time "
                             "(generation plus decode slower than real time, or startup)", late_ms, meta.chunk, index)
                    next_due = now
                if next_due > now:
                    time.sleep(next_due - now)
                next_due += interval
                self._send_frame(index, jpeg, meta)
        except Exception as exc:
            self._fail("streamer thread", exc)

    def _send_frame(self, index: int, jpeg: bytes, meta: ChunkMeta) -> None:
        self.publish_frame(pack_frame(meta.chunk, index, jpeg))
        self.recorder.offer(meta.video, jpeg)
        self._gate.sent(1)
        with self._stats_lock:
            self.frames_sent += 1
            self._sent_times.append(time.monotonic())
        if index != 0:
            return
        pending = self._resume_pending
        if pending is not None and pending[0] == meta.video:
            self._resume_pending = None
            ms = (time.monotonic() - pending[1]) * 1e3
            with self._stats_lock:
                self.last_recovery = {"video": meta.video, "chunk": meta.chunk, "resumed_ms": round(ms)}
            log.info("recovery: video resumed %.0f ms after the crash (chunk %d, video %d)", ms, meta.chunk, meta.video)
        self._gate.add(meta.frames - meta.est_frames)  # correct the estimate made at generation time
        self.book.chunk_shown(meta.chunk)
        with self._stats_lock:
            self.chunk_shown = meta.chunk
            self.decode_ms.add(meta.decode_ms)
            self.jpeg_ms.add(meta.jpeg_ms)
        self.publish_control(self.state_message(meta.chunk, meta.progress))
        if meta.chunk > 0 and meta.chunk % self.config.summary_every == 0:
            s = self.stats()
            log.info("summary at chunk %d: gen avg %s ms, decode avg %s ms, jpeg avg %s ms, %s fps, "
                     "lead limit %d frames, stalls %d, decode blocked %d, clients %s",
                     meta.chunk, s["gen_ms"]["avg"], s["decode_ms"]["avg"], s["jpeg_ms"]["avg"], s["fps"],
                     self._lead_limit(), s["stalls"], s["decode_blocked"], s["clients"])

    def state_message(self, chunk: int, progress: float) -> dict:
        """One contract 4 message: the reader's state fields plus what the engine measures."""
        try:
            state = dict(self.reader.state_fields())
        except Exception:
            log.exception("reader.state_fields failed")
            state = {"changes": [], "heard": None, "keyword": None, "unmatched": False}
        return {**state, "chunk": chunk, "progress": round(min(1.0, max(0.0, progress)), 3),
                **self.book.latest_fields()}

    # --- commands --------------------------------------------------------------------------

    def _require_running(self) -> None:
        state = self.lifecycle.state
        if state not in (Lifecycle.GENERATING, Lifecycle.RESETTING):
            raise BadMessage(f"engine is {state.value}, not accepting commands")

    def handle_control(self, message: object, heard_ns: int,
                       reply: Callable[[dict], None] = lambda _m: None) -> Future:
        """A contract 1 message from /control.

        Checks it against the schema here (raises BadMessage), then hands it to the reader worker thread.
        `reply` is called from that thread with {"error": ...} if the reader rejects it.
        The returned Future gives the reset's done event for a reset tap, else None.
        """
        msg = check(1, message)
        self._require_running()
        rec = self.book.heard("control", msg.get("text"), msg.get("t_client"), heard_ns)
        return self._reader_pool.submit(self._read, msg, rec, reply)

    def _read(self, msg: dict, rec: CommandRecord, reply: Callable[[dict], None]) -> threading.Event | None:
        """Reader worker thread: reader.handle, then steer, or reset for a reset tap."""
        is_reset = msg["type"] == "tap" and msg.get("action") == "reset"
        try:
            result = self.reader.handle(msg)
            steer = result.get("steer")
            if steer is not None:
                check_target(steer)
        except (ValueError, BadMessage) as exc:  # BadMessage is a ValueError too
            self.book.drop(rec, f"reader rejected it: {exc}")
            reply({"error": f"command reader: {exc}"})
            return None
        except Exception as exc:
            log.exception("reader.handle failed")
            self.book.drop(rec, f"reader error: {exc!r}")
            reply({"error": f"command reader failed: {exc!r}"})
            return None
        self.encoder.prefetch(self._likely_prompts())
        if is_reset:
            seed = None
            if steer is not None:
                self.book.decided(rec, steer)
                for prompt in (steer["prompt"], steer.get("without_prompt")):
                    if prompt:
                        self.encoder.encode(prompt)  # into the cache now, so the reset itself does not wait
                seed = (steer, rec)
            else:
                self.book.drop(rec, "reset to the starting prompt")
            try:
                return self.request_reset(seed)
            except BadMessage as exc:
                self.book.drop(rec, str(exc))
                reply({"error": str(exc)})
                return None
        if steer is None:
            self.book.drop(rec, f"no steer for {msg['type']}")
            return None
        self.book.decided(rec, steer)
        self._embed(rec, steer)
        return None

    def reset(self, heard_ns: int) -> Future:
        """POST /reset: the same path as a reset tap, so the reader resets its scene too."""
        return self.handle_control({"type": "tap", "action": "reset"}, heard_ns)

    def handle_target(self, message: object, heard_ns: int) -> CommandRecord:
        """A contract 2 message from POST /target, bypassing the reader. Passed to the Controller unchanged."""
        target = check_target(message)
        self._require_running()
        rec = self.book.heard("target", None, None, heard_ns)
        self.book.decided(rec, target)
        self._embed(rec, target)
        return rec

    def _embed(self, rec: CommandRecord, target: dict) -> None:
        """Encode every prompt the target names, then apply it on the steer worker thread."""
        futures = [self.encoder.request(p) for p in (target["prompt"], target.get("without_prompt")) if p]
        remaining = [len(futures)]
        lock = threading.Lock()

        def one_done(_fut) -> None:  # runs on the encoder thread, or right here if already cached
            with lock:
                remaining[0] -= 1
                last = remaining[0] == 0
            if last:
                self._steer_pool.submit(self._on_embedding, rec, target, futures)

        for fut in futures:
            fut.add_done_callback(one_done)

    def _on_embedding(self, rec: CommandRecord, target: dict, futures: list) -> None:
        """Steer worker thread. Once the embeddings are in the encoder cache, Controller.handle does not wait for
        encoding while the engine's lock is held."""
        for fut in futures:
            try:
                fut.result()
            except Exception as exc:
                self.book.drop(rec, f"encoding failed: {exc!r}")
                return
        try:
            result = self.book.embedded(rec, lambda: self._apply_target(target))
        except ValueError as exc:
            self.book.drop(rec, f"controller refused: {exc}")
            return
        if result is not None:
            log.info("command %d: controller %s", rec.seq, result)

    # --- stats -----------------------------------------------------------------------------

    def stats(self) -> dict:
        with self._stats_lock:
            now = time.monotonic()
            recent = [t for t in self._sent_times if now - t <= 5.0]
            fps = None
            if len(recent) >= 2 and now - recent[0] > 0:
                fps = round((len(recent) - 1) / (recent[-1] - recent[0]), 2) if recent[-1] > recent[0] else None
            out = {
                "state": self.lifecycle.state.value,
                "failure": self.failure,
                "backend": self.config.backend,
                "reader": self.config.reader,
                "chunk_generated": self.chunk_generated,
                "chunk_shown": self.chunk_shown,
                "gen_ms": {"last": _round(self.gen_ms.last), "avg": _round(self.gen_ms.avg)},
                "decode_ms": {"last": _round(self.decode_ms.last), "avg": _round(self.decode_ms.avg)},
                "jpeg_ms": {"last": _round(self.jpeg_ms.last), "avg": _round(self.jpeg_ms.avg)},
                "fps": fps,
                "frames_sent": self.frames_sent,
                "stalls": self.stalls,
                "decode_blocked": self.decode_blocked,
                "decode_queue": self._decode_queue.qsize(),
                "playout_frames": self._playout_queue.qsize(),
                "frames_ahead": self._gate.ahead,
                "reanchor": {
                    "state": self.reanchor_state.value,
                    "count": self.reanchors,
                    "last_strength": None if self.last_anchor is None else self.last_anchor.strength,
                    "last_reason": None if self.last_anchor is None else self.last_anchor.reason,
                    "last_chunk": self.last_anchor_chunk,
                },
                "video": self.video,
                "recovery": {"count": self.recoveries, "last": self.last_recovery,
                             "crashes_in_window": len(self._crash_times)},
            }
        out["reanchor"]["chunks_since_anchor"] = None if self.policy is None else self.policy.chunks_since_anchor
        if self.backend is not None:
            out.update(fps_target=self.backend.fps, frames_per_chunk=self.backend.frames_per_chunk,
                       width=self.backend.width, height=self.backend.height)
        if self.encoder is not None:
            out["encoder"] = {"cache_size": self.encoder.cache_size, "pending": self.encoder.pending, "hits": self.encoder.hits,
                              "misses": self.encoder.misses, "last_encode_ms": _round(self.encoder.last_encode_ms)}
        out["recorder"] = self.recorder.stats()
        out["clients"] = self.client_counts()
        return out


CONTROLLER_CACHE = 512


class EngineFailed(RuntimeError):
    """Too many crashes in a short time: in-process recovery gave up."""


def check_target(message: object) -> dict:
    """Contract 2 schema, plus the Controller's rule that a final (non-provisional) message has cap 1.0."""
    target = check(2, message)
    if not target["provisional"] and target["cap"] != 1.0:
        raise BadMessage(f"a final (non-provisional) target must have cap 1.0, got {target['cap']}")
    return target


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    buf = BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

"""The engine: generation thread, decode thread, streamer thread, and the command flow into the steerer.

    generation thread   cond = steerer.get_cond(); latents = backend.generate_chunk(cond); decode_queue.put(...)
    decode thread       decode_queue -> backend.decode_chunk -> JPEG -> playout_queue  (no other shared state)
    streamer thread     playout_queue -> paced at backend.fps -> publish(frame); on a chunk's first frame,
                        stamp `shown` and publish one contract 4 message

Commands: contract 1 -> command reader -> contract 2 -> encoder service (its own thread) -> steer worker thread ->
steering.Controller.handle (which calls steerer.set_target) -> takes effect on the next get_cond.
The frame cache is never rebuilt on a command; only the steerer's target changes.
"""

import collections
import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from steering import Controller, Steerer

from . import standin_reader
from .backends import load_backend
from .commands import CommandBook, CommandRecord
from .encoder import EncoderService
from .framing import pack_frame
from .lifecycle import Lifecycle, LifecycleMachine
from .messages import BadMessage, check

log = logging.getLogger("engine")


@dataclass
class EngineConfig:
    backend: str = "fake"
    width: int = 832
    height: int = 480
    steps: int = 4
    device: str = "cuda:0"
    decode_device: str = "cuda:0"
    backend_options: dict = field(default_factory=dict)
    base_prompt: str = "a wide shot of a meadow with a single oak tree"
    jpeg_quality: int = 80
    precompute_file: str | None = None
    decode_queue_size: int = 2      # chunks waiting for decode; generation blocks (and logs) when full
    playout_max_frames: int = 64    # hard bound on decoded frames waiting to be sent
    lead_margin: float = 1.3        # see LeadGate
    summary_every: int = 20         # chunks between rolling summary log lines


@dataclass
class ChunkMeta:
    """Travels with a chunk through both queues."""
    chunk: int
    progress: float
    gen_ms: float
    est_frames: int
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

    def wait(self, limit: Callable[[], int], stop: threading.Event) -> None:
        with self._cond:
            while self.ahead > limit() and not stop.is_set():
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
                 publish_control: Callable[[dict], None], reader=standin_reader.read_command,
                 scene=None) -> None:
        self.config = config
        self.publish_frame = publish_frame
        self.publish_control = publish_control
        self.read_command = reader
        self.scene = scene if scene is not None else standin_reader.StandinScene()
        self.lifecycle = LifecycleMachine()
        self.book = CommandBook()
        self.encoder: EncoderService | None = None
        self.backend = None
        self.steerer: Steerer | None = None
        self.controller: Controller | None = None
        self.base_embedding = None
        self.failure: str | None = None

        self._stop = threading.Event()
        self._decode_queue: queue.Queue = queue.Queue(maxsize=config.decode_queue_size)
        self._playout_queue: queue.Queue = queue.Queue(maxsize=config.playout_max_frames)
        self._gate = LeadGate()
        self._reset_lock = threading.Lock()
        self._reset_waiters: list[threading.Event] = []
        self._threads: list[threading.Thread] = []
        # Runs Controller.handle off the generation thread and the asyncio loop.
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
        self._steer_pool.shutdown(wait=False, cancel_futures=True)
        if self.encoder:
            self.encoder.stop()
        self.lifecycle.transition(Lifecycle.STOPPED, "shutdown")

    def _fail(self, what: str, exc: BaseException) -> None:
        log.exception("%s failed", what)
        self.failure = f"{what}: {exc!r}"
        if self.lifecycle.state not in (Lifecycle.FAILED, Lifecycle.STOPPED):
            self.lifecycle.transition(Lifecycle.FAILED, self.failure)
        self._stop.set()

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
            self.lifecycle.transition(Lifecycle.WARMING, "backend loaded")
            self.encoder = EncoderService(self.backend.encode_prompt_with_ids)
            self.encoder.start()
            self.base_embedding = self.encoder.encode(cfg.base_prompt)
            self._precompute()
            self._new_steerer()
            for target, name in ((self._decode_loop, "decode"), (self._stream_loop, "streamer")):
                t = threading.Thread(target=target, name=name, daemon=True)
                self._threads.append(t)
                t.start()
            if self._stop.is_set():
                return
            self.lifecycle.transition(Lifecycle.GENERATING, "warm")
            self._generation_loop()
        except Exception as exc:
            self._fail("generation thread", exc)

    def _precompute(self) -> None:
        path = self.config.precompute_file
        if not path:
            return
        prompts = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
        t0 = time.perf_counter()
        for p in prompts:
            if self._stop.is_set():
                return
            self.encoder.encode(p)
        log.info("precomputed %d prompts from %s in %.1f s", len(prompts), path, time.perf_counter() - t0)

    def _new_steerer(self) -> None:
        """A steerer at the base prompt, and the steering owner's Controller that applies contract 2 to it."""
        self.steerer = Steerer(self.base_embedding)
        self.controller = Controller(self.steerer, self.encoder.encode, cache_size=CONTROLLER_CACHE)

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
            if self._reset_waiters:
                self._do_reset()
                last_cond = None
            self._gate.wait(self._lead_limit, self._stop)
            if self._stop.is_set():
                break
            cond, progress = self.book.take_cond(lambda: (self.steerer.get_cond(), self.steerer.progress), chunk)
            if cond is not last_cond:
                log.debug("chunk %d: conditioning changed (progress %.2f)", chunk, progress)
            last_cond = cond
            t0 = time.perf_counter()
            latents = backend.generate_chunk(cond)
            gen_ms = (time.perf_counter() - t0) * 1e3
            self.book.chunk_generated(chunk)
            with self._stats_lock:
                self.gen_ms.add(gen_ms)
                self.chunk_generated = chunk
            est = backend.frames_per_chunk
            self._gate.add(est)
            meta = ChunkMeta(chunk=chunk, progress=progress, gen_ms=gen_ms, est_frames=est)
            if not self._put(self._decode_queue, (latents, meta), f"decode is behind, generation blocked at chunk {chunk}"):
                break
            chunk += 1

    def _put(self, q: queue.Queue, item, blocked_msg: str) -> bool:
        """Put with backpressure: block (and log how long) when the queue is full. False if stopping."""
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            pass
        t0 = time.perf_counter()
        while not self._stop.is_set():
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

    def request_reset(self) -> threading.Event:
        """Ask for a reset at the next chunk boundary. The returned event is set when it is done."""
        state = self.lifecycle.state
        if state not in (Lifecycle.GENERATING, Lifecycle.RESETTING):
            raise BadMessage(f"cannot reset while {state.value}")
        done = threading.Event()
        with self._reset_lock:
            self._reset_waiters.append(done)
        return done

    def _do_reset(self) -> None:
        with self._reset_lock:
            waiters, self._reset_waiters = self._reset_waiters, []
        self.lifecycle.transition(Lifecycle.RESETTING, "reset requested")
        t0 = time.perf_counter()
        self._decode_queue.join()  # the decoder finishes the chunks of the old video first
        self.backend.reset()
        self.book.reset(self._new_steerer)
        self.lifecycle.transition(Lifecycle.GENERATING, f"reset done in {(time.perf_counter() - t0) * 1e3:.0f} ms")
        for w in waiters:
            w.set()

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
        self._gate.sent(1)
        with self._stats_lock:
            self.frames_sent += 1
            self._sent_times.append(time.monotonic())
        if index != 0:
            return
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
        """One contract 4 message."""
        return {"chunk": chunk, "changes": [dict(c) for c in self.scene.changes],
                "progress": round(min(1.0, max(0.0, progress)), 3), **self.book.latest_fields()}

    # --- commands --------------------------------------------------------------------------

    def _require_running(self) -> None:
        state = self.lifecycle.state
        if state not in (Lifecycle.GENERATING, Lifecycle.RESETTING):
            raise BadMessage(f"engine is {state.value}, not accepting commands")

    def handle_control(self, message: object, heard_ns: int) -> None:
        """A contract 1 message from /control. Raises BadMessage if it is invalid."""
        msg = check(1, message)
        self._require_running()
        if msg["type"] == "tap" and msg["action"] == "reset":
            self.request_reset()
        rec = self.book.heard("control", msg.get("text"), msg.get("t_client"), heard_ns)
        try:
            target = self.read_command(msg, self.scene)
            if target is None:
                self.book.no_operation(rec, f"no operation for {msg['type']}", report=msg["type"] == "final")
                return
            check_target(target)
        except Exception as exc:
            self.book.drop(rec, f"reader error: {exc}")
            raise BadMessage(f"command reader failed: {exc}") from None
        self.book.decided(rec, target, keyword=getattr(self.scene, "keyword", None))
        self._embed(rec, target)

    def handle_target(self, message: object, heard_ns: int) -> CommandRecord:
        """A contract 2 message from POST /target, bypassing the reader."""
        target = check_target(message)
        self._require_running()
        rec = self.book.heard("target", None, None, heard_ns)
        self.book.decided(rec, target)
        self._embed(rec, target)
        return rec

    def _embed(self, rec: CommandRecord, target: dict) -> None:
        fut = self.encoder.request(target["prompt"])
        fut.add_done_callback(lambda f: self._steer_pool.submit(self._on_embedding, rec, target, f))

    def _on_embedding(self, rec: CommandRecord, target: dict, fut) -> None:
        """Steer worker thread. The embedding is now in the encoder cache, so handle does not wait for encoding."""
        if fut.exception() is not None:
            self.book.drop(rec, f"encoding failed: {fut.exception()!r}")
            return
        try:
            result = self.book.embedded(rec, lambda: self.controller.handle(target))
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
            }
        if self.backend is not None:
            out.update(fps_target=self.backend.fps, frames_per_chunk=self.backend.frames_per_chunk,
                       width=self.backend.width, height=self.backend.height)
        if self.encoder is not None:
            out["encoder"] = {"cache_size": self.encoder.cache_size, "hits": self.encoder.hits,
                              "misses": self.encoder.misses, "last_encode_ms": _round(self.encoder.last_encode_ms)}
        out["clients"] = self.client_counts()
        return out


CONTROLLER_CACHE = 512


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

"""A CPU fake backend whose picture depends on the conditioning tensor.

encode_prompt hashes the prompt into a deterministic tensor of shape [1, 16, 32].
The background colour comes from token 0 and the moving ball's colour from token 1, so as the steerer blends
the conditioning over a few chunks, the colours change smoothly.
generate_chunk and decode_chunk sleep for configurable times to stand in for GPU work.
reanchor(strength, source="last") stands in for the real model's re-anchor: it is recorded (reanchor_calls,
last_reanchor_strength, last_reanchor_source; source "clean" is the refresh from the clean anchor)
and the next chunk's frames get a white border and a "re-anchor" label, so a browser test can see it.

Options (pass as --backend-opt key=value):
    gen_ms=170, decode_ms=60, encode_ms=80, frames_per_chunk=4, first_chunk_frames=1, fps=16
    fail_on_chunk=-1, fail_count=1, fail_in=generate, shared_device=false, short_on_chunks=none

shared_device=true models one GPU that runs the text encoder, generation and decode, as on the real machine
(umT5-xxl and the generator share one GPU). Their simulated work then shares one device by fair time sharing:
while n jobs run at once, each advances at 1/n speed. So a prompt encoded while a chunk is generating makes that
chunk take longer. Without it every job sleeps its full time independently.

fail_on_chunk=N injects a crash to test recovery: from chunk N on (chunks count across resets), generate_chunk
(fail_in=generate) or decode_chunk (fail_in=decode) raises RuntimeError, fail_count times in total.

short_on_chunks=40,80 makes those chunks (numbered as fail_on_chunk, across resets) 1 frame long, as the real model's
chunk after a rebase decodes to only its first frame. Their frames say "short chunk", so a recording shows them.

Model passes, to measure the cost of a cut on the CPU (all off by default):
    pass_ms=0, denoise_passes=2, cut_passes=4, defer_context=false, cut_fade_frames=0
pass_ms > 0 replaces the single gen_ms sleep with the real model's passes, pass_ms each: denoise_passes denoising
passes and one context pass (the KV-cache write) per chunk, and cut_passes denoising passes for a re-anchor chunk
(4 today, 2 with the real backend's steer_steps=1000,500). defer_context=true runs a chunk's context pass at the start
of the next chunk and skips it when that chunk re-anchors, as the real backend's option does.
interrupt() stops the chunk in flight at its next pass boundary with ChunkInterrupted, but only a chunk that
continues the video (not a re-anchor chunk, not the first chunk after reset); the stopped call changes nothing,
so the next call takes the same chunk number. With pass_ms=0 a chunk is one pass and has no boundary.
cut_fade_frames=N dissolves into a re-anchor chunk from the last frame decoded before it over N frames, with the
real decoder's formula, so the engine's flush can be checked against it.
"""

import hashlib
import threading
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

TOKENS, DIM = 16, 32


class SharedDevice:
    """One simulated GPU shared by fair time sharing: while n jobs run at once, each advances at 1/n speed."""

    STEP_S = 0.002

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs = 0

    def run(self, seconds: float) -> None:
        """Block until `seconds` of work is done on this device."""
        with self._lock:
            self._jobs += 1
        try:
            left, last = seconds, time.perf_counter()
            while left > 0:
                time.sleep(min(self.STEP_S, left))
                now = time.perf_counter()
                with self._lock:
                    jobs = self._jobs
                left -= (now - last) / jobs
                last = now
        finally:
            with self._lock:
                self._jobs -= 1


class ChunkInterrupted(RuntimeError):
    """generate_chunk was stopped by interrupt(); the engine knows it by `chunk_interrupted`."""

    chunk_interrupted = True


def load(width: int, height: int, steps: int, device: str = "cpu", decode_device: str = "cpu", **options):
    return FakeBackend(width, height, steps, **options)


def _chunk_numbers(value: str | int | None) -> frozenset[int]:
    """`short_on_chunks` as a set: None or "none" for no chunks, an int, or comma-separated ints like "40,80"."""
    if value is None:
        return frozenset()
    if isinstance(value, int):
        return frozenset({value})
    text = str(value).strip()
    if text.lower() in ("", "none"):
        return frozenset()
    try:
        return frozenset(int(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise ValueError(f"short_on_chunks expects comma-separated chunk numbers, got {value!r}") from None


def _flag(value: bool | str) -> bool:
    """A --backend-opt flag: a bool, or a string such as 1, true, yes, on."""
    return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")


def _rgb(vec: torch.Tensor) -> np.ndarray:
    """Map the first three values of a token vector to a colour, smoothly."""
    return (127.5 + 127.5 * torch.tanh(vec[:3].float() * 1.5)).numpy()


class FakeBackend:
    def __init__(self, width: int, height: int, steps: int, gen_ms: float = 170, decode_ms: float = 60,
                 encode_ms: float = 80, frames_per_chunk: int = 4, first_chunk_frames: int = 1, fps: int = 16,
                 fail_on_chunk: int = -1, fail_count: int = 1, fail_in: str = "generate",
                 shared_device: bool = False, short_on_chunks: str | int | None = None, pass_ms: float = 0,
                 denoise_passes: int = 2, cut_passes: int = 4, defer_context: bool | str = False,
                 cut_fade_frames: int = 0):
        self.width, self.height, self.steps = int(width), int(height), int(steps)
        self.fps = int(fps)
        self.frames_per_chunk = int(frames_per_chunk)
        self.first_chunk_frames = int(first_chunk_frames)
        self.gen_s, self.decode_s, self.encode_s = float(gen_ms) / 1e3, float(decode_ms) / 1e3, float(encode_ms) / 1e3
        self.device = SharedDevice() if shared_device else None
        self.pass_s = float(pass_ms) / 1e3  # 0: a chunk is one gen_ms sleep
        self.denoise_passes, self.cut_passes = int(denoise_passes), int(cut_passes)
        if self.denoise_passes < 1 or self.cut_passes < 1:
            raise ValueError("denoise_passes and cut_passes must be at least 1")
        self.defer_context = _flag(defer_context)
        self._context_pending = False  # defer_context: the last chunk's context pass has not run yet
        self.context_passes_skipped = 0
        self._interrupt = threading.Event()
        self.interrupted_chunks = 0
        self.cut_fade_frames = int(cut_fade_frames)
        if self.cut_fade_frames < 0:
            raise ValueError(f"cut_fade_frames must be >= 0, got {cut_fade_frames}")
        self._last_frame: np.ndarray | None = None  # decoder state: the last frame decoded since reset
        self._fade_from: np.ndarray | None = None  # the frame a running dissolve starts from
        self._fade_done = 0
        self._chunks_generated = 0       # counts every generate_chunk call; not cleared by reset
        self._chunks_since_reset = 0     # the "frame cache": restarts the motion after reset
        self._frame_phase = 0            # decoder state: frames decoded since reset
        self.reanchor_calls = 0          # reanchor() calls; not cleared by reset
        self.last_reanchor_strength: float | None = None
        self.last_reanchor_source: str | None = None
        self._anchor_next: float | None = None  # strength the next generate_chunk re-anchors with, once
        if fail_in not in ("generate", "decode"):
            raise ValueError(f"fail_in must be generate or decode, got {fail_in!r}")
        self.fail_on_chunk, self.fail_in = int(fail_on_chunk), fail_in
        self.failures_left = int(fail_count) if self.fail_on_chunk >= 0 else 0
        self.short_on_chunks = _chunk_numbers(short_on_chunks)
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        self._yy, self._xx = yy.astype(np.float32), xx.astype(np.float32)
        self._shade = (0.75 + 0.25 * (self._yy / max(1, self.height - 1)))[..., None]
        try:
            self._font = ImageFont.load_default(size=max(14, self.height // 18))
        except TypeError:
            self._font = ImageFont.load_default()

    def _work(self, seconds: float) -> None:
        """Stand-in for GPU work: on the shared device if there is one, else a plain sleep."""
        if self.device is not None:
            self.device.run(seconds)
        elif seconds > 0:
            time.sleep(seconds)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        self._work(self.encode_s)
        seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "big")
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(1, TOKENS, DIM, generator=gen)

    def encode_prompt_with_ids(self, prompt: str) -> tuple[torch.Tensor, list[int]]:
        """The embedding plus deterministic stand-in token ids (one per word, from a hash)."""
        ids = [int.from_bytes(hashlib.sha256(w.encode()).digest()[:2], "big") for w in prompt.split()]
        return self.encode_prompt(prompt), ids

    def reset(self) -> None:
        self._chunks_since_reset = 0
        self._frame_phase = 0
        self._context_pending = False
        self._interrupt.clear()
        self._last_frame, self._fade_from, self._fade_done = None, None, 0

    def interrupt(self) -> None:
        """Any thread: stop the chunk in flight at its next pass boundary (see the module docstring)."""
        self._interrupt.set()

    def reanchor(self, strength: float, source: str = "last") -> None:
        """Re-anchor the next generate_chunk with `strength` in (0, 1], from the "last" latent or the "clean" anchor."""
        strength = float(strength)
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"reanchor strength must be in (0, 1], got {strength!r}")
        if source not in ("last", "clean"):
            raise ValueError(f"reanchor source must be 'last' or 'clean', got {source!r}")
        self.reanchor_calls += 1
        self.last_reanchor_strength = strength
        self.last_reanchor_source = source
        self._anchor_next = strength

    def _maybe_fail(self, where: str, chunk: int) -> None:
        if where == self.fail_in and self.failures_left > 0 and chunk >= self.fail_on_chunk:
            self.failures_left -= 1
            raise RuntimeError(f"fake backend: injected {where} failure at chunk {chunk}")

    def _passes(self) -> None:
        """The chunk's model passes (pass_ms > 0); raises ChunkInterrupted at a boundary when interrupt() was called."""
        self._interrupt.clear()  # an interrupt that missed the previous chunk is moot
        anchor = self._anchor_next is not None
        stoppable = not anchor and self._chunks_since_reset > 0
        done = 0
        if self._context_pending:
            self._context_pending = False
            if anchor:
                self.context_passes_skipped += 1  # the re-anchor drops the frame cache it would write
            else:
                self._work(self.pass_s)
                done += 1
        for _ in range(self.cut_passes if anchor else self.denoise_passes):
            self._stop_if_interrupted(done, stoppable)
            self._work(self.pass_s)
            done += 1
        if self.defer_context:
            self._context_pending = True
        else:
            self._stop_if_interrupted(done, stoppable)
            self._work(self.pass_s)

    def _stop_if_interrupted(self, done: int, stoppable: bool) -> None:
        if done and stoppable and self._interrupt.is_set():
            self._interrupt.clear()
            self.interrupted_chunks += 1
            raise ChunkInterrupted(f"fake chunk {self._chunks_generated} stopped at a pass boundary by interrupt()")

    def generate_chunk(self, cond: torch.Tensor) -> dict:
        if self.pass_s > 0:
            self._passes()
        else:
            self._interrupt.clear()  # one pass: no boundary to stop at
            self._work(self.gen_s)
        self._maybe_fail("generate", self._chunks_generated)
        short = self._chunks_since_reset > 0 and self._chunks_generated in self.short_on_chunks
        if self._chunks_since_reset == 0:
            n = self.first_chunk_frames
        else:
            n = 1 if short else self.frames_per_chunk
        latents = {"number": self._chunks_generated, "frames": n, "short": short,
                   "background": _rgb(cond[0, 0]), "ball": _rgb(cond[0, 1]), "anchor": self._anchor_next,
                   "fade": self._anchor_next is not None and self.cut_fade_frames > 0}
        self._anchor_next = None
        self._chunks_generated += 1
        self._chunks_since_reset += 1
        return latents

    def decode_chunk(self, latents: dict) -> np.ndarray:
        t0 = time.perf_counter()
        self._maybe_fail("decode", latents["number"])
        frames = self._fade_in(self._draw(latents), latents)
        self._work(self.decode_s - (time.perf_counter() - t0))
        return frames

    def _draw(self, latents: dict) -> np.ndarray:
        """The chunk's frames: the background, the moving ball, and the labels."""
        frames = np.empty((latents["frames"], self.height, self.width, 3), dtype=np.uint8)
        bg = (self._shade * latents["background"]).astype(np.uint8)
        r = self.height / 8
        for i in range(latents["frames"]):
            phase = self._frame_phase / self.fps  # seconds since reset
            cx = self.width / 2 + self.width * 0.35 * np.sin(phase * 1.3)
            cy = self.height / 2 + self.height * 0.3 * np.sin(phase * 2.1)
            frame = bg.copy()
            inside = (self._xx - cx) ** 2 + (self._yy - cy) ** 2 <= r * r
            frame[inside] = latents["ball"].astype(np.uint8)
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            short = "  short chunk" if latents["short"] else ""
            draw.text((12, 10), f"fake chunk {latents['number']}  frame {i}{short}", fill=(255, 255, 255),
                      font=self._font, stroke_width=2, stroke_fill=(0, 0, 0))
            if latents["anchor"] is not None:
                border = max(4, self.height // 40)
                draw.rectangle((0, 0, self.width - 1, self.height - 1), outline=(255, 255, 255), width=border)
                draw.text((12, 10 + self.height // 12), f"re-anchor {latents['anchor']:.2f}", fill=(255, 255, 255),
                          font=self._font, stroke_width=2, stroke_fill=(0, 0, 0))
            frames[i] = np.asarray(img)
            self._frame_phase += 1
        return frames

    def _fade_in(self, frames: np.ndarray, latents: dict) -> np.ndarray:
        """Decoder state after drawing: the dissolve into a re-anchor chunk (cut_fade_frames), and the last frame."""
        if latents["fade"] and self._last_frame is not None:
            self._fade_from, self._fade_done = self._last_frame, 0
        elif latents["anchor"] is not None:
            self._fade_from, self._fade_done = None, 0  # a cut without a fade ends a running dissolve
        if self._fade_from is not None and len(frames):
            frames = self._fade(frames)
        if len(frames):
            self._last_frame = frames[-1]
        return frames

    def _fade(self, out: np.ndarray) -> np.ndarray:
        """The real decoder's dissolve: output frame k after the cut is (1 - a) * old + a * new, a = (k + 1) / (N + 1)."""
        n = self.cut_fade_frames
        k = np.arange(len(out), dtype=np.float32) + self._fade_done
        a = np.minimum((k + 1) / (n + 1), 1.0).reshape(-1, 1, 1, 1)
        out = np.rint(self._fade_from.astype(np.float32) * (1 - a) + out.astype(np.float32) * a).astype(np.uint8)
        self._fade_done += len(out)
        if self._fade_done >= n:
            self._fade_from = None
        return out

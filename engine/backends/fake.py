"""A CPU fake backend whose picture depends on the conditioning tensor.

encode_prompt hashes the prompt into a deterministic tensor of shape [1, 16, 32].
The background colour comes from token 0 and the moving ball's colour from token 1, so as the steerer blends
the conditioning over a few chunks, the colours change smoothly.
generate_chunk and decode_chunk sleep for configurable times to stand in for GPU work.
reanchor(strength) stands in for the real model's re-anchor: it is recorded (reanchor_calls, last_reanchor_strength)
and the next chunk's frames get a white border and a "re-anchor" label, so a browser test can see it.

Options (pass as --backend-opt key=value):
    gen_ms=170, decode_ms=60, encode_ms=80, frames_per_chunk=4, first_chunk_frames=1, fps=16
    fail_on_chunk=-1, fail_count=1, fail_in=generate, shared_device=false

shared_device=true models one GPU that runs the text encoder, generation and decode, as on the real machine
(umT5-xxl and the generator share one GPU). Their simulated work then shares one device by fair time sharing:
while n jobs run at once, each advances at 1/n speed. So a prompt encoded while a chunk is generating makes that
chunk take longer. Without it every job sleeps its full time independently.

fail_on_chunk=N injects a crash to test recovery: from chunk N on (chunks count across resets), generate_chunk
(fail_in=generate) or decode_chunk (fail_in=decode) raises RuntimeError, fail_count times in total.
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


def load(width: int, height: int, steps: int, device: str = "cpu", decode_device: str = "cpu", **options):
    return FakeBackend(width, height, steps, **options)


def _rgb(vec: torch.Tensor) -> np.ndarray:
    """Map the first three values of a token vector to a colour, smoothly."""
    return (127.5 + 127.5 * torch.tanh(vec[:3].float() * 1.5)).numpy()


class FakeBackend:
    def __init__(self, width: int, height: int, steps: int, gen_ms: float = 170, decode_ms: float = 60,
                 encode_ms: float = 80, frames_per_chunk: int = 4, first_chunk_frames: int = 1, fps: int = 16,
                 fail_on_chunk: int = -1, fail_count: int = 1, fail_in: str = "generate",
                 shared_device: bool = False):
        self.width, self.height, self.steps = int(width), int(height), int(steps)
        self.fps = int(fps)
        self.frames_per_chunk = int(frames_per_chunk)
        self.first_chunk_frames = int(first_chunk_frames)
        self.gen_s, self.decode_s, self.encode_s = float(gen_ms) / 1e3, float(decode_ms) / 1e3, float(encode_ms) / 1e3
        self.device = SharedDevice() if shared_device else None
        self._chunks_generated = 0       # counts every generate_chunk call; not cleared by reset
        self._chunks_since_reset = 0     # the "frame cache": restarts the motion after reset
        self._frame_phase = 0            # decoder state: frames decoded since reset
        self.reanchor_calls = 0          # reanchor() calls; not cleared by reset
        self.last_reanchor_strength: float | None = None
        self._anchor_next: float | None = None  # strength the next generate_chunk re-anchors with, once
        if fail_in not in ("generate", "decode"):
            raise ValueError(f"fail_in must be generate or decode, got {fail_in!r}")
        self.fail_on_chunk, self.fail_in = int(fail_on_chunk), fail_in
        self.failures_left = int(fail_count) if self.fail_on_chunk >= 0 else 0
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

    def reanchor(self, strength: float) -> None:
        """Re-anchor the next generate_chunk with `strength` in (0, 1]."""
        strength = float(strength)
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"reanchor strength must be in (0, 1], got {strength!r}")
        self.reanchor_calls += 1
        self.last_reanchor_strength = strength
        self._anchor_next = strength

    def _maybe_fail(self, where: str, chunk: int) -> None:
        if where == self.fail_in and self.failures_left > 0 and chunk >= self.fail_on_chunk:
            self.failures_left -= 1
            raise RuntimeError(f"fake backend: injected {where} failure at chunk {chunk}")

    def generate_chunk(self, cond: torch.Tensor) -> dict:
        self._work(self.gen_s)
        self._maybe_fail("generate", self._chunks_generated)
        n = self.first_chunk_frames if self._chunks_since_reset == 0 else self.frames_per_chunk
        latents = {"number": self._chunks_generated, "frames": n,
                   "background": _rgb(cond[0, 0]), "ball": _rgb(cond[0, 1]), "anchor": self._anchor_next}
        self._anchor_next = None
        self._chunks_generated += 1
        self._chunks_since_reset += 1
        return latents

    def decode_chunk(self, latents: dict) -> np.ndarray:
        t0 = time.perf_counter()
        self._maybe_fail("decode", latents["number"])
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
            draw.text((12, 10), f"fake chunk {latents['number']}  frame {i}", fill=(255, 255, 255),
                      font=self._font, stroke_width=2, stroke_fill=(0, 0, 0))
            if latents["anchor"] is not None:
                border = max(4, self.height // 40)
                draw.rectangle((0, 0, self.width - 1, self.height - 1), outline=(255, 255, 255), width=border)
                draw.text((12, 10 + self.height // 12), f"re-anchor {latents['anchor']:.2f}", fill=(255, 255, 255),
                          font=self._font, stroke_width=2, stroke_fill=(0, 0, 0))
            frames[i] = np.asarray(img)
            self._frame_phase += 1
        self._work(self.decode_s - (time.perf_counter() - t0))
        return frames

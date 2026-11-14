"""A CPU fake backend whose picture depends on the conditioning tensor.

encode_prompt hashes the prompt into a deterministic tensor of shape [1, 16, 32].
The background colour comes from token 0 and the moving ball's colour from token 1, so as the steerer blends
the conditioning over a few chunks, the colours change smoothly.
generate_chunk and decode_chunk sleep for configurable times to stand in for GPU work.

Options (pass as --backend-opt key=value):
    gen_ms=170, decode_ms=60, encode_ms=80, frames_per_chunk=4, first_chunk_frames=1, fps=16
"""

import hashlib
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

TOKENS, DIM = 16, 32


def load(width: int, height: int, steps: int, device: str = "cpu", decode_device: str = "cpu", **options):
    return FakeBackend(width, height, steps, **options)


def _rgb(vec: torch.Tensor) -> np.ndarray:
    """Map the first three values of a token vector to a colour, smoothly."""
    return (127.5 + 127.5 * torch.tanh(vec[:3].float() * 1.5)).numpy()


class FakeBackend:
    def __init__(self, width: int, height: int, steps: int, gen_ms: float = 170, decode_ms: float = 60,
                 encode_ms: float = 80, frames_per_chunk: int = 4, first_chunk_frames: int = 1, fps: int = 16):
        self.width, self.height, self.steps = int(width), int(height), int(steps)
        self.fps = int(fps)
        self.frames_per_chunk = int(frames_per_chunk)
        self.first_chunk_frames = int(first_chunk_frames)
        self.gen_s, self.decode_s, self.encode_s = float(gen_ms) / 1e3, float(decode_ms) / 1e3, float(encode_ms) / 1e3
        self._chunks_generated = 0       # counts every generate_chunk call; not cleared by reset
        self._chunks_since_reset = 0     # the "frame cache": restarts the motion after reset
        self._frame_phase = 0            # decoder state: frames decoded since reset
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        self._yy, self._xx = yy.astype(np.float32), xx.astype(np.float32)
        self._shade = (0.75 + 0.25 * (self._yy / max(1, self.height - 1)))[..., None]
        try:
            self._font = ImageFont.load_default(size=max(14, self.height // 18))
        except TypeError:
            self._font = ImageFont.load_default()

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        time.sleep(self.encode_s)
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

    def generate_chunk(self, cond: torch.Tensor) -> dict:
        time.sleep(self.gen_s)
        n = self.first_chunk_frames if self._chunks_since_reset == 0 else self.frames_per_chunk
        latents = {"number": self._chunks_generated, "frames": n,
                   "background": _rgb(cond[0, 0]), "ball": _rgb(cond[0, 1])}
        self._chunks_generated += 1
        self._chunks_since_reset += 1
        return latents

    def decode_chunk(self, latents: dict) -> np.ndarray:
        t0 = time.perf_counter()
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
            frames[i] = np.asarray(img)
            self._frame_phase += 1
        left = self.decode_s - (time.perf_counter() - t0)
        if left > 0:
            time.sleep(left)
        return frames

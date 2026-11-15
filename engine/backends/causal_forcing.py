"""Causal Forcing / Causal Forcing++ backend for the real-time engine.

The model is the official Causal Forcing code (github.com/thu-ml/Causal-Forcing, Apache-2.0)
on top of Wan2.1-T2V-1.3B. The third-party code is NOT vendored here: it is imported from a
checkout whose path comes from the `repo_dir` option or the CAUSAL_FORCING_REPO environment
variable (default ~/third_party/Causal-Forcing). That checkout must contain two symlinks made
by engine/setup_gpu.sh: `wan_models` -> the weights dir and `checkpoints` -> the Causal Forcing
weights dir, because the official code opens weights through those relative paths.

One generate_chunk() call produces one block of latent frames:
- variant "framewise_2step" (Causal Forcing++, the default): 1 latent frame = 4 video frames
  = 0.25 s at 16 fps. The first call of a video uses the 4-step first-frame schedule the
  checkpoint was distilled with and yields 1 video frame after decode.
- variant "chunkwise_4step" (Causal Forcing): 3 latent frames = 12 video frames = 0.75 s.

Endless generation: the frame (self-attention) KV cache is a fixed-size rolling window of
`attn_window_frames` latent frames (plus optional `sink_frames` that are never evicted). The
eviction logic is the official code's local-attention path (local_attn_size / sink_size).
Keys are stored with RoPE already applied at their absolute frame index, so the only thing
that grows is the frame index. The official RoPE table has 1024 positions, which would end
generation after 1024 latent frames. This backend rebuilds the same table with
`max_latent_frames` positions (default 65536), with the same formula. With a pure rolling
window (sink_frames=0) attention only sees relative distances inside the window, which is
what the model was trained on.

Conditioning: each call is conditioned on `cond`, the umT5-xxl embedding [1, 512, 4096]
bf16 (padding tokens zeroed), passed as conditional_dict["prompt_embeds"]. The Wan model
caches the cross-attention K/V of the text in a second cache (crossattn_cache, flag
"is_init"). generate_chunk() resets that cache whenever `cond` is a different object from
the one used for the previous call, so a new prompt takes effect on the very next chunk.
The frame KV cache is never touched by a conditioning change.

Threads and devices: generation, prompt encoding and decoding each run on their own CUDA
stream. The decoder object owns all decode state and receives only the ChunkLatents passed
in, so it can move to `decode_device` or to another process later.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import torch

DEFAULT_REPO_DIR = os.path.expanduser(os.environ.get("CAUSAL_FORCING_REPO", "~/third_party/Causal-Forcing"))
DEFAULT_TAEHV_PATH = os.path.expanduser(os.environ.get("TAEHV_CHECKPOINT", "~/weights/taehv/taew2_1.pth"))

FPS = 16
VIDEO_FRAMES_PER_LATENT = 4  # Wan VAE temporal compression
LATENT_CHANNELS = 16
SPATIAL_DOWNSAMPLE = 8  # VAE
PATCH = 2  # transformer patch size on latents (so pixels must be multiples of 16)
NUM_BLOCKS = 30
NUM_HEADS = 12
HEAD_DIM = 128
TEXT_LEN = 512

# Official configs and checkpoints, paths relative to repo_dir.
VARIANTS = {
    "framewise_2step": {
        "config": "configs/causal_forcing_dmd_framewise_2step.yaml",
        "checkpoint": "checkpoints/causal-forcing++/framewise-2step.pt",
        "ema": True,
    },
    "framewise_1step": {
        "config": "configs/causal_forcing_dmd_framewise_1step.yaml",
        "checkpoint": "checkpoints/causal-forcing++/framewise-1step.pt",
        "ema": True,
    },
    "chunkwise_4step": {
        "config": "configs/causal_forcing_dmd_chunkwise.yaml",
        "checkpoint": "checkpoints/chunkwise/causal_forcing.pt",
        "ema": False,
    },
}


class VaeType(str, enum.Enum):
    WAN = "wan"  # official Wan2.1 VAE decoder, streaming with a feature cache (demo_utils/vae_block3.py)
    TAEHV = "taehv"  # tiny decoder shipped by the official demo (demo_utils/taehv.py)


class GenState(enum.Enum):
    EMPTY = "empty"  # no frame generated since load() or reset(); next call starts a new video
    STREAMING = "streaming"  # frame cache holds past frames; next call continues the video
    REANCHOR_PENDING = "reanchor_pending"  # next call drops the frame history and restarts it from the last frame


class DecoderState(enum.Enum):
    EMPTY = "empty"  # no chunk of the current video decoded yet
    STREAMING = "streaming"  # streaming decode state holds the previous chunks of this video


@dataclasses.dataclass(frozen=True)
class ChunkLatents:
    """What generate_chunk returns and decode_chunk consumes. Never modified after creation."""

    latents: torch.Tensor  # [1, F, 16, H/8, W/8], generation device
    video_id: int  # increases on every reset(); the decoder starts fresh when it changes
    chunk_index: int  # 0 for the first chunk of a video
    ready: torch.cuda.Event  # recorded on the generation stream after `latents` was written


@contextlib.contextmanager
def _in_dir(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _import_repo(repo_dir: str):
    repo_dir = os.path.abspath(os.path.expanduser(repo_dir))
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"Causal Forcing checkout not found at {repo_dir}; set CAUSAL_FORCING_REPO or repo_dir")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    return repo_dir


def _timestep_list(scheduler, steps_1000: list[int], warp: bool) -> torch.Tensor:
    """Same mapping as pipeline/causal_inference.py: optional warp onto the shifted schedule."""
    t = torch.tensor(steps_1000, dtype=torch.long)
    if warp:
        timesteps = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        t = timesteps[1000 - t]
    return t


class _Decoder:
    """Streaming VAE decode. Owns all of its state; only ChunkLatents cross the boundary."""

    def __init__(self, repo_dir: str, vae_type: VaeType, device: torch.device, latent_hw: tuple[int, int], dtype=torch.float16):
        self.vae_type = vae_type
        self.device = device
        self.dtype = dtype
        self.latent_hw = latent_hw
        self.stream = torch.cuda.Stream(device=device)
        self.state = DecoderState.EMPTY
        self.video_id: Optional[int] = None
        self.next_chunk_index = 0
        self._wan_cache: Optional[list[torch.Tensor]] = None
        self._taehv_context: Optional[torch.Tensor] = None  # last latents, used as temporal context

        with torch.cuda.device(device):
            if vae_type is VaeType.WAN:
                from demo_utils import vae_block3
                from demo_utils.vae_block3 import VAEDecoderWrapper
                from demo_utils.constant import ZERO_VAE_CACHE

                # The official wrapper prints a debug line on every call; silence it in that module only.
                vae_block3.print = lambda *a, **k: None

                model = VAEDecoderWrapper()
                sd = torch.load("wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", map_location="cpu")
                model.load_state_dict({k: v for k, v in sd.items() if "decoder." in k or "conv2" in k})
                # The official zero cache is sized for 60x104 latents (480x832). Scale its spatial
                # dims to this resolution; channels and the 2-frame temporal depth stay the same.
                lh, lw = latent_hw
                self._zero_cache_shapes = [
                    (t.shape[0], t.shape[1], t.shape[2], t.shape[3] * lh // 60, t.shape[4] * lw // 104) for t in ZERO_VAE_CACHE
                ]
            else:
                from demo_utils.taehv import TAEHV

                model = TAEHV(checkpoint_path=DEFAULT_TAEHV_PATH)
            self.model = model.eval().requires_grad_(False).to(device=device, dtype=dtype)

    def _start_video(self, video_id: int):
        self.video_id = video_id
        self.next_chunk_index = 0
        self.state = DecoderState.EMPTY
        self._wan_cache = None
        self._taehv_context = None

    @torch.no_grad()
    def decode(self, chunk: ChunkLatents) -> np.ndarray:
        if chunk.video_id != self.video_id:
            self._start_video(chunk.video_id)
        if chunk.chunk_index != self.next_chunk_index:
            raise ValueError(f"decode_chunk got chunk {chunk.chunk_index}, expected {self.next_chunk_index} (chunks must be decoded in order)")

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.stream.wait_event(chunk.ready)
            z = chunk.latents
            if z.device == self.device:
                z.record_stream(self.stream)
            z = z.to(device=self.device, dtype=self.dtype, non_blocking=True)

            if self.vae_type is VaeType.WAN:
                if self.state is DecoderState.EMPTY:
                    self._wan_cache = [torch.zeros(s, device=self.device, dtype=self.dtype) for s in self._zero_cache_shapes]
                pixels, cache = self.model(z, *self._wan_cache)
                self._wan_cache = list(cache)
                if self.state is DecoderState.EMPTY:
                    pixels = pixels[:, 3:]  # official demo: drop the first 3 frames of a video
            else:
                n_ctx = 0 if self._taehv_context is None else self._taehv_context.shape[1]
                full = z if n_ctx == 0 else torch.cat([self._taehv_context, z], dim=1)
                self._taehv_context = full[:, -3:]
                pixels = self.model.decode_video(full, parallel=False).mul_(2).sub_(1)
                # official demo: drop 3 frames on the first block, else the frames of the context latents
                drop = 3 if self.state is DecoderState.EMPTY else n_ctx * VIDEO_FRAMES_PER_LATENT
                pixels = pixels[:, drop:]

            # [1, T, 3, H, W] in [-1, 1] -> uint8 [T, H, W, 3]
            frames = ((pixels[0].float().clamp_(-1, 1) + 1.0) * 127.5).round_().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
            out = frames.cpu().numpy()  # waits for this stream only

        self.state = DecoderState.STREAMING
        self.next_chunk_index += 1
        return out


class Backend:
    fps: int = FPS

    def __init__(
        self,
        width: int,
        height: int,
        steps: Optional[int],
        device: str,
        decode_device: str,
        *,
        variant: str = "framewise_2step",
        repo_dir: str = DEFAULT_REPO_DIR,
        vae: str = "wan",
        attn_window_frames: int = 21,
        sink_frames: int = 0,
        max_latent_frames: int = 65536,
        reanchor_every_frames: int = 0,
        reanchor_strength: float = 0.6,
        text_encoder_device: Optional[str] = None,
        compile: bool = False,
        seed: int = 0,
    ):
        if width % 16 or height % 16:
            raise ValueError("width and height must be multiples of 16 (VAE x8, then patch x2)")
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}; choose from {sorted(VARIANTS)}")
        self.width, self.height = width, height
        self.variant = variant
        self.device = torch.device(device)
        self.decode_device = torch.device(decode_device)
        self.text_device = torch.device(text_encoder_device or device)
        self.dtype = torch.bfloat16
        self.latent_hw = (height // SPATIAL_DOWNSAMPLE, width // SPATIAL_DOWNSAMPLE)
        self.frame_seq_length = (self.latent_hw[0] // PATCH) * (self.latent_hw[1] // PATCH)
        self.attn_window_frames = attn_window_frames
        self.sink_frames = sink_frames
        self.max_latent_frames = max_latent_frames
        if sink_frames >= attn_window_frames:
            raise ValueError("sink_frames must be smaller than attn_window_frames")
        self.reanchor_every_frames = reanchor_every_frames  # 0: no periodic re-anchor
        self.reanchor_strength = float(reanchor_strength)

        repo_dir = _import_repo(repo_dir)
        from omegaconf import OmegaConf

        spec = VARIANTS[variant]
        with _in_dir(repo_dir):
            from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
            from wan.modules.model import rope_params

            cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(spec["config"]))
            self.num_frame_per_block = int(getattr(cfg, "num_frame_per_block", 1))
            self.frames_per_chunk = self.num_frame_per_block * VIDEO_FRAMES_PER_LATENT
            self.context_noise = int(cfg.context_noise)

            # Generator (causal Wan 1.3B) with the official rolling local-attention KV cache.
            model_kwargs = dict(getattr(cfg, "model_kwargs", {}))
            gen = WanDiffusionWrapper(**model_kwargs, is_causal=True, local_attn_size=attn_window_frames, sink_size=sink_frames)
            sd = torch.load(spec["checkpoint"], map_location="cpu")
            sd = sd["generator_ema" if spec["ema"] else "generator"]
            sd = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in sd.items()}
            gen.load_state_dict(sd)
            del sd
            if self.num_frame_per_block > 1:
                gen.model.num_frame_per_block = self.num_frame_per_block
            # Same RoPE formula as the official model (wan/modules/causal_model.py), more positions.
            d = gen.model.dim // gen.model.num_heads
            gen.model.freqs = torch.cat(
                [
                    rope_params(max_latent_frames, d - 4 * (d // 6)),
                    rope_params(max_latent_frames, 2 * (d // 6)),
                    rope_params(max_latent_frames, 2 * (d // 6)),
                ],
                dim=1,
            )
            self.generator = gen.to(device=self.device, dtype=self.dtype)
            self.scheduler = gen.get_scheduler()

            # Denoising schedules.
            warp = bool(cfg.warp_denoising_step)
            self._warp = warp
            distilled = [int(t) for t in cfg.denoising_step_list]
            if steps is None or steps == len(distilled):
                chosen = distilled
            else:
                # Evenly spaced subset of [0, 1000), the usual way to run a DMD model with fewer steps.
                chosen = [int(round(1000 - i * 1000 / steps)) for i in range(steps)]
            self.steps = len(chosen)
            self.denoising_steps = _timestep_list(self.scheduler, chosen, warp)
            first = getattr(cfg, "denoising_step_list_first_chunk", None)
            self.first_chunk_steps = _timestep_list(self.scheduler, [int(t) for t in first], warp) if first is not None else self.denoising_steps

            # Text encoder (umT5-xxl), its own lock and stream so it can run beside generation.
            te = WanTextEncoder()
            self.text_encoder = te.to(device=self.text_device, dtype=self.dtype).eval().requires_grad_(False)
            self._text_lock = threading.Lock()
            self._text_stream = torch.cuda.Stream(device=self.text_device)

            self.decoder = _Decoder(repo_dir, VaeType(vae), self.decode_device, self.latent_hw)

        if compile:
            self.generator.model.compile(mode="max-autotune-no-cudagraphs")

        self._gen_lock = threading.Lock()
        self._gen_stream = torch.cuda.Stream(device=self.device)
        self._rng = torch.Generator(device=self.device).manual_seed(seed)
        self._init_caches()
        self.state = GenState.EMPTY
        self.video_id = 0
        self.chunk_index = 0
        self.frame_index = 0  # latent frames generated in this video (absolute RoPE position)
        self._last_cond: Optional[torch.Tensor] = None
        self.crossattn_resets = 0  # how many times the text K/V cache was recomputed (for tests)
        self._last_latent: Optional[torch.Tensor] = None  # clean latents of the previous call
        self._frames_since_anchor = 0  # latent frames since the video start or the last re-anchor
        self._reanchor_strength: Optional[float] = None  # strength of the pending re-anchor
        self.reanchors = 0  # how many re-anchors ran (for tests)

    # ---- caches -------------------------------------------------------------------------------

    def _init_caches(self):
        size = self.attn_window_frames * self.frame_seq_length
        with torch.cuda.device(self.device):
            self.kv_cache = [
                {
                    "k": torch.zeros([1, size, NUM_HEADS, HEAD_DIM], dtype=self.dtype, device=self.device),
                    "v": torch.zeros([1, size, NUM_HEADS, HEAD_DIM], dtype=self.dtype, device=self.device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                }
                for _ in range(NUM_BLOCKS)
            ]
            self.crossattn_cache = [
                {
                    "k": torch.zeros([1, TEXT_LEN, NUM_HEADS, HEAD_DIM], dtype=self.dtype, device=self.device),
                    "v": torch.zeros([1, TEXT_LEN, NUM_HEADS, HEAD_DIM], dtype=self.dtype, device=self.device),
                    "is_init": False,
                }
                for _ in range(NUM_BLOCKS)
            ]

    def _clear_frame_cache(self):
        for c in self.kv_cache:
            c["global_end_index"].zero_()
            c["local_end_index"].zero_()

    def _reanchor_schedule(self, strength: float) -> torch.Tensor:
        """Timesteps for a re-anchored frame: start at `strength` (0..1 of the noise range,
        warped like the distilled schedule), then the first-frame steps below it."""
        raw = min(1000, max(1, int(round(strength * 1000))))
        start = _timestep_list(self.scheduler, [raw], self._warp)[0]
        rest = [t for t in self.first_chunk_steps.tolist() if t < float(start) - 1e-3]
        return torch.tensor([float(start)] + rest)

    def _reset_crossattn_cache(self):
        for c in self.crossattn_cache:
            c["is_init"] = False
        self.crossattn_resets += 1

    # ---- public interface -----------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt_with_ids(self, prompt: str) -> tuple[torch.Tensor, list[int]]:
        """Embedding [1, 512, 4096] (padding rows zeroed) plus the real umT5 token ids, in order.

        ids[i] belongs to embedding row i; padding is not included. Same steps as the official
        WanTextEncoder.forward, done here so the ids are returned too.
        """
        with self._text_lock, torch.cuda.device(self.text_device), torch.cuda.stream(self._text_stream):
            te = self.text_encoder
            ids, mask = te.tokenizer([prompt], return_mask=True, add_special_tokens=True)
            ids = ids.to(self.text_device)
            mask = mask.to(self.text_device)
            n = int(mask.gt(0).sum().item())
            emb = te.text_encoder(ids, mask)
            emb[:, n:] = 0.0
            emb = emb.to(device=self.device, dtype=self.dtype)
            self._text_stream.synchronize()
        return emb, ids[0, :n].tolist()

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        return self.encode_prompt_with_ids(prompt)[0]

    def reset(self, seed: Optional[int] = None) -> None:
        """Start a new video. With `seed`, the noise of the new video is reseeded, so the same
        seed and the same conds give the same latents again."""
        with self._gen_lock:
            if seed is not None:
                self._rng.manual_seed(int(seed))
            with torch.cuda.device(self.device), torch.cuda.stream(self._gen_stream):
                for c in self.kv_cache:
                    c["global_end_index"].zero_()
                    c["local_end_index"].zero_()
            self._reset_crossattn_cache()
            self._last_cond = None
            self._last_latent = None
            self._reanchor_strength = None
            self._frames_since_anchor = 0
            self.frame_index = 0
            self.chunk_index = 0
            self.video_id += 1
            self.state = GenState.EMPTY

    def request_reanchor(self, strength: float) -> None:
        """Re-anchor on the next generate_chunk call: the frame history is dropped and the
        next frame starts from the last frame noised to `strength` (0..1), denoised under the
        cond of that call with the first-frame schedule. Layout comes from the last frame,
        everything the text asks for can change. Higher strength changes more.
        The video continues for the viewer (same video_id, chunk_index keeps counting, the
        decoder keeps its state). Ignored before the first frame of a video."""
        if not 0.0 < strength <= 1.0:
            raise ValueError("strength must be in (0, 1]")
        with self._gen_lock:
            if self.state is GenState.EMPTY:
                return
            self._reanchor_strength = float(strength)
            self.state = GenState.REANCHOR_PENDING

    @torch.no_grad()
    def generate_chunk(self, cond: torch.Tensor) -> ChunkLatents:
        with self._gen_lock, torch.cuda.device(self.device), torch.cuda.stream(self._gen_stream):
            if self.frame_index + self.num_frame_per_block > self.max_latent_frames:
                raise RuntimeError("reached max_latent_frames; call reset() or load with a larger max_latent_frames")
            # `cond` may have been written on another thread's stream (the steerer's blend).
            self._gen_stream.wait_stream(torch.cuda.default_stream(self.device))
            if cond is not self._last_cond:
                self._reset_crossattn_cache()
                self._last_cond = cond
            cond_dict = {"prompt_embeds": cond}

            f = self.num_frame_per_block
            lh, lw = self.latent_hw
            if (self.state is GenState.STREAMING and self.reanchor_every_frames > 0
                    and self._frames_since_anchor >= self.reanchor_every_frames):
                # Periodic re-anchor: a named transition, same as request_reanchor().
                self._reanchor_strength = self.reanchor_strength
                self.state = GenState.REANCHOR_PENDING

            noise = torch.randn([1, f, LATENT_CHANNELS, lh, lw], device=self.device, dtype=self.dtype, generator=self._rng)
            if self.state is GenState.EMPTY:
                noisy, schedule = noise, self.first_chunk_steps
            elif self.state is GenState.REANCHOR_PENDING:
                # Restart the frame history from the last frame (see request_reanchor).
                self._clear_frame_cache()
                self.frame_index = 0
                self._frames_since_anchor = 0
                self.reanchors += 1
                schedule = self._reanchor_schedule(self._reanchor_strength)
                prev = self._last_latent[:, -f:]
                noisy = self.scheduler.add_noise(
                    prev.flatten(0, 1), noise.flatten(0, 1), schedule[0] * torch.ones([f], device=self.device, dtype=torch.long)
                ).unflatten(0, prev.shape[:2])
                self._reanchor_strength = None
            else:
                noisy, schedule = noise, self.denoising_steps
            current_start = self.frame_index * self.frame_seq_length

            for i, t in enumerate(schedule):
                timestep = torch.ones([1, f], device=self.device, dtype=torch.int64) * t
                _, denoised = self.generator(
                    noisy_image_or_video=noisy,
                    conditional_dict=cond_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                )
                if i < len(schedule) - 1:
                    nxt = schedule[i + 1]
                    noisy = self.scheduler.add_noise(
                        denoised.flatten(0, 1),
                        torch.randn(denoised.flatten(0, 1).shape, device=self.device, dtype=denoised.dtype, generator=self._rng),
                        nxt * torch.ones([f], device=self.device, dtype=torch.long),
                    ).unflatten(0, denoised.shape[:2])

            # Write the clean frame into the frame KV cache (official step 3.3).
            self.generator(
                noisy_image_or_video=denoised,
                conditional_dict=cond_dict,
                timestep=torch.ones([1, f], device=self.device, dtype=torch.int64) * self.context_noise,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start,
            )

            ready = torch.cuda.Event()
            ready.record(self._gen_stream)
            out = ChunkLatents(latents=denoised, video_id=self.video_id, chunk_index=self.chunk_index, ready=ready)
            self.frame_index += f
            self._frames_since_anchor += f
            self.chunk_index += 1
            self._last_latent = denoised
            self.state = GenState.STREAMING
            return out

    def decode_chunk(self, latents: ChunkLatents) -> np.ndarray:
        return self.decoder.decode(latents)

    # ---- helpers for benchmarks -------------------------------------------------------------------

    def sync_generation(self) -> None:
        """Block until all queued generation work is done (for timing)."""
        self._gen_stream.synchronize()


def load(width: int, height: int, steps: Optional[int] = None, device: str = "cuda:0", decode_device: str = "cuda:0", **options) -> Backend:
    """Build the backend. `steps=None` or 0 uses the schedule the checkpoint was distilled for.

    Options: variant ("framewise_2step" default, "framewise_1step", "chunkwise_4step"),
    repo_dir, vae ("wan" or "taehv"), attn_window_frames (21), sink_frames (0),
    max_latent_frames (65536), reanchor_every_frames (0: off), reanchor_strength (0.6),
    text_encoder_device, compile (False), seed (0).
    """
    # Options may arrive as strings from the server's --backend-opt KEY=VALUE.
    types = {"attn_window_frames": int, "sink_frames": int, "max_latent_frames": int, "seed": int,
             "reanchor_every_frames": int, "reanchor_strength": float,
             "compile": lambda v: v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")}
    options = {k: types[k](v) if k in types else v for k, v in options.items()}
    steps = int(steps) if steps is not None and int(steps) > 0 else None  # 0 or None: distilled schedule
    t0 = time.time()
    b = Backend(int(width), int(height), steps, device, decode_device, **options)
    b.load_seconds = time.time() - t0
    return b

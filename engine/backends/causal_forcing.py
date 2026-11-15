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
The frame KV cache is not touched by a conditioning change, unless a rebase is asked for:
generate_chunk(cond, rebase=Rebase(...)) or the automatic options (steer_rebase, refresh_every)
clear the frame cache and re-encode the last few clean frames under the current cond at
positions starting from 0 (LongLive's KV recache, arXiv 2509.22622). All of that is off by
default, and then the calls, the RNG draws and their order are exactly what they were before.

Clean anchor: reanchor(strength) (a steer) starts the new first frame from the last latent;
reanchor(strength, source="clean") (the periodic refresh) starts it from the clean anchor, the
latent saved `clean_anchor_after` (3) chunks after the last reset or steer re-anchor. The last
latent has already drifted (over-contrast, blocky grass) and a refresh from it bakes that in; a
refresh from the clean anchor does not (measured, MODEL_NOTES.md).

Threads and devices: generation, prompt encoding and decoding each run on their own CUDA
stream. The decoder object owns all decode state and receives only the ChunkLatents passed
in, so it can move to `decode_device` or to another process later.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import enum
import inspect
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


class ChunkStart(enum.Enum):
    CONTINUE = "continue"  # the latent continues the previous one (decoded with the stream's context)
    FIRST_FRAME = "first_frame"  # a first-frame latent: the first chunk of a video, or a hard cut (keep-0 rebase)


class FirstFramePad(str, enum.Enum):
    HOLD = "hold"  # a hard cut mid-video emits its 1 clean frame repeated to the usual chunk length
    NONE = "none"  # a hard cut mid-video emits its 1 frame, as the first chunk of a video does


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
    start: ChunkStart = ChunkStart.CONTINUE  # FIRST_FRAME: the decoder restarts its stream state for it
    fade: bool = False  # a re-anchor cut to dissolve into from the last shown frame (cut_fade_frames > 0)


@dataclasses.dataclass(frozen=True)
class Rebase:
    """Restart the frame cache before a chunk, keeping the last `keep` clean frames (LongLive's
    KV recache, arXiv 2509.22622, with positions restarted at 0). Plain ints, so callers can build
    one, or a plain dict with the same keys, without importing torch.

    keep: most recent clean latent frames re-encoded under the current cond at positions
        0..keep-1 (clamped to what exists and to attn_window_frames - 1). 0 = hard cut: the
        next frame is made as the first frame of a video.
    memory_noise: 0, or a timestep on the 0-1000 scale; the kept frames are re-encoded noised
        to that level (weaker memory) instead of clean.
    anchor_step: None, or n: the new frame starts from the last clean latent noised to step n of
        its schedule and is denoised with schedule[n:] (n = 0 is pure noise).
    from_clean: with anchor_step > 0, start from the clean anchor (see Backend.reanchor) instead
        of the last latent; falls back to the last latent when there is no clean anchor yet.
    """

    keep: int = 3
    memory_noise: int = 0
    anchor_step: Optional[int] = None
    from_clean: bool = False

    @classmethod
    def of(cls, value) -> Optional["Rebase"]:
        if value is None or isinstance(value, Rebase):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"rebase must be a Rebase, a dict or None, got {type(value).__name__}")


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

    def __init__(self, repo_dir: str, vae_type: VaeType, device: torch.device, latent_hw: tuple[int, int], dtype=torch.float16,
                 first_frame_pad: FirstFramePad = FirstFramePad.HOLD, frames_per_chunk: int = VIDEO_FRAMES_PER_LATENT,
                 cut_fade_frames: int = 0):
        self.vae_type = vae_type
        self.cut_fade_frames = int(cut_fade_frames)  # dissolve length after a cut chunk with `fade`; 0 = off
        self.first_frame_pad = FirstFramePad(first_frame_pad)
        self.frames_per_chunk = frames_per_chunk
        self.restarts = 0  # mid-video stream restarts (hard cuts), for tests and logs
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
        self._restart_stream()
        self._last_frame: Optional[np.ndarray] = None  # the last frame emitted in this video, uint8 [H, W, 3]
        self._fade_from: Optional[np.ndarray] = None  # the frame a running dissolve starts from
        self._fade_done = 0  # output frames of the running dissolve already emitted

    def _restart_stream(self):
        """Forget the decoded past (as for a new video), keeping video_id and the chunk count."""
        self.state = DecoderState.EMPTY
        self._wan_cache = None
        self._taehv_context = None

    @torch.no_grad()
    def decode(self, chunk: ChunkLatents) -> np.ndarray:
        if chunk.video_id != self.video_id:
            self._start_video(chunk.video_id)
        if chunk.chunk_index != self.next_chunk_index:
            raise ValueError(f"decode_chunk got chunk {chunk.chunk_index}, expected {self.next_chunk_index} (chunks must be decoded in order)")
        # A hard cut makes a first-frame latent. Decoding it as a continuation, with the old scene's
        # context, gives a washed-out, ghosted chunk and a soft next one; decode it as a new video's
        # first chunk instead (1 frame), padded to the usual length unless first_frame_pad is "none".
        cut = chunk.start is ChunkStart.FIRST_FRAME and self.state is DecoderState.STREAMING
        if cut:
            self._restart_stream()
            self.restarts += 1
            self._fade_from, self._fade_done = None, 0  # a cut without `fade` also ends a running dissolve
            if chunk.fade and self.cut_fade_frames > 0 and self._last_frame is not None:
                self._fade_from, self._fade_done = self._last_frame, 0

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

        if cut and self.first_frame_pad is FirstFramePad.HOLD and 0 < len(out) < self.frames_per_chunk:
            out = np.concatenate([out, np.repeat(out[-1:], self.frames_per_chunk - len(out), axis=0)])
        if self._fade_from is not None and len(out):
            out = self._fade(out)
        if len(out):
            self._last_frame = out[-1]
        self.state = DecoderState.STREAMING
        self.next_chunk_index += 1
        return out


    def _fade(self, out: np.ndarray) -> np.ndarray:
        """Dissolve from _fade_from: output frame k after the cut is (1 - a) * old + a * new, with
        a = (k + 1) / (N + 1), counted over emitted frames, so it runs on across chunk boundaries."""
        n = self.cut_fade_frames
        k = np.arange(len(out), dtype=np.float32) + self._fade_done
        a = np.minimum((k + 1) / (n + 1), 1.0).reshape(-1, 1, 1, 1)
        out = np.rint(self._fade_from.astype(np.float32) * (1 - a) + out.astype(np.float32) * a).astype(np.uint8)
        self._fade_done += len(out)
        if self._fade_done >= n:
            self._fade_from = None
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
        text_encoder_device: Optional[str] = None,
        compile: bool = False,
        seed: int = 0,
        first_frame_pad: str = "hold",
        **auto_rebase,
    ):
        if width % 16 or height % 16:
            raise ValueError("width and height must be multiples of 16 (VAE x8, then patch x2)")
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}; choose from {sorted(VARIANTS)}")
        first_frame_pad = FirstFramePad(first_frame_pad)  # fails before any weights load
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
        unknown = set(auto_rebase) - set(inspect.signature(self._init_runtime).parameters)
        if unknown:
            raise TypeError(f"unknown backend options {sorted(unknown)}")

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
            self._warp = warp

            # Text encoder (umT5-xxl), its own lock and stream so it can run beside generation.
            te = WanTextEncoder()
            self.text_encoder = te.to(device=self.text_device, dtype=self.dtype).eval().requires_grad_(False)
            self._text_lock = threading.Lock()
            self._text_stream = torch.cuda.Stream(device=self.text_device)

            self.decoder = _Decoder(repo_dir, VaeType(vae), self.decode_device, self.latent_hw,
                                    first_frame_pad=FirstFramePad(first_frame_pad), frames_per_chunk=self.frames_per_chunk)

        if compile:
            self.generator.model.compile(mode="max-autotune-no-cudagraphs")

        self._init_runtime(seed, **auto_rebase)
        self.decoder.cut_fade_frames = self.cut_fade_frames

    def _init_runtime(
        self,
        seed: int,
        *,
        steer_rebase: bool = False,
        steer_keep: int = 3,
        steer_memory_noise: int = 0,
        steer_anchor_step: int = -1,
        steer_min_gap: int = 8,
        steer_fire_on_change: bool = True,
        refresh_every: int = 0,
        refresh_keep: int = 6,
        refresh_memory_noise: int = 0,
        clean_anchor_after: int = 3,
        cut_fade_frames: int = 0,
        cut_fade_refresh: bool = True,
        refresh_noise: str = "fixed",
        refresh_anchor_t: Optional[int] = None,
    ) -> None:
        """Generation state: lock, stream, RNG, caches, counters, automatic rebase settings.
        Split out of __init__ so tests can build a Backend around a stub generator."""
        if (steer_rebase or refresh_every > 0) and self.num_frame_per_block != 1:
            raise ValueError("steer_rebase / refresh_every need a framewise variant (1 latent frame per chunk)")
        self._auto_steer = (
            Rebase(steer_keep, steer_memory_noise, None if steer_anchor_step < 0 else steer_anchor_step) if steer_rebase else None
        )
        self._auto_refresh = Rebase(refresh_keep, refresh_memory_noise) if refresh_every > 0 else None
        self._steer_min_gap = int(steer_min_gap)
        self._steer_fire_on_change = bool(steer_fire_on_change)
        self._refresh_every = int(refresh_every)
        self._steer_pending = False  # the cond changed and has not stopped changing yet
        self._last_rebase_chunk = 0  # chunk_index of the last rebase; 0 after reset()
        self._pending_reanchor: Optional[Rebase] = None  # set by reanchor(), used by the next chunk
        # The clean anchor: the latent of the chunk `clean_anchor_after` chunks after the start of
        # the video or after the last rebase that did not start from the clean anchor (a steer).
        # A refresh (reanchor(..., source="clean")) starts from it instead of the drifted last latent.
        if refresh_noise not in ("fresh", "fixed"):
            raise ValueError(f"refresh_noise must be 'fresh' or 'fixed', got {refresh_noise!r}")
        if refresh_anchor_t is not None and not 0 < int(refresh_anchor_t) <= 1000:
            raise ValueError(f"refresh_anchor_t must be in 1..1000 or None, got {refresh_anchor_t}")
        # A refresh from the clean anchor: "fixed" draws its noise from a generator seeded by the clean
        # anchor, so every refresh from one anchor gives the same picture (fresh: a new sample each time).
        self.refresh_noise = refresh_noise
        # The 0-1000 level (unwarped, like the config's steps) the clean anchor is noised to on a
        # refresh, instead of the strength's anchor step; None: the anchor step.
        self.refresh_anchor_t = None if refresh_anchor_t is None else int(refresh_anchor_t)
        if int(cut_fade_frames) < 0:
            raise ValueError(f"cut_fade_frames must be >= 0, got {cut_fade_frames}")
        # Dissolve from the last shown frame over this many frames after a re-anchor cut (0 = off);
        # cut_fade_refresh: refreshes (reanchor source "clean") fade too, else only steers.
        self.cut_fade_frames = int(cut_fade_frames)
        self.cut_fade_refresh = bool(cut_fade_refresh)
        if clean_anchor_after < 0:
            raise ValueError(f"clean_anchor_after must be >= 0, got {clean_anchor_after}")
        self._clean_anchor_after = int(clean_anchor_after)
        self._clean_anchor: Optional[torch.Tensor] = None
        self._clean_anchor_cond: Optional[torch.Tensor] = None  # the cond the clean anchor was made under
        self._clean_capture_chunk: Optional[int] = self._clean_anchor_after  # chunk_index to save, or None
        self.clean_anchor_chunk: Optional[int] = None  # chunk_index the clean anchor was saved from
        # Last clean latents of this video, oldest first (references to the returned latents, no
        # copies). Only kept for framewise variants, which are the only ones that can rebase.
        self._history: collections.deque = collections.deque(maxlen=self.attn_window_frames)
        self.rebases = 0  # rebases done (explicit + automatic), for tests and logs
        self.recached_frames = 0  # latent frames re-encoded by rebases
        self.auto_rebases = {"change": 0, "arrival": 0, "refresh": 0}
        self.last_rebase: Optional[Rebase] = None  # the last rebase as executed (keep after clamping)
        self._gen_lock = threading.Lock()
        self._gen_stream = torch.cuda.Stream(device=self.device)
        self._rng = torch.Generator(device=self.device).manual_seed(seed)
        self._video_seed = int(seed)  # this video's seed: a fixed-noise refresh is seeded by it and the anchor chunk
        self._init_caches()
        self.state = GenState.EMPTY
        self.video_id = 0
        self.chunk_index = 0
        self.frame_index = 0  # latent frames generated in this video (absolute RoPE position)
        self._last_cond: Optional[torch.Tensor] = None
        self.crossattn_resets = 0  # how many times the text K/V cache was recomputed (for tests)

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
            # A reseeded video replays exactly; an unseeded one gets the next seed, so its refreshes differ.
            self._video_seed = int(seed) if seed is not None else self._video_seed + 1
            with torch.cuda.device(self.device), torch.cuda.stream(self._gen_stream):
                self._clear_frame_cache()
            self._reset_crossattn_cache()
            self._last_cond = None
            self.frame_index = 0
            self.chunk_index = 0
            self.video_id += 1
            self.state = GenState.EMPTY
            self._history.clear()
            self._steer_pending = False
            self._pending_reanchor = None
            self._last_rebase_chunk = 0
            self._clean_anchor = None
            self._clean_anchor_cond = None
            self._clean_capture_chunk = self._clean_anchor_after
            self.clean_anchor_chunk = None

    def _clear_frame_cache(self) -> None:
        # Same as the official pipeline between videos (pipeline/causal_inference.py, "reset kv
        # cache"): only the end indices; stale K/V beyond local_end_index is never attended.
        for c in self.kv_cache:
            c["global_end_index"].zero_()
            c["local_end_index"].zero_()

    # ---- rebase ---------------------------------------------------------------------------------

    @property
    def first_frame_steps(self) -> int:
        """Length of the first-frame denoising schedule (the valid anchor steps are 0..this-1)."""
        return len(self.first_chunk_steps)

    def anchor_step_for_strength(self, strength: float) -> Optional[int]:
        """Map a 0..1 cut strength to an anchor step: 1.0 -> 0 (pure noise, full cut), lower ->
        a later step (more of the old picture kept), <= 0 -> None (no anchor)."""
        if strength <= 0:
            return None
        n = self.first_frame_steps
        return min(n - 1, int((1.0 - min(float(strength), 1.0)) * n))

    def reanchor(self, strength: float, source: str = "last") -> None:
        """Ask for a re-anchor on the next generate_chunk: a hard cut of the frame cache (keep 0)
        whose first frame starts from a picture noised to anchor_step_for_strength(strength).
        Strength 1.0 repaints from pure noise; lower keeps more of the old picture; <= 0 cancels.

        source "last" (a steer): start from the last latent; the clean anchor is dropped and saved
        again `clean_anchor_after` chunks later, so it shows the new scene. source "clean" (a
        refresh): start from the clean anchor, which is kept; with no clean anchor yet, as "last"
        but without dropping or re-saving anything."""
        if source not in ("last", "clean"):
            raise ValueError(f"reanchor source must be 'last' or 'clean', got {source!r}")
        step = self.anchor_step_for_strength(strength)
        self._pending_reanchor = None if step is None else Rebase(keep=0, anchor_step=step, from_clean=source == "clean")

    def _plan_rebase(self, changed: bool, explicit: Optional[Rebase]) -> tuple[Optional[Rebase], Optional[str]]:
        """Which rebase (if any) to do before this chunk; updates the automatic-mode state."""
        if self.state is GenState.EMPTY:
            # First chunk of a video: the frame cache is already empty.
            self._steer_pending = False
            return None, None
        gap = self.chunk_index - self._last_rebase_chunk
        auto, reason = None, None
        if self._auto_steer is not None:
            if changed:
                first_change = not self._steer_pending
                self._steer_pending = True
                if first_change and self._steer_fire_on_change and gap >= self._steer_min_gap:
                    auto, reason = self._auto_steer, "change"
            elif self._steer_pending:
                # The cond stopped changing (a ramp ended, or a hard switch one chunk ago).
                self._steer_pending = False
                if gap >= self._steer_min_gap:
                    auto, reason = self._auto_steer, "arrival"
        if auto is None and self._auto_refresh is not None and gap >= self._refresh_every:
            auto, reason = self._auto_refresh, "refresh"
        if explicit is not None:
            return explicit, None
        return auto, reason

    @staticmethod
    def _same_cond(a: torch.Tensor, b: Optional[torch.Tensor]) -> bool:
        """Is `a` the conditioning `b`? The steerer returns the same object once a move has ended."""
        if b is None:
            return False
        return a is b or (a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b)))

    def _timestep(self, t_1000: int) -> torch.Tensor:
        """A 0-1000 timestep in the backend's convention (warped like the denoising steps)."""
        if not 0 <= t_1000 <= 1000:
            raise ValueError(f"timestep must be in 0..1000, got {t_1000}")
        return _timestep_list(self.scheduler, [int(t_1000)], self._warp)[0]

    def _rebase(self, plan: Rebase, cond_dict: dict) -> int:
        """Clear the frame cache and re-encode the last `keep` clean frames at positions 0..keep-1
        under the current cond. Returns keep after clamping. The text cache was already reset."""
        hist = self._history
        keep = max(0, min(int(plan.keep), len(hist), self.attn_window_frames - 1))
        self._clear_frame_cache()
        self.frame_index = 0
        fs = self.frame_seq_length
        if plan.memory_noise > 0:
            # Warped, like the denoising steps: the model was distilled on the shifted timestep
            # grid, so memory_noise=500 is the same noise level as the "500" step of its 2-step
            # schedule, an input level the model was trained to see. Unwarped would mean a much
            # lower sigma than the same number in the config (500 -> sigma 0.5 instead of ~0.83).
            t_mem = self._timestep(plan.memory_noise)
        kept = list(hist)[len(hist) - keep:]  # oldest first
        for i, lat in enumerate(kept):
            if plan.memory_noise > 0:
                flat = lat.flatten(0, 1)
                x = self.scheduler.add_noise(
                    flat,
                    torch.randn(flat.shape, device=self.device, dtype=lat.dtype, generator=self._rng),
                    t_mem * torch.ones([flat.shape[0]], device=self.device, dtype=torch.long),
                ).unflatten(0, lat.shape[:2])
                timestep = torch.ones([1, 1], device=self.device, dtype=torch.int64) * t_mem
            else:
                x = lat
                timestep = torch.ones([1, 1], device=self.device, dtype=torch.int64) * self.context_noise
            # One frame per pass at position i, the same call as the context pass after a frame.
            self.generator(
                noisy_image_or_video=x,
                conditional_dict=cond_dict,
                timestep=timestep,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=i * fs,
            )
        self.frame_index = keep
        self.rebases += 1
        self.recached_frames += keep
        self._last_rebase_chunk = self.chunk_index
        return keep

    # ---- generation -----------------------------------------------------------------------------

    @torch.no_grad()
    def generate_chunk(self, cond: torch.Tensor, rebase=None) -> ChunkLatents:
        """One chunk. `rebase` (a Rebase, a dict of its fields, or None) restarts the frame cache
        before this chunk; with steer_rebase / refresh_every set, the backend may also do so by
        itself. An explicit `rebase` always wins for that chunk. Ignored on the first chunk of a
        video. Rebase needs a framewise variant."""
        with self._gen_lock, torch.cuda.device(self.device), torch.cuda.stream(self._gen_stream):
            if self.frame_index + self.num_frame_per_block > self.max_latent_frames:
                raise RuntimeError("reached max_latent_frames; call reset() or load with a larger max_latent_frames")
            # `cond` may have been written on another thread's stream (the steerer's blend).
            self._gen_stream.wait_stream(torch.cuda.default_stream(self.device))
            pending = self._pending_reanchor
            if pending is not None:
                self._pending_reanchor = None
                if rebase is None:
                    rebase = pending
            changed = cond is not self._last_cond
            plan = reason = None
            if rebase is not None or self._auto_steer is not None or self._auto_refresh is not None:
                plan, reason = self._plan_rebase(changed, Rebase.of(rebase))
                if plan is not None and self.num_frame_per_block != 1:
                    raise ValueError("rebase needs a framewise variant (1 latent frame per chunk)")
            if changed or plan is not None:
                self._reset_crossattn_cache()
                self._last_cond = cond
            cond_dict = {"prompt_embeds": cond}

            f = self.num_frame_per_block
            lh, lw = self.latent_hw
            noisy = torch.randn([1, f, LATENT_CHANNELS, lh, lw], device=self.device, dtype=self.dtype, generator=self._rng)
            schedule = self.first_chunk_steps if self.state is GenState.EMPTY else self.denoising_steps
            start = ChunkStart.FIRST_FRAME if self.state is GenState.EMPTY else ChunkStart.CONTINUE
            fade = False
            rng = self._rng  # a fixed-noise refresh uses its own generator for this chunk
            if plan is not None:
                x_last = self._history[-1] if self._history else None
                from_clean = plan.from_clean and self._clean_anchor is not None
                if from_clean and not self._same_cond(cond, self._clean_anchor_cond):
                    # The scene changed without a steer re-anchor (a provisional or a steer too weak
                    # to repaint): the clean anchor shows the old scene, so this refresh is a steer.
                    from_clean = False
                rng = self._rng
                if from_clean:
                    x_last = self._clean_anchor  # a refresh: the clean anchor stays as it is
                    if self.refresh_noise == "fixed":
                        # Seeded by the anchor, not drawn from the main stream: the same picture every refresh.
                        # Reduced to 64 bits, the range manual_seed accepts, so any reset seed works.
                        rng = torch.Generator(device=self.device).manual_seed(
                            (1_000_003 * self._video_seed + int(self.clean_anchor_chunk)) % (1 << 64))
                        noisy = torch.randn(noisy.shape, device=self.device, dtype=self.dtype, generator=rng)
                elif not plan.from_clean or self._clean_anchor is not None:
                    # A steer (or any other rebase): the old clean anchor shows the old scene.
                    self._clean_anchor = None
                    self._clean_anchor_cond = None
                    self.clean_anchor_chunk = None
                    self._clean_capture_chunk = self.chunk_index + self._clean_anchor_after
                keep = self._rebase(plan, cond_dict)
                if keep == 0:
                    schedule = self.first_chunk_steps  # a hard cut: this is a first frame again
                    start = ChunkStart.FIRST_FRAME  # and the decoder restarts for it
                    fade = self.cut_fade_frames > 0 and (self.cut_fade_refresh or not plan.from_clean)
                n = plan.anchor_step
                if n is not None:
                    if reason is not None:
                        n = min(max(0, n), len(schedule) - 1)  # automatic mode: clamp, never fail mid-stream
                    elif not 0 <= n < len(schedule):
                        raise ValueError(f"anchor_step {n} outside the schedule of {len(schedule)} steps")
                if from_clean and keep == 0 and self.refresh_anchor_t is not None:
                    # A refresh at a chosen level: noise the clean anchor to refresh_anchor_t and
                    # denoise with that level and the first-frame steps below it.
                    t0 = self._timestep(self.refresh_anchor_t)
                    noisy = self.scheduler.add_noise(
                        x_last.flatten(0, 1), noisy.flatten(0, 1), t0 * torch.ones([f], device=self.device, dtype=torch.long),
                    ).unflatten(0, noisy.shape[:2])
                    schedule = torch.stack([t0] + [s for s in self.first_chunk_steps if s < t0])
                elif n is not None and n > 0 and x_last is not None:
                    # Start from the last clean frame noised to step n (reuses the noise drawn above).
                    noisy = self.scheduler.add_noise(
                        x_last.flatten(0, 1),
                        noisy.flatten(0, 1),
                        schedule[n] * torch.ones([f], device=self.device, dtype=torch.long),
                    ).unflatten(0, noisy.shape[:2])
                    schedule = schedule[n:]
                else:
                    n = None if n is None else 0
                    from_clean = False
                self.last_rebase = Rebase(keep, int(plan.memory_noise), n, from_clean)
                if reason is not None:
                    self.auto_rebases[reason] += 1
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
                        torch.randn(denoised.flatten(0, 1).shape, device=self.device, dtype=denoised.dtype, generator=rng),
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
            if f == 1:
                self._history.append(denoised)  # a reference, for a later rebase
            if self.chunk_index == self._clean_capture_chunk:
                self._clean_anchor = denoised  # a reference; returned latents are never modified
                self._clean_anchor_cond = cond
                self.clean_anchor_chunk = self.chunk_index
                self._clean_capture_chunk = None

            ready = torch.cuda.Event()
            ready.record(self._gen_stream)
            out = ChunkLatents(latents=denoised, video_id=self.video_id, chunk_index=self.chunk_index, ready=ready, start=start,
                               fade=fade)
            self.frame_index += f
            self.chunk_index += 1
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
    max_latent_frames (65536), text_encoder_device, compile (False), seed (0).
    Automatic rebase (all off by default, see generate_chunk and MODEL_NOTES.md): steer_rebase (0),
    steer_keep (3), steer_memory_noise (0), steer_anchor_step (-1 = none), steer_min_gap (8),
    steer_fire_on_change (1), refresh_every (0 = off), refresh_keep (6), refresh_memory_noise (0).
    clean_anchor_after (3): chunks after a reset or a steer re-anchor at which the clean anchor is
    saved; reanchor(strength, source="clean") starts from it.
    first_frame_pad ("hold" or "none"): a hard cut (keep-0 rebase, every re-anchor) is decoded as a
    new video's first chunk, 1 frame; "hold" repeats it to the usual chunk length, "none" emits 1.
    cut_fade_frames (0 = off): after a re-anchor cut, dissolve from the last shown frame into the new
    picture over this many output frames, across chunk boundaries; never on a new video.
    cut_fade_refresh (1): refreshes (reanchor source "clean") fade too; 0 fades only steers.
    refresh_noise ("fresh" or "fixed"): "fixed" seeds a refresh's noise by its clean anchor, so every
    refresh from one anchor gives the same picture. refresh_anchor_t (None, or 1..1000): the level a
    refresh noises the clean anchor to, instead of the strength's anchor step (500 = step 2, 250 = step 3).
    """
    # Options may arrive as strings from the server's --backend-opt KEY=VALUE.
    flag = lambda v: v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")  # noqa: E731
    types = {"attn_window_frames": int, "sink_frames": int, "max_latent_frames": int, "seed": int, "compile": flag,
             "steer_rebase": flag, "steer_keep": int, "steer_memory_noise": int, "steer_anchor_step": int,
             "steer_min_gap": int, "steer_fire_on_change": flag, "refresh_every": int, "refresh_keep": int,
             "refresh_memory_noise": int, "clean_anchor_after": int, "cut_fade_frames": int,
             "cut_fade_refresh": flag, "refresh_noise": str,
             "refresh_anchor_t": lambda v: None if str(v).lower() == "none" else int(v)}
    options = {k: types[k](v) if k in types else v for k, v in options.items()}
    steps = int(steps) if steps is not None and int(steps) > 0 else None  # 0 or None: distilled schedule
    t0 = time.time()
    b = Backend(int(width), int(height), steps, device, decode_device, **options)
    b.load_seconds = time.time() - t0
    return b

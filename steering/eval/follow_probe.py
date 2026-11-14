"""Follow probe: does the picture follow a hard prompt switch, with and without a frame-cache rebase.

    python -m steering.eval.follow_probe --out DIR [--width 672 --height 384 --vae taehv --seed 0]
        [--switch-chunk 20 --chunks 80] [--variants all|name,name] [--pair meadow_winter|meadow_city|all]
    python -m steering.eval.follow_probe --out DIR --long 90 [--variants all|name,name]

Loads the backend once through the engine's loader (`engine.backends.load_backend`, default
`causal_forcing`). Switch mode, per pair and variant: `reset(seed)`, base prompt until
`--switch-chunk`, then the target prompt, passing the variant's rebase on the switch chunk only.
Long mode (`--long SECONDS`): one prompt, the variant's rebase every N chunks (a refresh that
restarts the frame positions), to see whether the picture decays over a long run.

Writes `DIR/<pair>/<variant>/frames/NNNNN.jpg` (every video frame), `chunk_ms.txt`,
`sheet.png` (red border at the switch), `clip.txt` when CLIP is available, and `DIR/summary.md`.
Backends without `generate_chunk(cond, rebase=...)` (the engine's fake) run only the variants
without a rebase; the others are listed as "unsupported".
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from .run_engine import _sync, encoder_of, seed_and_reset, split_encoded, to_uint8_frames
from .sheet import contact_sheet

# Same camera and composition wording in each pair, so only the scene content differs.
_CAMERA = (
    "Cinematic wide shot, slow steady dolly forward at eye level, 35mm lens, a wooden fence "
    "post in the left foreground, a dirt path leading to distant hills, centred horizon."
)
MEADOW = (
    "A wide green meadow in high summer, tall grass and yellow wildflowers swaying in a warm "
    "breeze, golden afternoon sunlight, clear blue sky with a few white clouds. " + _CAMERA
)
WINTER = (
    "The same wide meadow in deep winter, thick fresh snow covering the ground and the fence "
    "post, snowflakes drifting down, cold pale blue light, heavy grey overcast sky. " + _CAMERA
)
CITY = (
    "A narrow city street at night in heavy rain, wet asphalt reflecting red and blue neon "
    "signs, glowing shop windows, raindrops splashing in puddles, dark buildings. " + _CAMERA
)
PAIRS = {"meadow_winter": (MEADOW, WINTER), "meadow_city": (MEADOW, CITY)}

# Rebase given on the switch chunk (None = plain hard switch of the text conditioning).
VARIANTS: dict[str, Optional[dict]] = {
    "hardcut": None,
    "keep6": {"keep": 6},
    "keep3": {"keep": 3},
    "keep1": {"keep": 1},
    "keep3_n500": {"keep": 3, "memory_noise": 500},
    "keep6_n500": {"keep": 6, "memory_noise": 500},
    "keep1_n700": {"keep": 1, "memory_noise": 700},
    "keep0": {"keep": 0},
    "keep0_a1": {"keep": 0, "anchor_step": 1},
    "keep0_a2": {"keep": 0, "anchor_step": 2},
    "keep0_a3": {"keep": 0, "anchor_step": 3},
}

# Long mode: (rebase every N chunks, rebase) or None.
LONG_VARIANTS: dict[str, Optional[tuple[int, dict]]] = {
    "none": None,
    "refresh40_keep6": (40, {"keep": 6}),
    "refresh60_keep6": (60, {"keep": 6}),
    "refresh40_keep3": (40, {"keep": 3}),
    "refresh40_keep6_n300": (40, {"keep": 6, "memory_noise": 300}),
}

LAST_SECONDS = 5.0


# ---------------------------------------------------------------- backend glue


def supports_rebase(backend: Any) -> bool:
    try:
        return "rebase" in inspect.signature(backend.generate_chunk).parameters
    except (TypeError, ValueError):
        return False


def skip_reason(backend: Any, rebase: Optional[dict]) -> Optional[str]:
    """Why this variant cannot run on `backend`, or None."""
    if rebase is None:
        return None
    if not supports_rebase(backend):
        return "unsupported"
    n = rebase.get("anchor_step")
    steps = getattr(backend, "first_frame_steps", None)
    if n is not None and steps is not None and n >= steps:
        return f"skipped (first-frame schedule has {steps} steps)"
    return None


def parse_opts(items: Sequence[str]) -> dict[str, str]:
    out = {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep or not k:
            raise ValueError(f"--backend-opt expects KEY=VALUE, got {item!r}")
        out[k] = v
    return out


def load(args: argparse.Namespace) -> Any:
    from engine.backends import load_backend

    opts = parse_opts(args.backend_opt)
    if args.backend == "causal_forcing":
        opts.setdefault("vae", args.vae)
    return load_backend(args.backend, width=args.width, height=args.height, steps=args.steps,
                        device=args.device, decode_device=args.decode_device, **opts)


# ---------------------------------------------------------------- one run


@dataclass
class Run:
    """What one variant produced: per-chunk timing and one sample frame per chunk."""

    chunk_ms: list[float] = field(default_factory=list)
    chunk_first_frame: list[int] = field(default_factory=list)  # video frame index where each chunk starts
    samples: list[np.ndarray] = field(default_factory=list)  # last frame of each chunk, uint8 [H, W, 3]
    frames: int = 0
    sat: list[float] = field(default_factory=list)  # per video frame
    contrast: list[float] = field(default_factory=list)


def frame_stats(frame: np.ndarray) -> tuple[float, float]:
    """(mean HSV saturation in 0..1, RMS contrast = std of luma in 0..1) of uint8 [H, W, 3]."""
    f = frame.astype(np.float32) / 255.0
    mx, mn = f.max(axis=2), f.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    luma = f @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return float(sat.mean()), float(luma.std())


def generate(backend: Any, conds: Sequence[Any], rebases: dict[int, dict], frames_dir: str,
             stats: bool = False, jpeg_quality: int = 90) -> Run:
    """Generate one chunk per entry of `conds` (rebase given on the chunks in `rebases`),
    decode, and save every video frame as JPEG."""
    os.makedirs(frames_dir, exist_ok=True)
    run = Run()
    for i, cond in enumerate(conds):
        t0 = time.perf_counter()
        if i in rebases:
            lat = backend.generate_chunk(cond, rebase=rebases[i])
        else:
            lat = backend.generate_chunk(cond)
        _sync()
        run.chunk_ms.append((time.perf_counter() - t0) * 1e3)
        frames = to_uint8_frames(backend.decode_chunk(lat)).numpy()
        run.chunk_first_frame.append(run.frames)
        for fr in frames:
            Image.fromarray(fr, "RGB").save(os.path.join(frames_dir, f"{run.frames:05d}.jpg"), quality=jpeg_quality)
            if stats:
                s, c = frame_stats(fr)
                run.sat.append(s)
                run.contrast.append(c)
            run.frames += 1
        run.samples.append(frames[-1].copy())
    return run


def samples_tensor(run: Run) -> torch.Tensor:
    """Float [N, 3, H, W] in 0..1, one per chunk."""
    return torch.from_numpy(np.stack(run.samples)).permute(0, 3, 1, 2).float().div_(255.0)


# ---------------------------------------------------------------- CLIP


class Clip:
    """Lazy CLIP scorer; `None` scores when transformers or the model cannot be loaded."""

    def __init__(self, enabled: bool, device: str) -> None:
        self.enabled = enabled and importlib.util.find_spec("transformers") is not None
        self.device = device
        self.scorer = None
        self.error: Optional[str] = None if self.enabled else "CLIP disabled or transformers not importable"

    def curve(self, images: torch.Tensor, prompts: Sequence[str]) -> Optional[torch.Tensor]:
        """Cosine similarity [N, P], or None."""
        if not self.enabled:
            return None
        try:
            from .clip_score import ClipScorer, similarity_curve

            if self.scorer is None:
                self.scorer = ClipScorer(device=self.device)
            return similarity_curve(self.scorer.embed_images(images), self.scorer.embed_texts(list(prompts)))
        except Exception as exc:  # no download, no memory: report, do not fail the run
            self.enabled = False
            self.error = f"CLIP failed: {type(exc).__name__}: {exc}"
            return None


def follow_metrics(curve: torch.Tensor, switch: int, chunks_per_s: float) -> tuple[float, Optional[int]]:
    """(mean of CLIP(target) - CLIP(base) over the last 5 s, chunks after the switch until
    target first beats base or None) from a per-chunk [N, 2] (base, target) curve."""
    diff = (curve[:, 1] - curve[:, 0]).float()
    last = max(1, int(round(LAST_SECONDS * chunks_per_s)))
    tail = float(diff[-last:].mean())
    after = (diff[switch:] > 0).nonzero()
    return tail, (int(after[0]) if len(after) else None)


# ---------------------------------------------------------------- the two modes


def write_lines(path: str, lines: Sequence[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_switch(backend: Any, out: str, pairs: Sequence[str], variants: Sequence[str], *, seed: int,
               switch: int, chunks: int, clip: Clip) -> list[str]:
    """Switch mode; returns the summary.md lines."""
    fps = float(getattr(backend, "fps", 16))
    fpc = float(getattr(backend, "frames_per_chunk", 4))
    chunks_per_s = fps / fpc
    encode = encoder_of(backend)
    lines = [
        f"# Follow probe: hard switch at chunk {switch} of {chunks}",
        "",
        f"seed {seed}; {chunks_per_s:g} chunks per second; CLIP: {clip.error or 'on'}. "
        "`tail` = CLIP(target) - CLIP(base) over the last 5 s (one frame per chunk); "
        "`cross` = chunks after the switch until CLIP(target) > CLIP(base).",
    ]
    for pair in pairs:
        base_prompt, target_prompt = PAIRS[pair]
        base, _ = split_encoded(encode(base_prompt))
        target, _ = split_encoded(encode(target_prompt))
        conds = [base] * switch + [target] * (chunks - switch)
        lines += ["", f"## {pair}", "", "| variant | switch ms | max ms after | tail | cross |", "|---|---|---|---|---|"]
        for name in variants:
            rebase = VARIANTS[name]
            why = skip_reason(backend, rebase)
            if why:
                lines.append(f"| {name} | {why} | | | |")
                continue
            folder = os.path.join(out, pair, name)
            seed_and_reset(backend, seed)
            run = generate(backend, conds, {switch: rebase} if rebase is not None else {}, os.path.join(folder, "frames"))
            write_lines(os.path.join(folder, "chunk_ms.txt"), [f"{i} {ms:.1f}" for i, ms in enumerate(run.chunk_ms)])
            images = samples_tensor(run)
            labels = [f"c{i}" for i in range(len(run.samples))]
            contact_sheet(images, every=max(1, chunks // 40), labels=labels, mark=[switch]).save(os.path.join(folder, "sheet.png"))
            tail = cross = "-"
            curve = clip.curve(images, [base_prompt, target_prompt])
            if curve is not None:
                t, c = follow_metrics(curve, switch, chunks_per_s)
                tail, cross = f"{t:+.4f}", ("never" if c is None else str(c))
                write_lines(os.path.join(folder, "clip.txt"), ["chunk sim_base sim_target"] + [
                    f"{i} {float(curve[i, 0]):.4f} {float(curve[i, 1]):.4f}" for i in range(curve.shape[0])])
            after = run.chunk_ms[switch + 1:] or [float("nan")]
            lines.append(f"| {name} | {run.chunk_ms[switch]:.0f} | {max(after):.0f} | {tail} | {cross} |")
    return lines


def run_long(backend: Any, out: str, seconds: float, variants: Sequence[str], *, seed: int) -> list[str]:
    """Long mode; returns the summary.md lines."""
    fps = float(getattr(backend, "fps", 16))
    fpc = float(getattr(backend, "frames_per_chunk", 4))
    chunks = max(1, math.ceil(seconds * fps / fpc))
    every5 = max(1, int(round(5.0 * fps / fpc)))  # chunks per 5 s
    prompt = MEADOW
    cond, _ = split_encoded(encoder_of(backend)(prompt))
    rows: dict[str, tuple[list[float], list[float]]] = {}
    lines = [f"# Follow probe: long run, {seconds:g} s ({chunks} chunks), one prompt", "",
             f"seed {seed}. Per 5 s of video: mean HSV saturation and RMS contrast (luma std), both 0..1. "
             "Decay shows as both climbing.", "", "| variant | status | max chunk ms |", "|---|---|---|"]
    bin_frames = int(round(5.0 * fps))
    for name in variants:
        spec = LONG_VARIANTS[name]
        why = skip_reason(backend, spec[1] if spec else None)
        if why:
            lines.append(f"| {name} | {why} | |")
            continue
        folder = os.path.join(out, "long", name)
        seed_and_reset(backend, seed)
        rebases = {} if spec is None else {i: spec[1] for i in range(spec[0], chunks, spec[0])}
        run = generate(backend, [cond] * chunks, rebases, os.path.join(folder, "frames"), stats=True)
        write_lines(os.path.join(folder, "chunk_ms.txt"), [f"{i} {ms:.1f}" for i, ms in enumerate(run.chunk_ms)])
        images = samples_tensor(run)
        labels = [f"{run.chunk_first_frame[i] / fps:.0f}s" for i in range(len(run.samples))]
        contact_sheet(images, every=every5, labels=labels, mark=sorted(rebases)).save(os.path.join(folder, "sheet.png"))
        bins = range(0, run.frames, bin_frames)
        rows[name] = ([float(np.mean(run.sat[b:b + bin_frames])) for b in bins],
                      [float(np.mean(run.contrast[b:b + bin_frames])) for b in bins])
        lines.append(f"| {name} | ok | {max(run.chunk_ms):.0f} |")
    for title, k in (("saturation", 0), ("RMS contrast", 1)):
        if not rows:
            break
        n = max(len(v[k]) for v in rows.values())
        lines += ["", f"## {title} per 5 s", "", "| variant | " + " | ".join(f"{5 * i}s" for i in range(n)) + " |",
                  "|---|" + "---|" * n]
        for name, v in rows.items():
            lines.append(f"| {name} | " + " | ".join(f"{x:.3f}" for x in v[k]) + " |")
    return lines


# ---------------------------------------------------------------- CLI


def pick(spec: str, known: Sequence[str], what: str) -> list[str]:
    if spec.strip() == "all":
        return list(known)
    names = [n.strip() for n in spec.split(",") if n.strip()]
    bad = [n for n in names if n not in known]
    if bad or not names:
        raise ValueError(f"unknown {what} {bad}; choose from {list(known)}")
    return names


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--backend", default="causal_forcing", help="engine backend name (fake for a smoke test)")
    p.add_argument("--backend-opt", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--width", type=int, default=672)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--steps", type=int, default=0, help="0 = the distilled schedule")
    p.add_argument("--vae", default="taehv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--decode-device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--switch-chunk", type=int, default=20)
    p.add_argument("--chunks", type=int, default=80)
    p.add_argument("--variants", default="all")
    p.add_argument("--pair", default="all", help="meadow_winter, meadow_city or all")
    p.add_argument("--long", type=float, default=None, metavar="SECONDS", help="long single-prompt run instead")
    p.add_argument("--no-clip", action="store_true")
    p.add_argument("--clip-device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    try:
        if args.long is not None:
            variants = pick(args.variants, list(LONG_VARIANTS), "long variants")
        else:
            variants = pick(args.variants, list(VARIANTS), "variants")
            pairs = pick(args.pair, list(PAIRS), "pairs")
    except ValueError as e:
        p.error(str(e))
    if args.long is None and not 0 < args.switch_chunk < args.chunks:
        p.error("need 0 < --switch-chunk < --chunks")

    os.makedirs(args.out, exist_ok=True)
    backend = load(args)
    header = f"backend {args.backend} {args.width}x{args.height} steps={args.steps} opts={args.backend_opt}"
    if args.backend == "causal_forcing":
        header += f" vae={args.vae}"
    if args.long is not None:
        lines = run_long(backend, args.out, args.long, variants, seed=args.seed)
    else:
        lines = run_switch(backend, args.out, pairs, variants, seed=args.seed, switch=args.switch_chunk,
                           chunks=args.chunks, clip=Clip(not args.no_clip, args.clip_device))
    text = "\n".join(lines[:1] + ["", header] + lines[1:]) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

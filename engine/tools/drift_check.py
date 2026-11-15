"""Long-run stability and prompt-switch check for the Causal Forcing backend.

Drives the Backend directly (no server): one prompt for --calls calls, optionally a switch to
--prompt-b at --switch-at (hard, or blended over --ramp calls). Writes to --out:
- sheet.jpg: one decoded frame every --sheet-every calls, labelled with the video time;
- switch.jpg (with --prompt-b): one frame per second from 2 s before the switch to --switch-span s after;
- stats.csv: per call, generation ms, latent mean/std and decoded-frame mean, std, saturation,
  clipped-pixel fraction, so collapse shows as a curve;
- summary.json: the options and the stats averaged over 5 s windows.

Run on the GPU machine from the repo root, for example:
  python -m engine.tools.drift_check --calls 960 --out ~/runs/drift_w21 --opt attn_window_frames=21
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from engine.backends import causal_forcing

CALLS_PER_SECOND = 4  # one call = one latent frame = 4 video frames at 16 fps


def frame_stats(frames: np.ndarray) -> dict:
    """frames: uint8 [T, H, W, 3]. Mean over the chunk."""
    f = frames.astype(np.float32)
    mx, mn = f.max(axis=-1), f.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    gray = f.mean(axis=-1)
    return {
        "px_mean": float(gray.mean()),
        "px_std": float(gray.std()),
        "sat": float(sat.mean()),
        "clip": float(((f <= 4) | (f >= 251)).mean()),
    }


def tile(frame: np.ndarray, label: str, w: int, h: int) -> Image.Image:
    im = Image.fromarray(frame).resize((w, h), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(label) + 6, 14], fill=(0, 0, 0))
    d.text((3, 1), label, fill=(255, 255, 0))
    return im


def sheet(tiles: list[Image.Image], cols: int) -> Image.Image:
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * w, rows * h), (40, 40, 40))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * w, (i // cols) * h))
    return out


def parse_opts(items: list[str]) -> dict:
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=672)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--vae", default="taehv")
    p.add_argument("--calls", type=int, default=960)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default="a wide shot of a meadow with a single oak tree")
    p.add_argument("--prompt-b", default=None)
    p.add_argument("--switch-at", type=int, default=120)
    p.add_argument("--ramp", type=int, default=0, help="calls to blend A->B linearly (0: hard switch)")
    p.add_argument("--reanchor", type=float, default=0.0,
                   help="call request_reanchor(strength) at the switch (0: no re-anchor)")
    p.add_argument("--switch-span", type=int, default=16, help="seconds after the switch shown in switch.jpg")
    p.add_argument("--sheet-every", type=int, default=20)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--opt", action="append", default=[], help="backend option KEY=VALUE (repeatable)")
    args = p.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    opts = parse_opts(args.opt)
    b = causal_forcing.load(args.width, args.height, vae=args.vae, **opts)
    cond_a = b.encode_prompt(args.prompt)
    cond_b = b.encode_prompt(args.prompt_b) if args.prompt_b else None

    def cond_for(i: int, current: torch.Tensor) -> torch.Tensor:
        """Same tensor object while the conditioning is unchanged, a new one when it changes."""
        if cond_b is None or i < args.switch_at:
            return cond_a
        k = i - args.switch_at + 1
        if args.ramp <= 0 or k >= args.ramp:
            return cond_b
        a = k / args.ramp
        return (cond_a.float() * (1 - a) + cond_b.float() * a).to(cond_a.dtype)

    b.reset(seed=args.seed)
    rows, tiles, sw_tiles = [], [], []
    tw, th = 280, 160
    cond = cond_a
    for i in range(args.calls):
        cond = cond_for(i, cond)
        if cond_b is not None and args.reanchor > 0 and i == args.switch_at:
            b.request_reanchor(args.reanchor)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lat = b.generate_chunk(cond)
        b.sync_generation()
        gen_ms = (time.perf_counter() - t0) * 1000
        frames = b.decode_chunk(lat)
        z = lat.latents.float()
        row = {"call": i, "t_s": i / CALLS_PER_SECOND, "gen_ms": gen_ms,
               "lat_mean": float(z.mean()), "lat_std": float(z.std())}
        row.update(frame_stats(frames))
        rows.append(row)
        label = f"{i / CALLS_PER_SECOND:.0f}s c{i}"
        if i % args.sheet_every == 0 or i == args.calls - 1:
            tiles.append(tile(frames[-1], label, tw, th))
        if cond_b is not None and (i - args.switch_at) % CALLS_PER_SECOND == 0 and \
                -2 * CALLS_PER_SECOND <= i - args.switch_at <= args.switch_span * CALLS_PER_SECOND:
            sw_tiles.append(tile(frames[-1], f"{(i - args.switch_at) / CALLS_PER_SECOND:+.0f}s", tw, th))
        if i % 40 == 0:
            print(f"call {i}: gen {gen_ms:.0f} ms, px_mean {row['px_mean']:.1f} std {row['px_std']:.1f} "
                  f"sat {row['sat']:.2f} clip {row['clip']:.3f} lat_std {row['lat_std']:.3f}", flush=True)

    sheet(tiles, args.cols).save(os.path.join(out, "sheet.jpg"), quality=85)
    if sw_tiles:
        sheet(sw_tiles, 6).save(os.path.join(out, "switch.jpg"), quality=85)
    with open(os.path.join(out, "stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    win = 5 * CALLS_PER_SECOND
    windows = []
    for s in range(0, len(rows), win):
        part = rows[s:s + win]
        windows.append({k: round(statistics.fmean(r[k] for r in part), 3) for k in
                        ("t_s", "gen_ms", "lat_mean", "lat_std", "px_mean", "px_std", "sat", "clip")})
    steady = [r["gen_ms"] for r in rows[30:]]
    summary = {"args": vars(args), "gen_ms_median": round(statistics.median(steady), 1) if steady else None,
               "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "windows_5s": windows}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: summary[k] for k in ("gen_ms_median", "peak_alloc_gib")}))
    for wnd in windows[:: max(1, len(windows) // 12)]:
        print(wnd)


if __name__ == "__main__":
    main()

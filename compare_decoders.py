"""Phase 1: decode the same Causal Forcing++ latents with TAEHV and the stock Wan VAE decoder.

Run from ~/decode/madhacks with ~/engine-venv:
    python compare_decoders.py --out ~/decode/out
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from engine.backends import causal_forcing as cf

MEADOW = ("A wide sunlit meadow in late spring, tall green grass and yellow wildflowers swaying in a gentle breeze, "
          "a large oak tree with dense leaves in the middle distance, rolling hills and a blue sky with soft white clouds, "
          "fine detail, natural light, slow camera pan")
CITY = ("A busy city street at night, neon signs and glowing shop windows, wet asphalt reflecting colourful lights, "
        "cars with headlights and red tail lights passing by, tall buildings with lit windows, light rain, "
        "cinematic, fine detail, slow camera movement")


def decode_all(decoder, chunks):
    frames, ms = [], []
    for c in chunks:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f = decoder.decode(c)  # returns a CPU numpy array, so the decode is finished here
        ms.append((time.perf_counter() - t0) * 1e3)
        frames.append(f)
    return frames, ms


def flicker_stats(frames_per_chunk):
    """Mean abs difference between consecutive frames, split into inside a chunk and across a chunk boundary."""
    inside, boundary = [], []
    prev = None
    for f in frames_per_chunk:
        for i in range(f.shape[0]):
            cur = f[i].astype(np.float32)
            if prev is not None:
                (inside if i > 0 else boundary).append(float(np.abs(cur - prev).mean()))
            prev = cur
    return {"inside_mean": float(np.mean(inside)), "boundary_mean": float(np.mean(boundary)),
            "boundary_over_inside": float(np.mean(boundary) / np.mean(inside))}


def lap_var(img):
    g = img.astype(np.float32).mean(axis=2)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--chunks", type=int, default=80)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    b = cf.load(672, 384, vae="taehv", seed=0)
    print(f"loaded in {b.load_seconds:.0f} s", flush=True)
    with cf._in_dir(cf.DEFAULT_REPO_DIR):
        wan = cf._Decoder(cf.DEFAULT_REPO_DIR, cf.VaeType.WAN, b.decode_device, b.latent_hw)
    taehv = b.decoder

    results = {}
    for name, prompt in (("meadow", MEADOW), ("city", CITY)):
        b.reset(seed=0)
        cond = b.encode_prompt(prompt)
        chunks = []
        for _ in range(a.chunks):
            chunks.append(b.generate_chunk(cond))
        b.sync_generation()
        torch.save([c.latents.cpu() for c in chunks], os.path.join(a.out, f"{name}_latents.pt"))

        t_frames, t_ms = decode_all(taehv, chunks)
        w_frames, w_ms = decode_all(wan, chunks)
        np.savez_compressed(os.path.join(a.out, f"{name}_frames_small.npz"),
                            taehv=np.concatenate(t_frames)[::4, ::2, ::2], wan=np.concatenate(w_frames)[::4, ::2, ::2])
        # Full frames as JPEG q95 for sheets.
        for tag, fr in (("taehv", t_frames), ("wan", w_frames)):
            d = os.path.join(a.out, f"{name}_{tag}")
            os.makedirs(d, exist_ok=True)
            for ci, f in enumerate(fr):
                for i in range(f.shape[0]):
                    Image.fromarray(f[i]).save(os.path.join(d, f"c{ci:03d}_f{i}.png"))
        steady = slice(10, None)
        tall, wall = np.concatenate(t_frames), np.concatenate(w_frames)
        results[name] = {
            "taehv_ms_median": float(np.median(t_ms[steady])), "taehv_ms_p90": float(np.percentile(t_ms[steady], 90)),
            "wan_ms_median": float(np.median(w_ms[steady])), "wan_ms_p90": float(np.percentile(w_ms[steady], 90)),
            "taehv_ms_first": t_ms[:3], "wan_ms_first": w_ms[:3],
            "frames_taehv": int(tall.shape[0]), "frames_wan": int(wall.shape[0]),
            "flicker_taehv": flicker_stats(t_frames), "flicker_wan": flicker_stats(w_frames),
            "lapvar_taehv": float(np.mean([lap_var(f) for f in tall[::8]])),
            "lapvar_wan": float(np.mean([lap_var(f) for f in wall[::8]])),
            "mean_rgb_taehv": tall.reshape(-1, 3).mean(0).tolist(), "mean_rgb_wan": wall.reshape(-1, 3).mean(0).tolist(),
            "abs_diff_taehv_wan": float(np.abs(tall.astype(np.float32) - wall.astype(np.float32)).mean())
            if tall.shape == wall.shape else None,
        }
        print(name, json.dumps(results[name]), flush=True)
    results["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()

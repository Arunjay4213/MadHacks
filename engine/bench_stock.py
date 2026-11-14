"""Gate 1 benchmark: endless chunk-after-chunk generation through the Backend interface.

Run from the repo root on the GPU machine, for example:
  python -m engine.bench_stock --width 832 --height 480 --chunks 60
  python -m engine.bench_stock --width 832 --height 480 --chunks 60 --threaded
  python -m engine.bench_stock --width 576 --height 320 --chunks 300 --threaded --save-mp4 late.mp4 --save-from 280

Two modes:
- sequential (default): per chunk, generation ms and decode ms measured separately, each
  with a device synchronize around it.
- threaded (--threaded): generation in one thread, decode in another, a queue between them,
  the way the engine runs them. Reports the steady-state wall-clock time per chunk, taken as
  the median gap between consecutive decoded chunks after the warm-up chunks.

Optional prompt switch (--prompt-b, --switch-at) for the conditioning test: the embedding
object handed to generate_chunk changes at that chunk; the frame cache is left alone.
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import threading
import time

import numpy as np
import torch

from engine.backends import causal_forcing


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _summary(values: list[float]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    return {
        "median": round(statistics.median(v), 1),
        "mean": round(statistics.fmean(v), 1),
        "p90": round(v[int(0.9 * (len(v) - 1))], 1),
        "max": round(v[-1], 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--steps", type=int, default=None, help="denoising steps per chunk (default: what the checkpoint was distilled for)")
    p.add_argument("--chunks", type=int, default=60)
    p.add_argument("--warmup", type=int, default=5, help="chunks left out of the steady-state numbers")
    p.add_argument("--threaded", action="store_true")
    p.add_argument("--variant", default="framewise_2step", choices=sorted(causal_forcing.VARIANTS))
    p.add_argument("--vae", default="wan", choices=["wan", "taehv"])
    p.add_argument("--window", type=int, default=21, help="attention window in latent frames (frame KV cache size)")
    p.add_argument("--sink", type=int, default=0, help="sink frames kept at the start of the window")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--decode-device", default="cuda:0")
    p.add_argument("--prompt", default="A red fox walking through a snowy pine forest at sunrise, cinematic, detailed fur, soft light.")
    p.add_argument("--prompt-b", default=None)
    p.add_argument("--switch-at", type=int, default=10)
    p.add_argument("--save-mp4", default=None)
    p.add_argument("--save-from", type=int, default=0, help="first chunk written to --save-mp4")
    p.add_argument("--save-to", type=int, default=None, help="last chunk (exclusive) written to --save-mp4")
    p.add_argument("--json", default=None, help="write the summary as JSON here")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    backend = causal_forcing.load(
        args.width, args.height, args.steps, device=args.device, decode_device=args.decode_device,
        variant=args.variant, vae=args.vae, attn_window_frames=args.window, sink_frames=args.sink, compile=args.compile,
    )
    mem_loaded = torch.cuda.memory_allocated(args.device) / 2**30
    torch.cuda.reset_peak_memory_stats(args.device)  # peak numbers below cover generation, not loading
    print(f"loaded in {backend.load_seconds:.1f}s, {mem_loaded:.2f} GiB allocated, steps={backend.steps}, "
          f"latent frames/chunk={backend.num_frame_per_block}, video frames/chunk={backend.frames_per_chunk}", flush=True)

    # Prompt encode timing (first call includes warm-up, so encode twice).
    t0 = time.perf_counter()
    cond_a = backend.encode_prompt(args.prompt)
    encode_first_ms = _ms(t0)
    t0 = time.perf_counter()
    cond_a = backend.encode_prompt(args.prompt)
    encode_ms = _ms(t0)
    cond_b = backend.encode_prompt(args.prompt_b) if args.prompt_b else None
    print(f"cond shape={tuple(cond_a.shape)} dtype={cond_a.dtype} encode_ms first={encode_first_ms:.0f} warm={encode_ms:.0f}", flush=True)

    def cond_for(i: int):
        return cond_b if (cond_b is not None and i >= args.switch_at) else cond_a

    chunk_seconds = backend.frames_per_chunk / backend.fps
    save_to = args.save_to if args.save_to is not None else args.chunks
    saved: list[np.ndarray] = []
    gen_ms: list[float] = []
    dec_ms: list[float] = []
    done_at: list[float] = []
    frames_out = 0

    def keep(i: int, frames: np.ndarray):
        nonlocal frames_out
        frames_out += frames.shape[0]
        if args.save_mp4 and args.save_from <= i < save_to:
            saved.append(frames)

    t_start = time.perf_counter()
    if not args.threaded:
        for i in range(args.chunks):
            torch.cuda.synchronize(args.device)
            t0 = time.perf_counter()
            lat = backend.generate_chunk(cond_for(i))
            backend.sync_generation()
            gen_ms.append(_ms(t0))
            t0 = time.perf_counter()
            frames = backend.decode_chunk(lat)  # returns host memory, so it has waited for the GPU
            dec_ms.append(_ms(t0))
            done_at.append(time.perf_counter())
            keep(i, frames)
            if i % args.log_every == 0:
                print(f"chunk {i}: gen {gen_ms[-1]:.0f} ms, decode {dec_ms[-1]:.0f} ms, frames {frames.shape}, "
                      f"peak {torch.cuda.max_memory_allocated(args.device) / 2**30:.2f} GiB", flush=True)
    else:
        q: queue.Queue = queue.Queue(maxsize=2)
        errors: list[BaseException] = []

        def gen_loop():
            try:
                for i in range(args.chunks):
                    t0 = time.perf_counter()
                    lat = backend.generate_chunk(cond_for(i))
                    backend.sync_generation()
                    gen_ms.append(_ms(t0))
                    q.put((i, lat))
            except BaseException as e:  # surface errors from the thread
                errors.append(e)
            finally:
                q.put(None)

        th = threading.Thread(target=gen_loop, daemon=True)
        th.start()
        while True:
            item = q.get()
            if item is None:
                break
            i, lat = item
            t0 = time.perf_counter()
            frames = backend.decode_chunk(lat)
            dec_ms.append(_ms(t0))
            done_at.append(time.perf_counter())
            keep(i, frames)
            if i % args.log_every == 0:
                print(f"chunk {i}: gen {gen_ms[i]:.0f} ms, decode {dec_ms[-1]:.0f} ms, frames {frames.shape}, "
                      f"peak {torch.cuda.max_memory_allocated(args.device) / 2**30:.2f} GiB", flush=True)
        th.join()
        if errors:
            raise errors[0]
    total_s = time.perf_counter() - t_start

    w = min(args.warmup, max(0, args.chunks - 2))
    gaps = [(b - a) * 1000.0 for a, b in zip(done_at[w:], done_at[w + 1:])]
    if args.threaded:
        wall = _summary(gaps)
    else:
        wall = _summary([g + d for g, d in zip(gen_ms[w:], dec_ms[w:])])
    per_075 = 0.75 / chunk_seconds
    result = {
        "variant": args.variant,
        "width": args.width,
        "height": args.height,
        "steps": backend.steps,
        "latent_frames_per_chunk": backend.num_frame_per_block,
        "video_frames_per_chunk": backend.frames_per_chunk,
        "vae": args.vae,
        "window": args.window,
        "sink": args.sink,
        "compile": args.compile,
        "threaded": args.threaded,
        "chunks": args.chunks,
        "gen_ms": _summary(gen_ms[w:]),
        "decode_ms": _summary(dec_ms[w:]),
        "wall_ms_per_chunk": wall,
        "budget_ms_per_chunk": round(chunk_seconds * 1000.0, 1),
        "wall_ms_per_0.75s_video": round(wall["median"] * per_075, 1) if wall else None,
        "realtime": bool(wall and wall["median"] <= chunk_seconds * 1000.0),
        "gen_first10_last10_ms": [round(statistics.fmean(gen_ms[w:w + 10]), 1), round(statistics.fmean(gen_ms[-10:]), 1)] if len(gen_ms) >= w + 10 else None,
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated(args.device) / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(args.device) / 2**30, 2),
        "alloc_after_load_gib": round(mem_loaded, 2),
        "encode_ms_warm": round(encode_ms, 1),
        "frames_out": frames_out,
        "total_s": round(total_s, 1),
        "crossattn_resets": backend.crossattn_resets,
        "cond_shape": list(cond_a.shape),
        "cond_dtype": str(cond_a.dtype),
    }
    print("RESULT " + json.dumps(result), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    if args.save_mp4 and saved:
        import imageio

        imageio.mimwrite(args.save_mp4, np.concatenate(saved), fps=backend.fps, quality=8)
        print(f"wrote {args.save_mp4}", flush=True)


if __name__ == "__main__":
    main()

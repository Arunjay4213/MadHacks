"""Run the steering scenarios on a real engine backend.

    python -m steering.eval.run_engine --backend engine.backends.<name> --out DIR

The backend module must provide `load(width, height, steps)` returning an
object with `encode_prompt(prompt)` (or `encode`), `reset()`,
`generate_chunk(cond) -> latents` and `decode_chunk(latents) -> uint8
[T, H, W, 3]`. Each scenario is seeded and starts from `reset()`; it writes
`<name>.report.txt`, `<name>.mp4` (when a video writer is importable) and
optionally `<name>.frames.pt`, plus one `summary.txt` for the whole run.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from ..align import Aligner
from . import scenarios as sc


# ---------------------------------------------------------------- backend glue


def load_backend(module_name: str, width: int, height: int, steps: int) -> Any:
    """Import `module_name` and return `module.load(width=, height=, steps=)`."""
    module = importlib.import_module(module_name)
    if not hasattr(module, "load"):
        raise AttributeError(f"backend module {module_name!r} has no load()")
    return module.load(width=width, height=height, steps=steps)


def seed_and_reset(backend: Any, seed: int | None) -> str:
    """Seed (if `seed` is given) and call `backend.reset()`; return how it seeded.

    Prefers `backend.seed(int)`, then `backend.set_seed(int)`, then
    `backend.reset(seed=int)`, else `torch.manual_seed` (and all CUDA
    devices) right before `reset()`.
    """
    if seed is None:
        backend.reset()
        return "none"
    for name in ("seed", "set_seed"):
        fn = getattr(backend, name, None)
        if callable(fn):
            fn(int(seed))
            backend.reset()
            return f"backend.{name}"
    if _accepts_kw(backend.reset, "seed"):
        backend.reset(seed=int(seed))
        return "backend.reset(seed=)"
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    backend.reset()
    return "torch.manual_seed"


def _accepts_kw(fn: Callable, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def encoder_of(backend: Any) -> Callable[[str], Any]:
    """The backend's encoder, preferring the form that also returns token ids."""
    for name in ("encode_prompt_with_ids", "encode_prompt", "encode"):
        fn = getattr(backend, name, None)
        if callable(fn):
            return fn
    raise AttributeError("backend has neither encode_prompt() nor encode()")


def split_encoded(x: Any) -> tuple[Tensor, list[int] | None]:
    """Return (embedding, token ids or None) from a tuple `(emb, ids)` or a tensor."""
    if isinstance(x, Tensor):
        return x, None
    if isinstance(x, (tuple, list)) and len(x) == 2 and isinstance(x[0], Tensor):
        return x[0], [int(i) for i in x[1]]
    raise TypeError(
        f"encode must return a Tensor or (Tensor, token_ids), got {type(x).__name__}"
    )


def make_encode(backend: Any, align: bool) -> Callable[[str], Any]:
    """Encode for `scenarios.run`: (emb, ids) when `align`, else a bare tensor."""
    raw = encoder_of(backend)

    def encode(prompt: str) -> Any:
        emb, ids = split_encoded(raw(prompt))
        if not align:
            return emb
        if ids is None:
            raise TypeError(
                "--align needs the backend's encoder to return (embedding, token_ids); "
                "it returned a bare tensor"
            )
        return emb, ids

    return encode


def to_uint8_frames(x: Any) -> Tensor:
    """Decoded frames (tensor or array, uint8 [T, H, W, 3] or [H, W, 3]) on CPU."""
    t = torch.as_tensor(x).detach().cpu()
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() != 4 or t.shape[-1] != 3 or t.shape[0] < 1:
        raise ValueError(f"decode_chunk must return [T, H, W, 3], got {tuple(t.shape)}")
    if t.dtype != torch.uint8:
        raise ValueError(f"decode_chunk must return uint8, got {t.dtype}")
    return t


def frames_to_float(x: Any) -> Tensor:
    """uint8 [T, H, W, 3] -> float32 [T, 3, H, W] in [0, 1], on CPU."""
    return to_uint8_frames(x).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class EngineAdapter:
    """`generate(cond)` for `scenarios.run`: generate_chunk, decode_chunk, to CPU.

    Records the generate and decode wall time of every call separately
    (CUDA synchronised) in `gen_s` and `dec_s`.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.gen_s: list[float] = []
        self.dec_s: list[float] = []

    def generate(self, cond: Tensor) -> Tensor:
        t0 = time.perf_counter()
        latents = self.backend.generate_chunk(cond)
        _sync()
        t1 = time.perf_counter()
        frames = frames_to_float(self.backend.decode_chunk(latents))
        t2 = time.perf_counter()
        self.gen_s.append(t1 - t0)
        self.dec_s.append(t2 - t1)
        return frames


# ---------------------------------------------------------------- outputs


def write_video(path: str, frames: Tensor, fps: float) -> str | None:
    """Write float [T, 3, H, W] frames as mp4; return the writer used or None.

    Tries imageio (needs imageio-ffmpeg), then torchvision.io.write_video
    (needs PyAV). Returns None, leaving no file, if neither works.
    """
    u8 = (frames.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    try:
        import imageio

        imageio.mimwrite(path, list(u8.numpy()), fps=fps)
        return "imageio"
    except Exception:
        _remove(path)
    try:
        from torchvision.io import write_video as tv_write

        tv_write(path, u8, fps=fps)
        return "torchvision"
    except Exception:
        _remove(path)
    return None


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def summary_line(result: sc.Result, gen_s: Sequence[float], dec_s: Sequence[float]) -> str:
    """One line: cuts, per-event reaction ms and peak ratio, timing."""
    m = result.metrics
    events = []
    for (event, res), s in zip(result.handled, m["events"]):
        tag = res["kind"] if res["applied"] else res["reason"]
        if s is None:
            events.append(f"s{event.step}:{tag}:-")
            continue
        react = "-" if s["reaction_ms"] is None else f"{s['reaction_ms']:.0f}ms"
        ratio = "-" if s["peak_ratio"] is None else f"x{s['peak_ratio']:.1f}"
        events.append(f"s{event.step}:{tag}:{react}/{ratio}")
    timing = (
        f"step mean {m['generate_mean_s'] * 1e3:.0f} / max {m['generate_max_s'] * 1e3:.0f} ms"
    )
    if gen_s and dec_s:
        timing += (
            f" (gen {sum(gen_s) / len(gen_s) * 1e3:.0f}/{max(gen_s) * 1e3:.0f},"
            f" decode {sum(dec_s) / len(dec_s) * 1e3:.0f}/{max(dec_s) * 1e3:.0f})"
        )
    return (
        f"{result.scenario.name:<20} ok   cuts={m['cuts']} "
        f"events=[{' '.join(events)}] {timing}"
    )


# ---------------------------------------------------------------- the run


def select_scenarios(spec: str, ramp: int) -> list[sc.Scenario]:
    """`all` or a comma list of names from `default_scenarios(ramp)`."""
    every = sc.default_scenarios(ramp)
    if spec.strip() == "all":
        return every
    by_name = {s.name: s for s in every}
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in by_name]
    if unknown or not names:
        raise ValueError(f"unknown scenarios {unknown}; choose from {sorted(by_name)}")
    return [by_name[n] for n in names]


def run_all(
    backend: Any,
    scenarios: Sequence[sc.Scenario],
    out: str,
    fps: float,
    *,
    align: bool = False,
    seed: int | None = 0,
    hook_name: str | None = None,
    save_frames: bool = False,
    video: bool = True,
    header: str = "",
) -> tuple[str, int]:
    """Run each scenario on `backend`, write files to `out`; return (summary, failures)."""
    if align and "aligner" not in inspect.signature(sc.run).parameters:
        raise RuntimeError(
            "--align needs steering.eval.scenarios.run(..., aligner=...), "
            "which this checkout does not have yet"
        )
    hook = None
    if hook_name:
        fn = getattr(backend, hook_name, None)
        if not callable(fn):
            raise AttributeError(f"backend has no callable {hook_name!r} for the cache-reset hook")
        hook = fn
    encode = make_encode(backend, align)
    os.makedirs(out, exist_ok=True)

    lines: list[str] = []
    seeding = writers = None
    failures = 0
    for scenario in scenarios:
        adapter = EngineAdapter(backend)
        try:
            seeding = seed_and_reset(backend, seed)
            kw: dict[str, Any] = {"on_cond_changed": hook}
            if align:
                kw["aligner"] = Aligner(length=_rows(encode(scenario.initial_prompt)[0]))
            result = sc.run(scenario, adapter.generate, encode, fps, **kw)
            base = os.path.join(out, scenario.name)
            with open(base + ".report.txt", "w") as f:
                f.write(result.report() + "\n")
            if save_frames:
                result.save_frames(base + ".frames.pt")
            if video:
                writers = write_video(base + ".mp4", result.frames, fps) or "none"
            lines.append(summary_line(result, adapter.gen_s, adapter.dec_s))
        except Exception as exc:
            failures += 1
            last = traceback.format_exception_only(type(exc), exc)[-1].strip()
            lines.append(f"{scenario.name:<20} FAILED {last}")

    head = [header] if header else []
    head.append(
        f"seeding: {seeding or 'n/a'} (seed={seed}); align: {align}; "
        f"cache-reset hook: {hook_name or 'none (backend resets its own text cache)'}"
    )
    if not video:
        head.append("video: disabled")
    elif writers == "none":
        head.append("video: SKIPPED (no imageio+ffmpeg or torchvision+av importable)")
    else:
        head.append(f"video: {writers or 'n/a'}")
    head.append(f"{len(scenarios) - failures}/{len(scenarios)} scenarios ok")
    summary = "\n".join(head + lines) + "\n"
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(summary)
    return summary, failures


def _rows(emb: Tensor) -> int:
    """Token rows of a [1, L, D] or [L, D] embedding."""
    return emb.shape[-2]


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", required=True, help="dotted module name, e.g. engine.backends.wan")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--steps", type=int, default=2, help="denoising steps passed to load()")
    p.add_argument("--fps", type=float, default=16.0)
    p.add_argument("--ramp", type=int, default=4)
    p.add_argument("--scenarios", default="all", help="'all' or a comma list of names")
    p.add_argument("--align", action="store_true", help="use steering.align.Aligner")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-cache-reset-hook", metavar="NAME", default=None,
                   help="call backend.NAME() whenever the conditioning changes")
    p.add_argument("--save-frames", action="store_true", help="also write <name>.frames.pt")
    p.add_argument("--no-video", action="store_true", help="do not try to write mp4s")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    try:
        chosen = select_scenarios(args.scenarios, args.ramp)
    except ValueError as e:
        p.error(str(e))

    backend = load_backend(args.backend, args.width, args.height, args.steps)
    header = (
        f"backend {args.backend} {args.width}x{args.height} steps={args.steps} "
        f"fps={args.fps:g} ramp={args.ramp}"
    )
    summary, failures = run_all(
        backend, chosen, args.out, args.fps, align=args.align, seed=args.seed,
        hook_name=args.force_cache_reset_hook, save_frames=args.save_frames,
        video=not args.no_video, header=header,
    )
    print(summary, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""Check on the real model that the order of the text rows does not matter.

    python -m steering.eval.row_order_check --backend engine.backends.<name> --out DIR

Token alignment (`steering.align`) moves text rows to new positions; that is
only safe if the video model's cross-attention ignores row order. This
renders the same seeded clip from five conditionings of one prompt:

    A   as encoded              A2  A again (noise floor / determinism)
    B   real rows scattered     C   real rows reversed in place
    D   a different prompt (control: must differ)

and compares B, C, D and A2 with A. Writes `row_order_check.txt`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from .run_engine import encoder_of, load_backend, seed_and_reset, split_encoded, to_uint8_frames

DEFAULT_PROMPT = "a wide shot of a meadow with a single oak tree, gentle wind, cinematic"
CONTROL_PROMPT = "a city street at night in heavy rain"

# Verdict thresholds, in 0..255 pixel levels of mean absolute difference.
DET_TOL = 1.0  # A vs A2 above this: the run is not deterministic
RATIO = 3.0  # B, C may differ from A by up to RATIO x the A-vs-A2 difference...
FLOOR = 1.0  # ...or by this much, whichever is larger
D_MIN = 5.0  # D must differ by at least D_MIN x that tolerance to count as "clear"
D_SCALE = 0.5  # B or C at >= D_SCALE x the D difference is order dependence


# ---------------------------------------------------------------- tensors


def _rows(emb: Tensor) -> Tensor:
    """[L, D] view of a [1, L, D] or [L, D] embedding."""
    if emb.dim() == 3 and emb.shape[0] == 1:
        return emb[0]
    if emb.dim() == 2:
        return emb
    raise ValueError(f"embedding must be [1, L, D] or [L, D], got {tuple(emb.shape)}")


def real_count(emb: Tensor, ids: Sequence[int] | None) -> int:
    """Number of real token rows: len(ids), else the non-zero rows (must lead)."""
    rows = _rows(emb)
    if ids is not None:
        n = len(ids)
    else:
        nonzero = rows.abs().sum(dim=-1) != 0
        n = int(nonzero.sum())
        if n and not bool(nonzero[:n].all()):
            raise ValueError("non-zero rows are not a prefix; cannot infer the real count")
    if not 1 <= n <= rows.shape[0]:
        raise ValueError(f"real token count {n} outside [1, {rows.shape[0]}]")
    return n


def scatter_positions(n: int, length: int, a: int = 7, b: int = 13) -> list[int]:
    """Position (a*i + b) mod length for token i < n; raises if two collide."""
    pos = [(a * i + b) % length for i in range(n)]
    if len(set(pos)) != n:
        raise ValueError(f"scatter ({a}*i + {b}) mod {length} collides for n={n}")
    return pos


def build_conditionings(emb: Tensor, n: int, control: Tensor) -> dict[str, Tensor]:
    """A, B, C, D as described in the module docstring; same shape as `emb`.

    B is a full permutation of the rows: real row i goes to
    `scatter_positions(n, L)[i]`, the padding rows fill the other positions
    in order (so with zero padding it is "zeros elsewhere"). C reverses rows
    0..n-1. D is `control` (another prompt's encoding) as is.
    """
    rows = _rows(emb)
    if control.shape != emb.shape:
        raise ValueError(f"control shape {tuple(control.shape)} != {tuple(emb.shape)}")
    length = rows.shape[0]
    real = scatter_positions(n, length)
    taken = set(real)
    rest = [p for p in range(length) if p not in taken]
    perm = torch.tensor(real + rest, device=rows.device)
    b = torch.empty_like(rows)
    b[perm] = rows
    c = rows.clone()
    c[:n] = rows[:n].flip(0)

    def shaped(x: Tensor) -> Tensor:
        return x.unsqueeze(0) if emb.dim() == 3 else x

    return {"A": emb, "B": shaped(b), "C": shaped(c), "D": control}


# ---------------------------------------------------------------- comparison


def compare(x: Tensor, y: Tensor) -> dict[str, float]:
    """max_abs, mean_abs (0..255 levels) and PSNR (dB, inf if equal) of uint8 frames."""
    if x.shape != y.shape:
        raise ValueError(f"frame shapes differ: {tuple(x.shape)} vs {tuple(y.shape)}")
    d = (x.to(torch.float64) - y.to(torch.float64)).abs()
    mse = float((d**2).mean())
    psnr = math.inf if mse == 0 else 10.0 * math.log10(255.0**2 / mse)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "psnr": psnr}


def verdict(stats: Mapping[str, Mapping[str, float]]) -> tuple[str, list[str]]:
    """PASS, FAIL or INCONCLUSIVE from comparisons of A2, B, C, D with A.

    Uses mean_abs. Inconclusive if A vs A2 exceeds DET_TOL (not
    deterministic) or D is not clearly different. FAIL if B or C reaches
    D_SCALE x D. PASS if B and C are within max(RATIO x A2, FLOOR).
    """
    noise = stats["A2"]["mean_abs"]
    b, c, d = (stats[k]["mean_abs"] for k in ("B", "C", "D"))
    tol = max(RATIO * noise, FLOOR)
    why = [
        f"noise floor A-vs-A2 = {noise:.3f}; tolerance max({RATIO:g} x noise, {FLOOR:g}) = {tol:.3f}",
        f"B = {b:.3f}, C = {c:.3f}, control D = {d:.3f} (mean abs, 0..255 levels)",
    ]
    if noise > DET_TOL:
        return "INCONCLUSIVE", why + [
            f"A vs A2 differs by {noise:.3f} > {DET_TOL:g}: generation is not deterministic "
            "under this seeding, so differences cannot be attributed to row order"
        ]
    if d < D_MIN * tol:
        return "INCONCLUSIVE", why + [
            f"control D differs by only {d:.3f} < {D_MIN:g} x tolerance: the conditioning "
            "barely affects the frames at this setting"
        ]
    if max(b, c) >= D_SCALE * d:
        return "FAIL", why + [
            f"B or C differs by >= {D_SCALE:g} x the control: row order changes the video "
            "about as much as a different prompt; token alignment is NOT safe"
        ]
    if b <= tol and c <= tol:
        return "PASS", why + [
            "B and C match A within the noise tolerance while D differs clearly: "
            "row order does not matter; token alignment is safe"
        ]
    return "INCONCLUSIVE", why + [
        "B or C is above the noise tolerance but well below the control; likely float "
        "summation-order drift growing over chunks, check the first-chunk numbers"
    ]


# ---------------------------------------------------------------- the run


def render(backend: Any, cond: Tensor, chunks: int, seed: int | None) -> tuple[Tensor, int, str]:
    """Seed, reset, `chunks` steps with `cond` held fixed.

    Returns (uint8 [T, H, W, 3] frames on CPU, frames in the first chunk, seeding).
    """
    seeding = seed_and_reset(backend, seed)
    out = []
    for _ in range(chunks):
        out.append(to_uint8_frames(backend.decode_chunk(backend.generate_chunk(cond))))
    return torch.cat(out), out[0].shape[0], seeding


def check(
    backend: Any, prompt: str, chunks: int, seed: int | None, control_prompt: str = CONTROL_PROMPT
) -> tuple[str, str, dict[str, Tensor]]:
    """Render A, A2, B, C, D; return (verdict, report text, frames by label)."""
    encode = encoder_of(backend)
    emb, ids = split_encoded(encode(prompt))
    control, _ = split_encoded(encode(control_prompt))
    n = real_count(emb, ids)
    conds = build_conditionings(emb, n, control)
    # A2 is a fresh tensor object so the backend rebuilds its text cache.
    order = [("A", conds["A"]), ("A2", conds["A"].clone()), ("B", conds["B"]),
             ("C", conds["C"]), ("D", conds["D"])]
    frames: dict[str, Tensor] = {}
    first = 0
    seeding = ""
    for label, cond in order:
        frames[label], first, seeding = render(backend, cond, chunks, seed)

    stats = {k: compare(frames["A"], frames[k]) for k in ("A2", "B", "C", "D")}
    head = {k: compare(frames["A"][:first], frames[k][:first]) for k in stats}
    result, why = verdict(stats)

    length = _rows(emb).shape[0]
    lines = [
        "row order check",
        f"prompt: {prompt!r}",
        f"control D: {control_prompt!r}",
        f"real tokens: {n} of {length} rows ({'from ids' if ids is not None else 'inferred: non-zero rows'})",
        f"B positions: (7*i + 13) mod {length} -> {scatter_positions(n, length)[:8]}...",
        f"chunks: {chunks}, frames per run: {frames['A'].shape[0]}, seed: {seed} via {seeding}",
        "",
        f"{'vs A':<5} {'max_abs':>8} {'mean_abs':>9} {'psnr_dB':>8} {'chunk0_mean':>12}",
    ]
    for k, s in stats.items():
        lines.append(
            f"{k:<5} {s['max_abs']:>8.1f} {s['mean_abs']:>9.3f} {s['psnr']:>8.2f} "
            f"{head[k]['mean_abs']:>12.3f}"
        )
    lines += ["", f"VERDICT: {result}", *(f"  {w}" for w in why)]
    return result, "\n".join(lines) + "\n", frames


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", required=True, help="dotted module name, e.g. engine.backends.wan")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--steps", type=int, default=2, help="denoising steps passed to load()")
    p.add_argument("--chunks", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--save-frames", action="store_true", help="also write <label>.frames.pt")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.chunks < 1:
        p.error("--chunks must be >= 1")

    backend = load_backend(args.backend, args.width, args.height, args.steps)
    result, report, frames = check(backend, args.prompt, args.chunks, args.seed)
    report = f"backend {args.backend} {args.width}x{args.height} steps={args.steps}\n" + report
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "row_order_check.txt"), "w") as f:
        f.write(report)
    if args.save_frames:
        for label, x in frames.items():
            torch.save(x, os.path.join(args.out, f"{label}.frames.pt"))
    print(report, end="")
    return {"PASS": 0, "FAIL": 1}.get(result, 2)


if __name__ == "__main__":
    sys.exit(main())

"""Per-token interpolation between two conditioning tensors."""

import torch
from torch import Tensor

MAX_W = 1.3


def _check(a: Tensor, b: Tensor, w: float) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if not (0.0 <= w <= MAX_W):  # also rejects NaN
        raise ValueError(f"w must be in [0, {MAX_W}], got {w}")


def blend(a: Tensor, b: Tensor, w: float, eps: float = 1e-4) -> Tensor:
    """Interpolate each token vector (last dim) from `a` (w=0) to `b` (w=1).

    Direction is slerped between unit vectors and magnitude is lerped between
    norms. Tokens with a norm below `eps`, or whose directions are nearly
    collinear or opposite (sin of the angle below `eps`), use plain
    `a + w * (b - a)`. w in (1, 1.3] extrapolates past `b`.

    Returns a new tensor with `a`'s dtype and device; never modifies inputs.
    w == 0 and w == 1 return exact copies of `a` and `b`.
    Raises ValueError on shape mismatch or w outside [0, 1.3].
    """
    _check(a, b, w)
    if w == 0.0:
        return a.clone()
    if w == 1.0:
        return b.to(device=a.device, dtype=a.dtype, copy=True)

    work = torch.float64 if a.dtype == torch.float64 else torch.float32
    x = a.to(work)
    y = b.to(device=a.device, dtype=work)

    nx = x.norm(dim=-1, keepdim=True)
    ny = y.norm(dim=-1, keepdim=True)
    ux = x / nx.clamp_min(eps)
    uy = y / ny.clamp_min(eps)

    # atan2 of (sin, cos) is accurate at small angles, unlike acos.
    cos = (ux * uy).sum(dim=-1, keepdim=True)
    sin = (uy - cos * ux).norm(dim=-1, keepdim=True)
    theta = torch.atan2(sin, cos)

    spherical = (nx >= eps) & (ny >= eps) & (sin >= eps)
    safe_sin = torch.where(spherical, sin, torch.ones_like(sin))
    direction = (torch.sin((1.0 - w) * theta) * ux + torch.sin(w * theta) * uy) / safe_sin
    # Extrapolating can push the lerped norm below zero; clamp so the
    # direction never flips.
    magnitude = (nx + w * (ny - nx)).clamp_min(0.0)

    linear = x + w * (y - x)
    out = torch.where(spherical, direction * magnitude, linear)
    return out.to(a.dtype)


def apply_strength(without: Tensor, with_change: Tensor, s: float) -> Tensor:
    """Scale a change: s=0 gives `without`, s=1 gives `with_change`, up to 1.3.

    Same as `blend(without, with_change, s)`, with the same errors.
    """
    return blend(without, with_change, s)

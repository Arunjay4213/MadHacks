"""Judge a steer from rendered frames: cuts, peak change, reaction time.

All functions take a video tensor `frames` of shape [T, C, H, W], floats in
[0, 1]. Pure torch, no model.
"""

import torch
from torch import Tensor


def frame_change(frames: Tensor) -> Tensor:
    """Mean absolute difference between consecutive frames, shape [T-1].

    Entry j is the change from frame j to frame j+1.
    """
    if frames.dim() != 4 or frames.shape[0] < 2:
        raise ValueError(f"expected [T>=2, C, H, W], got {tuple(frames.shape)}")
    f = frames.to(torch.float32)
    return (f[1:] - f[:-1]).abs().flatten(1).mean(dim=1)


def _median_mad(x: Tensor) -> tuple[float, float]:
    med = x.median()
    return float(med), float((x - med).abs().median())


def find_cuts(
    frames: Tensor, k: float = 6.0, window: int = 8, floor: float = 0.01
) -> list[int]:
    """Frame indices where a hard cut lands (the first frame after the cut).

    Change j counts as a cut when it exceeds median + max(k * MAD, floor) of
    the previous `window` changes (at least 2 needed) and is also more than
    twice the next change. The second rule separates a one-frame jump from a
    cross-fade, whose change stays raised for many frames.
    """
    d = frame_change(frames)
    cuts = []
    for j in range(2, len(d)):
        med, mad = _median_mad(d[max(0, j - window) : j])
        if d[j] <= med + max(k * mad, floor):
            continue
        if j + 1 < len(d) and d[j] <= 2.0 * d[j + 1]:
            continue
        cuts.append(j + 1)
    return cuts


def transition_span(
    scores: Tensor, lo: float = 0.1, hi: float = 0.9, min_rise: float = 1e-3
) -> tuple[int, int] | None:
    """First indices where a rising per-frame score crosses `lo` and `hi`.

    The score is normalised so the mean of its first 10% is 0 and the mean of
    its last 10% is 1. Returns None if the rise is below `min_rise` or below
    3 standard deviations of those two end segments (noise).
    """
    s = scores.to(torch.float64).flatten()
    m = max(1, round(0.1 * len(s)))
    head, tail = s[:m], s[-m:]
    rise = float(tail.mean() - head.mean())
    noise = float(torch.cat([head - head.mean(), tail - tail.mean()]).std()) if m > 1 else 0.0
    if rise <= max(min_rise, 3.0 * noise):
        return None
    norm = (s - head.mean()) / rise
    above_lo = (norm >= lo).nonzero()
    above_hi = (norm >= hi).nonzero()
    if len(above_lo) == 0 or len(above_hi) == 0:
        return None
    return int(above_lo[0]), int(above_hi[0])


def summarize(frames: Tensor, steer_frame: int, fps: float, floor: float = 1e-3) -> dict:
    """Summary of how the video reacted to a steer.

    `steer_frame` is the first frame made with the new conditioning, so the
    change into it (index steer_frame - 1) is the first post-steer change.
    Keys: cuts, pre_median, pre_mad, peak_change, peak_frame, peak_ratio
    (peak / pre-steer median, None if that median is 0), reaction_ms (time
    from steer_frame until a change first exceeds pre median + max(3 MAD,
    floor), counted at the frame that change lands on; None if never).
    """
    d = frame_change(frames)
    if not (2 <= steer_frame <= len(d)):
        raise ValueError(f"steer_frame must be in [2, {len(d)}], got {steer_frame}")
    pre, post = d[: steer_frame - 1], d[steer_frame - 1 :]
    med, mad = _median_mad(pre)
    peak_j = int(post.argmax())
    peak = float(post[peak_j])

    reaction_ms = None
    over = (post > med + max(3.0 * mad, floor)).nonzero()
    if len(over):
        reaction_ms = int(over[0]) * 1000.0 / fps

    return {
        "cuts": find_cuts(frames),
        "pre_median": med,
        "pre_mad": mad,
        "peak_change": peak,
        "peak_frame": steer_frame + peak_j,
        "peak_ratio": peak / med if med > 0 else None,
        "reaction_ms": reaction_ms,
    }

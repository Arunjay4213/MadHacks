"""Smooth start-and-stop curve for pacing a move."""


def ease(r: float) -> float:
    """Min-jerk curve r^3 (10 - 15 r + 6 r^2), with r clamped to [0, 1].

    Guarantees ease(0) = 0, ease(1) = 1, ease(0.5) = 0.5, monotonic
    non-decreasing, and ease(r) + ease(1 - r) = 1.
    """
    r = min(max(float(r), 0.0), 1.0)
    return r**3 * (10.0 - 15.0 * r + 6.0 * r**2)

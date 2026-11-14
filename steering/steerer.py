"""Moves the conditioning one chunk at a time toward a target; interruptible."""

import threading

from torch import Tensor

from .blend import blend
from .ease import ease


class Steerer:
    """Holds the conditioning tensor the generation loop uses for each chunk.

    `set_target` (control thread) starts a move from wherever the
    conditioning is now. `get_cond` (generation thread) advances the move by
    one chunk and returns the tensor for that chunk. Both are thread safe.
    Tensors passed in or returned are never modified in place.
    """

    def __init__(self, initial_cond: Tensor) -> None:
        self._lock = threading.Lock()
        self._now = initial_cond
        self._start = initial_cond
        self._target = initial_cond
        self._ramp = 0
        self._k = 0
        self._cap = 1.0

    def set_target(self, target: Tensor, ramp_chunks: int, cap: float = 1.0) -> None:
        """Start a move from the current conditioning to `target`.

        The move takes `ramp_chunks` calls to `get_cond` and stops at
        blend(start, target, cap). Takes effect on the next `get_cond`.
        Raises ValueError if the shape differs, ramp_chunks is not an int >= 1,
        or cap is not in (0, 1]. On error, the current move is unchanged.
        """
        if isinstance(ramp_chunks, bool) or not isinstance(ramp_chunks, int):
            raise ValueError(f"ramp_chunks must be an int, got {ramp_chunks!r}")
        if ramp_chunks < 1:
            raise ValueError(f"ramp_chunks must be >= 1, got {ramp_chunks}")
        if isinstance(cap, bool) or not (0.0 < cap <= 1.0):  # also rejects NaN
            raise ValueError(f"cap must be in (0, 1], got {cap!r}")
        with self._lock:
            if target.shape != self._now.shape:
                raise ValueError(
                    f"target shape {tuple(target.shape)} != {tuple(self._now.shape)}"
                )
            self._target = target.to(device=self._now.device, dtype=self._now.dtype)
            self._start = self._now
            self._ramp = ramp_chunks
            self._k = 0
            self._cap = float(cap)

    def get_cond(self) -> Tensor:
        """Advance the current move by one chunk and return the conditioning.

        With cap 1.0 the result equals the target exactly after `ramp_chunks`
        calls. Once a move is finished, returns the same tensor object.
        """
        with self._lock:
            if self._k < self._ramp:
                self._k += 1
                if self._k == self._ramp and self._cap == 1.0:
                    self._now = self._target
                else:
                    w = min(ease(self._k / self._ramp), self._cap)
                    self._now = blend(self._start, self._target, w)
            return self._now

    @property
    def current(self) -> Tensor:
        """The tensor `get_cond` last returned (initially the initial cond).

        Read-only: does not advance the move.
        """
        with self._lock:
            return self._now

    @property
    def progress(self) -> float:
        """Fraction of the current move done (k / ramp); 1.0 when idle."""
        with self._lock:
            return 1.0 if self._ramp == 0 else self._k / self._ramp

    @property
    def is_moving(self) -> bool:
        """True while the current move has chunks left."""
        with self._lock:
            return self._k < self._ramp

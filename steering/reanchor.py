"""Decides when the engine should re-anchor (repaint from a noised last frame) and how hard.

No model code: the engine calls `on_target` after each `Controller.handle`,
`next_chunk` once per chunk after `steerer.get_cond()`, and `on_reset` when a
new video starts, then passes any returned anchor to the backend as
`backend.reanchor(anchor.strength, source=anchor.source)`.
"""

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_EPS = 1e-9  # progress comparisons tolerate float rounding
SOURCES = ("last", "clean")  # where the backend's re-anchor starts from; see Anchor.source


@dataclass(frozen=True)
class Anchor:
    """One re-anchor for the next chunk."""

    strength: float  # in (0, 1]
    reason: str  # "steer" or "refresh"
    # "last": start from the last latent (a steer: repaint toward the new target).
    # "clean": start from the clean anchor, the latent the backend saved a few chunks after the
    # last reset or steer (a refresh: undo drift without baking it in).
    source: str = "last"


def _check_int(name: str, value: object, low: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{name} must be an int >= {low}, got {value!r}")


def _check_unit(name: str, value: object, allow_zero: bool = False) -> None:
    ok = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and not math.isnan(value)
        and (0.0 <= value if allow_zero else 0.0 < value)
        and value <= 1.0
    )
    if not ok:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be a number in {interval}, got {value!r}")


@dataclass(frozen=True)
class ReanchorConfig:
    """Tuning knobs.

    Fire timing is measured on the real model, not guessed: an anchor repaints
    with the conditioning current on its chunk and the scene stays locked to it,
    so the (last) anchor must fire once the ramp has reached the full target
    (progress 1.0). Firing mid-ramp repaints toward a half-blended prompt.
    The refresh is measured too (MODEL_NOTES.md, 360 chunks, no commands): from
    the last (already drifted) latent it bakes the damage in and degrades from
    ~chunk 180; from the clean anchor at strength 0.5 (anchor step 2 of 4) every
    24 chunks the picture stays clean and the same scene. The steer strengths
    and the gap are still untuned.
    """

    steer_strength: float = 0.85  # for a normal final/confirm
    sudden_strength: float = 1.0  # when ramp_chunks <= sudden_max_ramp
    sudden_max_ramp: int = 2
    slow_min_ramp: int = 10  # ramp_chunks >= this is a "slow" steer
    slow_strength: float = 0.5  # each anchor of a slow steer
    slow_anchors: int = 3  # anchors spread over a slow ramp
    fire_at_progress: float = 1.0  # single anchor fires on the first chunk with progress >= this (1.0: the full target)
    provisional_strength: float = 0.0  # 0 means provisional targets never re-anchor
    refresh_every_chunks: int | None = 24  # None disables the refresh
    refresh_strength: float = 0.5  # anchor step 2 of the 4-step first-frame schedule
    refresh_source: str = "clean"  # "clean" (the clean anchor) or "last" (the drifted last latent)
    min_gap_chunks: int = 8
    min_strength: float = 0.05  # anything that scales below this is skipped

    def __post_init__(self) -> None:
        for name in ("steer_strength", "sudden_strength", "slow_strength",
                     "fire_at_progress", "refresh_strength"):
            _check_unit(name, getattr(self, name))
        for name in ("provisional_strength", "min_strength"):
            _check_unit(name, getattr(self, name), allow_zero=True)
        _check_int("sudden_max_ramp", self.sudden_max_ramp, 0)
        _check_int("slow_min_ramp", self.slow_min_ramp, 1)
        _check_int("slow_anchors", self.slow_anchors, 1)
        _check_int("min_gap_chunks", self.min_gap_chunks, 1)
        if self.refresh_source not in SOURCES:
            raise ValueError(f"refresh_source must be one of {SOURCES}, got {self.refresh_source!r}")
        if self.refresh_every_chunks is not None:
            _check_int("refresh_every_chunks", self.refresh_every_chunks, 1)
        if self.slow_min_ramp <= self.sudden_max_ramp:
            raise ValueError(
                f"slow_min_ramp ({self.slow_min_ramp}) must be > sudden_max_ramp "
                f"({self.sudden_max_ramp}), or a ramp would be both sudden and slow"
            )


@dataclass
class _Pending:
    """A scheduled steer: thresholds still to fire, in increasing order."""

    thresholds: list[float]
    strength: float
    ramp: int
    seen: int = 0  # next_chunk calls since it was scheduled


class ReanchorPolicy:
    """When and how strongly to re-anchor; thread safe (one lock).

    Call `on_target(msg, result)` right after `Controller.handle(msg)`,
    `next_chunk(steerer.progress, steerer.is_moving)` once per chunk after
    `steerer.get_cond()`, and `on_reset()` when a new video starts.
    Construction counts as a reset.
    """

    def __init__(self, config: ReanchorConfig = ReanchorConfig()) -> None:
        self._cfg = config
        self._lock = threading.Lock()
        self._pending: _Pending | None = None
        self._since = 0  # chunks since the last anchor or reset

    @property
    def config(self) -> ReanchorConfig:
        return self._cfg

    @property
    def chunks_since_anchor(self) -> int:
        """next_chunk calls since the last anchor or reset."""
        with self._lock:
            return self._since

    @property
    def pending(self) -> bool:
        """True while a steer anchor is scheduled and not yet (fully) fired."""
        with self._lock:
            return self._pending is not None

    def fires_next_chunk(self) -> bool:
        """True when the next next_chunk call will return a steer anchor for sure: a pending sudden steer
        (ramp 1, the live final) whose gap is already met. The engine may then stop the chunk in flight."""
        with self._lock:
            p = self._pending
            return (p is not None and p.seen == 0 and p.ramp == 1
                    and self._since + 1 >= self._cfg.min_gap_chunks)

    def on_target(self, msg: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        """Schedule (or cancel) a steer anchor for an applied contract-2 message.

        Not applied (stale, unchanged): the pending steer is left alone.
        Applied: the new target replaces the pending one (latest wins), even if
        the new one schedules nothing (provisional with provisional_strength 0,
        or strength under min_strength): the old anchor would repaint toward a
        target that is no longer the goal.
        """
        if not result.get("applied"):
            return
        kind = result.get("kind")
        if kind not in ("final", "confirm", "provisional"):
            return
        plan = self._plan(msg, kind)
        with self._lock:
            self._pending = plan

    def on_reset(self) -> None:
        """A new video: drop any pending steer and restart the gap/refresh counts."""
        with self._lock:
            self._pending = None
            self._since = 0

    def next_chunk(self, progress: float, is_moving: bool) -> Anchor | None:
        """Call once per chunk after get_cond; returns the anchor for this chunk or None."""
        cfg = self._cfg
        with self._lock:
            self._since += 1
            gap_ok = self._since >= cfg.min_gap_chunks
            p = self._pending
            if p is not None:
                p.seen += 1
                due = self._due(p, progress, is_moving)
                if due == 0 or not gap_ok:
                    return None  # not due yet, or deferred by the gap
                del p.thresholds[:due]  # thresholds due together fire once
                if not p.thresholds:
                    self._pending = None
                self._since = 0
                return Anchor(p.strength, "steer")
            every = cfg.refresh_every_chunks
            if every is not None and not is_moving and gap_ok and self._since >= every:
                self._since = 0
                return Anchor(cfg.refresh_strength, "refresh", cfg.refresh_source)
            return None

    # --- internals ------------------------------------------------------------

    def _plan(self, msg: Mapping[str, Any], kind: str) -> _Pending | None:
        cfg = self._cfg
        ramp = int(msg["ramp_chunks"])
        scale = msg.get("strength")
        scale = 1.0 if scale is None else min(float(scale), 1.0)
        single = [cfg.fire_at_progress]
        if kind == "provisional":
            if cfg.provisional_strength <= 0.0:
                return None
            cap = float(msg.get("cap", 1.0))
            strength, thresholds = cfg.provisional_strength * min(cap, 1.0), single
        elif ramp >= cfg.slow_min_ramp:
            n = cfg.slow_anchors
            strength = cfg.slow_strength * scale
            # Evenly spaced, last one at 1.0: the final repaint uses the full target.
            thresholds = [i / n for i in range(1, n + 1)]
        elif ramp <= cfg.sudden_max_ramp:
            strength, thresholds = cfg.sudden_strength * scale, single
        else:
            strength, thresholds = cfg.steer_strength * scale, single
        if strength < cfg.min_strength or strength <= 0.0:
            return None
        return _Pending(list(thresholds), min(strength, 1.0), ramp)

    @staticmethod
    def _due(p: _Pending, progress: float, is_moving: bool) -> int:
        """How many of p's thresholds are due on this chunk (caller holds the lock)."""
        if p.seen == 1 and p.ramp >= 2 and not is_moving:
            # The first chunk of a ramp >= 2 is always moving, so this reading
            # predates the steerer seeing the new target: ignore it.
            return 0
        if not is_moving:
            return len(p.thresholds)  # ramp over: anything left fires now
        # Never trust a progress ahead of the chunks this target has had.
        progress = min(progress, p.seen / p.ramp)
        return sum(1 for t in p.thresholds if t <= progress + _EPS)

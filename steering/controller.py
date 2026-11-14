"""Turns contract-2 messages into steerer calls (early steering included)."""

import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence

from torch import Tensor

from .align import Aligner, Layout
from .blend import MAX_W, blend
from .steerer import Steerer

# What `encode` returns: a tensor, or (tensor, real token ids) with an aligner.
Encoded = Tensor | tuple[Tensor, Sequence[int]]
# Full identity of a target: (prompt, strength, without_prompt).
Key = tuple[str, float | None, str | None]


def _validate_strength(strength: object, without: object) -> tuple[float | None, str | None]:
    """Return (strength, without_prompt), both None or both set, or raise ValueError."""
    if (strength is None) != (without is None):
        raise ValueError("strength and without_prompt must be given together")
    if strength is None:
        return None, None
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or math.isnan(strength)
        or not 0.0 < strength <= MAX_W
    ):
        raise ValueError(f"strength must be a number in (0, {MAX_W}], got {strength!r}")
    if not isinstance(without, str) or not without:
        raise ValueError(f"without_prompt must be a non-empty str, got {without!r}")
    return float(strength), without


def _validate(msg: Mapping) -> tuple[Key, int, bool, float]:
    """Return (key, ramp_chunks, provisional, cap) or raise ValueError."""
    prompt = msg.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"prompt must be a non-empty str, got {prompt!r}")
    ramp = msg.get("ramp_chunks")
    if isinstance(ramp, bool) or not isinstance(ramp, int) or ramp < 1:
        raise ValueError(f"ramp_chunks must be an int >= 1, got {ramp!r}")
    provisional = msg.get("provisional", False)
    if not isinstance(provisional, bool):
        raise ValueError(f"provisional must be a bool, got {provisional!r}")
    cap = msg.get("cap", 1.0)
    if (
        isinstance(cap, bool)
        or not isinstance(cap, (int, float))
        or math.isnan(cap)
        or not 0.0 < cap <= 1.0
    ):
        raise ValueError(f"cap must be a float in (0, 1], got {cap!r}")
    if not provisional and cap != 1.0:
        raise ValueError(f"a final (non-provisional) message must have cap 1.0, got {cap}")
    strength, without = _validate_strength(msg.get("strength"), msg.get("without_prompt"))
    return (prompt, strength, without), ramp, provisional, float(cap)


class Controller:
    """Applies contract-2 messages to a Steerer.

    Embeddings come from an LRU cache keyed by the exact prompt string and
    filled by `encode`, which is always called outside the controller's lock.
    `cache_size` must be >= the number of prompts passed to `precompute`, or
    early ones are evicted like any other entry. Cached tensors are never
    modified. `handle` and `reset` are thread safe; when messages race, the
    one whose `handle` call started last wins and older ones are dropped as
    "stale". A `reset` makes every `handle` call started before it stale.

    A message may carry `strength` and `without_prompt` (both or neither);
    the target is then blend(E(without_prompt), E(prompt), strength). A target
    is identified by (prompt, strength, without_prompt) for "unchanged" and
    "confirm".

    With an `aligner`, `encode` must return (embedding, real token ids). The
    cache keeps that raw pair; each applied message is placed by the aligner
    right before `set_target`, and the aligner's layout is committed only if
    `set_target` succeeds. New tokens avoid positions whose row is non-zero
    in `steerer.current` (see `Aligner`). `without_prompt` is placed against
    the prompt's new layout and never committed. The aligner must already
    hold the layout of the steerer's initial conditioning: build the Steerer
    from `aligner.place(initial_emb, initial_ids)`.
    """

    def __init__(
        self,
        steerer: Steerer,
        encode: Callable[[str], Encoded],
        cache_size: int = 64,
        aligner: Aligner | None = None,
    ) -> None:
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError(f"cache_size must be an int >= 1, got {cache_size!r}")
        if aligner is not None and aligner.layout is None:
            raise ValueError(
                "aligner has no layout yet: build the Steerer from "
                "aligner.place(initial_emb, initial_ids) first"
            )
        self._steerer = steerer
        self._encode = encode
        self._aligner = aligner
        self._cache_size = cache_size
        self._cache: OrderedDict[str, Encoded] = OrderedDict()
        self._lock = threading.Lock()
        self._next_seq = 0  # handed out at entry to `handle`
        self._last_seq = -1  # seq of the newest message that reached a decision
        self._committed: Key | None = None
        self._pending: Key | None = None

    @property
    def committed_prompt(self) -> str | None:
        """Prompt of the last non-provisional target applied (None before any)."""
        with self._lock:
            return None if self._committed is None else self._committed[0]

    @property
    def pending_prompt(self) -> str | None:
        """Prompt of the provisional move in flight, else None."""
        with self._lock:
            return None if self._pending is None else self._pending[0]

    def precompute(self, prompts: Iterable[str]) -> None:
        """Encode and cache `prompts` so handling them later needs no encoding."""
        for prompt in prompts:
            self._embedding(prompt)

    def handle(self, msg: Mapping) -> dict:
        """Apply one contract-2 message; returns what was done.

        Returns {"applied", "kind", "reason", "cache_hit"}; kind is
        "provisional", "confirm", "final" or None, reason is "stale",
        "unchanged" or None; cache_hit is True only if every prompt it names
        was cached. Raises ValueError on a bad message; on any error
        (including from `encode`) nothing changes.
        """
        key, ramp, provisional, cap = _validate(msg)
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
        emb, without, cache_hit = self._embeddings(key)  # may be slow; no lock held

        def result(kind: str | None, reason: str | None = None) -> dict:
            return {"applied": reason is None, "kind": kind, "reason": reason,
                    "cache_hit": cache_hit}

        with self._lock:
            if seq < self._last_seq:
                return result(None, "stale")
            # A repeated guess is a no-op too: restarting a capped move toward
            # the same target would compound the cap past its limit.
            if (self._pending is None and key == self._committed) or (
                provisional and key == self._pending
            ):
                self._last_seq = seq
                return result(None, "unchanged")
            # The target may fail (shape mismatch); state changes only after it.
            target, layout = self._target(emb, without, key[1])
            if provisional:
                self._set_target(target, layout, ramp, cap)
                self._last_seq = seq
                self._pending = key
                return result("provisional")
            kind = "confirm" if key == self._pending else "final"
            self._set_target(target, layout, ramp, 1.0)
            self._last_seq = seq
            self._pending = None
            self._committed = key
            return result(kind)

    def reset(
        self, prompt: str, strength: float | None = None, without_prompt: str | None = None
    ) -> Tensor:
        """Start over on `prompt` with no ramp (a new video); returns the new cond.

        Encodes outside the lock, places the prompt in the identity layout
        (with an aligner), calls `steerer.reset`, commits this target, clears
        any pending guess and makes every `handle` call already started
        stale. Raises ValueError on bad arguments; on any error (including
        from `encode`) nothing changes.
        """
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"prompt must be a non-empty str, got {prompt!r}")
        strength, without_prompt = _validate_strength(strength, without_prompt)
        key: Key = (prompt, strength, without_prompt)
        emb, without, _ = self._embeddings(key)
        with self._lock:
            target, layout = self._target(emb, without, strength, against=())
            self._steerer.reset(target)
            if self._aligner is not None:
                self._aligner.commit(layout)
            self._last_seq = self._next_seq  # older handle calls are now stale
            self._next_seq += 1
            self._pending = None
            self._committed = key
            return target

    def _target(
        self,
        emb: Encoded,
        without: Encoded | None,
        strength: float | None,
        against: Layout | None = None,
    ) -> tuple[Tensor, Layout | None]:
        """Return (target tensor, prompt's layout or None); caller holds the lock.

        Places with the aligner (against the committed layout, or `against`)
        without committing; `without` is placed against the prompt's layout.
        """
        if self._aligner is None:
            target = emb if without is None else blend(without, emb, strength)
            return target, None
        # New tokens avoid rows still non-zero now (e.g. an older layout fading
        # out after a capped or interrupted move). A reset starts a new stream.
        occupied = None
        if against is None:
            cur = self._steerer.current
            occupied = (cur[0] if cur.dim() == 3 else cur).ne(0).any(-1)
        placed, layout = self._aligner.prepare(*emb, occupied=occupied, against=against)
        if without is None:
            return placed, layout
        # Shared tokens land on the prompt's positions, so each blends with
        # itself; prompt-only tokens blend with zero rows.
        base, _ = self._aligner.prepare(*without, occupied=occupied, against=layout)
        return blend(base, placed, strength), layout

    def _set_target(self, target: Tensor, layout: Layout | None, ramp: int, cap: float) -> None:
        """Set the target, then commit the layout; caller holds the lock."""
        self._steerer.set_target(target, ramp, cap)
        if layout is not None:
            self._aligner.commit(layout)  # only once the steerer accepted it

    def _embeddings(self, key: Key) -> tuple[Encoded, Encoded | None, bool]:
        """Return (E(prompt), E(without_prompt) or None, both cache hits)."""
        emb, hit = self._embedding(key[0])
        if key[2] is None:
            return emb, None, hit
        without, hit_w = self._embedding(key[2])
        return emb, without, hit and hit_w

    def _embedding(self, prompt: str) -> tuple[Encoded, bool]:
        """Return (encoded, cache_hit), encoding outside the lock on a miss."""
        with self._lock:
            emb = self._cache.get(prompt)
            if emb is not None:
                self._cache.move_to_end(prompt)
                return emb, True
        emb = self._encode(prompt)
        if self._aligner is not None and not (
            isinstance(emb, tuple) and len(emb) == 2 and isinstance(emb[0], Tensor)
        ):
            raise TypeError(
                "with an aligner, encode must return (embedding, token_ids), "
                f"got {type(emb).__name__}"
            )
        if self._aligner is not None:
            emb = (emb[0], tuple(int(i) for i in emb[1]))  # cached; keep it immutable
        with self._lock:
            self._cache[prompt] = emb
            self._cache.move_to_end(prompt)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return emb, False

"""Turns contract-2 messages into steerer calls (early steering included)."""

import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping

from torch import Tensor

from .steerer import Steerer


def _validate(msg: Mapping) -> tuple[str, int, bool, float]:
    """Return (prompt, ramp_chunks, provisional, cap) or raise ValueError."""
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
    return prompt, ramp, provisional, float(cap)


class Controller:
    """Applies contract-2 messages to a Steerer.

    Embeddings come from an LRU cache keyed by the exact prompt string and
    filled by `encode`, which is always called outside the controller's lock.
    `cache_size` must be >= the number of prompts passed to `precompute`, or
    early ones are evicted like any other entry. Cached tensors are never
    modified. `handle` is thread safe; when messages race, the one whose
    `handle` call started last wins and older ones are dropped as "stale".
    """

    def __init__(
        self, steerer: Steerer, encode: Callable[[str], Tensor], cache_size: int = 64
    ) -> None:
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError(f"cache_size must be an int >= 1, got {cache_size!r}")
        self._steerer = steerer
        self._encode = encode
        self._cache_size = cache_size
        self._cache: OrderedDict[str, Tensor] = OrderedDict()
        self._lock = threading.Lock()
        self._next_seq = 0  # handed out at entry to `handle`
        self._last_seq = -1  # seq of the newest message that reached a decision
        self._committed: str | None = None
        self._pending: str | None = None

    @property
    def committed_prompt(self) -> str | None:
        """Last non-provisional prompt applied (None before any)."""
        with self._lock:
            return self._committed

    @property
    def pending_prompt(self) -> str | None:
        """Prompt of the provisional move in flight, else None."""
        with self._lock:
            return self._pending

    def precompute(self, prompts: Iterable[str]) -> None:
        """Encode and cache `prompts` so handling them later needs no encoding."""
        for prompt in prompts:
            self._embedding(prompt)

    def handle(self, msg: Mapping) -> dict:
        """Apply one contract-2 message; returns what was done.

        Returns {"applied", "kind", "reason", "cache_hit"}; kind is
        "provisional", "confirm", "final" or None, reason is "stale",
        "unchanged" or None. Raises ValueError on a bad message; on any error
        (including from `encode`) nothing changes.
        """
        prompt, ramp, provisional, cap = _validate(msg)
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
        emb, cache_hit = self._embedding(prompt)  # may be slow; no lock held

        def result(kind: str | None, reason: str | None = None) -> dict:
            return {"applied": reason is None, "kind": kind, "reason": reason,
                    "cache_hit": cache_hit}

        with self._lock:
            if seq < self._last_seq:
                return result(None, "stale")
            # A repeated guess is a no-op too: restarting a capped move toward
            # the same target would compound the cap past its limit.
            if (self._pending is None and prompt == self._committed) or (
                provisional and prompt == self._pending
            ):
                self._last_seq = seq
                return result(None, "unchanged")
            # set_target may raise (shape mismatch); state changes only after it.
            if provisional:
                self._steerer.set_target(emb, ramp, cap)
                self._last_seq = seq
                self._pending = prompt
                return result("provisional")
            kind = "confirm" if prompt == self._pending else "final"
            self._steerer.set_target(emb, ramp, 1.0)
            self._last_seq = seq
            self._pending = None
            self._committed = prompt
            return result(kind)

    def _embedding(self, prompt: str) -> tuple[Tensor, bool]:
        """Return (embedding, cache_hit), encoding outside the lock on a miss."""
        with self._lock:
            emb = self._cache.get(prompt)
            if emb is not None:
                self._cache.move_to_end(prompt)
                return emb, True
        emb = self._encode(prompt)
        with self._lock:
            self._cache[prompt] = emb
            self._cache.move_to_end(prompt)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return emb, False

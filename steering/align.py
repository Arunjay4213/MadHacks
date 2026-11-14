"""Keep each token's row at a stable position when a prompt changes.

The steerer blends two conditioning tensors position by position. When a
phrase is added or removed, the tokens after it shift, so a plain per-position
blend mixes unrelated rows (a word with a comma, a word with padding). The
video model's text cross-attention has no positional information for text
tokens and padding rows are zero, so the order of the rows does not matter to
it. `Aligner` therefore moves each real token row so that a token shared with
the previous prompt keeps the position it had there.
"""

import difflib
import threading
from collections.abc import Sequence

import torch
from torch import Tensor

# (position, token id) for each real token, in prompt order.
Layout = tuple[tuple[int, int], ...]


class Aligner:
    """Places token rows so shared tokens keep their positions across prompts.

    Remembers the layout of the last committed prompt. The first call (or the
    first after `reset`) is the identity. Later calls match the new token ids
    against the previous ones with difflib; a matched token keeps its
    previous position, and each unmatched one gets the lowest free position,
    preferring positions the previous layout did not use (so a new word fades
    in from zero rather than out of a word that is leaving).

    `place` = `prepare` + `commit`. Use `prepare`/`commit` to commit only
    after the placed tensor was actually used. All methods are thread safe.
    """

    def __init__(self, length: int = 512) -> None:
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError(f"length must be an int >= 1, got {length!r}")
        self.length = length
        self._lock = threading.Lock()
        self._layout: Layout | None = None

    @property
    def layout(self) -> Layout | None:
        """Layout of the last committed prompt (None before any / after reset)."""
        with self._lock:
            return self._layout

    def place(self, emb: Tensor, ids: Sequence[int]) -> Tensor:
        """Return `emb` with its real rows moved to their positions; commit.

        `emb` is [L, D] or [1, L, D] with L == length, whose first len(ids)
        rows are the real tokens. Returns a new tensor of the same shape,
        dtype and device; every position not holding a token is exactly zero.
        Raises ValueError on a wrong shape or len(ids) not in [1, length].
        """
        with self._lock:
            out, layout = self._prepare(emb, ids)
            self._layout = layout
            return out

    def prepare(self, emb: Tensor, ids: Sequence[int]) -> tuple[Tensor, Layout]:
        """Like `place` but does not commit; returns (placed, layout)."""
        with self._lock:
            return self._prepare(emb, ids)

    def commit(self, layout: Layout) -> None:
        """Make `layout` (from `prepare`) the previous layout for the next call."""
        with self._lock:
            self._layout = tuple(layout)

    def reset(self) -> None:
        """Forget the previous layout; the next call is the identity."""
        with self._lock:
            self._layout = None

    def _prepare(self, emb: Tensor, ids: Sequence[int]) -> tuple[Tensor, Layout]:
        ids = [int(i) for i in ids]
        n = len(ids)
        rows = emb[0] if emb.dim() == 3 and emb.shape[0] == 1 else emb
        if rows.dim() != 2 or rows.shape[0] != self.length:
            raise ValueError(
                f"emb must be [{self.length}, D] or [1, {self.length}, D], "
                f"got {tuple(emb.shape)}"
            )
        if not 1 <= n <= self.length:
            raise ValueError(f"need 1 <= len(ids) <= {self.length}, got {n}")

        positions = self._assign(ids)
        out_rows = torch.zeros_like(rows)
        out_rows[torch.tensor(positions, device=rows.device)] = rows[:n]
        out = out_rows.unsqueeze(0) if emb.dim() == 3 else out_rows
        return out, tuple(zip(positions, ids))

    def _assign(self, ids: list[int]) -> list[int]:
        if self._layout is None:
            return list(range(len(ids)))
        prev_pos = [p for p, _ in self._layout]
        prev_ids = [i for _, i in self._layout]
        out: list[int | None] = [None] * len(ids)
        matcher = difflib.SequenceMatcher(None, prev_ids, ids, autojunk=False)
        for a, b, size in matcher.get_matching_blocks():
            for k in range(size):
                out[b + k] = prev_pos[a + k]
        taken = {p for p in out if p is not None}
        was_used = set(prev_pos)
        free = [p for p in range(self.length) if p not in taken and p not in was_used]
        free += [p for p in range(self.length) if p not in taken and p in was_used]
        it = iter(free)
        return [p if p is not None else next(it) for p in out]

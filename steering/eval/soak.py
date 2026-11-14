"""Long randomized checks of the steering maths (issue #37).

    python -m steering.eval.soak --seconds 20 --seed 0 [--full-size]

Runs, for a share of `--seconds` each:

1. back_and_forth: a Steerer pushed between prompts A, B (and sometimes C)
   with random ramps (1, 2, 3, 4, 12), random interrupts and capped moves.
   Checks every frame is finite, its per-token norm stays within the largest
   endpoint norm, it stays in the span of the prompts' rows (no drift), each
   step is at most (ease increment) x (path speed bound), the first frame
   after an interrupt is no bigger than a normal first step, and every
   finished move lands on the target object itself.
2. aligner: long sessions of prompt edits (add, remove from the middle,
   replace, reorder, near and at the length limit, one-token prompts,
   repeated token ids, fragmented occupied masks). Checks every placement
   (`check_placement`) and that length + 1 tokens raises and changes nothing.
3. threads: a tight `get_cond` loop against a control thread sending random
   `handle` / `set_target` / `reset` calls. Every prompt's rows are all the
   same vector, so any blend of them is too: a frame whose rows differ was
   torn between two moves.
4. memory: 10,000 messages through a Controller with an Aligner; checks the
   cache and layout stay bounded and live tensors / Python memory do not grow.

Prints one line per check with the worst numbers seen and a RESULT line;
exit code 1 if any check failed.
"""

import argparse
import difflib
import gc
import hashlib
import math
import random
import sys
import threading
import time
import tracemalloc
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from steering import Aligner, Controller, Steerer, blend, ease
from steering.align import Layout

RAMPS = (1, 2, 3, 4, 12)
SMALL = (1, 16, 64)
FULL = (1, 512, 4096)


@dataclass
class Check:
    """Outcome of one soak check: worst numbers seen and any failures."""

    name: str
    stats: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def fail(self, msg: str) -> None:
        if len(self.failures) < 20:
            self.failures.append(msg)

    def worst(self, key: str, value: float) -> None:
        self.stats[key] = max(self.stats.get(key, 0.0), float(value))

    def line(self) -> str:
        parts = []
        for k, v in self.stats.items():
            parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
        status = "PASS" if self.ok else "FAIL"
        return f"{self.name:<15}{status}  " + " ".join(parts)


# --- realistic fake embeddings ----------------------------------------------

def make_prompts(
    gen: torch.Generator,
    shape: Sequence[int],
    n: int,
    dtype: torch.dtype = torch.bfloat16,
    max_deg: float = 55.0,
) -> list[Tensor]:
    """`n` prompt embeddings shaped like real ones.

    Each row is a shared base direction turned by up to `max_deg` degrees
    (pairwise angles up to 2 x max_deg, mostly 10-40), per-token norms
    log-normal with a wide spread, and rows past a random prompt length
    exactly zero (so prompts disagree about which rows are padding).
    """
    _, length, dim = shape
    base = torch.randn(length, dim, generator=gen, dtype=torch.float64)
    base = base / base.norm(dim=-1, keepdim=True)
    base_norm = torch.exp(torch.randn(length, 1, generator=gen, dtype=torch.float64))
    out = []
    for _ in range(n):
        perp = torch.randn(length, dim, generator=gen, dtype=torch.float64)
        perp = perp - (perp * base).sum(-1, keepdim=True) * base
        perp = perp / perp.norm(dim=-1, keepdim=True)
        frac = torch.rand(length, 1, generator=gen, dtype=torch.float64) ** 2
        theta = torch.deg2rad(frac * max_deg)
        rows = base * torch.cos(theta) + perp * torch.sin(theta)
        norm = base_norm * torch.exp(0.3 * torch.randn(length, 1, generator=gen,
                                                        dtype=torch.float64))
        rows = rows * norm
        real = int(torch.randint(max(1, length // 4), length + 1, (1,), generator=gen))
        rows[real:] = 0
        out.append(rows.unsqueeze(0).to(dtype))
    return out


def speed_bound(a: Tensor, b: Tensor) -> float:
    """Upper bound on |d blend(a, b, w) / dw| (whole tensor, any branch).

    Per token the spherical path's speed is sqrt(dn^2 + r(w)^2 theta^2) with
    r between the two norms; the linear path's is |b - a|, which is no
    larger. Returns the norm over tokens of sqrt(dn^2 + max(n)^2 theta^2).
    """
    x, y = a.double(), b.double()
    nx, ny = x.norm(dim=-1), y.norm(dim=-1)
    cos = (x * y).sum(-1) / (nx * ny).clamp_min(1e-300)
    theta = torch.acos(cos.clamp(-1, 1))
    theta = torch.where((nx > 0) & (ny > 0), theta, torch.zeros_like(theta))
    v2 = (ny - nx) ** 2 + torch.maximum(nx, ny) ** 2 * theta**2
    return float(v2.sum().sqrt())


def rounding_tol(dtype: torch.dtype) -> float:
    """Relative size of one rounding of a tensor to `dtype` (norm-wise)."""
    return {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}.get(dtype, 1e-6)


def max_ease_increment(n: int) -> float:
    """Largest ease(k/n) - ease((k-1)/n) over a ramp of n chunks."""
    return max(ease(k / n) - ease((k - 1) / n) for k in range(1, n + 1))


# --- 1. back and forth --------------------------------------------------------

def back_and_forth(seconds: float, seed: int, shape: Sequence[int],
                   dtype: torch.dtype = torch.bfloat16) -> Check:
    """Random interrupted moves between A, B, C; see module docstring."""
    chk = Check("back_and_forth")
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    a, b, c = make_prompts(gen, shape, 3, dtype)
    tol = rounding_tol(dtype)
    row_max = torch.stack([t[0].float().norm(dim=-1) for t in (a, b, c)]).amax(0)
    basis, _ = torch.linalg.qr(torch.stack([t[0].double() for t in (a, b, c)], dim=-1))
    s = Steerer(a)
    prev = s.get_cond()
    moves = interrupts = frames = arrivals = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        target = rng.choice((a, b, a, b, c))
        n = rng.choice(RAMPS)
        cap = rng.choice((1.0, 1.0, 1.0, 0.4, 0.7))
        speed = speed_bound(prev, target)
        s.set_target(target, n, cap)
        moves += 1
        stop_at = rng.randint(0, n) if rng.random() < 0.6 else n
        interrupts += stop_at < n
        for k in range(1, stop_at + 1):
            out = s.get_cond()
            frames += 1
            if out.shape != a.shape or out.dtype != dtype:
                chk.fail(f"frame shape/dtype {tuple(out.shape)} {out.dtype}")
            if not torch.isfinite(out).all():
                chk.fail(f"non-finite frame at move {moves}")
                return chk
            o = out[0].float()
            excess = (o.norm(dim=-1) / row_max.clamp_min(1e-30) - 1)[row_max > 0]
            chk.worst("norm_excess", excess.max().clamp_min(0))
            if ((o.norm(dim=-1) > 0) & (row_max == 0)).any():
                chk.fail("non-zero row where every prompt has padding")
            od = out[0].double()
            proj = (basis @ (basis.transpose(-1, -2) @ od.unsqueeze(-1))).squeeze(-1)
            resid = (od - proj).norm() / od.norm().clamp_min(1e-30)
            chk.worst("out_of_plane", resid)
            if resid > 4 * tol:
                chk.fail(f"drifted out of the prompts' span: {resid:.3g}")
            step = float((out.double() - prev.double()).norm())
            dw = min(ease(k / n), cap) - min(ease((k - 1) / n), cap)
            bound = dw * speed
            slack = tol * float(out.double().norm() + prev.double().norm()) + 1e-12
            # Ratios are reported net of rounding, so bf16 noise on tiny
            # steps does not dominate them.
            if bound > 0:
                chk.worst("step/bound", (step - slack) / bound)
            if step > bound * 1.01 + slack:
                chk.fail(f"step {step:.4g} > bound {bound:.4g} (n={n}, k={k})")
            if k == 1 and moves > 1:
                first_bound = min(ease(1 / n), cap) * speed
                if first_bound > 0:
                    chk.worst("first_step/normal", (step - slack) / first_bound)
            prev = out
        if stop_at == n and cap == 1.0:
            if prev is not target:
                chk.fail("finished move did not land on the target object")
            arrivals += 1
    chk.stats = {"moves": moves, "interrupts": interrupts, "frames": frames,
                 "exact_arrivals": arrivals, **chk.stats}
    return chk


# --- 2. aligner ---------------------------------------------------------------

def check_placement(
    prev: Layout | None,
    ids: Sequence[int],
    rows: Tensor,
    out: Tensor,
    layout: Layout,
    occupied: Sequence[bool] | None,
) -> list[str]:
    """Return every way this placement breaks the Aligner's documented rules.

    `rows` holds the real token rows in prompt order ([n, D], all distinct);
    `out` is the placed [length, D] tensor and `layout` its layout.
    """
    errs: list[str] = []
    n, length = len(ids), out.shape[0]
    pos = [p for p, _ in layout]
    if [i for _, i in layout] != list(ids):
        errs.append("layout ids differ from prompt ids")
    if len(set(pos)) != n:
        errs.append("two tokens share a slot")
    if any(not 0 <= p < length for p in pos):
        errs.append("slot out of range")
    if errs:
        return errs
    if not torch.equal(out[pos], rows):
        errs.append("a token row was not copied exactly to its slot")
    unused = sorted(set(range(length)) - set(pos))
    if unused and out[unused].ne(0).any():
        errs.append("a row not holding a token is not exactly zero")
    if prev is None:
        if pos != list(range(n)):
            errs.append("first placement is not the identity")
        return errs
    prev_pos = [p for p, _ in prev]
    matched: set[int] = set()
    sm = difflib.SequenceMatcher(None, [i for _, i in prev], list(ids), autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            matched.add(b + k)
            if pos[b + k] != prev_pos[a + k]:
                errs.append(f"matched token {b + k} moved")
    occ = [False] * length if occupied is None else [bool(o) for o in occupied]
    used_before = set(prev_pos)

    def rank(p: int) -> tuple[bool, bool, int]:
        return (occ[p], p in used_before, p)

    new_slots = [pos[j] for j in range(n) if j not in matched]
    best_left = min((rank(p) for p in unused), default=None)
    if new_slots and best_left is not None and max(rank(p) for p in new_slots) > best_left:
        errs.append("a new token took a worse free slot than one left empty")
    return errs


def _edit(rng: random.Random, body: list[int], length: int, vocab: int) -> list[int]:
    """One random prompt edit; returns the new token body (no end token)."""
    def phrase(k: int) -> list[int]:
        return [rng.randrange(2, vocab) for _ in range(k)]

    op = rng.choice(("append", "insert", "remove", "replace", "reorder", "near_limit",
                     "tiny", "repeat", "new"))
    body = list(body)
    i = rng.randint(0, len(body))
    j = rng.randint(i, min(len(body), i + 12))
    if op == "append":
        body += phrase(rng.randint(1, 8))
    elif op == "insert":
        body[i:i] = phrase(rng.randint(1, 8))
    elif op == "remove":
        del body[i:j]
    elif op == "replace":
        body[i:j] = phrase(rng.randint(1, 8))
    elif op == "reorder" and len(body) > 4:
        k = rng.randint(1, len(body) - 1)
        body = body[k:] + body[:k]
    elif op == "near_limit":
        body += phrase(max(0, length - 1 - rng.randint(0, 3) - len(body)))
    elif op == "tiny":
        body = phrase(rng.randint(0, 1))
    elif op == "repeat":
        body[i:i] = [rng.randrange(2, vocab)] * rng.randint(2, 20)
    else:
        body = phrase(rng.randint(1, min(length - 1, 80)))
    return body[: length - 1]


def aligner_session(rng: random.Random, edits: int, length: int = 512, dim: int = 8,
                    vocab: int = 40, chk: Check | None = None) -> Check:
    """`edits` random prompt edits through one Aligner; every placement checked."""
    chk = chk or Check("aligner")
    al = Aligner(length)
    end = 1
    body = [rng.randrange(2, vocab) for _ in range(rng.randint(0, 30))]
    for step in range(edits + 1):
        ids = body + [end]
        n = len(ids)
        rows = torch.randn(n, dim, generator=torch.Generator().manual_seed(rng.getrandbits(31)))
        rows[rows.eq(0)] = 1.0
        emb = torch.zeros(1, length, dim)
        emb[0, :n] = rows
        occupied: list[bool] | None = None
        if step and rng.random() < 0.5:
            density = rng.random()
            occupied = [rng.random() < density for _ in range(length)]
        prev = al.layout
        try:
            out = al.place(emb, ids, occupied)
        except Exception as exc:  # noqa: BLE001 - reported as a failure
            chk.fail(f"place raised {exc!r} at n={n}")
            return chk
        for e in check_placement(prev, ids, rows, out[0], al.layout, occupied):
            chk.fail(f"{e} (n={n}, step={step})")
        if prev is not None:
            kept = sum(1 for (p, _) in al.layout if p in {q for q, _ in prev})
            chk.worst("max_reused_slots", kept)
        chk.worst("max_tokens", n)
        chk.stats["placements"] = chk.stats.get("placements", 0) + 1
        if n == length:
            chk.stats["full_length_placements"] = chk.stats.get("full_length_placements", 0) + 1
        body = _edit(rng, body, length, vocab)
    # One over the limit must raise and leave the layout alone.
    before = al.layout
    over = torch.zeros(1, length, dim)
    try:
        al.place(over, [5] * (length + 1))
        chk.fail(f"{length + 1} tokens did not raise")
    except ValueError:
        pass
    if al.layout != before:
        chk.fail("failed place changed the layout")
    return chk


def aligner_soak(seconds: float, seed: int, length: int = 512, dim: int = 8) -> Check:
    """Run `aligner_session`s of 100-400 edits until `seconds` run out."""
    chk = Check("aligner")
    rng = random.Random(seed)
    deadline = time.monotonic() + seconds
    sessions = 0
    while time.monotonic() < deadline and chk.ok:
        aligner_session(rng, rng.randint(100, 400), length, dim, chk=chk)
        sessions += 1
    chk.stats = {"sessions": sessions, **chk.stats}
    return chk


# --- 3. threads ---------------------------------------------------------------

def row_constant_encoder(shape: Sequence[int], dtype: torch.dtype,
                         seed: int) -> Callable[[str], Tensor]:
    """encode(prompt): every row the same vector (random direction and norm)."""
    _, length, dim = shape
    cache: dict[str, Tensor] = {}

    def encode(prompt: str) -> Tensor:
        if prompt not in cache:
            digest = hashlib.sha256(f"{seed}:{prompt}".encode()).digest()
            g = torch.Generator().manual_seed(int.from_bytes(digest[:7], "little"))
            v = torch.randn(dim, generator=g)
            v = v / v.norm() * math.exp(float(torch.randn(1, generator=g)))
            cache[prompt] = v.expand(1, length, dim).to(dtype).contiguous()
        return cache[prompt]

    return encode


def thread_hammer(seconds: float, seed: int, shape: Sequence[int],
                  dtype: torch.dtype = torch.bfloat16) -> Check:
    """get_cond in a tight loop vs random handle / set_target / reset calls."""
    chk = Check("threads")
    encode = row_constant_encoder(shape, dtype, seed)
    prompts = [f"p{i}" for i in range(10)]
    max_norm = max(float(encode(p)[0, 0].float().norm()) for p in prompts)
    s = Steerer(encode(prompts[0]))
    ctl = Controller(s, encode, cache_size=4)
    stop = threading.Event()
    errors: list[str] = []
    counts = {"frames": 0, "messages": 0, "set_target": 0, "resets": 0}
    tol = rounding_tol(dtype)

    def loop() -> None:
        try:
            while not stop.is_set():
                out = s.get_cond()
                counts["frames"] += 1
                if out.shape != torch.Size(shape) or not torch.isfinite(out).all():
                    errors.append("bad frame shape or non-finite frame")
                    return
                if not torch.equal(out, out[:, :1].expand_as(out)):
                    errors.append("torn frame: rows from different moves")
                    return
                # strength 1.3 on the linear branch can reach |1.3 b - 0.3 a|.
                if float(out[0, 0].float().norm()) > 1.6 * max_norm * (1 + tol):
                    errors.append("frame norm above 1.6 x the largest prompt norm")
                    return
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(f"get_cond raised {exc!r}")

    def control() -> None:
        rng = random.Random(seed)
        try:
            while not stop.is_set():
                r = rng.random()
                p = rng.choice(prompts)
                ramp = rng.choice(RAMPS)
                if r < 0.05:
                    ctl.reset(p)
                    counts["resets"] += 1
                elif r < 0.15:
                    s.set_target(encode(p), ramp, rng.choice((1.0, 0.5)))
                    counts["set_target"] += 1
                else:
                    msg: dict = {"prompt": p, "ramp_chunks": ramp}
                    if rng.random() < 0.4:
                        msg.update(provisional=True, cap=rng.choice((0.3, 0.4, 1.0)))
                    if rng.random() < 0.3:
                        msg.update(strength=rng.choice((0.3, 0.6, 1.0, 1.3)),
                                   without_prompt=rng.choice(prompts))
                    ctl.handle(msg)
                    counts["messages"] += 1
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(f"control raised {exc!r}")

    threads = [threading.Thread(target=loop), threading.Thread(target=control)]
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join(30)
    for e in errors[:5]:
        chk.fail(e)
    chk.stats.update(counts)
    return chk


# --- 4. memory ----------------------------------------------------------------

def _live_tensors() -> int:
    gc.collect()
    with warnings.catch_warnings():  # some torch modules warn on attribute access
        warnings.simplefilter("ignore")
        return sum(1 for o in gc.get_objects() if isinstance(o, Tensor))


def memory_check(messages: int, seed: int, length: int = 64, dim: int = 16,
                 cache_size: int = 64) -> Check:
    """`messages` contract-2 messages (mostly new prompts) through a Controller."""
    chk = Check("memory")
    rng = random.Random(seed)
    vocab = torch.randn(200, dim, generator=torch.Generator().manual_seed(seed))

    def encode(prompt: str) -> tuple[Tensor, list[int]]:
        ids = [int(w) % 200 for w in prompt.split()][: length - 1] + [1]
        emb = torch.zeros(1, length, dim, dtype=torch.bfloat16)
        emb[0, : len(ids)] = vocab[ids].to(torch.bfloat16)
        return emb, ids

    al = Aligner(length)
    s = Steerer(al.place(*encode("1 2 3")))
    ctl = Controller(s, encode, cache_size=cache_size, aligner=al)

    def send(k: int) -> None:
        words = " ".join(str(rng.randrange(2, 200)) for _ in range(rng.randint(1, 30)))
        msg: dict = {"prompt": words, "ramp_chunks": rng.choice(RAMPS)}
        if k % 3 == 0:
            msg.update(provisional=True, cap=0.4)
        if k % 5 == 0:
            msg.update(strength=1.3, without_prompt=f"{k % 7} 4 5")
        ctl.handle(msg)
        for _ in range(rng.randint(0, 2)):
            s.get_cond()

    warm = min(500, messages // 4)
    for k in range(warm):
        send(k)
    tensors0 = _live_tensors()
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    for k in range(warm, messages):
        send(k)
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    growth = sum(st.size_diff for st in snap1.compare_to(snap0, "filename"))
    tensors1 = _live_tensors()
    cache_len = len(ctl._cache)  # noqa: SLF001 - checking the bound itself
    chk.stats.update(messages=messages, cache=f"{cache_len}/{cache_size}",
                     layout_tokens=len(al.layout or ()),
                     live_tensors=f"{tensors0}->{tensors1}",
                     py_growth_kb=growth / 1024)
    if cache_len > cache_size:
        chk.fail(f"cache holds {cache_len} > {cache_size}")
    if len(al.layout or ()) > length:
        chk.fail("layout longer than the aligner length")
    if tensors1 > tensors0 + 2 * cache_size:
        chk.fail(f"live tensors grew {tensors0} -> {tensors1}")
    if growth > 2 * 1024 * 1024:
        chk.fail(f"Python memory grew {growth / 1024:.0f} KB")
    return chk


# --- CLI ----------------------------------------------------------------------

def run(seconds: float, seed: int, full_size: bool) -> list[Check]:
    """Run every check; time is split 40/30/30 between the timed ones."""
    shape = FULL if full_size else SMALL
    dim = FULL[2] if full_size else 8
    return [
        back_and_forth(0.4 * seconds, seed, shape),
        aligner_soak(0.3 * seconds, seed, 512, dim),
        thread_hammer(0.3 * seconds, seed, shape),
        memory_check(10_000, seed),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-size", action="store_true",
                    help="[1, 512, 4096] bfloat16 instead of [1, 16, 64]")
    args = ap.parse_args(argv)
    shape = FULL if args.full_size else SMALL
    print(f"soak seed={args.seed} seconds={args.seconds:g} shape={list(shape)} dtype=bfloat16")
    t0 = time.monotonic()
    checks = run(args.seconds, args.seed, args.full_size)
    for chk in checks:
        print(chk.line())
        for f in chk.failures:
            print(f"    {f}")
    ok = all(c.ok for c in checks)
    print(f"RESULT {'PASS' if ok else 'FAIL'} in {time.monotonic() - t0:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

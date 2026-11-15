"""Property tests for the steering maths (issue #37): seeded, fast, deterministic.

Most cases use [1, 16, 64]; one full-size [1, 512, 4096] bfloat16 case.
Longer randomized runs of the same checks: `python -m steering.eval.soak`.
"""

import math
import random

import pytest
import torch
from torch import Tensor

from steering import Aligner, Controller, Steerer, blend, ease
from steering.eval.soak import (
    aligner_session,
    back_and_forth,
    check_placement,
    make_prompts,
    max_ease_increment,
    memory_check,
    rounding_tol,
    speed_bound,
    thread_hammer,
)

SHAPE = (1, 16, 64)
RAMPS = (1, 2, 3, 4, 12)
EPS = 1e-4  # blend's default eps


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _unit(v: Tensor) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True)


def _pair_at_angles(theta: Tensor, nx: Tensor, ny: Tensor, dim: int = 64,
                    seed: int = 0) -> tuple[Tensor, Tensor]:
    """float64 rows a_i, b_i with exact angle theta_i and norms nx_i, ny_i."""
    g = _gen(seed)
    u = _unit(torch.randn(len(theta), dim, generator=g, dtype=torch.float64))
    p = torch.randn(len(theta), dim, generator=g, dtype=torch.float64)
    p = _unit(p - (p * u).sum(-1, keepdim=True) * u)
    th = theta.double().unsqueeze(-1)
    v = u * torch.cos(th) + p * torch.sin(th)
    return u * nx.double().unsqueeze(-1), v * ny.double().unsqueeze(-1)


def _angle(x: Tensor, y: Tensor) -> Tensor:
    """Per-token angle, accurate at small angles (atan2 of sin and cos)."""
    ux, uy = _unit(x.double()), _unit(y.double())
    cos = (ux * uy).sum(-1, keepdim=True)
    return torch.atan2((uy - cos * ux).norm(dim=-1), cos.squeeze(-1))


# --- blend: finiteness ------------------------------------------------------

# bf16 at both extremes of its range (the same exponent range as float32), float32 at unit scale.
@pytest.mark.parametrize("dtype, scale", [(torch.bfloat16, 1e-30), (torch.float32, 1.0), (torch.bfloat16, 1e30)])
def test_blend_finite_for_any_scale(dtype: torch.dtype, scale: float) -> None:
    g = _gen(1)
    a = (torch.randn(SHAPE, generator=g) * scale).to(dtype)
    b = (torch.randn(SHAPE, generator=g) * scale).to(dtype)
    b[0, :3] = 0                      # padding rows
    b[0, 3] = a[0, 3] * -1            # exactly opposite
    b[0, 4] = a[0, 4] * 2             # collinear
    a[0, 5] = 0
    for w in (0.0, 1e-6, 0.1, 0.5, 0.9, 1.0, 1.15, 1.3):
        out = blend(a, b, w)
        assert out.dtype == dtype and out.shape == a.shape
        assert torch.isfinite(out).all(), (scale, w)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_blend_finite_near_switch_thresholds(dtype: torch.dtype) -> None:
    # Angles straddling sin = eps at both ends, norms straddling eps.
    d = torch.tensor([0.5, 0.99, 1.0, 1.01, 2.0]) * EPS
    theta = torch.cat([d, math.pi - d, torch.full((5,), 0.3)])
    nx = torch.cat([torch.ones(10), d * 1.0])
    a, b = _pair_at_angles(theta, nx, torch.ones(15))
    for w in torch.linspace(0, 1.3, 27).tolist():
        assert torch.isfinite(blend(a.to(dtype), b.to(dtype), w)).all()


# --- blend: norm bounds -----------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_blend_spherical_norm_within_endpoint_norms(dtype: torch.dtype) -> None:
    g = _gen(2)
    n = 400
    theta = torch.rand(n, generator=g, dtype=torch.float64) * math.radians(170) + 1e-3
    nx = torch.exp(2 * torch.randn(n, generator=g, dtype=torch.float64))
    ny = torch.exp(2 * torch.randn(n, generator=g, dtype=torch.float64))
    a, b = _pair_at_angles(theta, nx, ny)
    a, b = a.to(dtype), b.to(dtype)
    lo = torch.minimum(a.double().norm(dim=-1), b.double().norm(dim=-1))
    hi = torch.maximum(a.double().norm(dim=-1), b.double().norm(dim=-1))
    tol = 2 * rounding_tol(dtype) if dtype != torch.float32 else 1e-5
    worst = 0.0
    for w in torch.linspace(0, 1, 41).tolist():
        m = blend(a, b, w).double().norm(dim=-1)
        assert (m <= hi * (1 + tol)).all() and (m >= lo * (1 - tol)).all(), w
        worst = max(worst, float((m / hi - 1).max()), float((1 - m / lo).max()))
    assert worst <= tol


def test_blend_linear_branch_norm_bound() -> None:
    # Linear branch: |a + w (b - a)| <= (1 - w)|a| + w|b| <= max, and
    # >= |(1 - w)|a| - w|b||; it can pass through zero (no lower bound).
    g = _gen(3)
    v = torch.randn(6, 64, generator=g, dtype=torch.float64)
    a = torch.cat([v, torch.zeros(2, 64, dtype=torch.float64)])
    b = torch.cat([-v * torch.tensor([[1.0], [2.0], [0.5], [1.0], [3.0], [1.0]],
                                     dtype=torch.float64), v[:2]])
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    for w in torch.linspace(0, 1, 21).tolist():
        m = blend(a, b, w).norm(dim=-1)
        assert (m <= (1 - w) * na + w * nb + 1e-12).all()
        assert (m >= ((1 - w) * na - w * nb).abs() - 1e-12).all()
    assert blend(a[:1], b[:1], 0.5).norm() < 1e-12  # equal and opposite: through zero


# --- blend: progress in w and continuity --------------------------------------

def test_blend_angle_and_norm_move_linearly_in_w() -> None:
    # On the spherical branch the angle from a is exactly w * theta and the
    # norm exactly the lerp of norms, so a step in w is the same fraction
    # of the angle and of the norm change, for every token.
    g = _gen(4)
    theta = torch.rand(64, generator=g, dtype=torch.float64) * math.radians(110) + 0.01
    nx = torch.exp(torch.randn(64, generator=g, dtype=torch.float64))
    ny = torch.exp(torch.randn(64, generator=g, dtype=torch.float64))
    a, b = _pair_at_angles(theta, nx, ny)
    for w in (0.1, 0.25, 0.5, 0.8, 1.0, 1.2, 1.3):
        out = blend(a, b, w)
        expected = (nx + w * (ny - nx)).clamp_min(0)
        assert torch.allclose(out.norm(dim=-1), expected, atol=1e-9)
        alive = expected > 0  # extrapolation may clamp a token to zero
        assert torch.allclose(_angle(a, out)[alive], (w * theta)[alive], atol=1e-9)


def test_blend_lipschitz_in_w() -> None:
    a, b = make_prompts(_gen(5), SHAPE, 2, torch.float64)
    bound = speed_bound(a, b)
    ws = torch.linspace(0, 1.3, 261).tolist()
    prev = blend(a, b, ws[0])
    for w0, w in zip(ws, ws[1:]):
        cur = blend(a, b, w)
        assert float((cur - prev).norm()) <= (w - w0) * bound * (1 + 1e-9)
        prev = cur


def test_blend_continuous_across_sin_threshold_near_zero_angle() -> None:
    # Just below sin = eps the token goes linear, just above it slerps. The
    # two differ in direction by about w(1 - w) theta |ny - nx| / |out| (lerp
    # weights the direction by norm, slerp does not): first order in theta,
    # so the jump is below eps * |ny - nx|, far under bf16 rounding (2^-9).
    d = torch.tensor([0.999, 1.001]) * EPS
    a, b = _pair_at_angles(d, torch.tensor([1.0, 1.0]), torch.tensor([3.0, 3.0]))
    for w in (0.1, 0.5, 0.9, 1.3):
        out = blend(a, b, w)
        lin = a + w * (b - a)
        assert torch.equal(out[0], lin[0])                     # linear side
        assert float((out[1] - lin[1]).norm()) < EPS * 2.0, w  # spherical side


def test_blend_zero_row_to_token_and_back_is_exactly_linear() -> None:
    g = _gen(6)
    y = torch.randn(4, 64, generator=g)
    z = torch.zeros(4, 64)
    for w in torch.linspace(0, 1.3, 14).tolist():
        assert torch.allclose(blend(z, y, w), w * y, atol=1e-6)
        assert torch.allclose(blend(y, z, w), (1 - w) * y, atol=1e-6)
    assert torch.equal(blend(z, y, 1.0), y) and torch.equal(blend(y, z, 1.0), z)


@pytest.mark.xfail(strict=True, reason=(
    "Inherent to slerp: near theta = pi the slerp midpoint has full norm along "
    "an arbitrary perpendicular, the linear fallback passes through ~0. The "
    "fallback is documented; real token angles stay below ~110 degrees."))
def test_blend_continuous_across_sin_threshold_near_pi() -> None:
    d = torch.tensor([0.99, 1.01]) * EPS
    a, b = _pair_at_angles(math.pi - d, torch.ones(2), torch.ones(2))
    out = blend(a.float(), b.float(), 0.5).norm(dim=-1)
    assert abs(float(out[0] - out[1])) < 0.01


@pytest.mark.xfail(strict=True, reason=(
    "Defect, left for review: a row with a small but non-zero norm (above "
    "eps) slerps its direction by w regardless of norms, so a token that has "
    "nearly faded out reappears at up to ~0.36 |b| on its way to b. Fixing it "
    "changes the documented 'direction slerped by w' rule."))
def test_blend_nearly_faded_row_does_not_resurrect() -> None:
    old, new = _pair_at_angles(torch.tensor([math.pi / 2]), torch.tensor([1e-3]),
                               torch.tensor([1.0]))
    out = blend(old.float(), new.float(), 0.5)[0].double()
    u_new = new[0] / new[0].norm()
    off_target = (out - (out @ u_new) * u_new).norm()
    # Linear gives 0.5 * 1e-3 here; anything near that is fine.
    assert off_target < 0.01


# --- blend: extrapolation -----------------------------------------------------

def test_blend_extrapolation_spherical_magnitude_clamps_to_zero() -> None:
    # 1.3 |b| - 0.3 |a| < 0 when |b| < 0.23 |a|: the token goes to exactly zero
    # (never flips), and stays finite.
    theta = torch.full((4,), 0.4)
    nx = torch.ones(4)
    ny = torch.tensor([0.1, 0.2, 0.25, 0.5])
    a, b = _pair_at_angles(theta, nx, ny)
    for dtype in (torch.float32, torch.bfloat16):
        out = blend(a.to(dtype), b.to(dtype), 1.3).double()
        assert torch.isfinite(out).all()
        assert out[:2].abs().max() == 0
        assert (_angle(a[2:], out[2:]) - 1.3 * 0.4).abs().max() < 2e-2
        assert ((out * a).sum(-1)[2:] > 0).all()


@pytest.mark.xfail(strict=True, reason=(
    "Defect, left for review: at w > 1 the linear fallback a + w(b - a) "
    "flips a token whose target row is zero (or below eps) to -0.3 a, while "
    "the spherical branch clamps the same case to zero. Happens with "
    "strength 1.3 when without_prompt has a token the prompt lacks. Fixing it "
    "changes the documented linear rule."))
def test_blend_extrapolation_linear_branch_never_flips() -> None:
    a = torch.randn(3, 64, generator=_gen(8))
    out = blend(a, torch.zeros_like(a), 1.3)
    assert ((out * a).sum(-1) >= 0).all()


# --- Steerer: pacing ----------------------------------------------------------

@pytest.mark.parametrize("n", RAMPS)
def test_ramp_steps_follow_min_jerk_increments(n: int) -> None:
    # In w-space (angle and norm per token) every step is exactly the ease
    # increment; in Euclidean distance each step is at most the increment
    # times the path speed bound.
    g = _gen(10 + n)
    theta = torch.rand(16, generator=g, dtype=torch.float64) * math.radians(110) + 0.05
    nx = torch.exp(torch.randn(16, generator=g, dtype=torch.float64))
    ny = torch.exp(torch.randn(16, generator=g, dtype=torch.float64))
    a, b = _pair_at_angles(theta, nx, ny)
    s = Steerer(a)
    s.set_target(b, n)
    bound = speed_bound(a, b)
    prev, worst = a, 0.0
    for k in range(1, n + 1):
        out = s.get_cond()
        dw = ease(k / n) - ease((k - 1) / n)
        d_angle = _angle(a, out) - _angle(a, prev)
        assert torch.allclose(d_angle, dw * theta, atol=1e-9)
        assert torch.allclose(out.norm(dim=-1) - prev.norm(dim=-1), dw * (ny - nx), atol=1e-9)
        step = float((out - prev).norm())
        assert step <= dw * bound * (1 + 1e-9)
        worst = max(worst, step / bound)
        prev = out
    assert prev is b
    assert worst <= max_ease_increment(n) * (1 + 1e-9)


# A one-chunk move cut by a ramp, a ramp cut by a jump (m = 1), and a slow move cut by a slow move.
@pytest.mark.parametrize("n, m", [(1, 4), (4, 1), (12, 12)])
def test_interrupt_at_every_point_is_continuous_and_lands_exactly(n: int, m: int) -> None:
    a, b, c = make_prompts(_gen(20), SHAPE, 3, torch.float32)
    for k_stop in range(n + 1):
        s = Steerer(a)
        s.set_target(b, n)
        prev = a
        for _ in range(k_stop):
            prev = s.get_cond()
        s.set_target(c, m)
        first = s.get_cond()
        # The new move starts from the frame on screen: its first step is the
        # normal first step of an m-chunk move from there.
        assert torch.allclose(first, blend(prev, c, ease(1 / m)) if m > 1 else c, atol=1e-6)
        step = float((first - prev).norm())
        assert step <= ease(1 / m) * speed_bound(prev, c) * (1 + 1e-4) + 1e-6
        for _ in range(m - 1):
            first = s.get_cond()
        assert first is c


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_repeated_interrupted_back_and_forth_does_not_drift(dtype: torch.dtype) -> None:
    chk = back_and_forth(0.4, seed=3, shape=SHAPE, dtype=dtype)
    assert chk.ok, chk.failures
    assert chk.stats["moves"] > 100 and chk.stats["exact_arrivals"] > 10
    assert chk.stats["out_of_plane"] <= 4 * rounding_tol(dtype)
    assert chk.stats["norm_excess"] <= 2 * rounding_tol(dtype)


@pytest.mark.parametrize("n", (4, 12))
def test_bf16_constant_early_interrupts_still_converge(n: int) -> None:
    # Restarting a move toward the same target at every chunk advances
    # ease(1/n) of the remaining distance each time. In bf16 that step is
    # below the rounding of the frame (ease(1/12) = 0.5%), so a restart from
    # the rounded frame never moves; the steerer must keep its position at
    # full precision. float32 is the reference.
    a, b = make_prompts(_gen(30), SHAPE, 2, torch.bfloat16)
    remaining = {}
    for dtype in (torch.float32, torch.bfloat16):
        s = Steerer(a.to(dtype))
        for _ in range(300):
            s.set_target(b.to(dtype), n)
            out = s.get_cond()
        remaining[dtype] = float((out.float() - b.float()).norm() / (a.float() - b.float()).norm())
    assert remaining[torch.bfloat16] <= remaining[torch.float32] + 0.02, remaining


# --- Controller sequences -----------------------------------------------------

class DictEncode:
    """encode from a fixed dict of tensors."""

    def __init__(self, table: dict[str, Tensor]) -> None:
        self.table = table

    def __call__(self, prompt: str) -> Tensor:
        return self.table[prompt]


def _drain(s: Steerer, n: int) -> Tensor:
    out = s.current
    for _ in range(n):
        out = s.get_cond()
    return out


def test_capped_confirm_then_other_final_then_reverse() -> None:
    base, p, q = make_prompts(_gen(40), SHAPE, 3, torch.float32)
    s = Steerer(base)
    ctl = Controller(s, DictEncode({"base": base, "p": p, "q": q}))
    ctl.reset("base")
    assert ctl.handle({"prompt": "p", "ramp_chunks": 4, "provisional": True, "cap": 0.4})["kind"] == "provisional"
    held = _drain(s, 4)
    assert torch.allclose(held, blend(base, p, 0.4), atol=1e-6)
    assert _drain(s, 3) is held                        # holds at the cap
    assert ctl.handle({"prompt": "p", "ramp_chunks": 2})["kind"] == "confirm"
    assert _drain(s, 2) is p
    assert ctl.handle({"prompt": "q", "ramp_chunks": 3})["kind"] == "final"
    assert _drain(s, 3) is q
    assert ctl.handle({"prompt": "p", "ramp_chunks": 3})["kind"] == "final"
    assert _drain(s, 3) is p
    # Provisional for q, then the final goes back to p (cancel the guess).
    ctl.handle({"prompt": "q", "ramp_chunks": 4, "provisional": True, "cap": 0.4})
    _drain(s, 2)
    assert ctl.handle({"prompt": "p", "ramp_chunks": 4})["kind"] == "final"
    assert _drain(s, 4) is p


def test_strength_sequence_against_same_without_prompt() -> None:
    without, full = make_prompts(_gen(41), SHAPE, 2, torch.bfloat16)
    # Half the rows shrink to under 0.23 x their old norm: clamp at 1.3.
    full = full.clone()
    full[0, :8] = (without[0, :8].float() * 0.1 + full[0, :8].float() * 0.01).to(torch.bfloat16)
    s = Steerer(without)
    ctl = Controller(s, DictEncode({"w": without, "f": full}))
    for strength in (0.3, 0.6, 1.0, 1.3, 0.6):
        r = ctl.handle({"prompt": "f", "ramp_chunks": 4, "strength": strength,
                        "without_prompt": "w"})
        assert r["applied"] and r["kind"] == "final"
        out = _drain(s, 4)
        assert torch.equal(out, blend(without, full, strength))
        assert torch.isfinite(out).all()
        if strength == 1.0:
            assert torch.equal(out, full)
        if strength == 1.3:
            shrunk = without[0, :8].float().norm(dim=-1) * 0.23 > full[0, :8].float().norm(dim=-1)
            assert out[0, :8][shrunk].abs().max() == 0


# --- Aligner long session -----------------------------------------------------

@pytest.mark.parametrize("seed", range(3))
def test_aligner_long_session_invariants(seed: int) -> None:
    chk = aligner_session(random.Random(seed), edits=150, length=512, dim=4)
    assert chk.ok, chk.failures
    assert chk.stats["max_tokens"] >= 500


def test_aligner_exactly_full_and_one_over() -> None:
    length = 32
    al = Aligner(length)
    emb = torch.randn(1, length, 4, generator=_gen(50))
    al.place(emb, list(range(length)))
    # A completely different full prompt: every slot is reused, none twice.
    out = al.place(emb.flip(1), list(range(100, 100 + length)))
    assert sorted(p for p, _ in al.layout) == list(range(length))
    assert torch.equal(out.abs().sum(-1).ne(0), torch.ones(1, length, dtype=torch.bool))
    before = al.layout
    with pytest.raises(ValueError):
        al.place(emb, list(range(length + 1)))
    with pytest.raises(ValueError):
        al.place(emb, [])
    assert al.layout == before


def test_aligner_fragmented_slots_and_duplicate_ids() -> None:
    length = 40
    al = Aligner(length)
    ids = [7] * 20 + [1]
    rows = torch.randn(21, 4, generator=_gen(51))
    emb = torch.zeros(1, length, 4)
    emb[0, :21] = rows
    al.place(emb, ids)
    occupied = [p % 2 == 0 for p in range(length)]  # every other slot busy
    ids2 = [7] * 10 + [9] * 15 + [7] * 5 + [1]
    rows2 = torch.randn(31, 4, generator=_gen(52))
    emb2 = torch.zeros(1, length, 4)
    emb2[0, :31] = rows2
    prev = al.layout
    out, layout = al.prepare(emb2, ids2, occupied)
    assert check_placement(prev, ids2, rows2, out[0], layout, occupied) == []


# --- threads and memory -------------------------------------------------------

def test_threads_never_tear_or_fail() -> None:
    chk = thread_hammer(0.5, seed=0, shape=SHAPE)
    assert chk.ok, chk.failures
    assert chk.stats["frames"] > 100 and chk.stats["messages"] > 100


def test_memory_bounded_over_many_messages() -> None:
    chk = memory_check(2000, seed=0, cache_size=16)
    assert chk.ok, chk.failures


# --- full size ----------------------------------------------------------------

def test_full_size_bf16_ramp_interrupt_and_bounds() -> None:
    shape = (1, 512, 4096)
    a, b, c = make_prompts(_gen(60), shape, 3, torch.bfloat16)
    tol = rounding_tol(torch.bfloat16)
    hi = torch.maximum(a[0].float().norm(dim=-1), b[0].float().norm(dim=-1))
    for w in (0.25, 0.5, 0.75, 1.3):
        out = blend(a, b, w)
        assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
        if w <= 1:
            assert (out[0].float().norm(dim=-1) <= hi * (1 + 2 * tol)).all()
    s = Steerer(a)
    s.set_target(b, 4)
    prev = s.get_cond()
    s.set_target(c, 4)                              # interrupt after one chunk
    first = s.get_cond()
    step = float((first.double() - prev.double()).norm())
    assert step <= ease(0.25) * speed_bound(prev, c) * 1.01 + tol * float(prev.double().norm()) * 2
    assert _drain(s, 3) is c

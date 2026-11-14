import math

import pytest
import torch

from steering import apply_strength, blend

T, D = 8, 32


def _pair(seed: int = 0, shape=(T, D), dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(*shape, generator=g, dtype=torch.float64).to(dtype)
    b = torch.randn(*shape, generator=g, dtype=torch.float64).to(dtype)
    return a, b


def _angle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.double(), y.double()
    cos = (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1))
    return torch.acos(cos.clamp(-1, 1))


# --- endpoints and geometry -------------------------------------------------

@pytest.mark.parametrize("shape", [(T, D), (2, T, D)])
def test_endpoints(shape):
    a, b = _pair(shape=shape)
    assert torch.allclose(blend(a, b, 0.0), a, atol=1e-6)
    assert torch.allclose(blend(a, b, 1.0), b, atol=1e-6)


@pytest.mark.parametrize("shape", [(T, D), (2, T, D)])
def test_output_shape(shape):
    a, b = _pair(shape=shape)
    assert blend(a, b, 0.4).shape == a.shape


def test_midpoint_norm_is_mean_of_norms():
    a, b = _pair(seed=1)
    m = blend(a, b, 0.5)
    expected = (a.norm(dim=-1) + b.norm(dim=-1)) / 2
    assert torch.allclose(m.norm(dim=-1), expected, atol=1e-5)


def test_midpoint_direction_equidistant_by_angle():
    a, b = _pair(seed=2)
    m = blend(a, b, 0.5)
    assert torch.allclose(_angle(m, a), _angle(m, b), atol=1e-5)
    assert torch.allclose(_angle(m, a), _angle(a, b) / 2, atol=1e-5)


def test_slerp_differs_from_lerp_for_generic_pair():
    a, b = _pair(seed=3)
    lerp = a + 0.5 * (b - a)
    assert not torch.allclose(blend(a, b, 0.5), lerp, atol=1e-3)


# --- fallback cases ---------------------------------------------------------

def _special_pairs():
    g = torch.Generator().manual_seed(4)
    v = torch.randn(T, D, generator=g)
    return {
        "collinear": (v, 3.0 * v),
        "opposite": (v, -2.0 * v),
        "zero_a": (torch.zeros(T, D), v),
        "zero_b": (v, torch.zeros(T, D)),
        "both_zero": (torch.zeros(T, D), torch.zeros(T, D)),
        "identical": (v, v.clone()),
    }


@pytest.mark.parametrize("name", list(_special_pairs()))
def test_fallback_cases_finite_and_exact_ends(name):
    a, b = _special_pairs()[name]
    for w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.3]:
        out = blend(a, b, w)
        assert torch.isfinite(out).all(), (name, w)
    assert torch.allclose(blend(a, b, 0.0), a, atol=1e-6)
    assert torch.allclose(blend(a, b, 1.0), b, atol=1e-6)


@pytest.mark.parametrize("name", list(_special_pairs()))
def test_fallback_cases_are_linear(name):
    a, b = _special_pairs()[name]
    out = blend(a, b, 0.3)
    assert torch.allclose(out, a + 0.3 * (b - a), atol=1e-5)


def test_mixed_batch_selects_fallback_per_token():
    a, b = _pair(seed=5)
    a[0] = 0.0                      # zero token
    b[1] = 2.0 * a[1]               # collinear token
    b[2] = -a[2]                    # opposite token
    out = blend(a, b, 0.5)
    assert torch.isfinite(out).all()
    lerp = a + 0.5 * (b - a)
    assert torch.allclose(out[:3], lerp[:3], atol=1e-5)
    # Remaining tokens are spherical: norm is the mean of norms, not the lerp norm.
    expected = (a[3:].norm(dim=-1) + b[3:].norm(dim=-1)) / 2
    assert torch.allclose(out[3:].norm(dim=-1), expected, atol=1e-5)
    assert torch.allclose(_angle(out[3:], a[3:]), _angle(out[3:], b[3:]), atol=1e-5)
    assert torch.allclose(blend(a, b, 0.0), a, atol=1e-6)
    assert torch.allclose(blend(a, b, 1.0), b, atol=1e-6)


# --- dtypes, devices, purity ------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_dtype_preserved(dtype):
    a, b = _pair(dtype=dtype)
    out = blend(a, b, 0.5)
    assert out.dtype == dtype
    assert out.device == a.device
    assert torch.isfinite(out).all()
    # Reference is computed in float32, so compare at the coarser precision.
    tol = 1e-5 if dtype == torch.float64 else 1e-2
    ref = blend(a.float(), b.float(), 0.5).to(dtype)
    assert torch.allclose(out.double(), ref.double(), atol=tol * 10, rtol=tol)


def test_float64_midpoint_precise():
    a, b = _pair(seed=6, dtype=torch.float64)
    m = blend(a, b, 0.5)
    expected = (a.norm(dim=-1) + b.norm(dim=-1)) / 2
    assert torch.allclose(m.norm(dim=-1), expected, atol=1e-10)


def test_inputs_not_modified():
    a, b = _pair(seed=7)
    a_copy, b_copy = a.clone(), b.clone()
    for w in [0.0, 0.5, 1.0, 1.3]:
        blend(a, b, w)
    assert torch.equal(a, a_copy)
    assert torch.equal(b, b_copy)


def test_output_is_not_an_input():
    a, b = _pair(seed=8)
    assert blend(a, b, 0.0) is not a
    assert blend(a, b, 1.0) is not b


# --- errors -----------------------------------------------------------------

def test_shape_mismatch_raises():
    a, _ = _pair()
    with pytest.raises(ValueError):
        blend(a, torch.zeros(T, D + 1), 0.5)
    with pytest.raises(ValueError):
        blend(a, torch.zeros(1, T, D), 0.5)


@pytest.mark.parametrize("w", [-0.01, -1.0, 1.31, 2.0, math.nan])
def test_w_out_of_range_raises(w):
    a, b = _pair()
    with pytest.raises(ValueError):
        blend(a, b, w)


def test_extrapolation_allowed_up_to_1_3():
    a, b = _pair(seed=9)
    out = blend(a, b, 1.3)
    assert torch.isfinite(out).all()
    # Goes past b: further from a than b is, by angle.
    assert (_angle(out, a) > _angle(b, a)).all()


# --- apply_strength ---------------------------------------------------------

def test_apply_strength_endpoints():
    a, b = _pair(seed=10)
    assert torch.allclose(apply_strength(a, b, 0.0), a, atol=1e-6)
    assert torch.allclose(apply_strength(a, b, 1.0), b, atol=1e-6)


def test_apply_strength_partial_lies_between_by_angle():
    a, b = _pair(seed=11)
    out = apply_strength(a, b, 0.3)
    total = _angle(a, b)
    to_a, to_b = _angle(out, a), _angle(out, b)
    assert (to_a > 0).all() and (to_a < total).all()
    assert (to_b > 0).all() and (to_b < total).all()
    assert torch.allclose(to_a, 0.3 * total, atol=1e-5)


def test_apply_strength_extrapolates_without_error():
    a, b = _pair(seed=12)
    out = apply_strength(a, b, 1.2)
    assert torch.isfinite(out).all()


def test_apply_strength_errors():
    a, b = _pair()
    with pytest.raises(ValueError):
        apply_strength(a, b, -0.1)
    with pytest.raises(ValueError):
        apply_strength(a, b, 1.5)
    with pytest.raises(ValueError):
        apply_strength(a, torch.zeros(T, D + 1), 0.5)

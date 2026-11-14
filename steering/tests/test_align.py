import random

import pytest
import torch

from steering import Aligner, blend

L, D = 16, 8
TABLE = torch.randn(100, D, generator=torch.Generator().manual_seed(0))  # one row per id


def _enc(ids: list[int], length: int = L) -> torch.Tensor:
    """[length, D] tensor: the real rows (one fixed vector per id), then zeros."""
    out = torch.zeros(length, D)
    out[: len(ids)] = TABLE[ids]
    return out


def _positions(al: Aligner) -> list[int]:
    return [p for p, _ in al.layout]


def _check_rows(out: torch.Tensor, emb: torch.Tensor, al: Aligner) -> None:
    """Each token's row sits at its layout position; everything else is zero."""
    used = _positions(al)
    for i, p in enumerate(used):
        assert torch.equal(out[p], emb[i])
    rest = [p for p in range(out.shape[0]) if p not in used]
    assert torch.count_nonzero(out[rest]) == 0


def test_first_call_is_identity():
    al = Aligner(L)
    ids = [1, 2, 3, 9]
    emb = _enc(ids)
    out = al.place(emb, ids)
    assert torch.equal(out, emb)
    assert al.layout == ((0, 1), (1, 2), (2, 3), (3, 9))


def test_append_phrase_keeps_old_positions_including_end_token():
    al = Aligner(L)
    al.place(_enc([1, 2, 3, 9]), [1, 2, 3, 9])
    new = [1, 2, 3, 5, 6, 9]
    emb = _enc(new)
    out = al.place(emb, new)
    # 9 (the end token) stays at 3; the new words take the next free slots.
    assert al.layout == ((0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (3, 9))
    _check_rows(out, emb, al)


def test_remove_middle_phrase():
    al = Aligner(L)
    stack = [1, 2, 7, 8, 3, 4, 9]
    al.place(_enc(stack), stack)
    minus = [1, 2, 3, 4, 9]
    emb = _enc(minus)
    out = al.place(emb, minus)
    assert _positions(al) == [0, 1, 4, 5, 6]
    assert torch.count_nonzero(out[2:4]) == 0  # the removed words' positions
    _check_rows(out, emb, al)


def test_remove_then_readd_returns_to_original_layout():
    al = Aligner(L)
    stack = [1, 2, 7, 8, 3, 4, 9]
    al.place(_enc(stack), stack)
    al.place(_enc([1, 2, 3, 4, 9]), [1, 2, 3, 4, 9])
    out = al.place(_enc(stack), stack)
    assert _positions(al) == list(range(7))
    assert torch.equal(out, _enc(stack))


def test_completely_different_prompt_uses_unused_positions():
    al = Aligner(L)
    al.place(_enc([1, 2, 3]), [1, 2, 3])
    out = al.place(_enc([4, 5]), [4, 5])
    assert _positions(al) == [3, 4]
    assert torch.count_nonzero(out[:3]) == 0


def test_prefers_positions_unused_by_previous_layout_then_falls_back():
    al = Aligner(5)
    al.place(_enc([1, 2, 3], 5), [1, 2, 3])
    # 1 keeps 0; 7, 8 take never-used 3, 4; 9 falls back to the lowest freed one.
    al.place(_enc([1, 7, 8, 9], 5), [1, 7, 8, 9])
    assert al.layout == ((0, 1), (3, 7), (4, 8), (1, 9))


def test_fallback_when_every_position_was_used():
    al = Aligner(4)
    al.place(_enc([1, 2, 3, 4], 4), [1, 2, 3, 4])
    al.place(_enc([5, 6], 4), [5, 6])
    assert _positions(al) == [0, 1]


def test_output_is_permutation_of_real_rows_plus_zeros():
    rng = random.Random(0)
    gen = torch.Generator().manual_seed(1)
    al = Aligner(L)
    for _ in range(200):
        n = rng.randint(1, L)
        ids = [rng.randint(0, 5) for _ in range(n)]  # repeats on purpose
        emb = torch.randn(L, D, generator=gen)  # padding rows are ignored
        out = al.place(emb, ids)
        nonzero = out[out.abs().sum(dim=-1) != 0]
        assert nonzero.shape[0] == n
        key = lambda t: t[torch.argsort(t[:, 0])]  # noqa: E731
        assert torch.equal(key(nonzero), key(emb[:n]))
        assert len(set(_positions(al))) == n
        _check_rows(out, emb, al)


def test_input_not_modified():
    al = Aligner(L)
    al.place(_enc([1, 2, 3]), [1, 2, 3])
    emb = torch.randn(L, D)
    before = emb.clone()
    out = al.place(emb, [4, 1, 2, 3])
    assert torch.equal(emb, before)
    assert out.data_ptr() != emb.data_ptr()


@pytest.mark.parametrize(
    "shape, n",
    [((L + 1, D), 3), ((2, L, D), 3), ((L,), 3), ((1, 1, L, D), 3), ((L, D), 0), ((L, D), L + 1)],
)
def test_bad_shapes_raise_and_change_nothing(shape, n):
    al = Aligner(L)
    al.place(_enc([1, 2]), [1, 2])
    with pytest.raises(ValueError):
        al.place(torch.zeros(shape), list(range(n)))
    assert al.layout == ((0, 1), (1, 2))


def test_bad_length_raises():
    with pytest.raises(ValueError):
        Aligner(0)


def test_reset():
    al = Aligner(L)
    al.place(_enc([1, 2, 3]), [1, 2, 3])
    al.place(_enc([4, 5]), [4, 5])
    al.reset()
    assert al.layout is None
    assert torch.equal(al.place(_enc([4, 5]), [4, 5]), _enc([4, 5]))


def test_prepare_does_not_commit():
    al = Aligner(L)
    al.place(_enc([1, 2, 3]), [1, 2, 3])
    out, layout = al.prepare(_enc([1, 5, 2, 3]), [1, 5, 2, 3])
    assert al.layout == ((0, 1), (1, 2), (2, 3))
    al.commit(layout)
    assert al.layout == layout == ((0, 1), (3, 5), (1, 2), (2, 3))
    assert torch.equal(out[3], TABLE[5])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("batched", [False, True])
def test_dtype_and_shape_preserved(dtype, batched):
    al = Aligner(L)
    emb = _enc([1, 2, 3]).to(dtype)
    al.place(emb.unsqueeze(0) if batched else emb, [1, 2, 3])
    emb = _enc([1, 4, 2, 3]).to(dtype)
    emb = emb.unsqueeze(0) if batched else emb
    out = al.place(emb, [1, 4, 2, 3])
    assert out.shape == emb.shape and out.dtype == dtype and out.device == emb.device
    rows = out[0] if batched else out
    assert torch.equal(rows[3], TABLE[4].to(dtype))


def test_blend_after_removing_middle_phrase():
    stack = [1, 2, 7, 8, 3, 4, 9]
    minus = [1, 2, 3, 4, 9]
    a, b = _enc(stack), _enc(minus)

    al = Aligner(L)
    pa, pb = al.place(a, stack), al.place(b, minus)
    mid = blend(pa, pb, 0.5)
    for p in (0, 1, 4, 5, 6):  # survivors: identical rows blend to themselves
        assert torch.allclose(mid[p], pa[p], atol=1e-6)
    for p in (2, 3):  # removed words fade toward zero
        assert torch.allclose(mid[p].norm(), pa[p].norm() / 2, rtol=1e-5)

    # Without alignment the survivors after the removed phrase get mixed up.
    raw = blend(a, b, 0.5)
    assert not torch.allclose(raw[4], a[4], atol=1e-3)

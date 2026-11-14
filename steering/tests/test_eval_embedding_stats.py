import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from steering.blend import blend
from steering.eval.embedding_stats import (
    DEFAULT_PROMPTS,
    angles_deg,
    build_pairs,
    build_prompts,
    midpoint_norm_ratios,
    position_groups,
    shift_report,
    spherical_mask,
    suffix_alignment,
    token_shift,
)

D = 16


def _mixed_pair() -> tuple[torch.Tensor, torch.Tensor, list[bool]]:
    """Rows covering every branch of blend's spherical/linear choice."""
    g = torch.Generator().manual_seed(0)
    r = lambda: torch.randn(D, generator=g)  # noqa: E731
    v = r()
    rows = [
        (r(), r(), True),  # generic
        (torch.zeros(D), r(), False),  # a is padding
        (r(), torch.zeros(D), False),  # b is padding
        (torch.zeros(D), torch.zeros(D), False),  # both padding
        (r() * 1e-6, r(), False),  # a below eps
        (v, 2.0 * v, False),  # collinear
        (v, -v, False),  # opposite
        (v, v + 1e-7 * r(), False),  # nearly collinear, sin < eps
        (v, v + 0.05 * r(), True),  # small but real angle
    ]
    a = torch.stack([x for x, _, _ in rows])
    b = torch.stack([y for _, y, _ in rows])
    return a, b, [s for _, _, s in rows]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spherical_mask_matches_blend(dtype):
    a, b, expected = _mixed_pair()
    a, b = a.to(dtype), b.to(dtype)
    mask = spherical_mask(a, b)
    assert mask.tolist() == expected

    w = 0.5
    out = blend(a, b, w)
    lerp = a + w * (b - a)
    # Linear rows come out bit-identical to lerp; spherical rows keep the
    # lerped norm, which lerp itself does not (the angle shortens it).
    assert torch.equal(out[~mask], lerp[~mask])
    want_norm = (a.norm(dim=-1) + b.norm(dim=-1)) / 2
    assert torch.allclose(out[mask].norm(dim=-1), want_norm[mask], rtol=1e-5)
    assert not torch.allclose(out[mask][0], lerp[mask][0])


def test_angles_deg():
    e = torch.eye(3)
    a = torch.stack([e[0], e[0], e[0], e[0] + e[1]])
    b = torch.stack([e[0], e[1], -e[0], e[0]])
    assert torch.allclose(angles_deg(a, b), torch.tensor([0.0, 90.0, 180.0, 45.0]).double())


def test_position_groups():
    g = position_groups(5, 3, 8)
    assert int(g["both"].sum()) == 3
    assert int(g["one"].sum()) == 2
    assert int(g["neither"].sum()) == 3


def test_midpoint_norm_ratios_orthogonal_equal_norms():
    a = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
    b = torch.tensor([[0.0, 2.0], [0.0, 3.0]])
    slerp, lerp = midpoint_norm_ratios(a, b)
    assert torch.allclose(slerp, torch.ones(2).double())
    assert torch.allclose(lerp, torch.full((2,), math.sqrt(0.5)).double())


def test_suffix_alignment():
    #           base  removed  tail........
    ids_a = [1, 2, 7, 8, 3, 4, 5, 9]
    ids_b = [1, 2, 3, 4, 5, 9]
    assert suffix_alignment(ids_a, ids_b) == (4, 2, 4)
    assert suffix_alignment([1, 2, 1, 2], [1, 2]) == (4, 2, 0)
    assert suffix_alignment([1, 2, 3], [1, 2, 3]) == (3, 3, 0)


def test_token_shift_finds_same_word_and_padding():
    g = torch.Generator().manual_seed(1)
    table = torch.randn(20, D, generator=g)  # one fixed vector per token id
    ids_a = [1, 2, 7, 8, 3, 4, 5, 9]
    ids_b = [1, 2, 3, 4, 5, 9]
    L = 10
    a, b = torch.zeros(L, D), torch.zeros(L, D)
    a[: len(ids_a)] = table[ids_a]
    b[: len(ids_b)] = table[ids_b]

    r = token_shift(a, b, ids_a, ids_b)
    assert r["shift"] == 2
    assert r["tail_len"] == 4
    # Tail is at 4..7 in a and 2..5 in b: positions 2..5 are real in both,
    # 6..7 put a's last words over b's padding.
    assert r["tail_vs_padding"] == 2
    assert r["positional_angles"].numel() == 4
    assert r["positional_same_id"] == 0
    assert torch.all(r["positional_angles"] > 30.0)
    assert torch.allclose(r["aligned_angles"], torch.zeros(4).double(), atol=1e-4)

    text = "\n".join(shift_report(a, b, ids_a, ids_b))
    assert "VERDICT: yes" in text


def test_token_shift_tail_entirely_over_padding():
    # Short tail after a long removed phrase: b's tail sits on a's removed
    # words, a's tail sits on b's padding.
    ids_a = [1, 2, 7, 8, 6, 3, 9]
    ids_b = [1, 2, 3, 9]
    a, b = torch.randn(10, D), torch.randn(10, D)
    a[len(ids_a) :], b[len(ids_b) :] = 0.0, 0.0
    r = token_shift(a, b, ids_a, ids_b)
    assert (r["shift"], r["tail_len"]) == (3, 2)
    assert r["positional_angles"].numel() == 2
    assert r["positional_same_id"] == 0
    assert r["tail_vs_padding"] == 2
    assert "VERDICT: yes" in "\n".join(shift_report(a, b, ids_a, ids_b))


def test_token_shift_no_change():
    ids = [1, 2, 3]
    a = torch.randn(5, D)
    r = token_shift(a, a, ids, ids)
    assert r["shift"] == 0 and r["tail_len"] == 0
    assert "VERDICT: no shift" in "\n".join(shift_report(a, a, ids, ids))


def test_build_prompts_and_pairs():
    p = build_prompts(DEFAULT_PROMPTS)
    base = DEFAULT_PROMPTS["base"]
    ch = DEFAULT_PROMPTS["changes"]
    assert p["winter"] == base + ch["winter"]
    assert p["stack"] == base + ch["winter"] + ch["night"] + ch["rain"]
    assert p["stack_minus_middle"] == base + ch["winter"] + ch["rain"]
    pairs = build_pairs(DEFAULT_PROMPTS)
    assert ("base", "fog") in pairs
    assert ("stack", "stack_minus_middle") in pairs
    assert ("winter", "night") in pairs
    with pytest.raises(ValueError):
        build_prompts({**DEFAULT_PROMPTS, "stack": ["winter", "night"]})


def test_importing_eval_does_not_import_transformers():
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys, steering.eval, steering.eval.embedding_stats; "
        "assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)

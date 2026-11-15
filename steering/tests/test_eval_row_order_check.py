import math
import sys
import types

import pytest
import torch

from steering.eval import row_order_check as roc

pytestmark = pytest.mark.tools  # a tool's tests: run with pytest -m tools (pytest.ini)

L, D = 32, 8


def _encode(prompt: str) -> tuple[torch.Tensor, list[int]]:
    """[1, L, D] with one random row per word, zero padding."""
    words = prompt.split()
    g = torch.Generator().manual_seed(sum(map(ord, prompt)))
    emb = torch.zeros(1, L, D)
    emb[0, : len(words)] = torch.randn(len(words), D, generator=g)
    return emb, [sum(map(ord, w)) for w in words]


class FakeBackend:
    """Seeded fake; `mode` is 'invariant', 'ordered' or 'random' (non-deterministic)."""

    def __init__(self, mode: str = "invariant", bare: bool = False) -> None:
        self.mode, self.bare = mode, bare
        self.t = 0
        g = torch.Generator().manual_seed(1)
        self.proj = torch.randn(D, 3, generator=g)

    def encode_prompt(self, prompt: str):
        emb, ids = _encode(prompt)
        return emb if self.bare else (emb, ids)

    def reset(self) -> None:
        self.t = 0

    def generate_chunk(self, cond: torch.Tensor) -> torch.Tensor:
        rows = cond[0]
        if self.mode == "ordered":
            v = rows[0] * 3  # only the row at position 0 matters
        else:
            v = rows.sum(0)
        if self.mode == "random":
            noise = torch.randn(3, generator=torch.Generator().manual_seed(torch.seed())) * 0.3
        else:
            noise = torch.randn(3) * 0.01  # global RNG: deterministic after manual_seed
        self.t += 1
        return torch.tanh(v @ self.proj * 0.25 + 0.05 * self.t) + noise

    def decode_chunk(self, lat: torch.Tensor) -> torch.Tensor:
        px = (128 + 100 * lat.clamp(-1.2, 1.2)).round().clamp(0, 255).to(torch.uint8)
        return px.view(1, 1, 1, 3).expand(2, 4, 4, 3).clone()


# ---------------------------------------------------------------- pure parts


def test_scatter_positions_collision_free_for_512():
    pos = roc.scatter_positions(512, 512)
    assert sorted(pos) == list(range(512))
    assert pos[:3] == [13, 20, 27]


def test_scatter_positions_detects_collision():
    with pytest.raises(ValueError):
        roc.scatter_positions(5, 14)  # gcd(7, 14) = 7


def test_real_count_from_ids_and_inferred():
    emb, ids = _encode("one two three")
    assert roc.real_count(emb, ids) == 3
    assert roc.real_count(emb, None) == 3
    bad = emb.clone()
    bad[0, 0] = 0
    with pytest.raises(ValueError):
        roc.real_count(bad, None)


def test_build_conditionings():
    emb, ids = _encode("a b c d e")
    control, _ = _encode("x y")
    c = roc.build_conditionings(emb, 5, control)
    assert c["A"] is emb and c["D"] is control
    pos = roc.scatter_positions(5, L)
    for i, p in enumerate(pos):
        assert torch.equal(c["B"][0, p], emb[0, i])
    others = [p for p in range(L) if p not in pos]
    assert c["B"][0, others].abs().sum() == 0
    assert torch.equal(c["C"][0, :5], emb[0, :5].flip(0))
    assert torch.equal(c["C"][0, 5:], emb[0, 5:])
    # same rows, so a row-sum is unchanged
    for k in "BC":
        assert torch.allclose(c[k].sum(1), emb.sum(1))
    # 2-D input keeps its shape
    c2 = roc.build_conditionings(emb[0], 5, control[0])
    assert c2["B"].shape == (L, D)


def test_compare():
    x = torch.zeros(2, 4, 4, 3, dtype=torch.uint8)
    s = roc.compare(x, x)
    assert s["max_abs"] == 0 and s["mean_abs"] == 0 and s["psnr"] == math.inf
    y = x.clone()
    y[0, 0, 0, 0] = 255
    s = roc.compare(x, y)
    assert s["max_abs"] == 255 and s["psnr"] < math.inf
    with pytest.raises(ValueError):
        roc.compare(x, x[:1])


def _st(a2, b, c, d):
    return {k: {"mean_abs": v} for k, v in zip(("A2", "B", "C", "D"), (a2, b, c, d))}


@pytest.mark.parametrize(
    "stats,expected",
    [
        (_st(0.0, 0.2, 0.5, 40.0), "PASS"),
        (_st(0.5, 1.4, 1.0, 40.0), "PASS"),  # within 3x noise
        (_st(0.0, 30.0, 0.1, 40.0), "FAIL"),
        (_st(2.0, 0.0, 0.0, 40.0), "INCONCLUSIVE"),  # not deterministic
        (_st(0.0, 0.0, 0.0, 2.0), "INCONCLUSIVE"),  # control too weak
        (_st(0.0, 5.0, 0.0, 40.0), "INCONCLUSIVE"),  # in between
    ],
)
def test_verdict_rules(stats, expected):
    result, why = roc.verdict(stats)
    assert result == expected
    assert why


# ---------------------------------------------------------------- end to end


@pytest.mark.parametrize(
    "mode,expected", [("invariant", "PASS"), ("ordered", "FAIL"), ("random", "INCONCLUSIVE")]
)
def test_check_on_fake(mode, expected):
    result, report, frames = roc.check(FakeBackend(mode), "a b c d e f g", chunks=4, seed=0)
    assert result == expected, report
    assert f"VERDICT: {expected}" in report
    assert set(frames) == {"A", "A2", "B", "C", "D"}
    assert frames["A"].shape == (8, 4, 4, 3)


def test_check_bare_tensor_infers_count():
    result, report, _ = roc.check(FakeBackend(bare=True), "a b c", chunks=3, seed=0)
    assert result == "PASS"
    assert "inferred" in report


def test_main_through_importlib(tmp_path, monkeypatch, capsys):
    mod = types.ModuleType("fake_rowcheck_backend")
    mod.load = lambda width, height, steps: FakeBackend("ordered")
    monkeypatch.setitem(sys.modules, "fake_rowcheck_backend", mod)
    code = roc.main(["--backend", "fake_rowcheck_backend", "--chunks", "3",
                     "--out", str(tmp_path), "--save-frames"])
    assert code == 1
    text = (tmp_path / "row_order_check.txt").read_text()
    assert "VERDICT: FAIL" in text
    assert "VERDICT: FAIL" in capsys.readouterr().out
    assert (tmp_path / "B.frames.pt").exists()

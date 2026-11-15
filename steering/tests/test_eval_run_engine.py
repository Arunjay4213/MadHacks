import inspect
import sys
import types

import numpy as np
import pytest
import torch

from steering.eval import run_engine as re_
from steering.eval import scenarios as sc

pytestmark = pytest.mark.tools  # a tool's tests: run with pytest -m tools (pytest.ini)

L, D, H, W = 32, 8, 8, 8


class FakeBackend:
    """Fake engine: deterministic given the global torch seed, 2 frames per chunk.

    `fail_on_reset=k` makes generate_chunk raise during the k-th scenario.
    """

    def __init__(self, bare: bool = False, fail_on_reset: int | None = None) -> None:
        self.bare = bare
        self.fail_on_reset = fail_on_reset
        self.resets = 0
        self.t = 0
        self.cache_resets = 0
        self.proj = torch.randn(D, 3, generator=torch.Generator().manual_seed(2))

    def encode_prompt(self, prompt: str):
        words = prompt.replace(",", " ,").split()[:L]
        g = torch.Generator().manual_seed(sum(map(ord, prompt)))
        emb = torch.zeros(1, L, D)
        emb[0, : len(words)] = torch.randn(len(words), D, generator=g)
        ids = [sum(map(ord, w)) for w in words]
        return emb if self.bare else (emb, ids)

    def reset(self) -> None:
        self.resets += 1
        self.t = 0

    def drop_text_cache(self) -> None:
        self.cache_resets += 1

    def generate_chunk(self, cond: torch.Tensor) -> torch.Tensor:
        if self.fail_on_reset == self.resets:
            raise RuntimeError("CUDA out of memory (fake)")
        self.t += 1
        return torch.tanh(cond[0].sum(0) @ self.proj * 0.3) + torch.randn(3) * 0.01

    def decode_chunk(self, lat: torch.Tensor) -> torch.Tensor:
        px = (128 + 100 * lat).round().clamp(0, 255).to(torch.uint8)
        return px.view(1, 1, 1, 3).expand(2, H, W, 3).clone()


def _inject(monkeypatch, name: str, backend: FakeBackend) -> list:
    calls = []
    mod = types.ModuleType(name)

    def load(width, height, steps):
        calls.append((width, height, steps))
        return backend

    mod.load = load
    monkeypatch.setitem(sys.modules, name, mod)
    return calls


# ---------------------------------------------------------------- conversions


def test_frames_to_float_from_tensor_and_numpy():
    x = torch.zeros(3, 4, 5, 3, dtype=torch.uint8)
    x[1, 2, 3] = torch.tensor([255, 0, 51], dtype=torch.uint8)
    for src in (x, x.numpy()):
        f = re_.frames_to_float(src)
        assert f.shape == (3, 3, 4, 5) and f.dtype == torch.float32
        assert f[1, :, 2, 3].tolist() == pytest.approx([1.0, 0.0, 0.2])
        assert f.device.type == "cpu"
    assert re_.frames_to_float(np.zeros((4, 5, 3), np.uint8)).shape == (1, 3, 4, 5)


@pytest.mark.parametrize(
    "bad",
    [torch.zeros(2, 4, 5, 4, dtype=torch.uint8), torch.zeros(2, 4, 5, 3), torch.zeros(5, 3, dtype=torch.uint8)],
)
def test_frames_to_float_rejects(bad):
    with pytest.raises(ValueError):
        re_.frames_to_float(bad)


def test_split_and_make_encode():
    emb = torch.ones(1, L, D)
    assert re_.split_encoded(emb) == (emb, None)
    e, ids = re_.split_encoded((emb, (1, 2)))
    assert e is emb and ids == [1, 2]
    with pytest.raises(TypeError):
        re_.split_encoded("nope")

    tup, bare = FakeBackend(), FakeBackend(bare=True)
    assert isinstance(re_.make_encode(tup, align=False)("a b"), torch.Tensor)
    assert isinstance(re_.make_encode(bare, align=False)("a b"), torch.Tensor)
    e, ids = re_.make_encode(tup, align=True)("a b")
    assert e.shape == (1, L, D) and len(ids) == 2
    with pytest.raises(TypeError, match="--align needs"):
        re_.make_encode(bare, align=True)("a b")


def test_encoder_falls_back_to_encode():
    class Only:
        def encode(self, p):
            return torch.zeros(1, 2, 2)

    assert re_.encoder_of(Only())("x").shape == (1, 2, 2)
    with pytest.raises(AttributeError):
        re_.encoder_of(object())


# ---------------------------------------------------------------- seeding


class _Base:
    def __init__(self):
        self.log = []

    def reset(self):
        self.log.append("reset")


def test_seed_prefers_backend_methods():
    class WithSeed(_Base):
        def seed(self, s):
            self.log.append(("seed", s))

    class WithSetSeed(_Base):
        def set_seed(self, s):
            self.log.append(("set_seed", s))

    class ResetKw(_Base):
        def reset(self, seed=None):
            self.log.append(("reset", seed))

    b = WithSeed()
    assert re_.seed_and_reset(b, 3) == "backend.seed" and b.log == [("seed", 3), "reset"]
    b = WithSetSeed()
    assert re_.seed_and_reset(b, 4) == "backend.set_seed" and b.log == [("set_seed", 4), "reset"]
    b = ResetKw()
    assert re_.seed_and_reset(b, 5) == "backend.reset(seed=)" and b.log == [("reset", 5)]
    b = _Base()
    assert re_.seed_and_reset(b, None) == "none" and b.log == ["reset"]


def test_seed_falls_back_to_torch():
    b = _Base()
    assert re_.seed_and_reset(b, 7) == "torch.manual_seed"
    first = torch.rand(3)
    re_.seed_and_reset(b, 7)
    assert torch.equal(torch.rand(3), first)


def test_adapter_times_and_converts():
    a = re_.EngineAdapter(FakeBackend())
    out = a.generate(FakeBackend().encode_prompt("a b")[0])
    assert out.shape == (2, 3, H, W) and 0 <= out.min() and out.max() <= 1
    assert len(a.gen_s) == len(a.dec_s) == 1


# ---------------------------------------------------------------- full runs


def test_run_all_default_scenarios(tmp_path):
    backend = FakeBackend()
    scenarios = sc.default_scenarios(4)
    summary, failures = re_.run_all(backend, scenarios, str(tmp_path), 16.0, video=False)
    assert failures == 0, summary
    assert backend.resets == len(scenarios)
    for s in scenarios:
        assert "== " + s.name in (tmp_path / f"{s.name}.report.txt").read_text()
        assert not (tmp_path / f"{s.name}.frames.pt").exists()
        assert f"{s.name} " in summary
    assert (tmp_path / "summary.txt").read_text() == summary
    assert "seeding: torch.manual_seed" in summary
    assert f"{len(scenarios)}/{len(scenarios)} scenarios ok" in summary


def test_main_via_importlib(tmp_path, monkeypatch, capsys):
    backend = FakeBackend()
    calls = _inject(monkeypatch, "fake_engine_backend_ok", backend)
    code = re_.main([
        "--backend", "fake_engine_backend_ok", "--width", "64", "--height", "32", "--steps", "3",
        "--scenarios", "single,hard_swap", "--save-frames", "--no-video",
        "--force-cache-reset-hook", "drop_text_cache", "--out", str(tmp_path),
    ])
    assert code == 0
    assert calls == [(64, 32, 3)]
    out = capsys.readouterr().out
    assert "backend fake_engine_backend_ok 64x32 steps=3" in out
    assert "cache-reset hook: drop_text_cache" in out
    assert backend.cache_resets > 0
    frames = torch.load(tmp_path / "single.frames.pt")
    assert frames.dim() == 4 and frames.shape[1] == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([
        "single.report.txt", "single.frames.pt", "hard_swap.report.txt",
        "hard_swap.frames.pt", "summary.txt",
    ])


def test_failing_scenario_recorded_and_exit_1(tmp_path, monkeypatch, capsys):
    _inject(monkeypatch, "fake_engine_backend_bad", FakeBackend(fail_on_reset=2))
    code = re_.main([
        "--backend", "fake_engine_backend_bad", "--scenarios", "single,hard_swap,interrupt",
        "--no-video", "--out", str(tmp_path),
    ])
    assert code == 1
    summary = (tmp_path / "summary.txt").read_text()
    assert "hard_swap            FAILED RuntimeError: CUDA out of memory (fake)" in summary
    assert "2/3 scenarios ok" in summary
    assert (tmp_path / "interrupt.report.txt").exists()  # the run continued
    assert not (tmp_path / "hard_swap.report.txt").exists()


def test_unknown_scenario_and_missing_hook(tmp_path):
    with pytest.raises(ValueError):
        re_.select_scenarios("single,nope", 4)
    with pytest.raises(AttributeError):
        re_.run_all(FakeBackend(), sc.default_scenarios(4)[:1], str(tmp_path), 16.0,
                    hook_name="no_such_method", video=False)


def test_align_mode(tmp_path):
    scen = sc.default_scenarios(4)[:2]
    if "aligner" not in inspect.signature(sc.run).parameters:
        with pytest.raises(RuntimeError, match="aligner"):
            re_.run_all(FakeBackend(), scen, str(tmp_path), 16.0, align=True, video=False)
        return
    summary, failures = re_.run_all(FakeBackend(), scen, str(tmp_path), 16.0,
                                    align=True, video=False)
    assert failures == 0, summary
    assert "align: True" in summary
    # a bare-tensor backend cannot align: every scenario fails with a clear message
    summary, failures = re_.run_all(FakeBackend(bare=True), scen, str(tmp_path), 16.0,
                                    align=True, video=False)
    assert failures == 2 and "--align needs" in summary


def test_video_skipped_when_no_writer(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio", None)
    monkeypatch.setitem(sys.modules, "torchvision", None)
    monkeypatch.setitem(sys.modules, "torchvision.io", None)
    frames = torch.rand(3, 3, 4, 4)
    assert re_.write_video(str(tmp_path / "x.mp4"), frames, 16.0) is None
    assert not (tmp_path / "x.mp4").exists()
    summary, _ = re_.run_all(FakeBackend(), sc.default_scenarios(4)[:1], str(tmp_path), 16.0)
    assert "video: SKIPPED" in summary

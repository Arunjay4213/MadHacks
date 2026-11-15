import os

import pytest
import torch

from engine.backends.fake import FakeBackend
from steering.eval import follow_probe as fp

pytestmark = pytest.mark.tools  # a tool's tests: run with pytest -m tools (pytest.ini)

FAST = ["--backend", "fake", "--backend-opt", "gen_ms=0", "--backend-opt", "decode_ms=0",
        "--backend-opt", "encode_ms=0", "--width", "64", "--height", "48", "--device", "cpu",
        "--decode-device", "cpu", "--no-clip"]


def test_switch_smoke_on_fake_backend(tmp_path):
    out = str(tmp_path)
    assert fp.main(["--out", out, "--chunks", "6", "--switch-chunk", "3", "--pair", "meadow_winter"] + FAST) == 0
    summary = open(os.path.join(out, "summary.md")).read()
    assert "| hardcut | " in summary and "unsupported" not in summary.split("| hardcut |")[1].split("\n")[0]
    for name in fp.VARIANTS:
        if name != "hardcut":
            assert f"| {name} | unsupported |" in summary
    folder = os.path.join(out, "meadow_winter", "hardcut")
    assert len(os.listdir(os.path.join(folder, "frames"))) == 1 + 5 * 4  # fake: 1 frame, then 4 per chunk
    assert os.path.exists(os.path.join(folder, "sheet.png"))
    assert len(open(os.path.join(folder, "chunk_ms.txt")).read().splitlines()) == 6


def test_long_smoke_on_fake_backend(tmp_path):
    out = str(tmp_path)
    assert fp.main(["--out", out, "--long", "1"] + FAST) == 0
    summary = open(os.path.join(out, "summary.md")).read()
    assert "| none | ok |" in summary and "| refresh40_keep6 | unsupported |" in summary
    assert "## saturation per 5 s" in summary
    assert len(os.listdir(os.path.join(out, "long", "none", "frames"))) == 1 + 3 * 4


class RebaseFake(FakeBackend):
    """The fake backend plus a `rebase` keyword that is only recorded."""

    first_frame_steps = 2

    def __init__(self):
        super().__init__(32, 24, 0, gen_ms=0, decode_ms=0, encode_ms=0)
        self.rebases = []

    def reset(self, seed=None):
        super().reset()
        self.rebases = []

    def generate_chunk(self, cond, rebase=None):
        self.rebases.append(rebase)
        return super().generate_chunk(cond)


class CurveClip(fp.Clip):
    """Fake CLIP: base wins before chunk 5, target after."""

    def __init__(self):
        super().__init__(False, "cpu")
        self.error = None

    def curve(self, images, prompts):
        n = images.shape[0]
        target = torch.tensor([0.1 if i < 5 else 0.3 for i in range(n)])
        return torch.stack([torch.full((n,), 0.2), target], dim=1)


def test_rebase_only_on_switch_chunk_and_anchor_skips(tmp_path):
    b = RebaseFake()
    lines = fp.run_switch(b, str(tmp_path), ["meadow_city"], ["keep3", "keep0_a1", "keep0_a2"], seed=0,
                          switch=2, chunks=30, clip=CurveClip())
    assert b.rebases == [None, None, {"keep": 0, "anchor_step": 1}] + [None] * 27  # keep0_a1 ran last
    text = "\n".join(lines)
    assert "| keep0_a2 | skipped (first-frame schedule has 2 steps) |" in text
    row = next(line for line in lines if line.startswith("| keep3 |"))
    assert row.endswith("| +0.1000 | 3 |")  # tail over the last 5 s; target wins 3 chunks after the switch
    assert os.path.exists(os.path.join(str(tmp_path), "meadow_city", "keep3", "clip.txt"))


def test_follow_metrics_never():
    curve = torch.tensor([[0.3, 0.1]] * 30)
    tail, cross = fp.follow_metrics(curve, 10, 4.0)
    assert cross is None and abs(tail + 0.2) < 1e-6


def test_frame_stats():
    grey = torch.full((4, 4, 3), 128, dtype=torch.uint8).numpy()
    assert fp.frame_stats(grey) == (0.0, 0.0)
    red = grey.copy()
    red[..., 1:] = 0
    assert abs(fp.frame_stats(red)[0] - 1.0) < 1e-6

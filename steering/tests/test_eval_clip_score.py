import pytest
import torch

from steering.eval import clip_score as cs
from steering.eval import run_engine as re_
from steering.eval import scenarios as sc
from steering.tests.test_eval_run_engine import FakeBackend

pytestmark = pytest.mark.tools  # a tool's tests: run with pytest -m tools (pytest.ini)


def _ramp(t: int, steer: int, length: int) -> torch.Tensor:
    """0 before `steer`, linear rise to 1 over `length` frames, then 1."""
    x = torch.zeros(t, dtype=torch.float64)
    for i in range(steer, t):
        x[i] = min(1.0, (i - steer + 1) / length)
    return x


def test_similarity_curve_is_cosine():
    img = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    txt = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    c = cs.similarity_curve(img, txt)
    assert c.shape == (3, 2)
    assert c[0].tolist() == pytest.approx([1.0, 0.0])
    assert c[1].tolist() == pytest.approx([0.0, 1.0])
    assert c[2].tolist() == pytest.approx([2**-0.5, 2**-0.5])


def test_relative_score():
    curve = torch.tensor([[0.3, 0.1], [0.2, 0.25]])
    assert cs.relative_score(curve, 0, 1).tolist() == pytest.approx([-0.2, 0.05])


def test_transition_report_times_a_linear_ramp():
    # steer at 16, rises over 8 frames: 10% at frame 16, 90% at frame 23.
    r = cs.transition_report(_ramp(48, 16, 8) * 0.05, steer_frame=16, fps=16.0)
    assert r["rise"] == pytest.approx(0.05)
    assert r["t10_ms"] == pytest.approx(0.0)
    assert r["t90_ms"] == pytest.approx(7 * 1000 / 16)
    assert r["decreasing_frac"] == 0.0 and r["monotone"]
    assert r["window"][0] < 16 and r["window"][1] == 47


def test_transition_report_with_subsampled_frames_and_end():
    full = _ramp(64, 16, 8)
    full[40:] = -1.0  # a later event, excluded by end=40
    idx = list(range(0, 64, 2))
    r = cs.transition_report(full[idx], 16, 16.0, frame_idx=idx, end=40)
    assert r["window"][1] == 38
    assert r["t90_ms"] == pytest.approx(8 * 1000 / 16)  # frame 24 is the first sample >= 90%
    assert r["rise"] == pytest.approx(1.0)


def test_transition_report_flat_and_noisy():
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(48, generator=g, dtype=torch.float64) * 0.01
    r = cs.transition_report(noise, 16, 16.0)
    assert r["t10_ms"] is None and r["t90_ms"] is None
    # A zigzag after the steer is not monotone.
    zig = _ramp(48, 16, 4)
    zig[20::2] -= 0.6
    assert not cs.transition_report(zig, 16, 16.0)["monotone"]
    # Jitter under 2% of the rise on the plateau is not counted as going down.
    jit = _ramp(48, 16, 8)
    jit[30::2] -= 0.005
    assert cs.transition_report(jit, 16, 16.0)["decreasing_frac"] == 0.0


def test_transition_report_bad_input():
    with pytest.raises(ValueError):
        cs.transition_report(torch.zeros(10), 50, 16.0)
    with pytest.raises(ValueError):
        cs.transition_report(torch.zeros(10), 5, 16.0, frame_idx=[0, 1])


def test_format_text():
    idx = [0, 1, 2]
    curve = torch.tensor([[0.3, 0.2], [0.3, 0.25], [0.3, 0.31]], dtype=torch.float64)
    text = cs.format_table(idx, curve, cs.relative_score(curve, 0, 1), steer=1)
    assert "<- steer" in text.splitlines()[2]
    rep = cs.format_report({"window": (0, 9), "rise": 0.1, "t10_ms": None, "t90_ms": 62.5,
                            "decreasing_frac": 0.5, "monotone": False})
    assert "90% at 62 ms" in rep and "10% at -" in rep and "NOT monotone" in rep


def _by_name(name: str) -> sc.Scenario:
    return next(s for s in sc.default_scenarios(4) if s.name == name)


def test_scenario_pairs_from_report_rows():
    s = _by_name("stack_remove_middle")
    rows = [(e.step, e.step * 4, "final") for e in s.events]
    rows[1] = (rows[1][0], rows[1][1], "stale")
    pairs = cs.scenario_pairs(s, rows, 4.0)
    prompts = [e.msg["prompt"] for e in s.events]
    assert [p[0] for p in pairs] == [16, 64, 88]  # events at steps 4, 10, 16, 22
    assert pairs[0][1:3] == (s.initial_prompt, prompts[0])
    assert pairs[1][1:3] == (prompts[0], prompts[2])  # the stale message changed nothing
    assert pairs[0][3] == 64 and pairs[2][3] is None


def test_scenario_pairs_without_report_uses_frames_per_step():
    s = _by_name("interrupt")
    pairs = cs.scenario_pairs(s, [], 2.0)
    assert [(p[0], p[3]) for p in pairs] == [(8, 12), (12, None)]
    assert pairs[1][1] == sc.BASE + sc.WINTER and pairs[1][2] == sc.BASE + sc.NIGHT


class FakeScorer:
    """Stands in for ClipScorer: image feature = mean colour, text = hashed vector."""

    def embed_images(self, frames: torch.Tensor) -> torch.Tensor:
        return frames.mean(dim=(2, 3))

    def embed_texts(self, prompts):
        return torch.stack([torch.randn(3, generator=torch.Generator().manual_seed(len(p)))
                            for p in prompts])


def test_run_dir_on_run_engine_output(tmp_path, capsys):
    scen = [_by_name(n) for n in ("single", "interrupt", "strength_steps")]
    _, failures = re_.run_all(FakeBackend(), scen, str(tmp_path), 16.0,
                              save_frames=True, video=False)
    assert failures == 0
    torch.save(torch.rand(4, 3, 8, 8), tmp_path / "other.frames.pt")
    (tmp_path / "hard_swap.frames.pt").write_bytes(b"broken")
    assert cs.run_dir(str(tmp_path), FakeScorer(), 16.0, 1, 4) == 1
    out = capsys.readouterr()
    assert "other: skipped" in out.out and "hard_swap: FAILED" in out.err
    text = (tmp_path / "interrupt.clip.txt").read_text()
    assert "steer at frame 8" in text and "steer at frame 12" in text
    assert "sim_before" in text
    steps = (tmp_path / "strength_steps.clip.txt").read_text()
    assert steps.count("not scored") == 3  # same prompt at new strengths


def test_cli_validates_args():
    for argv in ([], ["x.pt"], ["x.pt", "--dir", "d"], ["x.pt", "--before", "a"]):
        with pytest.raises(SystemExit):
            cs.main(argv)

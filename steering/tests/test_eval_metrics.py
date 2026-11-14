import pytest
import torch

from steering.eval.metrics import find_cuts, frame_change, summarize, transition_span

T, C, H, W = 60, 3, 16, 16


def _img(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(C, H, W, generator=g)


def _noise(t: int, std: float, seed: int = 99) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(t, C, H, W, generator=g) * std


def _drift(img: torch.Tensor, t: int, rate: float = 0.002, std: float = 0.01) -> torch.Tensor:
    ramp = torch.arange(t).view(t, 1, 1, 1) * rate
    return (img.expand(t, C, H, W) * 0.9 + ramp + _noise(t, std)).clamp(0, 1)


def test_frame_change_shape_and_value():
    frames = torch.zeros(4, C, H, W)
    frames[2:] = 0.5
    d = frame_change(frames)
    assert d.shape == (3,)
    assert torch.allclose(d, torch.tensor([0.0, 0.5, 0.0]))


def test_frame_change_rejects_bad_shape():
    with pytest.raises(ValueError):
        frame_change(torch.zeros(1, C, H, W))
    with pytest.raises(ValueError):
        frame_change(torch.zeros(5, H, W))


def test_static_clip_has_no_cuts():
    frames = _img(0).expand(T, C, H, W).clone()
    assert find_cuts(frames) == []


def test_slow_drift_with_noise_has_no_cuts():
    assert find_cuts(_drift(_img(0), T)) == []


def test_hard_cut_is_found():
    frames = torch.cat([_drift(_img(0), 20), _drift(_img(1), T - 20)])
    assert find_cuts(frames) == [20]


def test_cross_fade_is_not_a_cut():
    a, b = _img(0), _img(1)
    fade = torch.linspace(0, 1, 12).view(12, 1, 1, 1)
    clean = torch.cat(
        [a.expand(20, C, H, W), a + fade * (b - a), b.expand(T - 32, C, H, W)]
    )
    frames = (clean + _noise(T, 0.01)).clamp(0, 1)
    assert find_cuts(frames) == []


def test_transition_span_on_known_ramp():
    s = torch.zeros(60)
    # Ramp values 0.025, 0.075, ... avoid landing exactly on lo or hi.
    s[20:40] = (torch.arange(20, dtype=torch.float64) + 0.5) / 20.0
    s[40:] = 1.0
    s = 0.2 + 0.1 * s  # e.g. similarity to the target prompt
    assert transition_span(s) == (22, 38)


def test_transition_span_none_when_flat_or_noisy():
    assert transition_span(torch.full((50,), 0.3)) is None
    g = torch.Generator().manual_seed(0)
    assert transition_span(0.3 + 0.01 * torch.randn(50, generator=g)) is None
    assert transition_span(torch.linspace(1.0, 0.0, 50)) is None  # falls, not rises


def test_summarize_steer_at_known_frame():
    fps, steer = 16.0, 30
    base = _img(0).expand(T, C, H, W).clone()
    # The picture starts changing 2 frames after the steer, then settles.
    change = torch.zeros(T, 1, 1, 1)
    change[steer + 2 : steer + 14, 0, 0, 0] = torch.arange(1, 13) * 0.02
    change[steer + 14 :] = 12 * 0.02
    frames = (base * 0.7 + change + _noise(T, 0.005)).clamp(0, 1)

    out = summarize(frames, steer, fps)
    assert out["cuts"] == []
    assert out["reaction_ms"] == pytest.approx(2 * 1000.0 / fps)
    assert steer + 2 <= out["peak_frame"] < steer + 14
    assert out["peak_ratio"] > 2.0


def test_summarize_no_reaction_and_hard_cut():
    frames = torch.cat([_drift(_img(0), 20), _drift(_img(1), T - 20)])
    still = summarize(_drift(_img(0), T), 30, 16.0)
    assert still["reaction_ms"] is None
    cut = summarize(frames, 18, 16.0)
    assert cut["cuts"] == [20]
    assert cut["peak_frame"] == 20
    assert cut["reaction_ms"] == pytest.approx(125.0)
    assert cut["peak_ratio"] > 10


def test_summarize_static_pre_steer_gives_none_ratio():
    frames = _img(0).expand(T, C, H, W).clone()
    frames[40:] = 0.0
    out = summarize(frames, 30, 16.0)
    assert out["peak_ratio"] is None
    assert out["reaction_ms"] == pytest.approx(10 * 1000.0 / 16.0)


def test_summarize_rejects_bad_steer_frame():
    frames = _drift(_img(0), 10)
    with pytest.raises(ValueError):
        summarize(frames, 1, 16.0)
    with pytest.raises(ValueError):
        summarize(frames, 10, 16.0)

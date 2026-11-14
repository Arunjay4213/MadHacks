import pytest
import torch

from steering import Aligner
from steering.eval.scenarios import (
    BASE,
    FAKE_ALIGN_LEN,
    NIGHT,
    RAIN,
    WINTER,
    Event,
    FakeModel,
    Scenario,
    default_scenarios,
    fake_encode,
    fake_encode_ids,
    main,
    pace_scenarios,
    run,
)

FPS = 16.0


def _by_name(ramp: int = 4) -> dict[str, Scenario]:
    return {s.name: s for s in default_scenarios(ramp)}


def _run(sc: Scenario, **kw):
    return run(sc, FakeModel(), fake_encode, FPS, **kw)


def _msg(prompt: str, ramp: int = 4) -> dict:
    return {"prompt": prompt, "ramp_chunks": ramp, "provisional": False, "cap": 1.0}


class _SpyModel:
    """Fake generate that records each cond and returns `frames` frames."""

    def __init__(self, frames: int = 4) -> None:
        self.model = FakeModel()
        self.frames = frames
        self.conds: list[torch.Tensor] = []

    def __call__(self, cond: torch.Tensor) -> torch.Tensor:
        self.conds.append(cond)
        return self.model(cond)[: self.frames]


def test_events_applied_at_their_step():
    sc = Scenario("t", BASE, (Event(3, _msg(BASE + WINTER, 2)),), 8)
    spy = _SpyModel()
    res = run(sc, spy, fake_encode, FPS)
    base = fake_encode(BASE)
    assert all(torch.equal(c, base) for c in spy.conds[:3])
    assert not torch.equal(spy.conds[3], base)
    assert torch.equal(spy.conds[4], fake_encode(BASE + WINTER))
    assert [r.progress for r in res.steps] == [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize("ramp", [1, 4, 7])
def test_cond_changed_hook_fires_once_per_ramp_step(ramp):
    sc = pace_scenarios((ramp,))[0]
    calls = []
    res = _run(sc, on_cond_changed=lambda: calls.append(len(calls)))
    changed = [r.step for r in res.steps if r.cond_changed]
    lead = sc.events[0].step
    assert changed == list(range(lead, lead + ramp))
    assert len(calls) == ramp


def test_cond_changed_hook_silent_when_idle():
    calls = []
    res = _run(Scenario("idle", BASE, (), 6), on_cond_changed=lambda: calls.append(1))
    assert calls == []
    assert not any(r.cond_changed for r in res.steps)
    assert res.handled == [] and res.metrics["events"] == []


def test_hard_swap_cuts_and_long_ramp_does_not():
    scs = _by_name()
    hard = _run(scs["hard_swap"])
    steer = hard.metrics["events"][0]["steer_frame"]
    assert steer == hard.steps[scs["hard_swap"].events[0].step].first_frame
    assert hard.metrics["cuts"] == [steer]
    assert _run(pace_scenarios((12,))[0]).metrics["cuts"] == []
    assert _run(scs["single"]).metrics["cuts"] == []  # default ramp 4, fake model


@pytest.mark.parametrize("name, prompt", [("early_wrong", BASE + NIGHT), ("early_right", BASE + WINTER)])
def test_early_steering_ends_on_final_prompt(name, prompt):
    spy = _SpyModel()
    res = run(_by_name()[name], spy, fake_encode, FPS)
    assert torch.equal(spy.conds[-1], fake_encode(prompt))
    kinds = [r["kind"] for _, r in res.handled]
    assert kinds == ["provisional", "confirm" if name == "early_right" else "final"]


def test_stack_remove_middle_ends_without_night():
    spy = _SpyModel()
    res = run(_by_name()["stack_remove_middle"], spy, fake_encode, FPS)
    assert torch.equal(spy.conds[-1], fake_encode(BASE + WINTER + RAIN))
    assert [r["kind"] for _, r in res.handled] == ["final"] * 4


def test_handle_results_recorded_including_unchanged():
    msg = _msg(BASE + WINTER, 2)
    sc = Scenario("t", BASE, (Event(2, msg), Event(6, msg)), 8)
    res = _run(sc)
    assert [e for e, _ in res.handled] == list(sc.events)
    assert res.handled[0][1]["applied"] and res.handled[0][1]["kind"] == "final"
    assert res.handled[1][1] == {"applied": False, "kind": None, "reason": "unchanged",
                                 "cache_hit": True}


def test_report_lists_every_event():
    sc = _by_name()["stack_remove_middle"]
    res = _run(sc)
    text = res.report()
    assert sc.name in text
    rows = [ln for ln in text.splitlines() if ln.split()[:1] and ln.split()[0].isdigit()]
    assert [int(r.split()[0]) for r in rows] == [e.step for e in sc.events]


@pytest.mark.parametrize("frames_per_step", [1, 3])
def test_frame_indices_consistent(frames_per_step):
    sc = _by_name()["interrupt"]
    res = run(sc, _SpyModel(frames_per_step), fake_encode, FPS)
    assert res.frames.shape == (sc.steps * frames_per_step, 3, 32, 32)
    assert [r.first_frame for r in res.steps] == [i * frames_per_step for i in range(sc.steps)]
    assert all(r.num_frames == frames_per_step for r in res.steps)
    for (event, _), m in zip(res.handled, res.metrics["events"]):
        assert m["steer_frame"] == event.step * frames_per_step


def test_metrics_timing_and_save(tmp_path):
    res = _run(_by_name()["single"])
    m = res.metrics
    assert 0 <= m["generate_mean_s"] <= m["generate_max_s"]
    path = tmp_path / "f.pt"
    res.save_frames(str(path))
    assert torch.equal(torch.load(path), res.frames)


@pytest.mark.parametrize(
    "steps, events",
    [(0, ()), (5, (Event(5, _msg(BASE + WINTER)),)), (5, (Event(-1, _msg(BASE + WINTER)),))],
)
def test_invalid_scenario_raises(steps, events):
    with pytest.raises(ValueError):
        Scenario("bad", BASE, events, steps)


@pytest.mark.parametrize("ramp", [1, 2, 4, 12])
def test_default_scenarios_valid_for_any_ramp(ramp):
    names = [s.name for s in default_scenarios(ramp)]
    assert names == ["single", "hard_swap", "interrupt", "pace_1", "pace_2", "pace_3",
                     "pace_4", "pace_12", "stack_remove_middle", "early_right", "early_wrong"]
    for sc in default_scenarios(ramp):
        assert sc.events[0].step >= 2  # room for pre-steer frames


def test_cli_requires_fake(capsys):
    assert main([]) == 2
    assert "README" in capsys.readouterr().err


def _rows(emb: torch.Tensor) -> list[tuple]:
    rows = emb.reshape(-1, emb.shape[-1])
    return sorted(tuple(r.tolist()) for r in rows if r.ne(0).any())


def test_stack_remove_middle_with_aligner_ends_on_placed_prompt():
    spy = _SpyModel()
    al = Aligner(FAKE_ALIGN_LEN)
    res = run(_by_name()["stack_remove_middle"], spy, fake_encode_ids, FPS, aligner=al)
    assert [r["kind"] for _, r in res.handled] == ["final"] * 4
    plain, ids = fake_encode_ids(BASE + WINTER + RAIN)
    last = spy.conds[-1]
    assert last.shape == plain.shape
    assert _rows(last) == _rows(plain)
    assert [i for _, i in al.layout] == ids
    for p, i in al.layout:  # each token's row sits at its placed position
        assert torch.equal(last[0, p], plain[0, ids.index(i)])


def test_fake_encode_ids_rows_keyed_by_id():
    a, ia = fake_encode_ids(BASE)
    b, ib = fake_encode_ids(BASE + WINTER)
    assert len(ia) == 17 and ib[: len(ia) - 1] == ia[:-1]
    assert torch.equal(a[0, : len(ia) - 1], b[0, : len(ia) - 1])
    assert torch.count_nonzero(a[0, len(ia):]) == 0
    assert len(fake_encode_ids(BASE + WINTER + NIGHT + RAIN)[1]) <= FAKE_ALIGN_LEN


@pytest.mark.parametrize("sc", default_scenarios(), ids=lambda s: s.name)
def test_every_default_scenario_runs_with_aligner(sc):
    res = run(sc, FakeModel(), fake_encode_ids, FPS, aligner=Aligner(FAKE_ALIGN_LEN))
    assert len(res.steps) == sc.steps


def test_cli_fake_align(capsys):
    assert main(["--fake", "--align", "--ramp", "2"]) == 0
    assert "stack_remove_middle" in capsys.readouterr().out

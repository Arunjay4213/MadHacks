import random
import threading

import pytest
import torch

from steering import Steerer, blend, ease

T, D = 8, 32


def _vecs(n: int, seed: int = 0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(T, D, generator=g).to(dtype) for _ in range(n)]


# --- basic moves ------------------------------------------------------------

def test_starts_idle_at_initial_cond():
    (a,) = _vecs(1)
    s = Steerer(a)
    assert torch.equal(s.get_cond(), a)
    assert s.progress == 1.0
    assert s.is_moving is False


def test_reaches_target_exactly_after_ramp_calls_and_stays():
    a, b = _vecs(2, seed=1)
    s = Steerer(a)
    s.set_target(b, 5)
    for k in range(1, 5):
        out = s.get_cond()
        assert torch.allclose(out, blend(a, b, ease(k / 5)), atol=1e-6)
        assert not torch.equal(out, b)
    final = s.get_cond()
    assert torch.equal(final, b)
    for _ in range(3):
        assert s.get_cond() is final


def test_ramp_one_jumps_on_next_call():
    a, b = _vecs(2, seed=2)
    s = Steerer(a)
    s.set_target(b, 1)
    assert torch.equal(s.get_cond(), b)
    assert not s.is_moving


def test_set_target_takes_effect_on_next_get_cond_only():
    a, b = _vecs(2, seed=3)
    s = Steerer(a)
    s.set_target(b, 4)
    # Nothing moves until the loop asks for the next chunk.
    assert s.progress == 0.0
    assert s.is_moving
    s.get_cond()
    assert s.progress == 0.25


def test_progress_and_is_moving_through_a_move():
    a, b = _vecs(2, seed=4)
    s = Steerer(a)
    s.set_target(b, 4)
    seen = [(s.progress, s.is_moving)]
    for _ in range(5):
        s.get_cond()
        seen.append((s.progress, s.is_moving))
    assert seen == [
        (0.0, True), (0.25, True), (0.5, True), (0.75, True),
        (1.0, False), (1.0, False),
    ]


# --- interruption -----------------------------------------------------------

def test_interruption_starts_from_cond_now_and_is_continuous():
    a, b, c = _vecs(3, seed=5)
    s = Steerer(a)
    s.set_target(b, 8)
    for _ in range(3):
        mid = s.get_cond()
    s.set_target(c, 8)
    first = s.get_cond()
    # The new move is a blend from where we were, not from a or b.
    assert torch.allclose(first, blend(mid, c, ease(1 / 8)), atol=1e-6)
    step = (first - mid).norm()
    distance = (c - mid).norm()
    assert step < 0.1 * distance
    for _ in range(7):
        out = s.get_cond()
    assert torch.equal(out, c)


def test_interruption_before_any_step_restarts_cleanly():
    a, b, c = _vecs(3, seed=6)
    s = Steerer(a)
    s.set_target(b, 4)
    s.set_target(c, 2)
    s.get_cond()
    assert torch.equal(s.get_cond(), c)


# --- cap --------------------------------------------------------------------

def test_cap_holds_at_partial_blend():
    a, b = _vecs(2, seed=7)
    s = Steerer(a)
    s.set_target(b, 4, cap=0.4)
    for _ in range(4):
        out = s.get_cond()
    assert torch.allclose(out, blend(a, b, 0.4), atol=1e-6)
    assert not s.is_moving
    assert s.progress == 1.0
    for _ in range(3):
        assert s.get_cond() is out


def test_cap_never_exceeded_during_move():
    a, b = _vecs(2, seed=8)
    s = Steerer(a)
    s.set_target(b, 6, cap=0.3)
    held = blend(a, b, 0.3)
    for k in range(1, 7):
        out = s.get_cond()
        w = min(ease(k / 6), 0.3)
        assert torch.allclose(out, blend(a, b, w), atol=1e-6)
    assert torch.allclose(out, held, atol=1e-6)


def test_confirm_after_cap_continues_to_target():
    a, b = _vecs(2, seed=9)
    s = Steerer(a)
    s.set_target(b, 3, cap=0.5)
    for _ in range(3):
        held = s.get_cond()
    s.set_target(b, 4)
    first = s.get_cond()
    assert torch.allclose(first, blend(held, b, ease(1 / 4)), atol=1e-6)
    for _ in range(3):
        out = s.get_cond()
    assert torch.equal(out, b)


def test_cancel_after_cap_moves_to_other_target():
    a, b, c = _vecs(3, seed=10)
    s = Steerer(a)
    s.set_target(b, 3, cap=0.5)
    for _ in range(3):
        held = s.get_cond()
    s.set_target(c, 4)
    first = s.get_cond()
    assert torch.allclose(first, blend(held, c, ease(1 / 4)), atol=1e-6)
    for _ in range(3):
        out = s.get_cond()
    assert torch.equal(out, c)


# --- dtype / device of target -----------------------------------------------

def test_target_cast_to_cond_dtype():
    (a,) = _vecs(1, seed=11, dtype=torch.bfloat16)
    (b,) = _vecs(1, seed=12, dtype=torch.float32)
    s = Steerer(a)
    s.set_target(b, 2)
    s.get_cond()
    out = s.get_cond()
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, b.to(torch.bfloat16))


# --- validation -------------------------------------------------------------

@pytest.mark.parametrize("ramp", [0, 1.5, True])
def test_bad_ramp_raises(ramp):
    a, b = _vecs(2)
    s = Steerer(a)
    with pytest.raises(ValueError):
        s.set_target(b, ramp)


@pytest.mark.parametrize("cap", [0.0, 1.01, float("nan")])
def test_bad_cap_raises(cap):
    a, b = _vecs(2)
    s = Steerer(a)
    with pytest.raises(ValueError):
        s.set_target(b, 4, cap=cap)


def test_shape_mismatch_raises():
    (a,) = _vecs(1)
    s = Steerer(a)
    with pytest.raises(ValueError):
        s.set_target(torch.zeros(T, D + 1), 4)
    with pytest.raises(ValueError):
        s.set_target(torch.zeros(1, T, D), 4)


def test_failed_set_target_leaves_state_unchanged():
    a, b = _vecs(2, seed=13)
    s = Steerer(a)
    s.set_target(b, 2)
    with pytest.raises(ValueError):
        s.set_target(b, 0)
    s.get_cond()
    assert torch.equal(s.get_cond(), b)


# --- purity -----------------------------------------------------------------

def test_inputs_and_outputs_not_mutated():
    a, b, c = _vecs(3, seed=14)
    a0, b0, c0 = a.clone(), b.clone(), c.clone()
    s = Steerer(a)
    s.set_target(b, 3)
    returned = [s.get_cond(), s.get_cond()]
    snapshots = [r.clone() for r in returned]
    s.set_target(c, 3, cap=0.5)
    returned += [s.get_cond() for _ in range(4)]
    snapshots += [r.clone() for r in returned[2:]]
    s.set_target(c, 2)
    for _ in range(3):
        s.get_cond()
    assert torch.equal(a, a0) and torch.equal(b, b0) and torch.equal(c, c0)
    for r, snap in zip(returned, snapshots):
        assert torch.equal(r, snap)


# --- threads ----------------------------------------------------------------

def test_threaded_smoke():
    targets = _vecs(40, seed=15)
    s = Steerer(torch.zeros(T, D))
    stop = threading.Event()
    errors: list[BaseException] = []
    outputs: list[torch.Tensor] = []

    def loop() -> None:
        try:
            while not stop.is_set():
                outputs.append(s.get_cond())
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    def control() -> None:
        rng = random.Random(15)
        try:
            for t in targets:
                s.set_target(t, rng.randint(1, 6), cap=rng.choice([1.0, 0.5]))
            s.set_target(targets[-1], 6)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    gen = threading.Thread(target=loop)
    ctl = threading.Thread(target=control)
    gen.start()
    ctl.start()
    ctl.join(timeout=10)
    stop.set()
    gen.join(timeout=10)

    assert not errors
    assert outputs
    assert all(torch.isfinite(o).all() for o in outputs)
    for _ in range(6):
        final = s.get_cond()
    assert torch.equal(final, targets[-1])
    assert not s.is_moving


def test_current_does_not_advance():
    a, b = _vecs(2)
    s = Steerer(a)
    assert s.current is a
    s.set_target(b, 4)
    for _ in range(3):
        assert s.current is a  # not advanced by reading
    assert s.progress == 0.0
    out = s.get_cond()
    assert s.current is out and s.progress == 0.25

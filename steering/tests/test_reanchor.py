import dataclasses
import hashlib
import random
import threading
import time

import pytest
import torch

from steering import Controller, Steerer
from steering.reanchor import Anchor, ReanchorConfig, ReanchorPolicy

SHAPE = (1, 4, 8)
GAP = ReanchorConfig().min_gap_chunks


def _emb(prompt: str) -> torch.Tensor:
    seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "little")
    return torch.randn(SHAPE, generator=torch.Generator().manual_seed(seed))


class Engine:
    """Real Steerer + Controller + policy, driven the way the engine loop will."""

    def __init__(self, config: ReanchorConfig = ReanchorConfig(), warm: bool = True) -> None:
        self.steerer = Steerer(_emb("meadow"))
        self.controller = Controller(self.steerer, _emb)
        self.policy = ReanchorPolicy(config)
        self.chunk = 0
        if warm:  # get past the gap that follows construction
            self.run(config.min_gap_chunks)

    def send(self, msg: dict) -> dict:
        result = self.controller.handle(msg)
        self.policy.on_target(msg, result)
        return result

    def step(self) -> Anchor | None:
        self.steerer.get_cond()
        self.chunk += 1
        return self.policy.next_chunk(self.steerer.progress, self.steerer.is_moving)

    def run(self, n: int) -> list[tuple[int, Anchor]]:
        """Run n chunks; returns (chunk number relative to the call, anchor)."""
        out = []
        for k in range(1, n + 1):
            a = self.step()
            if a is not None:
                out.append((k, a))
        return out


def _msg(prompt: str, ramp: int = 4, provisional: bool = False, cap: float = 1.0,
         strength: float | None = None) -> dict:
    m = {"prompt": prompt, "ramp_chunks": ramp, "provisional": provisional, "cap": cap}
    if strength is not None:
        m.update(strength=strength, without_prompt="meadow")
    return m


def _applied(kind: str) -> dict:
    return {"applied": True, "kind": kind, "reason": None, "cache_hit": True}


def _sim(policy: ReanchorPolicy, ramp: int, n: int | None = None) -> list[tuple[int, Anchor]]:
    """Simulate a ramp's progress readings (k / ramp) and n - ramp idle chunks after it."""
    n = ramp if n is None else n
    out = []
    for k in range(1, n + 1):
        kk = min(k, ramp)
        a = policy.next_chunk(kk / ramp, kk < ramp)
        if a is not None:
            out.append((k, a))
    return out


def _warm_policy(config: ReanchorConfig = ReanchorConfig()) -> ReanchorPolicy:
    p = ReanchorPolicy(config)
    for _ in range(config.min_gap_chunks):
        assert p.next_chunk(1.0, False) is None
    return p


# --- config -----------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"steer_strength": 0.0},
        {"steer_strength": 1.1},
        {"steer_strength": float("nan")},
        {"steer_strength": True},
        {"sudden_strength": -0.1},
        {"slow_strength": 2.0},
        {"refresh_strength": 0.0},
        {"fire_at_progress": 0.0},
        {"fire_at_progress": 1.01},
        {"provisional_strength": -0.1},
        {"provisional_strength": 1.5},
        {"min_strength": -0.01},
        {"min_strength": 1.2},
        {"sudden_max_ramp": -1},
        {"sudden_max_ramp": 2.0},
        {"slow_min_ramp": 0},
        {"slow_min_ramp": 2},  # not above sudden_max_ramp
        {"slow_anchors": 0},
        {"slow_anchors": True},
        {"refresh_every_chunks": 0},
        {"refresh_every_chunks": 60.0},
        {"min_gap_chunks": 0},
        {"min_gap_chunks": "8"},
    ],
)
def test_bad_config_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        ReanchorConfig(**kwargs)


def test_good_configs() -> None:
    ReanchorConfig()
    ReanchorConfig(provisional_strength=0.0, min_strength=0.0, sudden_max_ramp=0,
                   refresh_every_chunks=None, min_gap_chunks=1, fire_at_progress=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ReanchorConfig().min_gap_chunks = 3  # type: ignore[misc]


def test_initial_state() -> None:
    p = ReanchorPolicy()
    assert p.chunks_since_anchor == 0
    assert not p.pending


# --- rule 1: unapplied results ----------------------------------------------

@pytest.mark.parametrize("reason", ["stale", "unchanged"])
def test_unapplied_result_is_ignored(reason: str) -> None:
    p = _warm_policy()
    p.on_target(_msg("winter", 1), {"applied": False, "kind": None, "reason": reason,
                                    "cache_hit": True})
    assert not p.pending
    assert _sim(p, 1, 20) == []


def test_unchanged_leaves_pending_alone() -> None:
    e = Engine()
    assert e.send(_msg("winter", 4))["kind"] == "final"
    e.step()
    assert e.send(_msg("winter", 4))["reason"] == "unchanged"
    assert e.policy.pending
    assert e.run(4) == [(1, Anchor(0.85, "steer"))]  # the original ramp's chunk 2


# --- rule 2: final / confirm strengths --------------------------------------

@pytest.mark.parametrize(
    "ramp, strength",
    [(1, 1.0), (2, 1.0), (3, 0.85), (4, 0.85), (9, 0.85), (10, 0.5), (40, 0.5)],
)
def test_base_strength_by_ramp(ramp: int, strength: float) -> None:
    e = Engine(ReanchorConfig(min_gap_chunks=1))
    e.send(_msg("winter", ramp))
    anchors = e.run(ramp + 2)
    assert anchors and all(a == Anchor(strength, "steer") for _, a in anchors)
    assert len(anchors) == (3 if ramp >= 10 else 1)


@pytest.mark.parametrize(
    "ramp, msg_strength, expected",
    [
        (4, 0.3, 0.85 * 0.3),
        (4, 1.0, 0.85),
        (4, 1.3, 0.85),  # capped at 1.0
        (1, 0.5, 0.5),
        (1, 1.3, 1.0),
        (12, 0.3, 0.5 * 0.3),
    ],
)
def test_message_strength_scales(ramp: int, msg_strength: float, expected: float) -> None:
    e = Engine()
    e.send(_msg("winter", ramp, strength=msg_strength))
    anchors = e.run(ramp + GAP)
    assert anchors
    for _, a in anchors:
        assert a.reason == "steer"
        assert a.strength == pytest.approx(expected)


def test_scaled_below_min_strength_schedules_nothing() -> None:
    e = Engine()
    e.send(_msg("winter", 4, strength=0.05))  # 0.85 * 0.05 < 0.05
    assert not e.policy.pending
    assert e.run(20) == []


def test_confirm_schedules_a_steer() -> None:
    e = Engine()
    e.send(_msg("winter", 4, provisional=True, cap=0.4))
    assert not e.policy.pending  # provisional_strength 0
    e.step()
    assert e.send(_msg("winter", 4))["kind"] == "confirm"
    assert e.run(6) == [(2, Anchor(0.85, "steer"))]


# --- rule 3: provisional ----------------------------------------------------

def test_provisional_never_anchors_by_default() -> None:
    e = Engine()
    e.send(_msg("winter", 1, provisional=True, cap=0.4))
    assert e.run(30) == []


def test_provisional_strength_times_cap() -> None:
    cfg = ReanchorConfig(provisional_strength=0.4)
    e = Engine(cfg)
    e.send(_msg("winter", 4, provisional=True, cap=0.5))
    assert e.run(4) == [(2, Anchor(pytest.approx(0.2), "steer"))]
    # Same cap below min_strength: skipped.
    e2 = Engine(cfg)
    e2.send(_msg("winter", 4, provisional=True, cap=0.1))  # 0.04 < 0.05
    assert not e2.policy.pending and e2.run(10) == []


def test_provisional_then_confirm_replaces() -> None:
    e = Engine(ReanchorConfig(provisional_strength=0.4))
    e.send(_msg("winter", 4, provisional=True, cap=0.5))
    e.step()  # progress 0.25: not fired
    e.send(_msg("winter", 4))
    assert e.run(6) == [(2, Anchor(0.85, "steer"))]


# --- rule 4: single-anchor timing with real progress ------------------------

@pytest.mark.parametrize(
    "ramp, fire_at, chunks",
    [
        (1, 0.5, [1]),
        (2, 0.5, [1]),
        (4, 0.5, [2]),
        (5, 0.5, [3]),  # 2/5 < 0.5 <= 3/5
        (4, 0.75, [3]),
        (4, 1.0, [4]),
        (2, 1.0, [2]),
        (12, 0.5, [3, 11]),  # slow: 0.25 at chunk 3; 0.5 and 0.75 deferred by the gap, fire once
    ],
)
def test_fires_at_real_progress(ramp: int, fire_at: float, chunks: list[int]) -> None:
    e = Engine(ReanchorConfig(fire_at_progress=fire_at))
    e.send(_msg("winter", ramp))
    assert [k for k, _ in e.run(ramp + 20)] == chunks
    assert not e.policy.pending


def test_progress_at_fire_matches_steerer() -> None:
    for ramp in (1, 2, 4):
        e = Engine()
        e.send(_msg("winter", ramp))
        seen = []
        for _ in range(ramp):
            a = e.step()
            seen.append((e.steerer.progress, a is not None))
        fired = [p for p, f in seen if f]
        assert fired == [min(p for p, _ in seen if p >= 0.5)]


def test_latest_target_wins() -> None:
    e = Engine()
    e.send(_msg("winter", 4))
    e.step()  # 0.25
    e.send(_msg("night", 6, strength=0.5))
    anchors = e.run(10)
    assert anchors == [(3, Anchor(pytest.approx(0.425), "steer"))]  # 3/6 of the new ramp


def test_applied_target_that_schedules_nothing_cancels_pending() -> None:
    e = Engine()
    e.send(_msg("winter", 4))
    e.step()
    e.send(_msg("snow", 4, provisional=True, cap=0.4))  # applied, schedules nothing
    assert not e.policy.pending
    assert e.run(20) == []


# --- rule 5: slow steers ----------------------------------------------------

@pytest.mark.parametrize(
    "n, ramp, chunks",
    [
        (3, 12, [3, 6, 9]),
        (3, 10, [3, 5, 8]),  # 0.3, 0.5, 0.8
        (4, 20, [4, 8, 12, 16]),  # i/5
        (1, 10, [5]),
        (2, 30, [10, 20]),
    ],
)
def test_slow_thresholds_without_gap(n: int, ramp: int, chunks: list[int]) -> None:
    e = Engine(ReanchorConfig(slow_anchors=n, min_gap_chunks=1))
    e.send(_msg("winter", ramp, strength=0.6))
    anchors = e.run(ramp + 5)
    assert [k for k, _ in anchors] == chunks
    assert all(a.strength == pytest.approx(0.3) and a.reason == "steer" for _, a in anchors)


def test_slow_long_ramp_gets_all_anchors_with_gap() -> None:
    e = Engine()
    e.send(_msg("winter", 40))
    assert [k for k, _ in e.run(45)] == [10, 20, 30]


def test_slow_thresholds_landing_on_one_chunk_fire_once() -> None:
    p = _warm_policy(ReanchorConfig(min_gap_chunks=1))
    p.on_target(_msg("winter", 12), _applied("final"))
    assert p.next_chunk(1 / 12, True) is None
    # A reading that passes two thresholds at once (0.25, 0.5) fires once.
    assert p.next_chunk(6 / 12, True) is None  # capped by chunks seen: 2/12
    p2 = _warm_policy(ReanchorConfig(min_gap_chunks=1, slow_anchors=3))
    p2.on_target(_msg("winter", 10), _applied("final"))
    got = [p2.next_chunk(k / 10, k < 10) for k in range(1, 11)]
    assert [k + 1 for k, a in enumerate(got) if a] == [3, 5, 8]


# --- rule 6: min gap --------------------------------------------------------

def test_no_steer_within_gap_of_construction() -> None:
    e = Engine(warm=False)
    e.send(_msg("winter", 1))
    assert e.run(20) == [(GAP, Anchor(1.0, "steer"))]


def test_steer_blocked_by_gap_is_deferred_not_dropped() -> None:
    e = Engine()
    e.send(_msg("winter", 1))
    assert e.run(1) == [(1, Anchor(1.0, "steer"))]
    e.run(2)
    e.send(_msg("night", 2, strength=0.7))
    assert e.run(20) == [(GAP - 2, Anchor(pytest.approx(0.7), "steer"))]
    assert e.policy.chunks_since_anchor == 20 - (GAP - 2)


def test_steer_after_refresh_waits_gap() -> None:
    cfg = ReanchorConfig(refresh_every_chunks=20)
    e = Engine(cfg)
    anchors = e.run(20 - GAP)
    assert anchors == [(20 - GAP, Anchor(0.25, "refresh"))]
    e.send(_msg("winter", 1))
    assert e.run(20) == [(GAP, Anchor(1.0, "steer"))]


# --- rule 7: refresh --------------------------------------------------------

def test_long_idle_refresh_cadence_is_exact() -> None:
    p = ReanchorPolicy()
    chunks = [k for k in range(1, 6001) if p.next_chunk(1.0, False) is not None]
    assert chunks == list(range(60, 6001, 60))


def test_refresh_anchor_value_and_disable() -> None:
    p = ReanchorPolicy(ReanchorConfig(refresh_every_chunks=5, refresh_strength=0.3,
                                      min_gap_chunks=1))
    got = [p.next_chunk(1.0, False) for _ in range(5)]
    assert got == [None] * 4 + [Anchor(0.3, "refresh")]
    off = ReanchorPolicy(ReanchorConfig(refresh_every_chunks=None))
    assert all(off.next_chunk(1.0, False) is None for _ in range(1000))


def test_refresh_never_shorter_than_gap() -> None:
    p = ReanchorPolicy(ReanchorConfig(refresh_every_chunks=3, min_gap_chunks=8))
    chunks = [k for k in range(1, 41) if p.next_chunk(1.0, False)]
    assert chunks == [8, 16, 24, 32, 40]


def test_no_refresh_while_moving() -> None:
    p = ReanchorPolicy()
    assert all(p.next_chunk(0.5, True) is None for _ in range(200))
    assert p.next_chunk(1.0, False) == Anchor(0.25, "refresh")


def test_no_refresh_while_steer_pending() -> None:
    p = ReanchorPolicy(ReanchorConfig(refresh_every_chunks=10))
    p.on_target(_msg("winter", 400), _applied("final"))  # slow, first threshold at 100
    got = [p.next_chunk(k / 400, True) for k in range(1, 100)]
    assert all(a is None for a in got)
    assert p.pending


def test_steer_restarts_refresh_count() -> None:
    e = Engine()
    e.run(30)
    e.send(_msg("winter", 4))
    anchors = e.run(200)
    assert [(k, a.reason) for k, a in anchors] == [
        (2, "steer"), (62, "refresh"), (122, "refresh"), (182, "refresh")]


# --- rule 8: reset ----------------------------------------------------------

def test_reset_clears_pending_and_restarts_counts() -> None:
    e = Engine()
    e.run(50)
    e.send(_msg("winter", 4))
    e.step()
    e.controller.reset("desert")
    e.policy.on_reset()
    assert not e.policy.pending
    assert e.policy.chunks_since_anchor == 0
    assert [k for k, _ in e.run(130)] == [60, 120]  # refresh restarts from the reset


def test_steer_right_after_reset_waits_gap() -> None:
    e = Engine()
    e.policy.on_reset()
    e.send(_msg("winter", 1))
    assert e.run(GAP + 3) == [(GAP, Anchor(1.0, "steer"))]


# --- rule 9: ramp finished before the threshold -----------------------------

def test_progress_jump_to_finished_fires_next_chunk() -> None:
    p = _warm_policy(ReanchorConfig(fire_at_progress=0.9))
    p.on_target(_msg("winter", 8), _applied("final"))
    assert p.next_chunk(1 / 8, True) is None
    assert p.next_chunk(1.0, False) == Anchor(0.85, "steer")
    assert not p.pending


def test_slow_thresholds_deferred_by_gap_collapse() -> None:
    e = Engine()
    e.send(_msg("winter", 1))
    e.step()  # anchor; gap starts
    e.send(_msg("night", 10))
    anchors = e.run(30)
    # 0.25 (chunk 3), 0.5 (5) and 0.75 (8) all wait for the gap, which ends at
    # chunk 8: they fire once there, and nothing is left.
    assert anchors == [(GAP, Anchor(0.5, "steer"))]
    assert not e.policy.pending


def test_slow_leftover_fires_after_ramp_end() -> None:
    p = _warm_policy(ReanchorConfig(min_gap_chunks=4))
    p.on_target(_msg("winter", 12), _applied("final"))
    got = [p.next_chunk(min(k, 12) / 12, k < 12) for k in range(1, 20)]
    # 0.25 at 3; 0.5 due at 6 but the gap ends at 7; 0.75 due at 9, fires at 11;
    # nothing left by the ramp end.
    assert [k + 1 for k, a in enumerate(got) if a] == [3, 7, 11]
    p2 = _warm_policy(ReanchorConfig(min_gap_chunks=8))
    p2.on_target(_msg("winter", 12), _applied("final"))
    got2 = [p2.next_chunk(1 / 12, True), p2.next_chunk(2 / 12, True),
            p2.next_chunk(1.0, False), p2.next_chunk(1.0, False)]
    # Progress jumps to the end: all three thresholds fire once, on the next chunk.
    assert got2 == [None, None, Anchor(0.5, "steer"), None]
    assert not p2.pending


def test_stale_finished_reading_on_first_chunk_is_ignored() -> None:
    p = _warm_policy()
    p.on_target(_msg("winter", 4), _applied("final"))
    # Progress read before the steerer saw the new target (old move finished).
    assert p.next_chunk(1.0, False) is None
    assert p.next_chunk(1 / 4, True) is None
    assert p.next_chunk(2 / 4, True) == Anchor(0.85, "steer")


def test_progress_ahead_of_chunks_seen_is_not_trusted() -> None:
    p = _warm_policy()
    p.on_target(_msg("winter", 8), _applied("final"))
    assert p.next_chunk(0.9, True) is None  # 1/8 chunks seen
    assert p.next_chunk(0.9, True) is None
    assert p.next_chunk(0.9, True) is None
    assert p.next_chunk(0.9, True) == Anchor(0.85, "steer")  # 4/8


# --- rule 10 and fuzz -------------------------------------------------------

def test_strength_always_in_unit_interval_and_gap_holds() -> None:
    rng = random.Random(7)
    for trial in range(30):
        cfg = ReanchorConfig(
            steer_strength=rng.uniform(0.01, 1.0),
            sudden_strength=rng.uniform(0.01, 1.0),
            slow_strength=rng.uniform(0.01, 1.0),
            provisional_strength=rng.choice([0.0, rng.uniform(0.01, 1.0)]),
            refresh_strength=rng.uniform(0.01, 1.0),
            refresh_every_chunks=rng.choice([None, rng.randint(1, 50)]),
            min_gap_chunks=rng.randint(1, 10),
            slow_anchors=rng.randint(1, 5),
            fire_at_progress=rng.uniform(0.05, 1.0),
        )
        e = Engine(cfg)
        last = None
        for chunk in range(400):
            if rng.random() < 0.2:
                prov = rng.random() < 0.4
                e.send(_msg(rng.choice("abcdef"), rng.randint(1, 20), provisional=prov,
                            cap=rng.uniform(0.1, 1.0) if prov else 1.0,
                            strength=rng.choice([None, rng.uniform(0.01, 1.3)])))
            if rng.random() < 0.01:
                e.controller.reset("meadow")
                e.policy.on_reset()
                last = chunk - 1  # a reset counts like an anchor just before this chunk
            a = e.step()
            if a is not None:
                assert 0.0 < a.strength <= 1.0
                assert a.reason in ("steer", "refresh")
                if last is not None:
                    assert chunk - last >= cfg.min_gap_chunks, trial
                last = chunk


# --- interrupt storms -------------------------------------------------------

def test_storm_of_finals_never_fires_mid_ramp_then_fires_once() -> None:
    e = Engine()
    for k in range(100):
        e.send(_msg("ab"[k % 2] + "x", 4))
        assert e.step() is None  # each ramp is replaced before reaching 0.5
    e.send(_msg("calm", 4))
    assert e.run(30) == [(2, Anchor(0.85, "steer"))]


@pytest.mark.parametrize("ramp", [1, 2, 3])
def test_storm_respects_gap_and_fires_after(ramp: int) -> None:
    e = Engine()
    storm = []
    for k in range(100):
        e.send(_msg("ab"[k % 2] + "x", ramp))
        if e.step() is not None:
            storm.append(e.chunk)
    # Ramps of 1-2 fire on their first chunk, so the gap is the only limit;
    # a ramp of 3 is always replaced before reaching 0.5.
    assert len(storm) == (len(range(1, 101, GAP)) if ramp <= 2 else 0)
    e.send(_msg("calm", ramp))
    after = []
    for _ in range(GAP + ramp):
        if e.step() is not None:
            after.append(e.chunk)
    assert len(after) == 1
    allc = storm + after
    assert all(b - a >= GAP for a, b in zip(allc, allc[1:]))
    assert not e.policy.pending


# --- threads ----------------------------------------------------------------

def test_two_thread_hammer() -> None:
    cfg = ReanchorConfig(provisional_strength=0.3, refresh_every_chunks=20)
    steerer = Steerer(_emb("meadow"))
    controller = Controller(steerer, _emb)
    policy = ReanchorPolicy(cfg)
    stop = threading.Event()
    errors: list[BaseException] = []
    anchors: list[tuple[int, Anchor]] = []
    resets: list[int] = []
    chunk = [0]
    lock = threading.Lock()

    def control() -> None:
        rng = random.Random(1)
        try:
            while not stop.is_set():
                if rng.random() < 0.002:
                    controller.reset("meadow")
                    with lock:
                        policy.on_reset()
                        resets.append(chunk[0])
                    continue
                prov = rng.random() < 0.3
                msg = _msg(rng.choice("abcdefgh"), rng.randint(1, 16), provisional=prov,
                           cap=0.5 if prov else 1.0,
                           strength=rng.choice([None, 0.5, 1.2]))
                policy.on_target(msg, controller.handle(msg))
                _ = policy.pending, policy.chunks_since_anchor
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def generate() -> None:
        try:
            while not stop.is_set():
                steerer.get_cond()
                a = policy.next_chunk(steerer.progress, steerer.is_moving)
                with lock:
                    chunk[0] += 1
                    if a is not None:
                        assert 0.0 < a.strength <= 1.0
                        anchors.append((chunk[0], a))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=control), threading.Thread(target=generate)]
    for t in threads:
        t.start()
    time.sleep(1.0)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    assert not errors, errors
    assert chunk[0] > 100
    assert anchors
    nums = [n for n, _ in anchors]
    assert all(b - a >= cfg.min_gap_chunks for a, b in zip(nums, nums[1:]))

import hashlib
import random
import threading

import pytest
import torch

from steering import Aligner, Controller, Steerer, blend, ease

SHAPE = (1, 8, 16)


class FakeEncode:
    """Deterministic random tensor per prompt; counts calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> torch.Tensor:
        self.calls.append(prompt)
        seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "little")
        return torch.randn(SHAPE, generator=torch.Generator().manual_seed(seed))


def _emb(prompt: str) -> torch.Tensor:
    return FakeEncode()(prompt)


def _setup(cache_size: int = 64):
    enc = FakeEncode()
    s = Steerer(_emb("meadow"))
    return s, enc, Controller(s, enc, cache_size=cache_size)


def _msg(prompt: str, ramp: int = 4, provisional: bool = False, cap: float = 1.0) -> dict:
    return {"prompt": prompt, "ramp_chunks": ramp, "provisional": provisional, "cap": cap}


def _run(s: Steerer, n: int) -> list[torch.Tensor]:
    return [s.get_cond() for _ in range(n)]


# --- validation -------------------------------------------------------------

@pytest.mark.parametrize(
    "msg",
    [
        {"ramp_chunks": 4},
        {"prompt": "", "ramp_chunks": 4},
        {"prompt": 3, "ramp_chunks": 4},
        {"prompt": "x"},
        {"prompt": "x", "ramp_chunks": 0},
        {"prompt": "x", "ramp_chunks": 2.0},
        {"prompt": "x", "ramp_chunks": True},
        {"prompt": "x", "ramp_chunks": 4, "provisional": "yes"},
        {"prompt": "x", "ramp_chunks": 4, "provisional": 1},
        {"prompt": "x", "ramp_chunks": 4, "provisional": True, "cap": 0.0},
        {"prompt": "x", "ramp_chunks": 4, "provisional": True, "cap": 1.5},
        {"prompt": "x", "ramp_chunks": 4, "provisional": True, "cap": float("nan")},
        {"prompt": "x", "ramp_chunks": 4, "provisional": True, "cap": True},
        {"prompt": "x", "ramp_chunks": 4, "provisional": True, "cap": "0.4"},
        {"prompt": "x", "ramp_chunks": 4, "provisional": False, "cap": 0.4},
        {"prompt": "x", "ramp_chunks": 4, "cap": 0.4},
    ],
)
def test_bad_message_raises_and_changes_nothing(msg):
    s, enc, c = _setup()
    c.handle(_msg("winter", 2))
    c.handle(_msg("snow", 4, provisional=True, cap=0.4))
    s.get_cond()
    with pytest.raises(ValueError):
        c.handle(msg)
    assert enc.calls == ["winter", "snow"]
    assert c.committed_prompt == "winter"
    assert c.pending_prompt == "snow"
    assert s.progress == 0.25


def test_defaults_for_provisional_and_cap():
    s, _, c = _setup()
    r = c.handle({"prompt": "winter", "ramp_chunks": 1})
    assert r["kind"] == "final"
    assert torch.equal(s.get_cond(), _emb("winter"))


def test_initial_state():
    _, _, c = _setup()
    assert c.committed_prompt is None
    assert c.pending_prompt is None


def test_bad_cache_size_raises():
    with pytest.raises(ValueError):
        Controller(Steerer(_emb("a")), FakeEncode(), cache_size=0)


# --- provisional / confirm / final ------------------------------------------

def test_provisional_cap_is_respected():
    s, _, c = _setup()
    start, guess = _emb("meadow"), _emb("winter")
    r = c.handle(_msg("winter", 4, provisional=True, cap=0.4))
    assert r == {"applied": True, "kind": "provisional", "reason": None, "cache_hit": False}
    assert c.pending_prompt == "winter"
    assert c.committed_prompt is None
    for k, out in enumerate(_run(s, 6), start=1):
        w = min(ease(min(k, 4) / 4), 0.4)
        assert torch.allclose(out, blend(start, guess, w), atol=1e-6)
    assert torch.allclose(s.get_cond(), blend(start, guess, 0.4), atol=1e-6)


def test_provisional_then_matching_final_reaches_target():
    s, _, c = _setup()
    c.handle(_msg("winter", 4, provisional=True, cap=0.4))
    _run(s, 2)
    r = c.handle(_msg("winter", 3))
    assert r["kind"] == "confirm" and r["applied"] and r["cache_hit"]
    assert c.pending_prompt is None
    assert c.committed_prompt == "winter"
    outs = _run(s, 3)
    assert not torch.equal(outs[1], _emb("winter"))
    assert torch.equal(outs[2], _emb("winter"))


def test_provisional_then_different_final_reaches_real_target():
    s, _, c = _setup()
    calls = []
    orig = s.set_target
    s.set_target = lambda t, r, cap=1.0: (calls.append((t, cap)), orig(t, r, cap))[1]
    start, guess, real = _emb("meadow"), _emb("winter"), _emb("night")
    c.handle(_msg("winter", 4, provisional=True, cap=0.4))
    _run(s, 3)
    held = s.get_cond()
    limit = blend(start, guess, 0.4)
    assert torch.allclose(held, limit, atol=1e-6)
    r = c.handle(_msg("night", 4))
    assert r["kind"] == "final" and r["applied"]
    assert c.pending_prompt is None and c.committed_prompt == "night"
    # The guess was only ever targeted with the cap; then straight to the real one.
    assert [(torch.equal(t, guess), torch.equal(t, real), cap) for t, cap in calls] == [
        (True, False, 0.4), (False, True, 1.0)]
    outs = _run(s, 4)
    for k, out in enumerate(outs, start=1):
        # Every step is on the path from the held (capped) point to the real target.
        assert torch.allclose(out, blend(held, real, ease(k / 4)), atol=1e-6)
    assert torch.equal(outs[-1], real)


def test_final_without_provisional():
    s, _, c = _setup()
    r = c.handle(_msg("winter", 2))
    assert r["kind"] == "final"
    assert torch.equal(_run(s, 2)[-1], _emb("winter"))


def test_new_provisional_replaces_pending_guess():
    s, _, c = _setup()
    c.handle(_msg("winter", 2, provisional=True, cap=0.4))
    _run(s, 2)
    r = c.handle(_msg("night", 2, provisional=True, cap=0.4))
    assert r["kind"] == "provisional"
    assert c.pending_prompt == "night"


# --- no-ops -----------------------------------------------------------------

def test_final_same_as_committed_is_unchanged():
    s, enc, c = _setup()
    c.handle(_msg("winter", 4))
    s.get_cond()
    r = c.handle(_msg("winter", 4))
    assert r == {"applied": False, "kind": None, "reason": "unchanged", "cache_hit": True}
    assert s.progress == 0.25  # running move not restarted


def test_provisional_same_as_committed_is_unchanged():
    s, _, c = _setup()
    c.handle(_msg("winter", 1))
    s.get_cond()
    r = c.handle(_msg("winter", 4, provisional=True, cap=0.4))
    assert r["applied"] is False and r["reason"] == "unchanged"
    assert c.pending_prompt is None
    assert not s.is_moving


def test_repeated_provisional_is_unchanged():
    s, _, c = _setup()
    c.handle(_msg("winter", 2, provisional=True, cap=0.4))
    held = _run(s, 2)[-1]
    r = c.handle(_msg("winter", 2, provisional=True, cap=0.4))
    assert r["reason"] == "unchanged"
    assert s.get_cond() is held


def test_final_back_to_committed_while_guess_pending_is_applied():
    s, _, c = _setup()
    c.handle(_msg("winter", 1))
    s.get_cond()
    c.handle(_msg("night", 2, provisional=True, cap=0.4))
    _run(s, 2)
    r = c.handle(_msg("winter", 2))
    assert r["applied"] and r["kind"] == "final"
    assert c.pending_prompt is None
    assert torch.equal(_run(s, 2)[-1], _emb("winter"))


# --- cache ------------------------------------------------------------------

def test_cache_hit_and_miss():
    _, enc, c = _setup()
    assert c.handle(_msg("winter"))["cache_hit"] is False
    assert c.handle(_msg("night"))["cache_hit"] is False
    assert c.handle(_msg("winter"))["cache_hit"] is True
    assert enc.calls == ["winter", "night"]


def test_lru_eviction():
    _, enc, c = _setup(cache_size=2)
    c.precompute(["a", "b"])
    c.handle(_msg("a"))           # hit; "a" is now most recent
    c.handle(_msg("c"))           # miss; evicts "b"
    assert enc.calls == ["a", "b", "c"]
    assert c.handle(_msg("a"))["cache_hit"] is True
    assert c.handle(_msg("b"))["cache_hit"] is False
    assert enc.calls == ["a", "b", "c", "b"]


def test_precompute_avoids_encode_at_handle_time():
    s, enc, c = _setup()
    c.precompute(["winter", "night"])
    assert enc.calls == ["winter", "night"]
    r1 = c.handle(_msg("winter", 2, provisional=True, cap=0.4))
    r2 = c.handle(_msg("night", 2))
    assert r1["cache_hit"] and r2["cache_hit"]
    assert enc.calls == ["winter", "night"]


def test_cached_tensors_not_modified():
    s, _, c = _setup()
    c.precompute(["winter", "night"])
    w0, n0 = _emb("winter"), _emb("night")
    c.handle(_msg("winter", 2, provisional=True, cap=0.5))
    _run(s, 3)
    c.handle(_msg("night", 2))
    _run(s, 3)
    assert torch.equal(c._embedding("winter")[0], w0)
    assert torch.equal(c._embedding("night")[0], n0)


# --- errors from encode / steerer ------------------------------------------

def test_encode_raising_leaves_state_untouched():
    s, enc, c = _setup()
    c.handle(_msg("winter", 2))
    s.get_cond()

    def boom(prompt: str) -> torch.Tensor:
        raise RuntimeError("encoder down")

    c._encode = boom
    with pytest.raises(RuntimeError):
        c.handle(_msg("night", 2))
    assert c.committed_prompt == "winter" and c.pending_prompt is None
    assert s.progress == 0.5
    assert torch.equal(s.get_cond(), _emb("winter"))


def test_steerer_rejecting_embedding_leaves_state_untouched():
    s = Steerer(_emb("meadow"))
    c = Controller(s, lambda p: torch.zeros(2, 2))
    with pytest.raises(ValueError):
        c.handle(_msg("winter"))
    assert c.committed_prompt is None and c.pending_prompt is None
    assert not s.is_moving


# --- threads ----------------------------------------------------------------

def test_stale_message_is_dropped():
    s = Steerer(_emb("meadow"))
    release = threading.Event()
    entered = threading.Event()
    base = FakeEncode()

    def encode(prompt: str) -> torch.Tensor:
        if prompt == "slow":
            entered.set()
            assert release.wait(5)
        return base(prompt)

    c = Controller(s, encode)
    results: dict[str, dict] = {}
    old = threading.Thread(target=lambda: results.update(old=c.handle(_msg("slow", 1))))
    old.start()
    assert entered.wait(5)                     # older message is mid-encode
    results["new"] = c.handle(_msg("fast", 1))  # newer one starts later, lands first
    release.set()
    old.join(5)

    assert results["new"]["applied"] is True
    assert results["old"] == {"applied": False, "kind": None, "reason": "stale",
                              "cache_hit": False}
    assert c.committed_prompt == "fast"
    assert torch.equal(s.get_cond(), _emb("fast"))
    # The slow embedding was still cached, so it costs nothing next time.
    assert c.handle(_msg("slow", 1))["cache_hit"] is True


def test_threaded_smoke():
    prompts = [f"p{i}" for i in range(12)]
    s = Steerer(_emb("meadow"))
    c = Controller(s, FakeEncode(), cache_size=8)
    stop = threading.Event()
    errors: list[BaseException] = []

    def loop() -> None:
        try:
            while not stop.is_set():
                assert torch.isfinite(s.get_cond()).all()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    def control(seed: int) -> None:
        rng = random.Random(seed)
        try:
            for _ in range(60):
                p = rng.choice(prompts)
                if rng.random() < 0.5:
                    c.handle(_msg(p, rng.randint(1, 4), provisional=True, cap=0.4))
                else:
                    c.handle(_msg(p, rng.randint(1, 4)))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    gen = threading.Thread(target=loop)
    ctls = [threading.Thread(target=control, args=(i,)) for i in range(3)]
    gen.start()
    for t in ctls:
        t.start()
    for t in ctls:
        t.join(10)
    r = c.handle(_msg("final-target", 4))
    stop.set()
    gen.join(10)

    assert not errors
    assert r["applied"]
    assert c.committed_prompt == "final-target" and c.pending_prompt is None
    _run(s, 4)
    assert torch.equal(s.get_cond(), _emb("final-target"))


# --- aligner ----------------------------------------------------------------

AL_LEN, AL_DIM = 16, 8
_VOCAB = torch.randn(64, AL_DIM, generator=torch.Generator().manual_seed(3))


def _word_ids(prompt: str) -> list[int]:
    """One id per word (stable), plus an end token 0."""
    return [1 + hashlib.sha256(w.encode()).digest()[0] % 63 for w in prompt.split()] + [0]


class PairEncode:
    """(emb [1, AL_LEN, AL_DIM], ids): one fixed row per id, padding zeroed."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, prompt: str):
        self.calls.append(prompt)
        ids = _word_ids(prompt)
        emb = torch.zeros(1, AL_LEN, AL_DIM)
        emb[0, : len(ids)] = _VOCAB[ids]
        return emb, ids


def _aligned_setup(start: str = "oak tree wind"):
    enc = PairEncode()
    al = Aligner(AL_LEN)
    s = Steerer(al.place(*enc(start)))
    enc.calls.clear()
    return s, enc, al, Controller(s, enc, aligner=al)


def test_aligner_places_target():
    s, enc, al, c = _aligned_setup()
    r = c.handle(_msg("oak tree snow wind", 1))
    assert r["applied"] and r["kind"] == "final"
    out = s.get_cond()[0]
    ref = Aligner(AL_LEN)
    ref.place(*PairEncode()("oak tree wind"))
    assert torch.equal(out, ref.place(*PairEncode()("oak tree snow wind"))[0])
    # "wind" and the end token keep positions 2 and 3; "snow" takes position 4.
    ids = _word_ids("oak tree snow wind")
    assert [p for p, _ in al.layout] == [0, 1, 4, 2, 3]
    assert torch.equal(out[2], _VOCAB[ids[3]]) and torch.equal(out[4], _VOCAB[ids[2]])


def test_aligner_cache_stores_raw_pair():
    s, enc, al, c = _aligned_setup()
    c.handle(_msg("oak tree snow wind", 1))
    emb, ids = c._cache["oak tree snow wind"]
    raw_emb, raw_ids = PairEncode()("oak tree snow wind")
    assert torch.equal(emb, raw_emb) and list(ids) == raw_ids
    # Same prompt later from a different layout: placed afresh, not reused.
    c.handle(_msg("snow", 1))
    r = c.handle(_msg("oak tree snow wind", 1))
    assert r["cache_hit"] and enc.calls == ["oak tree snow wind", "snow"]


def test_aligner_not_advanced_by_unchanged():
    s, enc, al, c = _aligned_setup()
    c.handle(_msg("oak tree snow wind", 1))
    before = al.layout
    assert c.handle(_msg("oak tree snow wind", 1))["reason"] == "unchanged"
    assert al.layout == before
    c.handle(_msg("night", 2, provisional=True, cap=0.4))
    before = al.layout
    assert c.handle(_msg("night", 2, provisional=True, cap=0.4))["reason"] == "unchanged"
    assert al.layout == before


def test_aligner_not_advanced_by_stale():
    al = Aligner(AL_LEN)
    base = PairEncode()
    s = Steerer(al.place(*base("oak tree wind")))
    release, entered = threading.Event(), threading.Event()

    def encode(prompt: str):
        if prompt == "slow":
            entered.set()
            assert release.wait(5)
        return base(prompt)

    c = Controller(s, encode, aligner=al)
    results: dict[str, dict] = {}
    old = threading.Thread(target=lambda: results.update(old=c.handle(_msg("slow", 1))))
    old.start()
    assert entered.wait(5)
    c.handle(_msg("oak fast wind", 1))
    after_new = al.layout
    release.set()
    old.join(5)
    assert results["old"]["reason"] == "stale"
    assert al.layout == after_new


def test_aligner_unchanged_when_set_target_fails():
    s, enc, al, c = _aligned_setup()
    c.handle(_msg("oak tree snow wind", 1))
    before = al.layout

    def boom(*args, **kwargs):
        raise ValueError("rejected")

    s.set_target = boom
    with pytest.raises(ValueError):
        c.handle(_msg("night rain", 1))
    assert al.layout == before
    assert c.committed_prompt == "oak tree snow wind"


def test_aligner_requires_pair_from_encode():
    al = Aligner(AL_LEN)
    s = Steerer(al.place(*PairEncode()("oak")))
    c = Controller(s, lambda p: torch.zeros(1, AL_LEN, AL_DIM), aligner=al)
    with pytest.raises(TypeError, match="token_ids"):
        c.handle(_msg("winter"))
    assert c._cache == {} and c.committed_prompt is None


def test_aligner_without_initial_layout_raises():
    al = Aligner(AL_LEN)
    with pytest.raises(ValueError, match="place"):
        Controller(Steerer(torch.zeros(1, AL_LEN, AL_DIM)), PairEncode(), aligner=al)


def test_aligner_new_tokens_avoid_rows_still_fading():
    # Commit C, then a capped guess A moves part way: C's rows are still
    # non-zero but no longer in the aligner's (A's) layout. B's new tokens
    # must not land on them (or on A's).
    c_prompt, a_prompt, b_prompt = "oak tree wind", "snow night rain", "fog mist haze dawn"
    words = " ".join((c_prompt, a_prompt, b_prompt)).split()
    assert len({_word_ids(w)[0] for w in words}) == len(words)  # no id collisions
    s, enc, al, c = _aligned_setup(c_prompt)
    c.handle(_msg(a_prompt, 4, provisional=True, cap=0.4))
    _run(s, 2)
    cur = s.current[0]
    nonzero = {p for p in range(AL_LEN) if cur[p].ne(0).any()}
    assert {0, 1, 2} <= nonzero  # C's words, still fading out
    prev = dict(al.layout)  # A's layout: position -> id
    progress = s.progress
    assert c.handle(_msg(b_prompt, 4))["kind"] == "final"
    assert s.progress == 0.0 and progress == 0.5
    b_ids = _word_ids(b_prompt)
    for p, i in al.layout:
        if prev.get(p) != i:  # a new (unmatched) token
            assert p not in nonzero, (p, i)
    assert [p for p, _ in al.layout][:4] == [7, 8, 9, 10]
    assert al.layout[-1] == (3, b_ids[-1])  # the end token is matched

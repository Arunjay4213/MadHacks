"""Cut latency options of the Causal Forcing backend: defer_context and interrupt(), around the stub
generator of test_causal_forcing.py (CPU, no official checkout).

defer_context: each chunk's context pass (its clean frame written into the frame KV cache) runs at the
start of the next generate_chunk instead of at the end of its own, and is skipped when that next chunk
rebases (the rebase clears the frame cache). interrupt(): stop the generate_chunk in flight at its next
pass boundary with ChunkInterrupted, leaving the backend as if that call had never been made.
"""

import pytest
import torch

from engine.backends import causal_forcing as cf
from engine.tests.test_causal_forcing import (  # noqa: F401 (no_cuda is a fixture)
    GOLDEN,
    GOLDEN_STEER_FADES,
    GOLDEN_STEER_REBASES,
    GOLDEN_STEER_STARTS,
    GOLDEN_STEER_TAIL,
    GOLDEN_STEER_TRACE,
    _legacy_run,
    _same_trace,
    _steer_run,
    make_backend,
    no_cuda,
    summarize,
)

A = torch.full((1, 3, 8), 1.0)
C = torch.full((1, 3, 8), 2.0)


def without_skipped_context_passes(trace):
    """The default-mode trace with the context passes defer_context never runs: the last one of each
    video (dropped by reset, or never run at the end) and the one right before a rebase. In both cases
    the next call starts from an empty frame cache (kv_end 0)."""
    calls, sums = trace
    keep = []
    for j, call in enumerate(calls):
        context = call[1] == [0]
        nxt_empty = j + 1 == len(calls) or calls[j + 1][5] == 0
        if not (context and nxt_empty):
            keep.append(call)
    return keep, sums


def latents(outs):
    return [o.latents for o in outs]


# ---------------------------------------------------------------- defer_context


def test_defer_context_is_off_by_default(no_cuda):
    b = make_backend()
    assert b.defer_context is False and b.context_passes_skipped == 0 and b.interrupted_chunks == 0


def test_defer_context_same_passes_and_latents_on_the_legacy_run(no_cuda):
    (calls, sums), outs = _legacy_run(make_backend(defer_context=True))
    want = without_skipped_context_passes(GOLDEN)
    assert len(calls) == len(GOLDEN[0]) - 2  # the last context pass of each of the two videos
    assert _same_trace((calls, sums), want)
    assert [o.chunk_index for o in outs] == [0, 1, 2, 3, 4, 5, 0, 1]


def test_defer_context_through_steers_refresh_clean_anchor_and_reset(no_cuda):
    """The recorded steer run: two steers, a refresh from the clean anchor saved 3 chunks after the
    first steer, a seeded reset. Same passes in the same order minus the skipped context passes, the
    same latents, chunk starts, fades, rebases, clean anchor and main noise stream."""
    b = make_backend(cut_fade_frames=8, defer_context=True)
    trace, starts, fades, rebases, clean_chunk, tail = _steer_run(b)
    assert _same_trace(trace, without_skipped_context_passes(GOLDEN_STEER_TRACE))
    assert starts == GOLDEN_STEER_STARTS and fades == GOLDEN_STEER_FADES
    assert rebases == GOLDEN_STEER_REBASES and all(r.steer is None for r in rebases)
    assert clean_chunk is None and abs(tail - GOLDEN_STEER_TAIL) <= 2e-3
    assert b.context_passes_skipped == 3  # before each of the 3 re-anchors; the reset drops the 4th


@pytest.mark.parametrize("options", [
    {"steer_anchor_t": 800, "steer_noise": "fixed"},
    {"steer_anchor_t": 800, "steer_noise": "fixed", "steer_steps": "800,500"},
    {"steer_steps": "1000,500"},
    {"steer_noise": "fixed", "steer_noise_first_frame": True, "refresh_anchor_t": 500, "refresh_noise": "fresh"},
])
def test_defer_context_with_the_steer_start_options(no_cuda, options):
    """A steer from the last clean latent (steer_anchor_t) and the fixed noises see exactly what they
    see without defer_context: the deferred pass changes no latent and no RNG draw."""
    off = _steer_run(make_backend(cut_fade_frames=8, **options))
    on = _steer_run(make_backend(cut_fade_frames=8, defer_context=True, **options))
    assert _same_trace(on[0], without_skipped_context_passes(off[0]))
    assert on[1:] == off[1:]  # starts, fades, rebases (SteerStart included), clean anchor chunk, tail


def _auto_run(b):
    a, c, d = (torch.full((1, 3, 8), float(v)) for v in (1, 2, 3))
    tags = {id(a): "A", id(c): "B", id(d): "C"}
    b.reset(seed=0)
    seq = [a] * 9 + [c] * 3 + [d] + [d] * 12 + [a] * 2
    outs = [b.generate_chunk(cond) for cond in seq]
    return summarize(b, outs, tags), dict(b.auto_rebases), b.rebases, b.recached_frames


def test_defer_context_with_the_automatic_rebases(no_cuda):
    """steer_rebase (keep 3) and refresh_every (keep 6) re-encode kept frames after clearing the cache:
    the deferred pass before them is skipped and the re-encoded frames are the same."""
    opts = {"steer_rebase": True, "steer_min_gap": 4, "refresh_every": 5}
    off = _auto_run(make_backend(**opts))
    b = make_backend(defer_context=True, **opts)
    on = _auto_run(b)
    assert sum(on[1].values()) >= 3  # both kinds fired
    assert on[1] == off[1] and on[2:] == off[2:]
    assert _same_trace(on[0], without_skipped_context_passes(off[0]))
    assert b.context_passes_skipped == sum(on[1].values())


def test_defer_context_cut_latent_is_ready_after_its_denoising_passes(no_cuda):
    """With steer_steps=1000,500 and defer_context the cut chunk runs 2 passes before its latent is
    ready (no context pass of the old chunk, none of its own yet), against 5 by default."""
    for options, want, ctx_frames in (({}, 5, [1]), ({"steer_steps": "1000,500", "defer_context": True}, 2, [0])):
        b = make_backend(**options)
        b.reset(seed=0)
        for _ in range(6):
            b.generate_chunk(A)
        b.reanchor(1.0)
        n0 = len(b.generator.calls)
        out = b.generate_chunk(C)
        assert len(b.generator.calls) - n0 == want and out.start is cf.ChunkStart.FIRST_FRAME
        n0 = len(b.generator.calls)
        b.generate_chunk(C)
        ctx = [c for c in b.generator.calls[n0:] if c["t"] == [0]]
        # By default the next chunk runs its own context pass; deferred, it runs the cut's (frame 0).
        assert [c["frame"] for c in ctx] == ctx_frames and ctx[0]["cond"] is C


def test_deferred_pass_runs_with_its_own_cond_before_the_text_cache_reset(no_cuda):
    b = make_backend(defer_context=True)
    b.reset(seed=0)
    b.generate_chunk(A)
    b.generate_chunk(A)
    n0 = len(b.generator.calls)
    b.generate_chunk(C)  # a cond change, no rebase
    calls = b.generator.calls[n0:]
    assert calls[0]["t"] == [0] and calls[0]["frame"] == 1
    assert calls[0]["cond"] is A and calls[0]["text_init"] is True  # chunk 1's pass, A's text cache
    assert calls[1]["cond"] is C and calls[1]["text_init"] is False  # then the reset for C


def test_reset_drops_a_deferred_pass(no_cuda):
    b = make_backend(defer_context=True)
    b.reset(seed=0)
    b.generate_chunk(A)
    b.reset(seed=0)
    n0 = len(b.generator.calls)
    b.generate_chunk(A)
    calls = b.generator.calls[n0:]
    assert len(calls) == b.first_frame_steps and all(c["frame"] == 0 for c in calls)
    assert b.context_passes_skipped == 0  # dropped by reset, not counted as skipped


def test_load_converts_defer_context(monkeypatch):
    seen = {}

    class Capture:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(cf, "Backend", Capture)
    cf.load(64, 64, defer_context="1")
    assert seen == {"defer_context": True}
    cf.load(64, 64, defer_context="0")
    assert seen == {"defer_context": False}


# ---------------------------------------------------------------- interrupt()


class InterruptingGenerator:
    """The stub generator, calling backend.interrupt() during its call number `at` (0-based, counted
    from when it is installed), as the engine's steer worker would from another thread."""

    def __init__(self, b, at):
        self.inner, self.b, self.at, self.n = b.generator, b, at, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, **kw):
        if self.n == self.at:
            self.b.interrupt()
        self.n += 1
        return self.inner(**kw)


def snapshot(b):
    return {
        "chunk": b.chunk_index, "frame": b.frame_index, "history": [id(x) for x in b._history],
        "last_cond": id(b._last_cond), "clean": id(b._clean_anchor), "clean_chunk": b.clean_anchor_chunk, "capture": b._clean_capture_chunk,
        "last_rebase": b.last_rebase, "rebases": b.rebases, "auto": dict(b.auto_rebases),
        "steer_pending": b._steer_pending, "state": b.state, "video": b.video_id,
    }


def interrupted(b, cond, at):
    """Run one generate_chunk(cond) that interrupt() stops; returns the generator calls it made."""
    inner = b.generator
    b.generator = InterruptingGenerator(b, at)
    n0 = len(inner.calls)
    try:
        with pytest.raises(cf.ChunkInterrupted) as info:
            b.generate_chunk(cond)
    finally:
        b.generator = inner
    assert info.value.chunk_interrupted is True
    return inner.calls[n0:]


STEER_OPTIONS = [{}, {"steer_anchor_t": 800, "steer_steps": "800,500"}, {"steer_anchor_t": 800, "steer_noise": "fixed"}]


@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("at", [0, 1])
@pytest.mark.parametrize("options", STEER_OPTIONS)
def test_interrupt_leaves_the_backend_as_before_the_call(no_cuda, defer, at, options):
    b = make_backend(defer_context=defer, **options)
    b.reset(seed=0)
    for _ in range(6):
        b.generate_chunk(A)
    before, rng = snapshot(b), b._rng.get_state()
    calls = interrupted(b, C, at)  # a cond change on the stopped chunk too
    # Stopped at the boundary after the pass the interrupt arrived in (with defer_context, call 0 is
    # the deferred context pass of chunk 5).
    assert len(calls) == at + 1
    assert snapshot(b) == before
    assert torch.equal(b._rng.get_state(), rng)  # its noise draws are rolled back
    assert b.interrupted_chunks == 1 and b.rebases == 0


@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("at", [0, 1])
@pytest.mark.parametrize("options", STEER_OPTIONS)
def test_the_cut_after_an_interrupt_is_the_cut_one_chunk_earlier(no_cuda, defer, at, options):
    """Wherever the interrupt lands, the cut takes the stopped chunk's number and gives exactly the
    video a reanchor() made before that chunk gives: same latents, same noise afterwards, same clean
    anchor 3 chunks later. Under steer_noise=fresh the cut draws from the main stream where it was."""
    ref = make_backend(defer_context=defer, **options)
    ref.reset(seed=0)
    want = [ref.generate_chunk(A) for _ in range(6)]
    ref.reanchor(1.0)
    want += [ref.generate_chunk(C) for _ in range(5)]

    b = make_backend(defer_context=defer, **options)
    b.reset(seed=0)
    got = [b.generate_chunk(A) for _ in range(6)]
    interrupted(b, A, at)
    b.reanchor(1.0)
    got += [b.generate_chunk(C) for _ in range(5)]

    assert [o.chunk_index for o in got] == list(range(11))
    assert got[6].start is cf.ChunkStart.FIRST_FRAME
    assert all(torch.equal(x, y) for x, y in zip(latents(got), latents(want)))
    assert b.last_rebase == ref.last_rebase and b.clean_anchor_chunk == ref.clean_anchor_chunk == 9
    assert torch.equal(b._rng.get_state(), ref._rng.get_state())


@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("rerun", ["A", "C"])
def test_an_interrupt_with_no_cut_after_it_leaves_a_valid_cache(no_cuda, defer, rerun):
    """The steer did not fire after all: the stopped chunk is generated again at the same position.
    Its passes overwrite the noisy frame the stopped passes wrote, as the extra denoising passes of a
    chunk do; the text cache is rebuilt when it holds the stopped chunk's cond."""
    cond = A if rerun == "A" else C
    ref = make_backend(defer_context=defer)
    ref.reset(seed=0)
    want = [ref.generate_chunk(A) for _ in range(6)] + [ref.generate_chunk(cond) for _ in range(3)]

    b = make_backend(defer_context=defer)
    b.reset(seed=0)
    got = [b.generate_chunk(A) for _ in range(6)]
    interrupted(b, C, 1)
    n0 = len(b.generator.calls)
    got += [b.generate_chunk(cond) for _ in range(3)]
    rerun_calls = b.generator.calls[n0:]

    assert [o.chunk_index for o in got] == list(range(9))
    assert all(torch.equal(x, y) for x, y in zip(latents(got), latents(want)))
    first = [c for c in rerun_calls if c["t"] != [0]][0]
    assert first["frame"] == 6 and first["cond"] is cond
    assert first["text_init"] is False  # rebuilt: under A it held C's K/V from the stopped passes
    assert not any(c["t"] == [0] and c["frame"] == 5 for c in rerun_calls)  # chunk 5's context pass ran once


def test_interrupt_at_the_clean_capture_chunk(no_cuda):
    b = make_backend()
    b.reset(seed=0)
    outs = [b.generate_chunk(A) for _ in range(3)]
    interrupted(b, A, 0)  # chunk 3, the one the clean anchor is saved from
    assert b._clean_anchor is None and b._clean_capture_chunk == 3
    outs.append(b.generate_chunk(A))
    assert b.clean_anchor_chunk == 3 and b._clean_anchor is outs[3].latents


def test_interrupt_never_stops_a_cut_a_rebase_or_a_first_chunk(no_cuda):
    b = make_backend(steer_steps="1000,500")
    b.generator = InterruptingGenerator(b, 0)
    b.reset(seed=0)
    b.generate_chunk(A)  # the first chunk of a video
    for _ in range(3):
        b.generate_chunk(A)
    b.generator.n, b.generator.at = 0, 0
    b.reanchor(1.0)
    b.generate_chunk(C)  # a steer cut
    b.generator.n, b.generator.at = 0, 0
    b.reanchor(0.5, source="clean")
    b.generate_chunk(C)  # a refresh
    b.generator.n, b.generator.at = 0, 0
    b.generate_chunk(C, rebase=cf.Rebase(keep=2))  # an explicit rebase
    assert b.interrupted_chunks == 0 and b.chunk_index == 7


def test_interrupt_never_stops_an_automatic_rebase(no_cuda):
    b = make_backend(refresh_every=3)
    b.reset(seed=0)
    for _ in range(3):
        b.generate_chunk(A)
    b.generator = InterruptingGenerator(b, 0)
    b.generate_chunk(A)  # chunk 3: the periodic rebase
    assert b.auto_rebases["refresh"] == 1 and b.interrupted_chunks == 0


def test_a_late_interrupt_is_dropped_by_the_next_chunk(no_cuda):
    b = make_backend()
    b.reset(seed=0)
    for _ in range(3):
        b.generate_chunk(A)
    b.interrupt()  # between chunks: too late for the chunk it was meant for
    out = b.generate_chunk(A)
    assert out.chunk_index == 3 and b.interrupted_chunks == 0
    b.interrupt()
    b.reset(seed=0)  # a reset drops it too
    b.generate_chunk(A)
    assert b.generate_chunk(A).chunk_index == 1 and b.interrupted_chunks == 0


def test_interrupt_keeps_the_automatic_steer_detection(no_cuda):
    """steer_rebase watches cond changes: a stopped chunk whose cond changed must not eat the change."""
    opts = {"steer_rebase": True, "steer_min_gap": 2}
    seq = [A, C, C, C, C]
    ref = make_backend(**opts)
    ref.reset(seed=0)
    want = [ref.generate_chunk(cond) for cond in seq]

    b = make_backend(**opts)
    b.reset(seed=0)
    got = [b.generate_chunk(A)]
    interrupted(b, C, 0)  # chunk 1: the change, gap 1 < 2, so no rebase on it
    got += [b.generate_chunk(cond) for cond in seq[1:]]
    assert b.auto_rebases == ref.auto_rebases and ref.auto_rebases["arrival"] == 1
    assert all(torch.equal(x, y) for x, y in zip(latents(got), latents(want)))


def test_chunk_interrupted_is_a_runtime_error_with_the_engine_marker():
    exc = cf.ChunkInterrupted("x")
    assert isinstance(exc, RuntimeError) and exc.chunk_interrupted is True


def test_an_interrupt_during_the_last_pass_lets_the_chunk_finish(no_cuda):
    b = make_backend()
    b.reset(seed=0)
    for _ in range(4):
        b.generate_chunk(A)
    b.generator = InterruptingGenerator(b, 2)  # during the context pass: no boundary left
    out = b.generate_chunk(A)
    assert out.chunk_index == 4 and b.interrupted_chunks == 0
    b.generator = b.generator.inner
    assert b.generate_chunk(A).chunk_index == 5  # and the next chunk drops it


def test_interrupt_from_another_thread(no_cuda):
    """interrupt() only sets an event; the generating thread raises at its next pass boundary."""
    import threading

    b = make_backend()
    b.reset(seed=0)
    for _ in range(3):
        b.generate_chunk(A)
    entered, release = threading.Event(), threading.Event()
    inner = b.generator

    def slow(**kw):
        entered.set()
        release.wait(5)
        return inner(**kw)

    b.generator = slow
    errors = []

    def run():
        try:
            b.generate_chunk(A)
        except cf.ChunkInterrupted as exc:
            errors.append(exc)

    t = threading.Thread(target=run)
    t.start()
    assert entered.wait(5)
    b.interrupt()
    release.set()
    t.join(5)
    assert len(errors) == 1 and b.chunk_index == 3


def test_a_stopped_cond_change_leaves_no_automatic_steer_behind(no_cuda):
    """steer_fire_on_change=0: a change only arms the arrival rebase. A stopped chunk under a new cond
    must not arm it when the next chunk goes back to the old cond (no change happened)."""
    opts = {"steer_rebase": True, "steer_min_gap": 2, "steer_fire_on_change": False}
    ref = make_backend(**opts)
    ref.reset(seed=0)
    want = [ref.generate_chunk(A) for _ in range(6)]

    b = make_backend(**opts)
    b.reset(seed=0)
    got = [b.generate_chunk(A) for _ in range(3)]
    interrupted(b, C, 0)
    got += [b.generate_chunk(A) for _ in range(3)]
    assert b.auto_rebases == ref.auto_rebases == {"change": 0, "arrival": 0, "refresh": 0} and b.rebases == 0
    assert all(torch.equal(x, y) for x, y in zip(latents(got), latents(want)))

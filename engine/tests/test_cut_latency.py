"""steer_cut_steps and defer_context in the Causal Forcing backend, around the stub generator of
test_causal_forcing.py (CPU, no official checkout)."""

import pytest
import torch

from engine.backends import causal_forcing as cf
from engine.tests.test_causal_forcing import _legacy_run, make_backend, no_cuda, summarize  # noqa: F401


def a_c():
    return torch.full((1, 3, 8), 1.0), torch.full((1, 3, 8), 2.0)


def steer_chunk_calls(b, n_before=5, strength=1.0, source="last", same_cond=False):
    a, c = a_c()
    c = a if same_cond else c
    b.reset(seed=0)
    for _ in range(n_before):
        b.generate_chunk(a)
    n0 = len(b.generator.calls)
    b.reanchor(strength, source=source)
    out = b.generate_chunk(c)
    return out, b.generator.calls[n0:]


def test_steer_cut_steps_runs_the_short_schedule(no_cuda):
    b = make_backend(steer_cut_steps="1000,500")
    out, calls = steer_chunk_calls(b)
    ctx = int(b.context_noise)
    assert [call["t"] for call in calls] == [[int(t)] for t in b.denoising_steps] + [[ctx]]  # 3 passes, not 5
    assert calls[0]["kv_end"] == 0 and all(call["frame"] == 0 for call in calls)
    assert out.start is cf.ChunkStart.FIRST_FRAME


def test_steer_cut_steps_leave_refresh_and_partial_strength_alone(no_cuda):
    b = make_backend(steer_cut_steps="1000,500", clean_anchor_after=1)
    _, calls = steer_chunk_calls(b, strength=0.5, source="clean", same_cond=True)  # a refresh from the clean anchor, step 2 of 4
    assert [call["t"][0] for call in calls[:-1]] == [int(t) for t in b.first_chunk_steps[2:]]
    b2 = make_backend(steer_cut_steps="1000,500")
    _, calls = steer_chunk_calls(b2, strength=0.75)  # anchor step 1: unchanged first-frame tail
    assert [call["t"][0] for call in calls[:-1]] == [int(t) for t in b2.first_chunk_steps[1:]]


def test_first_chunk_of_a_video_keeps_the_first_frame_schedule(no_cuda):
    b = make_backend(steer_cut_steps="1000,500")
    b.reset(seed=0)
    b.generate_chunk(a_c()[0])
    assert [c["t"][0] for c in b.generator.calls[:-1]] == [int(t) for t in b.first_chunk_steps]


@pytest.mark.parametrize("bad", ["500,1000", "900,500", "1000,0", "x"])
def test_steer_cut_steps_is_checked(bad):
    with pytest.raises(ValueError):
        make_backend(steer_cut_steps=bad)


def test_defer_context_gives_the_same_calls_and_latents(no_cuda):
    want, _ = _legacy_run(make_backend())
    got, outs = _legacy_run(make_backend(defer_context=True))
    # Same passes in the same order, minus the context pass of each video's last chunk (dropped by the reset,
    # and not yet run at the end), and the same latents.
    calls = want[0]
    assert got[0] == calls[:19] + calls[20:27]
    assert got[1] == want[1]


def test_defer_context_skips_the_pass_before_a_cut(no_cuda):
    b = make_backend(defer_context=True)
    _, calls = steer_chunk_calls(b)
    ctx = int(b.context_noise)
    # No context pass for the chunk before the cut: the cut clears the frame cache anyway.
    assert [call["t"] for call in calls] == [[int(t)] for t in b.first_chunk_steps]
    assert b.context_passes_skipped == 1
    n0 = len(b.generator.calls)
    b.generate_chunk(a_c()[1])
    nxt = b.generator.calls[n0:]
    assert [c["t"] for c in nxt] == [[ctx]] + [[int(t)] for t in b.denoising_steps]  # cut's deferred pass first
    assert [c["frame"] for c in nxt] == [0, 1, 1]


def test_deferred_pass_uses_its_own_cond_before_the_text_cache_reset(no_cuda):
    b = make_backend(defer_context=True)
    a, c = a_c()
    b.reset(seed=0)
    b.generate_chunk(a)
    b.generate_chunk(a)
    n0 = len(b.generator.calls)
    b.generate_chunk(c)  # cond change, no rebase
    calls = b.generator.calls[n0:]
    assert calls[0]["cond"] is a and calls[0]["text_init"] is True  # chunk 1's pass, A's text cache
    assert calls[1]["cond"] is c and calls[1]["text_init"] is False  # then the reset for C


def test_defer_context_off_with_automatic_rebase(no_cuda):
    assert make_backend(defer_context=True, refresh_every=4).defer_context is False


def test_reset_drops_a_deferred_pass(no_cuda):
    b = make_backend(defer_context=True)
    b.reset(seed=0)
    b.generate_chunk(a_c()[0])
    b.reset(seed=0)
    n0 = len(b.generator.calls)
    b.generate_chunk(a_c()[0])
    assert all(c["frame"] == 0 for c in b.generator.calls[n0:]) and len(b.generator.calls) - n0 == 4


class InterruptingGenerator:
    """The stub generator, calling backend.interrupt() during its call number `at` (0-based)."""

    def __init__(self, inner, b, at):
        self.inner, self.b, self.at, self.n = inner, b, at, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, **kw):
        if self.n == self.at:
            self.b.interrupt()
        self.n += 1
        return self.inner(**kw)


def test_interrupt_stops_a_normal_chunk_and_keeps_its_state(no_cuda):
    b = make_backend(defer_context=True)
    a, c = a_c()
    b.reset(seed=0)
    for _ in range(3):
        b.generate_chunk(a)
    state = (b.chunk_index, b.frame_index, len(b._history))
    b.generator = InterruptingGenerator(b.generator, b, at=0)  # during the first pass of chunk 3
    with pytest.raises(cf.ChunkInterrupted):
        b.generate_chunk(a)
    assert (b.chunk_index, b.frame_index, len(b._history)) == state and b.interrupted_chunks == 1
    b.reanchor(1.0)
    out = b.generate_chunk(c)  # the cut runs at once, with its full schedule
    assert out.chunk_index == 3 and out.start is cf.ChunkStart.FIRST_FRAME


def test_interrupt_never_stops_a_cut_and_a_late_one_is_dropped(no_cuda):
    b = make_backend()
    a, c = a_c()
    b.reset(seed=0)
    for _ in range(3):
        b.generate_chunk(a)
    b.reanchor(1.0)
    b.generator = InterruptingGenerator(b.generator, b, at=0)
    b.generate_chunk(c)  # a re-anchor chunk is not stopped
    b.generate_chunk(c)  # the interrupt left over from it is dropped at the start of this chunk
    assert b.interrupted_chunks == 0

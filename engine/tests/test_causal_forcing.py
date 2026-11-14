"""The Causal Forcing backend around a stub generator, on CPU, without the official checkout.

The stub records every forward call (frame position, timestep, cond, input, whether the text
cache was initialised, the frame cache end index) and emulates the two caches just enough to
check what the backend does to them. CUDA streams/events/devices are replaced with no-ops.
"""

import contextlib

import pytest
import torch

from engine.backends import causal_forcing as cf

FS = 4  # frame_seq_length of the stub (2x2 patches on 4x4 latents)


class StubScheduler:
    """Flow-matching scheduler shaped like the official one: shift 5, 1000 steps, extra_one_step."""

    def __init__(self, shift: float = 5.0):
        sig = torch.linspace(1.0, 0.0, 1001)[:-1]
        self.sigmas = shift * sig / (1 + (shift - 1) * sig)
        self.timesteps = self.sigmas * 1000

    def add_noise(self, x, noise, t):
        if t.ndim == 2:
            t = t.flatten(0, 1)
        idx = torch.argmin((self.timesteps.unsqueeze(0) - t.unsqueeze(1).float()).abs(), dim=1)
        sigma = self.sigmas[idx].reshape(-1, 1, 1, 1)
        return ((1 - sigma) * x + sigma * noise).type_as(noise)


class StubGenerator:
    def __init__(self):
        self.calls = []

    def __call__(self, noisy_image_or_video, conditional_dict, timestep, kv_cache, crossattn_cache, current_start):
        x = noisy_image_or_video
        cond = conditional_dict["prompt_embeds"]
        self.calls.append({
            "frame": current_start // FS,
            "t": [int(v) for v in timestep.flatten().tolist()],
            "cond": cond,
            "x": x.clone(),
            "text_init": crossattn_cache[0]["is_init"],
            "kv_end": int(kv_cache[0]["global_end_index"].item()),
        })
        crossattn_cache[0]["is_init"] = True
        end = current_start + x.shape[1] * FS
        if end > kv_cache[0]["global_end_index"].item():
            kv_cache[0]["global_end_index"].fill_(end)
        denoised = 0.5 * x + 0.01 * cond.float().mean() + 0.001 * (current_start // FS) + 0.0001 * timestep.float().mean()
        return None, denoised


class _Null:
    def __init__(self, *a, **k):
        pass

    def wait_stream(self, *a):
        pass

    def record(self, *a):
        pass

    def synchronize(self):
        pass


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "default_stream", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "Stream", _Null)
    monkeypatch.setattr(torch.cuda, "Event", _Null)


def make_backend(seed=0, window=21, **options):
    b = cf.Backend.__new__(cf.Backend)
    b.device = torch.device("cpu")
    b.dtype = torch.float32
    b.latent_hw = (4, 4)
    b.frame_seq_length = FS
    b.attn_window_frames = window
    b.sink_frames = 0
    b.max_latent_frames = 65536
    b.num_frame_per_block = 1
    b.context_noise = 0
    b.generator = StubGenerator()
    b.scheduler = StubScheduler()
    b.denoising_steps = cf._timestep_list(b.scheduler, [1000, 500], True)
    b.first_chunk_steps = cf._timestep_list(b.scheduler, [1000, 750, 500, 250], True)
    b._warp = True
    b._init_runtime(seed, **options)
    return b


def conds(n=3):
    return [torch.full((1, 3, 8), float(i + 1)) for i in range(n)]


def summarize(b, outs, tags):
    calls = [(c["frame"], c["t"], tags[id(c["cond"])], round(float(c["x"].sum()), 4), c["text_init"], c["kv_end"])
             for c in b.generator.calls]
    return calls, [round(float(o.latents.sum()), 4) for o in outs]


# ---------------------------------------------------------------- off = unchanged


def _legacy_run(b):
    a, c, _ = conds()
    tags = {id(a): "A", id(c): "B"}
    outs = []
    b.reset(seed=0)
    for cond in (a, a, a, c, c, c):
        outs.append(b.generate_chunk(cond))
    b.reset(seed=1)
    for cond in (a, a):
        outs.append(b.generate_chunk(cond))
    return summarize(b, outs, tags), outs


# Recorded with the backend as it was before rebase existed (origin/main a403639), same stub.
GOLDEN = ([(0, [1000], 'A', 6.5901, False, 0), (0, [937], 'A', 25.77, True, 4), (0, [833], 'A', 19.8163, True, 4), (0, [625], 'A', 3.3423, True, 4), (0, [0], 'A', 20.2312, True, 4), (1, [1000], 'A', 8.6193, True, 4), (1, [833], 'A', 25.1555, True, 8), (1, [0], 'A', 36.7271, True, 8), (2, [1000], 'A', -13.7387, True, 8), (2, [833], 'A', -16.6387, True, 12), (2, [0], 'A', 16.086, True, 12), (3, [1000], 'B', -6.3662, False, 12), (3, [833], 'B', 6.8761, True, 16), (3, [0], 'B', 30.6594, True, 16), (4, [1000], 'B', -18.1832, True, 16), (4, [833], 'B', -5.0429, True, 20), (4, [0], 'B', 24.9559, True, 20), (5, [1000], 'B', -11.6108, True, 20), (5, [833], 'B', 5.7291, True, 24), (5, [0], 'B', 30.5979, True, 24), (0, [1000], 'A', 13.237, False, 0), (0, [937], 'A', -11.5417, True, 4), (0, [833], 'A', 23.3178, True, 4), (0, [625], 'A', 12.2185, True, 4), (0, [0], 'A', 24.6693, True, 4), (1, [1000], 'A', 4.4666, True, 4), (1, [833], 'A', 24.5751, True, 8), (1, [0], 'A', 36.4369, True, 8)], [20.2312, 36.7271, 16.086, 30.6594, 24.9559, 30.5979, 24.6693, 36.4369])


def _same_trace(got, want, tol=2e-3) -> bool:
    """Same call sequence; float sums within tol (torch builds round the 4th decimal differently)."""
    (gc, gs), (wc, ws) = got, want
    if len(gc) != len(wc) or len(gs) != len(ws):
        return False
    for g, w in zip(gc, wc):
        if g[:3] != w[:3] or g[4:] != w[4:] or abs(g[3] - w[3]) > tol:
            return False
    return all(abs(a - b) <= tol for a, b in zip(gs, ws))


def test_off_is_bit_for_bit_unchanged(no_cuda):
    (calls, sums), outs = _legacy_run(make_backend())
    assert _same_trace((calls, sums), GOLDEN)
    assert [o.chunk_index for o in outs] == [0, 1, 2, 3, 4, 5, 0, 1]


def test_off_with_auto_options_at_defaults_is_unchanged(no_cuda):
    b = make_backend(steer_rebase=False, refresh_every=0)
    (calls, sums), _ = _legacy_run(b)
    assert _same_trace((calls, sums), GOLDEN)
    assert b.rebases == 0 and b.recached_frames == 0


# ---------------------------------------------------------------- explicit rebase


def run_then_rebase(b, n_before, rebase, a=None, c=None):
    """`n_before` chunks under A, then one chunk under B with `rebase`. Returns (outs, calls of the
    rebase chunk, RNG state just before the rebase chunk, resets before)."""
    a = a if a is not None else torch.full((1, 3, 8), 1.0)
    c = c if c is not None else torch.full((1, 3, 8), 2.0)
    b.reset(seed=0)
    outs = [b.generate_chunk(a) for _ in range(n_before)]
    n0 = len(b.generator.calls)
    rng = b._rng.get_state()
    resets = b.crossattn_resets
    outs.append(b.generate_chunk(c, rebase=rebase))
    return outs, b.generator.calls[n0:], rng, resets, c


@pytest.mark.parametrize("keep", [0, 1, 3])
def test_rebase_call_sequence(no_cuda, keep):
    b = make_backend()
    outs, calls, _, resets, c = run_then_rebase(b, 5, cf.Rebase(keep=keep))
    ctx = int(b.context_noise)
    # Re-encode: one pass per kept frame, oldest first, positions 0..keep-1, new cond, context timestep.
    for i in range(keep):
        call = calls[i]
        assert call["frame"] == i and call["t"] == [ctx] and call["cond"] is c
        assert torch.equal(call["x"], outs[5 - keep + i].latents)
    assert calls[0]["kv_end"] == 0  # frame cache cleared before the first pass
    assert calls[0]["text_init"] is False  # text cache reset, rebuilt from the new cond
    assert all(call["text_init"] for call in calls[1:])
    assert b.crossattn_resets == resets + 1  # once, although the cond also changed
    # Then the new frame at position keep: first-frame schedule for a hard cut, else the normal one.
    sched = b.first_chunk_steps if keep == 0 else b.denoising_steps
    gen = calls[keep:]
    assert [call["t"] for call in gen] == [[int(t)] for t in sched] + [[ctx]]
    assert all(call["frame"] == keep and call["cond"] is c for call in gen)
    assert b.frame_index == keep + 1 and b.chunk_index == 6
    assert b.rebases == 1 and b.recached_frames == keep
    assert b.last_rebase == cf.Rebase(keep, 0, None)
    assert outs[-1].chunk_index == 5 and outs[-1].video_id == outs[0].video_id  # no decoder restart
    # The next chunk continues normally from position keep + 1.
    n0 = len(b.generator.calls)
    b.generate_chunk(c)
    nxt = b.generator.calls[n0:]
    assert [call["frame"] for call in nxt] == [keep + 1] * 3
    assert [call["t"] for call in nxt] == [[int(t)] for t in b.denoising_steps] + [[ctx]]


def test_rebase_same_cond_still_resets_text_cache(no_cuda):
    b = make_backend()
    a = torch.full((1, 3, 8), 1.0)
    outs, calls, _, resets, _ = run_then_rebase(b, 4, {"keep": 2}, a=a, c=a)
    assert calls[0]["text_init"] is False and b.crossattn_resets == resets + 1
    assert b.last_rebase.keep == 2


def test_memory_noise_uses_noisy_input_and_timestep(no_cuda):
    b = make_backend()
    outs, calls, rng, _, _ = run_then_rebase(b, 5, cf.Rebase(keep=2, memory_noise=500))
    t_mem = b._timestep(500)
    assert int(t_mem) > 500  # warped onto the shifted grid
    g = torch.Generator().set_state(rng)
    shape = outs[0].latents.shape
    torch.randn(shape, generator=g)  # the start noise of the new frame is drawn first
    for i in range(2):
        lat = outs[3 + i].latents
        noise = torch.randn(lat.flatten(0, 1).shape, generator=g)
        want = b.scheduler.add_noise(lat.flatten(0, 1), noise, t_mem * torch.ones([1], dtype=torch.long)).unflatten(0, shape[:2])
        assert calls[i]["t"] == [int(t_mem)] and calls[i]["frame"] == i
        assert torch.allclose(calls[i]["x"], want)
        assert not torch.equal(calls[i]["x"], lat)
    assert b.recached_frames == 2


@pytest.mark.parametrize("keep,n", [(0, 2), (0, 3), (1, 1)])
def test_anchor_step_start_latent_and_schedule(no_cuda, keep, n):
    b = make_backend()
    outs, calls, rng, _, _ = run_then_rebase(b, 4, cf.Rebase(keep=keep, anchor_step=n))
    sched = b.first_chunk_steps if keep == 0 else b.denoising_steps
    gen = calls[keep:]
    assert [call["t"] for call in gen] == [[int(t)] for t in sched[n:]] + [[0]]
    g = torch.Generator().set_state(rng)
    x_last = outs[3].latents
    noise = torch.randn(x_last.shape, generator=g)
    want = b.scheduler.add_noise(x_last.flatten(0, 1), noise.flatten(0, 1), sched[n] * torch.ones([1], dtype=torch.long))
    assert torch.allclose(gen[0]["x"], want.unflatten(0, x_last.shape[:2]))
    assert b.last_rebase == cf.Rebase(keep, 0, n)


def test_anchor_step_zero_is_pure_noise(no_cuda):
    b = make_backend()
    outs, calls, rng, _, _ = run_then_rebase(b, 4, cf.Rebase(keep=0, anchor_step=0))
    g = torch.Generator().set_state(rng)
    assert torch.equal(calls[0]["x"], torch.randn(outs[0].latents.shape, generator=g))
    assert len(calls) == b.first_frame_steps + 1


def test_anchor_step_out_of_range_raises(no_cuda):
    b = make_backend()
    with pytest.raises(ValueError):
        run_then_rebase(b, 4, cf.Rebase(keep=1, anchor_step=2))  # the normal schedule has 2 steps


def test_keep_clamped_to_buffer_and_window(no_cuda):
    b = make_backend()
    run_then_rebase(b, 2, cf.Rebase(keep=6))
    assert b.last_rebase.keep == 2 and b.frame_index == 3
    b = make_backend(window=4)
    outs, calls, _, _, _ = run_then_rebase(b, 10, cf.Rebase(keep=6))
    assert b.last_rebase.keep == 3 and [c["frame"] for c in calls[:3]] == [0, 1, 2]
    assert torch.equal(calls[0]["x"], outs[7].latents)


def test_rebase_on_first_chunk_is_ignored(no_cuda):
    b = make_backend()
    b.reset(seed=0)
    b.generate_chunk(torch.ones(1, 3, 8), rebase=cf.Rebase(keep=0))
    assert b.rebases == 0 and len(b.generator.calls) == b.first_frame_steps + 1


def test_rebase_of_and_strength_mapping(no_cuda):
    assert cf.Rebase.of({"keep": 1, "memory_noise": 700}) == cf.Rebase(1, 700, None)
    assert cf.Rebase.of(None) is None
    with pytest.raises(TypeError):
        cf.Rebase.of(3)
    b = make_backend()
    assert b.first_frame_steps == 4
    assert [b.anchor_step_for_strength(s) for s in (1.0, 1.5, 0.8, 0.5, 0.3, 0.01, 0.0, -1)] == [0, 0, 0, 2, 2, 3, None, None]


# ---------------------------------------------------------------- automatic mode


def drive(b, conds_seq):
    """Generate one chunk per cond; return the chunk indices at which a rebase happened."""
    b.reset(seed=0)
    fired = []
    for i, cond in enumerate(conds_seq):
        before = b.rebases
        b.generate_chunk(cond)
        if b.rebases != before:
            fired.append(i)
    return fired


def ramp_case():
    a = torch.zeros(1, 3, 8)
    ramp = [torch.full((1, 3, 8), 0.25 * (k + 1)) for k in range(4)]
    return [a] * 10 + ramp + [ramp[-1]] * 10


def test_auto_fire_on_change_then_arrival_skipped_by_gap(no_cuda):
    b = make_backend(steer_rebase=True, steer_keep=2)
    assert drive(b, ramp_case()) == [10]  # first changed chunk; arrival at 14 is within min_gap
    assert b.auto_rebases == {"change": 1, "arrival": 0, "refresh": 0}
    assert b.last_rebase.keep == 2


def test_auto_arrival_only(no_cuda):
    b = make_backend(steer_rebase=True, steer_fire_on_change=False)
    assert drive(b, ramp_case()) == [14]  # the first chunk whose cond stopped changing
    assert b.auto_rebases["arrival"] == 1


def test_auto_arrival_after_long_ramp(no_cuda):
    a = torch.zeros(1, 3, 8)
    ramp = [torch.full((1, 3, 8), 0.1 * (k + 1)) for k in range(9)]
    b = make_backend(steer_rebase=True)
    assert drive(b, [a] * 10 + ramp + [ramp[-1]] * 3) == [10, 19]


def test_auto_min_gap_from_reset_and_first_chunk(no_cuda):
    a, c = torch.zeros(1, 3, 8), torch.ones(1, 3, 8)
    b = make_backend(steer_rebase=True)
    assert drive(b, [a] * 3 + [c] * 10) == []  # change at 3 and arrival at 4 are within 8 of reset
    b = make_backend(steer_rebase=True, steer_min_gap=0)
    assert drive(b, [a] * 2 + [c] * 2)[0] == 2  # never on chunk 0 even with no gap


def test_auto_hard_switch_then_another(no_cuda):
    a, c, d = torch.zeros(1, 3, 8), torch.ones(1, 3, 8), torch.full((1, 3, 8), 2.0)
    b = make_backend(steer_rebase=True, steer_min_gap=4)
    # change at 8 fires; arrival at 9 is too close; change at 11 too close; arrival at 12 fires.
    assert drive(b, [a] * 8 + [c] * 3 + [d] * 4) == [8, 12]


def test_auto_refresh(no_cuda):
    a = torch.zeros(1, 3, 8)
    b = make_backend(refresh_every=5, refresh_keep=2)
    assert drive(b, [a] * 12) == [5, 10]
    assert b.auto_rebases["refresh"] == 2
    assert b.frame_index == 2 + 2  # rebased at chunk 10: 2 kept + chunks 10 and 11
    frames = [c["frame"] for c in b.generator.calls]
    assert max(frames) == 6  # positions stay small: at most keep + refresh_every - 1


def test_explicit_wins_over_auto(no_cuda):
    seq = ramp_case()
    b = make_backend(steer_rebase=True, steer_keep=3)
    b.reset(seed=0)
    for i, cond in enumerate(seq[:11]):
        b.generate_chunk(cond, rebase=cf.Rebase(keep=1) if i == 10 else None)
    assert b.rebases == 1 and b.last_rebase.keep == 1
    assert b.auto_rebases == {"change": 0, "arrival": 0, "refresh": 0}


def test_auto_needs_framewise(no_cuda):
    b = cf.Backend.__new__(cf.Backend)
    b.num_frame_per_block = 3
    with pytest.raises(ValueError):
        b._init_runtime(0, steer_rebase=True)


def test_reanchor_strength_maps_to_step_and_applies_once():
    import inspect

    from engine.backends import causal_forcing as cf

    src = inspect.getsource(cf.Backend.generate_chunk)
    assert "_pending_reanchor" in src
    b = cf.Backend.__new__(cf.Backend)
    b.first_chunk_steps = [1000, 937, 833, 625]
    b._pending_reanchor = None
    cf.Backend.reanchor(b, 1.0)
    assert b._pending_reanchor == cf.Rebase(keep=0, anchor_step=0)
    cf.Backend.reanchor(b, 0.5)
    assert b._pending_reanchor == cf.Rebase(keep=0, anchor_step=2)
    cf.Backend.reanchor(b, 0.25)
    assert b._pending_reanchor == cf.Rebase(keep=0, anchor_step=3)
    cf.Backend.reanchor(b, 0.0)
    assert b._pending_reanchor is None
    with pytest.raises(ValueError, match="source"):
        cf.Backend.reanchor(b, 0.5, source="golden")
    cf.Backend.reanchor(b, 0.5, source="clean")
    assert b._pending_reanchor == cf.Rebase(keep=0, anchor_step=2, from_clean=True)


# ---------------------------------------------------------------- clean anchor


def reanchored_chunk(b, cond, strength, source):
    """One chunk with reanchor(strength, source); returns (out, its generator calls, RNG state before it)."""
    b.reanchor(strength, source=source)
    n0 = len(b.generator.calls)
    rng = b._rng.get_state()
    out = b.generate_chunk(cond)
    return out, b.generator.calls[n0:], rng


def start_latent(b, x, step, rng):
    """What the first denoising pass of a hard cut gets: x noised to first_chunk_steps[step]."""
    g = torch.Generator().set_state(rng)
    noise = torch.randn(x.shape, generator=g)
    t = b.first_chunk_steps[step] * torch.ones([1], dtype=torch.long)
    return b.scheduler.add_noise(x.flatten(0, 1), noise.flatten(0, 1), t).unflatten(0, x.shape[:2])


def test_clean_anchor_saved_three_chunks_after_reset(no_cuda):
    b = make_backend()
    a = torch.full((1, 3, 8), 1.0)
    b.reset(seed=0)
    outs = [b.generate_chunk(a) for _ in range(3)]
    assert b._clean_anchor is None and b.clean_anchor_chunk is None
    outs += [b.generate_chunk(a) for _ in range(5)]
    assert b._clean_anchor is outs[3].latents and b.clean_anchor_chunk == 3
    b.reset(seed=0)
    assert b._clean_anchor is None and b.clean_anchor_chunk is None


def test_refresh_starts_from_the_clean_anchor_and_keeps_it(no_cuda):
    b = make_backend()
    a = torch.full((1, 3, 8), 1.0)
    b.reset(seed=0)
    outs = [b.generate_chunk(a) for _ in range(10)]
    clean = outs[3].latents
    out, calls, rng = reanchored_chunk(b, a, 0.5, "clean")
    assert torch.allclose(calls[0]["x"], start_latent(b, clean, 2, rng))
    assert not torch.allclose(calls[0]["x"], start_latent(b, outs[-1].latents, 2, rng))
    assert [c["t"] for c in calls] == [[int(t)] for t in b.first_chunk_steps[2:]] + [[0]]
    assert calls[0]["frame"] == 0 and b.frame_index == 1
    assert b.last_rebase == cf.Rebase(0, 0, 2, True)
    # The refresh does not replace the clean anchor, so every refresh starts from the same picture.
    outs += [out] + [b.generate_chunk(a) for _ in range(6)]
    assert b._clean_anchor is clean and b.clean_anchor_chunk == 3
    _, calls, rng = reanchored_chunk(b, a, 0.5, "clean")
    assert torch.allclose(calls[0]["x"], start_latent(b, clean, 2, rng))


def test_steer_reanchor_drops_and_recaptures_the_clean_anchor(no_cuda):
    b = make_backend()
    a, c = torch.full((1, 3, 8), 1.0), torch.full((1, 3, 8), 2.0)
    b.reset(seed=0)
    outs = [b.generate_chunk(a) for _ in range(10)]
    out, calls, rng = reanchored_chunk(b, c, 0.5, "last")  # a steer at chunk 10, from the last latent
    assert torch.allclose(calls[0]["x"], start_latent(b, outs[-1].latents, 2, rng))
    assert b.last_rebase == cf.Rebase(0, 0, 2, False)
    assert b._clean_anchor is None
    # A refresh before the new clean anchor exists falls back to the last latent and changes nothing.
    outs += [out, b.generate_chunk(c)]
    out, calls, rng = reanchored_chunk(b, c, 0.5, "clean")  # chunk 12
    assert torch.allclose(calls[0]["x"], start_latent(b, outs[-1].latents, 2, rng))
    assert b.last_rebase == cf.Rebase(0, 0, 2, False)
    outs += [out, b.generate_chunk(c)]  # chunk 13 = 10 + 3: the new clean anchor, under the new cond
    assert b._clean_anchor is outs[13].latents and b.clean_anchor_chunk == 13
    _, calls, rng = reanchored_chunk(b, c, 0.5, "clean")
    assert torch.allclose(calls[0]["x"], start_latent(b, outs[13].latents, 2, rng))


def test_clean_refresh_at_full_strength_is_pure_noise(no_cuda):
    b = make_backend()
    a = torch.full((1, 3, 8), 1.0)
    b.reset(seed=0)
    outs = [b.generate_chunk(a) for _ in range(6)]
    _, calls, rng = reanchored_chunk(b, a, 1.0, "clean")
    assert torch.equal(calls[0]["x"], torch.randn(outs[0].latents.shape, generator=torch.Generator().set_state(rng)))
    assert b.last_rebase == cf.Rebase(0, 0, 0, False)
    assert b.clean_anchor_chunk == 3  # kept


def test_clean_anchor_after_option(no_cuda):
    b = make_backend(clean_anchor_after=0)
    a = torch.full((1, 3, 8), 1.0)
    b.reset(seed=0)
    out = b.generate_chunk(a)
    assert b._clean_anchor is out.latents and b.clean_anchor_chunk == 0
    with pytest.raises(ValueError):
        make_backend(clean_anchor_after=-1)


# ---------------------------------------------------------------- chunk start and decoder restart


def test_chunk_start_marks_first_frames(no_cuda):
    b = make_backend()
    a = torch.full((1, 3, 8), 1.0)
    b.reset(seed=0)
    starts = [b.generate_chunk(a).start for _ in range(10)]
    starts.append(b.generate_chunk(a, rebase=cf.Rebase(keep=0, anchor_step=2)).start)
    starts.append(b.generate_chunk(a).start)
    starts.append(b.generate_chunk(a, rebase=cf.Rebase(keep=2)).start)
    b.reanchor(0.5, source="clean")
    starts.append(b.generate_chunk(a).start)
    F, C = cf.ChunkStart.FIRST_FRAME, cf.ChunkStart.CONTINUE
    assert starts == [F] + [C] * 9 + [F, C, C, F]
    b.reset(seed=0)
    assert b.generate_chunk(a).start is F


class StubTaehv:
    """decode_video stand-in: 4 frames per latent, each frame's value is its latent's mean."""

    def __init__(self):
        self.inputs = []

    def decode_video(self, x, parallel=False):
        self.inputs.append([round(float(v), 3) for v in x[0].float().mean(dim=(1, 2, 3))])
        vals = x[0].float().mean(dim=(1, 2, 3)).repeat_interleave(4)
        return vals.reshape(1, -1, 1, 1, 1).expand(1, len(vals), 3, 2, 2).clone()


def make_decoder(monkeypatch, pad="hold"):
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: None, raising=False)
    d = cf._Decoder.__new__(cf._Decoder)
    d.vae_type = cf.VaeType.TAEHV
    d.device = torch.device("cpu")
    d.dtype = torch.float32
    d.first_frame_pad = cf.FirstFramePad(pad)
    d.frames_per_chunk = 4
    d.restarts = 0
    d.stream = _Null()
    d.stream.wait_event = lambda e: None
    d.model = StubTaehv()
    d.video_id = None
    d._start_video(0)
    return d


def chunk(i, value, start=cf.ChunkStart.CONTINUE, video=1):
    return cf.ChunkLatents(latents=torch.full((1, 1, 2, 2, 2), value), video_id=video, chunk_index=i,
                           ready=None, start=start)


@pytest.mark.parametrize("pad, cut_frames", [("hold", 4), ("none", 1)])
def test_decoder_restarts_on_a_first_frame_mid_video(no_cuda, monkeypatch, pad, cut_frames):
    d = make_decoder(monkeypatch, pad)
    F = cf.ChunkStart.FIRST_FRAME
    # Values in (0, 1) map to pixels (2v - 1 + 1) * 127.5 = 255 v.
    outs = [d.decode(chunk(0, 0.1, F))] + [d.decode(chunk(i, 0.1 * (i + 1))) for i in range(1, 5)]
    assert [len(o) for o in outs] == [1, 4, 4, 4, 4]  # the first chunk of a video: 1 frame, as before
    cut = d.decode(chunk(5, 0.8, F))
    nxt = d.decode(chunk(6, 0.9))
    # The cut is decoded alone (no old context), like a first chunk; the next one with only the cut as context.
    assert d.model.inputs[-2] == [0.8] and d.model.inputs[-1] == [0.8, 0.9]
    assert len(cut) == cut_frames and (cut == round(0.8 * 255)).all()
    assert len(nxt) == 4 and (nxt == round(0.9 * 255)).all()
    assert d.restarts == 1 and d.video_id == 1 and d.next_chunk_index == 7
    assert d.decode(chunk(7, 0.5)).shape[0] == 4 and d.model.inputs[-1] == [0.8, 0.9, 0.5]


def test_decoder_first_frame_of_a_new_video_is_not_a_restart(no_cuda, monkeypatch):
    d = make_decoder(monkeypatch)
    assert len(d.decode(chunk(0, 0.1, cf.ChunkStart.FIRST_FRAME, video=3))) == 1
    assert d.restarts == 0


def test_first_frame_pad_option_is_checked():
    with pytest.raises(ValueError):
        cf.FirstFramePad("repeat")
    with pytest.raises(ValueError):
        cf.Backend(64, 64, None, "cpu", "cpu", first_frame_pad="repeat")

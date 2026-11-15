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


def test_hard_cut_rebase_call_sequence(no_cuda):
    # keep 0 is the only rebase the live engine uses: a re-anchor is a hard cut of the frame cache.
    b = make_backend()
    outs, calls, _, resets, c = run_then_rebase(b, 5, cf.Rebase(keep=0))
    ctx = int(b.context_noise)
    assert calls[0]["kv_end"] == 0  # frame cache cleared before the first pass
    assert calls[0]["text_init"] is False  # text cache reset, rebuilt from the new cond
    assert all(call["text_init"] for call in calls[1:])
    assert b.crossattn_resets == resets + 1  # once, although the cond also changed
    # The new frame at position 0 runs the first-frame schedule, then the context pass.
    assert [call["t"] for call in calls] == [[int(t)] for t in b.first_chunk_steps] + [[ctx]]
    assert all(call["frame"] == 0 and call["cond"] is c for call in calls)
    assert b.frame_index == 1 and b.chunk_index == 6
    assert b.rebases == 1 and b.recached_frames == 0
    assert b.last_rebase == cf.Rebase(0, 0, None)
    assert outs[-1].chunk_index == 5 and outs[-1].video_id == outs[0].video_id  # no decoder restart
    # The next chunk continues normally from position 1.
    n0 = len(b.generator.calls)
    b.generate_chunk(c)
    nxt = b.generator.calls[n0:]
    assert [call["frame"] for call in nxt] == [1] * 3
    assert [call["t"] for call in nxt] == [[int(t)] for t in b.denoising_steps] + [[ctx]]


@pytest.mark.parametrize("n", [2, 3])
def test_anchor_step_start_latent_and_schedule(no_cuda, n):
    keep = 0
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
        run_then_rebase(b, 4, cf.Rebase(keep=0, anchor_step=4))  # the first-frame schedule has 4 steps


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


def test_reanchor_applies_on_the_next_chunk_only(no_cuda):
    b = make_backend()
    a = torch.full((1, 3, 8), 1.0)
    b.reset(seed=0)
    for _ in range(6):
        b.generate_chunk(a)
    with pytest.raises(ValueError):
        b.reanchor(0.5, source="golden")
    for strength, step in [(1.0, 0), (0.5, 2), (0.25, 3)]:
        b.reanchor(strength)
        n0 = len(b.generator.calls)
        b.generate_chunk(a)
        # A hard cut: the first-frame schedule from the mapped step, at position 0.
        assert [c["t"] for c in b.generator.calls[n0:]] == [[int(t)] for t in b.first_chunk_steps[step:]] + [[0]]
        assert b.last_rebase == cf.Rebase(0, 0, step, False) and b.frame_index == 1
        n0 = len(b.generator.calls)
        b.generate_chunk(a)  # applied once: the next chunk is a normal one
        assert [c["t"] for c in b.generator.calls[n0:]] == [[int(t)] for t in b.denoising_steps] + [[0]]
    rebases = b.rebases
    b.reanchor(0.0)  # zero strength cancels: no cut
    b.generate_chunk(a)
    assert b.rebases == rebases


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
    b = make_backend(refresh_noise="fresh")
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
    b = make_backend(refresh_noise="fresh")
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
    b = make_backend(refresh_noise="fresh")
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


class StubWan:
    """Wan VAE wrapper stand-in: model(z, *cache) -> (pixels, cache), 4 frames per latent, value = latent mean."""

    def __init__(self):
        self.caches_in = []

    def __call__(self, z, *cache):
        self.caches_in.append(float(sum(c.sum() for c in cache)))
        vals = z[0].float().mean(dim=(1, 2, 3)).repeat_interleave(4)
        pixels = vals.reshape(1, -1, 1, 1, 1).expand(1, len(vals), 3, 2, 2).clone() * 2 - 1  # Wan gives [-1, 1]
        return pixels, [c + 1 for c in cache]  # the cache "remembers" every chunk since the last restart


def make_decoder(monkeypatch, pad="hold", fade=0, vae=cf.VaeType.TAEHV):
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: None, raising=False)
    d = cf._Decoder.__new__(cf._Decoder)
    d.vae_type = vae
    d.device = torch.device("cpu")
    d.dtype = torch.float32
    d.first_frame_pad = cf.FirstFramePad(pad)
    d.frames_per_chunk = 4
    d.cut_fade_frames = fade
    d.restarts = 0
    d.stream = _Null()
    d.stream.wait_event = lambda e: None
    if vae is cf.VaeType.WAN:
        d.model = StubWan()
        d._zero_cache_shapes = [(1, 1, 1, 1, 1)]
    else:
        d.model = StubTaehv()
    d.video_id = None
    d._start_video(0)
    return d


def chunk(i, value, start=cf.ChunkStart.CONTINUE, video=1, fade=False):
    return cf.ChunkLatents(latents=torch.full((1, 1, 2, 2, 2), value), video_id=video, chunk_index=i,
                           ready=None, start=start, fade=fade)


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


def test_clean_refresh_after_a_scene_change_without_steer_recaptures(no_cuda):
    """A cond change with no steer re-anchor (e.g. a provisional): the refresh must not bring the old scene back."""
    b = make_backend(refresh_noise="fresh")
    a, c = torch.full((1, 3, 8), 1.0), torch.full((1, 3, 8), 2.0)
    b.reset(seed=0)
    outs = [b.generate_chunk(a) for _ in range(6)]
    assert b.clean_anchor_chunk == 3
    outs += [b.generate_chunk(c) for _ in range(4)]  # the scene changed, no re-anchor
    out, calls, rng = reanchored_chunk(b, c, 0.5, "clean")  # chunk 10
    assert torch.allclose(calls[0]["x"], start_latent(b, outs[-1].latents, 2, rng))  # from the last latent
    assert b.last_rebase == cf.Rebase(0, 0, 2, False)
    assert b._clean_anchor is None
    outs += [out] + [b.generate_chunk(c) for _ in range(3)]  # chunk 13 = 10 + 3: saved again, under c
    assert b.clean_anchor_chunk == 13 and b._clean_anchor is outs[13].latents
    _, calls, rng = reanchored_chunk(b, c, 0.5, "clean")
    assert torch.allclose(calls[0]["x"], start_latent(b, outs[13].latents, 2, rng))
    assert b.last_rebase.from_clean


def test_clean_refresh_accepts_an_equal_cond_object(no_cuda):
    b = make_backend(refresh_noise="fresh")
    b.reset(seed=0)
    outs = [b.generate_chunk(torch.full((1, 3, 8), 1.0)) for _ in range(6)]
    _, calls, rng = reanchored_chunk(b, torch.full((1, 3, 8), 1.0), 0.5, "clean")  # equal values, new object
    assert torch.allclose(calls[0]["x"], start_latent(b, outs[3].latents, 2, rng))
    assert b.last_rebase.from_clean


# ---------------------------------------------------------------- cut fade (dissolve after a re-anchor)


def px(v):
    """The pixel value the stub decoders give a latent of value v: (2v - 1 + 1) * 127.5."""
    return round(v * 255)


def faded(prev, new, ks, n):
    """Expected pixel values of output frames number ks after a cut, fading from `prev` to `new` over n frames."""
    out = []
    for k in ks:
        a = min((k + 1) / (n + 1), 1.0)
        out.append(round((1 - a) * prev + a * new))
    return out


def values(frames):
    return [int(f[0, 0, 0]) for f in frames]


def run_cut(d, fade_flag=True):
    """5 chunks of a video, a re-anchor cut, then 3 more chunks; returns all outputs."""
    F = cf.ChunkStart.FIRST_FRAME
    outs = [d.decode(chunk(0, 0.1, F))] + [d.decode(chunk(i, 0.1 * (i + 1))) for i in range(1, 5)]
    outs.append(d.decode(chunk(5, 0.8, F, fade=fade_flag)))
    outs += [d.decode(chunk(i, 0.9)) for i in range(6, 9)]
    return outs


@pytest.mark.parametrize("vae", [cf.VaeType.TAEHV, cf.VaeType.WAN])
def test_cut_fade_weights_span_chunk_boundaries(no_cuda, monkeypatch, vae):
    d = make_decoder(monkeypatch, fade=8, vae=vae)
    outs = run_cut(d)
    last = px(0.5)  # the last frame shown before the cut (chunk 4)
    assert values(outs[4])[-1] == last
    # Output frames 0..7 after the cut: a = (k + 1) / 9, from the last shown frame to the new picture.
    want_cut = faded(last, px(0.8), range(0, 4), 8)  # the cut chunk: its 1 frame held to 4, faded
    want_next = faded(last, px(0.9), range(4, 8), 8)  # the next chunk: the fade continues across the boundary
    assert all(abs(g - w) <= 1 for g, w in zip(values(outs[5]), want_cut)), (values(outs[5]), want_cut)
    assert all(abs(g - w) <= 1 for g, w in zip(values(outs[6]), want_next)), (values(outs[6]), want_next)
    assert values(outs[7]) == [px(0.9)] * 4 and values(outs[8]) == [px(0.9)] * 4  # done after 8 frames
    assert [len(o) for o in outs[5:]] == [4, 4, 4, 4]
    # The weights rise strictly: no frame goes back towards the old picture.
    seq = values(outs[5]) + values(outs[6])
    assert all(b > a for a, b in zip(seq, seq[1:]))


@pytest.mark.parametrize("vae", [cf.VaeType.TAEHV, cf.VaeType.WAN])
def test_cut_fade_off_by_default_and_without_the_flag(no_cuda, monkeypatch, vae):
    plain = run_cut(make_decoder(monkeypatch, fade=0, vae=vae))
    assert values(plain[5]) == [px(0.8)] * 4 and values(plain[6]) == [px(0.9)] * 4
    no_flag = run_cut(make_decoder(monkeypatch, fade=8, vae=vae), fade_flag=False)
    assert [values(o) for o in no_flag] == [values(o) for o in plain]


def test_cut_fade_never_on_a_new_video(no_cuda, monkeypatch):
    d = make_decoder(monkeypatch, fade=8)
    F = cf.ChunkStart.FIRST_FRAME
    for i in range(5):
        d.decode(chunk(i, 0.1 * (i + 1), F if i == 0 else cf.ChunkStart.CONTINUE))
    out = d.decode(chunk(0, 0.8, F, video=2, fade=True))  # reset(): a new video, even with the flag set
    assert values(out) == [px(0.8)]
    assert values(d.decode(chunk(1, 0.9, video=2))) == [px(0.9)] * 4


def test_cut_fade_with_pad_none_counts_emitted_frames(no_cuda, monkeypatch):
    d = make_decoder(monkeypatch, pad="none", fade=8)
    outs = run_cut(d)
    last = px(0.5)
    assert len(outs[5]) == 1 and abs(values(outs[5])[0] - faded(last, px(0.8), [0], 8)[0]) <= 1
    assert all(abs(g - w) <= 1 for g, w in zip(values(outs[6]), faded(last, px(0.9), range(1, 5), 8)))
    assert all(abs(g - w) <= 1 for g, w in zip(values(outs[7]), faded(last, px(0.9), range(5, 9), 8)))
    assert values(outs[7])[-1] == px(0.9) and values(outs[8]) == [px(0.9)] * 4


def test_second_cut_during_a_fade_starts_from_the_last_shown_frame(no_cuda, monkeypatch):
    d = make_decoder(monkeypatch, fade=8)
    F = cf.ChunkStart.FIRST_FRAME
    for i in range(5):
        d.decode(chunk(i, 0.1 * (i + 1), F if i == 0 else cf.ChunkStart.CONTINUE))
    first = d.decode(chunk(5, 0.8, F, fade=True))
    shown = values(first)[-1]
    second = d.decode(chunk(6, 0.2, F, fade=True))  # another cut before the first fade ended
    assert all(abs(g - w) <= 1 for g, w in zip(values(second), faded(shown, px(0.2), range(0, 4), 8)))


def test_wan_cut_restarts_the_feature_cache(no_cuda, monkeypatch):
    d = make_decoder(monkeypatch, vae=cf.VaeType.WAN)
    run_cut(d)
    # caches_in: the summed cache each call got. Chunk 0 and the cut (chunk 5) start from zeros.
    assert d.model.caches_in[0] == 0 and d.model.caches_in[5] == 0
    assert d.model.caches_in[4] > 0 and d.model.caches_in[6] > 0


def test_backend_sets_the_fade_flag_on_re_anchor_chunks(no_cuda):
    a = torch.full((1, 3, 8), 1.0)
    for fade, refresh, want_steer, want_refresh in [(8, True, True, True), (8, False, True, False), (0, True, False, False)]:
        b = make_backend(cut_fade_frames=fade, cut_fade_refresh=refresh)
        b.reset(seed=0)
        outs = [b.generate_chunk(a) for _ in range(10)]
        assert not any(o.fade for o in outs)  # never on the first chunk of a video or a continuation
        b.reanchor(1.0, source="last")
        steer = b.generate_chunk(a)
        outs = [b.generate_chunk(a) for _ in range(5)]
        b.reanchor(0.5, source="clean")
        refresh_chunk = b.generate_chunk(a)
        assert (steer.fade, refresh_chunk.fade) == (want_steer, want_refresh)
        assert not any(o.fade for o in outs)
        b.generate_chunk(a, rebase=cf.Rebase(keep=2))  # keep >= 1: no cut, no fade
        assert not b.generate_chunk(a).fade


def test_cut_fade_options_are_checked():
    with pytest.raises(ValueError):
        make_backend(cut_fade_frames=-1)


def test_cut_without_fade_ends_a_running_dissolve(no_cuda, monkeypatch):
    d = make_decoder(monkeypatch, fade=8)
    F = cf.ChunkStart.FIRST_FRAME
    for i in range(5):
        d.decode(chunk(i, 0.1 * (i + 1), F if i == 0 else cf.ChunkStart.CONTINUE))
    d.decode(chunk(5, 0.8, F, fade=True))  # a steer: the dissolve starts
    out = d.decode(chunk(6, 0.2, F, fade=False))  # a refresh with cut_fade_refresh=0, mid-dissolve
    assert values(out) == [px(0.2)] * 4
    assert values(d.decode(chunk(7, 0.3))) == [px(0.3)] * 4


# ---------------------------------------------------------------- refresh noise and start timestep


def refreshes(b, n_refresh=2, gap=6):
    """reset, 6 chunks, then n_refresh clean refreshes `gap` chunks apart; returns [(out, calls)] per refresh."""
    a = torch.full((1, 3, 8), 1.0)
    b.reset(seed=0)
    for _ in range(6):
        b.generate_chunk(a)
    got = []
    for _ in range(n_refresh):
        out, calls, _ = reanchored_chunk(b, a, 0.5, "clean")
        got.append((out, calls))
        for _ in range(gap):
            b.generate_chunk(a)
    return got


def test_fixed_refresh_noise_gives_the_same_picture_every_refresh(no_cuda):
    (o1, c1), (o2, c2) = refreshes(make_backend(refresh_noise="fixed"))
    assert torch.equal(c1[0]["x"], c2[0]["x"])  # same start latent
    assert torch.equal(o1.latents, o2.latents)  # and the same denoised picture
    (f1, d1), (f2, d2) = refreshes(make_backend(refresh_noise="fresh"))  # fresh: a new sample each time
    assert not torch.equal(d1[0]["x"], d2[0]["x"])


def test_fixed_refresh_noise_changes_with_the_clean_anchor(no_cuda):
    b = make_backend(refresh_noise="fixed")
    a, c = torch.full((1, 3, 8), 1.0), torch.full((1, 3, 8), 2.0)
    b.reset(seed=0)
    for _ in range(6):
        b.generate_chunk(a)
    _, before, _ = reanchored_chunk(b, a, 0.5, "clean")
    reanchored_chunk(b, c, 1.0, "last")  # a steer: a new clean anchor 3 chunks later
    for _ in range(5):
        b.generate_chunk(c)
    _, after, _ = reanchored_chunk(b, c, 0.5, "clean")
    assert b.last_rebase.from_clean and not torch.equal(before[0]["x"], after[0]["x"])


def test_fixed_refresh_noise_leaves_steers_and_normal_chunks_alone(no_cuda):
    fixed, fresh = make_backend(refresh_noise="fixed"), make_backend()
    a, c = torch.full((1, 3, 8), 1.0), torch.full((1, 3, 8), 2.0)
    outs = {}
    for name, b in (("fixed", fixed), ("fresh", fresh)):
        b.reset(seed=0)
        seq = [b.generate_chunk(a) for _ in range(6)]
        seq.append(reanchored_chunk(b, c, 1.0, "last")[0])  # a steer: drawn from the main stream
        outs[name] = [o.latents for o in seq]
    assert all(torch.equal(x, y) for x, y in zip(outs["fixed"], outs["fresh"]))


@pytest.mark.parametrize("t, want_steps", [(125, None), (250, 1), (500, 2)])
def test_refresh_anchor_t_sets_the_start_level(no_cuda, t, want_steps):
    b = make_backend(refresh_anchor_t=t, refresh_noise="fresh")
    a = torch.full((1, 3, 8), 1.0)
    b.reset(seed=0)
    outs = [b.generate_chunk(a) for _ in range(6)]
    _, calls, rng = reanchored_chunk(b, a, 0.5, "clean")
    tw = b._timestep(t)
    sched = [tw] + [s for s in b.first_chunk_steps if s < tw]
    assert [c["t"] for c in calls] == [[int(s)] for s in sched] + [[0]]
    if want_steps is not None:
        assert len(sched) == want_steps  # 250 and 500 are on the first-frame grid (steps 3 and 2)
    g = torch.Generator().set_state(rng)
    noise = torch.randn(outs[3].latents.shape, generator=g)
    want = b.scheduler.add_noise(outs[3].latents.flatten(0, 1), noise.flatten(0, 1), tw * torch.ones([1], dtype=torch.long))
    assert torch.allclose(calls[0]["x"], want.unflatten(0, outs[3].latents.shape[:2]))
    assert b.last_rebase.from_clean


def test_refresh_anchor_t_only_for_refreshes_from_the_clean_anchor(no_cuda):
    b = make_backend(refresh_anchor_t=125)
    a = torch.full((1, 3, 8), 1.0)
    b.reset(seed=0)
    for _ in range(6):
        b.generate_chunk(a)
    _, calls, _ = reanchored_chunk(b, a, 0.5, "last")  # a steer keeps the strength mapping (step 2)
    assert [c["t"] for c in calls] == [[int(s)] for s in b.first_chunk_steps[2:]] + [[0]]


@pytest.mark.parametrize("options", [{"refresh_noise": "same"}, {"refresh_anchor_t": 0}, {"refresh_anchor_t": 1001}])
def test_refresh_options_are_checked(options):
    with pytest.raises(ValueError):
        make_backend(**options)


def test_refresh_noise_is_fixed_by_default_and_seeded_by_the_anchor(no_cuda):
    b = make_backend()
    assert b.refresh_noise == "fixed" and b.refresh_anchor_t is None
    a = torch.full((1, 3, 8), 1.0)
    b.reset(seed=0)
    outs = [b.generate_chunk(a) for _ in range(6)]
    _, calls, _ = reanchored_chunk(b, a, 0.5, "clean")
    g = torch.Generator().manual_seed(1_000_003 * b.video_id + 3)
    noise = torch.randn(outs[3].latents.shape, generator=g)
    t = b.first_chunk_steps[2] * torch.ones([1], dtype=torch.long)
    want = b.scheduler.add_noise(outs[3].latents.flatten(0, 1), noise.flatten(0, 1), t).unflatten(0, outs[3].latents.shape[:2])
    assert torch.allclose(calls[0]["x"], want)

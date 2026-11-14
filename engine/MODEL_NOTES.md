# Model notes: Causal Forcing++ on one L40S

## What runs

The base model is Causal Forcing++ (framewise, 2 denoising steps, 4 steps on the first frame of a video), built on Wan2.1-T2V-1.3B.
It is the official code from github.com/thu-ml/Causal-Forcing at commit eb8705b2f00b7c361193b25aa3aae9e7edc94667.
The chunkwise Causal Forcing model (3 latent frames per chunk, 4 steps) is also loadable with `variant="chunkwise_4step"` and was measured for comparison.
No code is copied into this repo; the backend imports the official checkout from `~/third_party/Causal-Forcing` (or `CAUSAL_FORCING_REPO`, or the `repo_dir` option).

| Item | Source | License | Where the license is stated |
|---|---|---|---|
| Causal Forcing code | github.com/thu-ml/Causal-Forcing | Apache-2.0 | `LICENSE` in the repo root |
| Causal Forcing weights | huggingface.co/zhuhz22/Causal-Forcing | Apache-2.0 | `license: apache-2.0` in the model card metadata (README.md) |
| Wan2.1-T2V-1.3B (DiT, umT5-xxl, VAE) | huggingface.co/Wan-AI/Wan2.1-T2V-1.3B | Apache-2.0 | `LICENSE.txt` in the model repo and the model card |
| TAEHV tiny decoder `taew2_1.pth` | github.com/madebyollin/taehv | MIT | `LICENSE` in the taehv repo; the official demo downloads it from there |
| Self-Forcing weights (downloaded, not used) | huggingface.co/gdhe17/Self-Forcing | Apache-2.0 | model card metadata |

None of these needed a token or a license click-through.

## Layout on the GPU machine

- Code: `~/third_party/Causal-Forcing` with symlinks `wan_models -> ~/weights` and `checkpoints -> ~/weights/Causal-Forcing` (the official code opens weights through these relative paths).
- Weights: `~/weights/Wan2.1-T2V-1.3B`, `~/weights/Causal-Forcing/{causal-forcing++,chunkwise}`, `~/weights/taehv/taew2_1.pth`, `~/weights/Self-Forcing`.
- Python: `~/engine-venv` (Python 3.10, torch 2.5.1+cu124, flash-attn 2.7.4.post1).
- Our repo: `~/madhacks`, synced with rsync from a laptop clone.
- Runs and logs: `~/runs`.
- `engine/setup_gpu.sh` rebuilds all of this on a fresh machine.

## Units

One `generate_chunk(cond)` call makes one latent frame, which decodes to 4 video frames (0.25 s at 16 fps).
The first call of a video yields 1 video frame, because the official streaming decode drops the first 3 frames.
So the budget for real time is 250 ms per call, which is 750 ms per 0.75 s of video.

## Measured on one L40S (Gate 1)

All rows: Causal Forcing++ framewise, 2 steps, 1 latent frame per call, attention window 21 frames, sink 0, bf16, flash-attn 2, no torch.compile.
Numbers are medians over calls 25 to 70, after the frame cache is full (generation time grows with the cache until call 21, then stays flat).
"Threaded" is generation in one thread and decode in another through a queue, both on the same GPU; it is the steady-state wall-clock time per call.
Budget for real time: 250 ms per call (750 ms per 0.75 s of video).

| Size | VAE | Gen ms | Decode ms | Gen + decode in sequence | Threaded wall ms per call | Per 0.75 s of video | Peak GPU alloc | Real time |
|---|---|---|---|---|---|---|---|---|
| 480x832 | Wan (stock) | 417 | 244 | 661 | 656 | 1968 | 24.6 GiB | no |
| 480x832 | TAEHV | 417 | 33 | 449 | 443 | 1329 | 19.5 GiB | no |
| 384x672 | Wan (stock) | 207 | 155 | 362 | 361 | 1082 | 20.7 GiB | no |
| 384x672 | TAEHV | 210 | 21 | 231 | 226 | 677 | 17.4 GiB | yes, about 10% margin |

On one GPU the two threads share the device, so in threaded mode the generation call takes as long as generation plus decode, and the wall time per call is close to the sequential sum.
The win of the threaded layout comes only when decode moves to a second GPU (`decode_device`); then the wall time per call would be the generation time alone.
Recommended real-time default on one L40S: 384x672, TAEHV, 2 steps, window 21.
Best quality that is close: 384x672 with the stock Wan VAE needs a second GPU for decode (generation 207 ms alone fits the budget).
480x832 does not fit on this GPU even with decode elsewhere (generation alone is 417 ms per call); a smaller attention window would cut that but was not measured yet.
Smaller sizes (320x576, 240x416) and the chunkwise model were not measured, because 384x672 already fits.

Memory with everything loaded (generator, umT5-xxl text encoder, decoder, frame cache) is about 17.0 GiB at 384x672 and 19.1 GiB at 480x832.
Encoding one prompt with umT5-xxl on this GPU takes about 53 ms warm (about 360 ms for the first call).
Loading takes about 70 s, mostly reading the 5.7 GB checkpoint and the umT5 weights.

The server passes `--steps 4` by default; for this model pass `--steps 2` or `--steps 0` (0 means the distilled schedule), otherwise the backend runs an evenly spaced 4-step schedule that is slower and was not distilled for.

## Endless generation

The official model keeps a self-attention KV cache of past frames (the frame cache).
With the default `local_attn_size=-1` the cache has room for 21 latent frames and generation breaks after that.
The official code also has a local-attention path: with `local_attn_size=N` (and optional `sink_size=S`) the cache is a fixed buffer of N frames, and when it is full the oldest non-sink frames are shifted out (see `CausalWanSelfAttention.forward` in `wan/modules/causal_model.py`).
The backend turns this on with `attn_window_frames` (default 21, the length the model was trained on) and `sink_frames` (default 0).
The keys in the cache already have RoPE applied at their absolute frame index, so the frame index keeps growing.
The official RoPE table has 1024 positions, which would stop generation after 1024 latent frames (about 4.3 minutes).
The backend rebuilds the same table with the same formula and 65536 positions (`max_latent_frames`), which is about 4.5 hours of video; past that, `generate_chunk` raises and the server should call `reset()`.
With `sink_frames=0` attention only sees distances inside the window, so the large absolute positions do not change what the model sees.
Memory is fixed after the first 21 calls because the cache buffer never grows.

Test (384x672, TAEHV, threaded, one prompt): the run was planned for 1200 calls (5 minutes of video, past the old 1024-position RoPE limit) but was stopped at about call 460 (about 115 s of video) to free the GPU for the live server.
Up to that point generation time stayed flat at 222 to 225 ms per call from call 50 to call 450 (one logged sample at 244 ms), decode stayed at 32 ms, and peak GPU memory stayed at 17.39 GiB.
Because the run was cut short, no mp4 from late in the run was saved and late-run picture quality was not checked by eye; the 90-call switch test below (22 s of video) still looked sane at its end.
The path past 1024 latent frames (the rebuilt RoPE table) has not been exercised yet.

## Conditioning and the second cache

Each call is conditioned on `cond`, the umT5-xxl output for the prompt: shape [1, 512, 4096], dtype bfloat16, padding rows set to zero.
It is passed to the model as `conditional_dict["prompt_embeds"]`; the streaming path has no classifier-free guidance, so it is one tensor.
The Wan model caches the text cross-attention keys and values per block (`crossattn_cache`, flag `is_init` in `WanT2VCrossAttention.forward`, `wan/modules/model.py`).
If that cache is not reset, a new `cond` is ignored.
`generate_chunk` resets it (sets `is_init=False` in all 30 blocks) whenever `cond is not self._last_cond`, so the steerer must hand over a new tensor object when the conditioning changes and the same object when it does not, and must never modify a tensor in place after handing it over.
Recomputing the text K/V is 30 small linear layers on 512 tokens, which is too small to see in the per-call timing.
The frame cache is not touched by a conditioning change (unless a rebase is asked for, see below).

Test (384x672, TAEHV, threaded): prompt A "a red fox walking through a snowy pine forest at sunrise" for calls 0 to 29, then prompt B "the same at night under a full moon, dark blue moonlight" from call 30 on, as a hard switch.
The video kept the same fox, trees and snow and did not restart.
The light changed gradually: frames right after the switch look like before, the scene is clearly dimmer about 4 s later, and dark blue moonlight about 14 s later.
The slow response is expected, because the frame cache still holds 21 frames (about 5 s) made under prompt A.
The median call time of that run (230 ms) was close to the run without a switch (226 ms); the single call that recomputed the text K/V was not timed on its own.

Seeding: `reset(seed=N)` reseeds all noise of the new video; two runs with the same seed and the same `cond` gave bit-identical latents (checked on 4 calls).

## Rebase: restarting the frame cache (experimental, off by default)

Two problems were measured on this model: after a prompt switch the new text is read but the scene barely changes, because every new frame attends to the cached clean frames made under the old prompt; and the picture decays after 20 to 50 s while the frame position grows without bound.
The training-free fix is the KV recache of LongLive (Yang et al., "LongLive: Real-time Interactive Long Video Generation", arXiv 2509.22622, Apache-2.0): at the switch, rebuild the frame cache from the last few clean frames re-encoded under the new prompt.
Our version also restarts the positions at 0, so a periodic rebase keeps positions inside the trained 21-frame range.

`generate_chunk(cond, rebase=Rebase(keep=3, memory_noise=0, anchor_step=None))` (or a dict with those keys), framewise variants only:

1. The text K/V cache is reset, and the frame cache is cleared the same way `reset()` does it (end indices to 0). The RNG stream, the decoder, the chunk counter and the video id are kept.
2. The last `keep` clean latents (a ring buffer of references, at most `attn_window_frames - 1`) are re-encoded oldest first, one frame per pass, at positions 0..keep-1, under the current cond, with the normal context pass (timestep `context_noise`). With `memory_noise=t` (0-1000) each is first noised to timestep t and passed with timestep t, which loosens how strongly the old frames pin the scene. t is warped onto the shifted grid like the denoising steps, so `memory_noise=500` is the noise level of the schedule's "500" step.
3. The new frame is made at position `keep`. With `keep=0` it is a first frame again (the 4-step first-frame schedule): a hard cut. `anchor_step=n` starts it from the last clean latent noised to step n of its schedule and denoises with `schedule[n:]` (n=0 is pure noise; larger n keeps more of the old picture). With `keep>0` the anchor uses the normal 2-step schedule (experimental).
4. Generation then continues normally and the window refills by itself.

Decoding is not restarted: the chunk keeps its place in the stream, and both decoders (`wan` with its feature cache, `taehv` with 3 latents of context) still give 4 video frames for it.
After `keep=0` that latent is a first-frame latent, which the Wan VAE normally decodes as 1 frame; here it is decoded as a 4-frame latent with the old scene in the decoder's temporal context.
Expect a short (about 0.25 s) blend or near-freeze, and possibly a one-chunk brightness or colour pop; this has not been looked at on the GPU yet.

Automatic mode, for the live server with no engine change (`--backend-opt k=v`): `steer_rebase=1` fires on the first chunk whose cond differs from the previous one (`steer_fire_on_change=1`, default) and again when the cond stops changing (the end of a ramp), never on the first chunk of a video and never within `steer_min_gap` chunks (8) of the last rebase or reset; it uses `steer_keep` (3), `steer_memory_noise` (0) and `steer_anchor_step` (-1 = none). `refresh_every=N` (0 = off) rebases with `refresh_keep` (6) and `refresh_memory_noise` (0) every N chunks to keep positions small. An explicit `rebase=` always wins for that chunk.
Counters for logs and tests: `rebases`, `recached_frames`, `auto_rebases`, `last_rebase`; `first_frame_steps` and `anchor_step_for_strength(s)` map a 0..1 strength to an anchor step.
With everything off, the calls, the RNG draws and their order are exactly as before (checked by `engine/tests/test_causal_forcing.py` against a recording of the old code).
The experiment driver is `python -m steering.eval.follow_probe` (see `steering/eval/README.md`).

### Re-anchor and the clean anchor (the periodic refresh)

`reanchor(strength, source="last")` queues `Rebase(keep=0, anchor_step=anchor_step_for_strength(strength))` for the next chunk: a first frame made with an empty frame cache.
With the 4-step first-frame schedule: strength >= 0.76 gives step 0 (full repaint from noise), 0.51-0.75 step 1, 0.26-0.5 step 2, <= 0.25 step 3.

The backend also keeps a **clean anchor**: the latent of the chunk `clean_anchor_after` (3) chunks after `reset()` or after any rebase that does not start from the clean anchor (a steer re-anchor).
`reanchor(strength, source="clean")` starts the first frame from the clean anchor noised to the anchor step instead of from the last latent, and keeps the clean anchor as it is.
If there is no clean anchor yet (the first 3 chunks of a video or of a steer), it falls back to the last latent and changes nothing.
`clean_anchor_chunk` says which chunk the current clean anchor came from; `last_rebase.from_clean` says whether the last rebase used it.
`ReanchorPolicy` sends steers with source "last" and refreshes with `refresh_source` ("clean").

Why: a refresh started from the last latent starts from a picture that has already drifted, so each refresh bakes the damage in (over-contrast, blocky neon grass).
Measured over 360 chunks with no commands (672x384, 2 steps, taehv, `quick_gold.py` on the GPU box):

| Refresh | Result |
|---|---|
| from the last latent, anchor step 1 every 40 chunks | degrades from about chunk 180 |
| full restart from noise (step 0) every 60 chunks | the scene and colours jump on every restart; unusable |
| from the clean anchor (saved at chunk 3), step 1 or 2, every 24 or 40 chunks | clean and the same scene for all 360 chunks |
| from the clean anchor, step 2 every 24 chunks | the steadiest; now the default (`refresh_every_chunks=24`, `refresh_strength=0.5`, `refresh_source="clean"`) |

## Using the backend from a plain Python script

Run from `~/madhacks` with `~/engine-venv` active:

```python
from engine.backends import causal_forcing
b = causal_forcing.load(672, 384, vae="taehv")     # recommended real-time config; steps=None: distilled schedule
cond = b.encode_prompt("a meadow at noon")          # [1, 512, 4096] bf16 on cuda:0
emb, ids = b.encode_prompt_with_ids("a meadow")     # same embedding plus the real umT5 token ids
b.reset(seed=0)                                     # start a new video; same seed + same conds = same latents
lat = b.generate_chunk(cond)                        # one latent frame
frames = b.decode_chunk(lat)                        # uint8 [T, H, W, 3]
```

`encode_prompt` may be called from another thread while `generate_chunk` runs; it has its own lock and CUDA stream.
`decode_chunk` may be called from another thread, in chunk order; the decoder keeps its own state and resets it when it sees the first chunk of a new video, so `reset()` never touches decoder state directly.
Generation, encoding and decoding each run on their own CUDA stream; `generate_chunk` waits for the default stream first, so a `cond` blended on another thread is safe to pass.

## Reproduce

```bash
bash engine/setup_gpu.sh                      # once, inside tmux
cd ~/madhacks && source ~/engine-venv/bin/activate
python -m engine.bench_stock --width 832 --height 480 --chunks 70 --warmup 25            # gen and decode separately
python -m engine.bench_stock --width 832 --height 480 --chunks 70 --warmup 25 --threaded # engine-style threads
python -m engine.bench_stock --width 672 --height 384 --vae taehv --threaded --chunks 90 --switch-at 30 \
  --prompt-b "A red fox walking through a snowy pine forest at night under a full moon" --save-mp4 switch.mp4
```

With the server, use `--backend causal_forcing --width 672 --height 384 --steps 0 --backend-opt vae=taehv`.

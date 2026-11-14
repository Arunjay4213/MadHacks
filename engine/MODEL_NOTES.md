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
The frame cache is never touched by a conditioning change.

Test (384x672, TAEHV, threaded): prompt A "a red fox walking through a snowy pine forest at sunrise" for calls 0 to 29, then prompt B "the same at night under a full moon, dark blue moonlight" from call 30 on, as a hard switch.
The video kept the same fox, trees and snow and did not restart.
The light changed gradually: frames right after the switch look like before, the scene is clearly dimmer about 4 s later, and dark blue moonlight about 14 s later.
The slow response is expected, because the frame cache still holds 21 frames (about 5 s) made under prompt A.
The median call time of that run (230 ms) was close to the run without a switch (226 ms); the single call that recomputed the text K/V was not timed on its own.

Seeding: `reset(seed=N)` reseeds all noise of the new video; two runs with the same seed and the same `cond` gave bit-identical latents (checked on 4 calls).

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

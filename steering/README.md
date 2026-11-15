# steering

Owner `@Tactacion` (Ishaan).
Moves the text-prompt conditioning from its current value toward a new target over a few video chunks, and decides when the backend must re-anchor the picture so the change shows.
The steerer is model-independent: it only sees tensors whose last dimension is the embedding dim.
It runs on CPU or GPU, with any float dtype (bfloat16 included).

The backend side of re-anchoring (the clean anchor, the decoder restart and the cut fade) lives in `engine/backends/causal_forcing.py`, which Ishaan also writes; it is described here under "Re-anchoring on the real model" because it is part of the same mechanism.

## The two calls

```python
from steering import Steerer

steerer = Steerer(initial_embedding)       # once, with the starting prompt

# control thread, when a command's embedding is ready (contract 2 -> 3)
steerer.set_target(new_embedding, ramp_chunks=4)            # full move
steerer.set_target(guess_embedding, ramp_chunks=4, cap=0.3)  # provisional

# generation loop, once per chunk
cond = steerer.get_cond()
```

- A new `set_target` starts from wherever the conditioning is right now, so commands can interrupt each other.
- With `cap < 1` the move stops part way.
  Call `set_target` again with the same target to confirm, or a different one to cancel.
- `steerer.progress` (0 to 1) and `steerer.is_moving` report the current move.
- `steerer.reset(cond)` jumps to `cond` with no move (a new stream; the shape may change).
- `steerer.current` is the tensor `get_cond` last returned (the initial one before any call), read without advancing the move.

Also exported:

- `Aligner` (see below).
- `blend(a, b, w)`: per-token slerp of direction and lerp of norm, with a linear fallback for zero or (anti)parallel tokens; w in [0, 1.3].
- `apply_strength(without, with_change, s)`: the same as `blend`, for "a little" and "a lot more".
- `ease(r)`: the min-jerk pacing curve.

## Controller (contract 2 -> steerer)

```python
from steering import Controller

ctl = Controller(steerer, encode)          # encode: prompt str -> embedding tensor
ctl.precompute(keyword_prompts)            # early-steering targets, encoded up front
ctl.handle({"prompt": p, "ramp_chunks": 4, "provisional": True, "cap": 0.4})
ctl.handle({"prompt": p, "ramp_chunks": 4})  # final: confirm, or replace a wrong guess
# -> {"applied": bool, "kind": "provisional"|"confirm"|"final"|None,
#     "reason": None|"unchanged"|"stale", "cache_hit": bool}
```

- Embeddings are cached (LRU, `cache_size=64`) by exact prompt string.
  `cache_size` must be at least the number of precomputed prompts.
- `encode` runs outside the controller's lock.
  When messages race, the one whose `handle` started last wins; an older one that finishes later is dropped with `reason: "stale"`.
- A final message equal to the committed prompt (nothing pending), or a provisional one equal to the committed or pending prompt, is a no-op (`reason: "unchanged"`).
- Bad messages raise ValueError (a final message must have cap 1.0); on any error nothing changes.
- Optional `strength` (a number in (0, 1.3]) and `without_prompt` (a non-empty string) come both or neither.
  The target is then `blend(E(without_prompt), E(prompt), strength)`; 1.0 is exactly `E(prompt)`.
  Both prompts go through the same cache; `cache_hit` is True only if both were hits.
  A provisional message may carry them too (`cap` applies on top).
- "Unchanged" and "confirm" compare the full target `(prompt, strength, without_prompt)`: the same prompt at a new strength is a new "final".
  `committed_prompt` and `pending_prompt` return the prompt only.

```python
cond = ctl.reset(prompt, strength=None, without_prompt=None)  # reset tap: new video
```

- It encodes outside the lock, then calls `steerer.reset(cond)` (a jump with no ramp; idle, progress 1.0; the shape may differ).
- It commits that target, clears any pending guess, and makes every `handle` call already in flight "stale".
- With an aligner the prompt is placed in the identity layout.
- On any error nothing changes.
- It returns the tensor now in the steerer.

## Aligner (keep shared tokens at the same positions)

Prompts are encoded and zero-padded after the real tokens, so appending or removing a phrase shifts later tokens, and a per-position blend mixes unrelated rows.
The video model's text cross-attention has no positional information for text tokens, and padding rows are zero anywhere, so row order does not matter to it (read from the model code).
`Aligner` moves each real token row so tokens shared with the previous prompt keep their position; new tokens take free positions, preferring ones the previous prompt did not use, so they fade in from zero.
The live engine does not use the aligner.

```python
from steering import Aligner, Controller, Steerer

aligner = Aligner(length=512)
# encode: prompt -> (embedding [1, 512, D] with padding zeroed, real token ids)
emb, ids = encode(initial_prompt)
steerer = Steerer(aligner.place(emb, ids))   # REQUIRED: initial cond in the aligner's layout
ctl = Controller(steerer, encode, aligner=aligner)
```

- `aligner.place(emb, ids)` returns a new tensor and remembers the layout.
  The first call (or the first after `reset()`) is the identity.
  The output is the real rows permuted plus exact zeros; nothing is lost or duplicated.
- `prepare(emb, ids)` returns `(placed, layout)` without committing; `commit(layout)` commits it.
- `place` and `prepare` take an optional `occupied` (a bool tensor or a sequence of `length`): positions still non-zero in the conditioning now, for example an older layout fading out after a capped or interrupted move.
  New tokens take free positions in this order: unused by the previous layout and not occupied; not occupied; unused by the previous layout; any.
  Matched tokens and the first call ignore it.
  `Controller` passes the rows of `steerer.current` that have any non-zero element.
- `prepare(..., against=layout)` matches against `layout` instead of the committed one and leaves the committed one alone.
  `Controller` places `without_prompt` against the prompt's new layout, so shared tokens blend with themselves and prompt-only tokens with zero rows; only the prompt's layout is ever committed.
  `against=()` is the identity placement (used by `Controller.reset`).
- With an aligner, `encode` must return `(emb, ids)` (TypeError otherwise).
  The cache keeps that raw pair; placement happens in `handle` just before `set_target`, only for applied messages, and the layout is committed only if `set_target` succeeds.
  `Controller` raises ValueError if the aligner has no layout yet (the initial prompt was not placed).
- Without an aligner nothing changes: `encode` returns a tensor.

## Tests

```
python3 -m venv ../.venv-steering
../.venv-steering/bin/pip install -r steering/requirements.txt
../.venv-steering/bin/python -m pytest steering/tests -q
```

Run from the repo root.

## Re-anchor policy (`steering/reanchor.py`)

Moving the embedding is not enough on the real model: it keeps copying the frames it already made (ask for winter and snow falls on a green meadow).
A re-anchor makes the backend drop its memory of old frames and generate the next frame from a noised copy of a picture under the current conditioning.
Strength is in (0, 1]: 1 starts from pure noise and keeps only what the prompt says, and a small strength keeps almost everything.
`ReanchorPolicy` decides when and how hard; it has no model code.
The engine turns it on with `--reanchor on` and sets its fields with `--reanchor-opt KEY=VALUE`.

| Case | Anchor |
|---|---|
| Result not applied (stale, unchanged) | nothing; a pending steer is kept |
| `final` / `confirm`, `ramp_chunks <= sudden_max_ramp` (2) | `sudden_strength` (1.0) once |
| `final` / `confirm`, ramp 3 to 9 | `steer_strength` (0.85) once |
| `final` / `confirm`, `ramp_chunks >= slow_min_ramp` (10) | `slow_anchors` (3) x `slow_strength` (0.5) at progress i/n, so the last is at 1.0 (the full target) |
| message `strength` s | strength x min(s, 1) |
| `provisional` | `provisional_strength` (0, so never) x min(cap, 1) |
| scaled strength < `min_strength` (0.05) | nothing, and it cancels a pending steer |
| single anchor timing | first chunk with progress >= `fire_at_progress` (1.0, the full target) |
| newer applied target before it fires | replaces it (latest wins) |
| ramp ended before a threshold | fires on the next allowed chunk |
| any two anchors | at least `min_gap_chunks` (8) apart; a blocked steer waits, thresholds due together fire once |
| idle (not moving, nothing pending) for `refresh_every_chunks` (24) | `refresh_strength` (0.5), reason "refresh", source `refresh_source` ("clean") |
| any steer anchor | source "last" |
| `on_reset()` (new video) | drops pending; gap and refresh count from the reset |

`commands/` sends `ramp_chunks` 1 for a normal command, so on the live engine a normal command is a "sudden" steer: one anchor at strength 1.0 on the next chunk.

Fire timing is measured, not guessed.
On the real model the anchor repaints with whatever conditioning is current on its chunk, and the scene then stays locked to it.
Firing half-way through the ramp (the old 0.5) repainted with a half-blended prompt, so "make it winter" gave a new green meadow; firing once the ramp has reached the full target works (PR #57, Gate 2).
So a single anchor waits for progress 1.0, and a slow steer's last anchor is at 1.0.
The steer strengths and the gap are still untuned.

With the default gap of 8, a slow ramp shorter than 14 chunks gets 2 anchors, not 3, and below 24 chunks the gap pushes later anchors back (the last one can land a few chunks after the ramp ends, still on the full target).

```python
from steering.reanchor import ReanchorPolicy
policy = ReanchorPolicy()                      # ReanchorConfig(...) to tune

# control thread
result = controller.handle(msg); policy.on_target(msg, result)
# generation thread, once per chunk
cond = steerer.get_cond(); a = policy.next_chunk(steerer.progress, steerer.is_moving)
if a: backend.reanchor(a.strength, source=a.source)  # takes effect on this chunk
latents = backend.generate_chunk(cond)
# new video
controller.reset(prompt); policy.on_reset()
```

## Re-anchoring on the real model

All numbers in this section are Ishaan's, measured on machine 2 (one L40S) at 672x384, 2 steps, TAEHV, unless a line says otherwise.
Sharpness is the variance of the Laplacian.
"Change" or "pop" is the mean absolute difference (0 to 255) between consecutive frames; normal motion is about 3 to 6, with a p99 of 17.

### Clean-anchor refresh (PR #83)

The backend saves a clean anchor: the latent of the chunk 3 chunks (`clean_anchor_after`) after a reset or a steer re-anchor.
`backend.reanchor(strength, source)` starts from the last latent for source `last` (every steer, which also drops the clean anchor so a new one is saved 3 chunks later) or from the clean anchor for source `clean` (every refresh).
A refresh from the last latent starts from a picture that has already drifted, so it bakes the damage in (over-contrast, blocky neon grass).

Over 360 chunks with no commands:

- From the last latent at anchor step 1 every 40 chunks, the picture degrades from about chunk 180.
- A full restart from noise every 60 chunks makes the scene and colours jump (unusable).
- From the clean anchor at step 1 or 2 every 24 or 40 chunks, it stays clean and the same scene for all 360 chunks, and step 2 every 24 is the steadiest.

Refresh interval on the car, with the decoder restart (mean saturation per 60 chunks, from PR #83):

| Interval | Saturation | By eye |
|---|---|---|
| every 24 (the default) | 0.23 0.22 0.20 0.21 0.21 0.21 | clean the whole run |
| every 48 | 0.25 0.24 0.21 0.23 0.20 0.21 | mostly clean, trees slightly blocky near the end |
| every 72 | 0.25 0.21 0.20 0.21 0.30 0.20 | the sky over-saturates and the trees go blocky |
| off | 0.25 0.28 0.27 0.13 0.05 0.03 | the colour is gone by the end |

The default stays at 24.
The live engine runs 48 (`--reanchor-opt refresh_every_chunks=48`) since engine PR #125, because each refresh re-composes the shot and 48 halves the number of jumps.

Why each refresh re-composed the shot: the refresh noises the clean anchor to step 2 (warped t = 833, sigma about 0.83) with fresh noise, so each refresh was a new sample near the anchor, and on a static scene the car and the snow layout changed every refresh.
The backend now draws a refresh's noise from a generator seeded by the clean anchor (`refresh_noise=fixed`, the default), so every refresh from one anchor returns to the same picture.
On the parked-car scene the change between the frames 1 s after two refreshes went from 30.4 to 10.9 (normal motion over 1 s: 12), with no drift over 360 chunks on the parked or the driving car (engine/MODEL_NOTES.md, `engine/samples/refresh_noise_fixed_parked.jpg`).

On the live engine, the 90 s after a Reset with no commands stayed clean and the same scene after PR #83, where before it was over-saturated by 25 to 30 s and broken by 40 to 60 s (`engine/samples/drift_after_83_90s.jpg`).

### Decoder restart (PR #83)

A keep-0 rebase makes a first-frame latent, but the streaming decoder kept the old video's context and decoded it as a 4-frame continuation.
That gave a washed-out, ghosted chunk and a soft next one on every refresh and every steer.
Now `ChunkLatents.start` is `ChunkStart.FIRST_FRAME` on the first chunk of a video and on every keep-0 rebase, and `_Decoder` restarts its stream state on it (TAEHV context cleared or the Wan cache zeroed), keeping `video_id` and the chunk count.
`first_frame_pad=hold` (the default) repeats the 1 clean frame to 4, so every chunk still has 4 frames; `none` emits the true 1 frame.

| Event | Sharpness: before, anchor chunk, next chunk (old decoder) | With the restart |
|---|---|---|
| meadow refresh | 2049, 498, 1448 | 2049, 1871, 2046 |
| car refresh | 541, 133, 296 | 541, 524, 505 |
| meadow steer | 2080, 110, 494 | 2080, 901, 1071 |
| car steer | 510, 71, 189 | 510, 361, 397 |

A steer used to take about 3 ghosted chunks to reach the new scene; now it is one clean cut.
Sheets: `engine/samples/reanchor_decoder_restart.jpg` and `reanchor_decoder_restart_steer.jpg`.

### Cut fade (PR #113)

A re-anchor is a hard cut.
With `--backend-opt cut_fade_frames=N`, the decoder dissolves from the last frame it showed into the new picture over the next N output frames: frame k after the cut is `(1 - a) * last shown + a * new`, with `a = (k + 1) / (N + 1)`.
It runs across chunk boundaries, never on the first chunk of a new video, and a second cut during a dissolve starts again from the frame last shown.
`cut_fade_refresh` (default 1) makes refreshes fade too; 0 fades only steers.

Car, 360 chunks, steers to snow, neon and desert, refresh every 24 chunks; the number is the largest frame-to-frame change around an anchor, averaged over events:

| | N = 0 | N = 8 |
|---|---|---|
| steer (3 events) | 82 (max 97) | 13 (max 15) |
| refresh (12 events) | 38 (max 56) | 8 (max 13) |

With N = 8 the new scene is fully in 0.5 s after the cut, with no extra generation time.
The cost is a double exposure during the fade: the lowest sharpness during a refresh fade was 226, against 369 without it.
On the live recording after the deploy (`after113_live_s002.mp4`, 2 minutes with no commands) Ishaan saw no double car, so `cut_fade_refresh` stays on (board post at 00:12 UTC).
The live engine runs `cut_fade_frames=8` since engine PR #116.
Sheets: `engine/samples/cut_fade_steer.jpg` and `cut_fade_refresh.jpg`.

### Still open

- Every steer re-composes the whole scene (a new camera angle and tree position), because strength 1.0 restarts from pure noise and continuity comes only from the prompt text (board post at 23:42 UTC).
- Every refresh also re-composes the scene: on the live recording the car moved between right of the tree, nearer, far left and small and distant at each 6 s refresh, so a refresh is a new sample, not a return to one saved frame (board post at 00:12 UTC).
  Ishaan is measuring fixed refresh noise, a lower refresh timestep, whether the anchor is re-saved, and interval 48.
- No variant kept the exact camera and layout across a world change, before the steer start options below.

### Steer start: keeping the layout (backend options, off by default)

Four backend options change how a steer chunk starts; they need no change to `steering/reanchor.py`.
A steer chunk is the one `backend.reanchor(strength, source="last")` asks for at a strength that maps to anchor step 0 (above 0.75), which is every command (`sudden_strength` 1.0, `steer_strength` 0.85); refreshes and slow steers at 0.5 are not touched.

- `steer_anchor_t=800`: start from the last clean latent noised to t = 800 instead of pure noise.
- `steer_noise=fixed`: all noise of that chunk comes from a generator seeded by the video's seed, the same for every steer of the video.
- `steer_steps=800,500`: the chunk's schedule (default: `steer_anchor_t` then the first-frame steps below it).
- `steer_noise_first_frame=1`: the first frame of each video uses the same fixed noise.

Measured on the GPU box, one seed so far, more runs in progress: with t = 800 and fixed noise the tree trunk moved 12, 12 and 21 px across steers to snow, night and desert (56 to 96 px today), the world still changed fully and sharpness was unchanged.
With fresh noise at t = 800 the world only partly arrives; t = 700 keeps more layout but the world does not arrive; t = 900 is like 800 with a little more reframing.
A 2-step schedule for the cut chunk (`steer_steps=1000,500`) took it from 410 to 247 ms with unchanged sharpness; with t = 800 that is being measured.
Details and validation rules: `engine/MODEL_NOTES.md`, "Steer start".

## Tried and rejected

| Variant | Result | Source |
|---|---|---|
| 3 or 4 denoising steps instead of 2 | Generation median 277 ms (3 steps) and 349 ms (4 steps) against 218 ms (2 steps), over the 250 ms budget; the picture looked worse (sharpness 783 / 721 / 688 for 2 / 3 / 4 steps, harsher contrast, purple sky) and the colour collapse came sooner (chunk 71 at 4 steps, 95 to 119 at 2 steps) | board post at 23:53 UTC, 120 chunks, seed 3 |
| Stock Wan VAE instead of TAEHV | 157 ms decode per chunk against 21.5 ms, picture almost identical | the same post |
| Step 1 steer (lower strength, to keep the layout) | pop 52 to 70, the camera jumps, winter arrives only partly | board post at 23:48 UTC, 12 variants, fields to snow to neon to desert, 28 chunks each |
| Step 2 steer | the world does not change | the same post |
| 4 steps on the two chunks after a steer | no gain, 180 ms more per chunk | the same post |
| Hidden 2-chunk warm-up | twice the generation, 0.5 s later | the same post |
| LongLive-style re-cache without a rebase | the world never changes, 2.3 s stall | the same post |
| Step-0 steer with the decoder restart, no fade | a clean hard cut, pop 67 to 79 | the same post |
| Old refresh: every 60 chunks at 0.25 from the last picture | breaks the picture on the refresh chunk | `engine/samples/refresh_breaks_picture.jpg` |

The best of the 12 scene-change variants was the step-0 steer with the decoder restart plus an 8-frame blend from the last shown frame (pop 10 to 11), which is what runs live.

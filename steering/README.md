# steering

Moves the text-prompt conditioning from its current value toward a new target
over a few video chunks, so a spoken command changes the scene without a cut.
Model-independent: it only sees tensors whose last dimension is the embedding
dim. Runs on CPU or GPU, any float dtype (bfloat16 included).

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

- A new `set_target` starts from wherever the conditioning is right now, so
  commands can interrupt each other.
- With `cap < 1` the move stops part way. Call `set_target` again with the same
  target to confirm, or a different one to cancel.
- `steerer.progress` (0 to 1) and `steerer.is_moving` report the current move.
- `steerer.reset(cond)` jumps to `cond` with no move (a new stream; the shape
  may change).
- `steerer.current` is the tensor `get_cond` last returned (the initial one
  before any call), read without advancing the move.

Also exported: `Aligner` (see below), `blend(a, b, w)` (per-token slerp of direction, lerp of norm,
linear fallback for zero or (anti)parallel tokens; w in [0, 1.3]),
`apply_strength(without, with_change, s)` (same as `blend`, for "a little" /
"a lot more"), and `ease(r)` (min-jerk pacing curve).

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
  `cache_size` must be >= the number of precomputed prompts.
- `encode` runs outside the controller's lock. When messages race, the one
  whose `handle` started last wins; an older one that finishes later is
  dropped with `reason: "stale"`.
- A final message equal to the committed prompt (nothing pending), or a
  provisional one equal to the committed or pending prompt, is a no-op
  (`reason: "unchanged"`).
- Bad messages raise ValueError (a final message must have cap 1.0); on any
  error nothing changes.
- Optional `strength` (number in (0, 1.3]) and `without_prompt` (non-empty
  str), both or neither: the target is
  `blend(E(without_prompt), E(prompt), strength)`; 1.0 is exactly `E(prompt)`.
  Both prompts go through the same cache; `cache_hit` is True only if both
  were hits. A provisional message may carry them too (`cap` applies on top).
- "Unchanged" and "confirm" compare the full target
  `(prompt, strength, without_prompt)`: the same prompt at a new strength is
  a new "final". `committed_prompt` / `pending_prompt` return the prompt only.

```python
cond = ctl.reset(prompt, strength=None, without_prompt=None)  # reset tap: new video
```

- Encodes outside the lock, then `steerer.reset(cond)` (jump, no ramp; idle,
  progress 1.0; shape may differ), commits that target, clears any pending
  guess, and makes every `handle` call already in flight "stale". With an
  aligner the prompt is placed in the identity layout. On any error nothing
  changes. Returns the tensor now in the steerer.

## Aligner (keep shared tokens at the same positions)

Prompts are encoded and zero-padded after the real tokens, so appending or
removing a phrase shifts later tokens and a per-position blend mixes
unrelated rows (a word with a comma, a word with padding). The video model's
text cross-attention has no positional information for text tokens and
padding rows are zero anywhere, so row order does not matter to it (read
from the model code; still to be confirmed on the real model). `Aligner`
moves each real token row so tokens shared with the previous prompt keep
their position; new tokens take free positions, preferring ones the previous
prompt did not use, so they fade in from zero.

```python
from steering import Aligner, Controller, Steerer

aligner = Aligner(length=512)
# encode: prompt -> (embedding [1, 512, D] with padding zeroed, real token ids)
emb, ids = encode(initial_prompt)
steerer = Steerer(aligner.place(emb, ids))   # REQUIRED: initial cond in the aligner's layout
ctl = Controller(steerer, encode, aligner=aligner)
```

- `aligner.place(emb, ids)` returns a new tensor and remembers the layout.
  First call (or after `reset()`) is the identity. Output = the real rows
  permuted plus exact zeros; nothing lost or duplicated.
- `prepare(emb, ids)` -> `(placed, layout)` without committing; `commit(layout)`.
- `place`/`prepare` take an optional `occupied` (bool tensor or sequence of
  `length`): positions still non-zero in the conditioning now, e.g. an older
  layout fading out after a capped or interrupted move. New tokens take free
  positions in this order: unused by the previous layout and not occupied;
  not occupied; unused by the previous layout; any. Matched tokens and the
  first call ignore it. `Controller` passes the rows of `steerer.current`
  that have any non-zero element.
- `prepare(..., against=layout)` matches against `layout` instead of the
  committed one and leaves the committed one alone. `Controller` places
  `without_prompt` against the prompt's new layout, so shared tokens blend
  with themselves and prompt-only tokens with zero rows; only the prompt's
  layout is ever committed. `against=()` is the identity placement (used by
  `Controller.reset`).
- With an aligner, `encode` must return `(emb, ids)` (TypeError otherwise).
  The cache keeps that raw pair; placement happens in `handle` just before
  `set_target`, only for applied messages, and the layout is committed only
  if `set_target` succeeds. `Controller` raises ValueError if the aligner has
  no layout yet (the initial prompt was not placed).
- Without an aligner nothing changes: `encode` returns a tensor.

## Tests

```
python3 -m venv ../.venv-steering
../.venv-steering/bin/pip install -r steering/requirements.txt
../.venv-steering/bin/python -m pytest steering/tests -q
```

Run from the repo root.

## Re-anchor policy (`steering/reanchor.py`)

Moving the embedding is not enough on the real model: it keeps copying the
frames it already made (ask for winter and snow falls on a green meadow). A
**re-anchor** makes the backend drop its memory of old frames and regenerate
the next frame from a noised copy of the last one under the current
conditioning. Strength is in (0, 1]: 1 keeps only the rough layout, small
keeps almost everything. It breaks motion a little, so it must be rare.
`ReanchorPolicy` decides when and how hard; it has no model code.

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
| idle (not moving, nothing pending) for `refresh_every_chunks` (60) | `refresh_strength` (0.25), reason "refresh" |
| `on_reset()` (new video) | drops pending; gap and refresh count from the reset |

Fire timing is **measured**, not guessed: on the real model the anchor
repaints with whatever conditioning is current on its chunk, and the scene then
stays locked to it. Firing half-way through the ramp (the old 0.5) repainted
with a half-blended prompt, so "make it winter" gave a new green meadow; firing
once the ramp has reached the full target works. So a single anchor waits for
progress 1.0, and a slow steer's last anchor is at 1.0. The strengths, gap and
refresh period are still untuned.

With the default gap of 8, a slow ramp shorter than 14 chunks gets 2 anchors,
not 3, and below 24 chunks the gap pushes later anchors back (the last one can
land a few chunks after the ramp ends, still on the full target).

```python
from steering.reanchor import ReanchorPolicy
policy = ReanchorPolicy()                      # ReanchorConfig(...) to tune

# control thread
result = controller.handle(msg); policy.on_target(msg, result)
# generation thread, once per chunk
cond = steerer.get_cond(); a = policy.next_chunk(steerer.progress, steerer.is_moving)
if a: backend.reanchor(a.strength)             # takes effect on this chunk
latents = backend.generate_chunk(cond)
# new video
controller.reset(prompt); policy.on_reset()
```

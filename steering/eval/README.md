# steering/eval

Tools to judge the steerer: one works on rendered frames, one on real
text-encoder embeddings. Importing `steering.eval` needs only torch.

## metrics.py: judge a steer from frames

Input is a video tensor `frames` of shape `[T, C, H, W]`, floats in [0, 1].

| Function | Returns |
|---|---|
| `frame_change(frames)` | `[T-1]` mean absolute difference; entry j is the change from frame j to j+1 |
| `find_cuts(frames, k=6.0, window=8, floor=0.01)` | frame indices where a hard cut lands (first frame after the cut) |
| `transition_span(scores, lo=0.1, hi=0.9)` | `(first index >= lo, first index >= hi)` of a rising per-frame score normalised between its first and last 10%, or `None` |
| `summarize(frames, steer_frame, fps)` | dict: `cuts`, `pre_median`, `pre_mad`, `peak_change`, `peak_frame`, `peak_ratio`, `reaction_ms` |

Rules:

- **Cut**: change j exceeds median + max(k * MAD, floor) of the previous
  `window` changes, and is also more than twice change j+1. The second rule
  is what keeps a cross-fade (raised change over many frames) from counting.
  The first two changes are never judged (no history).
- **transition_span** returns `None` when the rise is below `min_rise`
  (1e-3) or below 3 standard deviations of the two end segments.
- **summarize**: `steer_frame` is the first frame made with the new
  conditioning. Pre-steer changes are those before it. `reaction_ms` is the
  time from `steer_frame` to the frame where change first exceeds the
  pre-steer median + max(3 MAD, 1e-3); 0 means the first steered frame
  already moved. `peak_ratio` is `None` when the pre-steer median is 0.

```python
from steering.eval import summarize
print(summarize(frames, steer_frame=48, fps=16))
```

## embedding_stats.py: measure real T5 embeddings

```
python -m steering.eval.embedding_stats                        # google/umt5-small, CPU
python -m steering.eval.embedding_stats --model google/umt5-xxl --device cuda
python -m steering.eval.embedding_stats --prompts my.json --max-length 512
python -m steering.eval.embedding_stats --align                # also report aligned pairs
```

Needs `transformers`, `sentencepiece`, `protobuf` (only when run). Prompts are
encoded the way Wan2.1 does it (Causal-Forcing `utils/wan_wrapper.py`):
whitespace collapsed, tokenized with EOS and padded to `max_length`,
`UMT5EncoderModel` in float32, then every position past the real token count
set to 0.

`--prompts` takes JSON shaped like the built-in set:

```json
{"base": "a wide shot of a meadow ...",
 "changes": {"winter": ", in deep winter with snow on the ground", "...": "..."},
 "stack": ["winter", "night", "rain"]}
```

Pairs reported: base -> each change, stack -> stack minus its middle change,
and each change -> the next one. Per pair:

1. real token counts, and positions real in both / one only / padding in both;
2. per-token angle (degrees) over positions real in both, and per-token norms;
3. how many tokens `steering.blend.blend` sends down the spherical vs linear
   path at its default eps (`spherical_mask` copies blend's rule; a unit
   test checks it against blend's actual output);
4. for stack -> stack minus middle only: the tokens after the removed phrase
   (the shared tail). At every position where either copy of the tail sits,
   the angle between the two prompts at that position (what a per-token blend
   mixes, also counting pairs with padding) vs the angle of the same word
   aligned by the shift, and a plain verdict;
5. midpoint norm: |blend(a, b, 0.5)| and |lerp(a, b, 0.5)| divided by the mean
   of |a| and |b|, over positions real in both.

With `--align`, every pair is also reported with the "to" embedding placed
by `steering.align.Aligner` (a fresh one per pair: place "from", then place
"to"), header marked `[aligned]`: the same counts, angles and blend-path
numbers, with "real" meaning the positions the aligner put tokens at. For
stack -> stack minus middle, item 4 is printed as a before | aligned table
instead. Without `--align` the output is unchanged.

Saved runs are in `results/`. `results/umt5-small.txt` is from the small
model on CPU: indicative only, not the umt5-xxl encoder Wan actually uses.
`results/umt5-small-aligned.txt` is the same run with `--align`.

## scenarios.py: scripted steering runs

`run(scenario, generate, encode, fps, on_cond_changed=None, aligner=None)` replays timed
contract-2 messages through a real `Steerer` + `Controller` and scores the
clip with `metrics.py`. `default_scenarios(ramp)` mirrors the PLAN.md checks
(single, hard_swap, interrupt, pace_1..12, stack_remove_middle, early_right,
early_wrong), plus strength checks: `strength_low` (base -> winter at
`strength` 0.3, `without_prompt` = base), `strength_full` (the same at 1.0,
which must end on the same cond as `single`) and `strength_steps` (finals
for the same prompt at 0.3, 0.6, 1.0, 1.3, spaced ramp + 2 steps apart).
`run` passes every message to `Controller.handle` unchanged.
`python -m steering.eval.scenarios --fake` runs them all on a
built-in fake model (a smoke test only: its cut threshold is tuned, not real).
Add `--align` to run them with a `steering.align.Aligner` and
`fake_encode_ids` (word/punctuation tokens plus an end token, one fixed row
per token id, `[1, 64, 32]`).

With `aligner`, `encode` returns `(embedding, real token ids)` instead of a
tensor; `run` calls `aligner.reset()`, builds the Steerer from
`aligner.place(*encode(initial_prompt))`, and passes the aligner to the
Controller.

What the engine must provide to run them on the real model:

1. `encode(prompt: str) -> Tensor`: the conditioning tensor the loop feeds
   the model (same shape for every prompt), e.g. the umt5 embedding.
2. `generate(cond: Tensor) -> Tensor`: one generation step (one chunk) with
   that conditioning, returning the decoded video frames it produced,
   `[F, C, H, W]` floats in [0, 1], F >= 1. It keeps its own frame cache
   across calls; `run` calls it once per step, in order.
3. `on_cond_changed() -> None`: drop the text cross-attention K/V cache.
   `run` calls it right before any step whose cond is a different tensor
   from the previous step's. Never touch the frame cache here.
4. `fps` of the decoded video, so reaction times come out in milliseconds.

`Result.report()` prints a table; `Result.save_frames(path)` saves `[T, C, H, W]`.

## run_engine.py / row_order_check.py: on the GPU box

From the repo root, with `engine/` importable (`<name>` = the backend module):

```
python -m steering.eval.row_order_check --backend engine.backends.<name> --out eval_out/rows
python -m steering.eval.run_engine --backend engine.backends.<name> --align --out eval_out/scenarios
```

Defaults: 832x480, `--steps 2`, `--fps 16`, `--ramp 4`, `--seed 0`, all scenarios.
Add `--save-frames` for `.pt` frames, `--force-cache-reset-hook NAME` to call
`backend.NAME()` on every conditioning change, `--scenarios single,hard_swap` for a subset.

Send back the whole `eval_out/` folder:
- `rows/row_order_check.txt`: A2/B/C/D vs A (max/mean abs in 0..255 levels, PSNR,
  first-chunk mean) and a `VERDICT:` line. Exit code 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
- `scenarios/summary.txt`: one line per scenario (cuts, `s<step>:<kind>:<reaction ms>/x<peak
  ratio>` per message, step/gen/decode ms) or `FAILED <error>`; exit code 1 if any failed.
- `scenarios/<name>.report.txt` and `<name>.mp4` (only if imageio+ffmpeg or torchvision+av).

Verdict: PASS = scattered (B) and reversed (C) rows match A within max(3 x A-vs-A2, 1 level)
while another prompt (D) clearly differs, so `--align` is safe. FAIL = B or C differs by at
least half as much as D: do not align. INCONCLUSIVE = seeding is not deterministic (A2 > 1
level), D barely differs, or B/C sit in between; read the numbers before deciding.

## sheet.py / clip_score.py: after a GPU run comes back

Run on the returned `eval_out/scenarios/` (best with `--save-frames`; `.mp4` needs `imageio` + `imageio-ffmpeg`):

```
python -m steering.eval.sheet --dir eval_out/scenarios --cuts         # <name>.sheet.png
python -m steering.eval.clip_score --dir eval_out/scenarios --ramp 4  # <name>.clip.txt
python -m steering.eval.sheet eval_out/scenarios/single.mp4 --mark 16 --every 2
python -m steering.eval.clip_score eval_out/scenarios/single.frames.pt \
    --before "<old prompt>" --after "<new prompt>" --steer-frame 16 --fps 16
```

Sheets: red border = frame where a message was applied (from `<name>.report.txt`), blue = cut.
`clip_score` (CLIP `openai/clip-vit-base-patch32`, ~600 MB, `--model`/`--device` to change)
prints per-frame sim(before), sim(after), score = after - before, then the ms from the steer
frame to 10% and 90% of the rise and how often the smoothed score goes down after it.
`--dir` scores each applied message against the prompt in force before it; strength-only
changes (same prompt) are listed but not scored. Pass the run's `--ramp` and `--fps`.

## soak.py: long randomized checks of the steering maths

```
python -m steering.eval.soak --seconds 20 --seed 0              # [1, 16, 64] bf16
python -m steering.eval.soak --seconds 20 --seed 0 --full-size  # [1, 512, 4096] bf16
```

CPU only, torch only. Prints one line per check with the worst numbers seen,
then `RESULT PASS|FAIL`; exit code 1 on any failure.

| Check | What it does | Fails when |
|---|---|---|
| `back_and_forth` (40% of the time) | Steerer between prompts A, B (sometimes C), random ramps 1/2/3/4/12, interrupts at random chunks, some capped moves | a frame is non-finite; a token's norm exceeds the largest endpoint norm (+bf16 rounding); a frame leaves the span of the prompts' rows (drift); a step exceeds (ease increment) x (path speed bound); a finished move is not the target object |
| `aligner` (30%) | sessions of 100-400 prompt edits on `Aligner(512)`: add, insert, remove from the middle, replace, rotate, fill to 509-511 tokens, one-token prompts, runs of one id, random `occupied` masks | a token is lost, duplicated or shares a slot; a matched token moves; an unused row is not exactly zero; a new token takes a worse slot than one left empty; 513 tokens does not raise or changes the layout |
| `threads` (30%) | `get_cond` in a tight loop vs random `handle` (provisional, strength), `set_target` and `reset` | any exception; a frame of the wrong shape, non-finite, or torn (every prompt's rows are equal, so every valid frame's rows are too) |
| `memory` | 10,000 messages (mostly new prompts) through `Controller(cache_size=64, aligner=...)` | the cache exceeds its size, live tensors grow, or Python memory grows over 2 MB |

`step/bound` and `first_step/normal` are reported net of bf16 rounding;
`first_step/normal` <= 1 means the first frame after an interrupt is no
bigger than a normal first step. The same checks run briefly in
`steering/tests/test_properties.py`.

## follow_probe.py: does the picture follow a prompt switch (GPU)

Drives the real backend's frame-cache rebase (`engine/MODEL_NOTES.md`, "Rebase") on a hard prompt switch:

```
python -m steering.eval.follow_probe --out eval_out/follow                     # 672x384, taehv, all variants, both pairs
python -m steering.eval.follow_probe --out eval_out/follow --variants hardcut,keep3,keep0 --pair meadow_winter
python -m steering.eval.follow_probe --out eval_out/long --long 90             # decay: refresh vs none
```

Defaults: `--backend causal_forcing --width 672 --height 384 --vae taehv --seed 0 --switch-chunk 20 --chunks 80`.
The backend is loaded once; every variant starts from `reset(seed)`. Pairs: `meadow_winter`, `meadow_city`
(long prompts with the same camera wording). Variants: `hardcut` (text switch only), `keep6/3/1`
(rebase keeping that many frames), `keep3_n500`, `keep6_n500`, `keep1_n700` (noisy memory), `keep0`
(new first frame), `keep0_a1..a3` (first frame anchored on the old picture; skipped past the schedule).
`--long SECONDS` runs one prompt with `none`, `refresh40_keep6`, `refresh60_keep6`, `refresh40_keep3`,
`refresh40_keep6_n300` and reports mean saturation and RMS contrast per 5 s (decay shows as both climbing).

Output: `DIR/<pair>/<variant>/frames/*.jpg`, `chunk_ms.txt`, `sheet.png` (one frame per chunk, switch in red),
`clip.txt` (CLIP similarity per chunk to both prompts, when `transformers` loads; `--no-clip` to skip) and
`DIR/summary.md`: switch-chunk ms, max ms after it, CLIP(target) - CLIP(base) over the last 5 s, and chunks
until CLIP(target) first beats CLIP(base). On a backend without `rebase` (the fake) those variants read "unsupported".

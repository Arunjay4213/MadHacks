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

Saved runs are in `results/`. `results/umt5-small.txt` is from the small
model on CPU: indicative only, not the umt5-xxl encoder Wan actually uses.

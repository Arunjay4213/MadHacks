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

Also exported: `blend(a, b, w)` (per-token slerp of direction, lerp of norm,
linear fallback for zero or (anti)parallel tokens; w in [0, 1.3]),
`apply_strength(without, with_change, s)` (same as `blend`, for "a little" /
"a lot more"), and `ease(r)` (min-jerk pacing curve).

## Tests

```
python3 -m venv ../.venv-steering
../.venv-steering/bin/pip install -r steering/requirements.txt
../.venv-steering/bin/python -m pytest steering/tests -q
```

Run from the repo root.

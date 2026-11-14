# commands/: understanding commands

Owner `@prateekraouw`. Turns heard text (contract 1) into a target prompt for the steerer (contract 2) and the
scene-state fields of contract 4. Python 3, standard library only. Interfaces and rules: `PLAN.md` here.

## Use from the engine

```python
from commands.service import CommandService

svc = CommandService(base=None)        # None: DEFAULT_BASE from commands/state.py
start_prompt = svc.committed_prompt    # the engine starts the video on this
precompute = svc.likely_prompts()      # current state + each keyword; refresh after each command

r = svc.handle(contract1_msg)          # ValueError on an invalid message -> reply {"error": ...}
if r["steer"] is not None:
    controller.handle(r["steer"])      # steering.Controller, contract 2
contract4.update(svc.state_fields())   # changes, heard, keyword, unmatched; engine adds chunk, progress, stamps, ...
```

Call `handle` off the generation thread: it holds a lock, and with a slow reader it can block up to `slow_timeout_s`.
A reset tap always returns the base prompt in `steer` with `ramp_chunks` 1, even when nothing changed,
because the engine restarts the video on it.

## Pieces

| File | What |
|---|---|
| `state.py` | `SceneState`: base + changes + bounded undo history |
| `composer.py` | `compose(snapshot)`: one prompt in a fixed order with stable wording |
| `reader.py` | fast reader: keyword table, pace and strength words, name matching, partial-text spotting |
| `service.py` | `CommandService`: ties them together, early-steering guesses and rollback, optional slow reader |

The slow reader is an injected callable `fn(text, snapshot) -> op dict | None`, off by default.
No LLM provider is configured yet, so phrases the keyword table misses show as unmatched (red on the page).

## Tests

```sh
python3 -m unittest discover -s commands/tests -t .   # from the repo root
```

`tests/fixtures/phrases.json` holds the phrase cases (format in `PLAN.md`); `test_phrases.py` runs them through the reader.
If `jsonschema` is installed, every contract-2 message the service produces is also checked against `contracts/`.

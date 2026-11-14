# commands/ plan (teammate 3, owner `@prateekraouw`, issue #15, branch `15-command-pipeline`)

AGREED interfaces between Codex and Claude (2026-11-14).
Python 3, standard library only. Reuses merged `contracts/` (contract 1 in, contract 2 out, the contract-4 state fields).

## Files and owners (no overlap)

| File | Owner |
|---|---|
| `commands/state.py`, `commands/composer.py`, `commands/tests/test_state.py`, `commands/tests/test_composer.py`, `commands/tests/fixtures/phrases.json` (>= 40 phrases), `docs/` | Codex |
| `commands/__init__.py`, `commands/tests/__init__.py`, `commands/reader.py`, `commands/service.py`, `commands/tests/test_reader.py`, `commands/tests/test_service.py`, `commands/tests/test_phrases.py` (runs Codex's fixtures) | Claude |
| this file | shared; edit only by agreement |

Checks: `python3 -m unittest discover -s commands/tests -t .` from the repo root.

## 1. Operation dict (reader output, `state.apply` input)

| op | fields | effect |
|---|---|---|
| `add` | `label` str, `kind` str, `phrase` str, `strength` float (default 1.0) | Appends a change with a new id. `season`, `time`, `weather` replace an existing change of that kind; `extra` stacks. Same label as an existing change: updates its fields, keeps its id; False if identical. |
| `remove` | `id` str | Deletes that change. The reader resolves names to an id from the snapshot. Unknown id: False. |
| `undo` | none | Restores the state before the last change. False if no history. |
| `reset` | none | Constructor base, no changes; undoable. False if already there. |
| `set_strength` | `id` str, `strength` float | Absolute, finite, 0 to 1.3 inclusive. Not clamped: out of range raises ValueError. Unknown id or same value: False. |
| `new_scene` | `base` str | Replaces the base and clears the changes; undoable. |

Strength wording (composer): 0 omitted (kept in state), (0, 0.5) subtle, [0.5, 1) moderate, 1 the phrase unchanged, > 1 heavy.

## 2. State (`commands/state.py`, Codex)

```python
class SceneState:
    def __init__(self, base: str = DEFAULT_BASE): ...
    def apply(self, op: dict) -> bool   # True if changed. ValueError for malformed op or invalid values.
                                        # Valid but unknown id / unchanged: False, no history entry.
    def snapshot(self) -> dict          # deep copy: {"base", "changes": [{id, label, kind, phrase, strength}], "can_undo"}
```
Ids `c1`, `c2`, ... are never reused, even after undo or reset. History keeps the last 100 states.

## 3. Composer (`commands/composer.py`, Codex)

`compose(snapshot: dict) -> str`. Deterministic, no mutation: base, then season, weather, time, extra, then insertion order.
Each phrase is a self-contained noun phrase with identical wording every time (token alignment, board #6884167036).

## 4. Reader (`commands/reader.py`, Claude)

```python
KEYWORDS: dict[str, dict]   # spoken word -> {"label", "kind", "phrase"}; aliases share a label (snow -> winter, stream -> river)
RAMP = {"default": 4, "slow": 12, "sudden": 1, "confirm": 2}   # settings until the first real run

def read(text: str, snapshot: dict) -> dict      # {"op": dict | None, "keyword": str | None, "pace": "slow" | "sudden" | None}
def spot(partial_text: str) -> str | None        # KEYWORDS key in partial text, for early steering; None if remove/undo wording
def resolve(query: str, snapshot: dict) -> str | None  # change id: exact id, label, alias, phrase word, close match; most recent wins
```

Precedence in `read`: negation ("do not", "never") or a transform ("into", "replace") gives `op: None` (left to the slow reader, never guessed);
then undo, reset, explicit new scene ("new scene of ...", "take me to ..."), remove, keyword add/strength, then "show me X" as a new scene only if no keyword.

## 5. Slow reader (injected, optional)

No provider or key is available. `CommandService(slow_reader=fn)` takes `fn(text, snapshot) -> op dict | None`.
It runs on one worker thread with a timeout (default 0.4 s). At most one call is in flight: while an earlier call is still running
(it timed out but has not returned), new misses count as unmatched instead of queueing threads.
A timeout, an exception, or an op that `state.apply` rejects all count as unmatched.
A callback that can hang (network) must carry its own network timeout; the service cannot kill a Python thread.
PENDING: a live provider (model, account, key in `.env`) is not chosen; until then the slow reader is off and misses show as unmatched.

## 6. Service API (`commands/service.py`, Claude), what the engine calls

```python
class CommandService:
    def __init__(self, base: str | None = None, slow_reader=None, slow_timeout_s: float = 0.4): ...
    def handle(self, msg: dict) -> dict
        # msg: contract 1. Raises ValueError if invalid (engine replies {"error": ...}).
        # -> {"steer": <contract-2 message> | None,
        #     "state": {"changes": [{"id", "label", "strength"}], "heard": str|None, "keyword": str|None, "unmatched": bool}}
    def state_fields(self) -> dict          # the same "state" dict for the current state, for per-chunk contract-4 messages
    def likely_prompts(self) -> list[str]   # current state + each keyword, for the engine to precompute
    @property
    def committed_prompt(self) -> str
```
One lock around `handle`; a slow-reader miss can block up to `slow_timeout_s`, so call it off the generation thread.
The engine owns `chunk`, `progress`, stamps, the `t_client` echo and `applies_from_chunk`.

| Contract 1 | State | Steer |
|---|---|---|
| `partial`, new keyword spotted | unchanged | provisional, `cap` 0.4, prompt = compose(deepcopy(state) + that add) |
| `partial`, same keyword as guess | unchanged | None |
| `partial`, guess's keyword gone (revised) | unchanged | committed prompt, final (rollback) |
| `final`, op changes state | applied | compose(new), final; `ramp_chunks` from pace, or `confirm` if it equals the guess |
| `final`, no op / no-op / rejected | unchanged (`unmatched` true unless a valid no-op) | committed prompt if a guess is in flight, else None |
| `tap` remove/undo/reset | applied | compose(new) if changed; committed prompt if a guess is in flight; else None |

Every final and tap clears the guess.

## Open questions

- Q1 resolved: optional contract-2 pair `strength` + `without_prompt` accepted by the lead (#6884255875), not in `contracts/` yet.
  `commands/` sends wording only for now; using the pair for set_strength is a follow-up once the schema has it.
- Q2 `@Arunjay4213`: the engine calls `CommandService.handle()` as above (posted on the board).

## Fixture format (`commands/tests/fixtures/phrases.json`, Codex)

```json
[{"text": "also add a river", "before": ["winter"], "expect": {"op": "add", "label": "river"}, "pace": null},
 {"text": "remove the river", "before": ["winter", "river"], "expect": {"op": "remove", "label": "river"}}]
```
`before` lists labels added first (via KEYWORDS). `expect` is a subset match; for `remove`/`set_strength`, `label` names the change whose id is expected.
`expect: null` means no operation. `pace` is optional (absent means not checked).

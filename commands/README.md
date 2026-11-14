# commands/: understanding commands

Owner `@prateekraouw`. Turns heard text (contract 1) into a target prompt for the steerer (contract 2) and the
scene-state fields of contract 4. The dictionary path uses the Python standard
library. The optional CPU vector path uses pinned sentence-transformers. Interfaces
and rules: `PLAN.md` here.

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
| `keywords.json`, `dictionary.py` | embedded canonical vocabulary, aliases, validated loader and compiled whole-word matching |
| `semantic.py` | optional final-only CPU vector retrieval |
| `service.py` | `CommandService`: ties them together, early-steering guesses and rollback, optional slow reader |

The slow reader is an injected callable `fn(text, snapshot) -> op dict | None`, off by default.
The engine configures its existing Azure LLM reader for dictionary misses.

## Dictionary and open vocabulary

`keywords.json` contains 39 canonical scene changes and 281 spellings/aliases,
including day/daytime/morning/sunny/bright and common objects. Existing canonical
phrases are unchanged, so steering keeps stable wording. To extend it, add a
canonical `label`, `kind`, stable `phrase`, and `aliases`. Conflicting aliases fail
at startup rather than choosing an arbitrary meaning.

Matching normalizes Unicode spaces, full-width letters, dashes, punctuation and
curly apostrophes. Multiword aliases win over their substrings. Object nouns win
over generic adjective aliases: “add a bright red car” selects car, not daytime.
An explicit new object such as “add a glowing unicorn” is preserved as a named
change, even when it is absent from the dictionary; retrieval never substitutes
the nearest familiar animal. Whole-word partial matching stays deterministic.

## Local vector embeddings

Install once in the engine's Python 3.10 venv (these deployment pins do not support Python 3.13), save the small encoder in a fixed local folder before
the live run, and enable it at process startup:

```sh
pip install -r commands/requirements-semantic.txt
python -m commands.semantic --prepare
python -m commands.check_semantic
COMMAND_SEMANTIC=1 python -m engine.server --backend fake --reader commands --port 8000
```

Use the real backend flag for the real GPU engine; this model remains on CPU.
The fixed model folder is `~/.cache/madhacks/command-minilm`. Run setup as
the same OS user as the engine service (currently `ubuntu` on the GPU host),
then set `COMMAND_SEMANTIC=1` in that service environment and restart it.
If dependencies or prepared files are missing, startup logs a warning and disables
the vector layer; dictionary and the existing slow LLM still work. Runtime loading
is local-only, so a command never downloads a model.
Without that flag, the dictionary/open-object path still works without model
dependencies. A caller can also inject `CommandService(semantic_reader=reader)`.

The encoder is `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors).
Its normalized vectors are compared by cosine similarity. The reference index is
built once, recent queries are cached (256 entries), and only final dictionary
misses use it. Semantic additions need similarity ≥0.55 and a ≥0.05 lead over the
next canonical concept. Semantic removal searches only active changes and uses
stricter ≥0.60 / ≥0.10 thresholds (minimum ≥0.70 with one active change). Negation and unsupported transformations are
never turned into additions. A vector similarity score is not a probability and
cannot guarantee correct interpretation of every sentence.

The real-model self-check covers unseen paraphrases, literal new objects, negative
inputs, and a compound edit. It reports command-reader milliseconds separately
from download/startup, model generation and network/video latency. No remote LLM
or API key is used by this fallback.

The optional requirements preserve engine pins: torch 2.5.1, transformers 4.49.0,
numpy 1.24.4, plus sentence-transformers 3.4.1. A Linux x86_64 / Python 3.10 `uv pip
install --dry-run` resolves this set. The engine owner must install and check CPU
contention before enabling it on the GPU host; local CPU timings do not prove that.

Sources: [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[similarity documentation](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html).

## Multiple commands per sentence

“Add a dog and remove the lighthouse” applies both operations in order against a
temporary state, then emits one composed target. Undo restores the entire state
before that sentence. Explicit actions can be joined by and/then/semicolon, and
unambiguous noun lists can repeat an action (“add a dog and a cat”). Descriptions
such as “black and white” remain one phrase. A rejected clause rolls back the
whole sentence, including any provisional guess. A sentence is limited to 16
operations; nested batches are rejected. The external contracts remain unchanged.

## Tests

```sh
python3 -m unittest discover -s commands/tests -t .   # from the repo root
```

`tests/fixtures/phrases.json` holds the phrase cases (format in `PLAN.md`); `test_phrases.py` runs them through the reader.
If `jsonschema` is installed, every contract-2 message the service produces is also checked against `contracts/`.

## Integrated engine check

With the engine's verification dependencies installed, start the CPU fake backend
in one terminal and run the command scenarios in another:

```sh
python -m engine.server --backend fake --reader commands --port 8000
python -m commands.check_engine --url http://127.0.0.1:8000
```

The check resets the session, sends partial/final text and taps over the real
websockets, checks stacking/removal/undo/rollback, and validates the streamed state.
It does not measure real-model semantic quality or microphone latency.

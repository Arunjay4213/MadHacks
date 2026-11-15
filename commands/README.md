# commands/: understanding commands

Owner `@prateekraouw`.
Turns heard text (contract 1) into a target prompt for the steerer (contract 2) and the scene-state fields of contract 4.
The dictionary path uses only the Python standard library.
The optional CPU vector path uses pinned sentence-transformers.
Interfaces and rules: `PLAN.md` here.

## Use from the engine

```python
from commands.service import CommandService

svc = CommandService(base=None)        # None: DEFAULT_BASE from commands/state.py
start_prompt = svc.committed_prompt    # the engine starts the video on this
precompute = svc.likely_prompts()      # current state + each keyword and world; refresh after each command

r = svc.handle(contract1_msg)          # ValueError on an invalid message -> reply {"error": ...}
if r["steer"] is not None:
    controller.handle(r["steer"])      # steering.Controller, contract 2
contract4.update(svc.state_fields())   # changes, heard, keyword, unmatched; engine adds chunk, progress, stamps, ...
```

Call `handle` off the generation thread: it holds a lock, and with a slow reader it can block up to `slow_timeout_s`.
A reset tap always returns the base prompt in `steer` with `ramp_chunks` 1, even when nothing changed, because the engine restarts the video on it.
The default base is "a wide shot of a meadow with a single oak tree".

## Pieces

| File | What |
|---|---|
| `state.py` | `SceneState`: base + changes + undo history (the last 100 states) |
| `composer.py` | `compose(snapshot)`: one prompt in a fixed order with stable wording |
| `reader.py` | fast reader: keyword table, pace and strength words, name matching, partial-text spotting, compound sentences |
| `keywords.json`, `dictionary.py` | embedded canonical vocabulary, aliases, validated loader and compiled whole-word matching |
| `world.py` | place commands ("make it a desert") and the table of tested world sentences |
| `semantic.py` | optional final-only CPU vector retrieval |
| `rewriter.py` | the rules and the validator for the engine's Sonnet prompt enrichment (no network) |
| `service.py` | `CommandService`: ties them together, early-steering guesses and rollback, optional slow reader, `snapshot()` and `apply_final()` for the engine |

## Scene state

A scene is a base sentence plus a list of changes.
Each change has an `id`, a `label` (the chip), a `kind`, a `phrase` (the words that go into the prompt) and a `strength`.
The kinds are `world`, `season`, `weather`, `time` and `extra` (objects, animals, people).
`world`, `season`, `weather` and `time` hold one change each: a new one replaces the old one of the same kind, so "make it night" replaces "day" and "sunny" replaces "night".
Ramps: a normal or confirmed command uses `ramp_chunks` 1, "slowly" uses 12, and a provisional guess uses 4 with a cap of 0.4 (`RAMP` in `reader.py`, `PROVISIONAL_CAP` in `service.py`).
The ramp is 1 because the engine's re-anchor fires once the target is reached in full, so a longer ramp only adds delay.

## Dictionary and open vocabulary

`keywords.json` holds 39 canonical scene changes (26 objects, 5 times of day, 4 seasons, 4 kinds of weather) with 242 aliases, including day, daytime, morning, sunny and bright for `day`.
To extend it, add a canonical `label`, `kind`, stable `phrase`, and `aliases`.
Conflicting aliases fail at startup rather than choosing an arbitrary meaning.

Matching normalizes Unicode spaces, full-width letters, dashes, punctuation and curly apostrophes.
Multiword aliases win over their substrings.
Object nouns win over generic adjective aliases: "add a bright red car" selects car, not daytime.
For an object, the dictionary only recognises the object and picks the chip label; the user's own words go into the prompt, so "add a red car racing on a road" keeps red, racing and the road, and a bare "add a car" gets the dictionary phrase (PR #88).
An explicit new object such as "add a glowing unicorn" is kept as a named change, even when it is not in the dictionary; retrieval never substitutes the nearest familiar animal.
Partial text is matched only by whole words, so a guess is always deterministic.

## World changes

A place is one change of kind `world` (PR #88).
"make it a desert", "take me to the desert", "turn it into a beach", "turn this into", "change it to a" and "make this a" followed by a known place all replace the world and keep the subject and the objects (PR #115).
A new world also clears the season and weather changes in the same undo step, because the world sentence brings its own; the time of day and the objects stay.
So "make it winter", "add a red car", "make it a desert" gives a desert with the car and no snow.
Repeating the current world still clears a later season.
"turn the tree into a lighthouse" names one object, not the whole scene, so it is not a world change: it is left to the engine's final reader.

The known places are in `world.py`: country road, desert, snowy mountain road, neon city, ocean coast (also "beach" and "sea"), Mars, forest and meadow.
The first six are Ishaan's tested sentences, word for word (country road is the world of his tested car prompt); forest was added at the lead's request, and meadow lets "take me back to the meadow" return to the oak scene.
A later season, weather or time change removes the matching built-in part of the world sentence (for example the desert's afternoon sun after "make it night").
A sunlit time ("sunny", "make it day", "noon") also removes built-in weather that is not sunlit, such as overcast light, rain or falling snow, unless a weather change names the weather itself.
A world can word such a change in its own light sentence: on the snowy mountain road "sunny" becomes "Bright winter sunshine in a clear blue sky, sparkling snow, crisp shadows." in place of the overcast winter light, and the snowy road and forest stay.
That sentence belongs to winter, so after an explicit change to another season ("make it summer") a later "make it day" uses the plain daylight phrase and never brings winter back.
The state only gains a `day` change, so one undo brings the overcast light back.
An arbitrary custom base that mixes an unknown world with subjects is rejected rather than edited by unsafe text surgery.

## The car scene (road scenes)

`world.CAR_BASE` is the car scene: Ishaan's measured car template (board #6886548029) on the country road.
A base that is that template for any known world is a road scene (`world.is_road_scene`).
A spoken car base that names a known world as its road ("a red sports car driving along a country road", as the dictionary path keeps a new scene) is a road scene too, with that car as the subject, so "make it a desert" swaps the world when the LLM reader times out.
On a road scene the composer always writes the template: the subject sentence with the road, the camera sentence, one sentence per object, the scene and light of the world, one sentence per season, weather or time phrase, then the style sentence "Cinematic, highly detailed, realistic, sharp focus, motion blur on the road."
The style sentence is used only on road scenes, because on the meadow it grew a road.
Ishaan measured that appending a season to the car prompt does not change the picture, so on a road scene a winter season ("make it winter", "make it snowy") is shown as the snowy mountain road world.
The state still holds a season change labelled winter, so one undo or removing the chip goes back to the earlier world, and "make it a desert" after it clears it as usual.
On the meadow, winter is still the season phrase.

## Multiple commands per sentence

"Add a dog and remove the lighthouse" applies both operations in order against a temporary state, then emits one composed target (PR #70).
Explicit actions can be joined by and, then or a semicolon, and unambiguous noun lists can repeat an action ("add a dog and a cat").
Descriptions such as "black and white" stay one phrase.
"add snow near the river" adds only snow, and "add a dog and stop the rain" adds the dog and removes the rain.
Words after a preposition are a location for every verb, so "make it snow near the river" adds snow even when a river is present, and "turn it to night near the river" sets night.
After an add verb, "without" starts a removal: "add a dog without the rain" adds the dog and removes the rain.
A rejected clause rolls back the whole sentence, including any provisional guess.
A sentence is limited to 16 operations, and nested batches are rejected.

## Undo, remove and reset

- "undo", "go back", "revert", "take that back" and "never mind" restore the whole state before the last command, so one compound sentence or one world change is one undo step.
- "undo the winter" removes the winter, not the last change.
- "remove the river", or the × on a chip (a `remove` tap), removes one change and keeps the rest.
- "remove the forest" removes the world, and objects added inside it stay (a tree added earlier is kept).
- "remove the car near the forest" removes the car and keeps the forest.
  A world named after a preposition ("near", "in", "by") is only the location of the object named before it, never the removal target.
  This holds for every world, for example "remove the car near the meadow" keeps the meadow.
  If the object is not in the scene, nothing is removed: "remove the dragon near the meadow" with no dragon keeps the meadow.
  A world whose own name holds a preposition is still named in full: "remove the city on mars" removes the "city on mars" world.
- "reset", "start over" and "clear everything" go back to the base scene, and the engine starts a new video.
- "a little more snow" and "a little less snow" move the strength up or down from where it is.
- The vector matcher and the style fallback ("make it spooky") read strength words with the same `reader.strength_for` as the dictionary.
  "make it a little nocturnal" adds night at 0.3, and "make it much more nocturnal" with night present raises it.
  The style fallback never keeps a strength word in the chip: "make it a little spooky" adds spooky at 0.3, and "make it less spooky" with spooky present lowers it.
- Pace words ("slowly", "gradually", "gently", "little by little", "suddenly") set only the ramp, never the strength: "slowly make it winter" gives winter at 1.0 over 12 chunks, on every path, including the style fallback ("slowly make it spooky").

## Local vector embeddings

Install once in the engine's Python 3.10 venv (these pins do not support Python 3.13), save the small encoder in a fixed local folder before the live run, and enable it at process startup:

```sh
pip install -r commands/requirements-semantic.txt
python -m commands.semantic --prepare
python -m commands.check_semantic
COMMAND_SEMANTIC=1 python -m engine.server --backend fake --reader commands --port 8000
```

The model stays on the CPU, also next to the real GPU engine.
The fixed model folder is `~/.cache/madhacks/command-minilm`.
Run setup as the same OS user as the engine service (`ubuntu` on the GPU host).
The live engine unit sets `COMMAND_SEMANTIC=1` (PR #78), and its comment gives about 10 ms per query.
If dependencies or prepared files are missing, startup logs a warning and disables the vector layer; the dictionary still works.
Runtime loading is local only, so a command never downloads a model.
A caller can also inject `CommandService(semantic_reader=reader)`.

The encoder is `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors), compared by cosine similarity.
The reference index is built once, recent queries are cached (256 entries), and only final dictionary misses use it.
A semantic addition needs similarity of at least 0.55 and a lead of at least 0.05 over the next canonical concept.
A semantic removal searches only active changes, with stricter thresholds of 0.60 and 0.10 (at least 0.70 with one active change).
Negation and unsupported transformations are never turned into additions.
A similarity score is not a probability and cannot guarantee a correct reading of every sentence.

The optional requirements keep the engine pins: torch 2.5.1, transformers 4.49.0, numpy 1.24.4, plus sentence-transformers 3.4.1.

Sources: [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [similarity documentation](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html).

## The rewriter and how the engine uses it

`rewriter.py` holds the rules for prompt enrichment, with no network call and no cache.
`SYSTEM_PROMPT` tells a language model to write only a `details` sentence of world, light, texture and atmosphere for the scene; `validate_rewrite(prompt, reply)` keeps the composed prompt word for word as the prefix and checks the details (at most 60 added words and 100 in total; no camera, motion, events, history words, negation, or objects and weather the scene does not have).
`build_messages(prompt, command=None)` builds a details-only request.

The engine's Sonnet final reader (`engine/final_reader.py`, live since engine PR #125) uses `SYSTEM_PROMPT` and `validate_rewrite`, but not `build_messages`, because it needs one call that returns the operations and the details together within its 3 s wait.
For every final sentence the engine:

1. takes the full scene with `CommandService.snapshot()`,
2. asks Sonnet for the operations and the details in one call,
3. applies the operations with `CommandService.apply_final(text, operations, None)`, which validates them and applies them as one edit with the same `steer` and `state` shape as `handle()`,
4. checks the details with `validate_rewrite` and uses the rich prompt for that scene state from then on.

Partials, taps, undo and reset still go through `handle()`, and so does any final where Sonnet times out or fails.
Empty operations reject a final without changing state and roll back any provisional guess.
Open items from engine PR #120 (board post at 00:10 UTC): `validate_rewrite` returns None with no reason and refused details over single words such as "still", "haze" and "mist", and `MAX_TOTAL_WORDS` counts the composed prefix, so a long scene leaves few words for details; the engine works around both by cutting the refused clauses.
See `engine/README.md`, "Final reader (Sonnet)", for the engine side.

## Motion and continuity

Explicit object descriptions and motion stay in scene state across weather and light edits.
For example, adding a red car driving forward keeps its colour and direction when winter is added.
A motion edit updates the existing subject; bare repeated additions and partial guesses keep its details.
Language embeddings do not store visual history; clean anchors are the backend's job.
See [integration and frame-evaluation notes](../docs/prompt-continuity.md).

## Tests

```sh
python3 -m unittest discover -s commands/tests -t .   # from the repo root
```

`tests/fixtures/phrases.json` holds the phrase cases (format in `PLAN.md`); `test_phrases.py` runs them through the reader.
`test_demo_sentences.py` checks the composed prompt for every line of `engine/tools/demo_scripts/draft.json` and the world-change chains.
If `jsonschema` is installed, every contract-2 message the service produces is also checked against `contracts/`.

## Integrated engine check

With the engine's verification dependencies installed, start the CPU fake backend in one terminal and run the command scenarios in another:

```sh
python -m engine.server --backend fake --reader commands --port 8000
python -m commands.check_engine --url http://127.0.0.1:8000
```

The check resets the session, sends partial and final text and taps over the real websockets, checks stacking, removal, undo and rollback, and validates the streamed state.
It does not measure real-model picture quality or microphone delay.

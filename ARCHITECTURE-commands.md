# From a sentence to a prompt: how the command path works

This document describes the path a spoken or typed sentence takes until the engine has a prompt to steer to.
It was written on 2026-11-15 from `main` at cf031f8, after about five patches by different agents, so that one reader can see the whole design.
It is for agents, so it names files and functions instead of explaining Python.

Scope: `commands/` (reader, world, composer, state, service, semantic, rewriter, dictionary and `keywords.json`), `engine/reader.py`, `engine/final_reader.py`, `engine/car_template.py`, `engine/slow_reader.py`, `engine/people.py` and the loading of `engine/demo_prompts.json`.
Out of scope: `contracts/`, the engine streaming loop, `steering/` and `web/`.

## Live configuration

The live unit (`systemctl cat madhacks-engine`, read on 2026-11-15) runs `--reader commands --final-reader sonnet` with `COMMAND_SEMANTIC=1`.
So the live reader object is `StateCachingReader(FinalReadingService(CommandService(), FinalReader))`, built in `engine/reader.py:load_reader`.
`CommandService()` is built with no slow reader in this mode, so the nano slow reader is never called live.
`commands.rewriter` always imports, so `final_reader.prompt_spec()` always returns the rewriter spec and the engine stand-in spec is never used live.
The other branch of `load_reader`, `CommandService(slow_reader=from_env(), ...)`, runs only with `--final-reader off` (the default for local runs).

## The objects, outside in

| Object | File | Job |
|---|---|---|
| `StateCachingReader` | `engine/slow_reader.py` | Answers `state_fields()` from the last `handle()` result, so the streamer never waits on the service lock. |
| `FinalReadingService` | `engine/reader.py` | Sends each final sentence to the Sonnet final reader before the dictionary, and maps each composed prompt to the first rich prompt chosen for it. |
| `FinalReader` | `engine/final_reader.py` | Content block, pinned and cached lookup, the measured car template, the Sonnet call, reply checks, retry, disk cache. |
| `car_template` | `engine/car_template.py` | Routes a sentence on a vehicle scene (measured world, model, or not a vehicle) and writes the car template prompt. |
| `people` | `engine/people.py` | Replaces a named person with a visual description and builds the engine's own add for them. |
| `CommandService` | `commands/service.py` | Scene state, partial guesses, the dictionary reading of a final, `apply_final` for the engine's operations. |
| `reader` | `commands/reader.py` | One clause to one operation: keyword table, pace and strength words, name matching, clause splitting. |
| `world` | `commands/world.py` | Place commands, the table of tested world sentences, and the commands-side car template. |
| `composer` | `commands/composer.py` | Scene snapshot to one prompt, in a fixed order with stable wording. |
| `state` | `commands/state.py` | `SceneState`: base, changes, the operations, undo history. |
| `semantic` | `commands/semantic.py` | Optional CPU vector match for a final clause the dictionary misses. |
| `rewriter` | `commands/rewriter.py` | The system prompt rules and the validator for the Sonnet `details` field. |

## The path of a partial

1. The engine's reader worker thread calls `reader.handle(msg)` for a contract 1 `partial` (`engine/engine.py`, reader worker).
2. `StateCachingReader.handle` calls `FinalReadingService.handle`, which sees it is not a final and calls `CommandService.handle`.
3. `CommandService.handle` validates the message (`_validate`), takes the lock and calls `_partial(text)`.
4. `_partial` calls `reader.spot(text)`, which returns a keyword alias only when the text has no block word (`is_blocked`), no place command (`world_op`) and no undo, reset, remove, negation, transform or new scene words, and then picks an alias with `_pick`.
5. If the alias names the same label as the guess already in flight, there is no steer.
6. Otherwise `_hypothesis(entry)` adds the keyword's dictionary entry to a copy of the state and composes it with `composer.compose`.
7. If that prompt equals the committed prompt, a guess in flight is rolled back with a steer to the committed prompt (ramp 1), or there is no steer.
8. Otherwise the guess is stored in `_guess` and `_guess_prompt` and the steer is provisional: ramp 4, cap 0.4 (`RAMP["provisional"]`, `PROVISIONAL_CAP`).
9. `FinalReadingService._mapped` swaps the steer's prompt for the rich prompt if one is known for it (`_model_prompts`).
10. `StateCachingReader` keeps the returned state for `state_fields()`.

Partials never reach Sonnet, the content block or the car template.

## The path of a final, live

1. Steps 1 and 2 as for a partial; `FinalReadingService.handle` sees a final with a valid `t_client`.
2. It takes `CommandService.snapshot()` and classifies the sentence with `sentence_kind(text, snapshot)`: it splits it with `reader.split_commands`, reads every clause with `reader.read`, and names each clause with `clause_kind` (`BARE_CONTROL`, `SCENE_RESET`, `EDIT`).
3. `SentenceKind.CONTROL` (every clause is a bare undo or reset) goes to `CommandService.handle`, the dictionary path below, with no call and no content check.
4. Any other sentence goes to `FinalReader.__call__(text, snapshot)`:
   1. `blocked_word(text)` checks `BLOCKED_WORDS`; a hit returns `{"blocked": True}` with no call.
   2. The key is `(compose(snapshot), text)`; the pinned entries (`engine/demo_prompts.json`) are checked first, then the in-memory and disk cache.
   3. `_from_cache` uses a SONNET or PINNED entry only on the same scene (`scene_ids`), maps its change ids to the live ones, and checks every operation again with `slow_reader.validate` and `CACHED_MAX_PHRASE`.
   4. `_measured` asks `car_template.plan(text, snapshot)`; `Route.MEASURED` (a vehicle scene with no objects and one measured world word, or a new vehicle scene) is answered by the engine with `measured_operations` and `world_prompt`, with no call (`Outcome.TEMPLATE`).
   5. Otherwise `_call` runs on the thread pool and the caller waits `timeout_s` (3.0 s live).
   6. `_attempt` sends one user message: the scene, the composed prompt, the sentence with named people replaced (`people.for_model`), `details_max_words` (`details_budget`), and `car_template` fields when the route is not PLAIN.
   7. `_check` reads the reply: `blocked` gives BLOCKED; `people.with_person` puts in the engine's add of a named person; `expand_operations` validates and rewrites the operations; `scene_after` applies them to a rebuilt copy of the scene.
   8. When the new scene has a vehicle subject, `_check_vehicle` keeps the base as the bare vehicle (`bare_vehicle_operations`), checks `road`, `scene` and `light` (`field_problems`), and builds the template (`world_prompt`, `check_prompt`).
   9. Otherwise `_check` composes the new scene, checks the phrases of new objects (`phrase_problems`), and cuts bad details clauses with `repair_details`, which uses `rewriter.validate_rewrite` on `scene_prefix(scene)`.
   10. The result is a `Verdict` (ACCEPTED, REPAIRED, PHRASE, PLAIN, EMPTY, BLOCKED, INVALID, FAILED); PHRASE, PLAIN and INVALID are asked again once if the first call took under half the wait.
   11. On a timeout the entry is FALLBACK, kept in memory only, and a late answer is counted as `late_dropped`; SONNET and BLOCKED entries are saved to the disk cache.
   12. `_or_person` turns a FALLBACK on "add <named person>" into the engine's own add (`Outcome.PERSON`).
   13. `_given` remembers the template prompt of a vehicle scene (`_given_vehicle`), so the next call on that scene is told its current road, scene and light.
5. Back in `FinalReadingService.handle`:
   1. `{"blocked": True}` calls `CommandService.apply_final(text, [], None)`: no change, unmatched, and a guess in flight is rolled back.
   2. `SentenceKind.MIXED` (bare undo or reset plus edits, not blocked) goes to `CommandService.handle`, so the dictionary runs the undo or reset in order.
   3. A result calls `CommandService.apply_final(text, operations, None)`, which validates and applies the operations as one batch and steers to the composed prompt; `_model_prompts.setdefault(composed, rich)` then makes the rich prompt the one the model sees for that state, now and later.
   4. None (FALLBACK) goes to `CommandService.handle`, the dictionary path.

## The path of a final on the dictionary

This runs when the final reader gives None, for CONTROL and MIXED sentences, and for every final with `--final-reader off`.

1. `CommandService._final(text)`: text with non-Latin letters goes to `_final_original`, which asks the slow reader with the original text, or is unmatched when there is none (live).
2. `reader.split_commands(text)` gives 1 to 16 clauses (`MAX_CLAUSES`), or the sentence is unmatched.
3. Each clause is read on a trial copy of the state with `reader.read(clause, snapshot)`.
4. A clause with no operation tries, in order: the vector matcher `_ask_semantic` (at most 0.05 s, skipped for `is_blocked` text), the slow reader `_ask_slow` (None live), and `reader.style_fallback` ("make it spooky").
5. A clause with no operation, or one that does not apply or compose on the trial state, rejects the whole sentence.
6. The operations are applied as one operation or one batch (one undo step) with `_apply`, and `_commit` steers to the new composed prompt, with ramp 12 when a clause said "slowly".

`reader.read(clause, snapshot)` decides in this order: place command (`world_op`), negation or transform (none), move or position (none), undo ("undo the X" is a remove), reset, new scene, the keyword pick, remove (`resolve`), an unknown object after an add verb (`_pass_through`), "show me" as a new scene, a motion edit of an existing object, an add with the user's own description, a strength change of an existing change, and finally a plain dictionary add.

`composer.compose(snapshot)` writes the prompt: a road scene (`world.is_road_scene`) is the commands-side car template (`world.car_prompt`), with an active winter shown as the snowy mountain road (`road_scene_world`); a scene with a world change is the base with its world replaced (`compose_base`) and the built-in sky segments that clash with a time, weather or season change removed (`world_with_overrides`); any other scene is the base, then the season, weather and time phrases, then the objects.

## Where each rule is decided

| Rule | Where | Notes |
|---|---|---|
| Strength from words ("a little", "a lot", "more", "less") | `reader.strength_for` | Used by `read`, `semantic` and `style_fallback`. The same numbers (0.3, 1.3) are also written in the Sonnet system prompt (`final_reader._OPERATIONS`) and the nano system prompt (`slow_reader.SYSTEM_PROMPT`). |
| Strength range | `state._strength` (0 to 1.3), `slow_reader._strength` (0 to 1.3, rounded to 2 places) | Two checks of one range; only the second rounds. |
| Strength wording in the prompt | `composer.strength_phrase` | One owner; `car_template.scene_objects` calls it. |
| Pace (slowly, suddenly) to ramp | `reader._pace`, `reader.RAMP`, `service._final` and `service.apply_final` | `apply_final` runs the whole `read()` only to get the pace. `RAMP["confirm"]` and `RAMP["sudden"]` equal `RAMP["default"]`, so the swaps to them do nothing. |
| Provisional guess | `service._partial`, `service._commit`, `PROVISIONAL_CAP` | |
| One change per kind; world replaces world | `state.SceneState.apply` | One owner. |
| A world clears season and weather | `state.WORLD_CLEARS` in `SceneState.apply` | Owner. `final_reader.world_first` reorders a sentence's adds so a season said in the same sentence survives, and drops seasons a base edit copied. `car_template.measured_operations` also removes time changes for a measured world. |
| Built-in sky of a world vs. an explicit change | `world.world_with_overrides` | Owner; only for known worlds. |
| Winter on a road scene is the snowy mountain road | `world.road_scene_world` (dictionary path) and `car_template.WORLD_WORDS` (engine path) | Two implementations. |
| Night on a car scene | `car_template.WORLD_WORDS` maps night to the neon city; the dictionary path adds a night phrase to the template | The same sentence gives different prompts on the two paths. |
| Is this a vehicle or road scene | `world.is_road_scene` (exact template text, or `_CAR_ON_ROAD_RX`: "car" plus a verb plus a known place), `car_template.vehicle_subject` (`_SUBJECT`: any of 9 vehicle words), `final_reader.is_vehicle` (a vehicle word anywhere) | Three rules. A bare "a red sports car" base is a vehicle scene to the engine and not a road scene to `composer`. |
| Car template text | `world.car_prompt` (dictionary path and every cache key) and `car_template.world_prompt` (engine prompt) | Two templates. They differ in the camera noun, object punctuation, and season or weather sentences. |
| Car template constants | `world.CAMERA`, `STYLE`, `_REAR`, `_FORWARD` and copies in `car_template`; `final_reader.VEHICLE_STYLE` is `world.STYLE` again | Copied constants. |
| Known places | `world._TABLE` (`WORLDS`) | One owner; `car_template._measured` and `final_reader.canonical_world` read it. |
| Season, weather and time labels | `keywords.json` via `dictionary.KEYWORDS` | One owner; `final_reader._vocabulary` and `canonical_change` read it. `car_template.WORLD_WORDS` is a separate word table for vehicle scenes. |
| Sonnet merge into scene state | `final_reader.expand_operations` (`canonical_world`, `canonical_change`, `distinct_label`, `with_lane_cue`, `world_first`, edit_base expansion), then `service.apply_final` | |
| Rich prompt per scene state | `FinalReadingService._model_prompts.setdefault` | The first rich prompt for a composed prompt wins for the life of the process. |
| Pinned lookup | `final_reader.load_pinned` (replays `demo_prompts.json` from its base), `FinalReader.__call__`, `_from_cache` | Pinned is checked before the cache, and needs the same scene (`scene_ids`). |
| Cached lookup | `CacheFile`, `FinalReader.__call__`, `_from_cache` | Keyed by `(compose(snapshot), sentence)`; the file is tied to `fingerprint(system_prompt, model, PROMPT_BUILD)`. |
| Unsafe content block | `final_reader.BLOCKED_WORDS` and the model's `blocked` flag | Only for finals that reach the final reader. CONTROL sentences skip it (they add nothing), partials skip it, and with `--final-reader off` there is no block at all. |
| Deliberately unmatched text (negation, transforms) | `reader.is_blocked`, `reader._NEGATION_RX`, `_TRANSFORM_RX`, `world._BLOCKED_RX`, the inline regex in `semantic.SemanticReader.__call__` | Three word lists with small differences. The name `is_blocked` is easy to confuse with the content block. |
| Fallback order on the dictionary path | `service._final`: read, vector, slow reader, style | |
| Fallback order on the engine path | `FinalReadingService.handle` and `FinalReader.__call__`: block, pinned, cache, measured template, Sonnet, dictionary | |
| Phrase length limits | `slow_reader.MAX_PHRASE` 120, `final_reader.OBJECT_MAX_PHRASE` 240, `FINAL_MAX_PHRASE` 400, `CACHED_MAX_PHRASE` | |
| Batch size | `state.apply` (literal 16), `reader.MAX_CLAUSES` 16, `final_reader.BATCH_LIMIT` 16, `slow_reader.BATCH_LIMIT` 16 | One number in four places. |
| Base edit ("turn the tree into a lighthouse") keeps the changes | `final_reader.expand_operations` (edit_base) and `slow_reader.edit_base` (added in #210) | Two expansions with different orders: the slow reader puts world changes first, the final reader keeps the scene order and then runs `world_first`. |
| Details word limits | `rewriter.MAX_DETAIL_WORDS`, `MAX_TOTAL_WORDS` | `final_reader` reads them with `getattr(rewriter, ..., default)`; the default 100 is stale (the value is 180) and never used. |

## Duplicated rules and data

- The car template exists twice: `world.car_prompt` and `car_template.world_prompt`, with copied `CAMERA`, `STYLE`, `_REAR` and `_FORWARD`.
- `VEHICLE_WORDS` is defined in both `car_template.py` and `final_reader.py`; `final_reader.VEHICLE_STYLE` is the same text as `world.STYLE`.
- Vehicle and road scene detection is done three ways (see the table).
- Winter on a car scene is decided twice (`road_scene_world`, `car_template.WORLD_WORDS`), and night on a car scene gives different prompts on the two paths.
- The negation and transform word list exists three times (`reader`, `world`, `semantic`).
- The base edit expansion exists twice (`final_reader.expand_operations`, `slow_reader.edit_base`).
- The batch limit 16 is written four times; the label limit 24 is written as a bare number in `final_reader.distinct_label`.
- `final_reader._scene_for_model` and `slow_reader._scene_for_model` are the same function, and `final_reader.scene_record` is the same again.
- `_check` and `_check_vehicle` in `final_reader.py` repeat the same four lines that find new objects and their phrase problems.
- `normalize` has three thin wrappers: `reader._norm`, `car_template._normalize`, `people._normalize`.

## Dead or unused branches (live flags)

- The nano slow reader: `slow_reader.FoundrySlowReader`, `SYSTEM_PROMPT`, `from_env`, and in `commands/service.py` the `slow_reader` and `slow_timeout_s` arguments, `_ask_slow` and most of `_final_original`; live runs `--final-reader sonnet`, which builds `CommandService()` with no slow reader.
- The engine stand-in prompt spec: `final_reader.STANDIN`, `STANDIN_DETAILS`, `standin_validate`, `_standin_spec_validate`, `_scene_objects`, `STYLE`, `VEHICLE_STYLE`, the `try`/`except` in `prompt_spec`, and the `spec.check is None` and `spec.prefix is None` branches in `repair_details`, `details_limits` and `details_budget`.
- `rewriter.build_messages` and `_EXAMPLES`: the engine documents that it does not use them.
- The `prompt` argument of `CommandService.apply_final`: the engine always passes None; with it go `MAX_FINAL_PROMPT_CHARS` and about 12 lines.
- The ramp swaps to `RAMP["confirm"]` in `_commit` and `apply_final`: both values are 1.
- `world.world_phrase` is used only by tests.
- The `getattr(rewriter, "MAX_TOTAL_WORDS", 100)` and `getattr(rewriter, "MAX_DETAIL_WORDS", 60)` defaults.

## Implicit state

- `CommandService._guess` and `_guess_prompt` are two nullable fields that are always set and cleared together; "a guess is in flight" is `_guess_prompt is not None`.
- `CommandService._apply` returns True (changed), False (unchanged) or None (invalid), and `_commit` and `_final` decode that by hand (`changed is None` means unmatched).
- `CommandService._heard`, `_keyword` and `_unmatched` are three fields set in different combinations by `_partial`, `_final`, taps and `apply_final`.
- `CommandService._slow_thread` and `_semantic_thread` mean "a call is in flight" by being a live thread or None.
- `FinalReader._call` returns its first usable reply through a mutable list argument (`first_usable`) so the caller can use it on a timeout.
- `FinalReader.__call__` keeps `late` as a bool next to the FALLBACK outcome.
- `expand_operations` keeps `rebuilt` as a bool for "an edit_base or new_scene came first".
- `Entry.scene_before` is None for entries with nothing to map.
- Good examples already in the code: `FinalReaderState`, `Outcome`, `Verdict`, `car_template.Route` and `Words`, `SentenceKind`, `ClauseKind`.

## Found while reading (behaviour, not fixed here)

- On a Sonnet car scene, a sentence that falls back to the dictionary (a timeout or a failed call, for example "add a dog") steers to the commands-side composed prompt of the bare vehicle base (`a red sports car; a dog standing nearby; a coastal highway on a cliff top. ...`), not the car template, because `composer` does not treat a bare "a red sports car" base as a road scene. The behaviour lock records this.
- A partial can start a provisional move for a sentence that the content block later refuses (the final rolls it back); the block never runs on partials.
- The engine path and the dictionary path give different prompts for "make it night" on a car scene (neon city against a night phrase in the template).

## Ranked simplifications

Each is checked by the behaviour lock `engine/tests/test_prompt_lock.py` (sentence chains over the meadow and the car scene, on the dictionary path and on the live Sonnet path with a fake model) and the full suite, before and after.

| Rank | Change | Risk | Lines saved (about) | Status |
|---|---|---|---|---|
| 1 | One copy of the car template constants and other copied numbers: `car_template` imports `CAMERA`, `STYLE`, `REAR`, `FORWARD` from `commands.world`; `final_reader` imports `VEHICLE_WORDS` and uses `world.STYLE`; one `MAX_BATCH`; the label limit by name; no stale `getattr` defaults; one `scene_record`; one helper for new-object phrase problems. | None: same text and numbers. | 25 | Done, #223 |
| 2 | Delete the dead stand-in prompt spec in `final_reader.py` and its tests; `prompt_spec()` returns the rewriter spec with no fallback. | Low: never runs live; the lock checks the system prompt (cache fingerprint). | 70 code, 40 tests | Done, #233 |
| 3 | Delete `rewriter.build_messages` and `_EXAMPLES` and their tests. | None: no caller. | 30 code, 30 tests | Done, #234 |
| 4 | `CommandService`: delete the dead `prompt` argument of `apply_final` and the no-op confirm ramp swaps; read the pace with one function instead of a full `read()`. | Low: the engine never passes a prompt. | 20 | Done, #235 |
| 5 | `CommandService`: name the implicit states (one `_guess_label` field instead of two fields that moved together; an `Applied` enum instead of True, False or None). | Low. | 0 (clarity) | Done, #235 |
| 6 | Delete the nano slow reader: keep `validate` (the final reader's operation check) and `StateCachingReader`, delete the rest and the service's slow reader path. | Medium: no live change, but `--final-reader off` with Foundry keys would no longer ask nano. | 200 code and tests | Skipped: needs the lead (see below) |
| 7 | One car template and one vehicle scene rule for both paths. | High: changes composed prompts, which are the keys of `demo_prompts.json` and the disk cache. | 60 | Skipped: changes prompts |
| 8 | One negation and transform word list. | Medium: the three lists differ at the edges ("never mind", "replaced"). | 5 | Skipped: changes edge readings |
| 9 | One answer for winter and night on car scenes on both paths. | High: changes prompts. | 20 | Skipped: changes prompts |

Ranks 7 to 9 change behaviour, so they are left for a decision by the owners.

Rank 6 is left for the lead.
The live engine never calls the nano slow reader, but restarting with `--final-reader off` (Foundry keys still set) is the quickest fallback if the Sonnet deployment fails during the demo, because nano answers in about 0.5 s.
Deleting it removes that fallback, and #210 extended it on 2026-11-15, so it is still treated as in use.
If the lead decides it can go: keep `slow_reader.validate` (the final reader's operation check) and `StateCachingReader`, delete `FoundrySlowReader`, `SYSTEM_PROMPT`, `from_env`, `edit_base` and `engine/foundry.py`'s `FoundryChat`, and in `commands/service.py` the `slow_reader` path (`_ask_slow`, the slow call in `_final_original`, `slow_timeout_s`).

Smaller items seen but not done: `reader._norm` and `people._normalize` are thin wrappers of `dictionary.normalize`; `world.world_phrase` is used only by tests; `final_reader.pace_keeps_strength` imports the private `reader._pace`.

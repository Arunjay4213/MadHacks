# From a sentence to a prompt: how the command path works

This document describes the path a spoken or typed sentence takes until the engine has a prompt to steer to.
It was written on 2026-11-15 from `main` at cf031f8, after about five patches by different agents, so that one reader can see the whole design.
Issue #271 (the same day) changed the path of a final: Sonnet now writes the whole prompt of a free scene instead of operations merged into the composed scene; the sections below describe that.
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
| `FinalReadingService` | `engine/reader.py` | Decides who reads each final (dictionary, pinned, measured, Sonnet), reads the x on a free chip as "remove the <label>", and maps each composed prompt to the first rich prompt chosen for it. |
| `FinalReader` | `engine/final_reader.py` | `lookup` (content block, pinned, cached, measured car world), `ask` (the Sonnet call, the light check, retry, disk cache) and `result`. |
| `car_template` | `engine/car_template.py` | Routes a sentence on a vehicle scene (measured world or not), writes the car template prompt, and checks that a free prompt is still the template (`template_problems`, `asks_view_change`). |
| `people` | `engine/people.py` | Replaces a named person with a visual description for Sonnet, and builds the engine's own add or remove when Sonnet gives nothing usable. |
| `CommandService` | `commands/service.py` | Scene state, partial guesses, the dictionary reading of a final, `apply_final` for the engine's operations, and `dictionary_reading`, the fast-path rule. |
| `reader` | `commands/reader.py` | One clause to one operation: keyword table, pace and strength words, name matching, clause splitting. |
| `world` | `commands/world.py` | Place commands, the table of tested world sentences, and the commands-side car template. |
| `composer` | `commands/composer.py` | Scene snapshot to one prompt, in a fixed order with stable wording. |
| `state` | `commands/state.py` | `SceneState`: origin (composed or free), base, changes or the free prompt, the operations, undo history. |
| `semantic` | `commands/semantic.py` | Optional CPU vector match for a final clause the dictionary misses. |
| `rewriter` | `commands/rewriter.py` | `MAX_TOTAL_WORDS`; its details rules and validator have had no caller since #271. |

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

1. Steps 1 and 2 as for a partial; `FinalReadingService.handle` sees a final with a valid `t_client` and calls `_final`.
2. It takes `CommandService.snapshot()` and classifies the sentence with `sentence_kind(text, snapshot)`: it splits it with `reader.split_commands`, reads every clause with `reader.read`, and names each clause with `clause_kind` (`BARE_CONTROL`, `SCENE_RESET`, `EDIT`).
3. `SentenceKind.CONTROL` (every clause is a bare undo or reset) goes to `CommandService.handle`, the dictionary path below, with no call and no content check.
4. `FinalReader.lookup(text, snapshot)` gives an `Entry` known without a call, or None:
   1. `blocked_word(text)` checks `BLOCKED_WORDS`; a hit is `Outcome.BLOCKED`.
   2. The key is `(compose(snapshot), sentence_key(text))` (for a free scene `compose` is its own prompt); the pinned entries (`engine/demo_prompts.json`) are checked first, then the in-memory and disk cache.
      `_from_cache` uses a SONNET or PINNED entry only on the same scene (`scene_ids`: origin, free prompt, base and every chip), maps its chip ids to the live ones, and checks every operation again with `checked_operations`.
   3. `_measured` asks `car_template.plan(text, snapshot)`; `Route.MEASURED` (a composed vehicle scene with no objects and one measured world word, or a new vehicle scene from any scene) is answered with `measured_operations` and `world_prompt` (`Outcome.TEMPLATE`).
5. `SentenceKind.MIXED` (bare undo or reset plus edits): when the dictionary does not read every clause by itself, `FinalReader.ask` is called as the content check; a block gives no steer, anything else goes to `CommandService.handle`, which runs the undo or reset in order.
6. The fast path: with no entry, when `commands.service.dictionary_reading(text, snapshot)` reads the whole sentence (a composed scene only), `CommandService.handle` reads it, with no call; `_dictionary` gives a vehicle scene the car template through `fallback_prompt`.
7. Otherwise `FinalReader.ask(text, snapshot, committed_prompt)`: `_call` runs on the thread pool and the caller waits `timeout_s` (3.0 s live).
   1. `_attempt` sends one user message (`user_message`): the prompt on screen (the rich one), the chips, the sentence with a named person replaced (`people.for_model`), `car_scene` (`car_template.is_template` of the prompt on screen), and `view_change` and `person_chip` when they apply.
   2. `check_reply` is the light check: `blocked` gives BLOCKED; `understood` false or an empty prompt gives NOT_UNDERSTOOD; `fit_prompt` cuts a prompt over 180 words or 1600 characters at sentences; `reply_chips` cleans the chips; on a car scene that is not a new scene, `car_template.template_problems` checks the template (only a drivable road with `view_change`).
      The result is one operation `{"op": "free_scene", "prompt", "chips"}`, `Verdict.ACCEPTED`, or `REPAIRED` when something was cut or broken (still used).
   3. REPAIRED and INVALID (not a JSON object) are asked again once, with the reasons, when the first call took under half the wait; a REPAIRED first reply is kept in `Pending.usable` if the retry runs late.
   4. On a timeout the entry is FALLBACK, kept in memory only, and a late answer is counted as `late_dropped`; SONNET, BLOCKED and NOT_UNDERSTOOD entries are saved to the disk cache.
8. `FinalReader.result(entry, text, snapshot)`: `_or_person` turns a FALLBACK on "add <named person>" on a composed scene into the engine's own add (`Outcome.PERSON`), and `_given` remembers the template prompt of a composed vehicle scene for `fallback_prompt`.
9. Back in `FinalReadingService._final`:
   1. `{"blocked": True}` and `{"not_understood": True}` call `CommandService.apply_final(text, [])`: no change, unmatched, and a guess in flight is rolled back.
   2. A result calls `CommandService.apply_final(text, operations)`, which validates and applies the operations as one batch and steers to the composed prompt; `_model_prompts.setdefault(composed, rich)` keeps the rich prompt of a composed state (a free scene's composed prompt is already its prompt).
   3. None (FALLBACK) goes to `CommandService.handle`, the dictionary path; on a free scene it applies only a new scene, so anything else shows as not understood.
10. A tap on the x of a free scene's chip is read as the final "remove the <label>" through steps 2 to 9.

## The path of a final on the dictionary

This runs when the final reader gives None, for CONTROL and MIXED sentences, and for every final with `--final-reader off`.

1. `CommandService._final(text)`: text with non-Latin letters goes to `_final_original`, which asks the slow reader with the original text, or is unmatched when there is none (live).
2. `reader.split_commands(text)` gives 1 to 16 clauses (`MAX_CLAUSES`), or the sentence is unmatched.
3. Each clause is read on a trial copy of the state with `reader.read(clause, snapshot)`.
4. A clause with no operation tries, in order: the vector matcher `_ask_semantic` (at most 0.05 s, skipped for `is_blocked` text), the slow reader `_ask_slow` (None live), and `reader.style_fallback` ("make it spooky").
5. A clause with no operation, or one that does not apply or compose on the trial state, rejects the whole sentence.
6. The operations are applied as one operation or one batch (one undo step) with `_apply`, and `_commit` steers to the new composed prompt, with ramp 12 when a clause said "slowly".

`reader.read(clause, snapshot)` decides in this order: place command (`world_op`), negation or transform (none), move or position (none), undo ("undo the X" is a remove), reset, new scene, the keyword pick, remove (`resolve`), an unknown object after an add verb (`_pass_through`), "show me" as a new scene, a motion edit of an existing object, an add with the user's own description, a strength change of an existing change, and finally a plain dictionary add.

`composer.compose(snapshot)` writes the prompt: a road scene (`world.is_road_scene`) is the commands-side car template (`world.car_prompt`), with an active winter shown as the snowy mountain road (`road_scene_world`); a scene with a world change is the base with its world replaced (`compose_base`), in the world's road form when the base or an object is a vehicle and its landscape form otherwise (`world.Form`, #259), worded for its strength, and the built-in sky segments that clash with a time, weather or season change removed (`world_with_overrides`); any other scene is the base, then the season, weather and time phrases, then the objects.

## Where each rule is decided

| Rule | Where | Notes |
|---|---|---|
| Strength from words ("a little", "a lot", "more", "less") | `reader.strength_for` | Used by `read`, `semantic` and `style_fallback`. The same numbers (0.3, 1.3) are also written in the Sonnet system prompt (`final_reader._OPERATIONS`) and the nano system prompt (`slow_reader.SYSTEM_PROMPT`). |
| Strength range | `state._strength` (0 to 1.3), `slow_reader._strength` (0 to 1.3, rounded to 2 places) | Two checks of one range; only the second rounds. |
| Strength wording in the prompt | `composer.strength_phrase` | One owner; `car_template.scene_objects` calls it. |
| Pace (slowly, suddenly) to ramp | `reader._pace`, `reader.RAMP`, `service._final` and `service.apply_final` | `apply_final` runs the whole `read()` only to get the pace. `RAMP["confirm"]` and `RAMP["sudden"]` equal `RAMP["default"]`, so the swaps to them do nothing. |
| Provisional guess | `service._partial`, `service._commit`, `PROVISIONAL_CAP` | |
| One change per kind; world replaces world | `state.SceneState.apply` | One owner. |
| A world clears season and weather | `state.WORLD_CLEARS` in `SceneState.apply` | Owner. `car_template.measured_operations` also removes time changes for a measured world. |
| Built-in sky of a world vs. an explicit change | `world.world_with_overrides` | Owner; only for known worlds. |
| Winter on a road scene is the snowy mountain road | `world.road_scene_world` (dictionary path) and `car_template.WORLD_WORDS` (engine path) | Two implementations. |
| Night on a car scene | `car_template.WORLD_WORDS` maps night to the neon city; the dictionary path adds a night phrase to the template | The same sentence gives different prompts on the two paths. |
| Is this a vehicle or road scene | `world.is_road_scene` (exact template text, or `_CAR_ON_ROAD_RX`: "car" plus a verb plus a known place), `car_template.vehicle_subject` (`_SUBJECT`: any of 9 vehicle words), `final_reader.is_vehicle` (a vehicle word anywhere) | Three rules. A bare "a red sports car" base is a vehicle scene to the engine and not a road scene to `composer`. |
| Car template text | `world.car_prompt` (dictionary path and every cache key) and `car_template.world_prompt` (engine prompt) | Two templates. They differ in the camera noun, object punctuation, and season or weather sentences. |
| Car template constants | `world.CAMERA`, `STYLE`, `_REAR`, `_FORWARD` and copies in `car_template`; `final_reader.VEHICLE_STYLE` is `world.STYLE` again | Copied constants. |
| Known places | `world._TABLE` (`WORLDS`) | One owner; `car_template._measured` and `final_reader.canonical_world` read it. |
| Season, weather and time labels | `keywords.json` via `dictionary.KEYWORDS` | One owner; `final_reader._vocabulary` and `canonical_change` read it. `car_template.WORLD_WORDS` is a separate word table for vehicle scenes. |
| Sonnet answer into scene state | one `free_scene` operation through `service.apply_final` (#271) | Pinned and measured operations go through `final_reader.checked_operations` (`canonical_world`). |
| Rich prompt per scene state | `FinalReadingService._model_prompts.setdefault` | The first rich prompt for a composed prompt wins for the life of the process. |
| Pinned lookup | `final_reader.load_pinned` (replays `demo_prompts.json` from its base), `FinalReader.__call__`, `_from_cache` | Pinned is checked before the cache, and needs the same scene (`scene_ids`). |
| Cached lookup | `CacheFile`, `FinalReader.__call__`, `_from_cache` | Keyed by `(compose(snapshot), sentence)`; the file is tied to `fingerprint(system_prompt, model, PROMPT_BUILD)`. |
| Unsafe content block | `final_reader.BLOCKED_WORDS` and the model's `blocked` flag | Only for finals that reach the final reader. CONTROL sentences skip it (they add nothing), partials skip it, and with `--final-reader off` there is no block at all. |
| Deliberately unmatched text (negation, transforms) | `reader.is_blocked`, `reader._NEGATION_RX`, `_TRANSFORM_RX`, `world._BLOCKED_RX`, the inline regex in `semantic.SemanticReader.__call__` | Three word lists with small differences. The name `is_blocked` is easy to confuse with the content block. |
| Fallback order on the dictionary path | `service._final`: read, vector, slow reader, style | |
| Fallback order on the engine path | `FinalReadingService._final`: control, block, pinned, cache, measured template, dictionary fast path, Sonnet, dictionary | |
| Phrase and prompt length limits | `slow_reader.MAX_PHRASE` 120, `final_reader.CACHED_MAX_PHRASE`, `final_reader.MAX_PROMPT_WORDS` 180, `state.MAX_FREE_PROMPT_CHARS` 1600 | |
| Batch size | `state.apply` (literal 16), `reader.MAX_CLAUSES` 16, `final_reader.BATCH_LIMIT` 16, `slow_reader.BATCH_LIMIT` 16 | One number in four places. |
| Base edit ("turn the tree into a lighthouse") keeps the changes | Sonnet's free prompt live; `slow_reader.edit_base` (#210) with `--final-reader off` | |
| Free scene or composed scene | `state.Origin` (the snapshot's `origin`) | Owner; `service.dictionary_reading` and `SceneState.apply` refuse dictionary edits of a free scene. |

## Duplicated rules and data

- The car template exists twice: `world.car_prompt` and `car_template.world_prompt`, with copied `CAMERA`, `STYLE`, `_REAR` and `_FORWARD`.
- `VEHICLE_WORDS` is defined in both `car_template.py` and `final_reader.py`; `final_reader.VEHICLE_STYLE` is the same text as `world.STYLE`.
- Vehicle and road scene detection is done three ways (see the table).
- Winter on a car scene is decided twice (`road_scene_world`, `car_template.WORLD_WORDS`), and night on a car scene gives different prompts on the two paths.
- The negation and transform word list exists three times (`reader`, `world`, `semantic`).
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

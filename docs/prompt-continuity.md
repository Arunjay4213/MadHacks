# Prompt continuity and visual memory

Command retrieval embeddings map language to an edit; they do not encode prior video frames.
Scene state retains requested subjects, appearance and motion. The video backend's latent/frame cache retains visual history.
A faithful prompt cannot repair a corrupted anchor, and a clean anchor cannot restore details discarded by the reader.

## Canonical state

“Add a red car driving forward away from the camera” retains that description under the stable car label.
Winter, river and night edits leave it unchanged. “Make the car drive backwards” updates motion while retaining appearance and other modifiers.
A bare repeated addition or partial guess must not replace an already detailed car with the parked dictionary default.
Undo restores the description; removal/reset discard it.
Direct motion verbs: drive, move, walk, run, fly, swim, reverse and stop, plus matching participles. Other wording can use the LLM reader.
Intent preservation does not prove that generated motion follows it.

Confirmed ramps are one chunk: the anchor fires at full target. Explicit gradual edits retain 12 chunks; provisional guesses retain four and cap 0.4.

## Enrichment boundary

The commands.rewriter module exports SYSTEM_PROMPT, build_messages(prompt), and validate_rewrite(prompt, reply).
Input is exact canonical text, with an optional full final command as context.
With build_messages, the model replies with JSON containing only a details string for world, light and texture.
The engine does not use build_messages.
It makes one Sonnet call that reads the command and writes the details together, with SYSTEM_PROMPT as part of its rules and validate_rewrite to check the details.
That call returns JSON {blocked, operations, details}, not only a details string.
Validated output retains the source verbatim and appends bounded detail. Empty details return the source; invalid replies return None.
Target 60–100 total words when useful; maximum 60 added words and 100 total. No new actors, camera moves, motion, events, weather, or claims of seeing previous frames.
Lexical checks cannot prove semantic faithfulness; inspect outputs and compare frames.

Arunjay owns the Foundry call, timeout, bounded cache, likely-prompt prefetch and off switch.
Cache exact canonical input persistently and choose enriched or canonical text before encoding and applying the steer. Sonnet 5.5 is nondeterministic and rejects a temperature setting; cache stability is the engine’s responsibility.
Never swap in a late rewrite mid-scene. Embeddings and token IDs must correspond to the chosen text.
Keep network requests off the generation thread. Cache hits should add no network wait.

Example canonical text: a meadow with an oak tree; winter scenery with snow on the ground; a red car driving forward away from the camera.
Safe detail concerns existing snow texture and diffuse illumination. It must not add fog, a road, people, a camera pan, or reverse motion.

## Frame evaluation

Ishaan owns clean-anchor refresh. His measured diagnosis is that refreshing from the latest drifted latent repeatedly bakes in corruption.
Use the clean anchor saved after the last accepted steer/reset; his PR and frame comparisons determine the final parameters.
Compare identical seeds, duration and command timing with canonical versus enriched prompts.
Inspect initial/reset frames, steer anchors and later refreshes for direction, identity, camera stability and saturation/noise.
Measure cache-hit and cache-miss command-to-shown latency separately, plus generation chunk time.
Text tests alone are insufficient to enable this on the live demo.


## World replacement

Place commands are singleton world changes: desert, snowy mountains, neon city, ocean/beach, Mars, forest and meadow.
They replace the previous world, preserving subjects, objects and the time of day; undo restores the prior world.
A new world also clears the active season and weather, because the world brings its own (#115).
The built-in world dictionary uses Ishaan’s measured templates. Explicit later weather/time/season modifiers remove conflicting tagged atmosphere from those templates.
The default oak/meadow scene and Ishaan’s exact subject/world/style template support safe replacement.
Subject-only bases can acquire a world. An arbitrary base that mixes an unrecognized world description with subjects is rejected rather than edited with unsafe text surgery; normalize such a base into the template first.
World words are recognized only with setting commands, so adding a forest remains an object edit.
The model cannot recover attributes already discarded by a parser: canonical phrases therefore retain explicit user details before enrichment.

# Known issues

Open problems that are documented here to decide later.
Each item says what is wrong, what we know, and the options.
Last updated 2026-11-15 by Arunjay's Claude.

## 1. Two plan gates had no measured result

Gate 3 (stacking several changes and undo) and Gate 4 (pace words such as "slowly", strength words such as "a little", and early steering on partial speech) were measured on the live engine on 2026-11-15 about 03:00 UTC (PR #206, sheets in engine/samples/gate3_* and gate4*).
- **Gate 3: state is right, the picture is not, past about 5 objects.**
  All 14 steps were understood and the chips were right after every step, including 3 undos and "remove the lion".
  But with about 5 or more objects in the scene the model drops some: "add a tree" never showed a second tree and the lion vanished, and one undo showed fewer objects than the chips.
  Keep demo chains to about 4 objects at once.
- **Gate 4a, "slowly": partly.**
  Before 03:52 UTC a slow change did not visibly arrive on live (engine/samples/acceptance_live_0342.jpg): the reader gave strength 0.6 (fixed in #226), and the slow ramp re-anchored at slow_strength 0.5, which does not change the world, so each refresh returned to the old scene.
  Since 03:52 UTC the live engine runs `--reanchor-opt slow_strength=1.0` (#240): "slowly make it winter" reaches full winter in two dissolved steps about 2 s apart and stays (engine/samples/slow_winter_after_0353.jpg).
  Pace words are not used in the demo script.
- **Gate 4b, strength: shown.**
  "a little snowy" gives frosted grass, "snowy" gives deep snow; "a lot more snow" after winter is only a modest change.
- **Gate 4c, early steering: not shown.**
  A partial guess never re-anchors, so nothing visible changes before the sentence is final.
  The pitch must not claim that the scene changes before the speaker finishes.
- **Delay:** pinned sentences and undo about 0.5 s; sentences that need a new Sonnet call 1.7 to 3.2 s.

## 2. Two frontends and two product names

`web/` (Ani) calls the project "Halo".
`lucid-studio/` (Prateek) is called "Lucid Studio".
Judges should see one name and, ideally, one page.
Options: pick one page for the demo and keep the other as a fallback, and pick one name for the page, the pitch and Devpost.

## 3. Model limits we work around instead of solving

- **Every command recomposes the shot.**
  A command re-anchors from noise at full strength, so the camera angle and layout can jump.
  Ishaan's opt-in options (`steer_anchor_t=800`, `steer_noise=fixed`, PR #174) keep the layout (tree moves 7 to 16 px instead of up to 194 px), but winter then only partly arrives on the meadow.
  These numbers are from his newer 2-seed run (https://github.com/Arunjay4213/MadHacks/issues/1#issuecomment-6887678106): on the car chain winter, neon and desert arrive fully (car moves 5 px), on the meadow winter reaches a white share of 0.10 to 0.12 at +3 s against about 0.30 with defaults.
  The earlier one-seed table in `engine/MODEL_NOTES.md` (12 / 12 / 21 px, full snow, night and desert) did not include the meadow winter case.
  They are off on the live engine.
- **Added objects are small or at the edge.**
  Size and position words in the prompt help ("a large ... in the foreground at the left of the frame").
  The lion was the weakest case.
- **A car racing on a road, added to the plain meadow, is lost.**
  Live A/B on 2026-11-15 (issue #267): with the road clause ("a straight grey asphalt road ... runs across the scene") the road is drawn but the car is tiny, oncoming or gone in 4 of 4 runs, also with a longer road clause.
  The car template wording ("A red car, seen from directly behind ..., drives fast along a straight grey asphalt road through a green meadow ...") kept a large car on the road in 2 of 2.
  In the demo chain the same sentence after "make it a desert" works, because the desert world names its highway.
  Options: write a moving vehicle object on a scene without a vehicle base with the car template (a composer change that moves the chain's keys and pins), or keep the demo to the car scene for driving.
- **No real faces.**
  The 1.3B model cannot draw a specific person.
  Named people are mapped to a description of their look (Jensen Huang becomes a man in a black leather jacket).
  A real likeness would need an image-conditioned model or fine-tuning.
- **Resolution 384x672, not the model's native 480x832.**
  On one L40S, 480x832 is not real time (417 ms per chunk against a 250 ms budget).
  An H100 would allow it; it costs about 6.88 USD per hour on demand and is planned only for judging.
  More denoising steps and the full Wan decoder were tested and rejected: slower, and not better looking.
  torch.compile was tested on machine 2 (04:10 UTC): generation 193 to 148 ms median per chunk, but the frames are not equal to the uncompiled run (max pixel difference 255, sharpness up to 45% different per frame), so it stays off; CUDA graphs crash on the model's cache code.
- **A new sentence takes about 2.5 s.**
  Known and pinned sentences take about 0.6 s; a new one waits about 2 s for the Sonnet rewrite.
  In progress: Ishaan's latency plan (median 566 to 300 ms), faster end of speech on the page, and compiled model steps.
  Done, with limits: a sentence that names exactly one measured car world and nothing else skips Sonnet (`FinalReader._measured` in `engine/final_reader.py`, counted in `template_hits`), and so does a sentence that starts a new car scene in a measured world.
  It applies only while the car scene has no added objects: after "add a lion", "make it winter" goes to Sonnet (`car_template.plan`, `has_objects`).
  It also goes to Sonnet when the measured operations do not apply to the scene.
  Not done: skipping Sonnet for any other sentence the dictionary fully understood (objects, weather and time on any scene, and worlds on a scene without the car).
  Not fixed, because it is outside the demo: after a pace sentence such as "slowly make it winter" on the car scene, or "slowly make it night" on the winter meadow, the next pinned step ("add a dog") misses its pin and waits for Sonnet.
  The pinned "slowly" sentence saves a different change than the plain one (a season change instead of the snowy-road world), and pins are keyed by the scene before the sentence (Greptile on #245).
- **A base edit with 16 or more chips is dropped.**
  "turn the tree into a lighthouse" is applied as a new scene plus one add for every existing change, and `commands/state.py` allows at most 16 operations in one batch, so with 16 or more chips it is rejected on both the Sonnet and the dictionary path, and with 15 when the same sentence also adds something ("... and add a dog") (Greptile on #210).
  Not fixed: the demo keeps about 4 objects at once (Gate 3). A fix would add a base-only operation to `SceneState`.

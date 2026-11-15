# Known issues

Open problems that are documented here to decide later.
Each item says what is wrong, what we know, and the options.
Last updated 2026-11-15 by Arunjay's Claude.

## 1. Two plan gates had no measured result

Gate 3 (stacking several changes and undo) and Gate 4 (pace words such as "slowly", strength words such as "a little", and early steering on partial speech) were never measured on the real model until a run on 2026-11-15 about 03:00 UTC.
The result of that run is in the "Status" section of PLAN.md.
Early steering probably does not show: a provisional guess never triggers a re-anchor, and without a re-anchor the picture does not change (Gate 2 finding).
The pitch must only claim what that run shows.

## 2. Two frontends and two product names

`web/` (Ani) calls the project "Halo".
`lucid-studio/` (Prateek) is called "Lucid Studio".
Judges should see one name and, ideally, one page.
Options: pick one page for the demo and keep the other as a fallback, and pick one name for the page, the pitch and Devpost.

## 3. Model limits we work around instead of solving

- **Every command recomposes the shot.**
  A command re-anchors from noise at full strength, so the camera angle and layout can jump.
  Ishaan's opt-in options (`steer_anchor_t=800`, `steer_noise=fixed`, PR #174) keep the layout (tree moves 7 to 16 px instead of up to 194 px), but winter then only partly arrives on the meadow.
  They are off on the live engine.
- **Added objects are small or at the edge.**
  Size and position words in the prompt help ("a large ... in the foreground at the left of the frame").
  The lion was the weakest case.
- **No real faces.**
  The 1.3B model cannot draw a specific person.
  Named people are mapped to a description of their look (Jensen Huang becomes a man in a black leather jacket).
  A real likeness would need an image-conditioned model or fine-tuning.
- **Resolution 384x672, not the model's native 480x832.**
  On one L40S, 480x832 is not real time (417 ms per chunk against a 250 ms budget).
  An H100 would allow it; it costs about 6.88 USD per hour on demand and is planned only for judging.
  More denoising steps and the full Wan decoder were tested and rejected: slower, and not better looking.
- **A new sentence takes about 2.5 s.**
  Known and pinned sentences take about 0.6 s; a new one waits about 2 s for the Sonnet rewrite.
  In progress: Ishaan's latency plan (median 566 to 300 ms), faster end of speech on the page, and compiled model steps.
  Not done: skipping Sonnet when the dictionary fully understood the sentence on a scene that uses the measured template.

# Pitch and submission draft

Use this copy after checking it against the final demo. Remove every feature that
did not work in rehearsal; replace bracketed measurements with real results.

## Pitch

We build on Training-Free Interactive World Models to let you shape a scene as it unfolds.

A video that never stops generating and changes when you speak.
A sentence we already know (pinned or cached) changes the picture 429 to 674 ms after the command.
A new sentence adds about 2 s for the Sonnet call.

You say "make it winter" and it snows. You say "also add a river" and a river
appears while the snow stays. You say "remove the river" and only the river goes.

Nothing is trained. All of it happens while the model is running.

Your changes stay on screen as named chips.
Undo takes you back a step.
Say "a little" to keep a change subtle.
The video keeps moving while you shape the scene.

## Three slides: content

1. **Speak, and the video changes.** Show winter → winter + river → winter + river
   + night. Use captured demo images only once those changes work visibly.
2. **Words to state to pixels.** Text/mic → keyword or slow reader → reversible
   scene state → deterministic prompt → steerer → engine → browser. Mark the
   Training-Free Interactive World Models method and highlight the command, state and steering modules.
3. **What we measured and what remains.** Report [actual GPU], [resolution/FPS],
   [command-to-frame proxy], [visible switch behavior], and [failure cases]. List
   typed and voice results separately. CPU unit tests are not GPU benchmarks.

## Submission outline

- **Inspiration:** interactive visual storytelling controlled by speech.
- **What it does:** include only features demonstrated on the final integrated system.
- **How we built it:** built on Training-Free Interactive World Models, with a Python command/state layer, the team's steering and engine, Ani's browser interface, and Lucid Studio, Prateek's second, immersive frontend served at /studio/.
  Name the exact pretrained model and version used.
- **Challenges:** provisional speech revisions, stable phrases across prompt
  changes, coherent removals, and running generation within the compute budget.
- **What we learned:** fill in from the measured engine runs and rehearsals.
- **Limitations:** keyword vocabulary, optional slow-reader setup, semantic edits
  that disturb other content, and actual latency/failure results.

## Internal preparation notes

- The team's architecture, gates and role split are in the root `PLAN.md` and
  `AGENTS.md`; accepted messages are in `contracts/`.
- Competition and award comparisons stay out of the pitch.
- The engine team selects and verifies the upstream model, code and weight
  licenses. Copy their final exact model identifiers and source links here before
  submitting. Do not copy unlicensed SPED project files into this repository.
- The delays in the pitch are measured from command to changed picture: 429 to 674 ms for a pinned or cached sentence, plus about 2 s for the Sonnet call on a new sentence.
  Check them again on the final demo machine before presenting them.
- Do not claim "slowly", "a lot more snow" or early steering on partial words: none is measured on the real model, and a provisional guess never re-anchors.
- Do not claim the first real-time video generator/editor or use another project's
  latency numbers as our measurements.

No submission or external publication has been made by these drafts.

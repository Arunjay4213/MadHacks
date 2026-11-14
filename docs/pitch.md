# Pitch and submission draft

Use this copy after checking it against the final demo. Remove every feature that
did not work in rehearsal; replace bracketed measurements with real results.

## Pitch

A video that never stops generating and changes within a fraction of a second
when you speak.

You say "make it winter" and it snows. You say "also add a river" and a river
appears while the snow stays. You say "remove the river" and only the river goes.

Nothing is trained. All of it happens while the model is running.

We build on interactive world models to let you shape the scene as it unfolds.

Your changes stay on screen as named chips. Undo takes you back a step. Say
"slowly" to ease into a change, or "a little" to keep it subtle. The video keeps
moving while you shape the scene.

## Three slides: content

1. **Speak, and the video changes.** Show winter → winter + river → winter + river
   + night. Use captured demo images only once those changes work visibly.
2. **Words to state to pixels.** Text/mic → keyword or slow reader → reversible
   scene state → deterministic prompt → steerer → engine → browser. Mark the
   interactive world model and highlight the command, state and steering modules.
3. **What we measured and what remains.** Report [actual GPU], [resolution/FPS],
   [command-to-frame proxy], [visible switch behavior], and [failure cases]. List
   typed and voice results separately. CPU unit tests are not GPU benchmarks.

## Submission outline

- **Inspiration:** interactive visual storytelling controlled by speech.
- **What it does:** include only features demonstrated on the final integrated system.
- **How we built it:** built on interactive world models, with a Python command/state layer, the team's steering and engine,
  and Ani's browser interface. Name the exact pretrained model and version used.
- **Challenges:** provisional speech revisions, stable phrases across prompt
  changes, coherent removals, and running generation within the compute budget.
- **What we learned:** fill in from the measured engine runs and rehearsals.
- **Limitations:** keyword vocabulary, optional slow-reader setup, semantic edits
  that disturb other content, and actual latency/failure results.

## Internal preparation notes

- The team's architecture, gates and role split are in the root `PLAN.md` and
  `AGENTS.md`; accepted messages are in `contracts/`.
- Prior-project comparisons in the root plan are research references, not pitch copy.
- The engine team selects and verifies the upstream model, code and weight
  licenses. Copy their final exact model identifiers and source links here before
  submitting. Do not copy unlicensed SPED project files into this repository.
- Check "within a fraction of a second" against our end-to-end measurements before
  presenting it as an achieved result. The pitch above is the intended experience.
- Do not claim the first real-time video generator/editor or use another project's
  latency numbers as our measurements.

No submission or external publication has been made by these drafts.

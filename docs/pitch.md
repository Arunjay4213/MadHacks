# Pitch and submission draft

Use this copy after checking it against the final demo. Remove every feature that
did not work in rehearsal; replace bracketed measurements with real results.

## Pitch

We are building a generated world that responds to spoken changes. A command like
“make it winter” changes the target scene. “Also add a river” keeps winter and adds
a river. The user can remove a specific change, undo an operation, and control
the transition's pace and descriptive strength.

Our command layer tracks a base scene, named changes and undo history. A fast
keyword reader handles common commands; an optional injected language-model reader
handles other requests. Partial speech can propose a capped change without
committing it. Revised speech or an unmatched final rolls that guess back.

The video model and the idea of smoothly blending text conditioning are reused.
Our work connects explicit scene operations, reversible state and partial-speech
handling to the team's interruptible steering and live interface. No model is
trained by this command subsystem.

## Three slides: content

1. **Say a change, keep the rest.** Show winter → winter + river → winter + river
   + night. Use captured demo images only once those changes work visibly.
2. **Words to state to pixels.** Text/mic → keyword or slow reader → reversible
   scene state → deterministic prompt → steerer → engine → browser. Mark the
   inherited model and blending method, and highlight the team's added modules.
3. **What we measured and what remains.** Report [actual GPU], [resolution/FPS],
   [command-to-frame proxy], [visible switch behavior], and [failure cases]. List
   typed and voice results separately. CPU unit tests are not GPU benchmarks.

## Submission outline

- **Inspiration:** interactive visual storytelling controlled by speech.
- **What it does:** include only features demonstrated on the final integrated system.
- **How we built it:** Python command/state layer, the team's steering and engine,
  and Ani's browser interface. Name the exact pretrained model and version used.
- **Challenges:** provisional speech revisions, stable phrases across prompt
  changes, coherent removals, and running generation within the compute budget.
- **What we learned:** fill in from the measured engine runs and rehearsals.
- **Limitations:** keyword vocabulary, optional slow-reader setup, semantic edits
  that disturb other content, and actual latency/failure results.

## Credits and provenance

- The team's architecture, gates and role split are in the root `PLAN.md` and
  `AGENTS.md`; accepted messages are in `contracts/`.
- The root plan credits [Training-Free Interactive World Models](https://hrdlabs.org/research/interactive-world-models/)
  for the conditioning-blend method. Attribute that method in the spoken pitch
  and submission. Check the source's claims before quoting its timings or prizes.
- The engine team selects and verifies the upstream model, code and weight
  licenses. Copy their final exact model identifiers and source links here before
  submitting. Do not copy unlicensed SPED project files into this repository.
- Do not claim the first real-time video generator/editor. Do not quote inherited
  latency numbers as our measurements.

No submission or external publication has been made by these drafts.

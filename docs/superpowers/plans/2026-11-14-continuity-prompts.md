# Continuity-aware prompts Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Retain explicit scene subjects and motion across edits, and give the engine a bounded continuity-aware enrichment interface.
**Architecture:** Canonical scene state remains the source of truth. Commands preserve described objects and literal motion edits; engine caches enrichment by canonical prompt before text encoding. Visual memory remains clean latent anchors owned by steering, not language-retrieval embeddings.
**Tech Stack:** Existing Python standard library, existing engine Foundry client.
**Spec:** Arunjay board comment 6885482021 and Ishaan 6885561547; user authorizes implementation.

## Global Constraints
- No frontend changes, no added dependencies, no secret access.
- Engine and steering changes remain with their owners; request handoff before edits.
- Preserve canonical subject/motion/camera clauses word for word across weather edits.
- No late rewrite swaps: choose rewritten or fallback prompt before steer; cache stable text.
- Enrichment targets 60–100 words and never asserts access to video frames.

## Approaches
- Text retrieval alone cannot retain image history; rejected as visual memory.
- Full freeform rewrite risks changing subjects and token alignment; prefer immutable source clauses plus bounded enhancement, with owners agreeing the integration.
- Clean saved latent refresh is Ishaan's measured visual-memory fix; coordinate its landing.

## Review Focus
- Explicit red/forward/backward descriptions must survive winter/river edits.
- Reversing motion preserves subject identity and unrelated modifiers.
- Generic repeated additions must not overwrite an existing detailed subject.
- Reset/removal/undo must not resurrect deleted details through cached text.
- Invalid enrichment must fall back to canonical input; no speculative claims of observed frames.

### Task 1 — Canonical detail retention (Codex)
- [ ] Add failing tests in commands/tests/test_continuity.py for described object, motion change, weather stacking, generic repeat, undo/remove/reset.
- [ ] Modify reader.py only as needed, reusing add operations and stable labels.
- [ ] Run command and affected engine tests.

### Task 2 — Enrichment boundary (Claude + engine owner)
- [ ] Agree pure interface with engine owner; implement commands/rewriter.py with bounded validation and system instructions.
- [ ] Add focused failing/passing tests in commands/tests/test_rewriter.py.
- [ ] Engine owner implements cached/precomputed Foundry integration, timeout fallback, off switch.

### Task 3 — Review and integration
- [ ] Claude reviews Codex reader changes; Codex reviews rewrite module.
- [ ] Validate compound edits/undo and stable base/motion across world changes.
- [ ] Fetch/rebase, PR and merge commands; post interface and measured limits.
- [ ] Coordinate engine/steering PRs and frame comparison; report any unverified GPU behavior.

# Lucid Studio delivery checkpoint

Prateek explicitly lifted the pause on 2026-11-14 and authorized completion,
PR and merge for issue #40. Work is isolated at
`/Users/prateek/.codex/worktrees/3210/MadHacks` on `40-lucid-studio`.
The shared `74-continuity-prompts` checkout was not switched, stashed or reset.

## Completed

- Preserved editorial hero with a real Enter Studio button; natural scrolling into
  a contained editor, original prompt and generation, scene memory, and a reserved
  bottom-right voice rail. Short laptops and mobile use normal document scrolling.
- Ani web parity, including #82: backtick/button push-to-talk, one final per held
  utterance, leftover interim flush, deduped/restarting toggle voice, text partials,
  Wispr whole inserts, 2.5s/600ms typing/burst pauses, unfinished sentence guard,
  IME, focus restoration and Option/Alt+R reset.
- Server-authoritative keyed chips and highlighted/unmatched words, with #84
  hearing/heard/understood labels and correlation that never calls a draft understood; disabled
  disconnected mutations with editable retained text; persistent connection and
  >2s frame-stall notices; `?server=` and `?engine=` aliases.
- One-shot local-command latency and stage stamps captured from the same sample;
  six-byte frame parser, intrinsic size, async decode ordering and epoch cleanup.
- Shader reserved name remains `pulseActive`; fresh Chromium world/particle
  compilation is clean. Both WebGL contexts have independent fallbacks.
- External sources provide the poster immediately; a send-only custom adapter
  preserves the labeled procedural simulator. Simulation is never labeled AI video.
- Permission races cannot cancel a newer microphone request; late tracks are stopped.
- Automated PR review findings reproduced and fixed: ordered custom-adapter commands,
  retry of a second whole insert, hidden-source GPU pause and frame-before-control timing.

## Verification, 2026-11-14

- `npm test`: 22 passing.
- `npm run build`: passes (known lazy Three bundle size warning).
- `npm audit`: zero vulnerabilities.
- `LUCID_ENGINE_URL=http://127.0.0.1:8100 npm run test:browser`: 17 passing.
  Covers 1440x900, 1280x720, 390x844, 375x667; real production `/studio/`
  against a local CPU fake engine (winter/river/remove/undo/reset/unmatched),
  controlled speech, WebGL denial, source attribution, async decode and mic cleanup.
- Independent code review: four Important findings fixed and covered by browser or
  unit regressions (particle fallback, focus, send-only adapter, correlated stages).
- Real OS Wispr insertion and actual spoken recognition need a demo-machine check;
  browser tests use controlled speech/audio fixtures. Shared GPU engine was untouched.

## Backend handoff

Output remains `lucid-studio/dist`, Vite base `./`. Engine owner builds/copies dist
from main to serve `/studio/`; no frontend build artifacts are committed.
A non-default original prompt must be supplied as `initialPrompt` because contract 4
has no base-prompt field. Spatial edits remain simulator-only for the shared socket.

Local fake-engine reset followed immediately by a final can acknowledge state but
leave `applies_from_chunk` null. Reported to Arunjay on board #6885926316; no backend
edit made. The UI keeps “—” rather than inventing a delay. The integration test waits
for Reset to be scheduled before sending its next final.

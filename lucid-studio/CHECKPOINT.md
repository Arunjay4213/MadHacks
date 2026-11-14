# Frontend paused at Prateek's request

Paused 2026-11-14. Resume this work only when Prateek asks; current priority is
commands keyword recognition / embedded dictionary (issue #59).

Branch: `40-lucid-studio`, issue #40. Do not merge or deploy this checkpoint.

## User's current design

- Preserve the editorial hero. Enter Studio must be a real button.
- Natural scroll into a clean Adobe-style editor; no scroll zoom or full-screen takeover.
- Contained current-generation canvas, original prompt, and separated controls.
- Voice agent in a reserved bottom-right rail dock, not overlapping inputs or memory.
- All functionality of Ani's `web/`, with the improved styling. This parity pass is unfinished.
- Keep the procedural simulator and typed/external stream support; never call simulation live AI output.

## Implemented checkpoint

React/TypeScript/Vite app, Lenis natural scrolling, shader alpine world with
composable weather/river/light layers, R3F optical pulses, 6000 voice particles,
FFT + synthetic envelope, speech/text state hook, transient captions and pinning,
scene memory and undo/reset, source and websocket adapters. Shader reserved-word
bug `active` was fixed to `pulseActive` in `src/shaders/post.ts`; a fresh browser
load rendered the valley. Last recorded tests/build: 11 Vitest tests pass and Vite
production build passes, before the final rail-spacing changes.

Performance work: ~105KB gzip initial UI chunk with lazy GPU modules; procedural
target capped at 700k pixels, off-screen demand rendering, world pass bypass for
live textures, frame-version uploads, pointer refs and FFT state local to agent.

## Resume checklist

1. Fresh browser/console QA of latest rail spacing and Enter Studio button. Check
   hero poster, editor layout at laptop/mobile sizes, voice particles, mic errors.
2. Complete parity against current `web/app.js`, `web/lib.js`, `web/README.md`:
   hold-to-talk button/backtick (never Ctrl), leftover-interim final flush, text
   partials and focus retention for Wispr, whole inserts/IME handling, keyed server
   words with keyword highlight and unmatched styling, persistent connection/video
   stall notice after 2s, once-per-command delay + server stamp breakdown, disabled
   disconnected actions, `?server=` host:port/HTTPS/WSS aliases, reset shortcut.
3. Read Ani's latest reviews on board. Engine lookahead and unmatched-final
   t_client are engine-owned; they were flagged to Arunjay. Do not silently edit
   `web/`, `engine/` or `contracts/`.
4. Recheck async frame generation guards, source cleanup, mic cleanup, and failed
   WebGL fallback. Independent review caught stuck interim text and loading overlay;
   both were addressed, but the parity/refactor still needs end-to-end verification.
5. Run `npm test`, `npm run build`, `npm audit`; local fake-engine integration;
   then code review, PR and merge. No frontend PR has been opened yet.

Hosting: output `lucid-studio/dist`, Vite base `./`. `/studio/` defaults to page
origin; standalone Vite root simulates unless `?engine=<full-url>` is supplied.
Engine owner must build source before serving dist. Current preview was on port
5178; it is stopped for this pause. Third-party noise notice lives in
`public/THIRD_PARTY_NOTICES.txt`.

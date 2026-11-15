# Lucid Studio implementation plan

Goal: implement the supplied immersive frontend spec in an independent root app.
Spec: `DESIGN.md` plus the user's detailed component brief.
Stack: React 18, TypeScript, Vite, Three/R3F/drei, Framer Motion, Lenis, Tailwind.

## 1. State and adapter (Codex)
- Write and run failing `src/world.test.ts`: stacking/removal, snow alias,
  season/time replacement, undo/reset, intensity limits and normalized coordinates.
- Implement minimum pure transitions in `src/world.ts`; expose the exact
  `WorldStreamAdapter` from `src/types.ts`; run tests.
- Add a websocket bridge using contracts 1/4 and the six-byte frame header.
  In live mode, server state owns modifiers. Drop stale outbound commands.

## 2. World and optics (Claude, independent)
- Implement `src/components/WorldCanvas.tsx` and shader-only files. Consume
  `WorldCanvasProps`; render a continuous alpine valley and independently damped
  weather/river/light modifiers. Add external stream textures and optical pulse.
- Implement `AcousticSingularity.tsx` using 6000 GPU particles and audio bands.
  Keep resources bounded and dispose external video/texture resources.

## 3. Input and shell (Codex)
- Implement `useGenerativeWorld.ts`, `useAudioBands.ts`, `ThresholdShell.tsx`:
  voice interim/final updates, mic permission/error handling and cleanup, automatic
  restart, simulated envelope, typing/keyboard fallback and reduced motion.
- Test transcript final deduplication and binary frame parsing.

## 4. Ephemeral UI (Codex)
- Implement `WhisperCaption.tsx`, `LucidOverlay.tsx`, `App.tsx` and CSS. Words blur
  in then dissolve, click pins expire, Tab/hover exposes removable memory with
  wheel intensity. No persistent boxes or buttons in studio.

## 5. Integration and verification
- Run unit tests and TypeScript/Vite production build; fix actual failures.
- Browser-check threshold, studio lock, five simulated edits, independent removal,
  Tab memory, pin expiry, undo/reset, typed input, denied mic and responsive layout.
- Verify external websocket source against the CPU fake engine where available.
  Review shaders visually and handle WebGL failure without a blank page.

## 6. Delivery
- Document commands, input gestures, adapter capabilities and UI gaps in README.
- Review the full change, commit only this app, fetch/rebase, open PR closing #40,
  merge once checks pass, post integration notes to the board. Local preview only.

## User revision: editor and latency

Replace full-screen scroll takeover with a preserved hero and natural scroll into
a contained editor. Show original prompt and server-authoritative scene memory,
with a reserved bottom-right rail dock. Use one WebGL renderer, a first-frame hero
poster, localized FFT updates, lazy GPU components and the 700k procedural pixel
budget. Browser-check the revised editor and keep production asset base `./` for
the engine owner's `/studio/` route.

## Authorized resume, 2026-11-14

Prateek lifted the pause and explicitly authorized completing this design and
merging #40. Resume in the managed worktree; modify only `lucid-studio/`.

- [x] Input parity: regress leftover speech flush, whole insert/burst classification,
  shortcut modifiers; update input hook, both text inputs, voice dock and App.
- [x] Engine parity: regress optional strengths, heard/keyword/kind/fake/stamps,
  invalid payloads, URL aliases, stall states and one-shot latency. Update stream
  hook and keyed server words; disable mutations while disconnected.
- [x] QA: reproduce shader failures on a fresh browser; desktop/mobile contained
  layout, denied mic, restart, IME/Wispr, cleanup and local CPU fake engine.
- Delivery: PR [#94](https://github.com/Arunjay4213/MadHacks/pull/94) carries
  `Closes #40`; tests/build/audit and independent review are complete. Merge and
  board/engine/lauda notifications follow the authorized repository workflow.

Keep local procedural output explicitly labeled; label an engine fake only when
its message says `fake: true`. Keep source props distinct from a requested engine.
Use native scroll and the existing rail rather than a second editor implementation.

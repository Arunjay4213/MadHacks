# Lucid Studio design

The user's supplied Zero-Chrome Lucid Canvas specification is the design baseline.
This is a separate local React/TypeScript/Vite app; Ani's `web/` is unchanged.
Default source is a continuously animated procedural alpine world, labeled as a
simulation on entry and in the temporary help view. A source prop accepts a real
MediaStream or frame canvas, and a typed adapter sends mutations outward.

Use a shader-rendered world rather than a prerecorded landscape: the requested
winter/river/night/rain changes must be independently visible immediately. React
state owns simulation modifiers; a live-engine integration must use the server's
state. No invented GPU latency or semantic spatial-edit capability is claimed.

User revision: preserve the hero, remove scroll zoom and the full-screen takeover.
Normal Lenis scrolling (lerp 0.065) leads into an editor with a contained generation
canvas, original prompt, scene memory and quick command input. The voice agent
lives at the bottom right of the screen. A one-time shader poster preserves the
hero preview without a second continuously running renderer. Audio particles,
transcripts and spatial optics remain ephemeral; Tab reveals optical memory.
`/` opens typing; 1–5 simulate commands; V enables mic. Undo/reset are also visible
in the prompt column. Respect reduced motion and preserve keyboard access.

Optimize perceived latency by keeping input feedback independent of backend work:
pointer refs, audio updates localized to the voice agent, lazy shader chunks,
700k procedural pixel cap, off-screen pause and skipping procedural work behind
the live texture. Do not turn these bounds into an unmeasured latency claim.

The existing engine additionally needs connection/error/unmatched feedback,
authoritative state and measured frame timing. Keep these ephemeral. Its existing
contracts do not implement spatial coordinates or direct intensity scrubbing;
report those boundaries rather than changing teammates' code.

Codex owns state, input, audio analysis, threshold, overlays, tests and integration.
Claude owns WorldCanvas, AcousticSingularity and shader-only files. Shared props
are frozen in `src/types.ts`. All new files stay in `lucid-studio/`.

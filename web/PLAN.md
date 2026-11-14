# web/ plan (issue #6, branch `6-web-page`)

Owner: `@anirudhaugile` (Ani), since the lead's swap on 2026-11-14. Started by `@prateekraouw` (handoff commit 157ebbd).

## DOM interface (index.html must provide these ids)

| id | element | app.js does |
|---|---|---|
| `video` | `<canvas>` | draws decoded JPEG frames, newest only |
| `connection` | any | `textContent` = status; `data-state` = `connecting` / `open` / `closed` |
| `mode-banner` | any | `hidden = false` while the server says `fake: true`; `hidden = true` when it says `fake: false`. Leave it visible in markup so it shows before the first message. |
| `words` | any, `aria-live="polite"` | children replaced: text nodes plus one `<mark>` around the keyword; `data-unmatched="true"` when no operation; `data-kind` = `partial` / `final` |
| `changes` | any (list container) | children replaced from server state; `data-stale="true"` while disconnected |
| `command-form` | `<form>` | submit is `preventDefault`ed (no reload); submit sends `final` |
| `command` | `<input type="text">` | kept focused; input events send messages; cleared after `final` |
| `mic` | `<button>` | toggle listening; `aria-pressed`; `disabled` + `title` if no speech API |
| `ptt` | `<button>` | push-to-talk while pointer held; `aria-pressed` while held. The backtick key also works as push-to-talk. |
| `undo` | `<button>` | sends undo; `disabled` while disconnected |
| `latency` | any | `textContent` = `"— "` (unavailable) or `"<n> ms"` |
| `timing` | any | `textContent` = what the number measures, plus server stamps if any |
| `error` | any, `role="alert"` | `textContent` = last error, `hidden` when none |

Chips that app.js creates inside `#changes`: `<li class="chip" data-id>` containing `<span class="chip-label">` and `<button class="chip-remove" aria-label="Remove <label>">×</button>`.
So `#changes` should be a `<ul>`.

## Transport (agreed: `contracts/README.md`, "Wire details")

It sits in `PROTOCOL` in `lib.js` and in `fake-server.mjs`.
- One origin. `WS /control`: contract 1 in, contract 4 out, both JSON text. `WS /frames`: binary.
- Frame = uint32 big-endian chunk number + uint16 big-endian frame index + JPEG bytes. 4 frames per chunk.
- Contract 4 arrives once per chunk with every field; the page re-renders chips and words only when they change.
- The fake server adds `fake: true` (schemas allow extra fields); the banner shows only while it is true.
- `?server=host:port` points the page at another server, e.g. the engine on port 8000.

## Behaviour rules

- Chips render only from the last valid server `changes`. Clicking never changes chips locally.
- Commands are sent only while the control socket is open; otherwise they are dropped and the error says so. Nothing is replayed after a reconnect.
- Text box: paste / drop / replacement / multi-character insert sends one `final` with the whole value and clears the box.
  Typing sends `partial` per input and `final` on Enter or after 1500 ms idle (`TYPING_IDLE_MS`).
  Machine-fast keys (4 in a row, each within 30 ms: a tool simulating keystrokes) become final after 200 ms (`BURST_IDLE_MS`).
  Input during IME composition is ignored until the composed text lands.
  `?debug=1` logs each input's `inputType` to the console, to check how Wispr Flow inserts on the demo laptop.
- Speech: interim results send `partial` (only when the text changed). Each final result is sent once, tracked by its index.
  Stopping (mic off or PTT release) flushes leftover interim text as one `final`, once.
- Frames: each received frame gets a sequence number; after async decode it is drawn only if its sequence number is newer than the last drawn one.
- Delay: shown only if the server echoes a `t_client` this page sent plus `applies_from_chunk`.
  Value = time the first frame tagged with chunk >= `applies_from_chunk` is drawn onto the canvas, minus `t_client`, both on the page clock.
  That is a chunk-tag proxy, not proof the picture visibly changed, and not proof of paint on screen; `#timing` says so. Otherwise `—`.
  Each command is measured once: the per-chunk echo of the same `t_client` never produces a second (growing) number.
- Invalid server messages are ignored (error shown with `textContent`); they never throw out of handlers.

## Checks

- `cd web && npm install && npm test` (node --test: `test/lib.test.mjs`, `test/fake-server.test.mjs`).
- `npm start`, then open http://localhost:8787 in Chrome: paste, type, mic, PTT, chip x, undo, kill server to see reconnect.

## Status (PLAN.md "Teammate 2" items)

1. Page against the fake server: done. Text box keeps focus (buttons never take it; blur and window focus refocus).
2. Text box sends: done (paste at once, simulated keys after 200 ms, typing after 1.5 s; no Enter needed).
3. Live mic: done. Push-to-talk key is `PUSH_TO_TALK_CODE` in `lib.js` (default backtick), ignored with Ctrl/Cmd/Alt
   so it never clashes with Wispr Flow's Ctrl hotkey, and never types into the box. The mic toggle restarts Chrome's
   recognizer when it ends by itself; blocked-mic and network errors are explained; the mic is disabled outside a
   secure context (HTTPS or localhost). Needs a real voice test on the demo laptop.
4. to 6.: chips and undo on real state, delay and highlight polish, big-screen layout.

## Open

- Wispr Flow's real `inputType` must be checked on the demo laptop.
- Connecting to the engine (fake backend works now) needs its Python environment.

# web/ plan (issue #6, branch `6-web-page`)

Owner: `@prateekraouw`. Claude: `lib.js`, `app.js`, `fake-server.mjs`, `package.json`, tests, this file, README.
Codex: `index.html`, `style.css` (markup, styles, accessibility). Neither edits the other's files.

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

## Transport (MOCK / PROVISIONAL, not agreed by the lead; asked on the board)

All of it sits in `PROTOCOL` in `lib.js` and in `fake-server.mjs`.
- One origin. `WS /control`: contract 1 in, contract 4 out, both JSON text. `WS /frames`: binary.
- Frame = 4-byte big-endian uint32 chunk number + JPEG bytes.
- Contract 1 additions: `{"type":"tap","action":"undo"}`, `{"type":"tap","action":"reset"}`.
- Contract 4 additions: `t_client` (echo of the command), `applies_from_chunk`, `heard`, `keyword`, `unmatched`, `fake`, `error`.
  Every message carries the full `changes` list.

## Behaviour rules

- Chips render only from the last valid server `changes`. Clicking never changes chips locally.
- Commands are sent only while the control socket is open; otherwise they are dropped and the error says so. Nothing is replayed after a reconnect.
- Text box: paste / drop / replacement / multi-character insert sends one `final` with the whole value and clears the box.
  Typing sends `partial` per input and `final` on Enter or after 1500 ms idle.
- Speech: interim results send `partial` (only when the text changed). Each final result is sent once, tracked by its index.
  Stopping (mic off or PTT release) flushes leftover interim text as one `final`, once.
- Frames: each received frame gets a sequence number; after async decode it is drawn only if its sequence number is newer than the last drawn one.
- Delay: shown only if the server echoes a `t_client` this page sent plus `applies_from_chunk`.
  Value = time the first frame tagged with chunk >= `applies_from_chunk` is drawn onto the canvas, minus `t_client`, both on the page clock.
  That is a chunk-tag proxy, not proof the picture visibly changed, and not proof of paint on screen; `#timing` says so. Otherwise `—`.
- Invalid server messages are ignored (error shown with `textContent`); they never throw out of handlers.

## Checks

- `cd web && npm install && npm test` (node --test: `test/lib.test.mjs`, `test/fake-server.test.mjs`).
- `npm start`, then open http://localhost:8787 in Chrome: paste, type, mic, PTT, chip x, undo, kill server to see reconnect.

## Blockers

- Lead decision on transport points above (and Ani on contract 4 state fields).
- Real-server connection waits on `engine/`.
- Wispr Flow's real `inputType` must be checked on the demo laptop.

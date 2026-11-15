# web/: the demo page and its inputs

Owner `@anirudhaugile` (Ani), issue #6.
See `PLAN.md` here for the DOM ids, behaviour rules and status.

## Run it

`npm ci` comes first: it installs the exact dependency versions, and `npm test` and `npm start` need them.

```sh
cd web
npm ci          # first, once
npm test        # unit tests (lib.js) + fake-server integration tests
npm start       # MOCK fake server on http://127.0.0.1:8787 (PORT to change)
```

Open http://localhost:8787 in Chrome.
The live mic needs Chrome's speech recognition and a secure context: localhost works, a plain `http://<ip>` address does not.

The fake server is a mock.
It loops an ffmpeg test pattern (the committed frames in `test-pattern/` if ffmpeg is missing; regenerate them with `python tools/make-test-pattern.py`), spots a dozen keywords, and says `fake: true` in every message so the page shows the mock banner.
Its wire format follows `contracts/README.md`.

## What the page shows

The layout is the big-screen design from PR #104.

- The video, large and centred, drawn at the size the engine sends (672x384 for the live model) and scaled up at its own aspect ratio.
- A status pill at the top left: "Connecting…", "Connected" or "Reconnecting…", and a "MOCK server" banner when the server is the fake one.
- The delay badge at the top right: the time from your command to the first frame of the chunk it applies to, with the server's stage stamps in its tooltip.
  It shows a dash until a command has been measured, and an unmatched command is never measured.
- The words line under the video: "heard" with the full sentence, then "→ understood: <keyword>" or "→ not understood" (PR #84).
  Partial speech never shows as understood.
- The chips: one per change the server holds, each with a × to remove it.
- The dock at the bottom: the text box, Mic, Hold to talk, Undo, Reset and Full screen, with their keys shown.
- A notice on the video for a lost connection or a stalled stream.

## Controls and keys

| Control | What it does |
|---|---|
| Text box, Enter | Sends the sentence. |
| Typing pause of 2.5 s | Sends the sentence, unless it looks unfinished (ends in a comma, "and", "a", "on", "make it" and similar); then it waits for Enter. Endings like "start over", "undo it", "slowly" and "more" send normally (PR #114). |
| Wispr Flow: hold Ctrl, speak, release | The sentence lands in the text box and is sent at once. |
| Hold to talk, or hold the backtick key (`` ` ``, under Esc) | Push-to-talk: words show while you speak, and one press is one command, sent on release. A quick second press records at once, and commands reach the server in the order you pressed (PR #114). |
| Mic | Keeps listening hands-free until clicked again. |
| Undo | Steps back one change. |
| × on a chip | Removes that change only. |
| Reset, or Option+R / Alt+R | Goes back to the starting scene and starts a new video. |
| Full screen (Esc leaves) | Full screen for the demo; use Chrome's View, Enter Full Screen if the browser blocks it. |

Speech while disconnected is dropped, never replayed.

## How it is served

### From the engine at `/app/` (the normal way)

The engine serves this folder at `https://<tunnel address>/app/`, on the same origin as the video, so the mic works with nothing to install.
The tunnel address is posted on the board.
With no `?server=` the page connects to the host it was loaded from, at `/control` and `/frames`.

The page is four static files with no build step: `index.html`, `style.css`, `app.js`, `lib.js` (`app.js` imports `./lib.js`; nothing else is fetched).
Asset paths are relative, so it works under a sub-path such as `/app/` (`/app` redirects to `/app/`).
The engine serves only those four files; the rest of `web/` (`node_modules/`, `fake-server.mjs`, `test/`, `tools/`, `test-pattern/`, `package*.json`) returns 404.
The engine reads the files from disk on each request, so after a merge the engine owner copies `web/` from `main` to the engine machine and a hard refresh shows it, with no engine restart.

### From this laptop with `?server=`

Serve the page with `npm start` and point it at an engine with `?server=`:

| Engine is | Run this first | Open |
|---|---|---|
| on this laptop (`python -m engine.server --backend fake --reader commands --width 672 --height 384 --port 8000`) | nothing | http://localhost:8787/?server=127.0.0.1:8000 |
| on a remote machine you can SSH into | `ssh -N -L 8000:127.0.0.1:8000 <user>@<engine-host>` (keep it open) | http://localhost:8787/?server=127.0.0.1:8000 |
| behind the HTTPS tunnel | nothing | http://localhost:8787/?server=https://<tunnel address> |

`?server=` takes `host:port` or a full `ws://`, `wss://` or `https://` URL.
This is how to check a page change against the live engine before it is merged (PR #104 was checked this way).

## Demo day

Both pages are served by the engine: this one at `/app/` and Lucid Studio at `/studio/`.
On 2026-11-14 the lead named Lucid Studio the judges' page "for now" with `web/` as the tested fallback, and later approved the big-screen redesign of `web/` (PR #104); which page the judges see is the lead's call.

1. Open `https://<tunnel address>/app/` in Chrome and allow the microphone the first time Chrome asks.
2. Press Full screen.
3. Speak with Wispr Flow, push-to-talk or the Mic, or type.
4. Fix mistakes with Undo, the × on a chip, or Reset.
5. Read the screen: the chips are the changes the server holds, the words line says what was heard and understood, and the badge is the delay.

Measured on the live engine through the tunnel after PR #104: "make it winter" 496 ms and push-to-talk "make it night" 429 ms, command to first frame (Ani's board post at 23:59 UTC).

If the connection drops, the video shows "Connection lost, reconnecting…" and the page reconnects by itself.
Commands typed meanwhile are not sent (an error says so); say them again once it shows "Connected".
"Video paused" means the server is connected but not sending frames: press Reset, and if it stays paused, ask the engine owner or reload the page.
If the tunnel address changed, a new one is posted on the board.

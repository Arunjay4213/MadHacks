# web/: the demo page and its inputs

Owner `@anirudhaugile` (Ani), issue #6. See `PLAN.md` here for the DOM ids, behaviour rules and status.

```sh
cd web
npm install
npm test        # unit tests (lib.js) + fake-server integration tests
npm start       # MOCK fake server on http://127.0.0.1:8787 (PORT to change)
```

Open http://localhost:8787 in Chrome (the live mic needs Chrome's speech recognition, and a secure
context: localhost works, a plain `http://<ip>` address does not).
Push-to-talk: hold the Hold-to-talk button or the backtick key.

The fake server is a mock. It loops an ffmpeg test pattern (the committed frames in `test-pattern/` if ffmpeg is missing;
regenerate them with `python tools/make-test-pattern.py`), spots a dozen keywords, and says `fake: true` in every
message so the page shows the mock banner. Its wire format follows `contracts/README.md`.

## Connecting to the engine

Always serve the page from this laptop (`npm start`, then http://localhost:8787) so the live mic keeps working:
Chrome only allows the mic on localhost or HTTPS. Point it at the engine with `?server=`:

| Engine is | Run this first | Open |
|---|---|---|
| on this laptop (`python -m engine.server --backend fake --reader commands --width 672 --height 384 --port 8000`) | nothing | http://localhost:8787/?server=127.0.0.1:8000 |
| on a remote machine you can SSH into | `ssh -N -L 8000:127.0.0.1:8000 <user>@<engine-host>` (keep it open) | http://localhost:8787/?server=127.0.0.1:8000 |
| behind an HTTPS tunnel (`cloudflared tunnel --url http://127.0.0.1:8000` on the engine box) | nothing | http://localhost:8787/?server=https://<name>.trycloudflare.com |

`?server=` takes `host:port` or a full `ws://`, `wss://` or `https://` URL. Frames are drawn at whatever size the
engine sends (672x384 for the live model), scaled up to fill the screen at their own aspect ratio.

## Serving the page from the engine (`/app/`)

The page is four static files with no build step: `index.html`, `style.css`, `app.js`, `lib.js` (app.js imports
`./lib.js`; nothing else is fetched). Asset paths are relative, so it works under a sub-path such as `/app/`
(redirect `/app` to `/app/`). With no `?server=` it connects to the host it was loaded from, at `/control` and
`/frames`. Do not serve the rest of `web/`: `node_modules/`, `fake-server.mjs`, `test/`, `tools/`, `test-pattern/`,
`package*.json`.

## Demo day / fallback

`web/` is the tested fallback demo page (the lead chose `lucid-studio/` for now). To run it:

1. **Open the page.** Either
   - on this laptop: `cd web && npm start`, then in Chrome open
     `http://localhost:8787/?server=https://<engine tunnel address>` (the address is posted on the board), or
   - with nothing to run: `https://<engine tunnel address>/app/` once the engine serves `web/` there.

   Both give the mic (localhost and https are allowed). Allow the microphone the first time Chrome asks.
2. **Full screen:** Chrome menu **View → Enter Full Screen**.
3. **Speak commands:**
   - **Wispr Flow:** hold **Ctrl**, speak, release. The sentence lands in the text box and is sent at once.
   - **Live mic:** hold the **backtick** key (`` ` ``, under Esc) or the **Hold to talk** button, speak, release.
     Words show while you speak; one press is one command, sent when you release. **Mic** keeps listening
     hands-free until clicked again.
   - **Typing** works too: press Enter, or pause 2.5 s. Text that looks unfinished (ends in a comma or "and",
     "a", "on"…) waits for Enter, so one sentence is never split into two commands.
4. **Fix mistakes:** **Undo** steps back one change; the **×** on a chip removes that change; **Reset** (or
   **Option+R** / Alt+R) goes back to the starting scene. Reset also restarts the picture, which keeps it clean
   (the live model degrades after about 30 s without one).
5. **Read the screen:** chips are the changes the server holds; the highlighted word is the keyword it matched;
   red words matched no command. The number top right is the delay from your command to the first new frame;
   `—` means nothing measured yet.

**If the connection drops:** the video shows "Connection lost, reconnecting…" and the page reconnects by itself.
Commands typed meanwhile are not sent (an error says so); say them again once it shows "Connected". "Video paused"
means the server is connected but not sending frames: press **Reset**, and if it stays paused, ask the engine owner
or reload the page. If the tunnel address changed, a new one is posted on the board.

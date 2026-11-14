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

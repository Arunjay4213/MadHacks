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

The fake server is a mock. It loops an ffmpeg test pattern (a grey pixel if ffmpeg is missing;
`brew install ffmpeg` for a moving picture), spots a dozen keywords, and says `fake: true` in every
message so the page shows the mock banner. Its wire format follows `contracts/README.md`.

To point the page at another server, such as the engine (`python -m engine.server --backend fake --port 8000`),
add `?server=host:port`, for example http://localhost:8787/?server=127.0.0.1:8000.

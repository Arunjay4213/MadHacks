# web/: the demo page and its inputs

Owner `@prateekraouw`. See `PLAN.md` here for the DOM ids, behaviour rules and the provisional transport.

```sh
cd web
npm install
npm test        # unit tests (lib.js) + fake-server integration tests
npm start       # MOCK fake server on http://127.0.0.1:8787 (PORT to change)
```

Open the page in Chrome (the live mic needs Chrome's speech recognition).
Push-to-talk: hold the PTT button or the backtick key.

The fake server is a mock. It loops an ffmpeg test pattern (a grey pixel if ffmpeg is missing),
spots a dozen keywords, and says `fake: true` in every message so the page shows the mock banner.
Its wire format is a proposal waiting on the lead, not the agreed contract.

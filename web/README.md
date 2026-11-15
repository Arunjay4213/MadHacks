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

The page is branded "Mirage", the product name (decided by the lead on 2026-11-15), with "live voice-steered video" as the subtitle. The layout is one white window on a pale blue page, everything in one centred column.

- Header: "Mirage · live voice-steered video" on the left (plus a "Mock server" badge when the server is the fake one); on the right the status capsule. Connected, it shows the delay ("0.9 s": the time from your command to the first frame of the chunk it applies to, "Live" until a command has been measured; an unmatched command is never measured), green, amber over 1.5 s, red over 3 s, with the server's stage stamps on hover. Otherwise it is amber "Reconnecting…" (or "Connecting…" before the first connection) or grey "Offline" after about 8 s of failed retries.
- The video on a pale blue mat, at the engine's 672x384 shape, never cropped or stretched. The chips sit on it at the top left, one per change the server holds, each with a × to remove it, in blue, red and amber in the order the changes first appeared.
- Banners under the header: the video notice (connecting, connection lost, paused) and the error message.
- The voice line under the video, the words in orange quotes:
  - nothing yet: "Say something";
  - speech coming in: five blue voice bars, the settled words, the not-yet-final words in grey and a blue caret. The bars follow the real mic level while push-to-talk is held (a Web Audio analyser on its own short-lived mic stream, read once per animation frame and released on release); for Wispr Flow dictation they play a gentle wave;
  - typing: just the words, no bars;
  - a final the server understood: a green tick; one it did not: an amber "!" and "· Didn't catch that". Partial words never get the tick.
  With reduced motion on, the bars are static and the level meter does not run.
- The controls row: Undo and Reset on the left, the big Hold to talk mic in the centre (its outer ring grows with your voice while held), Keyboard and Full screen on the right. Every button has a tooltip with its shortcut; the mic's says "Hold to talk (`) · Wispr: Ctrl".
- The type box, hidden until you click Keyboard (or start typing): then it fades in under the controls with a send button, and stays open after a send. Keyboard again, or Esc while it is empty, hides it. The video gives up the box's height only while it is open. Hidden, it is still focused (never `display: none`), so Wispr Flow (Ctrl) always types into it; dictation shows on the voice line and does not open the box.
- Full screen: the video fills the screen; the voice line and the controls float at the bottom on a white card; the header is hidden; the type box stays focused there too, so Wispr Flow (Ctrl) still types into it and its words show on the card, and Keyboard opens the box on the card.
- Short screens (under 560 px tall, e.g. a landscape phone): the video keeps at least 240 px and the page scrolls.
- Phone width: everything stacks, the video goes full width, the chips scroll sideways and the page may scroll.

## Controls and keys

| Control | What it does |
|---|---|
| Text box, Enter | Sends the sentence. |
| Typing pause of 2.5 s | Sends the sentence, unless it looks unfinished (ends in a comma, "and", "a", "on", "make it" and similar); then it waits for Enter. Endings like "start over", "undo it", "slowly" and "more" send normally (PR #114). |
| Wispr Flow: hold Ctrl, speak, release | The sentence lands in the text box and is sent at once. Dictation that arrives in quick bursts shows the voice wave until it is sent. |
| Hold to talk, or hold the backtick key (`` ` ``, under Esc) | Push-to-talk: words show while you speak, and one press is one command. It is sent once the words have not changed for 200 ms (see below), or on release, whichever comes first. A quick second press records at once, and commands reach the server in the order you pressed (PR #114). |
| Keyboard, right group | Shows or hides the type box (`aria-expanded`). Typing printable characters also opens it. The separate hands-free browser-mic toggle was removed on purpose (owner's decision): Wispr Flow is Ctrl, hold-to-talk is the big mic or backtick. |
| Undo | Steps back one change. |
| × on a chip | Removes that change only. |
| Reset, or Option+R / Alt+R | Goes back to the starting scene and starts a new video. |
| Full screen (right group; Esc leaves) | Full screen for the demo; use Chrome's View, Enter Full Screen if the browser blocks it. |

Speech while disconnected is dropped, never replayed.

### Spoken commands finish after 200 ms of unchanged words (issue #188)

Chrome waits about 0.5 to 1 s of silence before it marks speech final, and the engine acts only on finals.
So push-to-talk (not the text box or Wispr Flow) sends the words as the command once they have not changed for 200 ms (`SPEECH_STABLE_MS` in `lib.js`).
Words that end like an unfinished sentence ("make it", "... and", "add a", a comma) wait, the same rule as typing (`looksUnfinished`).
When Chrome's own final arrives later it is dropped if it has the same words (case and punctuation aside).
If it adds words ("make it winter" then "make it winter at night"), only the new words are sent ("at night").
If Chrome rewrote the sentence, the new sentence is sent as its own command.
Press order and the no-sending-while-disconnected rule are unchanged.
The logic is `makeSpeechEndpointer` in `lib.js`, a state machine with the states `LISTENING`, `STABLE_SENT` and `CLOSED`.

Known risk: a speaker who pauses mid-sentence for longer than 200 ms after a complete-looking phrase ("make it winter ... at night") gets two commands, and the second ("at night") may not be understood on its own.

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

Both pages are served by the engine: this one at `/app/` and the studio page (`lucid-studio/`, also branded Mirage) at `/studio/`.
On 2026-11-14 the lead named the studio page the judges' page "for now" with `web/` as the tested fallback, and later approved the big-screen redesign of `web/` (PR #104); which page the judges see is the lead's call.

1. Open `https://<tunnel address>/app/` in Chrome and allow the microphone the first time Chrome asks.
2. Press Full screen.
3. Speak with Wispr Flow (Ctrl) or push-to-talk, or type (Keyboard).
4. Fix mistakes with Undo, the × on a chip, or Reset.
5. Read the screen: the chips are the changes the server holds, the voice line says what was heard and understood (tick or "Didn't catch that"), and the status capsule is the delay.

Measured on the live engine through the tunnel after PR #104: "make it winter" 496 ms and push-to-talk "make it night" 429 ms, command to first frame (Ani's board post at 23:59 UTC).

If the connection drops, the video shows "Connection lost, reconnecting…" and the page reconnects by itself.
Commands typed meanwhile are not sent (an error says so); say them again once it shows "Connected".
"Video paused" means the server is connected but not sending frames: press Reset, and if it stays paused, ask the engine owner or reload the page.
If the tunnel address changed, a new one is posted on the board.

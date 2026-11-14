# engine

The real-time engine: it generates video chunk after chunk, streams the frames to the browser, takes commands on a control socket, steers the model through the steering/ package, and measures the delay from a command to its first frame.

## Run it with the fake backend (no GPU)

From the repo root, with Python 3.12:

```
uv venv -p 3.12 .venv-engine
VIRTUAL_ENV=.venv-engine uv pip install -r engine/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv-engine/bin/python -m engine.server --backend fake --port 8000
```

Open http://127.0.0.1:8000/ to see the stream, type a prompt and press Enter.
With the fake backend the colours of the picture come from the conditioning tensor, so a new prompt shows as a smooth colour change over about four chunks.

End-to-end check against the running server (exits non-zero on failure):

```
.venv-engine/bin/python -m engine.tools.e2e_check --url http://127.0.0.1:8000
```

Unit tests:

```
.venv-engine/bin/python -m pytest engine/tests -q
```

## Command line

```
python -m engine.server --backend fake|causal_forcing --reader standin|commands --host 127.0.0.1 --port 8000 --width 832 --height 480 \
    --steps 4 --base-prompt "..." --jpeg-quality 80 --precompute-file prompts.txt \
    --device cuda:0 --decode-device cuda:0 --backend-opt key=value --lead-margin 1.3 --log-level info
```

- `--reader` picks the command reader: `standin` (default, `engine/standin_reader.py`) or `commands` (`commands.service.CommandService`, imported only when chosen).
- The starting prompt is the reader's `committed_prompt` when it has one, else `--base-prompt`.
- `--precompute-file` is a text file with one prompt per line; each is encoded at startup, so a known prompt needs no encoding time.
- `--backend-opt key=value` is passed to the backend's `load()` and may be repeated; numbers and true/false are converted.
- The fake backend's options are `gen_ms` (170), `decode_ms` (60), `encode_ms` (80), `frames_per_chunk` (4), `first_chunk_frames` (1) and `fps` (16).
- `--lead-margin` sets how much video generation keeps ready ahead of what is on screen (see "Pipeline").
- SIGINT or SIGTERM shuts down cleanly and moves the engine to STOPPED.

## Endpoints

All on one origin; the message shapes are in `contracts/README.md`.

| Endpoint | What it does |
|---|---|
| `GET /` | The engine's own test viewer (`engine/static/index.html`); the demo page is `web/`. |
| websocket `/frames` | Binary frames out: 4-byte big-endian chunk number, 2-byte big-endian frame index in that chunk, JPEG bytes. |
| websocket `/control` | Contract 1 in (JSON text); contract 4 out, one message per chunk; `{"error": "..."}` replies to a bad message. |
| `POST /target` | Body is a contract 2 message; it skips the command reader and goes unchanged to the encoder and `steering.Controller.handle`. |
| `POST /reset` | Same as a `{"type":"tap","action":"reset"}` on `/control`: goes through the reader, then starts a new video; returns when done. |
| `GET /stats` | JSON: lifecycle state, last generated and shown chunk, last and average generation, decode and JPEG ms, measured fps, queue depths, encoder cache, clients. |
| `GET /healthz` | 200 with the state, or 503 when FAILED or STOPPED. |

## Backend interface

A backend is `engine/backends/<name>.py`, loaded by name with `--backend`; the interface is in `engine/backends/__init__.py`.
One `generate_chunk` call is one chunk in every contract, and it may produce any number of frames.

## Pipeline

- Generation thread: `cond = steerer.get_cond()`, `latents = backend.generate_chunk(cond)`, then put `(latents, chunk info)` on the decode queue.
  The frame cache is never rebuilt on a command.
- Decode thread: takes from the decode queue, calls `backend.decode_chunk`, JPEG-encodes the frames and puts them on the playout queue.
  It shares nothing else with the rest of the engine, so it can later move to another GPU or process.
- Streamer thread: sends frames at `backend.fps` to every `/frames` client and, on the first frame of each chunk, sends one contract 4 message to every `/control` client.
  If a frame is not ready in time the stream waits and the log says "stream stalled".
- Each websocket client has its own small queue; a slow client loses frames from its own queue and never slows the engine.
- Both queues are bounded; when decode falls behind, generation blocks and the log says so.
- Lead gate: before each chunk, generation waits until the frames already made but not yet sent are no more than (generation + decode + JPEG time) x fps x `--lead-margin`.
  This keeps the stream fed while keeping as little video queued as possible, because every queued frame adds to the delay of a command.
- Timing log: one line per chunk with generation, decode and JPEG ms and the decode queue depth, and a summary line every 20 chunks.

## Commands

contract 1 on `/control` -> reader worker thread: `reader.handle` -> `steer` (contract 2) -> encoder service -> steer worker thread: `steering.Controller.handle` -> `steerer.set_target` -> the next `get_cond`.

- The reader API, agreed with the `commands/` owner, is in `engine/reader.py`:
  `handle(msg) -> {"steer": contract 2 or None, "state": {...}}`, `state_fields()`, `likely_prompts()` and `committed_prompt`.
- Every contract 1 message (`partial`, `final`, `tap`) goes to `reader.handle`, one at a time in arrival order on its own worker thread, because it may block for about 0.4 s.
  If it raises ValueError, the client gets `{"error": ...}` on `/control`.
- The stand-in reader takes a `final` text as the whole new prompt with a 4-chunk ramp, gives no steer for `partial` and `tap`, and keeps an empty changes list.
- The `steer` dict is passed unchanged to `steering.Controller.handle`, including optional fields such as `strength` and `without_prompt`.
  The engine first encodes `prompt` and `without_prompt` in its encoder service, so `handle` never waits for encoding.
- A reset tap also resets the engine: RESETTING, `backend.reset()` to clear the frame cache, then a new steerer and controller seeded directly with the embedding of the reset's `steer` prompt (no ramp), or the starting prompt if `steer` is None.
- The encoder service runs on its own thread with a priority queue, caches embeddings and token ids by exact prompt text (LRU, 512 entries), shares one encoding between identical requests, and logs cache hits, misses and encode ms.
- The reader's `likely_prompts()` are encoded at startup, and after every command they are queued again at background priority, behind any command's encoding.

## Lifecycle states

| State | Meaning | Next |
|---|---|---|
| LOADING | `backend.load()` is running | WARMING, STOPPED, FAILED |
| WARMING | encoding the base prompt and the precompute file | GENERATING, STOPPED, FAILED |
| GENERATING | producing chunks | RESETTING, STOPPED, FAILED |
| RESETTING | between chunks: wait for decode to finish, `backend.reset()`, new steerer and controller on the new video's prompt | GENERATING, STOPPED, FAILED |
| FAILED | a thread raised; `/stats` shows the error | STOPPED |
| STOPPED | shut down | none |

Every change goes through `LifecycleMachine.transition`, which logs it and rejects any change not in this table.
Chunk numbers keep rising across a reset.

## Command stages and delay stamps

Each command has one record whose stage is one of HEARD, DECIDED, EMBEDDED, SCHEDULED, GENERATED, SHOWN, or the end states DROPPED (no operation, an error, or a reset), SUPERSEDED (a newer command's target was set first) and UNCHANGED (the controller found the target already applied).

Stamps use the monotonic clock and are reported as integer ms after `heard`:

| Stamp | Taken when |
|---|---|
| heard | the server received the contract 1 message (or the `POST /target` request) |
| decided | the command reader returned an operation (contract 2) |
| embedded | the embedding was ready and `steering.Controller.handle` applied it, which calls `steerer.set_target` (for a reset with a prompt: the new steerer was seeded with it) |
| generated | generation finished for the first chunk whose `get_cond` came after that `set_target` |
| shown | the first frame of that chunk was handed to the `/frames` clients |

`set_target` and `get_cond` are ordered under one lock in the engine, so "the first `get_cond` after `set_target`" is exact.
That chunk's number is sent as `applies_from_chunk`.

## Contract 4 fields

The engine sends one message per chunk when the chunk's first frame is sent.

- `changes`, `heard`, `keyword`, `unmatched`: from `reader.state_fields()`, called for every message.
- `chunk`: that chunk's number.
- `progress`: `steerer.progress` right after the `get_cond` for that chunk.
- `stamps`, `t_client`, `applies_from_chunk`: from the most recent command, which is the newest one the reader gave a `steer` (or a `POST /target`).
  A message with no steer does not replace it.
- Stamps appear as they become known, so the page sees `heard` and `decided` first and `shown` last; `{}` before any command.

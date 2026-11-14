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

The check reads the server's command reader from `/stats` (or take `--reader standin|commands`) and sends commands that reader understands.
With `standin` it sends a raw new prompt, then the starting prompt again as a cache hit.
With `commands` it sends "new scene of a lighthouse on a cliff <time>", waits until the reader's likely prompts are encoded, then sends "make it winter" and "go back", and checks that both are cache hits and that `changes` shows the winter chip and then none.

Unit tests:

```
.venv-engine/bin/python -m pytest engine/tests -q
```

## The live engine on the GPU machine

The live engine runs on `madhacks-engine` (one NVIDIA L40S) in two tmux sessions: `engine` (the server) and `tunnel` (a Cloudflare quick tunnel).
The model is Causal Forcing++ framewise 2-step at 672x384 with the TAEHV decoder; see `engine/MODEL_NOTES.md` for the model, licenses and timing table.
One chunk is one latent frame, which is 4 video frames (0.25 s at 16 fps), so real time needs generation plus decode under 250 ms per chunk.
Measured while serving: generation about 210 ms and decode about 21 ms per chunk, 16.0 fps out, no stalls.
At 832x480 one chunk takes about 443 ms on this GPU, so that size is not real time on one L40S.

Start the server (first time on a new machine: `bash engine/setup_gpu.sh`, then `uv pip install -r engine/requirements.txt -r steering/requirements.txt` into `~/engine-venv`):

```
tmux new-session -d -s engine "cd ~/madhacks && source ~/engine-venv/bin/activate && python -m engine.server --backend causal_forcing --reader commands --width 672 --height 384 --backend-opt vae=taehv --host 127.0.0.1 --port 8000 2>&1 | tee -a ~/runs/engine-server.log"
```

Loading takes about 70 s; `curl -s localhost:8000/stats | jq .state` prints `GENERATING` when it is up.
The repo on the machine is a copy made with `rsync -a --delete --exclude .git ./ madhacks-engine:madhacks/` from a laptop clone, because the machine has no GitHub credentials.

Start the tunnel, which gives every teammate an https address without an open port:

```
tmux new-session -d -s tunnel "cloudflared tunnel --url http://127.0.0.1:8000 2>&1 | tee ~/runs/tunnel.log"
rg -o "https://[a-z0-9-]+\.trycloudflare\.com" ~/runs/tunnel.log | tail -1
```

Open the printed address for the engine test viewer, or give it to the web page as `?server=https://<address>`.
The address changes each time the tunnel restarts; restarting only the server keeps the address.
The engine has no login, so anyone who has the address can watch and send commands; keep it inside the team.
It is one shared video: every viewer sees every command and reset, so run automated checks against a local `--backend fake` server.
Without the tunnel, use an SSH port forward from a machine that has SSH access: `ssh -N -L 8000:127.0.0.1:8000 madhacks-engine`, then open http://127.0.0.1:8000/.

### Known problems on the real model (2026-11-14)

The picture degrades when a video runs long without a reset: it is clean for about 20 s, over-contrasted by about 30 s and broken by about 60 s.
Press Reset (or send the reset tap) to start a clean video.
A new prompt is read by the model but the frame history holds the old scene, so a command such as "make it winter" does not visibly change the scene yet.
Both are being worked on in the backend (re-anchoring the frame history at a command and at intervals).

## Command line

```
python -m engine.server --backend fake|causal_forcing --reader standin|commands --host 127.0.0.1 --port 8000 --width 832 --height 480 \
    --steps 0 --base-prompt "..." --jpeg-quality 80 --precompute-file prompts.txt \
    --device cuda:0 --decode-device cuda:0 --backend-opt key=value --lead-margin 1.3 --reanchor off --log-level info
```

- `--steps 0` (the default) runs the denoising schedule the backend's checkpoint was distilled for; the real model is distilled for 2 steps, so any other number changes the schedule.
- `--reader` picks the command reader: `standin` (default, `engine/standin_reader.py`) or `commands` (`commands.service.CommandService`, imported only when chosen).
- The starting prompt is the reader's `committed_prompt` when it has one, else `--base-prompt`.
- `--precompute-file` is a text file with one prompt per line; each is encoded at startup, so a known prompt needs no encoding time.
- `--backend-opt key=value` is passed to the backend's `load()` and may be repeated; numbers and true/false are converted.
- The fake backend's options are `gen_ms` (170), `decode_ms` (60), `encode_ms` (80), `frames_per_chunk` (4), `first_chunk_frames` (1) and `fps` (16).
- `--lead-margin` sets how much video generation keeps ready ahead of what is on screen (see "Pipeline").
- `--reanchor off|on` (default `off`) turns on the steering re-anchor policy (see "Re-anchor").
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
| `GET /stats` | JSON: lifecycle state, backend, reader, last generated and shown chunk, last and average generation, decode and JPEG ms, measured fps, queue depths, re-anchor state and last anchor, encoder cache and pending encodes, clients. |
| `GET /healthz` | 200 with the state, or 503 when FAILED or STOPPED. |
| `GET /app/` | The demo page from `web/` (see "The demo page at /app/"). |

## The demo page at /app/

The server also serves the demo page from the repo's `web/` folder at `/app/`, so the page and the engine share one origin.
On the https tunnel, `https://<address>/app/` gives the live mic with no extra setup, because the page connects to `/control` and `/frames` on the host it was loaded from when there is no `?server=`.
Only four files are served, from an explicit allow-list: `index.html` (at `/app/`), `style.css`, `app.js` and `lib.js`.
Every other path under `/app/` returns 404, including the rest of `web/` (`node_modules/`, `fake-server.mjs`, `test/`, `tools/`, `test-pattern/`, `package*.json`) and any path traversal.
`/app` redirects to `/app/` and keeps the query string.
The files are read from disk on each request with `Cache-Control: no-cache`, so an rsync of `web/` takes effect without a restart.
If `web/` is missing, `/app/` returns a 404 that says so, and the rest of the server works as usual.

## Lucid Studio at /studio/

The server serves the Lucid Studio build (a Vite app) read-only at `/studio/`, from `--studio-dir` (default `<repo>/lucid-studio/dist`).
Its asset names are hashed, so any regular file inside that folder is served, not an allow-list.
Dotfiles, missing files, directories, path traversal and symlinks that point outside the folder all return 404.
`/studio` redirects to `/studio/`, which serves `index.html`.
`index.html` and other files are sent with `Cache-Control: no-cache`; hashed files under `assets/` may be cached for a year.
If the folder is missing, `/studio/` returns "Lucid Studio build not found at <path>" with a 404, and the server still starts.
The Vite build must use `base: './'` or `base: '/studio/'` in its config, or its asset links point at `/assets/` and fail under `/studio/`.

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

## Re-anchor

`--reanchor on` puts `steering.reanchor.ReanchorPolicy` (see `steering/README.md`, "Re-anchor policy") into the generation loop.
With `--reanchor off`, the default, no policy exists and every chunk is exactly `backend.generate_chunk(cond)`.

- The policy is built next to the Controller in `Engine._new_steerer`.
- `on_target(msg, result)` runs right after every `Controller.handle`, in `Engine._apply_target`, under the CommandBook lock.
- `next_chunk(progress, is_moving)` runs once per chunk in `Engine._take_cond`, under the same lock and right after `get_cond`.
- `on_reset()` runs on every reset, right after `Controller.reset`.
- `Engine._generate(cond, anchor)` is the one backend call: with an anchor it calls `backend.reanchor(strength)` and then `backend.generate_chunk(cond)`.
- The state is an enum, `ReanchorState`: `off`, `on`, or `unsupported` when `--reanchor on` is given but the backend has no `reanchor(strength)` (logged once at load, then the engine runs as with `off`).
- The real backend (`causal_forcing`) cannot re-anchor yet, so it reports `unsupported`; the diffusion side is issue #45.
- The fake backend records each `reanchor` call and draws a white border and "re-anchor 1.00" on the next chunk's frames.
- `/stats` has `reanchor`: `state`, `count`, `last_strength`, `last_reason` (`steer` or `refresh`), `last_chunk`, and the policy's `chunks_since_anchor`.

## Commands

contract 1 on `/control` -> reader worker thread: `reader.handle` -> `steer` (contract 2) -> encoder service -> steer worker thread: `steering.Controller.handle` -> `steerer.set_target` -> the next `get_cond`.

- The reader API, agreed with the `commands/` owner, is in `engine/reader.py`:
  `handle(msg) -> {"steer": contract 2 or None, "state": {...}}`, `state_fields()`, `likely_prompts()` and `committed_prompt`.
- Every contract 1 message (`partial`, `final`, `tap`) goes to `reader.handle`, one at a time in arrival order on its own worker thread, because it may block for about 0.4 s.
  If it raises ValueError, the client gets `{"error": ...}` on `/control`.
- The stand-in reader takes a `final` text as the whole new prompt with a 4-chunk ramp, gives no steer for `partial` and `tap`, and keeps an empty changes list.
- The `steer` dict is passed unchanged to `steering.Controller.handle`, including optional fields such as `strength` and `without_prompt`.
  The engine first encodes `prompt` and `without_prompt` in its encoder service, so `handle` never waits for encoding.
- A reset tap also resets the engine: RESETTING, `backend.reset()` to clear the frame cache, then `steering.Controller.reset(prompt, strength, without_prompt)` with the reset's `steer` (or the starting prompt if `steer` is None).
  That jumps the steerer to the new target with no ramp, honours a strength pair, and keeps the controller's prompt cache.
- The encoder service runs on its own thread with a priority queue, caches embeddings and token ids by exact prompt text (LRU, 512 entries), shares one encoding between identical requests, and logs cache hits, misses and encode ms.
- The reader's `likely_prompts()` are encoded at startup, and after every command they are queued again at background priority, behind any command's encoding.

## Lifecycle states

| State | Meaning | Next |
|---|---|---|
| LOADING | `backend.load()` is running | WARMING, STOPPED, FAILED |
| WARMING | encoding the base prompt and the precompute file | GENERATING, STOPPED, FAILED |
| GENERATING | producing chunks | RESETTING, STOPPED, FAILED |
| RESETTING | between chunks: wait for decode to finish, `backend.reset()`, `Controller.reset` on the new video's prompt | GENERATING, STOPPED, FAILED |
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

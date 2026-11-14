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

## Demo harness

`engine/tools/demo_run.py` plays a timed script of commands against an engine and records what comes back, so a demo can be repeated and two versions of the engine can be compared frame by frame.
It needs `ffmpeg` on the PATH.

```
.venv-engine/bin/python -m engine.server --backend fake --reader commands --reanchor on --port 8769
.venv-engine/bin/python -m engine.tools.demo_run --url http://127.0.0.1:8769 --script engine/tools/demo_scripts/draft.json --out runs/baseline --label baseline
.venv-engine/bin/python -m engine.tools.demo_run --compare runs/baseline runs/new --out runs/compare
```

A script is a JSON file in `engine/tools/demo_scripts/`: a `name`, an `end` time in seconds, and `steps` of `{"at": 4, "say": "make it winter"}` or `{"at": 0, "reset": true}`.
A `say` step is sent on `/control` as a contract 1 `final` message with a fresh `t_client`, the same message the demo page sends for a typed command.
A `reset` step is a contract 1 reset tap with a `t_client` (optional on a tap), so the engine tracks it like a command and contract 4 tells the first chunk of the new video.
The harness sends nothing else; it only reads `GET /stats` besides the two sockets.

The output folder gets:

- `video.mp4`: every frame received between script second 0 and `end`, at the engine fps from `/stats` (libx264, CRF 20). A stall shortens the video instead of freezing it, because frames are placed by count, not by arrival time.
- `sheet.jpg`: one frame every 2 s, each labelled with the time, the chunk, what the frame shows ("before the run" until the first step reaches the screen, "start" after a reset, else the command), and a step that was sent but is not on screen yet.
- `frames/`: those sheet frames as the JPEGs the engine sent.
- `timings.json`: per command its `t_client`, the contract 4 stamps, `applies_from_chunk`, `unmatched`, the first frame received with chunk at least `applies_from_chunk` after the send, and `client_delay_ms` (that frame's receive time minus the send time); also `/stats` at start and end (`gen_ms`, `decode_ms`, `fps`, `stalls`, `reanchor`, `recorder`).
- `summary.txt`: the same as a short table.

`--compare DIR1 DIR2 ... --out OUT` writes `OUT/compare.jpg`: one row per run at the same time marks (every 4 s; `--every 2` for every sheet frame), labelled with each run's `--label` and its delay range.

It works on localhost and through the tunnel (`--url https://<address>`, which uses wss).
A run against a URL that is not localhost changes the shared live video for every viewer, so the harness prints a warning and waits 5 s before it starts (`--yes` skips the wait).

## The live engine on the GPU machine

The live engine runs on `madhacks-engine` (one NVIDIA L40S) in two tmux sessions: `engine` (the server) and `tunnel` (a Cloudflare quick tunnel).
The model is Causal Forcing++ framewise 2-step at 672x384 with the TAEHV decoder; see `engine/MODEL_NOTES.md` for the model, licenses and timing table.
One chunk is one latent frame, which is 4 video frames (0.25 s at 16 fps), so real time needs generation plus decode under 250 ms per chunk.
Measured while serving: generation about 210 ms and decode about 21 ms per chunk, 16.0 fps out, no stalls.
At 832x480 one chunk takes about 443 ms on this GPU, so that size is not real time on one L40S.

Start the server (first time on a new machine: `bash engine/setup_gpu.sh`, then `uv pip install -r engine/requirements.txt -r steering/requirements.txt` into `~/engine-venv`):

```
tmux new-session -d -s engine "cd ~/madhacks && source ~/engine-venv/bin/activate && python -m engine.server --backend causal_forcing --reader commands --width 672 --height 384 --backend-opt vae=taehv --reanchor on --reanchor-opt refresh_every_chunks=none --host 127.0.0.1 --port 8000 --record-dir /home/ubuntu/recordings 2>&1 | tee -a ~/runs/engine-server.log"
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
With the systemd units (next section) the server and the tunnel restart on their own; the tmux commands above are the manual way.
Without the tunnel, use an SSH port forward from a machine that has SSH access: `ssh -N -L 8000:127.0.0.1:8000 madhacks-engine`, then open http://127.0.0.1:8000/.

### Process supervision (systemd)

`engine/deploy/` holds two systemd units and a helper script:

- `madhacks-engine.service` runs the server with the live command line above, with `Restart=always`, `RestartSec=2` and no start limit.
- `madhacks-tunnel.service` runs `cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000` with `Restart=always`.
- `tunnel-address.sh` prints the current trycloudflare address from the tunnel unit's journal (only the current run, so never an old address); it waits up to 30 s for it.

They are system units with `User=ubuntu`, not user units.
A system unit starts at boot and keeps running with nobody logged in, with no `loginctl enable-linger` step, and its log is in the system journal, which the `ubuntu` user (group `adm`) can read.
Install them once (needs sudo), after stopping the tmux sessions `engine` and `tunnel` so port 8000 is free:

```
sudo cp ~/madhacks/engine/deploy/madhacks-engine.service ~/madhacks/engine/deploy/madhacks-tunnel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now madhacks-engine madhacks-tunnel
bash ~/madhacks/engine/deploy/tunnel-address.sh
journalctl -u madhacks-engine -f     # the server log
```

The tunnel unit expects `cloudflared` in `/usr/local/bin` or `/usr/bin`; if it is elsewhere, change `ExecStart` to its full path.
A quick tunnel gets a NEW address every time cloudflared restarts (a crash, `systemctl restart madhacks-tunnel`, or a reboot), so after that run `tunnel-address.sh` and post the new address again.
Restarting only the engine keeps the tunnel and its address.
The engine unit uses `KillMode=mixed`: `systemctl stop` sends SIGTERM to the server only, which finalizes the open recording before it exits.
With the default mode ffmpeg would get SIGTERM at the same moment and cut the file short.

## Crash recovery

A crash is handled in two layers, because loading the model takes 70 to 90 s and a crash should cost under 10 s.

1. In-process recovery.
   If the generation or decode thread raises, the engine logs the traceback and the generation thread moves GENERATING (or RESETTING) -> RECOVERING.
   It drops every chunk waiting for decode and every frame waiting to be sent, calls `backend.reset()`, and restarts the steerer on the starting prompt through `Controller.reset`, the same as a reset to the starting prompt.
   The model stays loaded.
   Then RECOVERING -> GENERATING, and a new video (and a new recording file) starts.
   The log says `recovery: video resumed N ms after the crash`, and `/stats` shows `recovery.count` and `recovery.last.resumed_ms`.
   With the fake backend the video resumes about 230 ms after the crash.
2. Process restart.
   If 3 crashes happen within 60 s (counting a crash during recovery itself), the engine moves to FAILED, finalizes the recording and exits with code 70.
   An error before generation starts (for example while loading the model) also ends in FAILED and exit code 70.
   systemd starts a fresh process 2 s later, which then loads the model again (70 to 90 s with the real model, about 3.5 s with the fake one).
   A crash that kills the process outright (`kill -9`, out of memory) goes straight to this layer.

The fake backend can inject crashes for testing: `--backend-opt fail_on_chunk=N` raises in `generate_chunk` from chunk N on, `fail_count` times in total (default 1); `--backend-opt fail_in=decode` raises in `decode_chunk` instead.
For example, `fail_on_chunk=40` shows one in-process recovery, and `fail_on_chunk=40 fail_count=3` shows the FAILED exit and, under systemd, the restart.

## Recorder

Every video the engine streams is saved as an mp4 on the engine machine, in `--record-dir` (default `~/recordings`; `--record-dir none` turns it off).
A video, and so a file, runs from server start, a reset or a recovery to the next reset, recovery or stop.
Files are named by their UTC start time and the session number in this process, for example `20261114T220704Z-s001.mp4`.

- The recorder takes the same JPEG frames the streamer sends, at `backend.fps`.
  The streamer only puts each frame on a bounded queue (64 frames); when the queue is full the frame is dropped and counted, so recording never slows streaming or generation.
- The recorder thread pipes the JPEGs into an `ffmpeg` subprocess (mjpeg in, libx264 `veryfast`, CRF 23, yuv420p, at `nice` 10, 2 threads), so encoding runs in another process.
  `engine/setup_gpu.sh` installs ffmpeg; if it is missing, the server logs one warning and the recorder stays DISABLED.
- On a reset and on a clean stop the file is finalized (ffmpeg's input is closed and it finishes the file).
- Files are fragmented mp4 with a fragment every 2 s, written to disk as each fragment finishes, so a file whose process was killed with `kill -9` still plays up to its last 2 s.
- When the folder holds more than `--record-max-gb` (default 20), the oldest `.mp4` files are deleted before a new file starts.
- If ffmpeg fails, the recorder logs it, moves to ERROR and drops that video's frames; the next video tries a new file.

| Recorder state | Meaning | Next |
|---|---|---|
| DISABLED | no record dir, or ffmpeg missing; frames are ignored | none |
| IDLE | running, no file open | RECORDING, ERROR, STOPPED |
| RECORDING | writing one video into one file | FINALIZING, ERROR |
| FINALIZING | ffmpeg's input is closed; waiting for it to finish the file | IDLE, ERROR, STOPPED |
| ERROR | ffmpeg failed for the current video | RECORDING (next video), ERROR, STOPPED |
| STOPPED | stopped; the last file is finalized | none |

`/stats` reports it under `recorder`: `state`, `file`, `frames_written` (in the current file), `frames_dropped` (whole run), `sessions` and `error`.
Check a recording with `ffprobe -v error -show_entries format=duration:stream=width,height,avg_frame_rate -of compact=p=0 FILE`.

### Known problems on the real model (2026-11-14)

The picture degrades when a video runs long without a reset: it is clean for about 20 s, over-contrasted by about 30 s and broken by about 60 s.
Press Reset (or send the reset tap) to start a clean video.
Steering works with `--reanchor on`: Gate 2 passed live after PR #57, with full winter about 2 s after "make it winter" (`engine/samples/gate2_live_make_it_winter_pass.jpg`).
The policy's periodic refresh (every 60 chunks at strength 0.25) visibly breaks the picture on the refresh chunk (`engine/samples/refresh_breaks_picture.jpg`).
The live engine therefore runs with the refresh off: `--reanchor-opt refresh_every_chunks=none`.

## Command line

```
python -m engine.server --backend fake|causal_forcing --reader standin|commands --host 127.0.0.1 --port 8000 --width 832 --height 480 \
    --steps 0 --base-prompt "..." --jpeg-quality 80 --precompute-file prompts.txt \
    --device cuda:0 --decode-device cuda:0 --backend-opt key=value --lead-margin 1.3 --reanchor off --reanchor-opt key=value --log-level info \
    --record-dir ~/recordings --record-max-gb 20
```

- `--steps 0` (the default) runs the denoising schedule the backend's checkpoint was distilled for; the real model is distilled for 2 steps, so any other number changes the schedule.
- `--reader` picks the command reader: `standin` (default, `engine/standin_reader.py`) or `commands` (`commands.service.CommandService`, imported only when chosen).
- The starting prompt is the reader's `committed_prompt` when it has one, else `--base-prompt`.
- `--precompute-file` is a text file with one prompt per line; each is encoded at startup, so a known prompt needs no encoding time.
- `--backend-opt key=value` is passed to the backend's `load()` and may be repeated; numbers and true/false are converted.
- The fake backend's options are `gen_ms` (170), `decode_ms` (60), `encode_ms` (80), `frames_per_chunk` (4), `first_chunk_frames` (1) and `fps` (16), plus `fail_on_chunk`, `fail_count` and `fail_in` to inject crashes (see "Crash recovery").
- `shared_device=true` makes the fake's encode, generation and decode share one simulated GPU by fair time sharing (two jobs at once each run at half speed), as umT5-xxl and the generator share one GPU on the live machine.
  To time the live machine on a laptop, use `--backend-opt gen_ms=213 --backend-opt decode_ms=21 --backend-opt encode_ms=53 --backend-opt shared_device=true`.
- `--lead-margin` sets how much video generation keeps ready ahead of what is on screen (see "Pipeline").
- `--reanchor off|on` (default `off`) turns on the steering re-anchor policy (see "Re-anchor").
- `--reanchor-opt key=value` sets one field of the policy's `ReanchorConfig` and may be repeated (see "Re-anchor").
- `--record-dir` and `--record-max-gb` control the recorder (see "Recorder"); `--record-dir none` turns it off.
- SIGINT or SIGTERM shuts down cleanly, finalizes the open recording and moves the engine to STOPPED.
- The process exits with code 70 when the engine reaches FAILED, so systemd restarts it.

## Endpoints

All on one origin; the message shapes are in `contracts/README.md`.

| Endpoint | What it does |
|---|---|
| `GET /` | The engine's own test viewer (`engine/static/index.html`); the demo page is `web/`. |
| websocket `/frames` | Binary frames out: 4-byte big-endian chunk number, 2-byte big-endian frame index in that chunk, JPEG bytes. |
| websocket `/control` | Contract 1 in (JSON text); contract 4 out, one message per chunk; `{"error": "..."}` replies to a bad message. |
| `POST /target` | Body is a contract 2 message; it skips the command reader and goes unchanged to the encoder and `steering.Controller.handle`. |
| `POST /reset` | Same as a `{"type":"tap","action":"reset"}` on `/control`: goes through the reader, then starts a new video; returns when done. |
| `GET /stats` | JSON: lifecycle state, backend, reader, last generated and shown chunk, last and average generation, decode and JPEG ms, measured fps, queue depths, re-anchor state and last anchor, encoder cache and pending encodes, current video number, recovery count and last recovery time, recorder, clients. |
| `GET /healthz` | 200 with the state (RECOVERING included), or 503 when FAILED or STOPPED. |
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
- The real backend (`causal_forcing`) has `reanchor(strength)` since PR #52, so it reports `on`.
  It cuts the frame cache and starts the next chunk from the last picture noised by an amount set by the strength.
- The fake backend records each `reanchor` call and draws a white border and "re-anchor 1.00" on the next chunk's frames.
- A re-anchor chunk is logged when it is generated and again on its decode line (`re-anchor steer 1.00`).
- `/stats` has `reanchor`: `state`, `count`, `last_strength`, `last_reason` (`steer` or `refresh`), `last_chunk`, `config` (the options that differ from the defaults), and the policy's `chunks_since_anchor`.

### Re-anchor options

`--reanchor-opt KEY=VALUE` sets one field of `steering.reanchor.ReanchorConfig`, without a change to the steering code.
It may be repeated; the fields and their defaults are in `steering/reanchor.py`.
Each value is converted to the field's type: an int, a float, or for `refresh_every_chunks` an int or `none`.
The live engine turns the periodic refresh off:

```
--reanchor on --reanchor-opt refresh_every_chunks=none
```

Other examples: `--reanchor-opt steer_strength=0.9`, `--reanchor-opt min_gap_chunks=8`.
The options are checked when the server starts, before the backend loads.
An unknown key, a value that does not convert, or a value `ReanchorConfig` rejects makes the server exit at once with a message such as `--reanchor-opt: unknown re-anchor option 'bogus'; known options: ...`.
The options are kept with `--reanchor off` too, but have no effect there.

## Commands

contract 1 on `/control` -> reader worker thread: `reader.handle` -> `steer` (contract 2) -> encoder service -> steer worker thread: `steering.Controller.handle` -> `steerer.set_target` -> the next `get_cond`.

- The reader API, agreed with the `commands/` owner, is in `engine/reader.py`:
  `handle(msg) -> {"steer": contract 2 or None, "state": {...}}`, `state_fields()`, `likely_prompts()` and `committed_prompt`.
- Every contract 1 message (`partial`, `final`, `tap`) goes to `reader.handle`, one at a time in arrival order on its own worker thread, because it may block for up to 1.5 s (the slow reader, next section).
  If it raises ValueError, the client gets `{"error": ...}` on `/control`.
- The stand-in reader takes a `final` text as the whole new prompt with a 4-chunk ramp, gives no steer for `partial` and `tap`, and keeps an empty changes list.
- The `steer` dict is passed unchanged to `steering.Controller.handle`, including optional fields such as `strength` and `without_prompt`.
  The engine first encodes `prompt` and `without_prompt` in its encoder service, so `handle` never waits for encoding.
- A reset tap also resets the engine: RESETTING, `backend.reset()` to clear the frame cache, then `steering.Controller.reset(prompt, strength, without_prompt)` with the reset's `steer` (or the starting prompt if `steer` is None).
  That jumps the steerer to the new target with no ramp, honours a strength pair, and keeps the controller's prompt cache.
- The encoder service runs on its own thread with a priority queue, caches embeddings and token ids by exact prompt text (LRU, 512 entries), shares one encoding between identical requests, and logs cache hits, misses and encode ms.
- The reader's `likely_prompts()` are encoded at startup, and after every command they are queued again at background priority, behind any command's encoding.
  Each new set replaces the background prompts still queued from the last one, because those belong to an older scene.
- Prefetch is paced by `PrefetchPacer` (state `held`, `saving` or `ready`, shown in `/stats` as `encoder.prefetch`), because the text encoder shares the GPU with the generator.
  While a command is in flight (HEARD to SHOWN) it is `held` and no background encode starts.
  Otherwise the time generation waits at the lead gate, which is spare GPU time, is saved up; once it covers one encode (the fastest of the last 20 encodes) it is `ready` and one background encode runs.
  Urgent encodes (a command's own prompt) never wait for it.
  Without the pacing, the about 20 likely prompts queued after each command (about 53 ms each on the L40S) ran back to back on the GPU right when the command's chunk was being generated, which doubled that chunk's generation time and starved playback.

## Slow reader (LLM fallback)

With `--reader commands`, a `final` that the commands/ matching does not understand ("add a lion", "make it day", "turn the tree into a lighthouse") goes to one LLM call, `engine/slow_reader.py`.
It is the last fallback: the keyword table and any other matching in commands/ run first, and only what they miss reaches the LLM.

- The model is an Azure AI Foundry deployment, by default `gpt-4.1-nano`, called as Azure OpenAI chat completions with JSON mode, temperature 0 and at most 120 tokens.
- It gets the current scene (base and changes with their ids) and the sentence, and returns one operation for `commands/state.py`: add (any object goes in as kind `extra`), remove, set_strength, undo, reset, new_scene, or none.
- The reply is checked before use: a bad JSON reply, an unknown kind, an id that is not in the scene or a strength out of range gives None, and the words show as unmatched.
- `CommandService` waits up to 1.5 s for it; the HTTP call itself gives up after 2 s.
- Measured from the engine machine on 2026-11-14: 390 to 550 ms per call with `gpt-4.1-nano` (`gpt-4.1-mini` took 1.0 to 1.3 s).
- Each call logs one line: `slow reader: 'add a lion' -> {...} (547 ms, ok)`.
- `CommandService.handle` holds its lock during the call, and the streamer thread asks for `state_fields()` on every chunk.
  So the engine wraps the service in `StateCachingReader`, which answers `state_fields()` from the state the last `handle` returned, and the stream never waits for the LLM.
- Credentials come from the environment: `FOUNDRY_ENDPOINT`, `FOUNDRY_KEY` and `FOUNDRY_DEPLOYMENT`.
  On the engine machine they are in `~/.madhacks-foundry.env` (mode 600, never in git), which the systemd unit loads with `EnvironmentFile=`.
  For a manual run, load it first: `set -a; . ~/.madhacks-foundry.env; set +a`.
- Without the variables the engine runs as before and logs one line: `slow reader off: FOUNDRY_ENDPOINT or FOUNDRY_KEY is not set ...`.

## Lifecycle states

| State | Meaning | Next |
|---|---|---|
| LOADING | `backend.load()` is running | WARMING, STOPPED, FAILED |
| WARMING | encoding the base prompt and the precompute file | GENERATING, STOPPED, FAILED |
| GENERATING | producing chunks | RESETTING, RECOVERING, STOPPED, FAILED |
| RESETTING | between chunks: wait for decode to finish, `backend.reset()`, `Controller.reset` on the new video's prompt | GENERATING, RECOVERING, STOPPED, FAILED |
| RECOVERING | generation or decode raised: drop the queued video, `backend.reset()`, `Controller.reset` on the starting prompt | GENERATING, STOPPED, FAILED |
| FAILED | too many crashes, or an error before generation started; `/stats` shows the error; the server exits with code 70 | STOPPED |
| STOPPED | shut down | none |

Every change goes through `LifecycleMachine.transition`, which logs it and rejects any change not in this table.
Chunk numbers keep rising across a reset and a recovery.
Commands and resets are refused while RECOVERING; it lasts a few ms.

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
`get_cond` is taken right before `generate_chunk`, after the lead gate wait, so that chunk is the first one not yet started.
`generated` is the first chunk of the ramp, not its end: with a 4-chunk ramp that chunk carries `ease(1/4)`, about 10% of the move; half the move is on screen one chunk (0.25 s) later and all of it three chunks later.
A command the reader found no operation for (an unmatched final, a tap with nothing to do) has only `heard`, and `applies_from_chunk` stays null.

Where the delay goes, measured on the fake backend at the live timings (generation 213 ms, decode 21 ms, 10 commands, cached prompts):

| Part | Mean | Range |
|---|---|---|
| heard to embedded (reader, cache hit) | 3 ms | 2 to 4 |
| embedded to the next chunk's `get_cond` (the chunk being generated finishes, then the lead gate) | 102 ms | 18 to 228 |
| that chunk's generation | 213 ms | 213 |
| generated to shown (decode, JPEG, and the about 2 older frames still ahead in the lead buffer) | 161 ms | 160 |
| heard to shown | 479 ms | 396 to 605 |

## Contract 4 fields

The engine sends one message per chunk when the chunk's first frame is sent.

- `changes`, `heard`, `keyword`, `unmatched`: from `reader.state_fields()`, called for every message.
- `chunk`: that chunk's number.
- `progress`: `steerer.progress` right after the `get_cond` for that chunk.
- `stamps`, `t_client`, `applies_from_chunk`: from the most recent command, which is the newest `final`, `tap` or `POST /target` the reader read, with or without an operation, or a `partial` that got a steer.
  So an unmatched final updates `t_client` together with `heard`, `keyword` and `unmatched`, and its stamps are `{"heard": 0}`.
  A `partial` with no steer, and a message the reader rejected or failed on, do not replace it.
- Stamps appear as they become known, so the page sees `heard` and `decided` first and `shown` last; `{}` before any command.

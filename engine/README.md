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

The demo harness tests (`test_demo_run*.py`) are marked `tools` and left out by default (root `pytest.ini`); add `-m tools` to run them.

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
If the engine connection drops during a run, the harness stops waiting at once, saves what it received so far, and exits with code 1 (PR #153, PR #161).
A run against a URL that is not localhost changes the shared live video for every viewer, so the harness prints a warning and waits 5 s before it starts (`--yes` skips the wait).

## Acceptance check

What a user gets for each sentence is tested from one data file, `acceptance/sentences.json`; the format is in the root `README.md`, "Acceptance".

- `engine/tests/test_acceptance_sentences.py` (default suite, CPU) runs every case through `FinalReadingService`, `FinalReader` and `CommandService`, with the recorded Sonnet replies of `acceptance/sonnet_replies.json` and with Sonnet down (`engine/tools/acceptance.py`).
- `python -m engine.tools.acceptance_record [--case NAME] [--refresh-stale]` records missing Sonnet replies; it needs `FOUNDRY_ENDPOINT` and `FOUNDRY_KEY`, and waits for each real call (not the live 3 s) so a slow call is saved before the case goes on.
- `python -m engine.tools.acceptance_live --url URL --out DIR [--calibrate] [--case NAME] [--max-calls 60] [--gap 8] [--yes]` plays the cases with `live: true` on a running engine and asks the Foundry deployment `claude-sonnet-5-5` yes/no questions about the frame 3 s after each step; `--calibrate-only` judges `acceptance/vision_calibration.json` and stops.
  A step is never judged on a frame from before the command: with no applied frame, or a video that stopped, it is `no frame` and the run fails.
  It changes the shared live video, so post on the board before a run on the live engine.

## The live engine on the GPU machine

The live engine runs on `madhacks-engine` (one NVIDIA L40S) in two tmux sessions: `engine` (the server) and `tunnel` (a Cloudflare quick tunnel).
The model is Causal Forcing++ framewise 2-step at 672x384 with the TAEHV decoder; see `engine/MODEL_NOTES.md` for the model, licenses and timing table.
One chunk is one latent frame, which is 4 video frames (0.25 s at 16 fps), so real time needs generation plus decode under 250 ms per chunk.
At 832x480 one chunk takes about 443 ms on this GPU, so that size is not real time on one L40S.

### Measured numbers

| What | Value | Source |
|---|---|---|
| Generation per chunk | 191.5 ms average | live `/stats`, read at about 00:15 UTC on 2026-11-15 |
| Decode per chunk (TAEHV) | 21.9 ms average | the same `/stats` read |
| Frame rate out | 16.0 fps | the same `/stats` read |
| Command to screen, draft script | 563 to 674 ms with the Sonnet reader's pinned prompts (637 to 640 ms before it) | harness runs after PR #125 (board post at 00:38 UTC) and after PR #113 (00:09 UTC) |
| Generation around commands | 190 to 210 ms idle, up to about 230 ms around commands, 16 fps held | the same post at 00:38 UTC |
| Command to screen on the page | 496 ms ("make it winter"), 429 ms (push-to-talk "make it night") | Ani's live check after PR #104 (board post at 23:59 UTC) |
| Sonnet final reader, new sentence | 1.87 s per call (p50), 2.3 to 3.1 s command to screen on a test engine; about 2 s extra live | test engine with the fake backend, 5 calls (board post at 23:31 UTC); live (00:38 UTC) |
| Sonnet final reader, pinned or cached sentence | no extra wait | the same post |
| Latency stages, pinned sentence (median ms) | reader 9, text encoder 1 (prefetched), wait for the chunk boundary 91, generation of the command's chunk 345 (a normal chunk is about 210; the re-anchor runs on it), decode and JPEG 19, playout queue 50, network about 40; server heard to shown 555 (494 to 637), client total 596 (536 to 674) | 15 commands in 3 live harness runs (board post at 00:55 UTC) |
| Latency, new sentence (Sonnet call) | reader 1679 ms median (1326 to 2772), Sonnet about 1380 ms per call, 2 of 8 needed a retry; client total 2290 ms (1920 to 3418) | 8 commands in live harness runs (the same post) |
| Foundry limits | prompt caching works but saves no time (1582 against 1577 ms); Azure returns 429 after about 20 calls in 40 s | 10-sentence bench from the laptop (the same post) |

The steps from a command to its first frame are in "Command stages and delay stamps" below.

### Live flags and why each is set

The live command line is `ExecStart` in `engine/deploy/madhacks-engine.service`:

```
python -m engine.server --backend causal_forcing --reader commands --width 672 --height 384 \
    --backend-opt vae=taehv --backend-opt cut_fade_frames=8 \
    --backend-opt steer_steps=1000,500 --backend-opt defer_context=1 \
    --reanchor on --reanchor-opt refresh_every_chunks=24 --reanchor-opt slow_strength=1.0 --final-reader sonnet --final-timeout-s 3.5 \
    --flush-on-cut on --interrupt-on-steer on \
    --host 127.0.0.1 --port 8000 --record-dir /home/ubuntu/recordings
```

The unit also sets `COMMAND_SEMANTIC=1` and loads `~/.madhacks-foundry.env`.

| Flag | Why |
|---|---|
| `--width 672 --height 384` | The largest size measured under the 250 ms budget on one L40S (Gate 1). |
| `--backend-opt vae=taehv` | The stock Wan VAE decodes in 157 ms per chunk against 21.5 ms for TAEHV, with an almost identical picture (Ishaan, board post at 23:53 UTC). |
| `--reanchor on` | Moving the embedding alone does not change the picture: the model keeps copying its old frames. A re-anchor restarts the next chunk from a noised copy of the picture under the new prompt (Gate 2, PR #57). |
| Refresh (default on with `--reanchor on`) | Without it the picture over-saturates by 25 to 30 s and breaks by 40 to 60 s; a refresh from the clean anchor keeps it clean (PR #83, board post at 23:47 UTC). |
| `--reanchor-opt refresh_every_chunks=24` | A refresh from the clean frame every 24 chunks (6 s) keeps the picture from drifting (PR #83). Since PR #128 a refresh draws its noise from a generator seeded by the clean anchor (`refresh_noise=fixed`, the default), so it returns to the same picture instead of re-arranging the scene; live since 01:50 UTC on 2026-11-15 (PR #184). Rollback: `--backend-opt refresh_noise=fresh` or `refresh_every_chunks=48`. |
| Decoder restart (always on, `first_frame_pad=hold` default) | A re-anchor chunk is a first frame; decoding it as a continuation gave a washed-out, ghosted chunk (sharpness 498 against 2049 before; 1871 after, PR #83). |
| `--backend-opt cut_fade_frames=8` | A re-anchor is a hard cut; dissolving from the last shown frame over 8 frames (0.5 s) brings the largest frame-to-frame change at a steer from 82 to 13 (PR #113). Refreshes fade too (`cut_fade_refresh` default 1); Ishaan saw no double car on the live recording, so it stays on (board post at 00:12 UTC). |
| `--backend-opt steer_steps=1000,500 --backend-opt defer_context=1` | A steer cut starts on a 2-step schedule (PR #174), and each chunk's context pass runs at the start of the next chunk and is skipped before a cut (PR #187), so a cut's latent is ready after 2 passes instead of 5. |
| `--flush-on-cut on` | When a steer cut's frames are decoded, the old-scene frames still queued for sending are dropped, so the cut is the next frame on screen; the 8-frame dissolve is shifted to start from the frame the viewer saw last, and the recorder saves exactly what was shown (PR #198). |
| `--interrupt-on-steer on` | When the policy says the next chunk is a steer cut, the old-scene chunk being generated is stopped at its next pass boundary and the cut takes its chunk number; refreshes, cuts and the first chunk of a video are never stopped (PR #187, PR #198). On the fake backend at live timings these four flags take command to screen from 554 to 296 ms median (621 to 349 p90) with no added stalls. |
| `--reanchor-opt slow_strength=1.0` | A slow command ("slowly make it winter", ramp of 12 chunks) re-anchors twice along its ramp (at about a third of the way and at the end; live log at 03:53 UTC: chunks 300 and 308). At the default 0.5 those anchors do not change the world and no new clean anchor is taken, so the next refresh pulled the picture back (live log at 03:16 UTC). At 1.0 each anchor is a full cut toward the ramp's current prompt, so the change arrives in 2 dissolved steps about 2 s apart and ends on the full target (live harness at 03:53 UTC: full winter that stays through later refreshes). |
| `--reader commands` | The `commands/` scene state, dictionary and composer read every sentence. |
| `COMMAND_SEMANTIC=1` | Turns on the CPU vector matcher in `commands/` for finals the dictionary misses (PR #73, PR #78). |
| `--final-reader sonnet` | Every final sentence goes to one Claude Sonnet 5.5 call that returns the operations and a rich prompt, with the dictionary as the instant result for partials and the backup on a timeout (PR #87, PR #120, live since PR #125). |
| `--final-timeout-s 3.5` | How long a final waits for Sonnet (also the default since #285). At 3.0 s, "make it slow" on a long chain (about 300 tokens of prompt) timed out 2 of 3 runs, with a median call of 1.98 s (regression sweep on 243a1e7); the lead rejected 5.5 s. |
| `--demo-prompts` (default `engine/demo_prompts.json`) | The demo sentences' Sonnet prompts are pinned, so every run draws the same words and a prompt checked on the GPU stays as checked (see "Final reader (Sonnet)"). |
| `--record-dir /home/ubuntu/recordings` | Every video is saved as an mp4, for the backup video and for checks such as Ishaan's refresh review. |

Rollback for the latency deploy: remove `--flush-on-cut on --interrupt-on-steer on` and the two `--backend-opt steer_steps=... defer_context=...` options. Rollback for the 01:50 UTC deploy (PR #184): `--backend-opt refresh_noise=fresh`, or back to `refresh_every_chunks=48`.

### Deploy and restart

First time on a new machine: `bash engine/setup_gpu.sh`, then `uv pip install -r engine/requirements.txt -r steering/requirements.txt` into `~/engine-venv`, and for the vector matcher `pip install -r commands/requirements-semantic.txt` and `python -m commands.semantic --prepare` as the `ubuntu` user.

The repo on the machine is a copy made with `rsync -a --delete --exclude .git ./ madhacks-engine:madhacks/` from a laptop clone of `main`, because the machine has no GitHub credentials.
Build the studio page in that clone first (`cd lucid-studio && npm ci && npm run build`), so `lucid-studio/dist` goes with the copy.

1. Post on the board before a restart: the video is gone for about 80 s, and the tunnel address stays the same.
2. Copy the repo as above.
3. If `engine/deploy/madhacks-engine.service` changed: `sudo cp ~/madhacks/engine/deploy/madhacks-engine.service /etc/systemd/system/ && sudo systemctl daemon-reload`.
4. `sudo systemctl restart madhacks-engine`.
5. Wait until `curl -s localhost:8000/stats | jq .state` prints `GENERATING` (loading takes about 70 to 90 s).
6. Run the harness (see "Demo harness") before and after the restart, and post both sheets.

A change to only `web/` or `lucid-studio/dist` needs the copy and no restart, because both are read from disk on each request.

The manual way, without systemd, is the same command line in a tmux session named `engine`, with `2>&1 | tee -a ~/runs/engine-server.log`.

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

### What the real model does now (2026-11-15)

- Steering works with `--reanchor on`: Gate 2 passed live after PR #57, with full winter about 2 s after "make it winter" (`engine/samples/gate2_live_make_it_winter_pass.jpg`).
- Long-run drift is fixed: 90 s after a Reset with no commands the picture stays clean and shows the same scene (`engine/samples/drift_after_83_90s.jpg`, after PR #83).
  Before PR #83 the picture was over-saturated by 25 to 30 s and broken by 40 to 60 s, and the old refresh (every 60 chunks at 0.25, from the last picture) broke the picture on the refresh chunk (`engine/samples/refresh_breaks_picture.jpg`).
- The draft script is clean in every frame, and the red car is red and visible (`engine/samples/harness_before_after_113.jpg`).
- Every steer and every refresh re-composes the shot: the camera, the tree and the car move, because a re-anchor restarts the picture from noise and continuity comes only from the prompt text (Ishaan, board posts at 23:42 and 00:12 UTC).
  The refresh every 48 chunks halves how often this happens; a backend fix is being measured by Ishaan.
- An added lion was missing or tiny before PR #125; with the pinned prompts (size and position words) it is on screen in the left foreground through night and sunny (`engine/samples/harness_before_after_sonnet.jpg`).
- "make it a desert" drops the snow and keeps the car and the lion; "add a red car racing on a road" after it gives a weak, cropped second car, so it is kept out of the script (`engine/samples/sonnet_desert_chain_live.jpg`).
- No variant tried kept the exact camera and layout across a world change (Ishaan, board post at 23:48 UTC).

## Command line

```
python -m engine.server --backend fake|causal_forcing --reader standin|commands --host 127.0.0.1 --port 8000 --width 832 --height 480 \
    --steps 0 --base-prompt "..." --jpeg-quality 80 --precompute-file prompts.txt \
    --device cuda:0 --decode-device cuda:0 --backend-opt key=value --lead-margin 1.3 --reanchor off --reanchor-opt key=value --log-level info \
    --record-dir ~/recordings --record-max-gb 20 --final-reader off --final-timeout-s 3.0 --final-cache ~/.cache/madhacks/final_reader.json \
    --short-chunk crossfade --flush-on-cut off --interrupt-on-steer off
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
- `--final-reader off|sonnet` (default `off`), `--final-timeout-s` and `--final-cache` control the Sonnet final reader (see "Final reader (Sonnet)").
- `--short-chunk fill|hold|crossfade` (default `crossfade`) sets how a chunk with fewer frames than `frames_per_chunk` is padded (see "Short chunks").
- `--flush-on-cut off|on` and `--interrupt-on-steer off|on` (both default `off`) make a steer cut reach the screen sooner (see "Cut latency").
- The fake backend's `pass_ms`, `denoise_passes` (2), `cut_passes` (4), `defer_context` (0) and `cut_fade_frames` (0) run the real model's passes and dissolve, to time "Cut latency" on a laptop; `pass_ms=0`, the default, keeps the single `gen_ms` sleep.
- The fake backend's `short_on_chunks=40,80` makes those chunks (numbered across resets, as `fail_on_chunk`) 1 frame long and labels them "short chunk", to test "Short chunks".
- `--record-dir` and `--record-max-gb` control the recorder (see "Recorder"); `--record-dir none` turns it off.
- SIGINT or SIGTERM shuts down cleanly, finalizes the open recording and moves the engine to STOPPED.
- The process exits with code 70 when the engine reaches FAILED, so systemd restarts it.

### Real backend options (`--backend-opt`, `causal_forcing`)

The full list with defaults is in the `load()` docstring of `engine/backends/causal_forcing.py`; the re-anchor ones:

- `clean_anchor_after` (3), `first_frame_pad` (`hold`), `cut_fade_frames` (0), `cut_fade_refresh` (1): see "Re-anchor" and `steering/README.md`.
- `refresh_noise` (`fixed`, the default since PR #128; `fresh` is the old behaviour): a refresh from the clean anchor draws all its noise from a generator seeded by the video's seed and the clean anchor's chunk, so every refresh from one anchor returns to the same picture.
- `refresh_anchor_t` (unset, or 1..1000): the level the clean anchor is noised to on a refresh, instead of the strength's anchor step (500 = step 2, 250 = step 3); unset on the live engine.
- Seeds: `reset(seed=N)` replays the same video, refreshes and fixed-noise steers included; any int seed works (the derived seed is reduced to 64 bits, PR #152); an unseeded reset takes the previous video's seed plus one, so it gets new refresh and steer noise (PR #144).
- `steer_anchor_t` (unset, or 1..1000): a steer (a re-anchor with source `last` at a strength above 0.75, which every command is) starts from the last clean latent noised to this level instead of pure noise; with no clean latent yet, from pure noise as before.
- `steer_noise` (`fresh` or `fixed`): `fixed` draws all of the steer chunk's noise from a generator seeded by the video's seed, the same for every steer of the video, without advancing the main noise stream.
- `steer_steps` (unset, or a comma list such as `800,500`): the steer chunk's schedule, strictly descending in 0..1000, starting at `steer_anchor_t` (or at 1000 when it is unset); unset means `steer_anchor_t` and the first-frame steps below it, or today's first-frame schedule.
- `steer_noise_first_frame` (0): 1 makes the first frame of every video use the fixed steer noise too.

The four steer options are off by default, and then nothing changes; a bad value stops the backend at load, before any weights are read.
They keep the shot's layout across a world change (tree trunk shift 12 to 21 px against 56 to 96 px, one seed so far, more runs in progress); see `engine/MODEL_NOTES.md`, "Steer start".
On the live engine only `steer_steps=1000,500` is set (the latency deploy, see "Live flags and why each is set"); `steer_anchor_t`, `steer_noise` and `steer_noise_first_frame` are off.

- `defer_context` (0): 1 runs each chunk's context pass (its frame KV cache write) at the start of the next chunk instead of at the end of its own, and skips it when that chunk rebases (every re-anchor), so a latent reaches the decoder one pass (about 90 ms) sooner and a cut does not pay for the old scene's pass. Latents and RNG draws are unchanged.
- `Backend.interrupt()` is not an option but a method: any thread may call it to stop a non-rebase chunk at its next pass boundary (`ChunkInterrupted`), leaving the backend as if the call had not been made, so a steer cut can take that chunk's number. The engine calls it with `--interrupt-on-steer on` (see "Cut latency").

See `engine/MODEL_NOTES.md`, "The cost of a cut": alone they give little on the live engine, which holds about 5 frames of old video ahead of a command.

## Endpoints

All on one origin; the message shapes are in `contracts/README.md`.

| Endpoint | What it does |
|---|---|
| `GET /` | The engine's own test viewer (`engine/static/index.html`); the demo page is `web/`. |
| websocket `/frames` | Binary frames out: 4-byte big-endian chunk number, 2-byte big-endian frame index in that chunk, JPEG bytes. |
| websocket `/control` | Contract 1 in (JSON text); contract 4 out, one message per chunk; `{"error": "..."}` replies to a bad message. |
| `POST /target` | Body is a contract 2 message; it skips the command reader and goes unchanged to the encoder and `steering.Controller.handle`. |
| `POST /reset` | Same as a `{"type":"tap","action":"reset"}` on `/control`: goes through the reader, then starts a new video; returns when done. |
| `GET /stats` | JSON: lifecycle state, backend, reader, last generated and shown chunk, last and average generation, decode and JPEG ms, measured fps, queue depths, re-anchor state and last anchor, encoder cache and pending encodes, current video number, recovery count and last recovery time, short chunks, recorder, clients. |
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

## The studio page at /studio/

The server serves the studio page build (`lucid-studio/`, branded Mirage) (a Vite app) read-only at `/studio/`, from `--studio-dir` (default `<repo>/lucid-studio/dist`).
Its asset names are hashed, so any regular file inside that folder is served, not an allow-list.
Dotfiles, missing files, directories, path traversal and symlinks that point outside the folder all return 404.
`/studio` redirects to `/studio/`, which serves `index.html`.
`index.html` and other files are sent with `Cache-Control: no-cache`; hashed files under `assets/` may be cached for a year.
If the folder is missing, `/studio/` returns "Studio build not found at <path>" with a 404, and the server still starts.
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

## Short chunks

After a rebase the model restarts its frame memory, and the decoder restarts on that chunk, so it decodes to 1 frame instead of `frames_per_chunk` (4).
Played as it is, the playout runs 3 frames (0.19 s) short and the stream stalls for about 47 ms at the next chunk.
The decode thread finds such a chunk by its frame count: fewer frames than `frames_per_chunk`, and not the first chunk of a video.
It pads the chunk to `frames_per_chunk` frames before JPEG encoding, by `--short-chunk` (the enum `ShortChunkMode`):

- `crossfade` (default): the missing slots blend linearly from the last frame decoded before the chunk to the chunk's first frame, ending on that frame.
  With 1 of 4 frames the slots are 25 %, 50 %, 75 % and then the new frame itself.
  The blend is integer math in uint16 and takes about 0.7 ms at 672x384 alone, and about 1.9 ms at 832x480 inside a running fake server.
- `hold`: the missing slots repeat the chunk's first frame.
- `fill`: no padding, the old behaviour, kept for comparison.

The first chunk of a video (at start, after a reset or after a recovery) is never padded.
It is a cut to a new picture, so a cross-fade from the old video would blend two unrelated scenes, and padding it would add 0.19 s to every reset.
The decode thread knows it by the video number in `ChunkMeta`: the last decoded frame belongs to another video, or there is none.
The decision is the enum `ChunkShape`: `full`, `video_start` or `short`.
The padding frames count as frames of the chunk, so the lead gate's estimate of `frames_per_chunk` frames stays right.
The padding time counts in the chunk's `decode_ms` (and so in `/stats` and the lead limit), because it runs on the decode thread between decode and JPEG encoding.
Each short chunk logs one line (`chunk 40: short chunk, 1 of 4 frames, mode crossfade: padded to 4 frames in 1.9 ms`).
`/stats` has `short_chunks`: `mode`, `count`, `last_chunk`, and `last` (`chunk`, decoded `frames`, `pad_ms`).
The real backend's default `first_frame_pad=hold` already repeats a re-anchor chunk's 1 frame to 4, so on the live engine no chunk is short (`/stats` showed `short_chunks.count` 0); this padding matters only with `--backend-opt first_frame_pad=none` or the fake backend's `short_on_chunks`.

## Re-anchor

`--reanchor on` puts `steering.reanchor.ReanchorPolicy` (see `steering/README.md`, "Re-anchor policy") into the generation loop.
With `--reanchor off`, the default, no policy exists and every chunk is exactly `backend.generate_chunk(cond)`.

- The policy is built next to the Controller in `Engine._new_steerer`.
- `on_target(msg, result)` runs right after every `Controller.handle`, in `Engine._apply_target`, under the CommandBook lock.
- `next_chunk(progress, is_moving)` runs once per chunk in `Engine._take_cond`, under the same lock and right after `get_cond`.
- `on_reset()` runs on every reset, right after `Controller.reset`.
- `Engine._generate(cond, anchor)` is the one backend call: with an anchor it calls `backend.reanchor(strength, source=anchor.source)` and then `backend.generate_chunk(cond)`.
- The state is an enum, `ReanchorState`: `off`, `on`, or `unsupported` when `--reanchor on` is given but the backend has no `reanchor(strength)` (logged once at load, then the engine runs as with `off`).
- The real backend (`causal_forcing`) has `reanchor(strength, source)` since PR #52, so it reports `on`.
  It cuts the frame cache and starts the next chunk from a picture noised by an amount set by the strength: the last picture for a steer (source `last`), or the clean anchor saved 3 chunks after the last reset or steer for a refresh (source `clean`, PR #83).
- The re-anchor chunk is marked `ChunkStart.FIRST_FRAME`, and the decoder restarts its stream state on it, as for a new video (PR #83); with `cut_fade_frames` set, the decoder dissolves into it from the last shown frame (PR #113).
- The fake backend records each `reanchor` call and draws a white border and "re-anchor 1.00" on the next chunk's frames.
- A re-anchor chunk is logged when it is generated and again on its decode line (`re-anchor steer 1.00`).
- `/stats` has `reanchor`: `state`, `count`, `last_strength`, `last_reason` (`steer` or `refresh`), `last_chunk`, `config` (the options that differ from the defaults), and the policy's `chunks_since_anchor`.

### Re-anchor options

`--reanchor-opt KEY=VALUE` sets one field of `steering.reanchor.ReanchorConfig`, without a change to the steering code.
It may be repeated; the fields and their defaults are in `steering/reanchor.py`.
Each value is converted to the field's type: an int, a float, or for `refresh_every_chunks` an int or `none`.
The live engine refreshes every 24 chunks, the default, and passes it explicitly (see "Live flags and why each is set"):

```
--reanchor on --reanchor-opt refresh_every_chunks=24
```

`refresh_every_chunks=none` turns the refresh off.
Other examples: `--reanchor-opt steer_strength=0.9`, `--reanchor-opt min_gap_chunks=8`.
The options are checked when the server starts, before the backend loads.
An unknown key, a value that does not convert, or a value `ReanchorConfig` rejects makes the server exit at once with a message such as `--reanchor-opt: unknown re-anchor option 'bogus'; known options: ...`.
The options are kept with `--reanchor off` too, but have no effect there.

## Cut latency (flush and interrupt)

About 5 frames (0.3 s) of the old scene are queued ahead of a command, in the playout queue and in the chunk being generated, and a steer cut waits behind them.
Two engine options remove that wait. Both are off by default, each is an enum, and each shows in `/stats` under `cut`.

- `--flush-on-cut on` (`FlushOnCut`): when a steer cut's frames are decoded, the decode thread drops the frames of the same video older than the cut that are still waiting to be sent, then queues the cut.
  It runs only once the cut's frames exist, so the stream never waits on an empty queue for them, and the lead gate counts the dropped frames as sent, so the next chunk starts on time.
  A refresh never flushes, and frames of an older video (the tail before a reset) are kept.
  The streamer and the flush share one lock: a frame the streamer took while the flush ran is dropped too, and the flush knows exactly which frame was the last one sent.
- `--interrupt-on-steer on` (`SteerInterrupt`, needs `--reanchor on` and a backend with `interrupt()`): when an applied target makes `ReanchorPolicy.fires_next_chunk()` true, the chunk being generated is stopped with `backend.interrupt()`.
  The chunk in flight is an `InFlight` state: `video_start`, `anchor` (a cut or a refresh), `continuation` or `stopping`; only `continuation` is ever stopped, so a refresh, a cut and the first chunk after a reset or recovery always finish.
  The stopped chunk raises `ChunkInterrupted`; it is never decoded, shown, timed or counted, and the cut takes its chunk number, so `applies_from_chunk`, `generated` and `shown` are the cut's.
  A stop that comes after the chunk's last pass boundary lets it finish (`late_interrupts` in `/stats`).
- Recorder: it saves exactly what was shown. Dropped frames and stopped chunks are not in the mp4, so a flushed cut shows the same jump in the recording as on screen.
- Dissolve (`cut_fade_frames`): the decoder dissolves into a cut from the last frame it decoded, which a flush has just dropped. The engine moves that dissolve to start from the frame the client saw last (`DissolveFix`: frame k gets (1 - a) × (shown - decoded) added, a = (k + 1) / (N + 1)), over the same 8 frames, so the old picture does not jump ahead before it fades.
  Without a flush the last decoded frame is the last shown one, and nothing changes.

Measured with `python -m engine.tools.cut_latency` (fake backend at the live pass timings: a pass 69 ms, decode 22 ms, `--reanchor on` with a refresh every 24 chunks, 24 cached ramp-1 commands per row, POST /target heard to shown):

| Variant | heard to shown, median / p90 ms | stalls | chunks stopped | old frames dropped | checks |
|---|---|---|---|---|---|
| today (5-pass cut) | 554 / 621 | 0 | 0 | 0 | ok |
| 2-step cut + defer_context | 545 / 605 | 0 | 0 | 0 | ok |
| + flush | 393 / 487 | 0 | 0 | 48 | ok |
| + flush + interrupt | 296 / 349 | 0 | 15 | 22 | ok |

Ishaan's prototype measured 566/674, 523/635, 414/471 and 300/376 for the same rows (PR #187). A 2-step cut alone saves a pass only after the old frames have played, so the gain comes from the flush and the interrupt.

The live flags for all of it: `--backend-opt steer_steps=1000,500 --backend-opt defer_context=1 --flush-on-cut on --interrupt-on-steer on`.

## Commands

contract 1 on `/control` -> reader worker thread: `reader.handle` -> `steer` (contract 2) -> encoder service -> steer worker thread: `steering.Controller.handle` -> `steerer.set_target` -> the next `get_cond`.

- The reader API, agreed with the `commands/` owner, is in `engine/reader.py`:
  `handle(msg) -> {"steer": contract 2 or None, "state": {...}}`, `state_fields()`, `likely_prompts()` and `committed_prompt`.
- Every contract 1 message (`partial`, `final`, `tap`) goes to `reader.handle`, one at a time in arrival order on its own worker thread, because it may block for up to 1.5 s (the slow reader) or 3.5 s (the Sonnet final reader), both below.
  If it raises ValueError, the client gets `{"error": ...}` on `/control`.
- The stand-in reader takes a `final` text as the whole new prompt with a 4-chunk ramp, gives no steer for `partial` and `tap`, and keeps an empty changes list.
- The `steer` dict is passed unchanged to `steering.Controller.handle`, including optional fields such as `strength` and `without_prompt`.
  The engine first encodes `prompt` and `without_prompt` in its encoder service, so `handle` never waits for encoding.
- A reset tap also resets the engine: RESETTING, `backend.reset()` to clear the frame cache, then `steering.Controller.reset(prompt, strength, without_prompt)` with the reset's `steer` (or the starting prompt if `steer` is None).
  That jumps the steerer to the new target with no ramp, honours a strength pair, and keeps the controller's prompt cache.
- Commands are ordered against a reset by when they were heard.
  A command heard before the reset tap is dropped by it.
  A command heard after it (for example "make it winter" a few ms after Reset) waits on the steer worker until the reset is done (`ResetTicket`, stage `heard`, then `requested`, then `done`; or `cancelled` if the reader rejects the tap), then applies to the new video, so it gets its own `applies_from_chunk` and all five stamps.
  The ticket exists from the moment the tap is heard, so a `POST /target`, which bypasses the reader, also waits while the tap is still being read.
  Its `embedded` stamp includes that wait, which is at most the reader's time for the tap plus about one chunk.
- The encoder service runs on its own thread with a priority queue, caches embeddings and token ids by exact prompt text (LRU, 512 entries), shares one encoding between identical requests, and logs cache hits, misses and encode ms.
- The reader's `likely_prompts()` are encoded at startup, and after every `final` and `tap` they are queued again at background priority, behind any command's encoding.
  Each new set replaces the background prompts still queued from the last one, because those belong to an older scene.
- A `partial` does not ask for `likely_prompts()` again: a partial never changes the committed scene, so the set queued after the last final or tap still holds.
  Typing sends one partial per keystroke, and the engine process is CPU bound on one core (the generation thread launches GPU work from Python and holds the GIL most of the time).
  Asking per keystroke cost about 47 scene copies each time, and on the live L40S unit it raised generation from about 195 to 240-280 ms per chunk and stalls from 26 to 135 while someone typed (journal, 2026-11-15 07:27 to 07:44 UTC).
  On the RTX PRO 6000 test box, typing at 15 characters per second, the reader thread held the GIL in 12 to 21% of py-spy samples before this change and under 1% after (issue #274).
- Prefetch is paced by `PrefetchPacer` (state `held`, `saving` or `ready`, shown in `/stats` as `encoder.prefetch`), because the text encoder shares the GPU with the generator.
  While a command is in flight (HEARD to SHOWN) it is `held` and no background encode starts.
  Otherwise the time generation waits at the lead gate, which is spare GPU time, is saved up; once it covers one encode (the fastest of the last 20 encodes) it is `ready` and one background encode runs.
  Urgent encodes (a command's own prompt) never wait for it.
  Without the pacing, the about 20 likely prompts queued after each command (about 53 ms each on the L40S) ran back to back on the GPU right when the command's chunk was being generated, which doubled that chunk's generation time and starved playback.

## Slow reader (LLM fallback)

With `--reader commands`, a `final` that the commands/ matching does not understand ("add a lion", "make it day", "turn the tree into a lighthouse") goes to one LLM call, `engine/slow_reader.py`.
It is the last fallback: the keyword table and any other matching in commands/ run first, and only what they miss reaches the LLM.
It is used only when the Sonnet final reader is off; the live engine runs `--final-reader sonnet` since PR #125, so the live engine does not call it.

- The model is an Azure AI Foundry deployment, by default `gpt-4.1-nano`, called as Azure OpenAI chat completions with JSON mode, temperature 0 and at most 120 tokens.
- It gets the current scene (base and changes with their ids) and the sentence, and returns one operation for `commands/state.py`: add (any object goes in as kind `extra`), remove, set_strength, undo, reset, new_scene, or none.
  A base edit ("turn the tree into a lighthouse") is `edit_base`, which becomes one batch of `new_scene` plus one `add` per existing change, so the other changes stay and one undo reverts it; `new_scene` alone clears every change.
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

## Final reader (Sonnet)

With `--reader commands --final-reader sonnet` (live since PR #125), `engine/reader.py` `FinalReadingService` decides who reads each `final` sentence, in this order (issue #271, free scenes):

1. A sentence made only of undo and reset clauses ("undo", "reset", "undo that and start over") goes to the keyword table, with no call.
2. The content block: a short word list (`BLOCKED_WORDS`) with no call; the model also sets `blocked: true` for sexual, violent or hateful requests.
   A blocked final gives no steer: the words show as unmatched.
3. A pinned entry (`engine/demo_prompts.json`), then a cached one (keys below).
4. A named person on a composed scene, when the sentence is nothing but adding or removing them ("jensen", "jensen huang", "add jensen huang next to the car", "remove jensen"; `people.whole_sentence`): the engine's own add (his hardcoded look, `people.PEOPLE`) or remove, with no call (`person_hits`).
   The sentence must be one clause (`commands.reader.split_commands`), and a spot ("next to the big red car") is at most four words after the article and stops before a word that starts another edit.
   So "add jensen huang next to the car and make it winter" or "..., delete the dog" is not this step: it goes to Sonnet with his look instead of his name, and the other edit applies too.
   A bare name is an add.
5. A measured car world (`car_template.plan`, `Route.MEASURED`): "make it winter" on the car scene, or a new car scene, composed by the engine with no call (`template_hits`).
6. The fast path: on a composed scene, when the keyword table reads every word of every clause by itself (`commands.service.dictionary_reading`: no vector matcher, no style fallback, no pass-through guess of an unknown object) and the result is one the table writes well (`engine/reader.py` `fast_path_ok`), the dictionary applies it with no call.
   Writes well means: undo, reset, remove, strength changes, explicit new scenes, and adds of a place, season, weather or time of day only on the plain base scene (no changes, no vehicle), at full strength, with no strength or amount word in the sentence ("some", "a little", "light", "a lot", "more", "less" and the other strength words of `commands/reader.py`).
   Object adds, partial strengths ("add a little fog") and adds on a vehicle scene go to Sonnet: the table's object phrases changed the count or the thing ("add an eagle" gave "a flock of birds", "add a human" a group of people) and had no size or position, "a little fog" read "a subtle hint of a thick fog", and on the car scene the template kept "clear blue summer sky" beside "a thick fog" (regression sweep on #273).
   On a scene with changes the table only appends its phrase ("make it snow" on the neon city kept "light rain falling" beside "winter scenery with snow on the ground"), and "gradually add some fog" wrote "a thick fog" (regression sweep on 243a1e7, #285): both go to Sonnet now.
   "make it winter" and "take me to the beach" on the plain meadow, and "remove the dog", are read whole; "make it night" on a winter scene, "add some fog", "add a dog", "add a red car", "kill the tree's leaves", "now i want a savannah" and "zoom out" are not.
7. Otherwise one Sonnet call writes the complete new prompt: a free scene.
   The scene becomes `commands.state.Origin.FREE`, its prompt is Sonnet's text word for word, and its chips are the labels Sonnet names.
8. When the call times out, fails or the reply is not a JSON object, the dictionary reads the sentence (the fallback).
   On a free scene the dictionary can only start a new scene, so anything else shows as not understood; the old scene is never silently redrawn.

A sentence that starts with undo or reset clauses and goes on with edits ("undo and add a dog", "reset and make it night") runs the controls on a trial copy of the scene (with its undo history), and the edits are read on the scene they give, through steps 2 to 7.
The controls and the edits' operations then apply as one batch (one undo step), and nothing applies when the edits are blocked or not understood.
So after two free-scene edits, "undo and add a dog" shows the scene before the last edit plus the dog, and "reset and add a dog" gets the same pinned prompt as "add a dog".
An answer saved for the whole sentence is used only when it is a block.
Any other mix (an undo after an edit, "add a dog and undo") goes to the dictionary, which runs the undo where it is said.
Sonnet is asked first, as the content check, only when the keyword table does not read every clause by itself, since its prompt is not used there.
A tap on the x of a free scene's chip is read as the sentence "remove the <label>" through steps 2 to 7.
Undo works across composed and free scenes: the scene history is a stack of snapshots, each with its `origin`.

The call:

- The Anthropic Messages API on the team's Foundry resource: `POST https://<resource>.services.ai.azure.com/anthropic/v1/messages`, where `<resource>` is the first label of `FOUNDRY_ENDPOINT`'s host.
  `FOUNDRY_ANTHROPIC_URL` replaces the whole URL (a local fake endpoint), and `FOUNDRY_REWRITE_DEPLOYMENT` the model (default `claude-sonnet-5-5`).
- Headers `x-api-key: $FOUNDRY_KEY` and `anthropic-version: 2023-06-01`.
  Body: `{"model": "claude-sonnet-5-5", "max_tokens": 1024, "system": <system prompt>, "messages": [{"role": "user", "content": <JSON>}], "thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}}`.
  No `temperature`: the model rejects it, and `between_tools` is how Sonnet 5.5 runs with thinking off.
- The call runs with thinking off (`between_tools`) at effort low and `max_tokens` 1024, within the time budget `--final-timeout-s` (default 3.5 s since #285: at 3.0 s "make it slow" on a long chain timed out 2 of 3 runs). Adaptive thinking at effort medium graded a little better on a short test prompt (4.54 against 4.40), but on the real system prompt the regression sweep (209 chains, 2026-11-15) measured p50 2.16 s against 1.65 s, 14 timeouts against 5 at 3.0 s, and replies that began with reasoning prose instead of JSON, so it is off. A reply that is not one JSON object is never used: it is asked again or the dictionary reads the sentence.
  `FINAL_READER_EFFORT` (low, medium or high), `FINAL_READER_THINKING` (the thinking type, passed as given; `between_tools` is thinking off) and `FINAL_READER_MAX_TOKENS` change the call without a code change, for measuring.
- The user message: `{"prompt": <the prompt the video model sees now, the rich one>, "chips": [<labels shown>], "sentence": <the sentence, a named person replaced by their description>, "car_scene": <true when the prompt on screen is the car template>}`, plus `"view_change": true` on a car scene when the sentence asks for another camera or view (`car_template.asks_view_change`) and `"person_chip"` when the sentence adds a named person.
- The reply: `{"blocked": bool, "understood": bool, "new_scene": bool, "prompt": str, "chips": [str]}`.
- The system prompt (`final_reader.SYSTEM_PROMPT` plus `car_template.instructions()`) asks for one complete scene description of at most 150 words: the subject and its place in the frame, its direction of travel, the camera, the setting, the light and the style, with Ishaan's measured car template and worlds as the example.
  Its rules: keep everything the sentence does not change word for word; on a change of place say for every object where it is, what it stands or drives on and which way it faces (a car goes on a road); explicit size and position words for objects; every moving thing states its direction of travel, and on the car scene it travels forward away from the camera like the car; counts agree in the whole prompt ("two oak trees", never "a single oak tree" and "a second tree"); a "no" becomes a positive description ("no clouds" is "a clear, cloudless deep blue sky"); a request repeated for something already in the prompt makes it stronger; the camera wording changes only when asked (zoom, wider or closer, side, aerial, front, low angle, pan) or when a vehicle starts driving; when the scene has grown past 100 words the new prompt is shorter than the old one, the oldest details compressed first; a new place named in any words ("city background", "middle of the city") replaces the old place and its landscape (the meadow, the oak, the grass) unless the user mentions it, and the thing the user talks about becomes the main subject; a vehicle drives by default (parked only when the user says so) on a road through the whole frame, with the car template's following camera unless another is asked for, and never "facing the camera" unless the user asks for its front; a new weather keeps the season ("make it rain" on a snowy night keeps the snow on the ground), and only what falls from the sky is replaced; `new_scene` only for an explicit new scene or a new main subject; one short chip per thing the user added or changed, and only chips whose words are still in the prompt (#285).
  "understood" is false only for a sentence that is not about the picture.

The light check (`final_reader.check_reply`) repairs what it can and never refuses a reasonable sentence:

- `blocked`: no steer. `understood` false, or an empty prompt: not understood (unmatched, a provisional guess is rolled back), saved like a block.
- A prompt over 300 words (`MAX_PROMPT_WORDS`) or 1900 characters is cut sentence by sentence from the end (the car template's style sentence stays last). umT5 reads 512 tokens; Sonnet's prompts measured 1.4 tokens a word and at least 3.9 characters a token, so both caps stay under 512. The earlier cap (180 words, 1600 characters) cut the sentence with the new object off long prompts (regression sweep on #273).
- A reply with a driving vehicle but without the car template's forward motion (`drives_without_template`: a vehicle word, a driving verb, and no "moving forward away from the camera"; a sentence that says "side view" is exempt) is `Verdict.NOT_FORWARD`: asked again when there is time, else the dictionary reads the sentence (on a composed vehicle scene that is the car template). Only the template's shot drew cars forward on the GPU; other wording reversed them, drove them off the road or turned them to the camera (live, 2026-11-15).
- A reply whose prompt is the prompt on screen (`Verdict.UNCHANGED`, often a cut that took the change away) is not used: it is asked again when there is time, else the dictionary reads the sentence, and on a free scene that shows as not understood, never as a chip with no change on screen.
- Chips are cleaned (lower case, whole words up to 24 characters, at most 16, no duplicates), never a reason to drop the reply.
- Chips match the prompt (`chips_in_prompt`): a chip whose words were in the prompt on screen and are in none of the new prompt is dropped and logged.
  A chip's words are its own content words and, for every keyword table entry named inside the label, all that entry's aliases ("winter": snow, snowy, frost, icy, ...; the "fog" of "light fog": mist, haze, ...).
  Keywords inside a label are read left to right, longest first ("lightning bugs" are fireflies, not the storm's "lightning").
  Filler and amount words ("second", "light", "thick", "some", ...) are no evidence when the label has other words; a label made only of them ("light") keeps them.
  A chip with no words in either prompt ("zoom out") and the named person's chip stay.
  Regression sweep on 243a1e7 (#285): "make it rain" on a snowy night took every snow word out and the "winter" chip stayed with nothing behind it.
- On a car scene that is not a new scene, the prompt must still start with the template's subject sentence (any subject, any road) and its camera sentence word for word, and the road must name a drivable surface (`car_template.DRIVABLE`).
  With `view_change` only a drivable road is needed, so "I want to see the car's side profile" can rewrite the camera.
  A reply that breaks this is still used and logged (`Verdict.REPAIRED`).
- A REPAIRED or unusable reply is asked again once, with the reasons, when the first call took at most 3/7 of the wait (`RETRY_SHARE`), so a second call still fits; a REPAIRED first reply is kept if the retry runs past the wait.
  On the live 3.5 s wait that is a first call under 1.5 s, as it was at 3.0 s with half the wait; with half of 3.5 s, retries after 1.6 s first calls ran past the wait (#285).
- The reply states are named in `Verdict` (accepted, repaired, not_understood, blocked, invalid, unchanged, not_forward, failed); `/stats` counts the final verdict of every call in `verdicts`.

The cache and the prompts:

- The cache is keyed by (composed prompt before the sentence, sentence); a free scene's composed prompt is its own prompt.
  The sentence part of every pinned and cache key is normalized (`sentence_key`: lower case, trimmed, spaces collapsed, end punctuation `. , ! ? ; :` and surrounding quotes stripped, inner words kept), so "Make it winter." from voice hits the pinned "make it winter"; the content check and Sonnet still get the original text.
  A hit is used only on the scene it was made for (`scene_ids`: origin, free prompt, base and every chip in order), with its chip ids mapped to the live ones.
- A Sonnet result, a block or a "not understood" is chosen once and saved to `--final-cache` (JSON, written atomically, tied to a fingerprint of the system prompt, the model and `final_reader.PROMPT_BUILD`), so a restart gives the same result.
  Change `PROMPT_BUILD` whenever the engine or `commands/` composes prompts differently from the same reply.
  A fallback is final for that key for the life of the process, and an answer that arrives after the wait is dropped (`late_dropped`); fallbacks are not saved.
- The model sees one prompt per composed scene state: the first rich prompt chosen for a composed prompt (pinned, measured or a car template fallback) replaces it in every later steer, in `likely_prompts` and in `committed_prompt`.
- Named people (`engine/people.py`): "add jensen huang" (also "jensen", "the nvidia ceo", or the bare name "jensen huang") on a composed scene is the engine's own add with no call (step 4). On a free scene it reaches Sonnet as "add <his visual description>" instead of the name, with `person_chip: "jensen huang"`, so his chip keeps the name; when the model gives nothing usable on a composed scene, the engine adds or removes him itself (`Outcome.PERSON`).
- Pinned, measured and named-person answers are composed operations, checked by `checked_operations` (`slow_reader.validate`, and `canonical_world` gives a known place its tested phrase).
- From the #272 stopgap, kept: `commands.world.compose_base` reads an edited oak scene base, an object the sentence only mentions as context ("i want to see the car's side profile") is never added by the keyword table, and a mover the dictionary adds on the car scene gets a heading with the car (`with_heading`) in `fallback_prompt`.
  Its `place_as_new_scene` went with the operations merge: Sonnet now writes the whole prompt, so a base that names a place cannot block a later change.
- `/stats` `final_reader`: `state` (`off`, `on`, `unavailable`), `reason`, `prompt_spec` (`free scene`), `calls`, `cache_hits`, `blocked`, `not_understood`, `timeouts`, `late_dropped`, `rejects`, `repaired`, `retries`, `errors`, `verdicts`, `pinned`, `pinned_hits`, `template_hits`, `person_hits`, `p50_ms`, `last_ms`, `cache_size`, `timeout_s`, `cache_file`.
- It is `unavailable` (and the engine runs as with `off`, logged once) when `FOUNDRY_ENDPOINT` or `FOUNDRY_KEY` is missing or the reader is not `commands`.
- `engine/reader.py` uses `CommandService.snapshot()` and `CommandService.apply_final(text, operations)`; `engine/tests/test_final_reader.py` runs them against the real `CommandService`.

Removed in #271 (they only existed to merge Sonnet operations into the composed scene): the `details` field and its repair (`PromptSpec`, `repair_details`, `why_rejected`, `engine_banned`, `details_budget`), the operations prompt, `_check`, `_check_vehicle`, `bare_vehicle_operations`, `expand_operations` with its base edit expansion, `world_first`, `distinct_label`, `canonical_change`, `with_lane_cue`, `pace_keeps_strength`, `phrase_problems`, `people.with_person`, and the car template's `field_problems` and `model_world`.

Local end-to-end run with a fake Sonnet endpoint (no key needed):

```
FOUNDRY_ENDPOINT=https://fake-resource.cognitiveservices.azure.com FOUNDRY_KEY=fake \
FOUNDRY_ANTHROPIC_URL=http://127.0.0.1:8771/anthropic/v1/messages \
python -m engine.server --backend fake --reader commands --final-reader sonnet --port 8770 --record-dir none --final-cache none
```

On the engine machine, load `~/.madhacks-foundry.env` first and add `--final-reader sonnet` to the usual command line.

`engine/demo_prompts.json` holds the real Sonnet prompts for the demo sentences, made by this final reader as one chain from a reset, in script order.
Since PR #125 it has 10 entries: the draft script (winter, red car, night, lion, sunny), "make it a desert", "add a red car racing on a road", and a car chain whose first prompt is Ishaan's tested car prompt word for word ("new scene of a red sports car driving along a country road", then winter and night).
The live engine loaded all 10 as pinned (`/stats` `final_reader.pinned` 10 after the 00:32 UTC restart).
Since the likely-sentences PR it has 133 entries: the first 19 (the two demo chains) unchanged, then chains of likely first and second steps from the meadow, the meadow in winter, the meadow at night, the car scene, the car in winter and the car at night.
A winter or night synonym said first ("make it snowy", "slowly make it winter", "suddenly make it night") is pinned with the same operations and prompt as "make it winter" or "make it night", so every second step pinned after that scene matches it too.
Inside a chain a removal ("remove the dog", "stop the rain") goes back to the scene before, and a car chain starts again with "new scene of a red sports car driving along a country road" said from the scene the chain before left; a key (`prompt_before`, `sentence`) never repeats, so the loader pins every entry.
Each entry is keyed like the cache, by `prompt_before` (the composed prompt of the scene before) and `sentence`, and also has `canonical` (the composed prompt it rewrote), `operations`, `prompt`, `model`, `fingerprint` and `generated_at`.
Entries are hand-editable: fix a bad `prompt` in place, and keep `prompt_before` and `sentence` unchanged so a loader still matches it.
`--demo-prompts PATH` (default `engine/demo_prompts.json`; `none` turns it off) loads it at start-up as pinned entries: they are never called or regenerated, never written to `--final-cache`, valid for any fingerprint, and checked before the cache.
Issue #259 (136 entries): the three "make it a neon city" entries on the meadow were asked again for the landscape form of the city, "a little more snow" on the car in winter was added after the car winter chain (with heavier snow in the scene sentence), and "make it a desert" and "take me to the beach" from the meadow were added at the end.
`python -m engine.tools.pin_drift` lists the entries whose composed prompts changed under new code (run it after a change to `commands/composer.py` or `commands/world.py`); `--regen`, with the Foundry key, asks those and any new entries again with the real Sonnet final reader and writes the file.
Live acceptance after #268 and #269 (06:57 UTC, `engine/samples/acceptance_live_0657.jpg`): 24 of 27; the meadow winter now shows deep snow; the three failures are the racing car on the plain meadow (KNOWN_ISSUES.md section 3) and two runs where a parked car was lost after a change of place (passed at 06:24).
Issue #267: the meadow "make it winter" prompt (and its 9 synonym copies) ended in "a soft white blanket over the grass, frost-dusted textures", which drew frosted grass on live; it now ends in "thick white snow blanketing the whole meadow, deep soft snowdrifts", edited by hand after an A/B on live (`engine/samples/winter_and_road_ab_0645.jpg`, rows 1 and 2: frost before, full snow after, 2 of 2 each).
The file is a list of entries, or `{"base", "entries"}`; the loader replays the entries in order from the base and skips, with a log line, an entry whose `prompt_before` is no scene the entries before it reached or whose operations do not apply.
An entry whose `prompt_before` is not the scene the entry before it leads to is read on the earlier scene of the file with that prompt, so a spoken variant ("make it snow") can follow the step it varies ("make it winter") mid-chain, and a chain can go on from any scene an earlier chain reached.
The demo script (`docs/demo-script.md`, 2026-11-15): two chains of pinned steps (`demo_chain` "meadow" and "car"), each step followed by its spoken variants (`variant_of`), made with this final reader and checked on the GPU at 672x384 (3 seeds) and 832x480 (2 seeds); 228 entries.

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
`commands/` now sends `ramp_chunks` 1 for a normal command (`RAMP` in `commands/reader.py`), so with `--reanchor on` the re-anchor fires on that same chunk and the new scene is there at once, dissolved in over 8 frames by `cut_fade_frames`.
A command the reader found no operation for (an unmatched final, a tap with nothing to do) has only `heard`, and `applies_from_chunk` stays null.

The live numbers are in "Measured numbers" near the top.
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

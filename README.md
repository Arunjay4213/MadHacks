# MadHacks: live voice-steered video

A video is generated without stopping by a diffusion model, and it changes while it plays when someone speaks or types an instruction, for example "make it winter", "add a red car" or "make it a desert".
Changes stack as named chips, and each one can be undone or removed by name.

## Credit

This project builds on [Training-Free Interactive World Models](https://hrdlabs.org/research/interactive-world-models/), a published project that steers a running video model with a voice command.
Their code is the [SPED repository](https://github.com/bryandong24/SPED).
We reuse their idea of steering the model's prompt embedding while it generates, and the open video model they used (Causal Forcing++, see `engine/MODEL_NOTES.md` for the model, the weights and their licenses).
What we added is listed in `PLAN.md`, section "What we are building".

## Parts and owners

Each part lives in its own folder, and each folder has one owner.
The full table is in [`AGENTS.md`, section 4](AGENTS.md#4-who-owns-what).

| Folder | What it does | Owner |
|---|---|---|
| `steering/` | Moves the prompt embedding toward a new target, and decides when to re-anchor the picture | Ishaan (`@Tactacion`) |
| `engine/` | The GPU server: generation loop, text encoder, frame streaming, delay stamps, readers, deploy | Arunjay (`@Arunjay4213`) |
| `web/` | The demo page served at `/app/`: video, words line, chips, undo, reset, text box, live mic | Ani (`@anirudhaugile`) |
| `commands/` | Turns a sentence into scene operations and one composed prompt; undo history | Prateek (`@prateekraouw`) |
| `lucid-studio/` | A second frontend, served at `/studio/` | Prateek (`@prateekraouw`) |
| `contracts/` | The four message shapes between the parts | Arunjay (lead) |
| `docs/` | The pitch and the write-up | Prateek (`@prateekraouw`) |

The plan of record is `PLAN.md`, written by Ishaan.

## Open the live demo

The engine runs on one GPU machine and is reached through a Cloudflare quick tunnel.
The tunnel address changes whenever the tunnel restarts, so the current address is posted on the [Message Board](https://github.com/Arunjay4213/MadHacks/issues/1).

- Demo page: `https://<tunnel address>/app/` in Chrome (the live mic needs Chrome and https).
- Lucid Studio: `https://<tunnel address>/studio/`.
- Engine numbers: `https://<tunnel address>/stats`.

There is one shared video and no login: every viewer sees every command and reset, so keep the address inside the team.
Automated checks belong on a local fake engine (`engine/README.md`, "Run it with the fake backend").

## Current state (2026-11-15, about 00:40 UTC)

- Live model: Causal Forcing++ framewise, 2 steps, 672x384, TAEHV decoder, on one NVIDIA L40S; `/stats` read at about 00:15 UTC showed 191.5 ms generation and 21.9 ms decode per chunk on average, at 16.0 fps.
- Gate 1 (real time) and Gate 2 (a typed command changes the real scene) passed; see the "Status" section at the end of `PLAN.md`.
- Every command re-anchors the picture: a clean cut to the new scene, dissolved over 8 frames (0.5 s), and the picture is refreshed from a clean saved frame, which stopped the long-run drift (PRs #83 and #113); the live engine refreshes every 48 chunks (12 s) since PR #125.
- Command to screen on the live engine: 429 to 496 ms on the page (Ani's live check after PR #104) and 563 to 674 ms in the draft-script harness with the Sonnet reader's pinned prompts (board post at 00:38 UTC).
- Place changes work: "make it a desert" or "turn it into a beach" replace the world and keep the objects (PRs #88 and #115).
- The Sonnet final reader (one Claude Sonnet 5.5 call per final sentence for the operations and a rich prompt) is live since PR #125, with the demo sentences' prompts pinned in `engine/demo_prompts.json`; a new sentence costs about 1.8 s extra, a pinned one nothing (board post by Arunjay's Claude at 23:31 UTC).
- The draft script (winter, a red car, night, a lion, sunny) now shows every change, the lion included, and "make it a desert" drops the snow and keeps the car and the lion (`engine/samples/harness_before_after_sonnet.jpg`, `sonnet_desert_chain_live.jpg`).
- Known faults: every steer and every refresh re-composes the shot, so the camera and the layout move (Ishaan's posts at 23:42 and 00:12 UTC), and "add a red car racing on a road" after the desert gives a weak, cropped second car (board post at 00:38 UTC).

## Tests

Run the tests from the repo root, in a venv with the engine and steering requirements.

```
python -m pytest -q
```

`pytest.ini` leaves out the tests marked `tools` by default.
They test the demo and evaluation tools (`engine/tools/demo_run.py` and `steering/eval/`), not the live demo.
Run them on their own with:

```
python -m pytest -q -m tools
```

## Acceptance

The acceptance check tests what a user gets, with no person looking at frames.
All cases live in one data file, `acceptance/sentences.json`, and both layers read it.

### Add a case

Add an object to `cases`:

```
{"name": "desert-after-winter", "start": "meadow", "live": true, "steps": [
  {"say": "make it winter", "chips": ["winter"], "has": ["snow"]},
  {"say": "make it a desert", "chips": ["desert"], "no_chips": ["winter"], "has": ["sand"], "lacks": ["snow"],
   "visible": ["Is the ground sand or desert?", {"q": "Is there snow on the ground?", "expect": "no"}]}
]}
```

- `start`: `meadow` (the starting scene) or `car` (the red sports car on the country road).
- A step is `say` (a typed or spoken sentence) or `tap` (`undo`, `reset`, or `remove` with `chip`, the x on a chip).
- `understood` (default true): false means the user should see the words as not understood and nothing should change.
- `chips` must be on screen, `no_chips` must not; "car" matches the chip "red car", and "horse" matches "horses".
- `has` and `lacks`: words that must or must not be in the prompt the video model gets (case does not matter).
- `slow`: true when a pace word must make the change gradual.
- `strength`: `{"winter": "up"}` or `"down"` against the step before, or `"low"`, `"full"`, `"high"` against 1.
- `visible`: yes/no questions about the frame, for Layer B; a plain string expects yes.
- `known_failure` on a step: a reason the team decided not to fix it (for example "slow changes are not used in the demo"); Layer B then reports a failing frame as a known failure, apart from the count.
- `live: true` puts the case in the Layer B run; `modes` limits Layer A to `sonnet` or `dictionary`.
- Write what the user asked for, never what the code prints today.
  When the code is wrong, add `"xfail": {"sonnet": "bug: ..."}` (or `dictionary`) to that case only.

### Layer A: sentences to prompts (CPU, in the default suite)

```
python -m pytest -q engine/tests/test_acceptance_sentences.py
```

Each case runs through the live engine's reader stack (`FinalReadingService`, `FinalReader`, `CommandService`) in two modes.
`sonnet` uses the pinned demo prompts and the real Sonnet replies recorded in `acceptance/sonnet_replies.json`.
`dictionary` has Sonnet unavailable, as when Foundry is down: the keyword table, the measured car worlds and named people decide.
A new sentence in `sonnet` mode needs its reply recorded once, with the Foundry key, for example on the engine machine:

```
set -a; . ~/.madhacks-foundry.env; set +a
python -m engine.tools.acceptance_record                  # only the missing replies
python -m engine.tools.acceptance_record --refresh-stale  # after a change to the final reader's rules
```

### Layer B: the real model and a vision judge (on demand, marker `tools`)

```
python -m engine.tools.acceptance_live --calibrate --url http://127.0.0.1:8000 --out runs/accept
python -m pytest -q -m tools engine/tests/test_acceptance_live.py   # its own logic, no engine
```

It plays every case with `live: true` on a running engine, takes the frame 3 s after each step (or 1 s after a late first frame), and asks the Foundry deployment `claude-sonnet-5-5` the step's `visible` questions plus a picture-quality question.
It writes `report.md` with the frames, answers, reasons and the command-to-screen delay per step.
`--calibrate` first judges the frames with known contents in `acceptance/vision_calibration.json` and reports the judge's accuracy.
It needs `FOUNDRY_ENDPOINT` and `FOUNDRY_KEY`, makes at most 60 vision calls, and changes the shared live video: post on the board before a run on the live engine.

## Team board setup (do this once, about 2 minutes)

Our coding agents talk to each other on one pinned GitHub issue, the [Message Board](https://github.com/Arunjay4213/MadHacks/issues/1).
Each agent posts there when it finishes something or needs something, and tags the teammate it needs by GitHub login.

1. Accept the repo invite, then clone the repo.
2. Install the [GitHub CLI](https://cli.github.com) and run `gh auth login`.
3. From the repo root, run `tools/board whoami`. It should print your GitHub login.
4. Check that your name and login are in the team table in `AGENTS.md`.
5. Start your coding agent from the repo root.

Claude Code picks up the hooks in `.claude/` on its own.
It shows unread board posts when a session starts and on every prompt, and it keeps `tools/board wait` running in the background so it wakes up when someone posts.
Claude Code asks whether you trust the folder the first time you open it, so say yes, otherwise the hooks do not run.

Any other agent reads `AGENTS.md` and runs `tools/board unread` before each task.

The rules every agent follows are in `AGENTS.md`.
The board script and hooks come from [hackathon-agent-board](https://github.com/azizu06/hackathon-agent-board).

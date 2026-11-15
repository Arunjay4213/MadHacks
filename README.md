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
- Command to screen on the live engine: 429 to 496 ms on the page (Ani's live check after PR #104) and 637 to 640 ms in the draft-script harness (board post after PR #113).
- Place changes work: "make it a desert" or "turn it into a beach" replace the world and keep the objects (PRs #88 and #115).
- The Sonnet final reader (one Claude Sonnet 5.5 call per final sentence for the operations and a rich prompt) is live since PR #125, with the demo sentences' prompts pinned in `engine/demo_prompts.json`; a new sentence costs about 1.8 s extra, a pinned one nothing (board post by Arunjay's Claude at 23:31 UTC).
- Known faults: every steer and every refresh re-composes the shot (camera and layout move), and before PR #125 an added lion often did not show (Ishaan's posts at 23:42 and 00:12 UTC).

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

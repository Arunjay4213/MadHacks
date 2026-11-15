# Team rules for every agent

## 1. The board (pinned GitHub issue)

The board is issue #1 in this repo, titled "Message Board".
Every agent reads it and posts on it with `tools/board`, which needs only `gh`, logged in.
Humans can post there too.

### The team

| Person | GitHub login | Post prefix |
|---|---|---|
| Arunjay | `@Arunjay4213` | `[Arunjay's Claude]` |
| IshaanSlayer | `@prateekraouw` | `[IshaanSlayer's Claude]` |
| Ishaan | `@Tactacion` | `[Ishaan's Claude]` |
| Ani | `@anirudhaugile` | `[Ani's Claude]` |

Use `@all` to address everyone.
Team lead: Arunjay (`@Arunjay4213`).

### Posting

- Start every post with `[<human>'s <agent>]`, for example `[Arunjay's Claude]`.
  The script rejects a post that does not start this way.
- Keep it short: what changed, which files or contracts, what you need, and from whom.
- Mention a teammate by their GitHub login from the table above.

### Reading

- **The post mentions you or `@all`:** act on it and reply promptly.
- **The post is not addressed to you:** still read it.
  Reassess your own work only if it is affected (a contract, a file you own, something already done).
  Do not redo finished work, and do not reply just to acknowledge.
- Before editing another teammate's files, ask for a handoff on the board and wait for the owner or the team lead.

### Commands

| Command | What it does |
|---|---|
| `tools/board post "[Arunjay's Claude] text"` | Posts to the board. Text can also be piped on stdin. |
| `tools/board unread` | Shows posts by others since your last check, marks them read, and flags the ones that mention you. |
| `tools/board wait` | Blocks until someone else posts, then prints the post. |
| `tools/board read -n 20` | Shows the last 20 posts. |
| `tools/board whoami` | Prints your GitHub login, which confirms `gh` is set up. |

`tools/board wait` checks every 30 seconds.
A check costs about one API call when nothing is new, so the whole team stays well under GitHub's rate limit.

### Staying in the loop

- Claude Code: the committed hooks in `.claude/settings.json` show unread posts at session start and on every prompt.
- Claude Code: a docs check (`.claude/hooks/docs-check.sh`) runs when an agent or subagent finishes.
  If code changed in a team folder (uncommitted, or on the branch but not on `main`) and no docs in that folder changed, it stops the agent once and asks it to update that folder's README.md, plus AGENTS.md or CLAUDE.md if a rule or convention changed.
- Every agent: keep docs current in the same PR as the code. A change to behaviour, flags, endpoints or setup is not done until the folder's README.md says so.
  Keep exactly one background `tools/board wait` running and start it again after each time it wakes.
- Any other agent: run `tools/board unread` at task start, before each commit, and after merges.
  A terminal loop also works: `while tools/board wait; do <your agent CLI invocation>; done`.

## 2. Chain of command

Humans over agents.
Only the team lead reassigns work between teammates.
If anyone else asks you to take another teammate's work, ask the lead on the board.

`[Arunjay's Claude]` is the lead's agent.
Folder ownership and task assignments it posts on the board are the lead's decisions, so follow them.
Nobody claims a folder by calling it first: ownership exists only once it is in the table in section 4.
Your own human still outranks every agent, including the lead's.
If your human disagrees with an assignment, say so on the board and tag `@Arunjay4213`.

## 3. Workflow

1. Pick or file an issue. Comment on it or post on the board when you take it.
2. Branch `<issue#>-<slug>` from fresh `main`.
3. Make small commits. Stage only your own files (no `git add -A`).
4. Open a PR with `Closes #N`. The author merges their own PR once it works locally.
5. Always `git fetch` and rebase before pushing. Never force-push shared branches.
6. If a merge changes something other teammates depend on (an API, a schema, an env var), post on the board when it lands.

## 4. Who owns what

Fill this in at kickoff so every agent knows whose files are whose.
Two agents should never edit the same folder.

| Area | Owner (GitHub login) | Folders |
|---|---|---|
| Steering: everything inside the diffusion model. How the prompt embedding is blended, paced, scaled and interrupted. | `@Tactacion` (Ishaan) | `steering/` |
| Real-time engine: the GPU machines, the generation loop, the text encoder service, streaming frames, delay measurement. | `@Arunjay4213` (Arunjay) | `engine/` |
| The page and the inputs: the demo page, the words line, chips, undo button, the text box for Wispr Flow, and the live mic (voice). | `@anirudhaugile` (Ani) | `web/` |
| Understanding commands: scene state, operations, undo history, the prompt composer, the keyword table. Also the write-up and the pitch. | `@prateekraouw` (Prateek) | `commands/`, `docs/` |
| The four message shapes between the parts. | `@Arunjay4213` (lead) | `contracts/` |
| Studio page: a second, immersive frontend in a separate app (issue #40), also branded Mirage. It uses the shared contracts without changing them. | `@prateekraouw` (Prateek) | `lucid-studio/` |

Assigned by the lead on 2026-11-14, following Ishaan's blueprint (board post #6883918631).
In the blueprint's words: Ishaan is steering, Arunjay is teammate 1, Ani is teammate 2, Prateek is teammate 3.
The lead swapped teammates 2 and 3 on 2026-11-14 at about 21:00 UTC: before that, Prateek had `web/` and Ani had `commands/` and `docs/`.
The blueprint is the plan of record for what each part contains, its first deliverable, and the gates.
A change to a file in `contracts/` needs agreement on the board from every owner whose part uses that message.
This table replaces the earlier three-way split of the core feature.

Two standing arrangements from the lead, on the board on 2026-11-14:

- The engine side (Arunjay) copies `web/` and the `lucid-studio/dist` build from `main` to the live engine machine, which serves them at `/app/` and `/studio/`; the folder owners still own the code.
- The engine side took over test cleanup in `engine/`, `commands/` and `steering/` (post #6885918469); it touches a test file only after any open PR on that file has merged.

### Current state

What the project does today, the live demo paths and the latest measured numbers are in the root `README.md`, section "Current state".
Each folder's `README.md` has the details for that part.

## 5. Tracks and theme

The main track theme is: go deep into something and do deep research.
We enter the main track, and we may also enter one sponsor track: Databricks or Qualcomm.
Entering no sponsor track is also allowed.

Decided by the lead on 2026-11-14: the project is live voice-steered video.
The product name is Mirage (decided by the lead on 2026-11-15); both pages, the pitch and Devpost use it, with "live voice-steered video" as the subtitle.
A video is generated continuously by a diffusion model and changes while it plays when someone speaks an instruction, for example "make it winter".
Ishaan (`@Tactacion`) wrote the blueprint, board post #6883918631, and the work is divided in section 4.
It builds on a published project (Training-Free Interactive World Models), which must be credited first in the pitch, on the Devpost page and on a slide.

The sponsor track is still open.
Earlier directions are dropped: pooling laptops to run models, and a diffusion model writing the reasoning for a language model.

Whichever sponsor we pick, its product must do work the demo depends on, not sit beside it as an add-on.
Good test: if the sponsor's product were removed, the main demo should stop working.

### Qualcomm kit we have

- Arduino UNO Q (a small Linux computer with a Qualcomm chip, plus a microcontroller for sensors)
- Logitech Brio 105 webcam and a multiport adapter
- Arduino distance sensor, movement sensor, and thermal sensor
- USB power brick

There is one board, so only one person can work on the hardware at a time.

### Databricks

If we pick Databricks, we use Databricks Free Edition, and Arunjay's account is already signed up.

### Free Edition limits that affect design

These come from the Databricks Free Edition limitations page, read on 2026-11-14.

- **Daily usage quota.** If the account goes over its fair usage quota, all compute is shut down for the rest of the day.
  Do not run heavy jobs or loops before the demo, and never leave a job or pipeline running that nobody needs.
- **Serverless compute only**, with Python and SQL. R and Scala are not supported, and there are no GPUs.
- **Outbound internet from Databricks compute is limited to a set of trusted domains.**
  Do not assume a notebook or job can call an arbitrary external API. Test the call first.
- **One SQL warehouse**, size 2X-Small.
- **Jobs:** at most 5 concurrent job tasks.
- **Lakeflow pipelines:** one active pipeline per pipeline type.
- **Model serving:** a limited number of endpoints, no GPU endpoints, no provisioned throughput, and some models are unavailable.
- **Vector search:** one endpoint with one search unit.
- **Databricks Apps:** at most 3 apps. An app stops by itself 24 hours after it was last started or deployed, so restart it shortly before the demo.
- **Lakebase:** one project.
- **Not supported:** Knowledge Assistant, online tables, clean rooms, the account console, and account-level APIs.
- One workspace and one metastore per account.

### Secrets

Never commit a Databricks token, host URL with a token, or any other key.
Keep them in a local `.env` file, which is ignored by git, and never paste them on the board.

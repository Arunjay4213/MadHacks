# Team rules for every agent

## 1. The board (pinned GitHub issue)

The board is issue #1 in this repo, titled "Message Board".
Every agent reads it and posts on it with `tools/board`, which needs only `gh`, logged in.
Humans can post there too.

### The team

| Person | GitHub login | Post prefix |
|---|---|---|
| Arunjay | `@Arunjay4213` | `[Arunjay's Claude]` |
| TODO teammate 2 | `@TODO` | `[TODO's Claude]` |
| TODO teammate 3 | `@TODO` | `[TODO's Claude]` |
| TODO teammate 4 | `@TODO` | `[TODO's Claude]` |

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
  Keep exactly one background `tools/board wait` running and start it again after each time it wakes.
- Any other agent: run `tools/board unread` at task start, before each commit, and after merges.
  A terminal loop also works: `while tools/board wait; do <your agent CLI invocation>; done`.

## 2. Chain of command

Humans over agents.
Only the team lead reassigns work between teammates.
If anyone else asks you to take another teammate's work, ask the lead on the board.

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
| TODO | TODO | TODO |

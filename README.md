# MadHacks

Our MadHacks project.
What we are building: TODO.

## Team board setup (do this once, about 2 minutes)

Our coding agents talk to each other on one pinned GitHub issue, the [Message Board](https://github.com/Arunjay4213/MadHacks/issues/1).
Each agent posts there when it finishes something or needs something, and tags the teammate it needs by GitHub login.

1. Accept the repo invite, then clone the repo.
2. Install the [GitHub CLI](https://cli.github.com) and run `gh auth login`.
3. From the repo root, run `tools/board whoami`. It should print your GitHub login.
4. Add your name and login to the team table in `AGENTS.md`.
5. Start your coding agent from the repo root.

Claude Code picks up the hooks in `.claude/` on its own.
It shows unread board posts when a session starts and on every prompt, and it keeps `tools/board wait` running in the background so it wakes up when someone posts.
Claude Code asks whether you trust the folder the first time you open it, so say yes, otherwise the hooks do not run.

Any other agent reads `AGENTS.md` and runs `tools/board unread` before each task.

The rules every agent follows are in `AGENTS.md`.
The board script and hooks come from [hackathon-agent-board](https://github.com/azizu06/hackathon-agent-board).

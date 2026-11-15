# Development tools from the overnight build

These scripts were written by the lead's agents on the night of 2026-11-14 to 2026-11-15.
They are kept here so they survive the agents' temporary scratch folder.
They are not part of the product, the engine does not import them, and the test suite does not run them.
Several still hold the absolute scratch paths and machine names they ran with (devbox, madhacks-engine, madhacks-test-3); adjust those before reusing them.
None of them holds a key: the ones that call Sonnet read `FOUNDRY_*` from the environment.

| Path | What it does |
|---|---|
| `rtest` | Copies a working tree to devbox and runs pytest there, so the laptop's memory stays free. Usage: `rtest [-C <worktree>] [pytest args]`. |
| `watchdog.sh` | Polls the agents' transcripts, laptop memory and GitHub every 5 minutes and exits with a report when an agent stopped or went quiet, an issue has no PR, or the laptop runs low on memory. |
| `sonnet_free.py` | Sends a list of commands to Sonnet with a plain "write the whole new prompt" system prompt; the experiment that led to the free-scene reader (#273). |
| `effort/` | The thinking and effort measurement: 25 sentences, 8 settings, blind grading (`sonnet_effort.py run`, then `analyze.py`). |
| `regression/` | The regression sweep: builds a corpus of chains from the engine logs, pins and acceptance cases (`build_corpus.py`), replays it through two code versions with real Sonnet (`sweep.py`) and compares them (`analyze.py`). |
| `livefeed/` | Reads the engine journal into records for the team status page's Map (`feed.py`, `RUNBOOK.md`) and the page source (`status.html`). |
| `deck/` | Build script and speaker notes for the pitch deck artifact (`build.py`, `notes.json`, `sync.sh`); images are not included. |

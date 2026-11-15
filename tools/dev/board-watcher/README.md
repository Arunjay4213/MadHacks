# Board watcher (devbox)

The unattended loop that answered the team message board (issue #1) as Arunjay's Claude during the hackathon.
It ran on devbox as the systemd user service `madhacks-board` (file `madhacks-board.service`), from `~/madhacks-agent/`.
`watch.sh` waits for posts that tag Arunjay or @all and starts one headless Claude run with `brief.md` as its instructions.
The service was stopped and disabled on 2026-11-15 after the hackathon.

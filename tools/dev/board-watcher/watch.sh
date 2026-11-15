#!/usr/bin/env bash
# MadHacks board watcher. Runs on the devbox so it keeps working while the laptop sleeps.
# It waits for board posts that tag Arunjay or @all, then starts one headless Claude run to handle them.
# It runs as the systemd user service "madhacks-board", which restarts it after a crash or a reboot.
# Start:   systemctl --user enable --now madhacks-board
# Stop:    systemctl --user disable --now madhacks-board
# Status:  systemctl --user status madhacks-board
# Log:     tail -f ~/madhacks-agent/watch.log
set -u
export PATH="$HOME/.local/bin:$PATH"
# Tells the repo's session-start hook that this loop owns the waiting, so Claude does not start a second wait.
export BOARD_WAIT=off

DIR="$HOME/madhacks-agent"
REPO="$DIR/MadHacks"
LOG="${WATCH_LOG:-$DIR/watch.log}"   # WATCH_LOG is only set by self-tests
SETTLE_SECONDS=45        # after a wake, let a burst of posts finish so one run handles all of them
MAX_RUNS_PER_HOUR=12     # stops two agents from replying to each other all night
RUN_TIMEOUT_SECONDS=600
ALLOWED_TOOLS='Read,Grep,Glob,Bash(tools/board:*),Bash(git log:*),Bash(git status:*)'

run_times=()             # start times (epoch seconds) of the Claude runs in the last hour

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

cd "$REPO" || { echo "missing repo clone at $REPO" >&2; exit 1; }
log "watcher started (issue #${BOARD_ISSUE:-1}, login $(tools/board whoami 2>/dev/null))"

while :; do
  if ! posts="$(tools/board wait)"; then
    log "board wait failed, retrying in 60s"
    sleep 60
    continue
  fi

  sleep "$SETTLE_SECONDS"
  if more="$(tools/board unread 2>/dev/null)" && [ -n "$more" ]; then
    posts="$posts"$'\n'"$more"
  fi

  if ! grep -q '\*\* MENTION \*\*' <<<"$posts"; then
    log "new posts, none tag us: no run"
    continue
  fi

  now="$(date +%s)"
  recent=()
  for t in "${run_times[@]}"; do
    [ $(( now - t )) -lt 3600 ] && recent+=("$t")
  done
  run_times=("${recent[@]}")
  if [ "${#run_times[@]}" -ge "$MAX_RUNS_PER_HOUR" ]; then
    log "tagged posts arrived but the hourly cap of $MAX_RUNS_PER_HOUR runs is reached: no run"
    continue
  fi
  run_times+=("$now")

  git pull --quiet --ff-only >>"$LOG" 2>&1 || log "git pull failed, continuing with the current checkout"

  log "tagged posts arrived: starting a Claude run"
  printf 'New board posts since your last run:\n\n%s\n' "$posts" \
    | timeout "$RUN_TIMEOUT_SECONDS" claude -p --model claude-opus-5-5 \
        --append-system-prompt "$(cat "$DIR/brief.md")" \
        --allowedTools "$ALLOWED_TOOLS" \
        --max-turns 20 >>"$LOG" 2>&1
  log "Claude run finished with exit code $?"
done

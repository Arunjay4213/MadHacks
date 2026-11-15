#!/usr/bin/env bash
# Wakes the lead (by exiting) when a subagent stopped, went quiet, or an issue has no PR.
T=/tmp/claude-1000/-home-arunj-projects-madhacks/8a322351-4ae7-4221-8b46-03960d2c9667/tasks
W=/tmp/claude-1000/-home-arunj-projects-madhacks/8a322351-4ae7-4221-8b46-03960d2c9667/scratchpad/watchdog
AGENTS="$(cat $W/agents)"
touch $W/done
while :; do
  now=$(date +%s); out=""
  for a in $AGENTS; do
    rg -qx "$a" $W/done && continue
    f=$T/$a.output; [ -f "$f" ] || continue
    age=$(( now - $(stat -L -c %Y "$f") ))
    last=$(tail -n 1 "$f")
    if printf '%s' "$last" | rg -q '"stop_reason":"end_turn"' && [ $age -gt 900 ]; then
      out="$out\nSTOPPED $a: $(printf '%s' "$last" | jq -r '[.message.content[]?|select(.type=="text")|.text]|join(" ")' 2>/dev/null | cut -c1-300)"
    fi
    ctx=$(tac "$f" | rg -m1 '"type":"assistant"' | jq -r '.message.usage | (.input_tokens + .cache_read_input_tokens + .cache_creation_input_tokens)' 2>/dev/null)
    if [ -n "$ctx" ] && [ "$ctx" -gt 520000 ] && ! rg -qx "ctx$a" $W/done; then
      out="$out\nCONTEXT $a: $ctx tokens, hand off before 600k"
    fi
    if [ $age -gt 1800 ] && ! printf '%s' "$last" | rg -q '"stop_reason":"end_turn"'; then
      out="$out\nQUIET $a: no transcript activity for $((age/60)) min"
    fi
  done
  # Laptop memory: alert when free memory is low or someone runs pytest locally.
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  [ "$avail" -lt 1200 ] && out="$out\nMEMORY laptop has only ${avail} MB available"
  lp=$(pgrep -af '^[^ ]*(python[0-9.]*|pytest) '| rg -- '-m pytest|/pytest ' | cut -c1-160)
  [ -n "$lp" ] && ! rg -qx "localpytest" $W/done && out="$out\nLOCAL PYTEST on the laptop:\n$lp"
  # Open issues older than 60 min with no open or merged PR that closes them.
  cd /home/arunj/projects/madhacks
  closers=$(gh pr list --state all --limit 60 --json body --jq '.[].body' 2>/dev/null | rg -io 'closes #[0-9]+' | rg -o '[0-9]+' | sort -u)
  for n in $(gh issue list --state open --limit 50 --json number,createdAt --jq ".[] | select(.number != 1) | select((now - (.createdAt|fromdate)) > 3600) | .number" 2>/dev/null); do
    printf '%s\n' "$closers" | rg -qx "$n" || rg -qx "issue$n" $W/done || out="$out\nORPHAN issue #$n has no PR after 60 min"
  done
  if [ -n "$out" ]; then echo "WATCHDOG $(date -u +%H:%M) UTC:"; printf "%b\n" "$out"; exit 0; fi
  sleep 300
done

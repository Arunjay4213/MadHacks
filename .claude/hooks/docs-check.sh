#!/usr/bin/env bash
# Stop / SubagentStop: if this branch or the working tree changed code in a team folder
# but not that folder's docs, block once and ask the agent to update the docs.
# Blocks at most once per distinct set of changed files per session. Never fails.
cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
input="$(cat)"
[ "$(printf '%s' "$input" | jq -r '.stop_hook_active // false' 2>/dev/null)" = "true" ] && exit 0
session="$(printf '%s' "$input" | jq -r '.session_id // "none"' 2>/dev/null)"

# Files changed: uncommitted (staged, unstaged, untracked) plus commits on this branch that main does not have.
base="$(git merge-base HEAD origin/main 2>/dev/null)"
changed="$( {
  git status --porcelain --untracked-files=all 2>/dev/null | sed 's/^...//; s/.* -> //'
  [ -n "$base" ] && git diff --name-only "$base" HEAD 2>/dev/null
} | sort -u)"
[ -z "$changed" ] && exit 0

folders=""
for f in engine commands steering web lucid-studio contracts; do
  code="$(printf '%s\n' "$changed" | rg "^$f/" | rg -v '/tests?/|\.test\.|/samples/|/screenshots/|\.md$|\.(png|jpe?g|gif|mp4|webm)$')"
  [ -z "$code" ] && continue
  # docs/ belongs to the commands/ owner, so it counts as docs only for commands/.
  pat="^$f/.*\.md$"; [ "$f" = commands ] && pat="$pat|^docs/"
  docs="$(printf '%s\n' "$changed" | rg "$pat")"
  [ -z "$docs" ] && folders="$folders $f/"
done
[ -z "$folders" ] && exit 0

stamp="$(git rev-parse --git-dir 2>/dev/null)/docs-check-$session"
key="$(printf '%s' "$changed" | sha1sum | cut -c1-16)"
[ "$(cat "$stamp" 2>/dev/null)" = "$key" ] && exit 0
printf '%s' "$key" > "$stamp" 2>/dev/null

reason="Docs check: you changed code in${folders} but no docs there. Before you finish, update what the change made out of date:
- that folder's README.md (behaviour, flags, endpoints, commands, setup, measured numbers);
- AGENTS.md if ownership, workflow, board rules or team decisions changed;
- CLAUDE.md if a convention every agent should follow changed.
Edit only docs in folders you own; for someone else's folder, ask on the board. One sentence per line, plain English. If nothing user-visible changed, say so in one line and stop."
jq -n --arg r "$reason" '{decision: "block", reason: $r}'
exit 0

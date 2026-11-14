#!/usr/bin/env bash
# Print the trycloudflare address of the running tunnel unit, from its journal.
# Only the current run of the unit is searched, so an address from before a restart is never printed.
# Usage: tunnel-address.sh [unit] [seconds to wait for the address, default 30]
set -euo pipefail
unit="${1:-madhacks-tunnel}"
wait_s="${2:-30}"
for ((i = 0; i <= wait_s; i++)); do
    id="$(systemctl show -p InvocationID --value "$unit")"
    if [[ -n "$id" ]]; then
        addr="$(journalctl --no-pager -o cat "_SYSTEMD_INVOCATION_ID=$id" \
            | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -n 1 || true)"
        if [[ -n "$addr" ]]; then
            echo "$addr"
            exit 0
        fi
    fi
    sleep 1
done
echo "no trycloudflare address for $unit (is it running? systemctl status $unit)" >&2
exit 1

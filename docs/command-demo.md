# Command demo and rehearsal

This is the script to rehearse after engine integration. It does not claim that the
GPU or voice path has passed. The command subsystem can be checked on CPU; visual
quality, strength, pace and spoken-word latency need the real model and browser.

## Three starting scenes

Use the default meadow first. Try the beach and city only after rehearsing them on
the same model, settings and machine used for judging.

1. A wide shot of a meadow with a single oak tree.
2. A wide shot of a quiet sandy beach beside the ocean.
3. A wide shot of a city square with a fountain.

## Three-minute run

| Time | Command / action | Check before saying it works |
|---|---|---|
| 0:00 | Start at the meadow | Moving video is coming from the real engine; show mock status if using a mock. |
| 0:15 | “Make it winter.” | Winter appears without a cut; a winter chip appears. |
| 0:30 | “Also add a river.” | River appears and winter stays. |
| 0:45 | “Slowly turn to night.” | Three chips; the transition is visibly slower than the calibrated default. |
| 1:00 | “A lot more snow.” | Snow is visibly heavier. State strength is wording, not a verified model blend weight. |
| 1:10 | “Remove the river.” | Winter and night remain; only river is removed. |
| 1:20 | “Go back.” | River returns: undo also reverses a removal. |
| 1:30 | Let someone outside the team speak | Unknown commands either reach the configured slow reader or are shown as unmatched. |
| 2:15 | Try “make it winter” on the live mic | Early steering is provisional; only describe it as faster after measuring it. |
| 2:30 | Show credits and explain our contribution | Name the reused model and method before describing our command/state work. |
| 2:50 | Return to the live video | Do not describe a backup recording as live. |

Use the text field for a Wispr Flow command only after checking its actual paste
behavior on the demo laptop. Interim speech and keyword timing use the browser's
live mic path; a finished Wispr paste does not supply word-by-word input.

## Rehearsal record

Run the script five times, including two runs with speakers outside the team.
For each run record machine/model/settings, typed or mic input, visible failures,
and whether reset recovered. Save a backup recording with an explicit recording
label. Do not replace this checklist with a claim that a rehearsal happened.

- Check three stacked changes, remove the middle change, and undo three times.
- Check a wrong provisional guess followed by revised/final speech: no lasting unwanted change.
- Check unsupported text and an LLM timeout: committed state remains intact.
- Check disconnect and reconnect with Ani: no stale commands are replayed.
- Check reset and then undo reset: the prior state returns.
- Calibrate ramp lengths on the actual engine. Current values count model calls,
  and are starting settings rather than measured durations.
- Read latency as the first correlated frame drawn by the browser, a proxy for
  response rather than proof of the first semantically changed pixel.

If the live engine fails, reset once. If it remains unavailable, use the labeled
backup recording and describe the failure plainly. Feature freeze and submission
timing follow the gates in the root `PLAN.md`.

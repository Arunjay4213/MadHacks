"""Record the real Sonnet replies the acceptance table needs (acceptance/sonnet_replies.json).

    set -a; . ~/.madhacks-foundry.env; set +a
    python -m engine.tools.acceptance_record            # record only the replies that are missing
    python -m engine.tools.acceptance_record --case NAME # only this case (repeatable)
    python -m engine.tools.acceptance_record --refresh-stale  # also re-ask replies made under another system prompt

Runs every case in `sonnet` mode with the real Foundry client behind the same stand-in the test uses: a reply already
in the file is replayed, a missing first reply is asked for once and saved, and a retry is refused (the live engine
never retries within its 3 s wait). A saved reply is never asked for again, so the test stays deterministic; delete an
entry from the file to record it afresh.
Each reply keeps the fingerprint of the final reader's system prompt and model it was made under; after a change to
the rules, `--refresh-stale` asks those sentences again, so the table checks what the live engine would now do. Calls are paced (Azure returns 429 after about 20 calls in 40 s).
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone

from engine.final_reader import fingerprint, load
from engine.foundry import FoundryMessages, settings_from_env
from engine.tools.acceptance import (REPLIES_FILE, Mode, MissingReply, ReplayChat, RetryRefused, case_modes,
                                     load_cases, reply_key, run_case)

MIN_GAP_S = 2.0      # between two real calls
HTTP_TIMEOUT_S = 30.0


class RecordingChat(ReplayChat):
    """Replays saved replies; asks the real model for a missing first reply and keeps it."""

    def __init__(self, replies: dict, real: FoundryMessages):
        super().__init__(replies)
        self.real = real
        self.model = real.model
        self.new: dict = {}
        self._last = 0.0

    def complete(self, system: str, messages: list[dict]) -> str:
        if len(messages) > 1:
            raise RetryRefused("no retry on the live timing")
        key = reply_key(json.loads(messages[-1]["content"]))
        if key in self.replies:
            return self.replies[key]
        time.sleep(max(0.0, self._last + MIN_GAP_S - time.monotonic()))
        self._last = time.monotonic()
        try:
            text = self.real.complete(system, messages)
        except Exception as exc:  # noqa: BLE001 - reported, not saved
            print(f"  call failed for {key[1]!r}: {exc!r}", file=sys.stderr)
            raise MissingReply(key[1]) from exc
        print(f"  recorded {key[1]!r}")
        self.replies[key] = text
        self.new[key] = text
        return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append", help="only this case (repeatable)")
    ap.add_argument("--refresh-stale", action="store_true",
                    help="ask again for replies recorded under another system prompt or model")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    settings = settings_from_env()
    if settings is None:
        print("error: FOUNDRY_ENDPOINT and FOUNDRY_KEY must be set (load ~/.madhacks-foundry.env)", file=sys.stderr)
        return 2
    real = FoundryMessages.from_settings(settings, HTTP_TIMEOUT_S)
    fp = fingerprint(load("sonnet", chat=real, cache_path=None, demo_prompts=None).system_prompt, real.model)
    existing = json.loads(REPLIES_FILE.read_text())["replies"] if REPLIES_FILE.exists() else []
    if args.refresh_stale:
        stale = [e for e in existing if e.get("fingerprint") != fp]
        existing = [e for e in existing if e.get("fingerprint") == fp]
        print(f"{len(stale)} replies were made under another system prompt or model; asking again")
    chat = RecordingChat({(e["prompt_before"], e["sentence"]): e["reply"] for e in existing}, real)
    for case in load_cases():
        if args.case and case["name"] not in args.case or Mode.SONNET not in case_modes(case):
            continue
        print(case["name"])
        run_case(case, Mode.SONNET, chat)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing += [{"sentence": s, "prompt_before": p, "reply": r, "model": real.model, "fingerprint": fp,
                  "recorded_at": now}
                 for (p, s), r in chat.new.items()]
    REPLIES_FILE.write_text(json.dumps({"about": "First Sonnet replies of the final reader, recorded by "
                                                 "engine/tools/acceptance_record.py; keyed by the composed prompt "
                                                 "before the sentence and the sentence as sent to the model.",
                                        "replies": existing}, indent=1) + "\n")
    print(f"{len(chat.new)} new replies, {len(existing)} in {REPLIES_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Live commands feeder for the MadHacks team status page (db collection `live_commands`).

Reads the engine logs and prints JSON for the newest FINAL commands (20 by default), oldest first.
Keystroke partials (logged "DROPPED (no steer for partial)" or steered "provisional") are left out, and so are
commands with no sentence (a reset tap or a POST /target).

Sources:
  live      ssh madhacks-engine 'journalctl -u madhacks-engine --since -30min -o cat'
  fullsize  ssh madhacks-test-3 'tail -n 20000 ~/runs/fullsize.log'   (skipped with a warning if unreachable)

The logs print only the time of day, in UTC. It is combined with today's UTC date, or yesterday's when that
time of day is still in the future.

Record shape (one db document per record, doc id = id):
  id            "c" + engine + command number + "-" + HHMMSS of HEARD (the suffix keeps ids unique when an
                engine restarts and its command numbers start again at 1)
  engine        "live" | "fullsize"
  at            ISO time in UTC of HEARD
  sentence      the final sentence, at most 120 characters
  path          PINNED      a pinned demo sentence, a saved Sonnet answer or a measured car world: no Sonnet call
                DICTIONARY  the keyword table decided (undo and reset, a word-list block, or Sonnet had no
                            usable answer in time)
                SONNET      Claude Sonnet on Azure Foundry decided
  outcome       SHOWN | NOT_UNDERSTOOD | BLOCKED
  ms_to_screen  the "GENERATED -> SHOWN at +Y ms" number, or null
  sonnet_ms     time of the Sonnet call(s) for this command, or null
  note          short reason for NOT_UNDERSTOOD and BLOCKED (and for a Sonnet fallback); null otherwise

Usage:
  python3 feed.py                          # JSON array of records
  python3 feed.py --writes stored.json     # ArtifactData batch `writes` for the refresh, where stored.json maps
                                           # each stored doc id to its version: {"clive33-034524": 1, ...}
  python3 feed.py --log live=FILE          # read a saved log instead of ssh (repeatable; for checks)
"""

import argparse
import ast
import dataclasses
import datetime as dt
import enum
import json
import re
import subprocess
import sys

COLLECTION = "live_commands"
SENTENCE_MAX = 120
LOG_CHARS = 100  # engine/final_reader.py shortens sentences in its log lines to this many characters

SOURCES = {
    "live": ("madhacks-engine", "journalctl -u madhacks-engine --since {since} -o cat"),
    "fullsize": ("madhacks-test-3", "tail -n 20000 ~/runs/fullsize.log"),
}
# Sentence keys of the pinned demo prompts and of the saved Sonnet answers on one engine machine.
KNOWN_SENTENCES_CMD = r'''python3 - <<'EOF'
import json, os
keys = set()
def key(s):
    s = " ".join(s.lower().split())
    return s.strip(" .,!?;:'\"").strip()
for path, pick in (("~/madhacks/engine/demo_prompts.json", lambda d: d),
                   ("~/.cache/madhacks/final_reader.json", lambda d: d.get("entries", []))):
    try:
        for e in pick(json.load(open(os.path.expanduser(path)))):
            if isinstance(e, dict) and isinstance(e.get("sentence"), str):
                keys.add(key(e["sentence"]))
    except Exception:
        pass
print(json.dumps(sorted(keys)))
EOF'''


class Path(enum.Enum):
    PINNED = "PINNED"
    DICTIONARY = "DICTIONARY"
    SONNET = "SONNET"


class Outcome(enum.Enum):
    SHOWN = "SHOWN"
    NOT_UNDERSTOOD = "NOT_UNDERSTOOD"
    BLOCKED = "BLOCKED"


class CommandKind(enum.Enum):
    UNKNOWN = "UNKNOWN"  # nothing yet says whether it is a partial or a final
    PARTIAL = "PARTIAL"
    FINAL = "FINAL"


class ReaderEvent(enum.Enum):
    """One final reader log line, tied to the command it was about."""
    ATTEMPT = "ATTEMPT"              # "'<s>' reply <verdict> in Z ms", a reply that was not JSON, or a failed call
    NO_USABLE = "NO_USABLE"          # "no usable reply for '<s>': <reasons>"
    NO_ANSWER = "NO_ANSWER"          # "no answer for '<s>' within 3.0 s": the keyword table decides
    MEASURED = "MEASURED"            # "'<s>' -> measured car world '<w>', no call"
    WORDLIST_BLOCK = "WORDLIST_BLOCK"  # "blocked '<s>' (word list: ...), no call"
    MODEL_BLOCK = "MODEL_BLOCK"      # "the model blocked '<s>'"
    MODEL_NOT_UNDERSTOOD = "MODEL_NOT_UNDERSTOOD"  # "the model did not understand '<s>'"


@dataclasses.dataclass
class Command:
    engine: str
    run: int
    n: int
    heard: str                 # HH:MM:SS.mmm
    text: str | None
    kind: CommandKind = CommandKind.UNKNOWN
    stage: str = "HEARD"
    reason: str = ""           # reason of the last transition, e.g. "no operation for final"
    shown_ms: int | None = None
    events: list = dataclasses.field(default_factory=list)  # (ReaderEvent, detail)
    sonnet_ms: int = 0
    asked_before: bool = False  # this engine run already sent this sentence to Sonnet (a later repeat is cached)


LINE = re.compile(r"^(?P<t>\d\d:\d\d:\d\d\.\d{3}) \S+ \S+ (?P<logger>[\w.]+): (?P<msg>.*)$")
HEARD = re.compile(r"^command (\d+): HEARD (\w+) (.*)$")
MOVE = re.compile(r"^command (\d+): ([A-Z]+) -> ([A-Z]+)(?: at \+(\d+) ms)?(?: \((.*)\))?$")
S = r"""('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")"""
READER = [
    (re.compile(rf"^{S} reply \w+ in (\d+) ms$"), ReaderEvent.ATTEMPT),
    (re.compile(rf"^the reply for {S} is not JSON \((\d+) ms\)"), ReaderEvent.ATTEMPT),
    (re.compile(rf"^call failed for {S} after (\d+) ms"), ReaderEvent.ATTEMPT),
    (re.compile(rf"^no usable reply for {S}: (.*)$"), ReaderEvent.NO_USABLE),
    (re.compile(rf"^no answer for {S} within ([\d.]+) s"), ReaderEvent.NO_ANSWER),
    (re.compile(rf"^{S} -> measured car world {S}, no call$"), ReaderEvent.MEASURED),
    (re.compile(rf"^blocked {S} \(word list: .*\), no call$"), ReaderEvent.WORDLIST_BLOCK),
    (re.compile(rf"^the model blocked {S}$"), ReaderEvent.MODEL_BLOCK),
    (re.compile(rf"^the model did not understand {S}$"), ReaderEvent.MODEL_NOT_UNDERSTOOD),
]
FINAL_REASONS = ("no operation for final", "no steer for final", "controller: final", "controller: confirm",
                 "new video starts on it")
PARTIAL_REASONS = ("no steer for partial", "controller: provisional")
# Plain words for the reasons engine/final_reader.py gives when it rejects a reply.
REASON_WORDS = [
    (re.compile(r"^no operations$"), "Sonnet found nothing to change"),
    (re.compile(r"base names its own world"), "base names its own world"),
    (re.compile(r"base names its own place"), "base names its own place"),
]


def short(text: str, n: int = LOG_CHARS) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def sentence_key(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .,!?;:'\"").strip()


def unrepr(s: str):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s.strip("'\"")


def parse(engine: str, lines) -> list[Command]:
    """Every command in one engine's log, in the order heard."""
    cmds: dict[tuple[int, int], Command] = {}
    order: list[Command] = []
    run, last_n = 0, 0
    asked: set[tuple[int, str]] = set()  # (run, sentence key) sent to Sonnet so far
    for raw in lines:
        m = LINE.match(raw.rstrip("\n"))
        if not m:
            continue
        t, logger, msg = m["t"], m["logger"], m["msg"]
        if logger == "engine.commands":
            h = HEARD.match(msg)
            if h:
                n = int(h[1])
                if n <= last_n:
                    run += 1  # the engine restarted: its command numbers start again
                last_n = n
                text = unrepr(h[3]) if h[3] != "None" else None
                c = Command(engine, run, n, t, text if isinstance(text, str) else None)
                if c.text is not None and (run, sentence_key(c.text)) in asked:
                    c.asked_before = True
                cmds[(run, n)] = c
                order.append(c)
                continue
            mv = MOVE.match(msg)
            if mv and (run, int(mv[1])) in cmds:
                c = cmds[(run, int(mv[1]))]
                c.stage, c.reason = mv[3], mv[5] or ""
                if any(c.reason.startswith(r) for r in FINAL_REASONS):
                    c.kind = CommandKind.FINAL
                elif any(c.reason.startswith(r) for r in PARTIAL_REASONS):
                    c.kind = CommandKind.PARTIAL
                if mv[3] == "SHOWN" and mv[4]:
                    c.shown_ms = int(mv[4])
            continue
        if logger in ("engine.final_reader", "engine.reader") and msg.startswith("final reader: "):
            body = msg[len("final reader: "):]
            for rx, ev in READER:
                r = rx.match(body)
                if not r:
                    continue
                said = unrepr(r[1])
                # The command still being read (HEARD, not yet DECIDED or DROPPED) with this sentence.
                owner = next((c for c in reversed(order[-80:]) if c.stage == "HEARD" and c.text is not None
                              and short(c.text) == said), None)
                if owner is None:
                    break
                detail = r[2] if r.lastindex and r.lastindex >= 2 else ""
                owner.events.append((ev, detail))
                owner.kind = CommandKind.FINAL
                if ev is ReaderEvent.ATTEMPT:
                    owner.sonnet_ms += int(detail)
                    asked.add((run, sentence_key(owner.text)))
                break
    return order


def plain_reason(reasons: str) -> str:
    first = reasons.split(";")[0].strip()
    for rx, words in REASON_WORDS:
        if rx.search(first):
            return words
    return first


def clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def to_record(c: Command, known: set[str], day_of) -> dict | None:
    """The page record for one command, or None when it is not a finished final with a sentence."""
    if c.text is None or c.kind is not CommandKind.FINAL:
        return None
    kinds = {ev for ev, _ in c.events}
    detail = {ev: d for ev, d in c.events}
    called = ReaderEvent.ATTEMPT in kinds

    if ReaderEvent.WORDLIST_BLOCK in kinds or ReaderEvent.MODEL_BLOCK in kinds:
        outcome = Outcome.BLOCKED
    elif c.stage == "SHOWN":
        outcome = Outcome.SHOWN
    elif c.stage == "DROPPED" and c.reason in ("no operation for final", "no steer for final"):
        outcome = Outcome.NOT_UNDERSTOOD
    else:
        return None  # still on its way, superseded, unchanged or dropped by a reset: not a result to show

    note = None
    if ReaderEvent.MEASURED in kinds:
        path = Path.PINNED
    elif ReaderEvent.WORDLIST_BLOCK in kinds:
        path, note = Path.DICTIONARY, "Blocked by the word list before any model call"
    elif ReaderEvent.MODEL_BLOCK in kinds:
        path, note = Path.SONNET, "Sonnet declined this request"
    elif ReaderEvent.MODEL_NOT_UNDERSTOOD in kinds and outcome is Outcome.NOT_UNDERSTOOD:
        path, note = Path.SONNET, "Sonnet did not understand it"
    elif ReaderEvent.NO_ANSWER in kinds:
        path, note = Path.DICTIONARY, "Sonnet took longer than 3 s, so the keyword table decided"
        if outcome is Outcome.NOT_UNDERSTOOD:
            note = "Sonnet took longer than 3 s and the keyword table found nothing to change"
    elif ReaderEvent.NO_USABLE in kinds:
        why = plain_reason(detail[ReaderEvent.NO_USABLE])
        if outcome is Outcome.SHOWN:
            lead = why if why.startswith("Sonnet") else "Sonnet reply rejected (" + why + ")"
            path, note = Path.DICTIONARY, lead + ", so the keyword table decided"
        else:
            path, note = Path.SONNET, why if why.startswith("Sonnet") else "Sonnet reply rejected: " + why
    elif called:
        path = Path.SONNET
        if outcome is Outcome.NOT_UNDERSTOOD:
            note = "Sonnet's change did not fit the scene"
    elif sentence_key(c.text) in known or c.asked_before:
        path = Path.PINNED
        if outcome is Outcome.NOT_UNDERSTOOD:
            note = "The saved answer had nothing to change in this scene"
    else:
        path = Path.DICTIONARY
        if outcome is Outcome.NOT_UNDERSTOOD:
            note = "The keyword table found nothing to change"

    hhmmss = c.heard[:8].replace(":", "")
    return {
        "id": f"c{c.engine}{c.n}-{hhmmss}",
        "engine": c.engine,
        "at": day_of(c.heard) + "T" + c.heard + "Z",
        "sentence": clip(c.text.strip(), SENTENCE_MAX),
        "path": path.value,
        "outcome": outcome.value,
        "ms_to_screen": c.shown_ms if outcome is Outcome.SHOWN else None,
        "sonnet_ms": c.sonnet_ms if called else None,
        "note": clip(note, 110) if note else None,
    }


def ssh(host: str, command: str, timeout: int = 40) -> str:
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, command],
                         capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"ssh {host} failed ({out.returncode}): {out.stderr.strip()[:200]}")
    return out.stdout


def day_resolver(now: dt.datetime):
    today, yesterday = now.date(), now.date() - dt.timedelta(days=1)
    now_s = now.strftime("%H:%M:%S")

    def day_of(t: str) -> str:
        # A time of day more than 10 minutes ahead of now was logged yesterday (UTC).
        later = t[:8] > now_s and (dt.datetime.combine(today, dt.time.fromisoformat(t[:8]),
                                                       tzinfo=dt.timezone.utc) - now).total_seconds() > 600
        return (yesterday if later else today).isoformat()
    return day_of


def batch_writes(records: list[dict], stored: dict[str, int]) -> list[dict]:
    """Set each record not stored yet (a finished command never changes); delete stored ids not in `records`."""
    keep = {r["id"] for r in records}
    writes = [{"op": "set", "collection": COLLECTION, "doc_id": r["id"], "data": r}
              for r in records if r["id"] not in stored]
    writes += [{"op": "delete", "collection": COLLECTION, "doc_id": i, "if_version": v}
               for i, v in sorted(stored.items()) if i not in keep]
    return writes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="-30min", help="journalctl --since for the live engine (default -30min)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--no-fullsize", action="store_true", help="skip the full-size test engine")
    ap.add_argument("--log", action="append", default=[], metavar="ENGINE=FILE",
                    help="read this engine's log from a file instead of ssh")
    ap.add_argument("--writes", metavar="STORED_JSON",
                    help="print ArtifactData batch writes instead; STORED_JSON maps stored doc ids to versions")
    args = ap.parse_args()

    files = dict(x.split("=", 1) for x in args.log)
    engines = [e for e in SOURCES if not (e == "fullsize" and args.no_fullsize)]
    now = dt.datetime.now(dt.timezone.utc)
    day_of = day_resolver(now)
    records = []
    for engine in engines:
        host, cmd = SOURCES[engine]
        try:
            if engine in files:
                with open(files[engine], encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            else:
                text = ssh(host, cmd.format(since=args.since))
            try:
                known = set(json.loads(ssh(host, KNOWN_SENTENCES_CMD)))
            except Exception as exc:  # noqa: BLE001 - pinned list is a hint only
                print(f"feed: {engine}: no pinned or saved sentences ({exc})", file=sys.stderr)
                known = set()
        except Exception as exc:  # noqa: BLE001 - one unreachable engine must not stop the other
            print(f"feed: {engine}: skipped ({exc})", file=sys.stderr)
            continue
        cmds = parse(engine, text.splitlines())
        recs = [r for r in (to_record(c, known, day_of) for c in cmds) if r]
        finals = sum(1 for c in cmds if c.kind is CommandKind.FINAL)
        print(f"feed: {engine}: {len(cmds)} commands heard, {finals} finals, {len(recs)} records", file=sys.stderr)
        records += recs

    records.sort(key=lambda r: r["at"])
    seen, unique = set(), []
    for r in records:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)
    newest = unique[-args.limit:]
    if args.writes:
        with open(args.writes, encoding="utf-8") as fh:
            stored = {str(k): int(v) for k, v in json.load(fh).items()}
        print(json.dumps(batch_writes(newest, stored), indent=1))
    else:
        print(json.dumps(newest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

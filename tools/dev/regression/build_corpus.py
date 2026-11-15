"""Build the replay corpus: chains of steps (each starts with a reset tap on the meadow).

Usage: python build_corpus.py REPO OUT.json LOG [LOG ...]
"""
import ast
import json
import re
import sys
from pathlib import Path

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
logs = sys.argv[3:]
CAR = "new scene of a red sports car driving along a country road"
BASE = "a wide shot of a meadow with a single oak tree"

HEARD = re.compile(r"command (\d+): HEARD control (.*)$")
DROP_PARTIAL = re.compile(r"command (\d+): HEARD -> DROPPED \(no steer for partial\)")
RESET = re.compile(r"lifecycle: \w+ -> RESETTING \(reset requested")
START = re.compile(r"final reader on:")

chains = []  # (source, [steps])


def flush(source, events):
    """events: list of dicts {cmd, text|None, partial, reset}. Split into chains at resets."""
    cur = []
    for e in events:
        if e["partial"]:
            continue
        if e["text"] is None:
            if e["reset"]:
                if cur:
                    chains.append((source, cur))
                cur = []
            else:
                cur.append({"tap": "undo"})
            continue
        cur.append({"say": e["text"]})
    if cur:
        chains.append((source, cur))


for path in logs:
    src = Path(path).name
    events, by_cmd, last_tap = [], {}, None
    for line in open(path, errors="replace"):
        if START.search(line):
            flush(src, events)
            events, by_cmd, last_tap = [], {}, None
            continue
        m = HEARD.search(line)
        if m:
            n, raw = int(m.group(1)), m.group(2).strip()
            text = None if raw == "None" else ast.literal_eval(raw)
            e = {"cmd": n, "text": text, "partial": False, "reset": False}
            events.append(e)
            by_cmd[n] = e
            last_tap = e if text is None else None
            continue
        m = DROP_PARTIAL.search(line)
        if m and int(m.group(1)) in by_cmd:
            by_cmd[int(m.group(1))]["partial"] = True
            continue
        if RESET.search(line) and last_tap is not None:
            last_tap["reset"] = True
            last_tap = None
    flush(src, events)

# Clean: drop empty texts, strip
corpus = []
for src, steps in chains:
    steps = [s for s in steps if "tap" in s or s["say"].strip()]
    # leading undos on a fresh meadow do nothing useful
    while steps and "tap" in steps[0]:
        steps.pop(0)
    if steps:
        corpus.append({"source": "live:" + src, "steps": steps})

# Pinned demo prompts: walk back through canonical prompts to build the chain that reaches prompt_before.
pins = json.loads((repo / "engine/demo_prompts.json").read_text())
by_canon = {}
for p in pins:
    by_canon.setdefault(p["canonical"], p)
car_canon = None
for p in pins:
    pass


def chain_for(p, depth=0):
    if depth > 20:
        return None
    if p["prompt_before"] == BASE:
        return [{"say": p["sentence"]}]
    prev = by_canon.get(p["prompt_before"])
    if prev is None:
        return None
    head = chain_for(prev, depth + 1)
    return None if head is None else head + [{"say": p["sentence"]}]


unreached = []
for i, p in enumerate(pins):
    c = chain_for(p)
    if c is None:
        # the car base is reached by the car sentence (which is itself not pinned on the base, or it is)
        c = [{"say": CAR}, {"say": p["sentence"]}] if p["prompt_before"] != BASE else None
        unreached.append(i)
    corpus.append({"source": f"pin:{i}", "steps": c, "pin_index": i, "pin_prompt_before": p["prompt_before"]})

# Acceptance cases.
cases = json.loads((repo / "acceptance/sentences.json").read_text())["cases"]
for case in cases:
    steps = [{"say": CAR}] if case.get("start") == "car" else []
    for s in case["steps"]:
        if "say" in s:
            steps.append({"say": s["say"]})
        elif s["tap"] == "remove":
            steps.append({"tap": "remove", "chip": s["chip"]})
        else:
            steps.append({"tap": s["tap"]})
    corpus.append({"source": f"acceptance:{case['name']}", "steps": steps, "expect": case["steps"],
                   "start": case.get("start", "meadow")})

# Demo scripts.
for f in sorted((repo / "engine/tools/demo_scripts").glob("*.json")):
    d = json.loads(f.read_text())
    cur = []
    for s in d["steps"]:
        if s.get("reset"):
            if cur:
                corpus.append({"source": f"demo:{f.stem}", "steps": cur})
            cur = []
        elif "say" in s:
            cur.append({"say": s["say"]})
        elif s.get("undo") or s.get("tap") == "undo":
            cur.append({"tap": "undo"})
    if cur:
        corpus.append({"source": f"demo:{f.stem}", "steps": cur})

# Deduplicate: identical chains, and live chains that are a prefix of another chain.
def key(steps):
    return tuple(json.dumps(s, sort_keys=True) for s in steps)

seen, uniq = {}, []
for c in corpus:
    k = key(c["steps"])
    if k in seen:
        seen[k]["also"].append(c["source"])
        continue
    c["also"] = []
    seen[k] = c
    uniq.append(c)
keys = [key(c["steps"]) for c in uniq]
final = []
for c, k in zip(uniq, keys):
    if c["source"].startswith("live:") and any(len(o) > len(k) and o[:len(k)] == k for o in keys):
        continue
    final.append(c)
for i, c in enumerate(final):
    c["id"] = i
out.write_text(json.dumps(final, indent=1))
n_steps = sum(len(c["steps"]) for c in final)
srcs = {}
for c in final:
    s = c["source"].split(":")[0]
    srcs[s] = srcs.get(s, 0) + 1
print(f"{len(final)} chains, {n_steps} steps, by source {srcs}; pins not reachable by canonical walk: {unreached}")

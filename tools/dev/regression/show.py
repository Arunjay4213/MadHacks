"""Print candidates side by side. python show.py corpus main pr [kind,kind] [--chips]"""
import json
import sys

corpus = {c["id"]: c for c in json.load(open(sys.argv[1]))}
m = {c["id"]: c for c in json.load(open(sys.argv[2]))}
r = {c["id"]: c for c in json.load(open(sys.argv[3]))}
kinds = set(sys.argv[4].split(",")) if len(sys.argv) > 4 and sys.argv[4] else None
cmp = json.load(open("compare1.json"))
W = int(__import__("os").environ.get("W", "260"))


def lab(s):
    return s["step"].get("say") or "<" + s["step"].get("tap", "") + ">"


def line(tag, s):
    chips = [c["label"] for c in s.get("chips", [])]
    return (f"  {tag}: und={s.get('understood')} path={s.get('path')} ms={s.get('ms')} tok={s.get('tokens')} "
            f"unm={s.get('unmatched')!r} chips={chips}\n      {(s.get('prompt') or '')[:W]}")


for x in cmp["regs"]:
    if kinds and not (set(x["worse"]) & kinds):
        continue
    cid, i = x["id"], x["i"]
    ms = {s["i"]: s for s in m[cid]["steps"]}
    rs = {s["i"]: s for s in r[cid]["steps"]}
    prev = [lab(ms[j]) for j in range(i) if j in ms]
    print(f"== chain {cid} {corpus[cid]['source']} step {i}: {lab(ms[i])!r}  WORSE {x['worse']}")
    print(f"  before: {prev[-6:]}")
    print(line("main", ms[i]))
    print(line("pr  ", rs[i]))
    if rs[i].get("logs") and "not_understood" in x["worse"]:
        for l in rs[i]["logs"][-3:]:
            print("      log:", l[:300])

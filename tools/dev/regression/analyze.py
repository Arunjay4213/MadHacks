"""Compare sweep results of main and #273.

python analyze.py CORPUS MAIN.json PR.json [MAIN2.json PR2.json ...] --out flags.json
Pairs trial k of main with trial k of #273. Prints candidates (PR worse on a flag in trial 1) and the 2-of-3 verdicts.
"""
import json
import re
import sys
from collections import defaultdict

VIEW = re.compile(r"\b(view|camera|angle|pov|point of view|side|front|zoom|aerial|above|overhead|bird|close|closer|"
                  r"behind|facing|face|perspective|shot|look at|from the)\b", re.I)
BEHIND = re.compile(r"from directly behind|seen from behind|from behind|rear lights", re.I)
CONTRA = [(re.compile(r"\bsingle\b", re.I), re.compile(r"\b(second|two|both|another|pair of)\b", re.I))]


def load(p):
    return {c["id"]: c for c in json.load(open(p))}


def word_set(labels):
    return {w for l in labels for w in re.findall(r"[a-z]+", (l or "").lower()) if len(w) > 2}


def expect_ok(exp, s):
    """Acceptance expectation for one step: list of problems."""
    probs = []
    if exp is None:
        return probs
    labels = [c["label"] or "" for c in s.get("chips", [])]
    want = exp.get("understood", True)
    if "say" in exp and s.get("understood") != want:
        probs.append("understood" if s.get("understood") else "not understood")
    def has_chip(c):
        return any(re.search(rf"(?<![a-z]){re.escape(c.lower())}(?:s|es)?(?![a-z])", l.lower()) for l in labels)
    for c in exp.get("chips", []):
        if not has_chip(c):
            probs.append(f"chip {c!r} missing")
    for c in exp.get("no_chips", []):
        if has_chip(c):
            probs.append(f"chip {c!r} not removed")
    low = (s.get("prompt") or "").lower()
    for w in exp.get("has", []):
        if w.lower() not in low:
            probs.append(f"lacks {w!r}")
    for w in exp.get("lacks", []):
        if w.lower() in low:
            probs.append(f"has {w!r}")
    return probs


def step_flags(chain, steps, corpus_chain):
    """Per step a dict of bad flags for one version's run of a chain."""
    out = {}
    stack = []
    exp_steps = corpus_chain.get("expect")
    offset = 1 if corpus_chain.get("start") == "car" else 0
    for s in steps:
        i = s["i"]
        st = s["step"]
        f = {}
        if s.get("error"):
            f["error"] = s["error"][:200]
        say = st.get("say")
        if say is not None and not s.get("understood"):
            f["not_understood"] = f"unmatched={s.get('unmatched')!r} path={s.get('path')}"
        if s.get("blocked"):
            f["blocked"] = True
        if "timeout" in (s.get("path") or "") or "error" in (s.get("path") or ""):
            f["sonnet_failed"] = s.get("path")
        if s.get("tokens", 0) > 512:
            f["over_512"] = s["tokens"]
        pb, p = s.get("prompt_before") or "", s.get("prompt") or ""
        if say is not None and BEHIND.search(pb) and not BEHIND.search(p) and not VIEW.search(say):
            f["lost_behind_camera"] = True
        for a, b in CONTRA:
            if a.search(p) and b.search(p):
                f["single_vs_many"] = True
        if st.get("tap") == "undo":
            if stack:
                want = stack.pop()
                if p != want:
                    f["undo_not_restored"] = True
        elif p != pb and s.get("steer"):
            stack.append(pb)
        if exp_steps is not None and i - offset >= 0 and i - offset < len(exp_steps):
            probs = expect_ok(exp_steps[i - offset], s)
            if probs:
                f["acceptance"] = probs
        out[i] = f
    return out


def compare(corpus, m, r):
    """Candidate regressions and improvements between one main run and one PR run."""
    regs, imps, same, total = [], [], 0, 0
    for cid, cc in corpus.items():
        if cid not in m or cid not in r:
            continue
        ms, rs = {s["i"]: s for s in m[cid]["steps"]}, {s["i"]: s for s in r[cid]["steps"]}
        mf, rf = step_flags(cc, m[cid]["steps"], cc), step_flags(cc, r[cid]["steps"], cc)
        for i in sorted(set(ms) & set(rs)):
            total += 1
            a, b = ms[i], rs[i]
            fa, fb = mf.get(i, {}), rf.get(i, {})
            worse = {k: v for k, v in fb.items() if k not in fa}
            better = {k: v for k, v in fa.items() if k not in fb}
            if "acceptance" in fa and "acceptance" in fb:
                extra = set(fb["acceptance"]) - set(fa["acceptance"])
                if extra:
                    worse["acceptance"] = sorted(extra)
            # pins: main hit a pin, PR did not or is slow
            if a.get("path") == "pinned" and (b.get("path") != "pinned" or b.get("ms", 0) > 100):
                worse["pin_lost"] = f"pr path={b.get('path')} ms={b.get('ms')}"
            if a.get("path") == "measured" and b.get("path") not in ("measured", "pinned"):
                worse["measured_world_lost"] = f"pr path={b.get('path')}"
            # chips lost (word level), only when PR understood
            mw, rw = word_set(c["label"] for c in a.get("chips", [])), word_set(c["label"] for c in b.get("chips", []))
            lost = mw - rw
            if lost and b.get("understood") and a.get("understood"):
                worse["chip_words_lost"] = sorted(lost)
            if worse:
                regs.append({"id": cid, "i": i, "worse": worse})
            if better and not worse:
                imps.append({"id": cid, "i": i, "better": better})
            if not worse and not better:
                same += 1
    return regs, imps, same, total


if __name__ == "__main__":
    args = sys.argv[1:]
    corpus = {c["id"]: c for c in json.load(open(args[0]))}
    runs = [load(p) for p in args[1:]]
    pairs = list(zip(runs[0::2], runs[1::2]))
    regs, imps, same, total = compare(corpus, *pairs[0])
    print(json.dumps({"total": total, "candidates": len(regs), "improvements": len(imps), "same": same}))
    kinds = defaultdict(int)
    for x in regs:
        for k in x["worse"]:
            kinds[k] += 1
    print("candidate kinds:", dict(kinds))
    ik = defaultdict(int)
    for x in imps:
        for k in x["better"]:
            ik[k] += 1
    print("improvement kinds:", dict(ik))
    json.dump({"regs": regs, "imps": imps, "same": same, "total": total}, open("compare1.json", "w"), indent=1)

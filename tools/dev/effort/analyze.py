"""sheet RESULTS.json SHEET.txt KEY.json   -> blind grading sheet (outputs shuffled per sentence, labelled)
   stats RESULTS.json KEY.json GRADES.json OUT.json -> per-setting latency, tokens, auto checks, mean score"""
import json, random, re, statistics, string, sys

NEG = re.compile(r"\b(no|not|without|never|none|nothing|absent|free of|devoid|lack(s|ing)?|empty of)\b|n't\b", re.I)


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def sheet(res, out, keyp):
    rows = [r for r in json.load(open(res))["rows"] if r["ok"]]
    key, rng, lines = {}, random.Random(11), []
    labels = [a + b for a in string.ascii_uppercase for b in "12"]
    for cid in sorted({r["case"] for r in rows}):
        rs = [r for r in rows if r["case"] == cid]
        rng.shuffle(rs)
        lines.append(f"\n===== {cid} | scene {rs[0]['scene']} | COMMAND: {rs[0]['command']}")
        for lab, r in zip(labels, rs):
            key[f"{cid}:{lab}"] = [r["setting"], r["rep"]]
            wc = len(r["text"].split())
            neg = sorted({m.group(0).lower() for m in NEG.finditer(r["text"])})
            lines.append(f"[{lab}] ({wc}w neg={neg}) {r['text']}")
    open(out, "w").write("\n".join(lines))
    json.dump(key, open(keyp, "w"))


def stats(res, keyp, gradesp, out):
    data = json.load(open(res))
    key, grades = json.load(open(keyp)), json.load(open(gradesp))
    score = {}
    for k, v in grades.items():
        st, rep = key[k]
        score[(k.split(":")[0], st, rep)] = v if isinstance(v, dict) else {"score": v, "fails": [], "note": ""}
    summary = {}
    for st in dict.fromkeys(r["setting"] for r in data["rows"]):
        rs = [r for r in data["rows"] if r["setting"] == st]
        ok = [r for r in rs if r["ok"]]
        lat = [r["latency_s"] for r in ok]
        gs = [score[(r["case"], st, r["rep"])] for r in ok if (r["case"], st, r["rep"]) in score]
        sc = [g["score"] for g in gs]
        fails = {}
        for g in gs:
            for f in g["fails"]:
                fails[f] = fails.get(f, 0) + 1
        summary[st] = {
            "calls": len(rs), "errors": len(rs) - len(ok),
            "p50_s": round(pct(lat, .5), 2), "p90_s": round(pct(lat, .9), 2), "max_s": round(max(lat), 2),
            "over_3s": sum(x > 3.0 for x in lat),
            "mean_output_tokens": round(statistics.mean(r["usage"]["output_tokens"] for r in ok)),
            "share_with_thinking": round(sum(r["thinking_blocks"] > 0 for r in ok) / len(ok), 2),
            "over_120_words": sum(len(r["text"].split()) > 120 for r in ok),
            "with_negation_word": sum(bool(NEG.search(r["text"])) for r in ok),
            "mean_score": round(statistics.mean(sc), 2) if sc else None,
            "scores_below_4": sum(x < 4 for x in sc),
            "fails_by_check": dict(sorted(fails.items())),
        }
    for r in data["rows"]:
        g = score.get((r["case"], r["setting"], r["rep"]))
        r["score"], r["fails"], r["grade_note"] = (g["score"], g["fails"], g["note"]) if g else (None, None, None)
    data["summary"] = summary
    json.dump(data, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    {"sheet": sheet, "stats": stats}[sys.argv[1]](*sys.argv[2:])

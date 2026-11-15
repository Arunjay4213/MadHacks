"""Replay corpus chains through one code copy's live reader stack (load_reader('commands', final reader sonnet)).

Run from inside the code copy:  cd ~/regsweep/main && python ../sweep.py CORPUS OUT [--only 1,2] [--upto id:step,...]
Needs FOUNDRY_* in the environment. CPU only.
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("COMMAND_SEMANTIC", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
GAP_S = float(os.environ.get("SWEEP_GAP_S", "3.0"))  # pause after a step that called Sonnet (Azure 429s at ~0.5 call/s)
TOK_DIR = str(Path.home() / "weights/Wan2.1-T2V-1.3B/google/umt5-xxl")

_tok = None
_records = []


class Grab(logging.Handler):
    def emit(self, record):
        try:
            _records.append(f"{record.name}: {record.getMessage()}"[:700])
        except Exception:
            pass


def tokens(text):
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOK_DIR)
    return len(_tok(text, add_special_tokens=True)["input_ids"])


def run_chain(arg):
    chain, upto = arg
    sys.path.insert(0, os.getcwd())
    from engine.final_reader import load
    from engine.reader import load_reader
    repo = Path(os.getcwd())
    wait = {"timeout_s": float(os.environ["SWEEP_TIMEOUT_S"])} if os.environ.get("SWEEP_TIMEOUT_S") else {}
    fr = load("sonnet", cache_path=None, demo_prompts=str(repo / "engine/demo_prompts.json"), **wait)
    svc = load_reader("commands", fr)
    steps_out = []
    t = 1
    out = svc.handle({"type": "tap", "action": "reset", "t_client": t})
    state = out.get("state", {})
    last_sentence_at = 0.0
    steps = chain["steps"] if upto is None else chain["steps"][: upto + 1]
    for i, step in enumerate(steps):
        t += 1
        before = fr.stats()
        prompt_before = svc.committed_prompt
        if "say" in step:
            msg = {"type": "final", "text": step["say"], "t_client": t}
        else:
            msg = {"type": "tap", "action": step["tap"], "t_client": t}
            if step["tap"] == "remove":
                import re
                want = step["chip"].lower()
                ids = [c["id"] for c in state.get("changes", [])
                       if re.search(rf"(?<![a-z]){re.escape(want)}(?:s|es)?(?![a-z])", c["label"].lower())]
                if not ids:
                    steps_out.append({"i": i, "step": step, "error": f"no chip {want!r} to remove",
                                      "prompt": prompt_before, "chips": [], "understood": False})
                    continue
                msg["id"] = ids[0]
        del _records[:]
        t0 = time.monotonic()
        try:
            out = svc.handle(msg)
        except Exception as exc:  # noqa: BLE001
            steps_out.append({"i": i, "step": step, "error": "raised " + repr(exc)[:300],
                              "trace": traceback.format_exc()[-1500:], "prompt": prompt_before,
                              "chips": [], "understood": False})
            continue
        ms = (time.monotonic() - t0) * 1000
        after = fr.stats()
        if after.get("calls", 0) > before.get("calls", 0):
            time.sleep(GAP_S)
        d = {k: after[k] - before[k] for k in after if isinstance(after.get(k), int) and isinstance(before.get(k), int)
             and not isinstance(after.get(k), bool) and after[k] != before[k]}
        steer = out.get("steer")
        state = out.get("state", {}) or state
        prompt = steer["prompt"] if steer else svc.committed_prompt
        unmatched = state.get("unmatched")
        understood = steer is not None and (not unmatched if "say" in step else True)
        if d.get("pinned_hits"):
            path = "pinned"
        elif d.get("template_hits"):
            path = "measured"
        elif d.get("cache_hits"):
            path = "cache"
        elif d.get("calls"):
            path = "sonnet" + ("+timeout" if d.get("timeouts") else "") + ("+error" if d.get("errors") else "")
        elif d.get("blocked"):
            path = "blocked-list"
        elif "tap" in step:
            path = "tap"
        else:
            path = "dictionary"
        logs = [r for r in _records if "final reader" in r or "engine.reader" in r]
        steps_out.append({
            "i": i, "step": step, "understood": understood, "unmatched": unmatched,
            "blocked": bool(d.get("blocked")) or any("blocked" in r and "no steer" in r for r in logs),
            "not_understood_model": bool(d.get("not_understood")),
            "chips": [{"label": c.get("label"), "strength": c.get("strength")} for c in state.get("changes", [])],
            "prompt": prompt, "prompt_before": prompt_before, "tokens": tokens(prompt) if prompt else 0,
            "ms": round(ms), "path": path, "stats_delta": d, "steer": steer is not None,
            "ramp": steer.get("ramp_chunks") if steer else None, "logs": logs[-8:],
        })
    tainted = any("error" in (x.get("path") or "") or any("429" in l for l in x.get("logs", [])) for x in steps_out)
    return {"id": chain["id"], "source": chain["source"], "steps": steps_out, "tainted": tainted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    ap.add_argument("--upto", default="", help="id:step,... run only to that step")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    chains = json.loads(Path(a.corpus).read_text())
    upto = {int(x.split(":")[0]): int(x.split(":")[1]) for x in a.upto.split(",") if x}
    only = {int(x) for x in a.only.split(",") if x} | set(upto)
    if only:
        chains = [c for c in chains if c["id"] in only]
    logging.basicConfig(level=logging.WARNING)
    root = logging.getLogger()
    root.handlers[0].setLevel(logging.ERROR)
    root.addHandler(Grab(level=logging.INFO))
    for name in ("engine", "commands"):
        logging.getLogger(name).setLevel(logging.INFO)
    # longest chains first so the pool finishes evenly
    chains.sort(key=lambda c: -len(c["steps"]))
    results = []
    t0 = time.time()
    todo = chains
    for attempt in range(4):
        again = []
        with mp.get_context("fork").Pool(a.workers) as pool:
            for n, r in enumerate(pool.imap_unordered(run_chain, [(c, upto.get(c["id"])) for c in todo])):
                if r["tainted"] and attempt < 3:
                    again.append(next(c for c in todo if c["id"] == r["id"]))
                else:
                    r["attempts"] = attempt + 1
                    results.append(r)
                if n % 10 == 0:
                    print(f"attempt {attempt}: {n + 1}/{len(todo)} chains, {time.time() - t0:.0f} s, "
                          f"{len(again)} to redo", flush=True)
                    Path(a.out).write_text(json.dumps(sorted(results, key=lambda r: r["id"]), indent=0))
        if not again:
            break
        print(f"redoing {len(again)} chains hit by 429s", flush=True)
        time.sleep(30)
        todo = again
    results.sort(key=lambda r: r["id"])
    Path(a.out).write_text(json.dumps(results, indent=0))
    print(f"done {len(results)} chains in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

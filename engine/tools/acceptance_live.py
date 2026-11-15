"""Layer B of the acceptance check: the real model on a running engine, judged by a vision model.

    set -a; . ~/.madhacks-foundry.env; set +a
    python -m engine.tools.acceptance_live --calibrate --out runs/accept-calib          # the judge on known frames
    python -m engine.tools.acceptance_live --url http://127.0.0.1:8000 --out runs/accept  # the cases with "live": true
    python -m engine.tools.acceptance_live --url ... --case undo-the-winter --out ...      # one case

It plays each chosen case of acceptance/sentences.json on the engine the way the demo page does (a reset tap, the
start scene, then each step as a contract 1 final or tap on /control, with the harness's sockets from
engine/tools/demo_run.py), records every frame, and takes the frame LOOK_AFTER_S (3 s, or a step's "look_after_s")
after each step is sent. When the step's first frame arrives later than that, it takes the frame SETTLE_S after the
first frame instead and marks the step late. Each step with a "visible" list is judged: the frame and the yes/no
questions (plus a standard picture-quality question) go to one vision call (Foundry deployment claude-sonnet-5-5 on
the Anthropic Messages API, as engine/final_reader.py calls it), which must answer JSON with a short reason per
question. A step passes when every answer matches what the case expects.

It writes OUT/report.md (frames, questions, answers, reasons, command-to-screen delay per step), OUT/results.json
and OUT/frames/. Exit code 0 when every judged step passes, 1 when one fails, 2 on a setup error.

It changes the shared live video for every viewer: post on the board first. Vision calls are capped (--max-calls,
default 60 including calibration) and paced (Azure returns 429 after about 20 calls in 40 s).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import enum
import io
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

LOOK_AFTER_S = 3.0      # the frame judged is this long after the step was sent ...
SETTLE_S = 1.0          # ... or this long after its first frame, when that came later (the cut dissolves in 0.5 s)
STEP_GAP_S = 8.0        # between two steps
RESET_WAIT_S = 4.0      # after the reset tap of each case
START_WAIT_S = 8.0      # after the start scene's sentence
CALL_GAP_S = 2.0        # between two vision calls
MAX_CALLS = 60
MIN_SIDE = 448          # small sheet crops are scaled up to this height before judging

# Asks only about failures a viewer notices at a glance: an AI frame always has small flaws in its details, and the
# first wording ("every object drawn whole") failed 2 of 7 good frames in calibration.
QUALITY_Q = ("Is the picture free of severe defects that a viewer would notice at a glance, such as large streaks, "
             "blocky or noisy texture over big areas, or an object smashed or broken apart? Small flaws in details "
             "such as odd legs, wheels or faces do not count.")

SYSTEM = """You check single frames of an AI-generated video for an automated test.
You get one frame and numbered yes/no questions about what the frame shows.
Answer each question from what is clearly visible in this frame only. Do not guess from what the scene probably is.
If a thing is too small, hidden or unclear to see, the answer is no.
Reply with one JSON object and nothing else:
{"answers": [{"id": 1, "answer": "yes" or "no", "reason": "at most 15 words on what you see"}]}
with exactly one answer per question, in order."""

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


class Verdict(enum.Enum):
    PASS = "pass"          # every answer matches
    FAIL = "fail"          # at least one answer does not match
    UNJUDGED = "unjudged"  # no usable answer (call failed, bad reply, or the call cap was reached)
    NO_FRAME = "no frame"  # nothing was recorded at that time


# --- the judge ---------------------------------------------------------------------------------------------------

def jpeg_for_judge(data: bytes, crop=None) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    if crop:
        im = im.crop(tuple(crop))
    if im.height < MIN_SIDE:
        im = im.resize((round(im.width * MIN_SIDE / im.height), MIN_SIDE), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=90)
    return out.getvalue()


def parse_answers(text: str, n: int) -> list[dict] | None:
    """[{"answer": bool, "reason": str}] * n, or None when the reply is not that shape."""
    try:
        data = json.loads(FENCE.sub("", text))
        answers = data["answers"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(answers, list) or len(answers) != n:
        return None
    out = []
    for a in answers:
        if not isinstance(a, dict) or str(a.get("answer", "")).strip().lower() not in ("yes", "no"):
            return None
        out.append({"answer": a["answer"].strip().lower() == "yes", "reason": str(a.get("reason", ""))[:200]})
    return out


class Judge:
    """One vision call per frame, capped at `max_calls`. `chat` has complete(system, messages) -> text."""

    def __init__(self, chat, max_calls: int = MAX_CALLS, gap_s: float = CALL_GAP_S):
        self.chat = chat
        self.max_calls = max_calls
        self.gap_s = gap_s
        self.calls = 0
        self.call_ms: list[float] = []
        self._last = 0.0

    def ask(self, jpeg: bytes, questions: list[str]) -> tuple[list[dict] | None, str]:
        """(answers or None, a note on why there are none)."""
        if self.calls >= self.max_calls:
            return None, f"call cap of {self.max_calls} reached"
        time.sleep(max(0.0, self._last + self.gap_s - time.monotonic()))
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                "data": base64.b64encode(jpeg).decode()}},
                   {"type": "text", "text": "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))}]
        for attempt in range(2):
            self.calls += 1
            self._last = t0 = time.monotonic()
            try:
                text = self.chat.complete(SYSTEM, [{"role": "user", "content": content}])
            except Exception as exc:  # noqa: BLE001 - one retry, then reported
                note = f"call failed: {type(exc).__name__}: {str(exc)[:120]}"
                if attempt == 0 and self.calls < self.max_calls:
                    time.sleep(5.0)
                    continue
                return None, note
            self.call_ms.append((time.monotonic() - t0) * 1e3)
            answers = parse_answers(text, len(questions))
            if answers is not None:
                return answers, ""
            note = f"reply not in the asked shape: {text[:120]!r}"
            if attempt == 0 and self.calls < self.max_calls:
                continue
            return None, note
        return None, "no answer"


def judge_frame(judge: Judge, jpeg: bytes, expected: list[tuple[str, bool]]) -> dict:
    answers, note = judge.ask(jpeg, [q for q, _ in expected])
    rows = []
    for i, (q, want) in enumerate(expected):
        a = answers[i] if answers else None
        rows.append({"q": q, "expect": "yes" if want else "no",
                     "answer": None if a is None else ("yes" if a["answer"] else "no"),
                     "reason": None if a is None else a["reason"],
                     "ok": None if a is None else a["answer"] == want})
    if answers is None:
        verdict = Verdict.UNJUDGED
    else:
        verdict = Verdict.PASS if all(r["ok"] for r in rows) else Verdict.FAIL
    return {"verdict": verdict.value, "questions": rows, "note": note}


def step_questions(step: dict) -> list[tuple[str, bool]]:
    from engine.tools.acceptance import question
    return [question(q) for q in step.get("visible", [])] + [(QUALITY_Q, True)]


# --- calibration -------------------------------------------------------------------------------------------------

def calibrate(judge: Judge, path: Path, out: Path) -> dict:
    from engine.tools.acceptance import REPO
    items = json.loads(path.read_text())["items"]
    rows, right, asked = [], 0, 0
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    for n, item in enumerate(items):
        expected = [(QUALITY_Q if q["q"] == "quality" else q["q"], q["expect"] == "yes") for q in item["questions"]]
        jpeg = jpeg_for_judge((REPO / item["image"]).read_bytes(), item.get("crop"))
        name = f"calibration/{n:02d}.jpg"
        (out / name).write_bytes(jpeg)
        res = judge_frame(judge, jpeg, expected)
        for r in res["questions"]:
            if r["ok"] is not None:
                asked += 1
                right += r["ok"]
        rows.append({"name": item["name"], "frame": name, **res})
        print(f"  calibration {item['name']}: {res['verdict']}")
    summary = {"questions": asked, "right": right, "accuracy": round(right / asked, 3) if asked else None,
               "items": rows}
    (out / "calibration.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"# Vision judge calibration", "",
             f"{right} of {asked} answers right ({summary['accuracy']:.0%}) on {len(items)} frames with known contents "
             f"(`acceptance/vision_calibration.json`).", ""]
    for r in rows:
        lines += [f"## {r['name']}: {r['verdict']}", "", f"![]({r['frame']})", ""]
        lines += question_table(r["questions"]) + [""]
    (out / "calibration.md").write_text("\n".join(lines) + "\n")
    return summary


def question_table(rows: list[dict]) -> list[str]:
    out = ["| question | expected | answer | reason |", "|---|---|---|---|"]
    for r in rows:
        mark = "" if r["ok"] is None else (" ok" if r["ok"] else " WRONG")
        out.append(f"| {r['q']} | {r['expect']} | {r['answer'] or '-'}{mark} | {r['reason'] or '-'} |")
    return out


# --- driving the engine ------------------------------------------------------------------------------------------

@dataclass
class Played:
    case: str
    index: int              # step number in the case
    label: str
    step: dict
    sent_t: float           # script seconds
    t_client: int
    next_t: float | None = None   # when the next message was sent (the frame must be from before it)
    reply: dict | None = None
    result: dict = field(default_factory=dict)


async def play(base: str, cases: list[dict], gap_s: float) -> tuple[list[Played], list, float]:
    """Run the cases on the engine; returns the played steps, the frames [(t, chunk, index, jpeg)] in script
    seconds, and the stop time."""
    import websockets
    from engine.tools import demo_run as h
    from engine.tools.acceptance import STARTS, step_label

    rec = h.Recording()
    wsb = h.ws_base(base)
    frames_task = asyncio.create_task(h.read_frames(wsb, rec))
    played: list[Played] = []
    async with websockets.connect(wsb + "/control", open_timeout=20) as control:
        control_task = asyncio.create_task(h.read_control(control, rec))
        if not await h.wait_for(lambda: rec.frames or rec.frames_closed, 20.0) or not rec.frames:
            raise RuntimeError(f"no frames on /frames ({rec.frames_closed and rec.frames_closed.reason})")
        rec.t0 = t0 = time.monotonic()
        last = [0]
        everything_sent: list[Played] = []

        def latest_state() -> dict:
            return rec.states[-1][1] if rec.states else {}

        async def send(msg: dict, case: str, index: int, label: str, step: dict | None) -> Played:
            if rec.frames_closed or rec.control_closed:
                raise RuntimeError("an engine connection closed during the run")
            t_client = last[0] = max(int(time.time() * 1000), last[0] + 1)
            await control.send(json.dumps({**msg, "t_client": t_client}))
            p = Played(case, index, label, step or {}, time.monotonic() - t0, t_client)
            if everything_sent:
                everything_sent[-1].next_t = p.sent_t
            everything_sent.append(p)
            print(f"  {p.sent_t:6.1f} s  {case}: {label}")
            return p

        for case in cases:
            await send({"type": "tap", "action": "reset"}, case["name"], -1, "reset", None)
            await asyncio.sleep(RESET_WAIT_S)
            for sentence in STARTS[case.get("start", "meadow")]:
                await send({"type": "final", "text": sentence}, case["name"], -1, f"start: {sentence}", None)
                await asyncio.sleep(START_WAIT_S)
            for i, step in enumerate(case["steps"]):
                if "say" in step:
                    msg = {"type": "final", "text": step["say"]}
                else:
                    msg = {"type": "tap", "action": step["tap"]}
                    if step["tap"] == "remove":
                        from engine.tools.acceptance import chip_matches
                        ids = [c["id"] for c in latest_state().get("changes", [])
                               if chip_matches(step["chip"], c.get("label", ""))]
                        msg["id"] = ids[0] if ids else "missing"
                played.append(await send(msg, case["name"], i, step_label(step), step))
                await asyncio.sleep(max(gap_s, step.get("look_after_s", LOOK_AFTER_S) + 3.0))
        t_stop = time.monotonic()
        control_task.cancel()
        await asyncio.gather(control_task, return_exceptions=True)
    frames_task.cancel()
    await asyncio.gather(frames_task, return_exceptions=True)
    for p in everything_sent:
        replies = [m for _t, m in rec.states if m.get("t_client") == p.t_client]
        p.reply = replies[-1] if replies else None
    frames = [(t - t0, c, i, j) for t, c, i, j in rec.frames if t >= t0]
    return played, frames, t_stop - t0


def pick_frame(p: Played, frames: list) -> tuple[int | None, dict]:
    """The index of the frame to judge for step `p`, and its timing: first applied frame, delay, lateness."""
    from engine.tools import demo_run as h
    fl = [h.Frame(t, c, i) for t, c, i, _ in frames]
    reply = p.reply or {}
    afc = reply.get("applies_from_chunk")
    first = h.first_frame_from(fl, p.sent_t, afc) if isinstance(afc, int) else None
    look = p.sent_t + p.step.get("look_after_s", LOOK_AFTER_S)
    late = first is not None and first.t + SETTLE_S > look
    if late:
        look = first.t + SETTLE_S
    if p.next_t is not None:
        look = min(look, p.next_t - 0.05)
    i = h.frame_at(fl, look)
    timing = {"sent_s": round(p.sent_t, 2), "look_s": round(look, 2), "late": late,
              "applies_from_chunk": afc, "unmatched": reply.get("unmatched"),
              "chips": [c.get("label") for c in reply.get("changes", [])],
              "delay_ms": None if first is None else round((first.t - p.sent_t) * 1000),
              "stamps": reply.get("stamps", {})}
    return i, timing


# --- report ------------------------------------------------------------------------------------------------------

def write_report(out: Path, played: list[Played], frames: list, calib: dict | None, judge: Judge, meta: dict) -> dict:
    steps = [{"case": p.case, "step": p.index, "label": p.label, **p.result} for p in played]
    judged = [s for s in steps if s.get("verdict") in (Verdict.PASS.value, Verdict.FAIL.value)]
    passed = [s for s in judged if s["verdict"] == Verdict.PASS.value]
    unjudged = [s for s in steps if s.get("verdict") in (Verdict.UNJUDGED.value, Verdict.NO_FRAME.value)]
    delays = sorted(s["timing"]["delay_ms"] for s in steps if s.get("timing", {}).get("delay_ms") is not None)
    summary = {"steps": len(steps), "judged": len(judged), "passed": len(passed),
               "failed": len(judged) - len(passed), "unjudged": len(unjudged),
               "late": sum(1 for s in steps if s.get("timing", {}).get("late")),
               "delay_ms_median": delays[len(delays) // 2] if delays else None,
               "delay_ms_range": [delays[0], delays[-1]] if delays else None,
               "vision_calls": judge.calls,
               "calibration_accuracy": calib.get("accuracy") if calib else None,
               "overall": "pass" if judged and len(passed) == len(judged) and not unjudged else "fail", **meta}
    (out / "results.json").write_text(json.dumps({"summary": summary, "steps": steps}, indent=2) + "\n")
    lines = ["# Acceptance run (real model, vision judge)", "",
             f"Overall: **{summary['overall'].upper()}**: {summary['passed']} of {summary['judged']} judged steps "
             f"pass, {summary['failed']} fail, {summary['unjudged']} unjudged; {summary['steps']} steps played, "
             f"{summary['late']} late (first frame after {LOOK_AFTER_S:.0f} s).",
             f"Command to screen (client side): median {summary['delay_ms_median']} ms, range "
             f"{summary['delay_ms_range']}. Vision calls: {judge.calls}"
             + (f"; judge calibration {calib['right']} of {calib['questions']} right ({calib['accuracy']:.0%})"
                if calib and calib.get("accuracy") is not None else "") + ".",
             f"Engine: {meta.get('url')}, {meta.get('backend')}, started {meta.get('started')}.", "",
             "| case | step | verdict | delay ms | wrong answers |", "|---|---|---|---|---|"]
    for s in steps:
        wrong = "; ".join(f"{r['q']} -> {r['answer']}" for r in s.get("questions", []) if r["ok"] is False)
        lines.append(f"| {s['case']} | {s['label']} | {s.get('verdict', '-')} | "
                     f"{s.get('timing', {}).get('delay_ms')} | {wrong} |")
    lines.append("")
    for s in steps:
        t = s.get("timing", {})
        lines += [f"## {s['case']}: {s['label']} ({s.get('verdict', 'not judged')})", "",
                  f"Sent at {t.get('sent_s')} s, frame at {t.get('look_s')} s"
                  + (" (late: the first frame came after the look time)" if t.get("late") else "")
                  + f", delay {t.get('delay_ms')} ms, chips {t.get('chips')}, unmatched {t.get('unmatched')}.", ""]
        if s.get("frame"):
            lines += [f"![]({s['frame']})", ""]
        if s.get("questions"):
            lines += question_table(s["questions"]) + [""]
        if s.get("note"):
            lines += [f"Note: {s['note']}", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def judge_played(judge: Judge, played: list[Played], frames: list, out: Path) -> None:
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for p in played:
        i, timing = pick_frame(p, frames)
        p.result = {"timing": timing}
        if i is None:
            p.result.update(verdict=Verdict.NO_FRAME.value)
            continue
        name = f"frames/{re.sub(r'[^a-z0-9]+', '-', p.case.lower())}-{p.index:02d}.jpg"
        (out / name).write_bytes(frames[i][3])
        p.result["frame"] = name
        if not p.step.get("visible"):
            continue  # played for the chain and the delay; the case asks nothing of this frame
        res = judge_frame(judge, jpeg_for_judge(frames[i][3]), step_questions(p.step))
        p.result.update(res)
        print(f"  {p.case}: {p.label}: {res['verdict']}" + (f" ({res['note']})" if res["note"] else ""))


def main(argv=None) -> int:
    from engine.foundry import FoundryMessages, settings_from_env
    from engine.tools import demo_run as h
    from engine.tools.acceptance import CASES_FILE, REPO, load_cases

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", action="append", help="only this case (repeatable); default: every case with live: true")
    ap.add_argument("--calibrate", action="store_true", help="judge the known frames first")
    ap.add_argument("--calibrate-only", action="store_true", help="judge the known frames and stop (no engine)")
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS)
    ap.add_argument("--gap", type=float, default=STEP_GAP_S, help="seconds between two steps")
    ap.add_argument("--yes", action="store_true", help="skip the wait before changing a shared video")
    args = ap.parse_args(argv)

    settings = settings_from_env()
    if settings is None:
        print("error: FOUNDRY_ENDPOINT and FOUNDRY_KEY must be set (load ~/.madhacks-foundry.env)", file=sys.stderr)
        return 2
    judge = Judge(FoundryMessages(settings.anthropic_url, settings.key, settings.rewrite_deployment, timeout_s=60.0),
                  max_calls=args.max_calls)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = None
    if args.calibrate or args.calibrate_only:
        calib = calibrate(judge, REPO / "acceptance" / "vision_calibration.json", out)
        print(f"calibration: {calib['right']} of {calib['questions']} right ({calib['accuracy']})")
        if args.calibrate_only:
            return 0

    cases = [c for c in load_cases(CASES_FILE) if (c["name"] in args.case if args.case else c.get("live"))]
    if args.case and len(cases) != len(set(args.case)):
        print(f"error: unknown case in {args.case}", file=sys.stderr)
        return 2
    base = args.url.rstrip("/")
    stats = h.get_stats(base)
    if not stats or stats.get("state") != "GENERATING":
        print(f"error: {base}/stats is not GENERATING", file=sys.stderr)
        return 2
    h.warn_shared(base, not args.yes)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    played, frames, _stop = asyncio.run(play(base, cases, args.gap))
    judge_played(judge, played, frames, out)
    summary = write_report(out, played, frames, calib, judge,
                           {"url": base, "backend": stats.get("backend"), "started": started})
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, indent=1))
    return 0 if summary["overall"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

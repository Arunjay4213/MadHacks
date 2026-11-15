"""The acceptance table: what a user should get for each sentence, checked through the live engine's own reader.

The cases are in `acceptance/sentences.json` at the repo root (format in README.md, "Acceptance"). Each case starts
from a named scene, says a chain of sentences (or taps undo, reset or a chip's x), and states for each step what the
user should get: understood or not, the chips, and words that must or must not be in the prompt the video model sees.

A case runs through the same objects the live engine builds with `--reader commands --final-reader sonnet`:
`engine.reader.FinalReadingService` around `commands.service.CommandService`, with `engine.final_reader`'s reader in
front. Only the Sonnet call is replaced, in one of two modes (`Mode`):

- `sonnet`: the pinned demo prompts (`engine/demo_prompts.json`) as live, and for any other sentence the real Sonnet
  reply recorded once in `acceptance/sonnet_replies.json` (`python -m engine.tools.acceptance_record` records the
  missing ones). A missing reply is reported as such, never guessed.
- `dictionary`: Sonnet unavailable (every call fails) and no pinned prompts, so the keyword table and the engine's
  own measured car worlds and named people decide, as on the live engine when Foundry is down.

The vector matcher (COMMAND_SEMANTIC) is off here: it needs a model download, and it only reads finals that the
dictionary misses.
"""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CASES_FILE = REPO / "acceptance" / "sentences.json"
REPLIES_FILE = REPO / "acceptance" / "sonnet_replies.json"
DEMO_PROMPTS = REPO / "engine" / "demo_prompts.json"

CAR_SENTENCE = "new scene of a red sports car driving along a country road"
# A start scene is the sentences said after a reset tap. The meadow is the engine's starting scene.
STARTS = {"meadow": [], "car": [CAR_SENTENCE]}

STEP_KEYS = {"say", "tap", "chip", "understood", "chips", "no_chips", "has", "lacks", "slow", "strength", "visible",
             "look_after_s", "note", "known_failure"}
CASE_KEYS = {"name", "start", "modes", "xfail", "note", "steps", "live"}
TAPS = ("undo", "reset", "remove")
# "up" and "down" compare a chip's strength with the step before; "low" (< 1), "full" (>= 1) and "high" (> 1)
# compare it with the default 1.
STRENGTH_WAYS = ("up", "down", "low", "full", "high")
ABSOLUTE = {"low": lambda x: x < 1.0, "full": lambda x: x >= 1.0, "high": lambda x: x > 1.0}


class Mode(enum.Enum):
    SONNET = "sonnet"          # pinned prompts plus recorded Sonnet replies
    DICTIONARY = "dictionary"  # Sonnet down: keyword table, measured car worlds, named people


class CaseError(ValueError):
    """The data file is malformed (a typo in a key, a missing field): fails loudly, never silently passes."""


# --- the data file -----------------------------------------------------------------------------------------------

def load_cases(path: Path = CASES_FILE) -> list[dict]:
    data = json.loads(Path(path).read_text())
    cases = data["cases"] if isinstance(data, dict) else data
    names = set()
    for case in cases:
        check_case(case)
        if case["name"] in names:
            raise CaseError(f"two cases are named {case['name']!r}")
        names.add(case["name"])
    return cases


def check_case(case: dict) -> None:
    where = f"case {case.get('name')!r}"
    if not isinstance(case.get("name"), str) or not case["name"]:
        raise CaseError("every case needs a 'name'")
    if extra := set(case) - CASE_KEYS:
        raise CaseError(f"{where} has unknown keys {sorted(extra)}")
    if case.get("start", "meadow") not in STARTS:
        raise CaseError(f"{where}: 'start' must be one of {sorted(STARTS)}")
    for m in case.get("modes", [m.value for m in Mode]):
        Mode(m)
    xfail = case.get("xfail", {})
    if not isinstance(xfail, dict) or any(Mode(m) is None or not str(r).startswith("bug:") for m, r in xfail.items()):
        raise CaseError(f"{where}: 'xfail' maps a mode to a reason that starts with 'bug:'")
    steps = case.get("steps")
    if not isinstance(steps, list) or not steps:
        raise CaseError(f"{where} needs 'steps'")
    for i, step in enumerate(steps):
        w = f"{where} step {i}"
        if extra := set(step) - STEP_KEYS:
            raise CaseError(f"{w} has unknown keys {sorted(extra)}")
        if ("say" in step) == ("tap" in step):
            raise CaseError(f"{w} needs exactly one of 'say' or 'tap'")
        if "tap" in step and step["tap"] not in TAPS:
            raise CaseError(f"{w}: 'tap' must be one of {TAPS}")
        if step.get("tap") == "remove" and not step.get("chip"):
            raise CaseError(f"{w}: a remove tap names the 'chip' whose x is tapped")
        for key in ("chips", "no_chips", "has", "lacks"):
            if not isinstance(step.get(key, []), list):
                raise CaseError(f"{w}: '{key}' must be a list")
        for label, way in step.get("strength", {}).items():
            if way not in STRENGTH_WAYS:
                raise CaseError(f"{w}: strength of {label!r} must be one of {STRENGTH_WAYS}")
        for q in step.get("visible", []):
            question(q)  # raises on a bad entry


def question(q) -> tuple[str, bool]:
    """A `visible` entry: a question whose expected answer is yes, or {"q": ..., "expect": "yes" | "no"}."""
    if isinstance(q, str) and q.strip():
        return q.strip(), True
    if isinstance(q, dict) and isinstance(q.get("q"), str) and q.get("expect") in ("yes", "no"):
        return q["q"].strip(), q["expect"] == "yes"
    raise CaseError(f"a 'visible' entry is a question or {{'q', 'expect': 'yes'|'no'}}, got {q!r}")


def case_modes(case: dict) -> list[Mode]:
    return [Mode(m) for m in case.get("modes", [m.value for m in Mode])]


def step_label(step: dict) -> str:
    if "say" in step:
        return step["say"]
    return f"tap {step['tap']}" + (f" x on {step['chip']!r}" if step["tap"] == "remove" else "")


# --- the stand-in for Sonnet --------------------------------------------------------------------------------------

class MissingReply(RuntimeError):
    pass


class RetryRefused(RuntimeError):
    """The second attempt for a sentence. Live, a call takes about 1.8 s of the 3 s wait, so the reader never asks
    again (it does only when the first call took under half the wait); refusing it here keeps the replay the same."""


def chip_key(user: dict) -> str:
    """The chips shown before the sentence, in order ("winter,lion"): the same prompt can come with other chips (a
    zero-strength chip is not in the prompt), and a reply names the chips to show, so they are in the key."""
    return ",".join(user.get("chips", []))


def reply_key(user: dict) -> tuple[str, str, str]:
    """(the prompt on screen before, the sentence as sent to the model, the chips shown)."""
    return user["prompt"], user["sentence"], chip_key(user)


class ReplayChat:
    """The final reader's chat client, answering from recorded first replies; `missing` lists what had none."""
    model = "claude-sonnet-5-5 (recorded)"

    def __init__(self, replies: dict[tuple[str, str], str]):
        self.replies = replies
        self.missing: list[tuple[str, str]] = []
        self.used: list[tuple[str, str]] = []

    def complete(self, system: str, messages: list[dict]) -> str:
        if len(messages) > 1:
            raise RetryRefused("no retry on the live timing")
        key = reply_key(json.loads(messages[-1]["content"]))
        if key not in self.replies:
            self.missing.append(key)
            raise MissingReply(key[1])
        self.used.append(key)
        return self.replies[key]


class DownChat:
    """Foundry down: every call fails, as a timeout or an HTTP error does live."""
    model = "unavailable"

    def complete(self, system: str, messages: list[dict]) -> str:
        raise ConnectionError("Sonnet is unavailable in this mode")


def entry_key(e: dict) -> tuple[str, str, str]:
    return e["prompt_before"], e["sentence"], e.get("chips", "")


def load_replies(path: Path = REPLIES_FILE) -> dict[tuple[str, str, str], str]:
    if not Path(path).exists():
        return {}
    return {entry_key(e): e["reply"] for e in json.loads(Path(path).read_text())["replies"]}


def build_service(mode: Mode, chat=None, timeout_s: float | None = None):
    """The live reader stack for `mode`; `chat` replaces the Sonnet client (a ReplayChat, or a recorder).
    `timeout_s` replaces the reader's 3 s wait (the recorder waits for the real call to finish)."""
    wait = {} if timeout_s is None else {"timeout_s": timeout_s}
    from commands.service import CommandService
    from engine.final_reader import load
    from engine.reader import FinalReadingService
    if mode is Mode.SONNET:
        chat = chat if chat is not None else ReplayChat(load_replies())
        reader = load("sonnet", chat=chat, cache_path=None, demo_prompts=str(DEMO_PROMPTS), **wait)
    else:
        chat = DownChat()
        reader = load("sonnet", chat=chat, cache_path=None, demo_prompts=None)
    return FinalReadingService(CommandService(semantic_reader=lambda *a: None), reader), reader, chat


# --- running a case ----------------------------------------------------------------------------------------------

@dataclass
class StepResult:
    label: str
    prompt: str                 # the prompt the video model is on after the step
    chips: list[dict]           # [{"id", "label", "strength"}]
    understood: bool
    ramp: int | None
    problems: list[str] = field(default_factory=list)


def chip_matches(expected: str, label: str) -> bool:
    """A chip the user expects ("car") matches a label that holds those words ("red car", "second red car"), also in
    the plural ("horse" matches "horses")."""
    return re.search(rf"(?<![a-z]){re.escape(expected.lower())}(?:s|es)?(?![a-z])", label.lower()) is not None


def _final(text: str, t: int) -> dict:
    return {"type": "final", "text": text, "t_client": t}


def send(svc, step: dict, t: int) -> dict:
    if "say" in step:
        return svc.handle(_final(step["say"], t))
    msg = {"type": "tap", "action": step["tap"], "t_client": t}
    if step["tap"] == "remove":
        chips = [c for c in svc.state_fields()["changes"] if chip_matches(step["chip"], c["label"])]
        if not chips:
            raise LookupError(f"no chip {step['chip']!r} to tap the x on")
        msg["id"] = chips[0]["id"]
    return svc.handle(msg)


def check_step(step: dict, out: dict, before: StepResult | None, prompt: str) -> StepResult:
    steer, state = out.get("steer"), out.get("state", {})
    chips = state.get("changes", [])
    if "say" in step:
        understood = steer is not None and not state.get("unmatched")
    else:
        understood = steer is not None
    r = StepResult(step_label(step), prompt, chips, understood, steer.get("ramp_chunks") if steer else None)
    labels = [c["label"] for c in chips]
    want = step.get("understood", True)
    if understood != want:
        r.problems.append("understood, but it should not be" if understood else
                          f"not understood (unmatched={state.get('unmatched')!r}, steer={'none' if steer is None else 'sent'})")
    if not want and before is not None and prompt != before.prompt:
        r.problems.append("not understood, yet the prompt changed")
    for c in step.get("chips", []):
        if not any(chip_matches(c, l) for l in labels):
            r.problems.append(f"chip {c!r} missing (chips: {labels})")
    for c in step.get("no_chips", []):
        if any(chip_matches(c, l) for l in labels):
            r.problems.append(f"chip {c!r} should be gone (chips: {labels})")
    low = prompt.lower()
    for p in step.get("has", []):
        if p.lower() not in low:
            r.problems.append(f"prompt lacks {p!r}")
    for p in step.get("lacks", []):
        if p.lower() in low:
            r.problems.append(f"prompt has {p!r}")
    if "slow" in step and r.ramp is not None and (r.ramp > 1) != step["slow"]:
        r.problems.append(f"{'slow' if step['slow'] else 'normal'} change expected, ramp_chunks={r.ramp}")
    for label, way in step.get("strength", {}).items():
        now = [c["strength"] for c in chips if chip_matches(label, c["label"])]
        was = [c["strength"] for c in (before.chips if before else []) if chip_matches(label, c["label"])]
        if way in ABSOLUTE:
            if not now:
                r.problems.append(f"strength of {label!r}: no such chip")
            elif not ABSOLUTE[way](now[0]):
                r.problems.append(f"strength of {label!r} should be {way}, got {now[0]}")
        elif not now or not was:
            r.problems.append(f"strength of {label!r}: chip missing before or after")
        elif (way == "up" and not now[0] > was[0]) or (way == "down" and not now[0] < was[0]):
            r.problems.append(f"strength of {label!r} should go {way}: {was[0]} -> {now[0]}")
    return r


def run_case(case: dict, mode: Mode, chat=None, timeout_s: float | None = None) -> tuple[list[StepResult], object]:
    """Run one case from a reset in `mode`; returns the step results (start steps excluded) and the chat."""
    svc, _reader, chat = build_service(mode, chat, timeout_s)
    t = 1
    out = svc.handle({"type": "tap", "action": "reset", "t_client": t})
    for sentence in STARTS[case.get("start", "meadow")]:
        t += 1
        out = svc.handle(_final(sentence, t))
        if out["steer"] is None:
            res = StepResult(f"start: {sentence}", svc.committed_prompt, [], False, None,
                             ["the start scene was not understood"])
            return [res], chat
    before = StepResult("start", svc.committed_prompt, svc.state_fields()["changes"], True, None)
    results = []
    for step in case["steps"]:
        t += 1
        try:
            out = send(svc, step, t)
        except LookupError as exc:
            results.append(StepResult(step_label(step), before.prompt, before.chips, False, None, [str(exc)]))
            continue
        prompt = out["steer"]["prompt"] if out.get("steer") else svc.committed_prompt
        before = check_step(step, out, before, prompt)
        results.append(before)
    return results, chat


def failure_report(case: dict, mode: Mode, results: list[StepResult], chat) -> list[str]:
    lines = []
    missing = getattr(chat, "missing", [])
    if missing:
        lines.append(f"no recorded Sonnet reply for {sorted({k[1] for k in missing})}; record them with "
                     f"`python -m engine.tools.acceptance_record` (needs FOUNDRY_ENDPOINT and FOUNDRY_KEY)")
    for i, r in enumerate(results):
        for p in r.problems:
            lines.append(f"step {i} {r.label!r}: {p}\n    prompt: {r.prompt[:300]}")
    return lines


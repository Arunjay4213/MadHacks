"""Layer A of the acceptance check: every case in acceptance/sentences.json, in both reader modes, on CPU.

Each case is said through the live engine's reader stack (engine/tools/acceptance.py) and checked against what the
user should get. A case the code gets wrong today is marked xfail (strict) in the data file with a 'bug:' reason, so
it fails the suite again once it is fixed and the mark is left behind.
"""

import pytest

from engine.tools.acceptance import CASES_FILE, Mode, case_modes, failure_report, load_cases, run_case


def _params():
    out = []
    for case in load_cases():
        for mode in case_modes(case):
            reason = case.get("xfail", {}).get(mode.value)
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            out.append(pytest.param(case, mode, id=f"{case['name']}-{mode.value}", marks=marks))
    return out


@pytest.mark.parametrize("case,mode", _params())
def test_acceptance_case(case, mode):
    results, chat = run_case(case, mode)
    problems = failure_report(case, mode, results, chat)
    assert not problems, f"{case['name']} ({mode.value}), from {CASES_FILE.name}:\n" + "\n".join(problems)


def test_the_table_covers_every_demo_prompt_line():
    """Every sentence of engine/demo_prompts.json is said in a case that starts from the matching scene."""
    import json
    from engine.tools.acceptance import CAR_SENTENCE, DEMO_PROMPTS
    said = {(c.get("start", "meadow"), s["say"]) for c in load_cases() for s in c["steps"] if "say" in s}
    for entry in json.loads(DEMO_PROMPTS.read_text()):
        if entry["sentence"] == CAR_SENTENCE:
            continue  # the car start scene itself
        start = "car" if "sports car" in entry["prompt_before"] else "meadow"
        assert (start, entry["sentence"]) in said, f"no case says {entry['sentence']!r} on the {start} scene"


def test_both_modes_run_by_default():
    assert {m for c in load_cases() for m in case_modes(c)} == set(Mode)

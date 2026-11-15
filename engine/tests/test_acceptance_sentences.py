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


def test_recorded_replies_are_from_the_current_rules():
    """A reply recorded under an older system prompt says nothing about what the live engine does now. Only the
    replies the cases use are checked; a full `acceptance_record` run drops the others."""
    import json
    import logging
    from engine.final_reader import fingerprint, load
    from engine.tools.acceptance import REPLIES_FILE, DownChat, entry_key
    logging.getLogger("engine.final_reader").setLevel(logging.ERROR)
    system = load("sonnet", chat=DownChat(), cache_path=None, demo_prompts=None).system_prompt
    used = set()
    for case in load_cases():
        if Mode.SONNET in case_modes(case):
            used |= set(run_case(case, Mode.SONNET)[1].used)
    stale = sorted({e["sentence"] for e in json.loads(REPLIES_FILE.read_text())["replies"]
                    if entry_key(e) in used and e.get("fingerprint") != fingerprint(system, e.get("model", ""))})
    assert not stale, ("replies recorded under another final reader system prompt; run "
                       f"`python -m engine.tools.acceptance_record --refresh-stale` with the Foundry key: {stale}")


def test_every_sonnet_step_has_a_recorded_reply():
    """Kept apart from the cases, so an xfail case cannot hide a missing reply."""
    missing = set()
    for case in load_cases():
        if Mode.SONNET in case_modes(case):
            _results, chat = run_case(case, Mode.SONNET)
            missing |= {(case["name"], key[1]) for key in chat.missing}
    assert not missing, ("no recorded Sonnet reply for these; run `python -m engine.tools.acceptance_record` with the "
                         f"Foundry key: {sorted(missing)}")


def test_refreshing_one_case_keeps_the_other_replies(tmp_path, monkeypatch):
    import json
    from engine.tools import acceptance_record as R

    path = tmp_path / "replies.json"
    old = [{"sentence": "s1", "prompt_before": "p", "chips": "", "reply": "r1", "fingerprint": "old"},
           {"sentence": "s2", "prompt_before": "p", "chips": "", "reply": "r2", "fingerprint": "old"}]
    path.write_text(json.dumps({"replies": old}))

    class Real:
        model = "m"

        def complete(self, system, messages):
            return "new"

    def one_case(case, mode, chat, timeout_s=None):
        chat.complete("sys", [{"role": "user", "content": json.dumps({"prompt": "p", "sentence": "s1", "chips": []})}])
        return [], chat

    monkeypatch.setattr(R, "REPLIES_FILE", path)
    monkeypatch.setattr(R, "settings_from_env", lambda: object())
    monkeypatch.setattr(R.FoundryMessages, "from_settings", classmethod(lambda cls, s, t, **options: Real()))
    monkeypatch.setattr(R, "load_cases", lambda: [{"name": "a", "steps": []}, {"name": "b", "steps": []}])
    monkeypatch.setattr(R, "run_case", one_case)
    monkeypatch.setattr(R, "MIN_GAP_S", 0)
    assert R.main(["--refresh-stale", "--case", "a"]) == 0
    saved = {e["sentence"]: e["reply"] for e in json.loads(path.read_text())["replies"]}
    assert saved == {"s1": "new", "s2": "r2"}
    # a full run (no --case) drops the entry no case asked for
    assert R.main([]) == 0
    assert [e["sentence"] for e in json.loads(path.read_text())["replies"]] == ["s1"]

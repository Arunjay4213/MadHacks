"""Layer B's own logic (engine/tools/acceptance_live.py) with a stand-in judge and no engine: reply parsing, verdicts,
the call cap, and which frame is judged. Marked `tools`: run with `pytest -m tools`."""

import json

import pytest

from engine.tools import acceptance_live as L

pytestmark = pytest.mark.tools


class Chat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system, messages):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def answers(*yes):
    return json.dumps({"answers": [{"id": i + 1, "answer": "yes" if y else "no", "reason": "r"}
                                   for i, y in enumerate(yes)]})


def test_parse_answers_needs_one_yes_or_no_per_question():
    assert L.parse_answers("```json\n" + answers(True, False) + "\n```", 2) == [
        {"answer": True, "reason": "r"}, {"answer": False, "reason": "r"}]
    assert L.parse_answers(answers(True), 2) is None
    assert L.parse_answers('{"answers": [{"answer": "maybe"}]}', 1) is None
    assert L.parse_answers("not json", 1) is None


def test_a_step_passes_only_when_every_answer_matches():
    judge = L.Judge(Chat([answers(True, False), answers(True, True)]), gap_s=0)
    expected = [("Is there a lion?", True), ("Is it night?", False)]
    assert L.judge_frame(judge, b"", expected)["verdict"] == "pass"
    res = L.judge_frame(judge, b"", expected)
    assert res["verdict"] == "fail" and [q["ok"] for q in res["questions"]] == [True, False]


def test_a_bad_reply_is_asked_once_more_then_unjudged_and_the_cap_holds():
    chat = Chat(["nonsense", answers(True)])
    judge = L.Judge(chat, gap_s=0)
    assert L.judge_frame(judge, b"", [("q", True)])["verdict"] == "pass" and chat.calls == 2
    judge = L.Judge(Chat(["x", "y"]), gap_s=0)
    assert L.judge_frame(judge, b"", [("q", True)])["verdict"] == "unjudged"
    capped = L.Judge(Chat([]), max_calls=0, gap_s=0)
    res = L.judge_frame(capped, b"", [("q", True)])
    assert res["verdict"] == "unjudged" and "cap" in res["note"]


def frames(times):
    return [(t, n, 0, b"") for n, t in enumerate(times)]  # chunk = the frame's number


def test_the_frame_is_three_seconds_after_the_send_or_one_second_after_a_late_first_frame():
    fl = frames([x / 4 for x in range(0, 80)])  # one frame every 0.25 s
    on_time = L.Played("c", 0, "make it winter", {}, sent_t=2.0, t_client=1, reply={"applies_from_chunk": 11})
    i, t = L.pick_frame(on_time, fl)
    assert fl[i][0] == 5.0 and not t["late"] and t["delay_ms"] == 750
    late = L.Played("c", 1, "add a dragon", {}, sent_t=2.0, t_client=2, reply={"applies_from_chunk": 20})
    i, t = L.pick_frame(late, fl)
    assert t["late"] and fl[i][0] == 6.0 and t["delay_ms"] == 3000
    capped = L.Played("c", 2, "x", {}, sent_t=2.0, t_client=3, next_t=4.0, reply={"applies_from_chunk": 9})
    i, _ = L.pick_frame(capped, fl)
    assert fl[i][0] < 4.0


def test_every_live_step_also_asks_the_quality_question():
    qs = L.step_questions({"visible": ["Is there a lion?", {"q": "Is it night?", "expect": "no"}]})
    assert qs == [("Is there a lion?", True), ("Is it night?", False), (L.QUALITY_Q, True)]

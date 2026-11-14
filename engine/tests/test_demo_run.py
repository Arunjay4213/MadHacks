from pathlib import Path

import pytest

from engine.tools.demo_run import (Frame, ScriptError, Sent, StepKind, command_delay, first_frame_from, frame_at,
                                   load_script, parse_script, shown_at)

SCRIPTS = Path(__file__).resolve().parents[1] / "tools" / "demo_scripts"


def test_draft_script_parses():
    s = load_script(SCRIPTS / "draft.json")
    assert s.name == "draft" and s.end == 46
    assert [(st.at, st.kind, st.text) for st in s.steps] == [
        (0, StepKind.RESET, None), (4, StepKind.SAY, "make it winter"), (12, StepKind.SAY, "add a red car"),
        (20, StepKind.SAY, "make it night"), (28, StepKind.SAY, "add a lion"), (36, StepKind.SAY, "sunny")]


def test_every_shipped_script_parses():
    for path in SCRIPTS.glob("*.json"):
        assert load_script(path).steps


@pytest.mark.parametrize("data", [
    {"steps": [{"at": 0, "reset": True}]},                         # no end
    {"end": 10, "steps": [{"at": 11, "say": "x"}]},                # a step after the end
    {"end": 10, "steps": [{"at": 1, "say": "x", "reset": True}]},  # a step that is both kinds
])
def test_bad_scripts_are_rejected(data):
    with pytest.raises(ScriptError):
        parse_script(data)


def test_bad_json_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ScriptError):
        load_script(p)


FRAMES = [Frame(0.0, 10, 0), Frame(0.5, 10, 1), Frame(1.0, 2, 0),  # a reset restarted chunk numbers at 1.0 s
          Frame(1.5, 3, 0), Frame(2.0, 5, 0), Frame(2.25, 5, 1), Frame(3.0, 8, 0)]


def test_frame_at():
    assert frame_at([], 1.0) is None
    assert frame_at(FRAMES, -1.0) == 0      # before the first frame: the first frame
    assert frame_at(FRAMES, 1.2) == 2       # the last frame at or before the time
    assert frame_at(FRAMES, 2.0) == 4
    assert frame_at(FRAMES, 99) == len(FRAMES) - 1


def test_first_frame_from_skips_frames_before_the_send():
    # chunk 10 was before the send; the old frames must not count even though their chunk is higher
    assert first_frame_from(FRAMES, 1.2, 5) == Frame(2.0, 5, 0)
    assert first_frame_from(FRAMES, 1.2, 9) is None


def test_command_delay():
    reply = {"stamps": {"heard": 0, "decided": 2, "embedded": 3, "generated": 400, "shown": 600},
             "applies_from_chunk": 5, "unmatched": False, "keyword": "winter", "changes": [{"label": "winter"}]}
    r = command_delay(Sent(1.0, StepKind.SAY, "make it winter", 1731612345890, 1.25, reply), FRAMES)
    assert r["first_frame"] == {"chunk": 5, "index": 0, "received_s": 2.0}
    assert r["client_delay_ms"] == 750     # 2.0 s receive minus 1.25 s send
    assert r["changes"] == ["winter"] and r["unmatched"] is False and r["replied"]


def test_command_delay_without_reply_or_frame():
    r = command_delay(Sent(1.0, StepKind.SAY, "x", 1, 1.25, None), FRAMES)
    assert r["client_delay_ms"] is None and not r["replied"] and r["stamps"] == {}
    r = command_delay(Sent(1.0, StepKind.RESET, None, 1, 1.25, {"applies_from_chunk": 99}), FRAMES)
    assert r["first_frame"] is None and r["client_delay_ms"] is None


def test_shown_at():
    reset = {"kind": "reset", "say": None, "sent_s": 0.0, "first_frame": {"chunk": 2, "received_s": 1.0}}
    winter = {"kind": "say", "say": "winter", "sent_s": 1.25, "first_frame": {"chunk": 5, "received_s": 2.0}}
    night = {"kind": "say", "say": "night", "sent_s": 2.5, "first_frame": None}
    results = [reset, winter, night]
    assert shown_at(results, Frame(0.0, 10, 0)) == ("before the run", "reset")  # the old video, reset in flight
    assert shown_at(results, Frame(1.0, 2, 0)) == ("start", None)              # the reset reached the screen
    assert shown_at(results, Frame(1.5, 3, 0)) == ("start", "winter")          # sent, not on screen yet
    assert shown_at(results, Frame(2.25, 5, 1)) == ("winter", None)
    assert shown_at(results, Frame(3.0, 8, 0)) == ("winter", "night")          # night never applied
    assert shown_at([], Frame(3.0, 8, 0)) == ("before the run", None)

import pytest

from engine.engine import check_target
from engine.messages import BadMessage, check, parse_json
from engine.reader import load_reader
from engine.standin_reader import StandinService


@pytest.mark.parametrize("msg", [
    {"type": "partial", "text": "make it win", "t_client": 1},
    {"type": "final", "text": "make it winter", "t_client": 2},
    {"type": "tap", "action": "remove", "id": "c2"},
    {"type": "tap", "action": "undo"},
    {"type": "tap", "action": "reset"},
    {"type": "final", "text": "x", "t_client": 3, "extra": "allowed"},
])
def test_valid_contract_1(msg):
    assert check(1, msg) is msg


@pytest.mark.parametrize("msg", [
    {"type": "final", "text": "no t_client"},
    {"type": "tap", "action": "remove"},
    {"type": "tap", "action": "jump"},
    {"type": "bogus"},
    ["not", "an", "object"],
    {"type": "partial", "text": 5, "t_client": 1},
])
def test_invalid_contract_1(msg):
    with pytest.raises(BadMessage):
        check(1, msg)


def test_bad_json():
    with pytest.raises(BadMessage):
        parse_json("{nope")


def test_contract_2_and_final_cap_rule():
    ok = {"prompt": "p", "ramp_chunks": 4, "provisional": False, "cap": 1.0}
    assert check_target(ok) == ok
    assert check_target({**ok, "provisional": True, "cap": 0.4})["cap"] == 0.4
    with_strength = {**ok, "strength": 0.3, "without_prompt": "w", "later_field": 1}
    assert check_target(with_strength) is with_strength  # passed through untouched
    for bad in ({**ok, "ramp_chunks": 0}, {**ok, "cap": 0.5}, {"prompt": "p"}, {**ok, "strength": 0.3},
                {**ok, "strength": 1.5, "without_prompt": "w"}):
        with pytest.raises(BadMessage):
            check_target(bad)


def test_standin_service():
    svc = StandinService()
    assert svc.committed_prompt is None and svc.likely_prompts() == []
    out = svc.handle({"type": "final", "text": " make it winter ", "t_client": 1})
    assert out["steer"] == {"prompt": "make it winter", "ramp_chunks": 4, "provisional": False, "cap": 1.0}
    check_target(out["steer"])
    assert out["state"] == {"changes": [], "heard": "make it winter", "keyword": None, "unmatched": False}
    assert svc.state_fields() == out["state"] and svc.committed_prompt == "make it winter"
    assert svc.handle({"type": "partial", "text": "make it", "t_client": 1})["steer"] is None
    assert svc.handle({"type": "tap", "action": "undo"})["steer"] is None
    with pytest.raises(ValueError):
        svc.handle({"type": "bogus"})


def test_load_reader():
    assert isinstance(load_reader("standin"), StandinService)
    with pytest.raises(ValueError):
        load_reader("nope")

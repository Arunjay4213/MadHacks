import pytest

from engine.engine import check_target
from engine.messages import BadMessage, check, parse_json
from engine.standin_reader import StandinScene, read_command


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
    for bad in ({**ok, "ramp_chunks": 0}, {**ok, "cap": 0.5}, {"prompt": "p"}):
        with pytest.raises(BadMessage):
            check_target(bad)


def test_standin_reader():
    scene = StandinScene()
    target = read_command({"type": "final", "text": " make it winter ", "t_client": 1}, scene)
    assert target == {"prompt": "make it winter", "ramp_chunks": 4, "provisional": False, "cap": 1.0}
    check_target(target)
    assert read_command({"type": "partial", "text": "make it", "t_client": 1}, scene) is None
    assert read_command({"type": "tap", "action": "undo"}, scene) is None
    assert scene.changes == [] and scene.keyword is None

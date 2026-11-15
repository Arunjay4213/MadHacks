"""Pace words set how fast a change appears, never its strength.

Real bug: Sonnet read "slowly make it winter" as winter at strength 0.6, so winter never fully arrived. The fake
chat replays that bad reply; the sentences go through the real commands.rewriter and the real CommandService.
"""

import json
import logging

from commands.reader import RAMP
from commands.service import CommandService
from engine.final_reader import FinalReader, FinalReaderState, operations_instructions, prompt_spec

WINTER = {"op": "add", "label": "winter", "kind": "season",
          "phrase": "deep winter, snow covering the ground, frost on bare branches"}
NIGHT = {"op": "add", "label": "night", "kind": "time", "phrase": "deep night, dark sky, soft cold moonlight"}


def reply(*ops):
    return json.dumps({"blocked": False, "operations": list(ops), "details": ""})


class ReplayChat:
    model = "fake-sonnet"

    def __init__(self, replies):
        self.replies = replies

    def complete(self, system, messages):
        return self.replies[json.loads(messages[-1]["content"])["sentence"]]


def play(replies, sentences, pinned=None):
    """Each sentence through the reader and the real CommandService: (strengths by label, ramp) per sentence."""
    reader = FinalReader(FinalReaderState.ON, chat=ReplayChat(replies), spec=prompt_spec(), pinned=pinned or {})
    svc = CommandService(None, semantic_reader=lambda *a: None)
    out = []
    for s in sentences:
        res = reader(s, svc.snapshot())
        assert res is not None and not res.get("blocked"), (s, res)
        steer = svc.apply_final(s, res["operations"], None)["steer"]
        out.append(({c["label"]: c["strength"] for c in svc.snapshot()["changes"]}, steer["ramp_chunks"]))
    return out


def test_the_instructions_say_pace_is_not_strength():
    text = operations_instructions()
    assert "Pace words (slowly, gradually, gently, little by little, suddenly) set how fast a change appears" in text
    assert "Strength stays 1.0 unless the sentence has a strength word" in text


def test_slowly_make_it_winter_is_full_winter_on_the_slow_ramp(caplog):
    with caplog.at_level(logging.INFO, logger="engine.final_reader"):
        [(strengths, ramp)] = play({"slowly make it winter": reply({**WINTER, "strength": 0.6})},
                                   ["slowly make it winter"])
    assert strengths == {"winter": 1.0}
    assert ramp == RAMP["slow"] == 12
    assert any("slowly make it winter" in r.getMessage() and "0.6 -> 1.0" in r.getMessage()
               for r in caplog.records if r.levelno == logging.INFO)


def test_suddenly_make_it_night_is_full_night_at_once():
    [(strengths, ramp)] = play({"suddenly make it night": reply({**NIGHT, "strength": 0.6})},
                               ["suddenly make it night"])
    assert strengths == {"night": 1.0}
    assert ramp == RAMP["sudden"]


def test_a_strength_word_without_a_pace_word_keeps_its_strength():
    [(strengths, _)] = play({"make it a little more snowy": reply({**WINTER, "strength": 0.3})},
                            ["make it a little more snowy"])
    assert strengths == {"winter": 0.3}


def test_a_pace_word_with_a_strength_word_keeps_the_strength():
    [(strengths, ramp)] = play({"slowly add a little snow": reply({**WINTER, "strength": 0.3})},
                               ["slowly add a little snow"])
    assert strengths == {"winter": 0.3}
    assert ramp == RAMP["slow"]


def test_a_paced_removal_stays_a_removal():
    sentences = ["make it winter", "slowly fade out the winter"]
    replies = {"make it winter": reply({**WINTER, "strength": 1.0}),
               "slowly fade out the winter": reply({"op": "set_strength", "id": "c1", "strength": 0})}
    (_, _), (strengths, _) = play(replies, sentences)
    assert strengths.get("winter", 0) == 0

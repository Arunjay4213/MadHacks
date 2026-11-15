"""Pace words set how fast a change appears, never its strength.

Real bug: Sonnet read "slowly make it winter" as winter at strength 0.6, so winter never fully arrived. The fake
chat replays that bad reply; the sentences go through the real commands.rewriter and the real CommandService.
"""

import json

import pytest

from commands.reader import RAMP
from commands.service import CommandService
from engine.final_reader import FinalReader, FinalReaderState, prompt_spec

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
        steer = svc.apply_final(s, res["operations"])["steer"]
        out.append(({c["label"]: c["strength"] for c in svc.snapshot()["changes"]}, steer["ramp_chunks"]))
    return out


@pytest.mark.parametrize("sentence, op, strength, ramp", [
    ("slowly make it winter", WINTER, 1.0, "slow"),  # the real bug: Sonnet said 0.6, winter never fully arrived
    ("suddenly make it night", NIGHT, 1.0, "sudden"),
    ("make it a little more snowy", WINTER, 0.3, "default"),  # a strength word without a pace word keeps it
    ("slowly add a little snow", WINTER, 0.3, "slow"),  # a pace word with a strength word keeps the strength
])
def test_pace_words_set_the_ramp_and_strength_words_the_strength(sentence, op, strength, ramp):
    model_strength = 0.6 if strength == 1.0 else strength
    [(strengths, got_ramp)] = play({sentence: reply({**op, "strength": model_strength})}, [sentence])
    assert strengths == {op["label"]: strength}
    assert got_ramp == RAMP[ramp]


def test_a_paced_removal_stays_a_removal():
    sentences = ["make it winter", "slowly fade out the winter"]
    replies = {"make it winter": reply({**WINTER, "strength": 1.0}),
               "slowly fade out the winter": reply({"op": "set_strength", "id": "c1", "strength": 0})}
    (_, _), (strengths, _) = play(replies, sentences)
    assert strengths.get("winter", 0) == 0

"""Fixes for open review comments on the final reader (issue #145), each with the real sentence it broke on.
A fake chat replays the reply shape Sonnet gave; the scene goes through the real CommandService."""

import json

from commands.service import CommandService
from engine.final_reader import FinalReader, FinalReaderState, prompt_spec
from engine.reader import FinalReadingService

WINTER = {"op": "add", "label": "winter", "kind": "season", "phrase": "snow on the ground", "strength": 1.0}


class ReplayChat:
    model = "fake-sonnet"

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def complete(self, system, messages):
        user = json.loads(messages[-1]["content"])
        self.calls.append(user)
        return json.dumps(self.replies[user["sentence"]])


def reply(ops, details="soft light"):
    return {"blocked": False, "operations": ops, "details": details}


def final(text):
    return {"type": "final", "text": text, "t_client": 1}


def service(replies):
    chat = ReplayChat(replies)
    reader = FinalReader(FinalReaderState.ON, chat=chat, spec=prompt_spec())
    return chat, reader, FinalReadingService(CommandService(semantic_reader=lambda *a: None), reader)


def test_a_base_edit_with_a_new_place_drops_the_copied_snow():
    # Greptile on PR #120 (final_reader.py:1049): the base edit copied winter, and world_first moved the desert
    # before the copy, so the desert kept its snow.
    sentence = "turn the tree into a lighthouse and make it a desert"
    _, _, svc = service({
        "make it winter": reply([WINTER]),
        sentence: reply([{"op": "edit_base", "base": "a wide shot of a single lighthouse"},
                         {"op": "add", "label": "desert", "kind": "world", "phrase": "a desert", "strength": 1.0}]),
    })
    svc.handle(final("make it winter"))
    out = svc.handle(final(sentence))
    assert [c["label"] for c in out["state"]["changes"]] == ["desert"]
    assert "snow" not in out["steer"]["prompt"]


def test_a_season_said_with_the_place_and_a_base_edit_stays():
    sentence = "turn the tree into a lighthouse and take me to a snowy desert"
    _, _, svc = service({
        "make it winter": reply([WINTER]),
        sentence: reply([{"op": "edit_base", "base": "a wide shot of a single lighthouse"}, WINTER,
                         {"op": "add", "label": "desert", "kind": "world", "phrase": "a desert", "strength": 1.0}]),
    })
    svc.handle(final("make it winter"))
    out = svc.handle(final(sentence))
    assert sorted(c["label"] for c in out["state"]["changes"]) == ["desert", "winter"]

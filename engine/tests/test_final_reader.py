"""The Sonnet final reader (engine/final_reader.py) and its CommandService wrapper (engine/reader.py).

A fake chat client stands in for Foundry: no network. The wrapper and its shim run against the real
commands.service.CommandService, so a change to its private fields fails here, not in the live demo.
"""

import json
import sys
import threading
import time
import types

import httpx
import pytest

from commands.service import CommandService
from commands.state import SceneState
from engine.final_reader import (FORWARD_MOTION, STYLE, CacheFile, FinalReader, FinalReaderState, expand_operations,
                                 fingerprint, load, prompt_spec, scene_after)
from engine.foundry import FoundryMessages
from engine.reader import FinalReadingService, apply_final, load_reader, service_snapshot

WINTER = {"op": "add", "label": "winter", "kind": "season", "phrase": "snow on the ground", "strength": 1.0}
LION = {"op": "add", "label": "lion", "kind": "extra", "phrase": "a lion in the grass", "strength": 0.5}


def reply(ops=(), details="", blocked=False):
    return json.dumps({"blocked": blocked, "operations": list(ops), "details": details})


class FakeChat:
    """Answers `replies[sentence]` (a reply string or an exception). A sentence in `slow` waits for its event."""

    model = "fake-sonnet"

    def __init__(self, replies=None, slow=None):
        self.replies = replies or {}
        self.slow = slow or {}
        self.calls = []

    def complete(self, system, messages):
        user = json.loads(messages[-1]["content"])
        self.calls.append((system, user))
        if user["sentence"] in self.slow:
            self.slow[user["sentence"]].wait(10)
        answer = self.replies.get(user["sentence"], reply())
        if isinstance(answer, Exception):
            raise answer
        return answer


def reader(chat, **kwargs):
    return FinalReader(FinalReaderState.ON, chat=chat, **kwargs)


def snap(*ops, base="a red car driving along a road"):
    s = SceneState(base)
    for op in ops:
        s.apply(op)
    return s.snapshot()


def test_operations_and_the_standin_prompt():
    chat = FakeChat({"make it winter": "```json\n" + reply([WINTER], "a snowy mountain road between pine forests, "
                                                                     "snow falling, cold blue light") + "\n```"})
    r = reader(chat)
    out = r("make it winter", snap())
    assert out["operations"] == [WINTER]
    assert out["prompt"] == ("A red car driving along a road, " + FORWARD_MOTION + ", a snowy mountain road between pine forests, snow "
                             "falling, cold blue light. " + STYLE)
    system, user = chat.calls[0]
    assert "edit_base" in system and "WORLD" in system
    assert user == {"scene": {"base": "a red car driving along a road", "changes": []},
                    "prompt": "a red car driving along a road", "sentence": "make it winter"}
    assert r("make it winter", snap()) == out and len(chat.calls) == 1  # cached by (prompt before, sentence)
    s = r.stats()
    assert (s["state"], s["calls"], s["cache_hits"], s["p50_ms"] is not None) == ("on", 1, 1, True)


def test_word_list_blocks_without_a_call():
    chat = FakeChat()
    r = reader(chat)
    assert r("show me a naked man", snap()) == {"blocked": True}
    assert not chat.calls and r.stats()["blocked"] == 1


def test_the_model_can_block():
    chat = FakeChat({"make them hurt each other": reply(blocked=True)})
    r = reader(chat)
    assert r("make them hurt each other", snap()) == {"blocked": True}
    assert r("make them hurt each other", snap()) == {"blocked": True} and len(chat.calls) == 1


def test_invalid_replies_fall_back():
    chat = FakeChat({"a": "not json", "b": reply([{"op": "add", "kind": "planet", "label": "x", "phrase": "p"}]),
                     "c": reply([]), "d": reply([{"op": "undo"}]), "e": reply([WINTER], "x " * 200),
                     "f": httpx.ConnectError("down")})
    r = reader(chat)
    for sentence in "abcdef":
        assert r(sentence, snap()) is None
    s = r.stats()
    assert s["rejects"] == 4 and s["errors"] == 2  # "not json" counts as a failed call
    assert r("f", snap()) is None and len(chat.calls) == 6  # a fallback is final in this process: no new call


def test_timeout_falls_back_once_and_drops_the_late_answer():
    gate = threading.Event()
    chat = FakeChat({"slow": reply([WINTER], "snow")}, slow={"slow": gate})
    r = reader(chat, timeout_s=0.1)
    t0 = time.monotonic()
    assert r("slow", snap()) is None
    assert time.monotonic() - t0 < 0.5 and r.stats()["timeouts"] == 1
    gate.set()
    t0 = time.monotonic()
    while r.stats()["late_dropped"] == 0 and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    assert r.stats()["late_dropped"] == 1
    assert r("slow", snap()) is None and len(chat.calls) == 1  # no late swap, no new call


def test_edit_base_keeps_changes_and_one_undo_reverts_it():
    before = snap(WINTER, LION, base="a wide shot of a meadow with a single oak tree")
    ops = expand_operations([{"op": "edit_base", "base": "a wide shot of a meadow with a single lighthouse"}], before)
    assert ops[0] == {"op": "new_scene", "base": "a wide shot of a meadow with a single lighthouse"}
    after = scene_after(before, ops)
    assert after["base"] == "a wide shot of a meadow with a single lighthouse"
    assert [(c["label"], c["kind"], c["phrase"], c["strength"]) for c in after["changes"]] == \
        [(c["label"], c["kind"], c["phrase"], c["strength"]) for c in before["changes"]]
    # applied as one batch to the real SceneState: one undo restores the old base and changes
    s = SceneState("a wide shot of a meadow with a single oak tree")
    s.apply(WINTER)
    s.apply(LION)
    assert s.apply({"op": "batch", "operations": ops})
    assert s.snapshot()["base"].endswith("lighthouse") and len(s.snapshot()["changes"]) == 2
    s.apply({"op": "undo"})
    assert s.snapshot()["base"].endswith("oak tree")
    assert [c["label"] for c in s.snapshot()["changes"]] == ["winter", "lion"]
    # ids of the old scene are gone after a base edit
    lion_id = before["changes"][1]["id"]
    assert expand_operations([{"op": "edit_base", "base": "x"}, {"op": "remove", "id": lion_id}], before) is None


def test_cache_file_survives_a_restart(tmp_path):
    path = tmp_path / "sub" / "cache.json"
    chat = FakeChat({"make it winter": reply([WINTER], "snow"), "boom": httpx.ConnectError("down"),
                     "make them hurt each other": reply(blocked=True)})
    r = reader(chat)
    r._file = CacheFile(path, fingerprint(r.system_prompt, chat.model))
    first = r("make it winter", snap())
    r("boom", snap())
    r("make them hurt each other", snap())
    data = json.loads(path.read_text())
    assert {e["outcome"] for e in data["entries"]} == {"sonnet", "blocked"}  # the fallback is not saved
    assert not list(path.parent.glob(".final-reader-*"))

    chat2 = FakeChat({"make it winter": reply([WINTER], "something else entirely")})
    r2 = reader(chat2, cache_file=CacheFile(path, fingerprint(r.system_prompt, chat.model)))
    assert r2("make it winter", snap()) == first and not chat2.calls  # never regenerated
    r3 = reader(FakeChat(), cache_file=CacheFile(path, fingerprint(r.system_prompt, "another model")))
    assert r3.stats()["cache_size"] == 0


def test_cached_operations_must_still_fit_the_scene():
    s = snap(LION)
    lion_id = s["changes"][0]["id"]
    chat = FakeChat({"remove it": reply([{"op": "remove", "id": lion_id}], "")})
    r = reader(chat)
    assert r("remove it", s)["operations"] == [{"op": "remove", "id": lion_id}]
    other = dict(s, changes=[dict(s["changes"][0], id="c9")])  # same prompt, other id
    assert r("remove it", other) is None


def test_prompt_spec_prefers_commands_rewriter(monkeypatch):
    assert prompt_spec(lambda: (_ for _ in ()).throw(ModuleNotFoundError("no"))).name == "engine stand-in"
    rules = types.SimpleNamespace(SYSTEM_PROMPT="his prompt", validate_rewrite=lambda p, r: f"{p}; {r['details']}")
    monkeypatch.setitem(sys.modules, "commands.rewriter", rules)
    spec = prompt_spec()
    assert spec.name == "commands.rewriter" and spec.details_instructions == "his prompt"
    chat = FakeChat({"make it winter": reply([WINTER], "snow")})
    r = reader(chat, spec=spec)
    assert r("make it winter", snap())["prompt"] == "a red car driving along a road; snow on the ground; snow"
    assert "his prompt" in chat.calls[0][0]


def test_load_states():
    assert load("off").state is FinalReaderState.OFF
    r = load("sonnet", environ={}, cache_path=None)
    assert r.state is FinalReaderState.UNAVAILABLE and "FOUNDRY" in r.stats()["reason"]
    r = load("sonnet", environ={"FOUNDRY_ENDPOINT": "https://madhacks-foundry.openai.azure.com", "FOUNDRY_KEY": "k"},
             cache_path=None)
    assert r.state is FinalReaderState.ON and r._chat.model == "claude-sonnet-5-5"
    assert r._chat._url == "https://madhacks-foundry.services.ai.azure.com/anthropic/v1/messages"
    load_reader("standin", r)
    assert r.state is FinalReaderState.UNAVAILABLE and "--reader commands" in r.reason
    assert r("make it winter", snap()) is None  # UNAVAILABLE behaves as OFF
    with pytest.raises(ValueError):
        load("gpt")


def test_request_body_shape():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"content": [{"type": "text", "text": reply([WINTER], "snow")}],
                                         "stop_reason": "end_turn"})
    client = httpx.Client(transport=httpx.MockTransport(handler),
                          headers={"x-api-key": "k", "anthropic-version": "2023-06-01"})
    chat = FoundryMessages("https://madhacks-foundry.services.ai.azure.com/anthropic/v1/messages", "k", client=client)
    r = reader(chat)
    assert r("make it winter", snap())["operations"] == [WINTER]
    body = json.loads(seen[0].content)
    assert set(body) == {"model", "max_tokens", "system", "messages", "thinking", "output_config"}
    assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 1024
    assert body["thinking"] == {"type": "between_tools"} and body["output_config"] == {"effort": "low"}
    assert body["system"] == r.system_prompt and body["messages"][0]["role"] == "user"
    assert seen[0].headers["x-api-key"] == "k" and seen[0].headers["anthropic-version"] == "2023-06-01"


# --- the wrapper against the real CommandService ----------------------------------------------------------


def final(text, t=1):
    return {"type": "final", "text": text, "t_client": t}


def service():
    return CommandService(semantic_reader=lambda *a: None)


def test_shim_against_the_real_command_service():
    svc = service()
    assert service_snapshot(svc)["base"] == "a wide shot of a meadow with a single oak tree"
    out = apply_final(svc, "winter and a lion", [WINTER, LION])
    assert [c["label"] for c in out["state"]["changes"]] == ["winter", "lion"]
    assert out["state"]["heard"] == "winter and a lion" and not out["state"]["unmatched"]
    assert out["steer"]["prompt"] == svc.committed_prompt and not out["steer"]["provisional"]
    out = apply_final(svc, "blocked words", [])
    assert out["steer"] is None and out["state"]["unmatched"]
    out = svc.handle(final("undo"))
    assert out["state"]["changes"] == []  # the two operations were one edit


def test_wrapper_end_to_end_with_the_real_service():
    oak = "a wide shot of a meadow with a single oak tree"
    chat = FakeChat({
        "make it winter": reply([WINTER], "a snowy meadow, cold blue light"),
        "add a lion": reply([LION], "a snowy meadow with a lion, cold blue light"),
        "turn the tree into a lighthouse": reply([{"op": "edit_base", "base": oak.replace("oak tree", "lighthouse")}],
                                                 "a snowy meadow around a white lighthouse, a lion nearby"),
        "make them fight with weapons": reply(blocked=True),
    })
    svc = FinalReadingService(service(), reader(chat))
    winter = svc.handle(final("make it winter"))
    assert winter["steer"]["prompt"].startswith("A wide shot of a meadow with a single oak tree, a snowy meadow")
    assert [c["label"] for c in winter["state"]["changes"]] == ["winter"]
    lion = svc.handle(final("add a lion"))
    assert [c["label"] for c in lion["state"]["changes"]] == ["winter", "lion"]
    out = svc.handle(final("turn the tree into a lighthouse"))
    assert [(c["label"], c["strength"]) for c in out["state"]["changes"]] == [("winter", 1.0), ("lion", 0.5)]
    assert "lighthouse" in out["steer"]["prompt"] and svc.committed_prompt == out["steer"]["prompt"]
    blocked = svc.handle(final("make them fight with weapons"))
    assert blocked["steer"] is None and blocked["state"]["unmatched"]
    n = len(chat.calls)
    back = svc.handle(final("go back"))  # the dictionary's undo, no call
    assert len(chat.calls) == n
    assert back["steer"]["prompt"] == lion["steer"]["prompt"]  # the same rich prompt as before: same embedding
    assert [c["label"] for c in back["state"]["changes"]] == ["winter", "lion"]
    # a fallback (no operations) uses the dictionary path
    out = svc.handle(final("add a dog"))
    assert "dog" in [c["label"] for c in out["state"]["changes"]]
    # partials and taps still go to the service
    assert svc.handle({"type": "partial", "text": "make it", "t_client": 9})["steer"] is None
    assert svc.handle({"type": "tap", "action": "reset"})["state"]["changes"] == []


def test_engine_reports_the_final_reader():
    from engine.engine import Engine, EngineConfig
    e = Engine(EngineConfig(backend="fake"), lambda b: None, lambda m: None)
    assert e.stats()["final_reader"]["state"] == "off"
    e.stop()


def test_standin_subject_gets_forward_motion_for_vehicles_only():
    from engine.final_reader import FORWARD_MOTION, standin_validate
    car = standin_validate("a red sports car on a road; winter", {"details": "a snowy mountain road."})
    assert FORWARD_MOTION in car and car.index(FORWARD_MOTION) < car.index("a snowy mountain road")
    meadow = standin_validate("a wide shot of a meadow with a single oak tree", {"details": "green grass."})
    assert FORWARD_MOTION not in meadow
    told = standin_validate("a red car driving away from the camera", {"details": "a desert highway."})
    assert told.count("away from the camera") == 1

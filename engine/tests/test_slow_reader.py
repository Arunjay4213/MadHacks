import json
import threading
import time

import httpx

from commands.service import CommandService
from engine.reader import load_reader
from engine.slow_reader import FoundrySlowReader, StateCachingReader, from_env, validate

SNAP = {"base": "a meadow", "changes": [{"id": "c1", "label": "winter", "kind": "season",
                                          "phrase": "snow", "strength": 1.0}], "can_undo": True}


def reader_replying(content, status=200, seen=None):
    """A FoundrySlowReader whose HTTP calls get `content` as the model's message text."""
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
        return httpx.Response(status, json=body)
    client = httpx.Client(transport=httpx.MockTransport(handler), headers={"api-key": "k"})
    return FoundrySlowReader("https://example.test/", "k", "gpt-4.1-nano", client=client)


def final(text):
    return {"type": "final", "text": text, "t_client": 1}


def test_request_shape():
    seen = []
    op = reader_replying(json.dumps({"op": "add", "label": "Lion", "kind": "extra", "phrase": "a lion"}), seen=seen)(
        "add a lion", SNAP)
    assert op == {"op": "add", "label": "lion", "kind": "extra", "phrase": "a lion", "strength": 1.0}
    req = seen[0]
    assert req.url.path == "/openai/deployments/gpt-4.1-nano/chat/completions"
    assert req.url.params["api-version"]
    assert req.headers["api-key"] == "k"
    body = json.loads(req.content)
    assert body["response_format"] == {"type": "json_object"} and body["temperature"] == 0
    user = json.loads(body["messages"][1]["content"])
    assert user["command"] == "add a lion" and user["scene"]["changes"][0]["id"] == "c1"


def test_bad_replies_give_none():
    assert reader_replying("not json")("x", SNAP) is None
    assert reader_replying('{"op": "none"}')("x", SNAP) is None
    assert reader_replying('{"op": "add", "label": "x", "kind": "planet", "phrase": "p"}')("x", SNAP) is None
    assert reader_replying('[1, 2]')("x", SNAP) is None
    assert reader_replying('{"op": "remove", "id": ["c1"]}')("x", SNAP) is None
    assert reader_replying("{}", status=500)("x", SNAP) is None


def test_validate_shapes():
    assert validate({"op": "remove", "id": "c1"}, SNAP) == {"op": "remove", "id": "c1"}
    assert validate({"op": "remove", "id": "c9"}, SNAP) is None  # unknown id
    assert validate({"op": "set_strength", "id": "c1", "strength": 0.3}, SNAP)["strength"] == 0.3
    assert validate({"op": "set_strength", "id": "c1", "strength": 2}, SNAP) is None
    assert validate({"op": "add", "label": "x", "kind": "extra", "phrase": "p", "strength": True}, SNAP) is None
    assert validate({"op": "add", "label": "a" * 40, "kind": "extra", "phrase": "p"}, SNAP) is None
    assert validate({"op": "new_scene", "base": " a  beach "}, SNAP) == {"op": "new_scene", "base": "a beach"}
    assert validate({"op": "undo", "extra": 1}, SNAP) == {"op": "undo"}
    assert validate({"op": "drop_table"}, SNAP) is None


def test_service_uses_it_after_the_keyword_table():
    svc = CommandService(slow_reader=reader_replying(
        json.dumps({"op": "add", "label": "lion", "kind": "extra", "phrase": "a lion in the grass"})), slow_timeout_s=2)
    out = svc.handle(final("turn the tree into a lion"))
    assert out["state"]["unmatched"] is False
    assert [c["label"] for c in out["state"]["changes"]] == ["lion"]
    assert "a lion in the grass" in out["steer"]["prompt"]


def test_from_env_missing_is_off(caplog):
    assert from_env({}) is None
    assert "slow reader off" in caplog.text
    assert from_env({"FOUNDRY_ENDPOINT": "https://e/", "FOUNDRY_KEY": "k"}).deployment == "gpt-4.1-nano"


def test_state_fields_does_not_wait_for_a_slow_handle():
    started = threading.Event()

    def slow(text, snapshot):
        started.set()
        time.sleep(0.5)
        return {"op": "add", "label": "lion", "kind": "extra", "phrase": "a lion"}

    reader = StateCachingReader(CommandService(slow_reader=slow, slow_timeout_s=2))
    t = threading.Thread(target=reader.handle, args=(final("turn the tree into a lion"),))
    t.start()
    started.wait(1)
    t0 = time.monotonic()
    assert reader.state_fields()["changes"] == []
    assert time.monotonic() - t0 < 0.05
    t.join()
    assert [c["label"] for c in reader.state_fields()["changes"]] == ["lion"]


def test_load_reader_commands_without_credentials(monkeypatch):
    monkeypatch.delenv("FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("FOUNDRY_KEY", raising=False)
    reader = load_reader("commands")
    assert reader.committed_prompt and reader.state_fields()["changes"] == []
    assert reader.handle(final("make it winter"))["state"]["changes"][0]["label"] == "winter"

"""Behaviour lock for the sentence-to-prompt path: commands/ and engine/reader.py, final_reader.py, car_template.py.

Each chain sends contract 1 messages, in order, through the reader the engine builds, and records what comes back
for every message: the steer (contract 2), the state fields (contract 4), the committed prompt and a hash of the
likely prompts, and at the end of the chain the likely prompts in full. The Sonnet chains also record every payload
sent to the model, and the system prompt (which is part of the disk cache fingerprint). The recording is engine/tests/fixtures/prompt_lock.json.

A refactor must leave the recording unchanged. Regenerate it only for an intended behaviour change:

    python -m engine.tests.test_prompt_lock --write
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).with_name("fixtures") / "prompt_lock.json"

DICTIONARY_CHAINS = {
    "meadow draft script": [
        "reset", "make it winter", "add a red car", "make it night", "add a lion", "sunny"],
    "meadow worlds": [
        "reset", "make it winter", "add a red car", "make it a desert", "add a red car racing on a road", "undo",
        "take me to the beach at night", "take me back to the meadow", "go to mars", "remove the car near the meadow",
        "remove the forest", "make it foggy"],
    "meadow edits": [
        "reset", "make it winter", "a little more snow", "a little less snow", "make it slowly rain",
        "add a dog and remove the rain", "add a dog and a cat", "remove the dog", "undo", "undo the winter",
        "add a glowing unicorn", "add a bright red car", "make it spooky", "don't add a dog",
        "turn the tree into a lighthouse", "the car drives forward", "add snow near the river", "make it nice",
        "add a lot of fog", "tap:undo", "tap:remove:c1", "add a 猫", "start over"],
    "meadow new scenes": [
        "reset", "new scene of a lighthouse on a cliff", "make it winter", "make it a desert", "show me a castle",
        "reset and add a dog", "start over with a beach"],
    "car chain": [
        "reset", "new scene of a red sports car driving along a country road", "make it winter", "make it a desert",
        "make it night", "add a lion", "remove the lion", "make it autumn", "undo", "take me to the beach",
        "make it sunny", "tap:reset"],
    "partials": [
        "reset", "partial:make it win", "partial:make it winter", "final:make it winter", "partial:add a li",
        "partial:add a lion", "partial:add a", "final:add a lion", "partial:undo", "partial:make it a desert",
        "partial:add a dog", "final:never mind"],
}

# The demo chains as the live engine reads them: pinned prompts first, then measured car worlds (no call), then a
# sentence the fake model answers, then sentences the model cannot answer (the dictionary fallback).
SONNET_CHAINS = {
    "sonnet meadow": [
        "tap:reset", "make it winter", "partial:add a red", "add a red car", "make it night", "add a lion", "sunny",
        "add jensen huang", "remove jensen huang", "make it a desert", "add a red car racing on a road",
        "undo", "undo", "add a small dog", "add a nude dog", "undo", "make it foggy", "tap:reset"],
    "sonnet car": [
        "tap:reset", "new scene of a red sports car driving along a country road", "add jensen huang",
        "remove jensen huang", "make it winter", "make it a desert", "make it night", "make it autumn",
        "make it sunset", "make it rain", "take me to the beach", "make it foggy", "make it country", "undo",
        "add a dog", "tap:reset"],
    "sonnet control": [
        "tap:reset", "make it winter", "undo", "reset", "make it winter and undo", "reset and add a dog",
        "start over with a beach"],
}

FAKE_REPLIES = {
    "add a small dog": {"blocked": False, "operations": [
        {"op": "add", "label": "dog", "kind": "extra",
         "phrase": "a large brown dog standing in the foreground at the right of the frame, facing the camera",
         "strength": 1.0}],
        "details": "fresh green grass, soft shadows under the oak, warm afternoon light, clear air"},
    "make it foggy": {"blocked": False, "operations": [
        {"op": "add", "label": "fog", "kind": "weather", "phrase": "thick fog", "strength": 1.0}],
        "details": "grey fog over the grass, soft diffuse light, damp air, muted colours",
        "road": "a quiet country road in thick fog",
        "scene": "Pale fields and dim trees fade into the fog on both sides.",
        "light": "Soft grey diffuse light under a white sky."},
    "start over with a beach": {"blocked": False, "operations": [
        {"op": "new_scene", "base": "a wide shot of a sandy beach"}],
        "details": "white sand, gentle waves, bright sun, clear blue sky"},
}


class FakeChat:
    """Answers FAKE_REPLIES; any other sentence fails, so the reader falls back to the dictionary."""

    model = "fake-sonnet"

    def __init__(self):
        self.calls = []

    def complete(self, system, messages):
        user = json.loads(messages[-1]["content"])
        self.calls.append(user)
        if user["sentence"] not in FAKE_REPLIES:
            raise RuntimeError("no reply")
        return json.dumps(FAKE_REPLIES[user["sentence"]])


def _message(step):
    kind, _, rest = step.partition(":")
    if kind == "tap":
        action, _, change_id = rest.partition(":")
        return {"type": "tap", "action": action, **({"id": change_id} if change_id else {})}
    if kind == "partial":
        return {"type": "partial", "text": rest, "t_client": 1}
    if kind == "final":
        return {"type": "final", "text": rest, "t_client": 1}
    return {"type": "final", "text": step, "t_client": 1}


def _run(svc, steps, chat=None):
    out = []
    for step in steps:
        result = svc.handle(_message(step))
        # A hash per step, so the prompts the engine encodes ahead (rich ones included) are checked on every scene.
        likely = hashlib.sha256(json.dumps(sorted(svc.likely_prompts())).encode()).hexdigest()[:16]
        record = {"say": step, "steer": result["steer"], "state": result["state"],
                  "committed": svc.committed_prompt, "likely_prompts_sha": likely}
        if chat is not None:
            record["model_calls"] = chat.calls[:]
            chat.calls.clear()
        out.append(record)
    return {"steps": out, "likely_prompts": sorted(svc.likely_prompts())}


def record() -> dict:
    from commands.service import CommandService
    from engine import final_reader
    from engine.reader import load_reader

    data = {}
    for name, steps in DICTIONARY_CHAINS.items():
        data[name] = _run(CommandService(semantic_reader=_no_semantic), steps)
    for name, steps in SONNET_CHAINS.items():
        chat = FakeChat()
        reader = final_reader.load("sonnet", timeout_s=5.0, cache_path=None, chat=chat)
        svc = load_reader("commands", final_reader=reader)
        data[name] = _run(svc, steps, chat)
        data[name]["stats"] = {k: v for k, v in reader.stats().items() if k not in ("p50_ms", "last_ms")}
        reader.stop()
    data["system_prompt"] = reader.system_prompt
    return data


def _no_semantic(text, snapshot):
    return None


@pytest.fixture(autouse=True)
def _dictionary_only(monkeypatch):
    # The live engine runs the vector matcher; the lock runs without it, as CI has no model.
    monkeypatch.delenv("COMMAND_SEMANTIC", raising=False)
    from commands import semantic
    semantic.configured_reader.cache_clear()
    yield
    semantic.configured_reader.cache_clear()


def test_sentence_to_prompt_is_unchanged():
    expected = json.loads(FIXTURE.read_text())
    got = json.loads(json.dumps(record()))
    assert set(got) == set(expected)
    for name in expected:
        if name == "system_prompt":
            assert got[name] == expected[name], "the Sonnet system prompt changed (the disk cache fingerprint too)"
            continue
        for want, have in zip(expected[name]["steps"], got[name]["steps"]):
            assert have == want, f"{name}: {want['say']!r}"
        assert len(got[name]["steps"]) == len(expected[name]["steps"])
        assert got[name] == expected[name], f"{name}: likely prompts or stats"


if __name__ == "__main__":
    import os
    os.environ.pop("COMMAND_SEMANTIC", None)
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: python -m engine.tests.test_prompt_lock --write")
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(record(), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {FIXTURE}")

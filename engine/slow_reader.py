"""The slow reader: one LLM call that turns a sentence the keyword table missed into one scene operation.

`CommandService` (commands/service.py) calls it as `fn(text, snapshot) -> operation dict | None` after its own
matching finds nothing (PLAN.md, "Two readers, fast and slow"). It is the last fallback, not the main reader.

The model is an Azure AI Foundry deployment (Azure OpenAI chat completions), by default `gpt-4.1-nano`, which
answers in about 0.4 to 0.8 s from the engine machine. The client and its FOUNDRY_* settings are in foundry.py,
shared with the prompt rewriter (rewriter.py).

The reply is checked against the operation shapes of commands/state.py before it is returned. A bad reply,
an HTTP error or a timeout gives None, which the service shows as unmatched (red words).
"""

import json
import logging
import os
import time

import httpx

from commands.state import KINDS

from .foundry import DEFAULT_DEPLOYMENT, FoundryChat, settings_from_env

log = logging.getLogger("engine.slow_reader")

HTTP_TIMEOUT_S = 2.0      # a bit above the service's wait, so a slow call ends soon after the service gives up
SERVICE_TIMEOUT_S = 1.5   # how long CommandService waits for the answer; nano measured 0.4 to 0.8 s
MAX_TOKENS = 120

MAX_STRENGTH = 1.3
MAX_LABEL = 24
MAX_PHRASE = 120
MAX_BASE = 200

SYSTEM_PROMPT = f"""You turn one spoken command into one edit of a video scene. Reply with one JSON object only.

The scene is a base sentence plus a list of changes. Each change has an id, a label (a short name shown on a
chip), a kind, a phrase (the words added to the video prompt) and a strength (0 to 1.3, normal 1.0).
Kinds: season (winter, autumn, ...), weather (rain, fog, sunny, ...), time (night, day, sunset, ...), and extra
(any object, animal, person or effect). Adding a season, weather or time replaces the old one of that kind.

Operations, as JSON:
{{"op": "add", "label": "<1 or 2 words, lowercase>", "kind": "<season|weather|time|extra>", "phrase": "<short visual description>", "strength": 1.0}}
{{"op": "remove", "id": "<id of an existing change>"}}
{{"op": "set_strength", "id": "<id of an existing change>", "strength": <0 to 1.3>}}
{{"op": "undo"}}
{{"op": "reset"}}
{{"op": "new_scene", "base": "<one sentence describing the whole new scene>"}}
{{"op": "none"}}

Rules:
- Write phrases as concrete visual descriptions for an image model, under 12 words, for example
  "a lion standing in the grass" or "bright daytime under a clear blue sky".
- To change something that is part of the base sentence (for example "turn the tree into a lighthouse"),
  use new_scene with the base sentence rewritten, changing only that thing.
- To remove or weaken something, use the id of the matching change from the current scene.
  If nothing matches, reply {{"op": "none"}}.
- "a little" means strength 0.3, "a lot" or "very" means 1.3.
- If the command is vague, not about the picture, or not one edit, reply {{"op": "none"}}."""


def _text(value, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value if value and len(value) <= limit else None


def _strength(value, default=None):
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= MAX_STRENGTH:
        return None
    return round(float(value), 2)


def validate(reply, snapshot: dict, max_phrase: int = MAX_PHRASE) -> dict | None:
    """The model's reply as a clean operation for SceneState.apply, or None if it is not exactly one valid edit.
    `max_phrase` is the longest phrase in characters (the final reader allows longer, positioned phrases)."""
    if not isinstance(reply, dict):
        return None
    op = reply.get("op")
    ids = {c["id"] for c in snapshot.get("changes", [])}
    if op == "add":
        label, phrase = _text(reply.get("label"), MAX_LABEL), _text(reply.get("phrase"), max_phrase)
        kind, strength = reply.get("kind"), _strength(reply.get("strength"), 1.0)
        if label is None or phrase is None or kind not in KINDS or strength is None:
            return None
        return {"op": "add", "label": label.lower(), "kind": kind, "phrase": phrase, "strength": strength}
    if op == "remove":
        return {"op": "remove", "id": reply["id"]} if reply.get("id") in ids else None
    if op == "set_strength":
        strength = _strength(reply.get("strength"))
        if reply.get("id") not in ids or strength is None:
            return None
        return {"op": "set_strength", "id": reply["id"], "strength": strength}
    if op in ("undo", "reset"):
        return {"op": op}
    if op == "new_scene":
        base = _text(reply.get("base"), MAX_BASE)
        return {"op": "new_scene", "base": base} if base else None
    return None  # "none" or anything unknown


def _scene_for_model(snapshot: dict) -> dict:
    return {
        "base": snapshot.get("base"),
        "changes": [{k: c[k] for k in ("id", "label", "kind", "phrase", "strength")} for c in snapshot.get("changes", [])],
    }


class FoundrySlowReader:
    """Callable slow reader: `reader(text, snapshot) -> operation dict | None`. Never raises."""

    def __init__(self, endpoint: str, key: str, deployment: str = DEFAULT_DEPLOYMENT,
                 client: httpx.Client | None = None) -> None:
        self.deployment = deployment
        self._chat = FoundryChat(endpoint, key, deployment, timeout_s=HTTP_TIMEOUT_S, client=client)

    def __call__(self, text: str, snapshot: dict) -> dict | None:
        start = time.monotonic()
        op, why = None, "ok"
        try:
            reply = self._ask(text, snapshot)
            op = validate(reply, snapshot)
            if op is None:
                why = f"no valid operation in {json.dumps(reply)[:200]}"
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            why = f"call failed: {exc!r}"[:200]
        log.info("slow reader: %r -> %s (%.0f ms, %s)", text[:80], op, (time.monotonic() - start) * 1e3, why)
        return op

    def _ask(self, text: str, snapshot: dict):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"scene": _scene_for_model(snapshot), "command": text})},
        ]
        content = self._chat.complete(messages, MAX_TOKENS, json_object=True)
        return json.loads(content)  # ValueError on bad JSON


def from_env(environ=os.environ) -> FoundrySlowReader | None:
    """The slow reader from FOUNDRY_* variables, or None (logged once) when they are missing."""
    settings = settings_from_env(environ)
    if settings is None:
        log.warning("slow reader off: FOUNDRY_ENDPOINT or FOUNDRY_KEY is not set, so commands the keyword table "
                    "does not know stay unmatched")
        return None
    log.info("slow reader on: Azure Foundry deployment %s, wait up to %.1f s", settings.deployment, SERVICE_TIMEOUT_S)
    return FoundrySlowReader(settings.endpoint, settings.key, settings.deployment)


class StateCachingReader:
    """Wraps CommandService so `state_fields()` never waits for a slow `handle()`.

    CommandService.handle holds its lock for the whole slow reader call (up to SERVICE_TIMEOUT_S), and the
    engine's streamer thread calls state_fields() for every chunk. Without this wrapper the stream would stall
    for as long as the LLM call takes. The state only changes inside handle(), and handle() returns the new
    state, so caching that return value gives the same answer without taking the lock.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._state = inner.state_fields()

    def handle(self, msg: dict) -> dict:
        result = self._inner.handle(msg)
        self._state = result["state"]
        return result

    def state_fields(self) -> dict:
        return self._state

    def likely_prompts(self) -> list[str]:
        return self._inner.likely_prompts()

    @property
    def committed_prompt(self):
        return self._inner.committed_prompt

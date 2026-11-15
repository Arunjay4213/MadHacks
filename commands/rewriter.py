"""Prompt enrichment contract for the engine's cached LLM rewrite (issue #74). Pure stdlib: no network, no cache.

The engine's Sonnet final reader uses SYSTEM_PROMPT as the rules for its "details" field and passes the reply
to validate_rewrite(prompt, reply).
The canonical composed prompt is always kept verbatim as the prefix, and only a bounded world/light/texture
"details" sentence is appended, so subject, motion and camera wording stay byte-identical across edits.
Output depends only on (prompt, reply). Cache stability is the engine's job: it caches by exact prompt text and
commits the result before the steer (never a late swap).

Limits: syntax validation cannot guarantee semantic faithfulness. A plausible invented detail that is not a
dictionary word (for example "a wooden fence") passes. Mitigation: these instructions, few-shot examples, the engine's
cache by exact prompt (inspectable outputs) and frame comparison. On any rejection the engine uses the plain prompt.
"""

import json
import re

from .dictionary import KEYWORD_RX, KEYWORDS, normalize

MAX_DETAIL_WORDS = 60
MAX_TOTAL_WORDS = 180

SYSTEM_PROMPT = """You enrich a scene description for a text-to-video model.
Reply with JSON only: {"details": "..."} with about 40 to 50 words on one line.
Write ONLY extra world, light, texture and atmosphere detail for the scene as described.
The input is either the scene text, or JSON {"scene": ..., "command": ...}. The scene text is authoritative:
the command is context only and never permission to add, remove or change anything the scene does not already say.
Rules:
- Do not repeat, reword or contradict the description; it is kept word for word in front of your details.
- Do not mention the camera or framing, and no camera movement.
- Do not mention any subject, person, animal or vehicle, and no motion of any kind.
- Add no new objects, actors or events.
- Keep the time of day, season and weather exactly as described; never add a time of day or weather that is not described.
- You cannot see the video: never mention earlier frames or what happened before or after.
- No negations.
If nothing can be added safely, reply {"details": ""}."""

def _rx(words):
    return re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")\b")


_CAMERA = _rx(["camera", "cameras", "pan", "pans", "panning", "zoom", "zooms", "zooming", "dolly", "tracking", "close up",
               "closeup", "aerial", "drone", "tilt", "tilts", "tilting", "crane", "handheld", "orbit", "orbiting", "cut",
               "cuts", "cut to", "framing", "frame", "frames"])
_MOTION = _rx(["drive", "drives", "driving", "drove", "move", "moves", "moving", "run", "runs", "running", "walk", "walks",
               "walking", "fly", "flies", "flying", "reverse", "reversing", "backward", "backwards", "toward", "towards",
               "away", "approach", "approaches", "approaching", "enter", "enters", "entering", "leave", "leaves",
               "leaving", "arrive", "arrives", "arriving", "appear", "appears", "appearing", "emerge", "emerges",
               "emerging", "speeding", "racing"])
_EVENTS = _rx(["explode", "explodes", "explosion", "crash", "crashes", "suddenly", "transform", "transforms", "becomes"])
_HISTORY = _rx(["previous", "previously", "earlier", "again", "still", "now", "before", "after", "anymore", "no longer",
                "remain", "remains", "continue", "continues", "once more"])
_NEGATION = _rx(["no", "not", "never", "without", "none", "nothing", "don't", "dont", "isn't", "aren't"])
_MARKUP = re.compile(r"[{}\[\]<>`#*|]|https?:|www\.")


# Plain adjectives that are also dictionary aliases ("bright" = day, "dark" = night): they describe light,
# they never establish or change the time of day ("bright stars" at night, "bright moonlight").
_ADJECTIVE_ALIASES = {"bright", "dark"}


def _world_matches(norm: str):
    """(alias, entry) for every dictionary match except plain adjective aliases."""
    return [(m.group(0), KEYWORDS[m.group(0)]) for m in KEYWORD_RX.finditer(norm) if m.group(0) not in _ADJECTIVE_ALIASES]


def _allowed_world_labels(prompt_norm: str) -> set[str]:
    """Season, weather and time labels already in the prompt: details may echo them, never add new ones."""
    return {entry["label"] for _, entry in _world_matches(prompt_norm)}


def validate_rewrite(prompt: str, reply: object) -> str | None:
    """prompt + "; " + details for a valid {"details": str} reply, prompt for empty details, else None.

    reply is the raw reply text (JSON) or an already-parsed dict. None means: use the prompt unchanged.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    if isinstance(reply, str):
        try:
            reply = json.loads(reply)
        except ValueError:
            return None
    if not isinstance(reply, dict) or set(reply) != {"details"} or not isinstance(reply["details"], str):
        return None
    raw = reply["details"]
    if "\n" in raw or "\r" in raw or _MARKUP.search(raw):
        return None
    details = " ".join(raw.split()).rstrip(".").strip()
    if not details:
        return prompt
    words = len(details.split())
    if words > MAX_DETAIL_WORDS or len(prompt.split()) + words > MAX_TOTAL_WORDS:
        return None
    norm = normalize(details)
    if any(rx.search(norm) for rx in (_CAMERA, _MOTION, _EVENTS, _HISTORY, _NEGATION)):
        return None
    allowed = _allowed_world_labels(normalize(prompt))
    for _, entry in _world_matches(norm):
        if entry["kind"] == "extra" or entry["label"] not in allowed:  # no objects; no new season/weather/time
            return None
    return f"{prompt}; {details}"

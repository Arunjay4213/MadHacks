"""Fast reader: a keyword table, no model. Turns heard text into one operation dict (PLAN.md section 1)."""

import difflib
import re

_SEASON = {
    "winter": "winter scenery with snow on the ground",
    "autumn": "autumn scenery with orange and red leaves",
    "spring": "spring scenery with fresh blossoms",
    "summer": "bright summer scenery with lush green grass",
}
_WEATHER = {
    "rain": "steady rain falling",
    "fog": "a thick fog in the air",
    "storm": "a dark thunderstorm with lightning",
    "wind": "strong wind blowing through everything",
}
_TIME = {
    "night": "night time under a full moon",
    "sunset": "a golden sunset sky",
    "sunrise": "a soft pink sunrise sky",
    "noon": "harsh midday sunlight",
}
_EXTRA = {
    "river": "a river running through the scene",
    "fire": "a crackling campfire",
    "rainbow": "a rainbow across the sky",
    "lighthouse": "a tall white lighthouse",
    "flowers": "colorful wildflowers everywhere",
    "birds": "a flock of birds flying overhead",
    "mountains": "snow-capped mountains in the distance",
    "boat": "a small wooden boat",
    "house": "a small wooden cabin",
    "deer": "a deer grazing nearby",
    "fireflies": "glowing fireflies",
    "lake": "a calm lake reflecting the sky",
}
# Spoken word -> canonical label.
_ALIASES = {
    "snow": "winter", "snowy": "winter", "wintry": "winter", "fall": "autumn", "leaves": "autumn",
    "rainy": "rain", "raining": "rain", "foggy": "fog", "mist": "fog", "misty": "fog",
    "stormy": "storm", "thunder": "storm", "lightning": "storm", "windy": "wind",
    "nighttime": "night", "dark": "night", "moon": "night", "dusk": "sunset", "dawn": "sunrise",
    "midday": "noon", "stream": "river", "creek": "river", "campfire": "fire", "flames": "fire",
    "flower": "flowers", "bird": "birds", "mountain": "mountains", "cabin": "house", "pond": "lake",
}

KEYWORDS: dict[str, dict] = {}
for _kind, _table in (("season", _SEASON), ("weather", _WEATHER), ("time", _TIME), ("extra", _EXTRA)):
    for _label, _phrase in _table.items():
        KEYWORDS[_label] = {"label": _label, "kind": _kind, "phrase": _phrase}
for _word, _label in _ALIASES.items():
    KEYWORDS[_word] = KEYWORDS[_label]

RAMP = {"default": 4, "slow": 12, "sudden": 1, "confirm": 2}  # chunks; settings until the first real run

_STRENGTH_LOW = 0.3
_STRENGTH_HIGH = 1.3  # "a lot": heavy wording in the composer
_STEP = 0.3


def _rx(words):
    return re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")\b")


_KEYWORD_RX = _rx(KEYWORDS)
_UNDO_RX = _rx(["undo", "go back", "revert", "take that back", "never mind", "nevermind"])
_RESET_RX = _rx(["reset", "start over", "start again", "clear everything", "clear it all"])
_REMOVE_RX = _rx(["remove", "no more", "get rid of", "take away", "delete", "without", "stop the", "lose the", "drop the"])
_SCENE_RX = re.compile(r"\b(?:new scene(?: of| with)?|change the scene to|switch the scene to|take me to)\s+(.+)")  # wins over keywords
_SHOW_RX = re.compile(r"\b(?:show me|switch to|go to)\s+(.+)")  # a new scene only when no keyword matches
# Not expressible as one operation; left to the slow reader, never guessed early.
_NEGATION_RX = re.compile(r"\b(?:don't|dont|do not|not|never(?! mind\b))\b")  # "never mind" is undo, not negation
_TRANSFORM_RX = _rx(["into", "instead of", "replace", "replaced", "swap"])
_SLOW_RX = _rx(["slowly", "gradually", "gently", "little by little", "over time"])
_SUDDEN_RX = _rx(["suddenly", "instantly", "immediately", "right now", "quickly", "at once", "all at once"])
_LOW_RX = _rx(["a little", "a bit", "slightly", "a touch of", "a hint of", "barely", "light", "lightly"])
_HIGH_RX = _rx(["a lot", "lots of", "much more", "way more", "tons of", "heavy", "heavily", "very", "really", "extremely"])
_MORE_RX = _rx(["more"])
_LESS_RX = _rx(["less", "fewer", "weaker", "softer"])


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")  # dictation and pastes use curly apostrophes
    return re.sub(r"[^a-z0-9' -]+", " ", text).strip()


def _pace(t: str):
    return "slow" if _SLOW_RX.search(t) else "sudden" if _SUDDEN_RX.search(t) else None


def resolve(query: str, snapshot: dict):
    """Change id for a spoken name: exact id, label, alias, phrase word, then close match. Most recent wins."""
    changes = snapshot.get("changes", [])
    q = _norm(query)
    words = set(q.split())
    labels = {KEYWORDS[w]["label"] for w in words if w in KEYWORDS}
    for test in (
        lambda c: c["id"] == query.strip(),
        lambda c: c["label"].lower() == q,
        lambda c: c["label"].lower() in labels or c["label"].lower() in words,
        lambda c: words & set(_norm(c["phrase"]).split()) - {"a", "the", "of", "with", "on", "in", "and"},
    ):
        hits = [c for c in changes if test(c)]
        if hits:
            return hits[-1]["id"]
    for w in words:
        close = difflib.get_close_matches(w, [c["label"].lower() for c in changes], n=1, cutoff=0.8)
        if close:
            return [c for c in changes if c["label"].lower() == close[0]][-1]["id"]
    return None


def _find(snapshot, label):
    hits = [c for c in snapshot.get("changes", []) if c["label"] == label]
    return hits[-1] if hits else None


def read(text: str, snapshot: dict) -> dict:
    """{"op": operation dict or None, "keyword": spoken key word or None, "pace": "slow" | "sudden" | None}."""
    t = _norm(text)
    pace = _pace(t)
    out = lambda op, keyword=None: {"op": op, "keyword": keyword, "pace": pace}  # noqa: E731
    if _NEGATION_RX.search(t) or _TRANSFORM_RX.search(t):
        return out(None)
    m = _UNDO_RX.search(t)
    if m:
        return out({"op": "undo"}, m.group(0))
    m = _RESET_RX.search(t)
    if m:
        return out({"op": "reset"}, m.group(0))
    m = _SCENE_RX.search(t)
    if m and m.group(1).strip(" .'"):
        return out({"op": "new_scene", "base": m.group(1).strip(" .'")})
    kw = _KEYWORD_RX.search(t)
    keyword = kw.group(0) if kw else None
    m = _REMOVE_RX.search(t)
    if m:
        target = resolve(t[m.end():] or t, snapshot)
        return out({"op": "remove", "id": target} if target else None, keyword)
    if not kw:
        m = _SHOW_RX.search(t)
        if m and m.group(1).strip(" .'"):
            return out({"op": "new_scene", "base": m.group(1).strip(" .'")})
        return out(None)
    entry = KEYWORDS[keyword]
    existing = _find(snapshot, entry["label"])
    current = existing["strength"] if existing else 0.0
    if _LOW_RX.search(t):
        strength = _STRENGTH_LOW
    elif _HIGH_RX.search(t):
        strength = _STRENGTH_HIGH
    elif _MORE_RX.search(t) and existing:
        strength = min(_STRENGTH_HIGH, round(current + _STEP, 2))
    elif _LESS_RX.search(t) and existing:
        strength = max(_STRENGTH_LOW / 3, round(current - _STEP, 2))
    else:
        strength = 1.0
    if existing:
        return out({"op": "set_strength", "id": existing["id"], "strength": strength}, keyword)
    return out({"op": "add", "label": entry["label"], "kind": entry["kind"], "phrase": entry["phrase"], "strength": strength}, keyword)


def spot(partial_text: str):
    """The KEYWORDS key in partial text, for early steering. None for undo/reset/remove/negation/transform/new-scene wording: never guessed."""
    t = _norm(partial_text)
    if any(rx.search(t) for rx in (_UNDO_RX, _RESET_RX, _REMOVE_RX, _NEGATION_RX, _TRANSFORM_RX, _SCENE_RX)):
        return None
    m = _KEYWORD_RX.search(t)
    return m.group(0) if m else None

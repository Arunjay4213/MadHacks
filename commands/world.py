"""World (setting) support: "make it a desert" replaces the world, never the subject (issue #74).

The world is one change of kind "world" (a singleton, like season), so ids, undo, chips and compound
sentences work unchanged. Phrases are Ishaan's exact tested sentences (board #6885673339, and #6886548029 for
the four measured road | scene | light worlds).
Pure stdlib. Imports only dictionary.py, so reader.py can import this module without a cycle.
"""

import re

from .dictionary import KEYWORD_RX, KEYWORDS, normalize
from .state import DEFAULT_BASE

# Ishaan's measured car template (board #6886548029). With the rear view named, the car stayed seen from behind in
# 10 of 10 GPU runs, and swapping road, scene and light changed the world (a place appended to a short prompt did not):
#   {subject}, seen from directly behind ..., drives fast along {road}, moving forward away from the camera. {CAMERA}
#   {scene} {light} {STYLE}
# A world phrase below is "{road}. {scene} {light}", so the road is its text before the first ". ".
CAR_SUBJECT = "A red sports car"
CAMERA = ("The camera follows directly behind the car in a smooth, low tracking shot at the same speed, keeping the car "
          "centred in the frame while the white dashed centre line rushes toward the viewer.")
STYLE = "Cinematic, highly detailed, realistic, sharp focus, motion blur on the road."  # road scenes only
_REAR = ", seen from directly behind with its rear lights facing the camera, drives fast along "
_FORWARD = ", moving forward away from the camera."

# Each world is a list of segments (text, tags). Joined, the segments are Ishaan's tested sentence verbatim.
# Tagged segments are the built-in sky/time/weather; world_with_overrides drops one only when an explicit
# change of the same kind (time/weather/season) says something different.
_D, _N, _SUN = {"time": "day"}, {"time": "night"}, {"time": "sunset"}
_TABLE = [
    # The world of Ishaan's earlier tested car prompt (the engine's CAR_PROMPT), split the same way.
    ("country road", [("a straight two-lane country road", {}),
                      (". Green fields and rows of trees line both sides of the road", {}),
                      (" under a clear blue summer sky", {"time": "day", "weather": "clear", "season": "summer"}),
                      (". Bright midday sunshine, crisp shadows", {"time": "day", "weather": "clear"})],
     ["country road", "countryside", "fields", "farmland", "the start", "the country"]),
    # desert, snowy mountain road (Ishaan's "winter"), neon city and ocean coast are his measured road | scene | light
    # worlds (board #6886548029), word for word, as "{road}. {scene} {light}", so this path matches the engine's.
    ("desert", [("a long straight desert highway", {}),
                (". Red sandstone mesas and golden sand dunes stretch to the horizon on both sides, dry scrub along the "
                 "roadside", {}),
                (", heat haze shimmering above the asphalt", {"weather": "haze", "season": "summer"}),
                (". Hot afternoon sun in a cloudless pale blue sky, strong hard shadows", {"time": "day", "weather": "clear"})],
     ["desert", "desert highway", "sand dunes", "dunes", "mesas"]),
    ("snowy mountain road", [("a snowy mountain road", {}),
                             (". Pine forests covered in thick white snow line both sides, high snowbanks along the edges "
                              "of the road", {}),
                             (", snowflakes falling", {"weather": "snow", "season": "winter"}),
                             (". Cold blue overcast winter light, soft diffuse shadows",
                              {"time": "day", "weather": "overcast", "season": "winter"})],
     ["snowy mountains", "snowy mountain road", "mountain road", "the mountains"]),
    ("neon city", [("a wide city street", {}), (" at night", _N),
                   (". Tall skyscrapers covered in glowing pink, blue and purple neon signs rise on both sides, the wet "
                    "asphalt reflects the colourful lights", {}),
                   (", light rain falling", {"weather": "rain"}),
                   (". Dark night sky", _N), (", vivid neon glow, cyberpunk atmosphere", {})],
     ["neon city", "city", "city street", "cyberpunk city", "downtown"]),
    ("ocean coast", [("a coastal cliff road", {}),
                     (". A bright turquoise ocean with white waves breaking on the rocks below on one side, palm trees and "
                      "green hills on the other side", {}),
                     (". Sunny tropical day, clear blue sky, sparkling light on the water",
                      {"time": "day", "weather": "clear", "season": "summer"})],
     ["ocean", "coast", "coastal road", "beach", "seaside", "sea", "tropical coast", "cliffs"]),
    ("mars", [("a dusty road across the red rocky surface of Mars, craters and red mountains", {}),
              (", a pale pink sky with two small moons", _D)],
     ["mars", "red planet", "martian surface", "surface of mars"]),
    # Added per lead (not from Ishaan's tested set): same template style.
    ("forest", [("a narrow road through a dense green pine forest, tall trees on both sides", {}),
                (", soft sunlight through the branches", {"time": "day", "weather": "clear"})],
     ["forest", "woods", "woodland", "pine forest"]),
    # The oak scene's own world, so "take me back to the meadow" can return to it.
    ("meadow", [("a green meadow with soft grass", {}), (", under a clear blue sky", {"time": "day", "weather": "clear"})],
     ["meadow", "the meadow", "grassland"]),
]
# A sunlit time change says the weather too: "sunny" and "make it day" (the dictionary's "bright daytime sunlight under a
# clear blue sky") conflict with a built-in overcast light, rain or falling snow, unless a weather change says otherwise.
_SUNLIT = frozenset({"clear", "haze"})
_IMPLIED = {"day": {"weather": _SUNLIT}, "noon": {"weather": _SUNLIT}}
# A world's own light sentence for an explicit change, in the style of Ishaan's measured light: on the snowy road a
# sunny day is winter sunshine, so the road stays snowy instead of getting the generic daylight sentence.
_LIGHT_WORDING = {("snowy mountain road", "day"): "bright winter sunshine in a clear blue sky, sparkling snow, crisp shadows"}
_SEGMENTS = {label: segments for label, segments, _ in _TABLE}
_PHRASES = {"".join(text for text, _ in segments) + ".": label for label, segments, _ in _TABLE}
WORLDS: dict[str, dict] = {}
for _label, _segments, _aliases in _TABLE:
    for _alias in [_label, *_aliases]:
        WORLDS[normalize(_alias)] = {"label": _label, "kind": "world", "phrase": "".join(t for t, _ in _segments) + "."}
DEFAULT_WORLD = WORLDS["country road"]

# Setting verbs only. "make it" needs an article ("make it a desert"), so "make it night" never matches.
# "turn it into", "change it to" and "make this a" name the whole scene ("it"/"this"), never one object, so
# "turn the tree into a lighthouse" stays an object transform for the slow reader.
_SETTING_RX = re.compile(
    r"^(?:please\s+)?(?P<verb>make (?:it|this)(?=\s+an?\b)|turn (?:it|this) into|change (?:it|this) (?:in)?to|"
    r"take me (?:back )?to|go (?:back )?to|switch to|"
    r"set (?:it|the scene|everything) (?:in|on|at)|move (?:it|us|me|the scene|everything) (?:back )?to|"
    r"change the (?:scene|setting|world|location|background|place|scenery) to)\s+(?:(a|an|the)\s+)?(.+)$")
# "take me to the beach at night": a known place plus season/weather/time words only.
_PLACE_RX = re.compile(r"^(?P<place>" + "|".join(sorted(map(re.escape, WORLDS), key=len, reverse=True)) +
                       r")\s+(?:at|during|in|with|under)\s+(?P<rest>.+)$")
_REST_STOP = {"a", "an", "the", "and", "of", "full", "heavy", "light", "some"}
_BLOCKED_RX = re.compile(r"\b(?:don't|dont|do not|not|never|instead of|into|replace|swap)\b")
_FILLER_TAIL_RX = re.compile(r"(?:\s+\b(?:please|now|instead|for me|again)\b)+$")
_VAGUE = {"it", "one", "nice", "something", "somewhere", "place", "there", "here", "that", "this", "good", "cool"}
_MAX_PLACE_WORDS = 6


def world_op(text: str) -> dict | None:
    """An add op of kind "world" for an explicit setting command, else None.

    Known places use Ishaan's sentences. Unknown places (1-6 words, no dictionary word) fall back to the spoken
    place. Season/weather/time/object words, negation and transforms return None for the reader to handle.
    """
    t = normalize(text)
    m = _SETTING_RX.match(t)
    # Checked after the verb, so the verb's own "into" is allowed; "into" later in the sentence still blocks.
    if not m or _BLOCKED_RX.search(t[m.end("verb"):]):
        return None
    article, place = m.group(2), _FILLER_TAIL_RX.sub("", m.group(3)).strip()
    if not place:
        return None
    if place in WORLDS:
        return {"op": "add", **WORLDS[place], "strength": 1.0}
    combo = _place_with_conditions(place)
    if combo:
        return combo
    if m.group("verb").startswith(("make", "turn", "change it", "change this")):
        return None  # "make it a little spooky", "turn it into a painting": a look, not a place: known places only
    words = place.split()
    if KEYWORD_RX.search(place) or len(words) > _MAX_PLACE_WORDS or set(words) <= _VAGUE:
        return None  # "make it a dog", "a desert at night", "misty grove": the reader handles dictionary words
    phrase = f"{article} {place}." if article else f"{place}."
    return {"op": "add", "label": place, "kind": "world", "phrase": phrase, "strength": 1.0}


def _place_with_conditions(place: str):
    """{"op": "batch", ...} for a known place plus season/weather/time words ("beach at night"), else None.
    Carries "label" so callers can show the world as the keyword; the batch is one undo step."""
    m = _PLACE_RX.match(place)
    if not m:
        return None
    rest = m.group("rest")
    matches = list(KEYWORD_RX.finditer(rest))
    leftover = KEYWORD_RX.sub(" ", rest).split()
    if not matches or any(w not in _REST_STOP for w in leftover):
        return None
    entries = [KEYWORDS[x.group(0)] for x in matches]
    if any(e["kind"] not in ("time", "weather", "season") for e in entries):
        return None
    world = WORLDS[m.group("place")]
    ops = [{"op": "add", **world, "strength": 1.0}] + [{"op": "add", **e, "strength": 1.0} for e in entries]
    return {"op": "batch", "operations": ops, "label": world["label"]}


def world_phrase(snapshot: dict) -> str:
    """The active world's phrase (most recent world change), else the default world."""
    worlds = [c for c in snapshot.get("changes", []) if c.get("kind") == "world" and c.get("strength", 1.0) > 0]
    return worlds[-1]["phrase"] if worlds else DEFAULT_WORLD["phrase"]


def world_with_overrides(world_change: dict | None, changes: list) -> str | None:
    """The active world's phrase with conflicting BUILT-IN sky/time/weather segments removed.

    A segment is dropped when an explicit change of the same kind (time, weather, season) names a different
    label: "desert" + "night" loses the afternoon sun but keeps the heat haze. A sunlit time (day, noon) also
    drops built-in weather that is not sunlit, unless an explicit weather change says the weather: "snowy
    mountain road" + "sunny" loses the overcast light and the falling snow but keeps the snowy road and forest.
    Other changes and the scene state are untouched. Unknown (fallback) world phrases are returned literally.
    None if no world.
    """
    if world_change is None:
        return None
    phrase = world_change["phrase"]
    segments = _SEGMENTS.get(_PHRASES.get(phrase))
    if segments is None:
        return phrase
    active = [c for c in changes if c.get("kind") in ("time", "weather", "season") and c.get("strength", 1.0) > 0]
    explicit = {c["kind"]: c.get("label") for c in active}  # zero strength = no effect
    implied = {kind: values for c in active for kind, values in _IMPLIED.get(c.get("label"), {}).items()
               if kind not in explicit}
    kept = [text for text, tags in segments
            if all(explicit.get(k, v) == v and v in implied.get(k, {v}) for k, v in tags.items())]
    return "".join(kept) + "."


def change_wording(world_change: dict | None, change: dict) -> str:
    """The phrase a change is written with in this world: the world's own light sentence for it if it has one
    ("sunny" on the snowy mountain road), else the change's phrase."""
    world = _PHRASES.get(world_change["phrase"]) if world_change else None
    return _LIGHT_WORDING.get((world, change.get("label")), change["phrase"])


_WORLD_ALIAS_RX = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, WORLDS), key=len, reverse=True)) + r")\b")
_OAK_SUBJECT = "a wide shot of a single oak tree"


def compose_base(base: str, world: str) -> str:
    """The base with its world replaced by `world` (a world phrase), never touching the subject.

    Road scenes (is_road_scene) never come here: the composer writes them with car_prompt.
    - DEFAULT_BASE (oak in a meadow): "a wide shot of a single oak tree; " + world.
    - A subject-only or generic base without its own world words: base + "; " + world.
    - A base that names its own world in an unknown way: ValueError (no guessing which words to cut).
    """
    if base == DEFAULT_BASE:
        return f"{_OAK_SUBJECT}; {world}"
    if _WORLD_ALIAS_RX.search(normalize(base)):
        raise ValueError(f"base names its own world; cannot replace it safely: {base!r}")
    return f"{base}; {world}"


def _sentence(phrase: str) -> str:
    phrase = phrase.strip().rstrip(".")
    return phrase[:1].upper() + phrase[1:] + "."


def car_prompt(world: str, objects=(), modifiers=(), subject: str = CAR_SUBJECT) -> str:
    """Ishaan's car template for a world phrase: subject with its road, camera, objects, scene and light, modifiers,
    style. Object and modifier phrases (already strength-worded) are one sentence each, word for word."""
    road, _, rest = world.strip().rstrip(".").partition(". ")
    parts = [f"{subject}{_REAR}{road}{_FORWARD}", CAMERA, *map(_sentence, objects),
             *([rest + "."] if rest else []), *map(_sentence, modifiers), STYLE]
    return " ".join(parts)


# The car scene: Ishaan's template on the country road. A base that is the template for any known world is a road
# scene too, and that world is its own (so "make it night" drops its built-in sunshine).
CAR_BASE = car_prompt(DEFAULT_WORLD["phrase"])
_ROAD_SCENES = {car_prompt(phrase): label for phrase, label in _PHRASES.items()}
SNOWY_WORLD = WORLDS["snowy mountain road"]


# A car base that names its road as a known world, as the dictionary path keeps "new scene of a red sports car
# driving along a country road". It is a road scene too: its world is that place, and a world change replaces it
# (compose_base could not, as the place is in the base), so "make it a desert" works when the LLM reader times out.
_CAR_ON_ROAD_RX = re.compile(
    r"^(?P<subject>(?:a|an|the)\s+(?:[a-z'-]+\s+){0,4}?car)\s+"
    r"(?:driving|drives|speeding|speeds|racing|races|cruising|cruises|going|goes|moving|moves)(?:\s+fast)?\s+"
    r"(?:along|on|through|across|down)\s+(?:(?:a|an|the)\s+)?"
    r"(?P<place>" + "|".join(sorted(map(re.escape, WORLDS), key=len, reverse=True)) + r")$")


def _car_on_road(base: str):
    return _CAR_ON_ROAD_RX.match(normalize(base)) if isinstance(base, str) else None


def is_road_scene(base: str) -> bool:
    return base in _ROAD_SCENES or _car_on_road(base) is not None


def road_scene_subject(base: str) -> str:
    """The capitalised subject of a road scene's template: the car the base names, else CAR_SUBJECT."""
    m = _car_on_road(base)
    if m is None:
        return CAR_SUBJECT
    subject = m.group("subject")
    return subject[:1].upper() + subject[1:]


def _own_world(base: str) -> dict:
    m = _car_on_road(base)
    return WORLDS[m.group("place")] if m else WORLDS[_ROAD_SCENES[base]]


def road_scene_world(base: str, changes: list) -> tuple[dict, list]:
    """The world change a road scene shows, and the changes left to render after it.

    A winter season on a road scene IS the snowy mountain road (Ishaan: appending a season to the car prompt does not
    change the picture, swapping the world does). A new world clears the season in state.py, so an active winter is
    always newer than the active world, and it wins. Without a world change, the base's own world is shown.
    """
    active = [c for c in changes if c.get("strength", 1.0) > 0]
    winter = next((c for c in active if c["kind"] == "season" and c["label"] == "winter"), None)
    if winter is not None:
        return SNOWY_WORLD, [c for c in changes if c is not winter]
    worlds = [c for c in active if c["kind"] == "world"]
    return (worlds[-1] if worlds else _own_world(base)), changes

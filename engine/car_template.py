"""The car template: the prompt of a vehicle-on-road scene, composed by the engine (used by final_reader.py).

Measured on GPU by the steering owner (board, 2026-11-14): appending a place to a short car prompt does not change
the world ("..., desert" gave green fields), but this full-sentence template does, and with the rear view named
it kept the car seen from behind in 10 of 10 runs:

    {Subject}, seen from directly behind with its rear lights facing the camera, drives fast along {road}, moving
    forward away from the camera. {CAMERA} {objects} {scene} {light} {STYLE}

{CAMERA} names the subject's vehicle word ("the car" for a car, which is Ishaan's sentence exactly; "the
motorcycle" for a motorcycle). {objects} are the phrases of the scene's objects, one sentence each, word for word,
so a lion added on a car scene stays in the prompt (none in Ishaan's measured prompts).

The subject sentence (apart from its road) and the camera sentence are the same, word for word, in every prompt
of a vehicle scene. A change of world swaps road, scene and light together.

Scene state: a vehicle scene has a bare vehicle subject as its base ("a red sports car") and its world as one change
of kind world (label "desert", "winter", ...). The engine never keeps the place in the base, so a change of world
never has to edit the base. (On the dictionary path, commands/world.py reads a car base that names a known road,
"a red sports car driving along a country road", as a road scene of its own.)

Routes (Route) for one final sentence:
- MEASURED: the sentence names exactly one of the measured worlds (WORLDS) and nothing else, on a vehicle scene
  with no objects, or it starts a new vehicle scene with a measured world (or no world: country). The engine
  builds the operations and the prompt; no model call.
- MODEL: a vehicle scene, but no measured world fits. Sonnet writes the whole prompt in the template form, and the
  engine checks the subject, road and camera sentences (template_problems).
- PLAIN: not a vehicle scene: the usual final reader path.
"""

import dataclasses
import enum
import re

from commands import world as commands_world
from commands.composer import strength_phrase
from commands.dictionary import normalize
from commands.world import CAMERA, FORWARD, REAR, STYLE  # the one copy of Ishaan's template sentences


@dataclasses.dataclass(frozen=True)
class World:
    name: str   # the label of its world change, shown on the chip
    road: str   # a noun phrase, no final period
    scene: str  # one sentence
    light: str  # one sentence


def _measured(name: str, label: str) -> World:
    """A measured world from commands/world.py, the one copy of Ishaan's text: its phrase is "{road}. {scene} {light}"."""
    road, _, rest = commands_world.WORLDS[label]["phrase"].partition(". ")
    scene, light = re.split(r"(?<=\.)\s+", rest)
    return World(name, road, scene, light)


# Ishaan's measured worlds (board, 2026-11-14), character for character. "country" is his earlier tested car
# prompt's world, the start of the demo chain. The five that commands/ also uses come from commands/world.py; the
# other three are only used here.
WORLDS = {w.name: w for w in [
    _measured("country", "country road"),
    _measured("winter", "snowy mountain road"),
    _measured("desert", "desert"),
    _measured("neon city", "neon city"),
    _measured("ocean coast", "ocean coast"),
    World("sunset", "a straight open road across rolling grassland",
          "Golden grass fields and a few lone trees on both sides, long shadows stretching across the road.",
          "The sun setting low on the horizon ahead, a glowing orange and pink sky, warm backlight."),
    World("autumn", "a winding country road through an autumn forest",
          "Trees with bright orange, red and yellow leaves line both sides, fallen leaves scattered across the road "
          "and swirling behind the car.",
          "Low warm golden afternoon sun, light mist between the trees."),
    World("rain", "a wet country road in heavy rain",
          "Rain pours down, puddles and spray on the dark asphalt, water kicked up behind the tyres, green fields and "
          "trees blurred by the rain.",
          "Dark grey storm clouds, dim diffuse light, glistening reflections on the road."),
]}
START_WORLD = "country"

# Spoken words -> measured world. "night" is the neon city: a plain night highway turned to orange streaks by
# chunk 29 on GPU (Ishaan), so there is no night world of its own.
WORLD_WORDS = {
    "winter": "winter", "wintry": "winter", "snow": "winter", "snowy": "winter", "snowing": "winter",
    "snowfall": "winter", "blizzard": "winter",
    "desert": "desert",
    "city": "neon city", "neon": "neon city", "cyberpunk": "neon city", "night": "neon city",
    "nighttime": "neon city",
    "beach": "ocean coast", "ocean": "ocean coast", "coast": "ocean coast", "coastal": "ocean coast",
    "sea": "ocean coast", "seaside": "ocean coast",
    "sunset": "sunset", "evening": "sunset", "dusk": "sunset",
    "autumn": "autumn", "autumnal": "autumn", "fall": "autumn",
    "rain": "rain", "rainy": "rain", "raining": "rain", "rainstorm": "rain",
    "country": "country", "countryside": "country",
}
# Words a world command may also have. Anything else (negation, "a little", an object, a second place) is not a
# plain world change and goes to the model.
_FILLER = frozenset("""
    make it it's its this a an the take me us to go back turn into switch change set in at now please let lets
    let's drive driving through along on scene world place setting road highway street time season and is be i
    want like we can you
""".split())
# The rest of a new scene's description after its vehicle: how it moves and the road words.
_ROAD_FILLER = _FILLER | frozenset("""
    drives driving speeding speeds racing races cruising cruises going goes moving moves fast quickly along on
    through across down over two lane two-lane straight long open highway road street
""".split())

VEHICLE_WORDS = ("car", "truck", "van", "bus", "motorcycle", "motorbike", "jeep", "taxi", "vehicle")
# A bare vehicle subject: an article, at most four describing words, the vehicle. "a red sports car".
_SUBJECT = re.compile(r"^(?P<subject>(?:a|an|the|one)\s+(?:[a-z'-]+\s+){0,4}?(?:" + "|".join(VEHICLE_WORDS) +
                      r"))(?:\s+(?P<rest>.*))?$")
# What may follow the subject in a vehicle base: how it moves or where it is (not "a car park", not "with stripes").
_REST_START = frozenset("""
    driving drives speeding speeds racing races cruising cruises going goes moving moves on along through across
    down in at
""".split())
_NEW_SCENE = re.compile(r"^(?:(?:a|start a|make a|give me a)\s+)?new scene (?:of|with)\s+(?P<desc>.+)$|"
                        r"^start over with\s+(?P<desc2>.+)$")


class Route(enum.Enum):
    MEASURED = "measured"  # the engine composes a measured world: no model call
    MODEL = "model"        # a vehicle scene with no measured world: Sonnet writes the template prompt
    PLAIN = "plain"        # not a vehicle scene: the usual path


@dataclasses.dataclass(frozen=True)
class Plan:
    route: Route
    subject: str | None = None  # the bare vehicle subject, lowercase ("a red sports car")
    world: World | None = None  # MEASURED only
    new_scene: bool = False     # the sentence starts a new scene


def split_vehicle(text) -> tuple[str, str] | None:
    """(subject, rest) when `text` is a vehicle subject, optionally followed by how it moves or where it is;
    else None. "a red sports car driving along a country road" -> ("a red sports car", "driving along ...")."""
    if not isinstance(text, str):
        return None
    m = _SUBJECT.match(normalize(text))
    if m is None:
        return None
    rest = m.group("rest") or ""
    if rest and rest.split()[0] not in _REST_START:
        return None
    return m.group("subject"), rest


def vehicle_subject(base) -> str | None:
    """The bare vehicle subject of a scene base, or None when the scene is not a vehicle on a road."""
    parts = split_vehicle(base)
    return parts[0] if parts else None


def is_bare(snapshot: dict) -> bool:
    """True when the base is the bare vehicle subject (no place in it), as the engine keeps it."""
    subject = vehicle_subject(snapshot.get("base"))
    return subject is not None and normalize(snapshot["base"]) == subject


class Words(enum.Enum):
    NO_WORLD = "no world"    # only filler words
    ONE_WORLD = "one world"  # one measured world, and only filler besides
    OTHER = "other"          # two worlds, or a word that is neither a world word nor filler


def world_of_words(words: list[str], filler: frozenset) -> tuple[Words, World | None]:
    """Which measured world the words name: (ONE_WORLD, world), (NO_WORLD, None) or (OTHER, None)."""
    named = {WORLD_WORDS[w] for w in words if w in WORLD_WORDS}
    if any(w not in WORLD_WORDS and w not in filler for w in words) or len(named) > 1:
        return Words.OTHER, None
    return (Words.ONE_WORLD, WORLDS[named.pop()]) if named else (Words.NO_WORLD, None)


def has_objects(snapshot: dict) -> bool:
    return any(c["kind"] == "extra" and c["strength"] > 0 for c in snapshot["changes"])


def plan(text: str, snapshot: dict) -> Plan:
    """The route for one final sentence on `snapshot`."""
    t = normalize(text)
    m = _NEW_SCENE.match(t)
    if m is not None:
        parts = split_vehicle(m.group("desc") or m.group("desc2"))
        if parts is None:
            return Plan(Route.PLAIN, new_scene=True)
        subject, rest = parts
        match, world = world_of_words(rest.split(), _ROAD_FILLER)
        if match is Words.OTHER:
            return Plan(Route.MODEL, subject, new_scene=True)
        return Plan(Route.MEASURED, subject, world or WORLDS[START_WORLD], new_scene=True)
    subject = vehicle_subject(snapshot.get("base"))
    if subject is None:
        return Plan(Route.PLAIN)
    match, world = world_of_words(t.split(), _FILLER)
    if match is Words.ONE_WORLD and not has_objects(snapshot):
        return Plan(Route.MEASURED, subject, world)
    return Plan(Route.MODEL, subject)


def measured_operations(p: Plan, snapshot: dict) -> list[dict]:
    """The operations of a MEASURED plan, as commands/state.py operations.

    The start world (country) is the bare vehicle scene itself, with no world chip: the new scene sentence maps
    that bare state to the country template, so removing a later world chip goes back to a state that still has
    the template (a world chip on it would leave the bare state with no template prompt after its removal).
    On a bare vehicle scene, another world removes the time changes (the world brings its own light) and adds the
    world, which clears season, weather and the old world in commands.state; country removes every change that
    is not an object. Otherwise (a new scene, or a base that names its own place): a new scene of the bare
    subject, then the world unless it is country."""
    start = p.world.name == START_WORLD
    add = [] if start else [{"op": "add", "label": p.world.name, "kind": "world", "phrase": f"{p.world.road}.",
                             "strength": 1.0}]
    if p.new_scene or not is_bare(snapshot):
        return [{"op": "new_scene", "base": p.subject}, *add]
    drop = ("world", "season", "weather", "time") if start else ("time",)
    return [*({"op": "remove", "id": c["id"]} for c in snapshot["changes"] if c["kind"] in drop), *add]


def camera_sentence(subject: str) -> str:
    """CAMERA for the subject's vehicle word: Ishaan's sentence exactly for a car."""
    noun = subject.split()[-1]
    return CAMERA.replace("behind the car", f"behind the {noun}").replace("keeping the car", f"keeping the {noun}")


def _object_sentences(objects) -> str:
    return " ".join(f"{o[:1].upper()}{o[1:].rstrip(' .,;')}." for o in objects)


def world_prompt(subject: str, world: World, objects=()) -> str:
    """The template, with the subject capitalised and one sentence per object phrase after the camera sentence."""
    middle = " ".join(p for p in (_object_sentences(objects), world.scene, world.light) if p)
    return (f"{subject[:1].upper()}{subject[1:]}{REAR}{world.road}{FORWARD} {camera_sentence(subject)} "
            f"{middle} {STYLE}")


def scene_objects(snapshot: dict) -> list[str]:
    """The phrases of the scene's objects that are on, in the order they were added, with commands.composer's
    strength wording (strength_phrase), so weakening a chip changes the prompt."""
    return [strength_phrase(c["phrase"], c["strength"]) for c in snapshot["changes"]
            if c["kind"] == "extra" and c["strength"] > 0]


def current_world(snapshot: dict) -> World | None:
    """The measured world of a vehicle scene: its world change's label when it is one of WORLDS (none: country)."""
    worlds = [c for c in snapshot["changes"] if c["kind"] == "world" and c["strength"] > 0]
    if not worlds:
        return WORLDS[START_WORLD] if is_bare(snapshot) else None
    return WORLDS.get(worlds[-1]["label"])


# --- the template check of a free prompt ----------------------------------------------------------------

# A road slot without one of these words left the car on the sand or in the water (live acceptance run, 2026-11-15).
DRIVABLE = frozenset({"road", "roads", "highway", "street", "lane", "avenue", "boulevard", "bridge", "motorway",
                      "freeway", "causeway", "drive", "promenade", "track", "asphalt"})
_TEMPLATE_HEAD = re.compile(r"^(?P<subject>[A-Z][^.!?]*?)" + re.escape(REAR) + r"(?P<road>[^.!?]+?)" + re.escape(FORWARD)
                            + r" ")


def template_parts(prompt) -> tuple[str, str] | None:
    """(subject, road) when `prompt` starts as the car template: the subject sentence "<Subject>, seen from directly
    behind with its rear lights facing the camera, drives fast along <road>, moving forward away from the camera."
    and then the camera sentence for that subject's vehicle word, word for word. Else None."""
    if not isinstance(prompt, str):
        return None
    m = _TEMPLATE_HEAD.match(prompt)
    if m is None:
        return None
    subject = m.group("subject")
    subject = subject[:1].lower() + subject[1:]
    if not prompt[m.end():].startswith(camera_sentence(subject)):
        return None
    return subject, m.group("road")


def is_template(prompt) -> bool:
    """True when `prompt` is a car template prompt (template_parts): the final reader then keeps it one."""
    return template_parts(prompt) is not None


# A sentence that asks for another camera or view of the car ("change the point of view to be facing the ocean,
# i want to see the car's side profile"): the template's rear view and camera sentence then yield to the model's.
_VIEW_RX = re.compile(r"\b(?:point of view|pov|viewpoint|view|camera|angle|perspective|profile|side on|from the "
                      r"(?:side|front|left|right|above)|bird'?s[ -]eye|aerial|drone|overhead|close[ -]?up|zoom|"
                      r"wider|closer|shot|pan|framing)\b")


def asks_view_change(sentence) -> bool:
    """True when the sentence asks for another camera position or view."""
    return isinstance(sentence, str) and bool(_VIEW_RX.search(normalize(sentence)))


def _drivable(text: str) -> bool:
    return bool(set(re.findall(r"[a-z]+", text.lower())) & DRIVABLE)


def template_problems(prompt: str, view_change: bool = False) -> list[str]:
    """Why a free prompt for a car scene is not the car template any more; [] when it is. Any subject may drive
    (the user may recolour or swap the car); the road must name a paved surface (DRIVABLE). With `view_change`
    (the sentence asks for another camera or view, asks_view_change) the rear view and the camera sentence may
    change: the prompt need only still name a drivable road under the vehicle."""
    if view_change:
        if _drivable(prompt):
            return []
        return ["the vehicle must still drive on a paved road the prompt names (road, highway, street, lane, bridge)"]
    parts = template_parts(prompt)
    if parts is None:
        return ['the scene is a car scene: keep the first sentence "<Subject>' + REAR + '<road>' + FORWARD +
                '" and then the camera sentence word for word (only the road may change), or set "new_scene" true '
                'when the user asks for the vehicle to go']
    if not _drivable(parts[1]):
        return [f'the road "{parts[1]}" must name the paved road the vehicle drives on (road, highway, street, '
                'lane, bridge), or the vehicle is drawn on sand, water or grass']
    return []


def check_prompt(prompt: str, subject: str, objects=()) -> bool:
    """True when `prompt` is the template for `subject`: the subject sentence (any road), the camera sentence
    word for word as the second sentence, the object sentences word for word, then exactly one scene and one
    light sentence and the style."""
    head = f"{subject[:1].upper()}{subject[1:]}{REAR}"
    if not prompt.startswith(head) or not prompt.endswith(" " + STYLE):
        return False
    first, sep, rest = prompt[len(head):].partition(FORWARD + " ")
    camera = camera_sentence(subject)
    if not sep or re.search(r"[.!?]", first) or not rest.startswith(camera + " "):
        return False
    middle = rest[len(camera) + 1:-len(STYLE) - 1]
    said = _object_sentences(objects)
    if said and not middle.startswith(said + " "):
        return False
    return _sentence_count(middle[len(said):]) == 2


def _sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


INSTRUCTIONS = """
The car template. When the main subject is a vehicle driving along a road, the prompt has this form, measured on
the GPU to keep the vehicle seen from behind and driving forward (anything else made it drive backwards):
  "<Subject>{rear}<road>{forward} {camera} <one sentence per object> <scene sentence> <light sentence> {style}"
- The camera sentence names the subject's vehicle word: "the car" for a car, "the truck" for a truck, and so on.
- "car_scene" true means the prompt on screen has this form. Keep the subject sentence and the camera sentence word
  for word; change only <road> when the place changes (and the subject's own words only when the user changes the
  vehicle, "make the car blue"). Objects, place, weather and light go in the sentences after the camera sentence.
- "take me to X", "go to X", "drive to X", "turn it into X" on a car scene: the vehicle keeps driving, now on a road
  in X. <road> always names the paved road it drives on (road, highway, street, lane, bridge): for a beach it is
  "a beachfront road with pale sand and waves beside it", never sand or water.
- "view_change" true means the user asks for another camera position or view ("I want to see the car's side
  profile"): then the rear view and the camera sentence yield. Write a camera sentence that matches the new view
  ("a smooth tracking shot from the side, the camera moving alongside the car at the same speed, the car seen in side
  profile driving from left to right along the coastal road, the ocean behind it"), keep the road and keep the
  vehicle's direction of travel consistent with that camera. The vehicle still drives on a named road.
- The vehicle goes only when the user says so ("remove the car", "just the beach, no car"): then "new_scene" is
  true and the prompt describes X without the vehicle.
- More or less of the current weather or season ("a little more snow", "less rain"): the same road, and the scene
  and light sentences rewritten with visibly more or less of it ("heavy snow falling thickly, deep snowdrifts";
  "a light dusting of snow").
- A new prompt whose main subject is a vehicle driving along a road ("a city at night with a car driving on the
  road") is written in this form too; the other objects become object sentences after the camera sentence.
Measured worlds (road | scene sentence | light sentence):
{examples}
"""


def instructions() -> str:
    return INSTRUCTIONS.format(rear=REAR, forward=FORWARD, camera=camera_sentence("a car"), style=STYLE,
                               examples="\n".join(f"- {w.name}: {w.road} | {w.scene} | {w.light}"
                                                  for w in WORLDS.values()))


def parse_world(prompt, subject: str, objects=()) -> World | None:
    """The road, scene and light of a template prompt for `subject` and `objects` (name "given"), or None when
    `prompt` is not one (check_prompt)."""
    if not isinstance(prompt, str) or not check_prompt(prompt, subject, objects):
        return None
    head = f"{subject[:1].upper()}{subject[1:]}{REAR}"
    road, _, rest = prompt[len(head):].partition(FORWARD + " ")
    middle = rest[len(camera_sentence(subject)) + 1:-len(STYLE) - 1][len(_object_sentences(objects)):].strip()
    scene, light = re.split(r"(?<=[.!?])\s+", middle)
    return World("given", road, scene, light)

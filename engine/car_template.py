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
- MODEL: a vehicle scene, but no measured world fits. Sonnet gives the operations plus road, scene and light, and
  the engine checks them (field_problems) and puts them in the template.
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
    MODEL = "model"        # a vehicle scene with no measured world: Sonnet fills road, scene and light
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


# --- the model's fields ---------------------------------------------------------------------------------

MAX_ROAD_WORDS = 14
# A road slot without one of these words left the car on the sand or in the water (live acceptance run, 2026-11-15).
DRIVABLE = frozenset({"road", "roads", "highway", "street", "lane", "avenue", "boulevard", "bridge", "motorway",
                      "freeway", "causeway", "drive", "promenade", "track", "asphalt"})
MAX_SCENE_WORDS = 40
MAX_LIGHT_WORDS = 20
# The subject and camera sentences are the engine's; the fields never move the camera or re-frame the shot.
CAMERA_WORDS = frozenset("""
    camera cameras shot shots tracking frame framing framed zoom zooms zooming pan pans panning lens angle
    aerial drone closeup close-up viewer centred centered
""".split())


def _sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def field_problems(reply: dict) -> list[str]:
    """Why the model's road, scene and light cannot go in the template; [] when they can."""
    out = []
    fields = {k: reply.get(k) for k in ("road", "scene", "light")}
    for k, v in fields.items():
        if not isinstance(v, str) or not v.strip():
            out.append(f'"{k}" is missing: a vehicle scene needs "road", "scene" and "light"')
        elif "\n" in v or "\r" in v:
            out.append(f'"{k}" has a line break')
    if out:
        return out
    road, scene, light = (" ".join(fields[k].split()) for k in ("road", "scene", "light"))
    if not re.match(r"^(?:a|an|the)\s", road) or re.search(r"[.!?;]", road):
        out.append('"road" must be a noun phrase like "a snowy mountain road", with no period')
    elif not set(re.findall(r"[a-z]+", road.lower())) & DRIVABLE:
        out.append('"road" must name the paved road the vehicle drives on (road, highway, street, lane, bridge), '
                   'or the vehicle is drawn on sand, water or grass')
    for k, v, limit in (("road", road, MAX_ROAD_WORDS), ("scene", scene, MAX_SCENE_WORDS),
                        ("light", light, MAX_LIGHT_WORDS)):
        if len(v.split()) > limit:
            out.append(f'"{k}" has {len(v.split())} words, the limit is {limit}')
        bad = sorted(set(re.findall(r"[a-z-]+", v.lower())) & CAMERA_WORDS)
        if bad:
            out.append(f'"{k}" has camera or framing words {", ".join(map(repr, bad))}: the camera sentence is fixed')
    for k, v in (("scene", scene), ("light", light)):
        if not v[:1].isupper() or not v.endswith(".") or _sentence_count(v) != 1:
            out.append(f'"{k}" must be one sentence that starts with a capital letter and ends with a period')
    return out


def model_world(reply: dict) -> World | None:
    """The model's road, scene and light as a World (name "model"), or None when field_problems finds any."""
    if field_problems(reply):
        return None
    return World("model", *(" ".join(reply[k].split()) for k in ("road", "scene", "light")))


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


INSTRUCTIONS = """
Vehicle scenes. When the NEW scene, after your operations, has a vehicle as its main subject (its base names a car,
truck, van, bus, motorcycle, jeep or taxi), the engine writes the video prompt from a fixed template:
  "<Subject>, seen from directly behind with its rear lights facing the camera, drives fast along <road>, moving
  forward away from the camera. The camera follows directly behind the car ... rushes toward the viewer. <scene>
  <light> Cinematic, highly detailed, realistic, sharp focus, motion blur on the road."
The engine writes the subject and camera sentences itself. For such a scene, reply with three more fields and
leave "details" empty: {"blocked": ..., "operations": [...], "details": "", "road": "...", "scene": "...", "light": "..."}
- "road": a noun phrase for the road of the new scene, starting with "a", at most 14 words, no period. It always
  names the paved road the vehicle drives on (road, highway, street, lane, bridge): the video model draws the vehicle
  on whatever this says. For a beach it is a road beside the beach ("a beachfront road with pale sand and waves
  beside it"), never sand or water.
- "take me to X", "go to X", "drive to X", "turn it into X" on a vehicle scene: the vehicle keeps driving, now on a
  road in X. The vehicle goes only when the user says so ("remove the car", "just the beach, no car"): then reply
  with ONE new_scene whose base describes X itself ("a wide sandy beach with gentle turquoise waves") and no add
  of kind world (a base that names a place cannot take another place).
- "scene": ONE sentence (at most 40 words) about what lines both sides of the road: landscape, buildings, weather
  you can see. Never the camera, the shot or the frame. The scene's objects (kind extra) are added by the engine
  from their phrases, word for word: do not describe them again.
- "light": ONE short sentence (at most 20 words) about the light and the sky.
- The world of the new scene: its place, season, weather and time of day. A change of place replaces the road,
  the scene and the light together. If "car_template" in the user message gives the current road, scene and
  light, keep what the sentence does not change.
- More or less of the current world ("a little more snow", "a lot more snow", "less rain" on the snowy road or in
  the rain): ONE set_strength on that world change (never a new add: it would replace the world), the same road,
  and "scene" and "light" rewritten with visibly more or less of it ("heavy snow falling thickly, deep snowdrifts"
  for more; "a light dusting of snow" for less). The engine draws only what road, scene and light say.
- For a new scene of a vehicle, use new_scene with the base being only the vehicle ("a blue pickup truck") and an
  add of kind world for its place.
Measured examples (road | scene | light):
{examples}
"""


def instructions() -> str:
    return INSTRUCTIONS.replace("{examples}", "\n".join(
        f"- {w.name}: {w.road} | {w.scene} | {w.light}" for w in WORLDS.values()))


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

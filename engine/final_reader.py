"""The final reader (--final-reader sonnet): one Claude Sonnet 5.5 call per FINAL sentence that returns both the
scene operations and the rich prompt the video model sees.

    reader = load("sonnet")
    reader(text, snapshot) -> None                                  use the dictionary path (fallback)
                            | {"blocked": True}                     refuse it: no steer, shown as unmatched
                            | {"operations": [op, ...], "prompt": str}

`snapshot` is commands.state.SceneState.snapshot() of the scene before the sentence. The operations use the
shapes of commands/state.py and are checked with slow_reader.validate, one by one, against that snapshot.
The prompt is built from the composed (canonical) prompt of the new scene and the model's details by a
PromptSpec: the engine's stand-in below, or commands.rewriter (SYSTEM_PROMPT, validate_rewrite) once it exists.

The model's reply, JSON: {"blocked": bool, "operations": [op, ...], "details": str}.
The validator gets {"details": str} only. A clause of the details it rejects is logged with the reason and cut;
details with nothing left keep the operations and use the composed prompt. An unusable reply is asked again once,
with the reasons, when there is time (Verdict, RETRY_VERDICTS).

Vehicle scenes (car_template.py): a scene whose subject is a vehicle on a road gets Ishaan's measured car template.
A sentence naming one measured world ("make it winter", "make it night" -> neon city) is composed by the engine
with no call (Outcome.TEMPLATE); any other sentence on a vehicle scene asks Sonnet for road, scene and light, which
the engine checks and puts in the template, so the subject and camera sentences are always the engine's.

Places: "take me to the desert" is one change of kind world (commands/world.py) that replaces the old world and
keeps the other changes. Only an explicit "new scene of ..." resets. Season, weather and time adds get the
keyword table's label and kind ("sunny" is day, of kind time).

Content block: a small word list (BLOCKED_WORDS) is checked first, with no call at all; the model also flags
sexual, violent or hateful requests with "blocked": true in the same reply.

Determinism: the cache is keyed by (canonical prompt of the scene before, sentence). A result is chosen once:
a Sonnet result or a block is saved to a JSON file (written atomically) and never regenerated, also across
restarts. A timeout (final_timeout_s, 3.0 s by default), a failed call or a reply that does not validate
gives the fallback (None): the dictionary result and the short prompt. The fallback is final for that key for
the life of the process, and a late answer is dropped (late_dropped). It is not saved to disk, so a passing
outage does not disable a sentence for good.
"""

import concurrent.futures
import copy
import dataclasses
import enum
import hashlib
import importlib
import json
import logging
import os
import re
import statistics
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from . import car_template
from .foundry import FoundryMessages, settings_from_env
from .slow_reader import validate as validate_operation

log = logging.getLogger("engine.final_reader")

DEFAULT_TIMEOUT_S = 3.0      # Sonnet measured 1.8 s p50, 2.4 s max from the engine machine
HTTP_TIMEOUT_S = 10.0        # a call may finish after the wait; its answer is then dropped
WORKERS = 4
MAX_OPERATIONS = 6
DEFAULT_CACHE_FILE = "~/.cache/madhacks/final_reader.json"
DEFAULT_DEMO_PROMPTS = str(Path(__file__).resolve().parent / "demo_prompts.json")
LOG_CHARS = 100
FENCE = re.compile(r"^\s*```[A-Za-z]*\s*|\s*```\s*$")

# Checked before any call, as whole words. Short on purpose: the model's own "blocked" flag is the main check.
BLOCKED_WORDS = frozenset("""
    nude nudes naked nudity porn porno pornographic sex sexual sexy erotic orgasm genitals nsfw boobs
    gore gory decapitate decapitated beheading behead dismember dismembered mutilate mutilated torture
    corpse corpses massacre slaughter murder murdered suicide rape raped
    nazi swastika kkk lynch lynching
""".split())
_WORD = re.compile(r"[a-z]+")


class FinalReaderState(enum.Enum):
    OFF = "off"                  # --final-reader off: the dictionary and the nano slow reader, as before
    ON = "on"                    # every final goes to Sonnet first
    UNAVAILABLE = "unavailable"  # --final-reader sonnet, but something it needs is missing: as OFF


class Outcome(enum.Enum):
    SONNET = "sonnet"      # operations and a prompt from the model: saved
    BLOCKED = "blocked"    # refused (word list or the model): saved when the model said so
    FALLBACK = "fallback"  # timeout, error or invalid reply: the dictionary path; final in memory only
    PINNED = "pinned"      # from the demo prompts file: never called, never saved, never evicted
    TEMPLATE = "template"  # a measured car world composed by the engine: never called, never saved


@dataclasses.dataclass(frozen=True)
class Entry:
    outcome: Outcome
    operations: tuple = ()
    prompt: str | None = None

    def result(self) -> dict | None:
        if self.outcome in (Outcome.SONNET, Outcome.PINNED, Outcome.TEMPLATE):
            return {"operations": [dict(op) for op in self.operations], "prompt": self.prompt}
        if self.outcome is Outcome.BLOCKED:
            return {"blocked": True}
        return None


def blocked_word(text: str) -> str | None:
    return next((w for w in _WORD.findall(text.lower()) if w in BLOCKED_WORDS), None)


# --- the prompt part ------------------------------------------------------------------------------------

STYLE = "Cinematic, highly detailed, realistic, sharp focus."
# "motion blur on the road" only fits a vehicle on a road. Measured on GPU (2026-11-14): with that tail every
# winter and night frame of the oak meadow grew a road or a tyre track, so it is used only when the scene has a road.
VEHICLE_STYLE = "Cinematic, highly detailed, realistic, sharp focus, motion blur on the road."

# Measured by the steering owner with optical flow (board, 2026-11-14): a vehicle subject drives backwards in 1 of
# 3 seeds with a short prompt, and forward in 3 of 3 with this wording. Added to SUBJECT when the subject is a
# vehicle and does not already say how it moves. LANE_CUE is also added to the phrase of a vehicle added on a road.
VEHICLE_WORDS = ("car", "truck", "van", "bus", "motorcycle", "motorbike", "jeep", "taxi", "vehicle")
LANE_CUE = "white dashed lane markings rush toward the viewer"
FORWARD_MOTION = f"speeding forward, driving away from the camera, the camera follows directly behind, {LANE_CUE}"
ROAD_WORDS = frozenset({"road", "roads", "roadside", "highway", "highways", "street", "streets", "lane", "lanes",
                        "asphalt", "motorway", "freeway", "pavement"})
# Never in details when the scene has no road: they grew a road or tyre tracks on the meadow.
ROAD_DETAIL_WORDS = ROAD_WORDS | {"tyre", "tyres", "tire", "tires", "track", "tracks", "blur"}
# Never in details: the positions are in the subject and object phrases, word for word. Details that placed
# things again re-framed the shot on GPU ("make it night" moved the tree into the distance, cut the car off).
POSITION_WORDS = frozenset({"foreground", "background", "midground", "left", "right", "centre", "center",
                            "middle", "distance", "distant", "beside", "behind", "front", "closeup", "shot"})
SIZE_WORDS = frozenset({"large", "small", "big", "huge", "tiny", "giant", "massive", "medium", "tall", "little",
                        "enormous"})
PHRASE_POSITION_WORDS = POSITION_WORDS - {"shot", "closeup"} | {"near", "next", "beneath", "under", "above"}
# Characters of a phrase the final reader accepts: a positioned object (asked for under 200) or a world sentence
# from commands/world.py (Ishaan's measured worlds, about 250) is longer than the nano slow reader's 120.
FINAL_MAX_PHRASE = 400
# A new object's phrase stays word for word in every later prompt, so it keeps the tighter limit it is asked for.
OBJECT_MAX_PHRASE = 200
# A saved or pinned operation was checked with FINAL_MAX_PHRASE before with_lane_cue added ", " + LANE_CUE, so a
# cache hit is checked with room for it (the 120-character default refused the pinned racing car).
CACHED_MAX_PHRASE = FINAL_MAX_PHRASE + len(", " + LANE_CUE)

STANDIN_DETAILS = """The "details" field is the WORLD of the new scene, for the Wan2.1 text-to-video model.
The engine builds the video prompt as SUBJECT + OBJECTS + WORLD + STYLE. SUBJECT is the scene's base sentence,
OBJECTS are the phrases of the scene's extra changes (objects, animals, people), word for word, and STYLE is a
fixed style sentence; all three are added by the engine, so never repeat, rephrase or contradict them, and say
nothing about the camera, shots, lenses or editing.
Write WORLD in about 50 words (never more than 60), plain present tense, no lists, no line breaks:
- First the ground or road of that place, then two or three concrete things on both sides, then the light or sky.
- The season, weather and time of day of the new scene must appear in WORLD. Objects, animals and people are
  already in OBJECTS: do not describe them again, and never say where they are.
- A place REPLACES the world. It is never added to the old one: "make it a desert" turns the meadow into a
  desert, and nothing of the old landscape (grass, meadow, fields) is described. The objects and animals of the
  scene stay, now in the new place.
- Do not add any object, creature, person or vehicle the scene does not name, and no motion or events beyond
  what it says. Mention a road only when the scene has one.
- If changes conflict (sunlight and rain), blend them plausibly, such as sun breaking through rain clouds.
Examples of WORLD:
- start: "a two-lane country road through green fields and trees under a clear blue summer sky, bright sunshine."
- desert: "a long straight desert highway between red sandstone mesas and sand dunes, heat haze, orange sunset sky."
- winter: "a snowy mountain road between pine forests covered in thick white snow, snow falling, cold blue overcast light."
- neon city: "a wet city street at night in the rain, tall skyscrapers with glowing neon signs on both sides, reflections on the asphalt."
"""


def standin_validate(prompt: str, reply, objects=()) -> str | None:
    """SUBJECT + OBJECTS + WORLD + STYLE from the canonical prompt of the new scene (SUBJECT is its part before the
    first ";"), the phrases of its extra changes (`objects`, kept word for word right after SUBJECT so a lion
    stays the same lion in every later prompt) and the reply's details. The canonical prompt for empty details;
    None for details that are not one plain paragraph of at most 80 words."""
    details = reply.get("details") if isinstance(reply, dict) else None
    if not isinstance(details, str):
        return None
    details = details.strip()
    if not details:
        return prompt
    if "\n" in details or len(details.split()) > 80:
        return None
    base = prompt.split(";")[0].strip()
    subject = ", ".join([with_forward_motion(base), *objects])
    subject = subject[:1].upper() + subject[1:]
    world = details if details.endswith(".") else details + "."
    return f"{subject}, {world} {VEHICLE_STYLE if is_vehicle(base) and has_road(prompt) else STYLE}"


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def is_vehicle(subject: str) -> bool:
    """True when `subject` names a vehicle (VEHICLE_WORDS), as a whole word."""
    return bool(_words(subject) & set(VEHICLE_WORDS))


def has_road(text: str) -> bool:
    """True when `text` names a road (ROAD_WORDS), as a whole word."""
    return bool(_words(text) & ROAD_WORDS)


def with_forward_motion(subject: str) -> str:
    """`subject` plus FORWARD_MOTION when it names a vehicle and says nothing about its direction yet."""
    if is_vehicle(subject) and not _words(subject) & {"forward", "away", "behind", "reversing", "backwards"}:
        return f"{subject}, {FORWARD_MOTION}"
    return subject


def scene_prefix(scene: dict) -> str:
    """The fixed part of the prompt of `scene`, from the commands/ composer: the subject, then the season and
    weather phrases, then the phrases of the scene's objects word for word, then the rest (world, time).
    Measured on GPU: with the object before the winter phrase, "add a red car" lost the winter (green grass).
    The objects stay early and verbatim, so a positioned lion stays where it is in every later prompt.
    A world change clears season and weather in commands.state, so no dropped season comes back here.
    A vehicle subject gets FORWARD_MOTION. This is the text the details are appended to (and checked against)."""
    from commands.composer import compose
    parts = [p.strip() for p in compose(scene).split(";")]
    subject, rest = parts[0], parts[1:]
    conditions = [p for p in rest if any(p.endswith(ph) for ph in _scene_phrases(scene, ("season", "weather")))]
    objects = [p for p in rest if p not in conditions and any(p.endswith(ph) for ph in _scene_phrases(scene, ("extra",)))]
    others = [p for p in rest if p not in conditions and p not in objects]
    return "; ".join([with_forward_motion(subject), *conditions, *objects, *others])


@dataclasses.dataclass(frozen=True)
class PromptSpec:
    """How the prompt part is asked for and checked. `validate(scene, reply)` gets the snapshot of the new scene
    and {"details": str}, and gives the text the model sees, or None to reject the details. With commands.rewriter,
    `prefix` and `check` (validate_rewrite itself) let the engine find out why details are rejected and repair them."""
    name: str
    details_instructions: str
    validate: Callable[[dict, dict], str | None]
    prefix: Callable[[dict], str] | None = None
    check: Callable[[str, dict], str | None] | None = None


def _scene_phrases(scene: dict, kinds: tuple[str, ...]) -> list[str]:
    """The phrases (without a final period) of the scene's changes of `kinds` that are on."""
    return [c["phrase"].rstrip(".") for c in scene["changes"] if c["kind"] in kinds and c["strength"] > 0]


def _scene_objects(scene: dict) -> list[str]:
    """The phrases of the scene's extra changes that are on (strength above 0), in the order they were added."""
    return [c["phrase"] for c in scene["changes"] if c["kind"] == "extra" and c["strength"] > 0]


def _standin_spec_validate(scene: dict, reply: dict) -> str | None:
    from commands.composer import compose
    return standin_validate(compose(scene), reply, _scene_objects(scene))


STANDIN = PromptSpec("engine stand-in", STANDIN_DETAILS, _standin_spec_validate)


def rewriter_validate(validate_rewrite: Callable[[str, dict], str | None]) -> Callable[[dict, dict], str | None]:
    """commands.rewriter.validate_rewrite on scene_prefix(scene): the composed prompt of the scene with its objects
    right after the subject and FORWARD_MOTION in a vehicle subject."""
    def validate(scene: dict, reply: dict) -> str | None:
        return validate_rewrite(scene_prefix(scene), reply)
    return validate


def prompt_spec(import_rules: Callable[[], object] = lambda: importlib.import_module("commands.rewriter")) -> PromptSpec:
    """commands.rewriter (SYSTEM_PROMPT, validate_rewrite) when it imports, else the engine stand-in.

    commands.rewriter.build_messages is not used. It is made for a separate call that only writes details:
    its user message is the finished scene text, where "the command is context only and never permission to
    add, remove or change anything", and its few-shot replies are {"details"} alone. The engine's one call has
    to apply the command (the operations) and describe the scene after it, in about 1.8 s; with
    build_messages that would be two calls in a row, past the 3 s wait. Its SYSTEM_PROMPT is used as the rules
    for the details field, and validate_rewrite checks them against the composed prompt of the new scene.
    """
    try:
        rules = import_rules()
        return PromptSpec("commands.rewriter", rules.SYSTEM_PROMPT, rewriter_validate(rules.validate_rewrite),
                          scene_prefix, rules.validate_rewrite)
    except Exception as exc:  # not there yet, or not the agreed shape
        log.info("final reader: commands.rewriter not usable (%r); using the engine's stand-in prompt", exc)
        return STANDIN


# --- why details are rejected, and the repair -----------------------------------------------------------

_CLAUSE = re.compile(r"(?<=[,;.])\s+")
# Words a model tends to use in details. Only those commands.rewriter.validate_rewrite really rejects are listed
# in the system prompt (probed at start-up), so the list follows the validator if its rules change.
_CANDIDATE_WORDS = """camera cameras pan pans panning zoom zooming dolly tracking aerial drone tilt crane handheld orbit
    cut cuts frame frames framing drive drives driving move moves moving run runs running walk walks walking fly flies
    flying reverse reversing backward backwards toward towards away approach approaching enter entering leave leaves
    leaving arrive arriving appear appears emerge emerging speeding racing explode explosion crash suddenly transform
    becomes previous previously earlier again still now before after anymore remain remains continue continues no not
    never without none nothing""".split()


def _validator_words(check: Callable[[str, dict], str | None]) -> list[str]:
    return [w for w in _CANDIDATE_WORDS if check("a scene", {"details": w}) is None]


def engine_banned(scene: dict) -> dict[str, str]:
    """Words the engine keeps out of details for `scene`, each with its reason."""
    from commands.composer import compose
    banned = {w: "a position or framing word (positions stay in the subject and object phrases)" for w in POSITION_WORDS}
    if not has_road(compose(scene)):
        banned.update({w: "a road word in a scene with no road" for w in ROAD_DETAIL_WORDS})
    for c in scene["changes"]:
        if c["kind"] == "extra" and c["strength"] > 0:
            for w in re.findall(r"[a-z]+", c["label"].lower())[-1:]:  # the noun: "red car" -> car
                banned[w] = f"names the scene's object {c['label']!r} again"
    return banned


@dataclasses.dataclass(frozen=True)
class Repair:
    prompt: str | None                 # the prompt with the kept details, or None: nothing usable
    dropped: tuple[tuple[str, str], ...] = ()  # (clause, why) for each clause cut


def why_rejected(check: Callable[[str, dict], str | None], prefix: str, details: str, banned: dict[str, str]) -> str:
    """The reason validate_rewrite (`check`) rejects `details` after `prefix`, found through its public behaviour:
    the word limits, then the single words it refuses. validate_rewrite itself returns None with no reason."""
    from commands import rewriter
    from commands.dictionary import normalize
    if not isinstance(details, str):
        return "details is not text"
    if "\n" in details or "\r" in details:
        return "a line break in details"
    max_detail = getattr(rewriter, "MAX_DETAIL_WORDS", 60)
    max_total = getattr(rewriter, "MAX_TOTAL_WORDS", 100)
    n, p = len(details.split()), len(prefix.split())
    if n > max_detail:
        return f"details has {n} words, the limit is {max_detail}"
    if p + n > max_total:
        return f"prompt {p} + details {n} words is over the limit of {max_total}"
    words = sorted({w for w in normalize(details).split() if check(prefix, {"details": w}) is None})
    engine = sorted({w for w in normalize(details).split() if w in banned})
    parts = []
    if words:
        parts.append(f"validate_rewrite refuses the words {', '.join(map(repr, words))}"
                     " (motion, camera, time order, negation, an object, or a season, weather or time the scene does not have)")
    if engine:
        parts.append("; ".join(f"{w!r} is {banned[w]}" for w in engine))
    return "; ".join(parts) or "validate_rewrite refuses it (no single word or limit explains it)"


def repair_details(spec: PromptSpec, scene: dict, details: str) -> Repair:
    """The prompt for `details`, cutting the clauses that break a rule: first the clauses with a word the
    validator or the engine refuses, then clauses from the end until it fits the word limits. Each cut is kept
    with its reason. Without spec.check (the stand-in) the details are taken or rejected whole."""
    if spec.check is None or spec.prefix is None:
        prompt = spec.validate(scene, {"details": details})
        return Repair(prompt, () if prompt is not None else ((details, "the engine stand-in refuses it"),))
    from commands.dictionary import normalize
    prefix = spec.prefix(scene)
    banned = engine_banned(scene)
    clauses = [c for c in _CLAUSE.split(details.strip()) if c.strip()]
    if not clauses:
        return Repair(spec.validate(scene, {"details": ""}))
    if spec.check(prefix, {"details": "soft light"}) is None:
        return Repair(None, ((details, f"the prompt before the details already has {len(prefix.split())} words"),))
    refused = {w for w in normalize(details).split() if w in banned or spec.check(prefix, {"details": w}) is None}
    kept, dropped = [], []
    for c in clauses:
        bad = sorted(set(normalize(c).split()) & refused)
        if bad:
            dropped.append((c, why_rejected(spec.check, prefix, c, banned)))
        else:
            kept.append(c)
    while kept:
        text = " ".join(kept).rstrip(",;. ")
        prompt = spec.validate(scene, {"details": text})
        if prompt is not None:
            return Repair(prompt, tuple(dropped))
        dropped.append((kept[-1], why_rejected(spec.check, prefix, text, banned)))
        kept.pop()
    return Repair(None, tuple(dropped))


MAIN_OBJECT_SMALL_WORDS = frozenset({"small", "tiny", "little", "distant"})


def phrase_problems(op: dict, main: bool = False) -> list[str]:
    """Rule breaks in the phrase of a new object (an add of kind extra): world words (a place, season, weather or
    time: they belong to the world, and "parked on the snowy ground" stays snowy in a desert), or no size or
    position words (the video model then draws it small or leaves it out). `main`: the first object of the
    scene, which must not be small (measured on GPU: "a small red car" was small and far right in every frame)."""
    from commands.dictionary import normalize
    phrase = normalize(op["phrase"])
    words = set(phrase.split())
    small = sorted(words & MAIN_OBJECT_SMALL_WORDS) if main else []
    found = sorted({a for a in [*_vocabulary(), *_known_worlds()] if a not in ("bright", "dark")
                    and re.search(rf"\b{re.escape(a)}\b", phrase)})
    out = []
    if found:
        out.append(f"the phrase of {op['label']!r} has world words {', '.join(map(repr, found))}: describe only the object, its size and its position")
    if small:
        out.append(f"the phrase of {op['label']!r}, the main object, says {', '.join(map(repr, small))}: make it large "
                   "and clearly visible, in the centre of the frame")
    if not words & SIZE_WORDS and not ({"clearly", "visible"} <= words):
        out.append(f"the phrase of {op['label']!r} has no size word (large, small, ...)")
    if not words & PHRASE_POSITION_WORDS:
        out.append(f"the phrase of {op['label']!r} has no position words (in the foreground at the left of the frame, beside the oak tree, ...)")
    return out


_PLACE_RULE = """- A PLACE or setting ("take me to the desert", "make it a desert", "go to Mars", "switch to a
  neon city") is ONE add of kind world: label = the place in 1 to 3 words, phrase = a short description of that
  place. It replaces the old world and KEEPS every object: a red car added before stays in the desert.
  A new place brings its own season and weather: the earlier season and weather changes are cleared, so a
  desert after "make it winter" has no snow. Add a season or weather again only if the user says it in the same
  sentence ("a snowy desert"). The time of day stays.
  Known places, use exactly these labels when one fits: {places}."""

_OPERATIONS = """You read one spoken sentence that edits a video scene, and reply with one JSON object only:
{{"blocked": <true or false>, "operations": [<operation>, ...], "details": "<text>"}}

The scene is a base sentence plus a list of changes. Each change has an id, a label (a short name shown on a
chip), a kind, a phrase (the words added to the video prompt) and a strength (0 to 1.3, normal 1.0).
Kinds: {kinds_help}. Adding a {singletons} replaces the old one of that kind.

Operations, applied in order (a sentence with several edits gives several operations, at most 6):
{{"op": "add", "label": "<1 or 2 words, lowercase>", "kind": "<{kinds}>", "phrase": "<short visual description>", "strength": 1.0}}
{{"op": "remove", "id": "<id of an existing change>"}}
{{"op": "set_strength", "id": "<id of an existing change>", "strength": <0 to 1.3>}}
{{"op": "edit_base", "base": "<the base sentence rewritten, changing only what the user asked>"}}
{{"op": "new_scene", "base": "<one sentence describing the whole new scene>"}}

Rules for operations:
- Write phrases as concrete visual descriptions. A season, weather or time phrase is under 12 words.
- A new object, animal or person (kind extra) gets a phrase of at most 25 words (under 200 characters) with
  explicit size and position words that a video model obeys: where it is in the frame and, when it helps, next
  to what. The first object added to a scene is the main one: never "small", it takes the centre and is large,
  for example "a red car parked beside the oak tree in the centre of the frame, facing the camera, large and
  clearly visible". A later object takes a free place, for example "a large male lion standing in the
  foreground at the left of the frame, facing the camera" when the centre is taken. The engine keeps
  that phrase word for word, right after the subject, in every later prompt, also after a change of place,
  season or time, so it must not name a place, ground, season, weather or time of day ("in the snow", "on the
  sand", "at night" are wrong): only the object, its size, its position and what it does.
- If an add names a kind the scene already has, its phrase describes a second one with a clear position. A
  colour, type or size the user names ALWAYS wins: "add a red car racing on a road" when the scene has a red car
  gives "a second small red car racing in the next lane at the right of the frame", differing only by position.
  A second one always gets its own label ("second red car"): an add with the label of an existing change
  replaces that change.
  Only when the user names no attribute ("add a car"), give the second one another colour or type.
- A vehicle on a road ("add a red car racing on a road") gets this cue at the end of its phrase: "{lane_cue}".
{place_rule}
- Seasons, weather and times of day use the keyword table's labels and kinds, so a sentence gives the same
  change however it is read: label (kind): words that mean it.
{vocabulary}
  "sunny" is the time change day: it replaces night. Use one of these labels whenever the words fit.
- A BASE EDIT turns something the base sentence names into something else and KEEPS every change ("turn the
  tree into a lighthouse"): use edit_base. The engine keeps the changes; never copy them yourself.
- A NEW SCENE only when the user asks for one in so many words ("new scene of a beach", "start over with a
  beach"): use new_scene, which clears every change. Its base is the user's words for the scene, as said.
- edit_base and new_scene come before any other operation of the sentence.
- To remove or weaken something, use the id of the matching change. "a little" means strength 0.3, "a lot" or
  "very" means 1.3.
- If the sentence is vague, not about the picture, or asks to undo or reset, reply with an empty operations list.

Set "blocked" to true, with no operations, if the sentence asks for sexual, violent (gore, killing, injury) or
hateful content. Otherwise "blocked" is false.

The user message is JSON: {{"scene": <the scene before the sentence>, "prompt": <its current video prompt>,
"sentence": <what the user said>, "details_max_words": <the most words "details" may have>}}. If it also has
"rejected", your earlier reply to the same sentence was not usable for the reasons given: reply again, fixed.

For the "details" field, follow the rules below. Where they speak of the scene or the description, that is the
NEW scene, after your operations. Any reply format they name is only the shape of that one field: you always
reply with the one JSON object above, {{"blocked", "operations", "details"}}.
"""

_DETAILS_LIMITS = """
Limits the engine checks on "details" (a clause that breaks one is cut; a reply with nothing left is rejected):
- Write 3 to 5 short clauses separated by commas, the most important first, at most "details_max_words" words
  in all. Only the ground, surfaces, textures, light, air and colours of the place.
- Never name or describe an object, animal, person or vehicle of the scene again, and never say where anything
  is: no position or framing words ({position}).
- Never these words, they are objects: {extras}.
- A season, weather or time word only for a label the NEW scene has after your operations. The words of each
  label are in the keyword table above; note that haze, mist, wind, breeze, frost, icy, moon, moonlight, stars,
  sunlight, sunshine, morning, evening and dusk are on it, so "heat haze" needs fog and "moonlight" needs night.
- Never these words (motion, camera, time order, negation): {validator}.
- Road words ({road}) only when the scene has a road.
"""


_KIND_HELP = {"world": "world (the place: desert, neon city, Mars, ...)", "season": "season (winter, autumn, ...)",
              "weather": "weather (rain, fog, sunny, ...)", "time": "time (night, day, sunset, ...)",
              "extra": "extra (any object, animal, person or effect)"}


def _known_worlds() -> dict:
    """commands.world.WORLDS: normalized alias -> {"label", "kind", "phrase"}."""
    from commands.world import WORLDS
    return WORLDS


def _vocabulary() -> dict[str, dict]:
    """commands.dictionary.KEYWORDS restricted to season, weather and time: normalized alias -> entry."""
    from commands.dictionary import KEYWORDS
    return {alias: e for alias, e in KEYWORDS.items() if e["kind"] in ("season", "weather", "time")}


def _vocabulary_lines() -> str:
    words: dict[tuple[str, str], list[str]] = {}
    for alias, e in _vocabulary().items():
        words.setdefault((e["label"], e["kind"]), [])
        if alias != e["label"]:
            words[(e["label"], e["kind"])].append(alias)
    return "\n".join(f"  {label} ({kind}): {', '.join(aliases) or label}" for (label, kind), aliases in words.items())


def operations_instructions() -> str:
    """The operations part of the system prompt, for the kinds this commands/ has."""
    from commands.state import KINDS
    places = ", ".join(sorted({w["label"] for w in _known_worlds().values()})) or "none"
    singletons = [k for k in KINDS if k != "extra"]
    return _OPERATIONS.format(kinds_help=", ".join(_KIND_HELP.get(k, k) for k in KINDS), kinds="|".join(KINDS),
                              singletons=", ".join(singletons[:-1]) + " or " + singletons[-1], lane_cue=LANE_CUE,
                              place_rule=_PLACE_RULE.format(places=places), vocabulary=_vocabulary_lines())


def details_limits(spec: PromptSpec) -> str:
    """The engine's limits on details, with the words commands.rewriter refuses; empty for the stand-in."""
    if spec.check is None:
        return ""
    from commands.dictionary import KEYWORDS
    extras = sorted({a for a, e in KEYWORDS.items() if e["kind"] == "extra"})
    return _DETAILS_LIMITS.format(position=", ".join(sorted(POSITION_WORDS)), extras=", ".join(extras),
                                  validator=", ".join(_validator_words(spec.check)), road=", ".join(sorted(ROAD_DETAIL_WORDS)))


def details_budget(spec: PromptSpec, snapshot: dict) -> int:
    """The words details may have: what the validator's total limit leaves after the prompt of the scene before,
    less room for a new phrase, between 12 and 45."""
    if spec.prefix is None:
        return 50
    from commands import rewriter
    left = getattr(rewriter, "MAX_TOTAL_WORDS", 100) - len(spec.prefix(snapshot).split()) - 15
    return max(12, min(45, left))


# --- the cache file -------------------------------------------------------------------------------------

class CacheFile:
    """{"fingerprint": str, "entries": [{"prompt_before", "sentence", "outcome", "operations", "prompt"}]},
    replaced atomically on save. A file written for another system prompt or model is ignored."""

    def __init__(self, path: str | Path, fingerprint: str) -> None:
        self.path = Path(path).expanduser()
        self.fingerprint = fingerprint
        self._lock = threading.Lock()

    def load(self) -> dict[tuple[str, str], Entry]:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            log.warning("final reader: cannot read %s (%r); starting with an empty cache", self.path, exc)
            return {}
        if not isinstance(data, dict) or data.get("fingerprint") != self.fingerprint:
            log.warning("final reader: %s was written for another prompt or model; starting empty", self.path)
            return {}
        out = {}
        for e in data.get("entries") or []:
            try:
                outcome = Outcome(e["outcome"])
                if outcome not in (Outcome.SONNET, Outcome.BLOCKED):
                    continue
                out[(e["prompt_before"], e["sentence"])] = Entry(
                    outcome, tuple(e.get("operations") or ()), e.get("prompt"))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def save(self, cache: dict[tuple[str, str], Entry]) -> None:
        entries = [{"prompt_before": k[0], "sentence": k[1], "outcome": v.outcome.value,
                    "operations": list(v.operations), "prompt": v.prompt}
                   for k, v in cache.items() if v.outcome in (Outcome.SONNET, Outcome.BLOCKED)]
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".final-reader-", suffix=".tmp")
                with os.fdopen(fd, "w") as f:
                    json.dump({"fingerprint": self.fingerprint, "entries": entries}, f, indent=1)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)  # atomic: a reader sees the old file or the new one, never half
            except OSError:
                log.exception("final reader: cannot save %s; the cache stays in memory", self.path)


# --- the pinned demo prompts ----------------------------------------------------------------------------

def load_pinned(path: str | Path | None) -> dict[tuple[str, str], Entry]:
    """The demo prompts file as PINNED entries keyed by (prompt_before, sentence). Never raises.

    File: [entry, ...] or {"base": str, "entries": [entry, ...]}, where an entry is
    {"sentence", "prompt_before", "operations": [op, ...], "prompt"} (other keys, such as "canonical", "model" or
    "fingerprint", are notes and ignored). The entries are replayed in order on a commands.state.SceneState that
    starts at `base` (commands.state.DEFAULT_BASE for the list form). An entry whose
    prompt_before is the composed prompt of the base starts again from the base (a reset came before it).
    An entry is skipped, with a log line, when its prompt_before is not the composed prompt of the replayed
    scene, its operations do not apply (expand_operations, then one batch), or its prompt is not text.
    """
    if path is None:
        return {}
    path = Path(path).expanduser()
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        log.info("final reader: no demo prompts file at %s", path)
        return {}
    except (OSError, ValueError) as exc:
        log.warning("final reader: cannot read the demo prompts %s (%r); none pinned", path, exc)
        return {}
    try:
        from commands.composer import compose
        from commands.state import DEFAULT_BASE, SceneState
        if isinstance(data, list):
            data = {"base": DEFAULT_BASE, "entries": data}
        base = data.get("base", DEFAULT_BASE)
        start = compose(SceneState(base).snapshot())
        entries = data["entries"]
        if not isinstance(entries, list):
            raise TypeError("entries is not a list")
    except Exception as exc:  # noqa: BLE001 - a bad file must never stop the engine
        log.warning("final reader: the demo prompts %s are not usable (%r); none pinned", path, exc)
        return {}
    pinned: dict[tuple[str, str], Entry] = {}
    scene = SceneState(base)
    for n, e in enumerate(entries):
        try:
            sentence, before, prompt = e["sentence"], e["prompt_before"], e["prompt"]
            if not all(isinstance(v, str) and v.strip() for v in (sentence, before, prompt)):
                raise ValueError("sentence, prompt_before and prompt must be nonempty text")
            if before == start:
                scene = SceneState(base)
            if compose(scene.snapshot()) != before:
                raise ValueError("prompt_before is not the scene the entries before it lead to")
            ops = expand_operations(e["operations"], scene.snapshot()) if isinstance(e["operations"], list) else None
            if not ops:
                raise ValueError("the operations are not valid for that scene")
            trial = copy.deepcopy(scene)
            trial.apply(ops[0] if len(ops) == 1 else {"op": "batch", "operations": ops})
            compose(trial.snapshot())
        except Exception as exc:  # noqa: BLE001 - skip the entry, keep the others
            log.warning("final reader: demo prompt %d (%r) skipped: %r", n, _short(e.get("sentence") if isinstance(e, dict) else e), exc)
            continue
        scene = trial
        pinned[(before, sentence)] = Entry(Outcome.PINNED, tuple(ops), prompt)
    log.info("final reader: %d of %d demo prompts pinned from %s", len(pinned), len(entries), path)
    return pinned


# --- the reader -----------------------------------------------------------------------------------------

class FinalReader:
    """Callable `reader(text, snapshot)`; see the module docstring. Never raises."""

    def __init__(self, state: FinalReaderState, reason: str = "", chat=None, spec: PromptSpec = STANDIN,
                 timeout_s: float = DEFAULT_TIMEOUT_S, cache_file: CacheFile | None = None,
                 pinned: dict[tuple[str, str], Entry] | None = None) -> None:
        if state is FinalReaderState.ON and chat is None:
            raise ValueError("an ON final reader needs a chat client")
        self.state = state
        self.reason = reason
        self.spec = spec
        self.timeout_s = timeout_s
        self.system_prompt = (f"{operations_instructions()}\n{spec.details_instructions}\n{details_limits(spec)}\n"
                              f"{car_template.instructions()}")
        self._chat = chat
        self._file = cache_file
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], Entry] = cache_file.load() if cache_file is not None else {}
        self._pinned: dict[tuple[str, str], Entry] = dict(pinned or {})  # checked before the cache
        self.pinned_hits = 0
        self.template_hits = 0  # measured car worlds composed by the engine, no call
        # composed prompt of a vehicle scene -> the template prompt last given for it, so the next model call on
        # that scene is told its current road, scene and light (after "make it foggy", the fog stays)
        self._given_vehicle: dict[str, str] = {}
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="final-reader")
        self.calls = 0
        self.cache_hits = 0
        self.blocked = 0
        self.timeouts = 0
        self.late_dropped = 0
        self.rejects = 0
        self.details_rejected = 0
        self.details_repaired = 0
        self.retries = 0
        self.errors = 0
        self.last_ms: float | None = None
        self._recent_ms: deque[float] = deque(maxlen=50)
        if self._cache:
            log.info("final reader: %d cached results loaded from %s", len(self._cache), cache_file.path)

    @classmethod
    def off(cls) -> "FinalReader":
        return cls(FinalReaderState.OFF)

    @classmethod
    def unavailable(cls, reason: str) -> "FinalReader":
        return cls(FinalReaderState.UNAVAILABLE, reason)

    def mark_unavailable(self, reason: str) -> None:
        """ON -> UNAVAILABLE, when the reader it plugs into cannot use it. Logged once."""
        if self.state is FinalReaderState.ON:
            log.warning("final reader unavailable, running without it: %s", reason)
            self.state, self.reason = FinalReaderState.UNAVAILABLE, reason

    def stop(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __call__(self, text: str, snapshot: dict) -> dict | None:
        if self.state is not FinalReaderState.ON:
            return None
        word = blocked_word(text)
        if word is not None:
            with self._lock:
                self.blocked += 1
            log.info("final reader: blocked %r (word list: %r), no call", _short(text), word)
            return {"blocked": True}
        from commands.composer import compose  # the commands/ owner's composer; imported only when ON
        key = (compose(snapshot), text)
        with self._lock:
            entry = self._pinned.get(key)
            if entry is not None:
                self.pinned_hits += 1
            else:
                entry = self._cache.get(key)
                if entry is not None:
                    self.cache_hits += 1
        if entry is not None:
            result = self._from_cache(key, entry, snapshot)
            if result is not None:
                self._given(entry, snapshot)
            return result
        measured = self._measured(text, snapshot)
        if measured is not None:
            with self._lock:
                self.template_hits += 1
            return self._given(measured, snapshot)
        with self._lock:
            self.calls += 1
        fut = self._pool.submit(self._call, text, snapshot)
        try:
            entry = fut.result(timeout=self.timeout_s)
        except concurrent.futures.TimeoutError:
            entry = None
        late = entry is None
        with self._lock:
            if late:
                self.timeouts += 1
                entry = Entry(Outcome.FALLBACK)
            self._cache[key] = entry
            if entry.outcome is Outcome.BLOCKED:
                self.blocked += 1
            snapshot_of_cache = dict(self._cache) if entry.outcome is not Outcome.FALLBACK else None
        if late:  # outside the lock: the callback takes it, and runs at once if the call has just finished
            fut.add_done_callback(lambda _f: self._late())
            log.info("final reader: no answer for %r within %.1f s; the dictionary result is final for it",
                     _short(text), self.timeout_s)
        if snapshot_of_cache is not None and self._file is not None:
            self._file.save(snapshot_of_cache)
        return self._given(entry, snapshot)

    def stats(self) -> dict:
        with self._lock:
            return {"state": self.state.value, "reason": self.reason or None, "prompt_spec": self.spec.name,
                    "calls": self.calls, "cache_hits": self.cache_hits,
                    "pinned": len(self._pinned), "pinned_hits": self.pinned_hits,
                    "template_hits": self.template_hits, "blocked": self.blocked,
                    "timeouts": self.timeouts, "late_dropped": self.late_dropped, "rejects": self.rejects,
                    "details_rejected": self.details_rejected, "details_repaired": self.details_repaired,
                    "retries": self.retries, "errors": self.errors,
                    "p50_ms": round(statistics.median(self._recent_ms), 1) if self._recent_ms else None,
                    "last_ms": None if self.last_ms is None else round(self.last_ms, 1),
                    "cache_size": len(self._cache), "timeout_s": self.timeout_s,
                    "cache_file": None if self._file is None else str(self._file.path)}

    # --- internals -------------------------------------------------------------------------

    def _late(self) -> None:
        with self._lock:
            self.late_dropped += 1
        log.info("final reader: a late answer was dropped")

    def _given(self, entry: Entry, snapshot: dict) -> dict | None:
        """entry.result(), remembering the prompt of a vehicle scene for the scene it leads to."""
        if entry.prompt is not None and entry.outcome in (Outcome.SONNET, Outcome.PINNED, Outcome.TEMPLATE):
            from commands.composer import compose
            try:
                after = scene_after(snapshot, list(entry.operations))
                if car_template.vehicle_subject(after["base"]) is not None:
                    with self._lock:
                        self._given_vehicle[compose(after)] = entry.prompt
            except (ValueError, TypeError, KeyError):
                pass  # only a hint for the next call
        return entry.result()

    def _current_vehicle_world(self, snapshot: dict, subject: str) -> "car_template.World | None":
        """The road, scene and light of the prompt last given for this vehicle scene, else its measured world."""
        from commands.composer import compose
        with self._lock:
            given = self._given_vehicle.get(compose(snapshot))
        return (car_template.parse_world(given, subject, car_template.scene_objects(snapshot))
                or car_template.current_world(snapshot))

    def _measured(self, text: str, snapshot: dict) -> Entry | None:
        """A TEMPLATE entry when `text` is a measured car world for `snapshot` (car_template.Route.MEASURED), else
        None. Deterministic: the same scene and sentence always give the same operations and words."""
        plan = car_template.plan(text, snapshot)
        if plan.route is not car_template.Route.MEASURED:
            return None
        from commands.composer import compose
        try:
            ops = expand_operations(car_template.measured_operations(plan, snapshot), snapshot)
            if ops == []:
                return None  # already that world: nothing to change, the model reads it
            if ops is None:
                raise ValueError("the operations are not valid for this scene")
            compose(scene_after(snapshot, ops))
        except (ValueError, TypeError, KeyError) as exc:
            log.warning("final reader: the measured car world for %r does not apply (%r); asking the model",
                        _short(text), exc)
            return None
        prompt = car_template.world_prompt(plan.subject, plan.world)
        log.info("final reader: %r -> measured car world %r, no call", _short(text), plan.world.name)
        return Entry(Outcome.TEMPLATE, tuple(ops), prompt)

    def _from_cache(self, key, entry: Entry, snapshot: dict) -> dict | None:
        """A cached SONNET or PINNED result's operations name change ids; check they still fit this snapshot."""
        if entry.outcome not in (Outcome.SONNET, Outcome.PINNED):
            return entry.result()
        if all(validate_operation(dict(op), snapshot, CACHED_MAX_PHRASE) == dict(op) for op in entry.operations):
            return entry.result()
        log.info("final reader: the cached operations for %r no longer fit the scene; dictionary path", _short(key[1]))
        return None

    def _call(self, text: str, snapshot: dict) -> Entry | None:
        """Worker thread: one Sonnet call, checked, and one retry with the reasons when the first reply is not
        fully usable and the first call took less than half the wait (so on the live 3 s wait at 1.8 s a call,
        the retry happens only in offline runs such as the demo prompts regeneration). Never raises."""
        t0 = time.perf_counter()
        first = self._attempt(text, snapshot, None)
        best = first
        if first.verdict in RETRY_VERDICTS:
            elapsed = time.perf_counter() - t0
            if elapsed < self.timeout_s / 2:
                with self._lock:
                    self.retries += 1
                log.warning("final reader: asking again for %r: %s", _short(text), "; ".join(first.reasons)[:400])
                second = self._attempt(text, snapshot, first)
                if VERDICT_RANK[second.verdict] >= VERDICT_RANK[first.verdict]:
                    best = second
            else:
                log.warning("final reader: no time to ask again for %r (%.0f ms of %.1f s used)", _short(text),
                            elapsed * 1e3, self.timeout_s)
        with self._lock:
            if best.verdict is Verdict.FAILED:
                self.errors += 1
            elif best.verdict in (Verdict.INVALID, Verdict.EMPTY):
                self.rejects += 1
            elif best.verdict is Verdict.PLAIN:
                self.details_rejected += 1
            elif best.verdict is Verdict.REPAIRED:
                self.details_repaired += 1
        entry = best.entry
        if best.verdict in (Verdict.FAILED, Verdict.INVALID, Verdict.EMPTY):
            log.warning("final reader: no usable reply for %r: %s", _short(text), "; ".join(best.reasons)[:400])
        elif entry.outcome is Outcome.BLOCKED:
            log.info("final reader: the model blocked %r", _short(text))
        else:
            if best.verdict is Verdict.PLAIN:
                log.warning("final reader: %r keeps its operations with the composed prompt (no usable details)",
                            _short(text))
            log.info("final reader: %r -> %d operations (%s), prompt %r", _short(text), len(entry.operations),
                     best.verdict.value, _short(entry.prompt))
        return entry

    def _attempt(self, text: str, snapshot: dict, previous: "Attempt | None") -> "Attempt":
        """One call. `previous` is the earlier attempt for this sentence: its reply and reasons are sent back."""
        from commands.composer import compose
        t0 = time.perf_counter()
        user = {"scene": _scene_for_model(snapshot), "prompt": compose(snapshot), "sentence": text,
                "details_max_words": details_budget(self.spec, snapshot)}
        plan = car_template.plan(text, snapshot)
        if plan.route is not car_template.Route.PLAIN:
            world = None if plan.new_scene else self._current_vehicle_world(snapshot, plan.subject)
            user["car_template"] = {"subject": plan.subject, **({"road": world.road, "scene": world.scene,
                                                                 "light": world.light} if world else {})}
        messages = [{"role": "user", "content": json.dumps(user)}]
        if previous is not None and previous.reply_text is not None:
            messages += [{"role": "assistant", "content": previous.reply_text},
                         {"role": "user", "content": json.dumps({**user, "rejected": list(previous.reasons)})}]
        reply_text = None
        try:
            reply_text = self._chat.complete(self.system_prompt, messages)
            reply = json.loads(FENCE.sub("", reply_text))
            error = None
        except Exception as exc:  # the HTTP call or the JSON
            reply, error = None, exc
        ms = (time.perf_counter() - t0) * 1e3
        with self._lock:
            self.last_ms = ms
            self._recent_ms.append(ms)
        if error is not None:
            if reply_text is None:  # the call itself failed: asking again would fail the same way
                log.warning("final reader: call failed for %r after %.0f ms: %s", _short(text), ms, repr(error)[:200])
                return Attempt(Verdict.FAILED, Entry(Outcome.FALLBACK), (f"the call failed: {error!r}"[:200],))
            log.warning("final reader: the reply for %r is not JSON (%.0f ms): %s", _short(text), ms, _short(reply_text))
            return Attempt(Verdict.INVALID, Entry(Outcome.FALLBACK), ("the reply was not one JSON object",), reply_text)
        attempt = self._check(reply, snapshot, text)
        log.info("final reader: %r reply %s in %.0f ms", _short(text), attempt.verdict.value, ms)
        return dataclasses.replace(attempt, reply_text=reply_text)

    def _plain(self, after: dict, canonical: str) -> str:
        """The spec's prompt with no details (the composed prompt, plus forward motion with commands.rewriter)."""
        try:
            prompt = self.spec.validate(after, {"details": ""})
        except Exception:  # noqa: BLE001 - the composed prompt is always usable
            prompt = None
        return prompt if isinstance(prompt, str) and prompt.strip() else canonical

    def _check(self, reply, snapshot: dict, text: str = "") -> "Attempt":
        if not isinstance(reply, dict):
            return Attempt(Verdict.INVALID, Entry(Outcome.FALLBACK), ("the reply is not a JSON object",))
        if reply.get("blocked") is True:
            return Attempt(Verdict.BLOCKED, Entry(Outcome.BLOCKED))
        ops = reply.get("operations")
        if not isinstance(ops, list) or not 1 <= len(ops) <= MAX_OPERATIONS:
            return Attempt(Verdict.EMPTY, Entry(Outcome.FALLBACK), ("no operations",))
        try:
            clean = expand_operations(ops, snapshot)
            after = scene_after(snapshot, clean) if clean is not None else None
        except (ValueError, TypeError, KeyError) as exc:
            return Attempt(Verdict.INVALID, Entry(Outcome.FALLBACK), (f"the operations do not apply: {exc!r}",))
        if clean is None:
            return Attempt(Verdict.INVALID, Entry(Outcome.FALLBACK), (invalid_operation_reason(ops, snapshot),))
        subject = car_template.vehicle_subject(after["base"])
        if subject is not None:
            return self._check_vehicle(reply, snapshot, clean, subject)
        from commands.composer import compose
        try:
            canonical = compose(after)  # with PR #88 a world the base cannot take raises ValueError
        except (ValueError, TypeError, KeyError) as exc:
            return Attempt(Verdict.INVALID, Entry(Outcome.FALLBACK), (f"the new scene does not compose: {exc!r}",))
        known = {(c["label"], c["phrase"]) for c in snapshot["changes"]}
        added = [c for c in after["changes"] if c["kind"] == "extra" and (c["label"], c["phrase"]) not in known]
        main = next((c for c in after["changes"] if c["kind"] == "extra"), None)  # the first object of the scene
        problems = [p for c in added for p in phrase_problems(c, main=c is main)]
        for p in problems:
            log.warning("final reader: %r: %s", _short(text), p)
        details = reply.get("details")
        details = details if isinstance(details, str) else ""
        try:  # only {"details"}: commands.rewriter.validate_rewrite rejects any other key
            repair = repair_details(self.spec, after, details)
        except Exception as exc:  # noqa: BLE001
            log.warning("final reader: checking the details raised %r", exc)
            repair = Repair(None, ((details, f"the check raised {exc!r}"),))
        for clause, why in repair.dropped:  # every rejection, with its reason and the words
            log.warning("final reader: details for %r rejected (%s): %s", _short(text), why, _short(clause, 80))
        if repair.prompt is None or not repair.prompt.strip():
            # The operations are still the model's reading of the sentence; only its wording is dropped.
            reasons = tuple(problems) + tuple(f"details {why}: {_short(c, 60)!r}" for c, why in repair.dropped)
            return Attempt(Verdict.PLAIN, Entry(Outcome.SONNET, tuple(clean), self._plain(after, canonical)),
                           reasons or ("no details",))
        if problems:
            return Attempt(Verdict.PHRASE, Entry(Outcome.SONNET, tuple(clean), repair.prompt), tuple(problems))
        verdict = Verdict.REPAIRED if repair.dropped else Verdict.ACCEPTED
        return Attempt(verdict, Entry(Outcome.SONNET, tuple(clean), repair.prompt),
                       tuple(f"details {why}: {_short(c, 60)!r}" for c, why in repair.dropped))


    def _check_vehicle(self, reply: dict, snapshot: dict, clean: list[dict], subject: str) -> "Attempt":
        """A reply whose new scene is a vehicle on a road: the operations with the base kept as the bare vehicle
        subject (commands.world.compose_base cannot replace a place named in the base), and the template with the
        model's road, scene and light. The subject and camera sentences are the engine's, word for word."""
        from commands.composer import compose
        try:
            ops = bare_vehicle_operations(clean, snapshot, subject)
            if ops is None:
                return Attempt(Verdict.INVALID, Entry(Outcome.FALLBACK), (
                    "the vehicle scene's base must be only the vehicle; use new_scene with the vehicle alone",))
            compose(scene_after(snapshot, ops))
        except (ValueError, TypeError, KeyError) as exc:
            return Attempt(Verdict.INVALID, Entry(Outcome.FALLBACK), (f"the new scene does not compose: {exc!r}",))
        problems = car_template.field_problems(reply)
        world = None if problems else car_template.model_world(reply)
        after = scene_after(snapshot, ops)
        objects = car_template.scene_objects(after)
        prompt = None if world is None else car_template.world_prompt(subject, world, objects)
        if prompt is None or not car_template.check_prompt(prompt, subject, objects):
            return Attempt(Verdict.INVALID, Entry(Outcome.FALLBACK),
                           tuple(problems) or ("road, scene and light do not make the template",))
        known = {(c["label"], c["phrase"]) for c in snapshot["changes"]}
        added = [c for c in after["changes"] if c["kind"] == "extra" and (c["label"], c["phrase"]) not in known]
        main = next((c for c in after["changes"] if c["kind"] == "extra"), None)
        phrase = [p for c in added for p in phrase_problems(c, main=c is main)]
        if phrase:  # usable, but a new object's phrase breaks a rule: asked again once when there is time
            return Attempt(Verdict.PHRASE, Entry(Outcome.SONNET, tuple(ops), prompt), tuple(phrase))
        return Attempt(Verdict.ACCEPTED, Entry(Outcome.SONNET, tuple(ops), prompt))


class Verdict(enum.Enum):
    """What one reply gave. RETRY_VERDICTS are asked again once, with the reasons, when there is time."""
    ACCEPTED = "accepted"    # operations and all of the details
    REPAIRED = "repaired"    # operations and the details minus the clauses that broke a rule (logged)
    PHRASE = "phrase"        # usable, but a new object's phrase breaks a rule (world words, no size or position)
    PLAIN = "plain"          # operations, but no usable details: the composed prompt
    EMPTY = "empty"          # no operations: the dictionary path (a vague sentence; not asked again)
    BLOCKED = "blocked"      # the model refused the sentence
    INVALID = "invalid"      # not usable: the dictionary path
    FAILED = "failed"        # the call failed: the dictionary path


RETRY_VERDICTS = frozenset({Verdict.PHRASE, Verdict.PLAIN, Verdict.INVALID})
VERDICT_RANK = {Verdict.FAILED: 0, Verdict.INVALID: 1, Verdict.EMPTY: 1, Verdict.PLAIN: 2, Verdict.PHRASE: 3,
                Verdict.REPAIRED: 4, Verdict.ACCEPTED: 5, Verdict.BLOCKED: 5}


@dataclasses.dataclass(frozen=True)
class Attempt:
    verdict: Verdict
    entry: Entry
    reasons: tuple[str, ...] = ()
    reply_text: str | None = None


def invalid_operation_reason(ops: list, snapshot: dict) -> str:
    """Which operation is invalid, and why, for the log and the retry."""
    for n, op in enumerate(ops):
        if not isinstance(op, dict):
            return f"operation {n + 1} is not an object"
        kind = op.get("op")
        if kind in ("undo", "reset"):
            return f"operation {n + 1} is {kind}: undo and reset stay with the dictionary"
        if kind == "add":
            phrase = op.get("phrase")
            limit = OBJECT_MAX_PHRASE if op.get("kind") == "extra" else FINAL_MAX_PHRASE
            if isinstance(phrase, str) and len(" ".join(phrase.split())) > limit:
                return (f"operation {n + 1} ({op.get('label')!r}): its phrase has {len(phrase)} characters, "
                        f"the limit is {limit}")
        if validate_operation(op if kind != "edit_base" else {"op": "new_scene", "base": op.get("base")}, snapshot,
                              FINAL_MAX_PHRASE) is None:
            return f"operation {n + 1} is not valid: {_short(json.dumps(op), 160)}"
    return "an operation names a change id after a base edit or a new scene, or there are too many"


BATCH_LIMIT = 16  # commands.state: a batch holds 1 to 16 operations


def expand_operations(ops: list, snapshot: dict) -> list[dict] | None:
    """The model's operations as operations of commands/state.py, or None if one is invalid.

    {"op": "edit_base", "base": B} becomes new_scene B followed by one add per change the scene has at that
    point (label, kind, phrase and strength copied), so a base edit keeps the changes and, applied as one
    batch, is undone in one step. commands/state.py has no operation that rewrites only the base (a world
    change replaces the place, not an object of the base such as the tree), so this expansion stays.
    An operation naming a change id after a base edit or a new scene is invalid (those ids are gone), and undo
    and reset stay with the dictionary.
    A world add clears the earlier season and weather in commands.state (#115); world_first keeps a season said
    in the same sentence.
    A vehicle added on a road gets LANE_CUE.
    """
    out: list[dict] = []
    rebuilt = False  # an edit_base or new_scene came first: the old ids no longer exist
    for op in ops:
        if isinstance(op, dict) and op.get("op") == "edit_base":
            clean = validate_operation({"op": "new_scene", "base": op.get("base")}, snapshot, FINAL_MAX_PHRASE)
            if clean is None:
                return None
            current = scene_after(snapshot, out)["changes"]
            out.append(clean)
            out += [{"op": "add", "label": c["label"], "kind": c["kind"], "phrase": c["phrase"],
                     "strength": c["strength"]} for c in current]
            rebuilt = True
            continue
        clean = validate_operation(op, snapshot, FINAL_MAX_PHRASE)
        if clean is None or clean["op"] in ("undo", "reset") or (rebuilt and "id" in clean):
            return None
        if clean["op"] == "add" and clean["kind"] == "extra" and len(clean["phrase"]) > OBJECT_MAX_PHRASE:
            return None
        rebuilt = rebuilt or clean["op"] == "new_scene"
        out.append(with_lane_cue(distinct_label(canonical_change(canonical_world(clean)), snapshot, out)))
    return world_first(out) if len(out) <= BATCH_LIMIT else None


def bare_vehicle_operations(ops: list[dict], snapshot: dict, subject: str) -> list[dict] | None:
    """`ops` (already expanded) with the scene's base kept as the bare vehicle `subject`: a new_scene of
    "a red sports car driving along a country road" becomes a new_scene of "a red sports car"; a scene whose base
    still names a place gets a base edit to the subject first. None when that edit cannot keep the operations
    (they name change ids)."""
    ops = [{**op, "base": car_template.vehicle_subject(op["base"]) or op["base"]} if op["op"] == "new_scene" else op
           for op in ops]
    if car_template.is_bare(scene_after(snapshot, ops)):
        return ops
    return expand_operations([{"op": "edit_base", "base": subject}, *ops], snapshot)


def world_first(ops: list[dict]) -> list[dict]:
    """The world adds of a sentence moved before its season and weather adds (each group keeps its order).
    commands.state clears season and weather on a world add (#115), so "take me to a snowy desert" read as
    winter then desert would lose the winter it asked for. A new_scene or edit_base stays first."""
    worlds = [o for o in ops if o["op"] == "add" and o["kind"] == "world"]
    if not worlds:
        return ops
    first = next((i for i, o in enumerate(ops) if o["op"] == "add" and o["kind"] in ("season", "weather")), None)
    if first is None or first > ops.index(worlds[0]):
        return ops
    rest = [o for o in ops if o not in worlds]
    return rest[:first] + worlds + rest[first:]


_ORDINALS = ("second", "third", "fourth", "fifth", "sixth")


def distinct_label(op: dict, snapshot: dict, earlier: list[dict]) -> dict:
    """An add of a second object (kind extra, its phrase says "second", "another", ...) with the label of an existing
    change gets its own label ("second red car"): commands.state replaces a change of the same label, so the
    first red car would be lost. Any other operation is returned as it is."""
    if op.get("op") != "add" or op.get("kind") != "extra":
        return op
    words = _words(op["phrase"])
    if not words & {"another", "other", *_ORDINALS}:
        return op
    taken = {c["label"] for c in snapshot["changes"]} | {o.get("label") for o in earlier}
    if op["label"] not in taken:
        return op
    noun = op["label"].split()[-1]
    for ordinal in _ORDINALS:
        for label in (f"{ordinal} {op['label']}", f"{ordinal} {noun}"):
            if label not in taken and len(label) <= 24:  # slow_reader.MAX_LABEL
                return {**op, "label": label}
    return op


def with_lane_cue(op: dict) -> dict:
    """An add of a vehicle on a road (kind extra) with LANE_CUE at the end of its phrase, once."""
    if op.get("op") != "add" or op.get("kind") != "extra":
        return op
    phrase = op["phrase"]
    if is_vehicle(phrase) and has_road(phrase) and "lane markings" not in phrase.lower():
        return {**op, "phrase": f"{phrase.rstrip(' .,')}, {LANE_CUE}"}
    return op


def canonical_change(op: dict) -> dict:
    """An add whose label is a season, weather or time word of the keyword table ("sunny", "snow") gets that
    table's label and kind (day of kind time, winter of kind season), so it replaces the right earlier change,
    as the keyword table's edit would. The model's phrase is kept. Other operations are returned as they are."""
    if op.get("op") != "add" or op.get("kind") == "world":
        return op
    from commands.dictionary import normalize
    entry = _vocabulary().get(normalize(op["label"]))
    return {**op, "label": entry["label"], "kind": entry["kind"]} if entry else op


def canonical_world(op: dict) -> dict:
    """A world add for a place commands.world knows ("desert", "the city") gets its label and tested phrase,
    the same change the keyword table makes for that place. Any other operation is returned as it is."""
    if op.get("op") != "add" or op.get("kind") != "world":
        return op
    from commands.dictionary import normalize
    known = _known_worlds().get(normalize(op["label"]))
    return {**op, "label": known["label"], "phrase": known["phrase"]} if known else op


def scene_after(snapshot: dict, operations: list[dict]) -> dict:
    """The snapshot after `operations`, through commands.state.SceneState's public API only: the scene is
    rebuilt by adding its changes in order (new ids), the operations' ids are mapped onto them, and the result
    is given back with the old ids. Raises ValueError (and others) when an operation does not apply."""
    from commands.state import SceneState
    trial = SceneState(snapshot["base"])
    to_trial = {}
    for c in snapshot["changes"]:
        trial.apply({"op": "add", "label": c["label"], "kind": c["kind"], "phrase": c["phrase"],
                     "strength": c["strength"]})
        to_trial[c["id"]] = trial.snapshot()["changes"][-1]["id"]
    for op in operations:
        op = copy.deepcopy(op)
        if "id" in op:
            op["id"] = to_trial[op["id"]]
        trial.apply(op)
    return trial.snapshot()


def _scene_for_model(snapshot: dict) -> dict:
    return {"base": snapshot.get("base"),
            "changes": [{k: c[k] for k in ("id", "label", "kind", "phrase", "strength")}
                        for c in snapshot.get("changes", [])]}


def fingerprint(system_prompt: str, model: str) -> str:
    return hashlib.sha256(json.dumps([system_prompt, model]).encode()).hexdigest()[:16]


def load(mode: str, timeout_s: float = DEFAULT_TIMEOUT_S, cache_path: str | None = DEFAULT_CACHE_FILE,
         environ=os.environ, chat=None, spec: PromptSpec | None = None,
         demo_prompts: str | None = DEFAULT_DEMO_PROMPTS) -> FinalReader:
    """The final reader for --final-reader `mode` ("off" or "sonnet"). With "sonnet" but no FOUNDRY_* settings
    it logs once and gives an UNAVAILABLE reader. `chat` replaces the Foundry client (tests); `cache_path` None
    keeps the cache in memory only. `demo_prompts` is the pinned demo prompts file (load_pinned), None for none."""
    if mode == "off":
        return FinalReader.off()
    if mode != "sonnet":
        raise ValueError(f"final reader must be 'off' or 'sonnet', got {mode!r}")
    if chat is None:
        settings = settings_from_env(environ)
        if settings is None:
            reason = "FOUNDRY_ENDPOINT or FOUNDRY_KEY is not set"
            log.warning("final reader unavailable, running without it: %s", reason)
            return FinalReader.unavailable(reason)
        chat = FoundryMessages.from_settings(settings, HTTP_TIMEOUT_S)
    spec = spec or prompt_spec()
    model = getattr(chat, "model", "")
    reader = FinalReader(FinalReaderState.ON, chat=chat, spec=spec, timeout_s=timeout_s,
                         pinned=load_pinned(demo_prompts))
    if cache_path:
        reader._file = CacheFile(cache_path, fingerprint(reader.system_prompt, model))
        reader._cache = reader._file.load()
    log.info("final reader on: model %s, prompt %s, wait up to %.1f s, cache %s", model or "(test client)", spec.name,
             timeout_s, cache_path or "in memory")
    return reader


def _short(text, n: int = LOG_CHARS) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 3] + "..."

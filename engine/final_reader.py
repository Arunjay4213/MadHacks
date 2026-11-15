"""The final reader (--final-reader sonnet): one Claude Sonnet 5.5 call per FINAL sentence that the fast paths do not
answer, which writes the complete new video prompt and the chips to show.

    reader = load("sonnet")
    reader.lookup(text, snapshot) -> Entry | None      the content block, pinned, cached and measured answers; no call
    reader.ask(text, snapshot, current_prompt) -> Entry one Sonnet call, checked (the light check), cached
    reader.result(entry, text, snapshot) -> None                       use the dictionary path (fallback)
                                           | {"blocked": True}          refuse it: no steer, shown as unmatched
                                           | {"not_understood": True}   Sonnet did not understand it: unmatched
                                           | {"operations": [op, ...], "prompt": str}
    reader(text, snapshot) is lookup, then ask, then result (no dictionary fast path: engine/reader.py adds it).

`snapshot` is commands.state.SceneState.snapshot() of the scene before the sentence. A Sonnet answer is one
operation, {"op": "free_scene", "prompt": p, "chips": [label, ...]}, which makes the scene a free scene
(commands.state.Origin.FREE) whose prompt is p. Pinned, measured and named-person answers are composed operations.

The model's reply, JSON: {"blocked": bool, "understood": bool, "new_scene": bool, "prompt": str, "chips": [str]}.
The light check (check_reply): blocked; understood false or an empty prompt is not understood; the prompt has at most
MAX_PROMPT_WORDS words and commands.state.MAX_FREE_PROMPT_CHARS characters; on a car scene (the prompt on screen is
the car template) that is not a new scene, the new prompt is still the car template with a drivable road
(car_template.template_problems). A reply that fails is asked again once, with the reasons, when the first call took
under RETRY_SHARE (3/7) of the wait; otherwise the dictionary reads the sentence (the fallback).

Content block: a small word list (BLOCKED_WORDS) is checked first, with no call at all; the model also flags
sexual, violent or hateful requests with "blocked": true in the same reply.

Determinism: the cache is keyed by (composed prompt of the scene before, sentence_key(sentence)); for a free scene the
composed prompt is its own prompt. A result is chosen once: a Sonnet result, a block or a "not understood" is saved to
a JSON file (written atomically) and never regenerated, also across restarts. A timeout (final_timeout_s, 3.5 s by
default), a failed call or a reply that fails the check gives the fallback, final for that key for the life of the
process on a composed scene (a late answer is dropped, late_dropped), and not saved to disk, so a passing outage does
not disable a sentence for good. On a free scene a fallback is not kept: the dictionary cannot edit that scene, so
the next try asks again.
"""

import collections
import concurrent.futures
import copy
import dataclasses
import enum
import hashlib
import json
import logging
import os
import re
import statistics
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

from . import car_template, people
from commands.state import MAX_BATCH, MAX_FREE_PROMPT_CHARS, MAX_LABEL, SceneState, free_chips, free_prompt, is_free

from .foundry import FoundryMessages, settings_from_env
from .slow_reader import validate as validate_operation

log = logging.getLogger("engine.final_reader")

# Thinking off at effort low. Adaptive thinking at effort medium graded a little better on a short test prompt, but on
# the real system prompt the regression sweep (2026-11-15, 209 chains) measured p50 2.16 s against 1.65 s, 14 timeouts
# against 5 at the 3 s wait, and replies that began with reasoning prose instead of JSON.
# The wait is 3.5 s: on a long chain (about 300 tokens of prompt) the sweep measured a median call of 1.98 s, and 2 of 3
# runs of "make it slow" timed out at 3.0 s (#285). The lead rejected 5.5 s as too slow on screen.
DEFAULT_TIMEOUT_S = 3.5
DEFAULT_EFFORT = "low"              # output_config.effort
DEFAULT_THINKING = "between_tools"  # thinking.type: between_tools is thinking off
DEFAULT_MAX_TOKENS = 1024
HTTP_TIMEOUT_S = 10.0        # a call may finish after the wait; its answer is then dropped
# A retry starts only when the first call took at most this share of the wait, so a second call as long as the first
# still fits with room to spare: 1.5 s of the live 3.5 s, as it was with half of 3.0 s. With half of 3.5 s (1.75 s),
# retries after 1.6 s first calls ran past the wait (#285).
RETRY_SHARE = 3 / 7
WORKERS = 4
DEFAULT_CACHE_FILE = "~/.cache/madhacks/final_reader.json"
DEFAULT_DEMO_PROMPTS = str(Path(__file__).resolve().parent / "demo_prompts.json")
LOG_CHARS = 100
FENCE = re.compile(r"^\s*```[A-Za-z]*\s*|\s*```\s*$")
# The video model's text encoder (umT5) reads at most 512 tokens. Sonnet's prompts measured at most 1.4 tokens a word
# and at least 3.9 characters a token (54 recorded prompts, the real tokenizer), so 300 words (420 tokens) and
# commands.state.MAX_FREE_PROMPT_CHARS (1900 characters, 487 tokens) stay under it. The cap only guards: a cut that takes
# the sentence with the change away gives an unchanged prompt (Verdict.UNCHANGED).
MAX_PROMPT_WORDS = 300
ASKED_PROMPT_WORDS = 100     # what the system prompt asks for (60 to 100), so a long reply still fits the cap
# Characters of a pinned or measured operation's phrase: a world sentence from commands/world.py is about 250, and a
# saved racing car's phrase ends with the lane cue the old final reader added (", white dashed lane markings ...").
CACHED_MAX_PHRASE = 400 + len(", white dashed lane markings rush toward the viewer")

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
    SONNET = "sonnet"                  # a free scene written by the model: saved
    BLOCKED = "blocked"                # refused (word list or the model): saved when the model said so
    NOT_UNDERSTOOD = "not_understood"  # the model said the sentence is vague or not about the picture: saved
    FALLBACK = "fallback"              # timeout, error or a reply that failed the check: the dictionary path
    PINNED = "pinned"                  # from the demo prompts file: never called, never saved, never evicted
    TEMPLATE = "template"              # a measured car world composed by the engine: never called, never saved
    PERSON = "person"                  # the engine's add of a named person when the model gave nothing usable

    @property
    def saved(self) -> bool:
        return self in (Outcome.SONNET, Outcome.BLOCKED, Outcome.NOT_UNDERSTOOD)


@dataclasses.dataclass(frozen=True)
class Entry:
    outcome: Outcome
    operations: tuple = ()
    prompt: str | None = None
    # The scene the operations were made for (scene_record). The key holds only its composed prompt, which leaves
    # out zero-strength chips and sorts changes by kind; a hit is used only on this same scene, with its change ids
    # mapped to the live ones (scene_ids). None: an entry with no operations to map.
    scene_before: dict | None = None

    def result(self) -> dict | None:
        if self.outcome in (Outcome.SONNET, Outcome.PINNED, Outcome.TEMPLATE, Outcome.PERSON):
            return {"operations": [dict(op) for op in self.operations], "prompt": self.prompt}
        if self.outcome is Outcome.BLOCKED:
            return {"blocked": True}
        if self.outcome is Outcome.NOT_UNDERSTOOD:
            return {"not_understood": True}
        return None


def scene_record(snapshot: dict) -> dict:
    """The part of a snapshot a cached entry is tied to: the base, the origin and free prompt, and the changes in
    order, with their ids."""
    return {"base": snapshot["base"], "origin": snapshot.get("origin", "composed"), "prompt": snapshot.get("prompt"),
            "changes": [{k: c[k] for k in ("id", "label", "kind", "phrase", "strength")} for c in snapshot["changes"]]}


def scene_ids(before: dict, snapshot: dict) -> dict[str, str] | None:
    """{id in `before`: id in `snapshot`} when both are the same scene (origin, free prompt, base, and every change's
    label, kind, phrase and strength in the same order); None when they differ."""
    def shape(s):
        return (s.get("origin", "composed"), s.get("prompt"), s["base"],
                [(c["label"], c["kind"], c["phrase"], c["strength"]) for c in s["changes"]])
    if shape(before) != shape(snapshot):
        return None
    return {b["id"]: c["id"] for b, c in zip(before["changes"], snapshot["changes"])}


def blocked_word(text: str) -> str | None:
    return next((w for w in _WORD.findall(text.lower()) if w in BLOCKED_WORDS), None)


# --- movers on the car scene (dictionary results) ---------------------------------------------------------

# Things that move by themselves. On a vehicle scene the camera follows the car forward, so one with no direction of its
# own is drawn moving backwards (live, 2026-11-15: "an airplane in the sky" from the dictionary flew backwards, #272).
# Sonnet's prompts state the direction themselves (the system prompt); this is for the dictionary's objects.
MOVER_WORDS = frozenset({"airplane", "aeroplane", "plane", "jet", "helicopter", "bird", "birds", "eagle", "drone",
                         "balloon", "car", "truck", "van", "bus", "motorcycle", "motorbike", "bike", "bicycle", "horse",
                         "horses", "dog", "deer", "boat"})
HEADING = "seen from behind, moving forward away from the camera in the same direction as the car"
# Words that already say how a thing moves, or that it stays where it is.
MOTION_SAID_WORDS = frozenset({"forward", "away", "behind", "reversing", "reverse", "backwards", "backward", "toward",
                               "towards", "approaching", "approaches", "oncoming", "parked", "parking", "stopped",
                               "stationary", "standing", "idle", "idling", "still", "waiting", "abandoned", "crashed",
                               "wrecked", "sitting"})
STAYING_WORDS = frozenset({"anchored", "moored", "docked", "sleeping", "asleep", "perched", "resting", "lying", "lies",
                           "grazing", "sits", "landed", "hovering", "floating", "stands", "still"})


def with_heading(phrase: str) -> str:
    """An object phrase on a vehicle scene with HEADING when it names a mover and says nothing about its direction."""
    words = set(re.findall(r"[a-z]+", phrase.lower()))
    if words & MOVER_WORDS and not words & (MOTION_SAID_WORDS | STAYING_WORDS):
        return f"{phrase.rstrip(' .,')}, {HEADING}"
    return phrase


# --- the system prompt ----------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a skilled prompt writer for a live text-to-video model (Wan 2.1, text encoder umT5). The
video plays continuously; each spoken sentence edits it. You get the prompt on screen and one sentence, and you write
the complete new prompt. Reply with one JSON object only:
{"blocked": <true or false>, "understood": <true or false>, "new_scene": <true or false>,
 "prompt": "<the complete new video prompt>", "chips": ["<label>", ...]}

The user message is JSON: {"prompt": <the video prompt on screen now>, "chips": <the labels shown now>,
"sentence": <what the user said>, "car_scene": <true when the prompt on screen is the car template below>}.
It may also have "view_change": true (the sentence asks for another camera or view), "person_chip" (the sentence adds
a named person, already described in the sentence; use this label as their chip) and "rejected" (your earlier reply
broke the rules given: reply again, fixed).

A good prompt is one complete scene description, one paragraph of concrete visual words, 60 to {asked_words}
words: the subject and where it is in the frame, what it does and its direction of travel, the camera, the setting
(ground, plants, buildings, background), the weather and light, and the style. The car template below, measured on
the GPU, is the example of a good prompt.

Rules:
- Return the complete new prompt, never a diff. Keep everything the sentence does not change word for word (the
  subject, the objects, their positions, the camera, the light), so the picture does not jump. Change only what the
  sentence asks, and make that change clearly visible.
- The prompt is at most {asked_words} words, also when the scene has grown: describe the older things more briefly
  (keep each of them, in fewer words) instead of growing the prompt. The thing the sentence adds or changes is always
  written in full. When the prompt on screen is already longer than {asked_words} words, the new prompt is shorter
  than it: compress the oldest details first (a long reply is slow, and a late reply is lost).
- Camera: keep the camera wording as it is unless the user asks for another camera, framing or view (zoom in or out,
  a wider or closer shot, side view, aerial or drone view, front view, low angle, pan), or a vehicle starts driving
  (the camera then follows it, below); then rewrite the camera wording to match ("zoom out": a wide establishing shot from further away, the subject smaller in the frame).
- No sounds, no story or time order ("now", "again", "still", "no longer").
- A video model cannot draw "no": it draws every noun it reads. A request to remove or not have something becomes a
  positive description of what is there instead, with the thing's word left out: "no clouds" -> "a clear, cloudless
  deep blue sky"; "no grass" -> "bare sandy ground"; "no people" -> "an empty street"; "no more snow" -> "green grass
  under a clear sky"; "get rid of the fog" -> "clear, crisp air with sharp distant hills"; "no trees" -> "an open
  grassy plain to the horizon"; "kill the tree's leaves" -> "bare branches". Such a sentence is always understood.
- A sentence that asks again for what the prompt already says means the user does not see it yet: make it stronger
  and clearer in the words (bigger, closer to the centre, more of it), so the picture changes.
- A change of place alone ("take me to the desert", "now I want a savannah", "make it a beach") replaces the place:
  the ground, the background and the light. The objects stay unless the user says otherwise. The video model cannot
  infer what happens to them, so for every object say where it is, what it stands or drives on, and which way it
  faces, in words that fit the new place. A thing that cannot stay where it was gets a plausible spot: an animal
  stands on the ground beside the road.
- A new place or setting named in any words ("city background", "in the city", "middle of the city", "at the
  beach") replaces the old place in the same way; "keep everything word for word" does not keep the old place. The
  landscape of the old place (the meadow, the oak tree, the grass) goes unless the user mentions it. The subject the
  user is talking about (the car) becomes the main subject, placed in the new place with a direction and camera that
  suit it: a car in a city drives along a street, with the camera behind it unless the user asks otherwise.
- A sentence that describes a whole new scene, a place together with its own main subject or action ("I now want a
  city at night with a car driving on the road", "show me a harbour with fishing boats", "a mountain village with
  people skating on a frozen lake"), starts a new scene: "new_scene" is true, the prompt describes only what the
  sentence names, and every earlier object, plant change and chip is dropped (the oak tree, bare branches, a lion),
  unless the sentence keeps it by name ("... and keep the lion").
- A car or other road vehicle is always on a paved road, street, bridge or car park, also when it is parked, also at a
  beach or in a field: never on sand, grass, snow or water.
- A place after near, by, beside, next to or behind that the prompt does not have only says where the new thing goes:
  "add snow near the river" adds snow and no river.
- Ground, plants and climate ("arid", "no grass", "dry", "lush", "cut the grass", "kill the tree's leaves") change the
  ground, the plants and the place in the prompt. Nothing stops a later change of place: the prompt is free text.
- Objects get explicit size and position words. The first object is large and in the centre ("a large brown horse
  standing beside the oak tree in the centre of the frame, facing the camera, large and clearly visible"); a later one
  takes a free place ("a large male lion standing in the foreground at the left of the frame, facing the camera").
- A car or other vehicle the user adds or names drives by default, on a road; it is parked only when the user says
  parked or stopped. When a vehicle drives, the road runs through the whole frame along its direction of travel, and
  the camera follows the vehicle in a smooth tracking shot at its speed and keeps it centred (the car template's
  camera), unless the user asks for a static or other camera. A driving vehicle is never "facing the camera" unless
  the user asks to see its front.
- Counts agree everywhere in the prompt. "add another tree" rewrites the scene with the new number ("a wide shot of
  a meadow with two oak trees, one in the centre and a second, taller one in the foreground at the left"), never
  "a single oak tree" next to "a second tree": the model then draws one.
- Every moving subject (vehicles, aircraft, boats, animals, people) states its direction of travel relative to the
  camera, consistent with the camera. On a car scene anything that moves travels forward away from the camera in the
  same direction as the car, seen from behind, unless the user names another direction. The video model draws the
  parts it reads, not the words "seen from behind", so say which parts face the camera and where the front points:
  an aircraft flying away is "seen from behind, its tail fin and the backs of its engines toward the camera, its nose
  pointing away toward the horizon" ("a large white passenger airplane high above the road in the upper centre of
  the frame, seen from behind, its tail fin and the backs of its engines toward the camera, its nose pointing away
  toward the horizon, flying forward in the same direction as the car"); a vehicle driving away shows "its rear
  lights and rear bumper toward the camera"; an animal or person walking away shows "its back and tail toward the
  camera, its head turned away". A side view shows "the whole length of the car from the side, front wheels and
  rear wheels both in view, its nose pointing to the left edge of the frame".
- Seasons, weather and time of day are written visibly ("thick white snow covering the whole ground", "a dark night
  sky with a full moon"). "sunny" replaces night. More or less of something ("a little more snow") rewrites it with
  visibly more or less of it. Amount words set how much: "some fog" is a light, thin fog, "a little" is faint, "a
  lot" or "heavy" is thick; never write "thick" for "some".
- A new weather keeps the current season unless the user asks to remove it: "make it rain" on a snowy night keeps
  the snow on the ground and the night, with rain falling over it. Only what falls from the sky is replaced: snow
  falling replaces rain falling.
- A named person: when the sentence holds their visual description (the engine puts it there for people it
  knows), write that description; for any other real person describe their usual look (hair, glasses, clothes),
  never the name. "person_chip" is their chip.
- "new_scene" is true only when the user asks for a new scene in so many words ("new scene of a beach", "start over
  with a castle"), replaces the main subject itself, or describes a whole new scene (a place with its own main
  subject or action, above); the prompt then describes only the new scene. Otherwise false.

The chips: one short label (1 to 3 words, lower case, at most {max_label} characters) for each thing the user added
or changed that is still in the picture (objects, place, season, weather, time, camera), in the order they were
added. Keep the current chips that still apply, with the same labels; drop a chip when its thing is gone. The chips
match the prompt: a chip whose words are no longer in the new prompt is dropped. A new scene starts with the chips of
what the user named.

"understood" is false, with an empty prompt, only when the sentence is not about the picture at all, asks for nothing
to change ("don't add a lion", "never mind", "leave it"), or asks to undo or reset: then nothing changes on screen. A short or loose sentence about the picture ("zoom out", "more trees",
"make it cozy") is understood: write your best reading of it. A bare noun phrase ("a lion", "fireworks", "a red
balloon", "a middle-aged man in a black leather jacket") means: add that thing to the scene. "blocked" is true, with an empty prompt, when the
sentence asks for sexual, violent (gore, killing or hurting people or animals) or hateful content; plants and
objects are not violence.
"""


def system_prompt() -> str:
    return (SYSTEM_PROMPT.replace("{asked_words}", str(ASKED_PROMPT_WORDS)).replace("{max_label}", str(MAX_LABEL))
            + car_template.instructions())


# --- the sentence part of a key -------------------------------------------------------------------------

# Characters stripped from both ends of a sentence key: end punctuation and surrounding quotes, as voice and
# Wispr Flow add them ("Make it winter.", "\u201cmake it winter\u201d").
_KEY_EDGE = ".,!?;:\"'`\u201c\u201d\u2018\u2019 "


def sentence_key(text: str) -> str:
    """The sentence part of a pinned or cache key: lower case, spaces collapsed, and end punctuation
    (. , ! ? ; :) and surrounding quotes stripped from both ends. Inner words, apostrophes and punctuation are
    kept, so "Make it winter." and "  make it   winter " share the key "make it winter" and "make it winter
    please" does not. Not commands.dictionary.normalize: that one turns hyphens, commas and accented letters
    inside the sentence into spaces, which changes words. Only the key uses this; the content check and Sonnet
    get the original text. A non-string raises TypeError, which the file loaders skip as a bad entry."""
    if not isinstance(text, str):
        raise TypeError(f"a sentence must be a string, got {type(text).__name__}")
    text = " ".join(text.casefold().split())
    return " ".join(text.strip(_KEY_EDGE).split())


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
                if not outcome.saved:
                    continue
                scene = e.get("scene_before")
                if outcome is Outcome.SONNET and not isinstance(scene, dict):
                    continue  # written before entries kept their scene: its ids cannot be mapped
                out[(e["prompt_before"], sentence_key(e["sentence"]))] = Entry(  # old raw keys too
                    outcome, tuple(e.get("operations") or ()), e.get("prompt"), scene)
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def save(self, cache: dict[tuple[str, str], Entry]) -> None:
        entries = [{"prompt_before": k[0], "sentence": k[1], "outcome": v.outcome.value,
                    "operations": list(v.operations), "prompt": v.prompt, "scene_before": v.scene_before}
                   for k, v in cache.items() if v.outcome.saved]
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
    """The demo prompts file as PINNED entries keyed by (prompt_before, sentence_key(sentence)). Never raises.

    File: [entry, ...] or {"base": str, "entries": [entry, ...]}, where an entry is
    {"sentence", "prompt_before", "operations": [op, ...], "prompt"} (other keys, such as "canonical", "model" or
    "fingerprint", are notes and ignored). The entries are replayed in order on a commands.state.SceneState that
    starts at `base` (commands.state.DEFAULT_BASE for the list form). An entry whose
    prompt_before is the composed prompt of the base starts again from the base (a reset came before it). An entry
    whose prompt_before is an earlier scene of the file (not the one the entry before it leads to) is read on that
    scene: a spoken variant of a step ("make it snow" beside "make it winter") follows the step it varies.
    An entry is skipped, with a log line, when its prompt_before is no scene the entries before it reached, its
    operations do not apply (checked_operations, then one batch), or its prompt is not text.
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
        from commands.state import DEFAULT_BASE
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
    scenes: dict[str, SceneState] = {}  # composed prompt -> the first replayed scene with it, for variants
    for n, e in enumerate(entries):
        try:
            sentence, before, prompt = e["sentence"], e["prompt_before"], e["prompt"]
            if not all(isinstance(v, str) and v.strip() for v in (sentence, before, prompt)):
                raise ValueError("sentence, prompt_before and prompt must be nonempty text")
            if before == start:
                scene = SceneState(base)
            elif compose(scene.snapshot()) != before:
                if before not in scenes:
                    raise ValueError("prompt_before is no scene the entries before it reached")
                scene = scenes[before]
            scenes.setdefault(before, scene)
            ops = checked_operations(e["operations"], scene.snapshot()) if isinstance(e["operations"], list) else None
            if not ops:
                raise ValueError("the operations are not valid for that scene")
            trial = copy.deepcopy(scene)
            trial.apply(ops[0] if len(ops) == 1 else {"op": "batch", "operations": ops})
            compose(trial.snapshot())
        except Exception as exc:  # noqa: BLE001 - skip the entry, keep the others
            log.warning("final reader: demo prompt %d (%r) skipped: %r", n, _short(e.get("sentence") if isinstance(e, dict) else e), exc)
            continue
        pinned[(before, sentence_key(sentence))] = Entry(Outcome.PINNED, tuple(ops), prompt, scene_record(scene.snapshot()))
        scene = trial
        scenes.setdefault(compose(trial.snapshot()), trial)
    log.info("final reader: %d of %d demo prompts pinned from %s", len(pinned), len(entries), path)
    return pinned


# --- the light check ------------------------------------------------------------------------------------

class Verdict(enum.Enum):
    """What one reply gave. RETRY_VERDICTS are asked again once, with the reasons, when there is time; a REPAIRED
    reply is used when the second one is not better or comes too late. Only an unusable reply falls back."""
    ACCEPTED = "accepted"              # a prompt that passed the light check
    REPAIRED = "repaired"              # usable: cut to fit the length cap, or a car-template rule broken (logged)
    NOT_UNDERSTOOD = "not_understood"  # understood false, or no prompt: unmatched, nothing changes
    BLOCKED = "blocked"                # the model refused the sentence
    INVALID = "invalid"                # not JSON or not an object: asked again, else the dictionary
    UNCHANGED = "unchanged"            # the prompt is the one on screen (often a cut took the change): as INVALID
    FAILED = "failed"                  # the call failed: the dictionary path


RETRY_VERDICTS = frozenset({Verdict.INVALID, Verdict.UNCHANGED, Verdict.REPAIRED})
VERDICT_RANK = {Verdict.FAILED: 0, Verdict.INVALID: 1, Verdict.UNCHANGED: 1, Verdict.NOT_UNDERSTOOD: 2,
                Verdict.REPAIRED: 3,
                Verdict.ACCEPTED: 4, Verdict.BLOCKED: 4}


@dataclasses.dataclass(frozen=True)
class Attempt:
    verdict: Verdict
    entry: Entry
    reasons: tuple[str, ...] = ()
    reply_text: str | None = None


@dataclasses.dataclass
class Pending:
    """One call in flight. `usable` is set by the worker to its first usable (REPAIRED) entry before it asks again,
    so a retry that runs past the wait does not lose the edit."""
    usable: Entry | None = None


def reply_chips(chips, person_chip: str | None = None) -> list[str]:
    """The model's chip labels as commands.state.free_chips takes them: text only, lower case, each cut to whole words
    of at most MAX_LABEL characters, at most 16, no duplicates; the named person's label added when it is missing.
    Chips are only what the page shows, so a bad label is cleaned, never a reason to drop the whole reply."""
    out: list[str] = []
    for chip in chips if isinstance(chips, list) else []:
        if not isinstance(chip, str):
            continue
        label = ""
        for word in chip.lower().split():
            if len(f"{label} {word}".strip()) > MAX_LABEL:
                break
            label = f"{label} {word}".strip()
        if label and label not in out:
            out.append(label)
    if person_chip and person_chip not in out:
        out.append(person_chip)
    return out[:16]


# Words of a chip label that say nothing about what is drawn ("second dog", "the city", "light fog"): never the evidence
# for a chip.
_CHIP_FILLER = frozenset("""a an the of in on at to and with some more less second third another light heavy thick thin
    little bit lot very""".split())


def _chip_terms(label: str) -> set[str]:
    """The words that show `label`'s thing is in a prompt: its own content words, and for every keyword table entry
    named inside it ("winter", the "fog" of "light fog") all that entry's aliases ("snow", "frost", ...; "mist",
    "haze", ...)."""
    from commands.dictionary import KEYWORDS, normalize
    words = normalize(label).split()
    terms = {w for w in words if w not in _CHIP_FILLER}
    labels = {KEYWORDS[" ".join(words[i:j])]["label"] for i in range(len(words)) for j in range(i + 1, len(words) + 1)
              if " ".join(words[i:j]) in KEYWORDS and " ".join(words[i:j]) not in _CHIP_FILLER}
    terms |= {k for k, e in KEYWORDS.items() if e["label"] in labels}
    return terms


def _says(prompt: str, term: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(term)}(?:s|es)?(?![a-z])", prompt.lower()) is not None


def chips_in_prompt(chips: list[str], before: str, after: str, keep: str | None = None) -> tuple[list[str], list[str]]:
    """(the chips to show, the chips dropped). A chip is dropped when its words left the prompt: some term of it is
    in the prompt on screen `before` and none is in the new prompt `after` ("winter" after a rain reply that took all
    the snow out). A chip with no words in either prompt ("zoom out", "slow motion") cannot be judged and stays, and
    so does `keep` (the named person's chip, whose name is never in the prompt)."""
    shown, dropped = [], []
    for chip in chips:
        terms = _chip_terms(chip)
        left = chip != keep and terms and any(_says(before, t) for t in terms) and not any(_says(after, t) for t in terms)
        (dropped if left else shown).append(chip)
    return shown, dropped


def _fits(text: str) -> bool:
    return len(text.split()) <= MAX_PROMPT_WORDS and len(text) <= MAX_FREE_PROMPT_CHARS


def fit_prompt(prompt: str) -> tuple[str, tuple[str, ...]]:
    """`prompt` cut to the length cap, with what was cut: whole sentences from the end (the car template's style
    sentence stays last), then words of a single sentence that is still too long."""
    if _fits(prompt):
        return prompt, ()
    from commands.world import STYLE
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", prompt) if x]
    style = [sentences.pop()] if len(sentences) > 2 and sentences[-1] == STYLE else []
    cut = []
    while len(sentences) > 1 and not _fits(" ".join(sentences + style)):
        cut.append(sentences.pop())
    text = " ".join(sentences + style)
    if not _fits(text):
        words = text.split()
        while words and not _fits(" ".join(words)):
            words.pop()
        text = " ".join(words)
    return text, (f"the prompt had {len(prompt.split())} words, over the limit of {ASKED_PROMPT_WORDS}; "
                  f"cut {len(cut)} sentences from the end",)


def retry_fits(elapsed_s: float, timeout_s: float) -> bool:
    """True when the first call (`elapsed_s`) took at most RETRY_SHARE of the wait: a retry then fits."""
    return elapsed_s <= timeout_s * RETRY_SHARE + 1e-9


def check_reply(reply, user: dict) -> Attempt:
    """The light check of one parsed reply for the user message `user` (the dictionary it was sent). It repairs what
    it can instead of refusing a reasonable sentence: only a reply that is not an object falls back."""
    if not isinstance(reply, dict):
        return Attempt(Verdict.INVALID, Entry(Outcome.FALLBACK), ("the reply is not a JSON object",))
    if reply.get("blocked") is True:
        return Attempt(Verdict.BLOCKED, Entry(Outcome.BLOCKED))
    prompt = reply.get("prompt")
    if reply.get("understood") is False or not isinstance(prompt, str) or not prompt.strip():
        return Attempt(Verdict.NOT_UNDERSTOOD, Entry(Outcome.NOT_UNDERSTOOD),
                       ("understood false" if reply.get("understood") is False else "no prompt",))
    prompt, problems = fit_prompt(" ".join(prompt.split()))
    if prompt == " ".join(str(user.get("prompt") or "").split()):
        why = "the prompt is the one on screen, so nothing would change" + (f" ({problems[0]})" if problems else "")
        return Attempt(Verdict.UNCHANGED, Entry(Outcome.FALLBACK), (why,))
    if user.get("car_scene") and reply.get("new_scene") is not True:
        problems += tuple(car_template.template_problems(prompt, view_change=bool(user.get("view_change"))))
    chips, dropped = chips_in_prompt(reply_chips(reply.get("chips"), user.get("person_chip")),
                                     str(user.get("prompt") or ""), prompt, user.get("person_chip"))
    if dropped:
        log.info("final reader: chips %s dropped: their words left the prompt", dropped)
    op = {"op": "free_scene", "prompt": prompt, "chips": chips}
    return Attempt(Verdict.REPAIRED if problems else Verdict.ACCEPTED, Entry(Outcome.SONNET, (op,), prompt), problems)


# --- the reader -----------------------------------------------------------------------------------------

class FinalReader:
    """See the module docstring. Never raises."""

    def __init__(self, state: FinalReaderState, reason: str = "", chat=None, timeout_s: float = DEFAULT_TIMEOUT_S,
                 cache_file: CacheFile | None = None, pinned: dict[tuple[str, str], Entry] | None = None) -> None:
        if state is FinalReaderState.ON and chat is None:
            raise ValueError("an ON final reader needs a chat client")
        self.state = state
        self.reason = reason
        self.timeout_s = timeout_s
        self.system_prompt = system_prompt()
        self._chat = chat
        self._file = cache_file
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], Entry] = cache_file.load() if cache_file is not None else {}
        self._pinned: dict[tuple[str, str], Entry] = dict(pinned or {})  # checked before the cache
        self.pinned_hits = 0
        self.template_hits = 0  # measured car worlds composed by the engine, no call
        self.person_hits = 0    # a sentence that only adds or removes a named person (engine/people.py), no call
        # composed prompt of a vehicle scene -> the template prompt last given for it, so a dictionary result on that
        # scene keeps its road, scene and light (fallback_prompt)
        self._given_vehicle: dict[str, str] = {}
        # composed prompt of a vehicle scene -> (the fallback prompt given for it, its world, its condition phrases),
        # since parse_world cannot read a prompt back once a condition sentence follows the objects (fallback_prompt)
        self._fallback_vehicle: dict[str, tuple[str, car_template.World, tuple[str, ...]]] = {}
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="final-reader")
        self.calls = 0
        self.cache_hits = 0
        self.blocked = 0
        self.not_understood = 0
        self.timeouts = 0
        self.late_dropped = 0
        self.rejects = 0
        self.repaired = 0
        self.verdicts: collections.Counter = collections.Counter()  # the final verdict of every call, by name
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

    def __call__(self, text: str, snapshot: dict, current_prompt: str | None = None) -> dict | None:
        """lookup, then ask, then result; no dictionary fast path (engine/reader.py FinalReadingService adds it)."""
        if self.state is not FinalReaderState.ON:
            return None
        entry = self.lookup(text, snapshot)
        if entry is None:
            entry = self.ask(text, snapshot, current_prompt)
        return self.result(entry, text, snapshot)

    def lookup(self, text: str, snapshot: dict) -> Entry | None:
        """The answer known without a call: the content block, a pinned or cached entry (a cached FALLBACK too: a
        fallback is final for the life of the process), or a measured car world. None: nothing known yet."""
        if self.state is not FinalReaderState.ON:
            return Entry(Outcome.FALLBACK)
        word = blocked_word(text)
        if word is not None:
            with self._lock:
                self.blocked += 1
            log.info("final reader: blocked %r (word list: %r), no call", _short(text), word)
            return Entry(Outcome.BLOCKED)
        key = self._key(text, snapshot)
        with self._lock:
            pinned, cached = self._pinned.get(key), self._cache.get(key)
        for found, counter in ((pinned, "pinned_hits"), (cached, "cache_hits")):
            hit = self._from_cache(key, found, snapshot) if found is not None else None
            if hit is not None:
                with self._lock:
                    setattr(self, counter, getattr(self, counter) + 1)
                return hit
        person = self._person(text, snapshot, people.whole_sentence(text, snapshot))
        if person is not None:
            with self._lock:
                self.person_hits += 1
            return person
        measured = self._measured(text, snapshot)
        if measured is not None:
            with self._lock:
                self.template_hits += 1
        return measured

    def ask(self, text: str, snapshot: dict, current_prompt: str | None = None) -> Entry:
        """One Sonnet call (and one retry when there is time) for `text` on `snapshot`, whose prompt on screen is
        `current_prompt` (the rich prompt the model sees; the composed prompt when None). Waits at most timeout_s."""
        if self.state is not FinalReaderState.ON:
            return Entry(Outcome.FALLBACK)
        from commands.composer import compose
        current = current_prompt if isinstance(current_prompt, str) and current_prompt.strip() else compose(snapshot)
        key = self._key(text, snapshot)
        with self._lock:
            self.calls += 1
        pending = Pending()
        fut = self._pool.submit(self._call, text, snapshot, current, pending)
        late = False
        try:
            entry = fut.result(timeout=self.timeout_s)
        except concurrent.futures.TimeoutError:
            late, entry = True, pending.usable or Entry(Outcome.FALLBACK)
            if pending.usable is not None:
                log.info("final reader: the retry for %r ran out of time; using its first reply", _short(text))
        with self._lock:
            if late:
                self.timeouts += 1
            if entry.outcome is Outcome.SONNET:
                entry = dataclasses.replace(entry, scene_before=scene_record(snapshot))
            # A fallback is final for the process on a composed scene, where the dictionary acts. On a free scene the
            # dictionary refuses edits, so a kept fallback would make the sentence fail there until a restart: the
            # next try asks again.
            if entry.outcome is not Outcome.FALLBACK or not is_free(snapshot):
                self._cache[key] = entry
            if entry.outcome is Outcome.BLOCKED:
                self.blocked += 1
            elif entry.outcome is Outcome.NOT_UNDERSTOOD:
                self.not_understood += 1
            snapshot_of_cache = dict(self._cache) if entry.outcome.saved else None
        if late:  # outside the lock: the callback takes it, and runs at once if the call has just finished
            fut.add_done_callback(lambda _f: self._late())
            log.info("final reader: no answer for %r within %.1f s; the dictionary reads it", _short(text),
                     self.timeout_s)
        if snapshot_of_cache is not None and self._file is not None:
            self._file.save(snapshot_of_cache)
        return entry

    def result(self, entry: Entry, text: str, snapshot: dict) -> dict | None:
        """The answer for FinalReadingService (module docstring), with the engine's add of a named person when the
        model gave nothing usable, remembering a vehicle scene's template prompt for fallback_prompt."""
        return self._given(self._or_person(entry, text, snapshot), snapshot)

    def user_message(self, text: str, snapshot: dict, current: str) -> dict:
        user = {"prompt": current, "chips": [c["label"] for c in snapshot["changes"]],
                "sentence": people.for_model(text), "car_scene": car_template.is_template(current)}
        if user["car_scene"] and car_template.asks_view_change(text):
            user["view_change"] = True
        mention = people.mention(text)
        if mention.intent is people.Intent.ADD:
            user["person_chip"] = mention.person.label
        return user

    def stats(self) -> dict:
        with self._lock:
            return {"state": self.state.value, "reason": self.reason or None, "prompt_spec": "free scene",
                    "calls": self.calls, "cache_hits": self.cache_hits,
                    "pinned": len(self._pinned), "pinned_hits": self.pinned_hits,
                    "template_hits": self.template_hits, "person_hits": self.person_hits, "blocked": self.blocked,
                    "not_understood": self.not_understood,
                    "timeouts": self.timeouts, "late_dropped": self.late_dropped, "rejects": self.rejects,
                    "repaired": self.repaired, "retries": self.retries, "errors": self.errors,
                    "verdicts": dict(sorted(self.verdicts.items())),
                    "p50_ms": round(statistics.median(self._recent_ms), 1) if self._recent_ms else None,
                    "last_ms": None if self.last_ms is None else round(self.last_ms, 1),
                    "cache_size": len(self._cache), "timeout_s": self.timeout_s,
                    "cache_file": None if self._file is None else str(self._file.path)}

    # --- internals -------------------------------------------------------------------------

    @staticmethod
    def _key(text: str, snapshot: dict) -> tuple[str, str]:
        from commands.composer import compose  # for a free scene its own prompt
        return compose(snapshot), sentence_key(text)  # pinned and cache keys are normalized the same way

    def _late(self) -> None:
        with self._lock:
            self.late_dropped += 1
        log.info("final reader: a late answer was dropped")

    def _given(self, entry: Entry, snapshot: dict) -> dict | None:
        """entry.result(), remembering the template prompt of a composed vehicle scene for the scene it leads to."""
        if entry.prompt is not None and entry.outcome in (Outcome.PINNED, Outcome.TEMPLATE, Outcome.PERSON):
            from commands.composer import compose
            try:
                after = scene_after(snapshot, list(entry.operations))
                if not is_free(after) and car_template.vehicle_subject(after["base"]) is not None:
                    with self._lock:
                        self._given_vehicle[compose(after)] = entry.prompt
            except (ValueError, TypeError, KeyError):
                pass  # only a hint for a later fallback
        return entry.result()

    def _or_person(self, entry: Entry, text: str, snapshot: dict) -> Entry:
        """`entry`, or for a FALLBACK on a composed scene and a sentence that adds or removes a named person
        (engine/people.py), the engine's own add or remove with the plain prompt (or the car template): the
        dictionary would label the chip with the description."""
        if entry.outcome is not Outcome.FALLBACK:
            return entry
        person = self._person(text, snapshot, people.person_add(text, snapshot) or people.person_remove(text, snapshot))
        return entry if person is None else person

    def _person(self, text: str, snapshot: dict, add: dict | None) -> Entry | None:
        """A PERSON entry for the engine's add or remove `add` of a named person on a composed scene (the plain
        prompt, or the car template on a vehicle scene), or None when there is none or it does not apply."""
        if add is None or is_free(snapshot):
            return None
        from commands.composer import compose
        try:
            ops = checked_operations([add], snapshot)
            if ops is None:
                raise ValueError("the engine's operation is not valid for this scene")
            after = scene_after(snapshot, ops)
            subject = car_template.vehicle_subject(after["base"])
            if subject is not None:
                world = self._current_vehicle_world(snapshot, subject) or car_template.WORLDS[car_template.START_WORLD]
                prompt = car_template.world_prompt(subject, world, car_template.scene_objects(after))
            else:
                prompt = compose(after)
        except (ValueError, TypeError, KeyError) as exc:
            log.warning("final reader: the engine's %s for %r does not apply (%r)", add["op"], _short(text), exc)
            return None
        log.info("final reader: %r -> the engine's %s of a named person, no call", _short(text), add["op"])
        return Entry(Outcome.PERSON, tuple(ops), prompt)

    def fallback_prompt(self, before: dict, after: dict) -> str | None:
        """The prompt for the dictionary's result of a sentence this reader gave None for (a timeout, an error or an
        unusable reply), from the scene `before` the sentence to the scene `after` it. On a vehicle scene it is the
        car template, so "add a dog" after a timeout keeps the car driving forward on the same road, scene and light,
        with the dog's phrase as an object sentence. None (the composed prompt, as before) on any other scene, or
        when the vehicle scene's world is not one the template knows.

        The world is the one last given for `before` when the dictionary kept the world, season, weather and time
        changes as they were; otherwise the measured world of `after`. Season, weather and time phrases that world
        does not already say (the ones just added, or all of them for a measured world) follow the object sentences;
        they are kept with the world, so a later fallback on that scene keeps both."""
        subject = car_template.vehicle_subject(after.get("base"))
        if subject is None or is_free(after) or is_free(before):
            return None
        from commands.composer import compose, strength_phrase

        def settings(scene: dict) -> list[tuple[str, float]]:
            return [(c["id"], c["strength"]) for c in scene["changes"] if c["kind"] != "extra"]

        old_ids = {c["id"] for c in before["changes"]}
        added = [c for c in after["changes"] if c["id"] not in old_ids]
        given = None
        if (before.get("base") == after["base"] and set(settings(before)) <= set(settings(after))
                and not any(c["kind"] == "world" for c in added)):
            given = self._given_world_said(before, subject)
        if given is not None:
            world, earlier = given
        else:
            world, earlier, added = car_template.current_world(after), (), after["changes"]
        if world is None:
            return None
        said = (*earlier, *(strength_phrase(c["phrase"], c["strength"]) for c in added
                            if c["kind"] in ("season", "weather", "time") and c["strength"] > 0))
        objects = [with_heading(o) for o in car_template.scene_objects(after)]
        prompt = car_template.world_prompt(subject, world, [*objects, *said])
        with self._lock:
            self._given_vehicle[compose(after)] = prompt
            self._fallback_vehicle[compose(after)] = (prompt, world, said)
        log.info("final reader: the dictionary's result on a vehicle scene gets the car template (%s)", world.name)
        return prompt

    def _given_world_said(self, snapshot: dict, subject: str) -> "tuple[car_template.World, tuple[str, ...]] | None":
        """The world of the prompt last given for this vehicle scene and the condition phrases it says after the
        objects (none for a Sonnet or measured prompt; a fallback prompt's own), or None."""
        from commands.composer import compose
        with self._lock:
            key = compose(snapshot)
            given, fallback = self._given_vehicle.get(key), self._fallback_vehicle.get(key)
        world = car_template.parse_world(given, subject, car_template.scene_objects(snapshot))
        if world is not None:
            return world, ()
        if fallback is not None and fallback[0] == given:
            return fallback[1], fallback[2]
        return None

    def _given_world(self, snapshot: dict, subject: str) -> "car_template.World | None":
        """The road, scene and light of the prompt last given for this vehicle scene, or None."""
        given = self._given_world_said(snapshot, subject)
        return None if given is None else given[0]

    def _current_vehicle_world(self, snapshot: dict, subject: str) -> "car_template.World | None":
        """The road, scene and light of the prompt last given for this vehicle scene, else its measured world."""
        return self._given_world(snapshot, subject) or car_template.current_world(snapshot)

    def _measured(self, text: str, snapshot: dict) -> Entry | None:
        """A TEMPLATE entry when `text` is a measured car world for `snapshot` (car_template.Route.MEASURED), else
        None. Deterministic: the same scene and sentence always give the same operations and words. On a free scene
        only a new vehicle scene is measured (a world change there is the model's)."""
        plan = car_template.plan(text, snapshot)
        if plan.route is not car_template.Route.MEASURED or (is_free(snapshot) and not plan.new_scene):
            return None
        from commands.composer import compose
        try:
            ops = checked_operations(car_template.measured_operations(plan, snapshot), snapshot)
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

    def _from_cache(self, key, entry: Entry, snapshot: dict) -> Entry | None:
        """The cached or pinned `entry` for `snapshot`, or None when it does not apply (a miss: the sentence is read
        again). A SONNET or PINNED entry applies only to the scene it was made for (scene_ids), with its change
        ids mapped to the live ones, and only when every operation still validates. BLOCKED, NOT_UNDERSTOOD and
        FALLBACK apply."""
        if entry.outcome not in (Outcome.SONNET, Outcome.PINNED):
            return entry
        ids = scene_ids(entry.scene_before, snapshot) if entry.scene_before is not None else None
        if ids is None:
            log.info("final reader: the saved result for %r was made for another scene with the same prompt; "
                     "reading it again", _short(key[1]))
            return None
        ops = tuple({**op, "id": ids[op["id"]]} if "id" in op else dict(op) for op in entry.operations)
        if checked_operations([dict(op) for op in ops], snapshot) == [dict(op) for op in ops]:
            return dataclasses.replace(entry, operations=ops, scene_before=scene_record(snapshot))
        log.info("final reader: the cached operations for %r no longer fit the scene; reading it again", _short(key[1]))
        return None

    def _call(self, text: str, snapshot: dict, current: str, pending: Pending | None = None) -> Entry:
        """Worker thread: one Sonnet call, checked, and one retry with the reasons when the reply fails the light
        check and a second call as long as the first still fits in the wait (retry_fits: on the live 3.5 s wait, a
        first call under 1.5 s). Never raises."""
        t0 = time.perf_counter()
        first = self._attempt(text, snapshot, current, None)
        best = first
        if first.verdict is Verdict.REPAIRED and pending is not None:
            pending.usable = first.entry  # a retry that runs past the wait must not lose this edit
        if first.verdict in RETRY_VERDICTS:
            elapsed = time.perf_counter() - t0
            if retry_fits(elapsed, self.timeout_s):
                with self._lock:
                    self.retries += 1
                log.warning("final reader: asking again for %r: %s", _short(text), "; ".join(first.reasons)[:400])
                second = self._attempt(text, snapshot, current, first)
                if VERDICT_RANK[second.verdict] >= VERDICT_RANK[first.verdict]:
                    best = second
            else:
                log.warning("final reader: no time to ask again for %r (%.0f ms of %.1f s used)", _short(text),
                            elapsed * 1e3, self.timeout_s)
        with self._lock:
            self.verdicts[best.verdict.value] += 1
            if best.verdict is Verdict.FAILED:
                self.errors += 1
            elif best.verdict in (Verdict.INVALID, Verdict.UNCHANGED):
                self.rejects += 1
            elif best.verdict is Verdict.REPAIRED:
                self.repaired += 1
        if best.verdict is Verdict.REPAIRED:
            log.warning("final reader: %r used with repairs: %s", _short(text), "; ".join(best.reasons)[:400])
        if best.verdict in (Verdict.FAILED, Verdict.INVALID, Verdict.UNCHANGED):
            log.warning("final reader: no usable reply for %r: %s", _short(text), "; ".join(best.reasons)[:400])
        elif best.verdict is Verdict.BLOCKED:
            log.info("final reader: the model blocked %r", _short(text))
        elif best.verdict is Verdict.NOT_UNDERSTOOD:
            log.info("final reader: the model did not understand %r", _short(text))
        else:
            log.info("final reader: %r -> free scene %r, chips %s", _short(text), _short(best.entry.prompt),
                     best.entry.operations[0]["chips"])
        return best.entry

    def _attempt(self, text: str, snapshot: dict, current: str, previous: "Attempt | None") -> Attempt:
        """One call. `previous` is the earlier attempt for this sentence: its reply and reasons are sent back."""
        t0 = time.perf_counter()
        user = self.user_message(text, snapshot, current)
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
        attempt = check_reply(reply, user)
        log.info("final reader: %r reply %s in %.0f ms", _short(text), attempt.verdict.value, ms)
        return dataclasses.replace(attempt, reply_text=reply_text)


# --- operations the engine itself applies (pinned, measured, named people) -------------------------------

def checked_operations(ops: list, snapshot: dict) -> list[dict] | None:
    """`ops` as commands/state.py operations, or None if one is invalid: each is checked with slow_reader.validate
    (phrases up to CACHED_MAX_PHRASE characters), a free_scene with commands.state.free_prompt and free_chips, and a
    world add of a place commands.world knows gets its tested phrase (canonical_world: the measured car worlds name
    only their road). Undo and reset stay with the dictionary, and an operation naming a change id after a new scene
    is invalid (those ids are gone)."""
    out: list[dict] = []
    rebuilt = False  # a new_scene or free_scene came first: the old ids no longer exist
    for op in ops if isinstance(ops, list) else [None]:
        if isinstance(op, dict) and op.get("op") == "free_scene":
            try:
                clean = {"op": "free_scene", "prompt": free_prompt(op.get("prompt")), "chips": free_chips(op.get("chips"))}
            except ValueError:
                return None
        else:
            clean = validate_operation(op, snapshot, CACHED_MAX_PHRASE) if isinstance(op, dict) else None
        if clean is None or clean["op"] in ("undo", "reset") or (rebuilt and "id" in clean):
            return None
        rebuilt = rebuilt or clean["op"] in ("new_scene", "free_scene")
        out.append(canonical_world(clean))
    return out if len(out) <= MAX_BATCH else None


def canonical_world(op: dict) -> dict:
    """A world add for a place commands.world knows ("desert", "the city") gets its label and tested phrase,
    the same change the keyword table makes for that place. Any other operation is returned as it is."""
    if op.get("op") != "add" or op.get("kind") != "world":
        return op
    from commands.dictionary import normalize
    from commands.world import WORLDS
    known = WORLDS.get(normalize(op["label"]))
    return {**op, "label": known["label"], "phrase": known["phrase"]} if known else op


def scene_after(snapshot: dict, operations: list[dict]) -> dict:
    """The snapshot after `operations`, applied to a copy of `snapshot` (commands.state.SceneState.from_snapshot,
    ids kept). Raises ValueError (and others) when an operation does not apply."""
    trial = SceneState.from_snapshot(snapshot)
    for op in operations:
        trial.apply(copy.deepcopy(op))
    return trial.snapshot()


# Changed whenever the engine builds prompts differently from the same reply (the car template, forward motion,
# the object order, ...), so the disk cache is not reused across that change even when the system prompt is the same.
PROMPT_BUILD = "2026-11-15 free scenes written by Sonnet (#271)"


def fingerprint(system_prompt: str, model: str, build: str = PROMPT_BUILD) -> str:
    return hashlib.sha256(json.dumps([system_prompt, model, build]).encode()).hexdigest()[:16]


def call_options(environ=os.environ) -> dict:
    """The Sonnet call's effort, thinking and max_tokens from FINAL_READER_EFFORT (low, medium, high; default
    DEFAULT_EFFORT, low), FINAL_READER_THINKING (the thinking type; default DEFAULT_THINKING, between_tools, which is
    thinking off; adaptive turns it on) and FINAL_READER_MAX_TOKENS (default DEFAULT_MAX_TOKENS, 1024), so they can be
    measured against each other without a code change."""
    out = {"effort": environ.get("FINAL_READER_EFFORT") or DEFAULT_EFFORT,
           "thinking": environ.get("FINAL_READER_THINKING") or DEFAULT_THINKING}
    if out["effort"] not in ("low", "medium", "high"):
        raise ValueError(f"FINAL_READER_EFFORT must be low, medium or high, got {out['effort']!r}")
    tokens = environ.get("FINAL_READER_MAX_TOKENS")
    out["max_tokens"] = int(tokens) if tokens else DEFAULT_MAX_TOKENS
    return out


def load(mode: str, timeout_s: float = DEFAULT_TIMEOUT_S, cache_path: str | None = DEFAULT_CACHE_FILE,
         environ=os.environ, chat=None, demo_prompts: str | None = DEFAULT_DEMO_PROMPTS) -> FinalReader:
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
        chat = FoundryMessages.from_settings(settings, HTTP_TIMEOUT_S, **call_options(environ))
    model = getattr(chat, "model", "")
    reader = FinalReader(FinalReaderState.ON, chat=chat, timeout_s=timeout_s, pinned=load_pinned(demo_prompts))
    if cache_path:
        reader._file = CacheFile(cache_path, fingerprint(reader.system_prompt, model))
        reader._cache = reader._file.load()
    log.info("final reader on: model %s (effort %s, thinking %s), free scenes, wait up to %.1f s, cache %s",
             model or "(test client)", getattr(chat, "effort", "-"), getattr(chat, "thinking", "-"), timeout_s,
             cache_path or "in memory")
    return reader


def _short(text, n: int = LOG_CHARS) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 3] + "..."

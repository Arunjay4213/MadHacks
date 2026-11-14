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
The validator gets {"details": str} only. Details it rejects keep the operations and use the composed prompt.

Places: with commands.state kind "world" (PR #88), "take me to the desert" is one world change that replaces
the old world and keeps the other changes; without it, a base edit. Only an explicit "new scene of ..." resets.

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


@dataclasses.dataclass(frozen=True)
class Entry:
    outcome: Outcome
    operations: tuple = ()
    prompt: str | None = None

    def result(self) -> dict | None:
        if self.outcome in (Outcome.SONNET, Outcome.PINNED):
            return {"operations": [dict(op) for op in self.operations], "prompt": self.prompt}
        if self.outcome is Outcome.BLOCKED:
            return {"blocked": True}
        return None


def blocked_word(text: str) -> str | None:
    return next((w for w in _WORD.findall(text.lower()) if w in BLOCKED_WORDS), None)


# --- the prompt part ------------------------------------------------------------------------------------

STYLE = "Cinematic, highly detailed, realistic, sharp focus."
# "motion blur on the road" only fits a vehicle on a road; on a meadow with a lion it is wrong.
VEHICLE_STYLE = "Cinematic, highly detailed, realistic, sharp focus, motion blur on the road."

# Measured by the steering owner with optical flow (board, 2026-11-14): a vehicle subject drives backwards in 1 of
# 3 seeds with a short prompt, and forward in 3 of 3 with this wording. Added to SUBJECT when the subject is a
# vehicle and does not already say how it moves.
VEHICLE_WORDS = ("car", "truck", "van", "bus", "motorcycle", "motorbike", "jeep", "taxi", "vehicle")
FORWARD_MOTION = ("speeding forward, driving away from the camera, the camera follows directly behind, "
                  "lane markings rush toward the viewer")

STANDIN_DETAILS = """The "details" field is the WORLD of the new scene, for the Wan2.1 text-to-video model.
The engine builds the video prompt as SUBJECT + WORLD + STYLE. SUBJECT is the scene's base sentence and STYLE is
a fixed style sentence; both are added by the engine, so never repeat, rephrase or contradict the subject, and say
nothing about the camera, shots, lenses or editing.
Write WORLD in about 50 words (never more than 60), plain present tense, no lists, no line breaks:
- First the ground or road of that place, then two or three concrete things on both sides, then the light or sky.
- Every change the new scene names (objects, animals, weather, season, time of day) must appear in WORLD.
- A place REPLACES the world. It is never added to the old one: "make it a desert" turns the meadow into a
  desert, and nothing of the old landscape (grass, meadow, fields) is described. The objects and animals of the
  scene stay, now in the new place.
- Do not add any object, creature, person or vehicle the scene does not name, and no motion or events beyond
  what it says.
- If a change adds a kind the scene already has, describe a second one with a clear position. A colour, type or
  size the user names ALWAYS wins: "add a red car racing on a road" when the scene has a red car gives "a second
  red car racing alongside it in the next lane", differing only by position. Only when the user names no
  attribute ("add a car"), give the second one another colour or type, for example "a second blue car driving
  alongside it in the next lane".
- If changes conflict (sunlight and rain), blend them plausibly, such as sun breaking through rain clouds.
Examples of WORLD:
- start: "a two-lane country road through green fields and trees under a clear blue summer sky, bright sunshine."
- desert: "a long straight desert highway between red sandstone mesas and sand dunes, heat haze, orange sunset sky."
- winter: "a snowy mountain road between pine forests covered in thick white snow, snow falling, cold blue overcast light."
- neon city: "a wet city street at night in the rain, tall skyscrapers with glowing neon signs on both sides, reflections on the asphalt."
"""


def standin_validate(prompt: str, reply) -> str | None:
    """SUBJECT + WORLD + STYLE from the canonical prompt of the new scene and the reply's details; the canonical
    prompt for empty details; None for details that are not one plain paragraph of at most 80 words."""
    details = reply.get("details") if isinstance(reply, dict) else None
    if not isinstance(details, str):
        return None
    details = details.strip()
    if not details:
        return prompt
    if "\n" in details or len(details.split()) > 80:
        return None
    base = prompt.split(";")[0].strip()
    subject = with_forward_motion(base)
    subject = subject[:1].upper() + subject[1:]
    world = details if details.endswith(".") else details + "."
    return f"{subject}, {world} {VEHICLE_STYLE if is_vehicle(base) else STYLE}"


def is_vehicle(subject: str) -> bool:
    """True when `subject` names a vehicle (VEHICLE_WORDS), as a whole word."""
    return bool(set(re.findall(r"[a-z]+", subject.lower())) & set(VEHICLE_WORDS))


def with_forward_motion(subject: str) -> str:
    """`subject` plus FORWARD_MOTION when it names a vehicle and says nothing about its direction yet."""
    words = set(re.findall(r"[a-z]+", subject.lower()))
    if is_vehicle(subject) and not words & {"forward", "away", "behind", "reversing", "backwards"}:
        return f"{subject}, {FORWARD_MOTION}"
    return subject


@dataclasses.dataclass(frozen=True)
class PromptSpec:
    """How the prompt part is asked for and checked. `validate(prompt, reply)` gets the canonical prompt of the
    new scene and the parsed reply, and gives the text the model sees, or None to reject the reply."""
    name: str
    details_instructions: str
    validate: Callable[[str, dict], str | None]


STANDIN = PromptSpec("engine stand-in", STANDIN_DETAILS, standin_validate)


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
        return PromptSpec("commands.rewriter", rules.SYSTEM_PROMPT, rules.validate_rewrite)
    except Exception as exc:  # not there yet, or not the agreed shape
        log.info("final reader: commands.rewriter not usable (%r); using the engine's stand-in prompt", exc)
        return STANDIN


class PlaceRule(enum.Enum):
    """How the model is told to handle a place command ("take me to the desert", "make it a desert")."""
    WORLD = "world"          # commands.state has kind "world" (PR #88): one singleton world change
    BASE_EDIT = "base_edit"  # an older commands/ without it: rewrite the base's place, keep the changes


_PLACE_RULES = {
    PlaceRule.WORLD: """- A PLACE or setting ("take me to the desert", "make it a desert", "go to Mars", "switch to a
  neon city") is ONE add of kind world: label = the place in 1 to 3 words, phrase = a short description of that
  place. It replaces the old world and KEEPS every other change: a red car added before stays in the desert.
  Known places, use exactly these labels when one fits: {places}.""",
    PlaceRule.BASE_EDIT: """- A PLACE or setting ("take me to the desert", "make it a desert") is a base edit: use edit_base
  with the place of the base sentence rewritten. It KEEPS every change: a red car added before stays.""",
}

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
- Write phrases as concrete visual descriptions, under 12 words, for example "a lion standing in the grass".
{place_rule}
- A BASE EDIT turns something the base sentence names into something else and KEEPS every change ("turn the
  tree into a lighthouse"): use edit_base. The engine keeps the changes; never copy them yourself.
- A NEW SCENE only when the user asks for one in so many words ("new scene of a beach", "start over with a
  beach"): use new_scene, which clears every change.
- edit_base and new_scene come before any other operation of the sentence.
- To remove or weaken something, use the id of the matching change. "a little" means strength 0.3, "a lot" or
  "very" means 1.3.
- If the sentence is vague, not about the picture, or asks to undo or reset, reply with an empty operations list.

Set "blocked" to true, with no operations, if the sentence asks for sexual, violent (gore, killing, injury) or
hateful content. Otherwise "blocked" is false.

The user message is JSON: {{"scene": <the scene before the sentence>, "prompt": <its current video prompt>,
"sentence": <what the user said>}}.

The rules for "details" follow. Where they speak of the scene or the description, that is the NEW scene, after
your operations. Whatever reply format they name, you reply with the one JSON object above.
"""

_KIND_HELP = {"world": "world (the place: desert, neon city, Mars, ...)", "season": "season (winter, autumn, ...)",
              "weather": "weather (rain, fog, sunny, ...)", "time": "time (night, day, sunset, ...)",
              "extra": "extra (any object, animal, person or effect)"}


def place_rule() -> PlaceRule:
    from commands.state import KINDS
    return PlaceRule.WORLD if "world" in KINDS else PlaceRule.BASE_EDIT


def _known_worlds() -> dict:
    """commands.world.WORLDS (normalized alias -> {"label", "kind", "phrase"}), or {} without PR #88."""
    try:
        from commands.world import WORLDS
        return WORLDS
    except ImportError:
        return {}


def operations_instructions() -> str:
    """The operations part of the system prompt, for the kinds this commands/ has."""
    from commands.state import KINDS
    rule = place_rule()
    places = ", ".join(sorted({w["label"] for w in _known_worlds().values()})) or "none"
    singletons = [k for k in KINDS if k != "extra"]
    return _OPERATIONS.format(kinds_help=", ".join(_KIND_HELP.get(k, k) for k in KINDS), kinds="|".join(KINDS),
                              singletons=", ".join(singletons[:-1]) + " or " + singletons[-1],
                              place_rule=_PLACE_RULES[rule].format(places=places))


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
        self.system_prompt = f"{operations_instructions()}\n{spec.details_instructions}"
        self._chat = chat
        self._file = cache_file
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], Entry] = cache_file.load() if cache_file is not None else {}
        self._pinned: dict[tuple[str, str], Entry] = dict(pinned or {})  # checked before the cache
        self.pinned_hits = 0
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="final-reader")
        self.calls = 0
        self.cache_hits = 0
        self.blocked = 0
        self.timeouts = 0
        self.late_dropped = 0
        self.rejects = 0
        self.details_rejected = 0
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
            return self._from_cache(key, entry, snapshot)
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
        return entry.result()

    def stats(self) -> dict:
        with self._lock:
            return {"state": self.state.value, "reason": self.reason or None, "prompt_spec": self.spec.name,
                    "calls": self.calls, "cache_hits": self.cache_hits,
                    "pinned": len(self._pinned), "pinned_hits": self.pinned_hits, "blocked": self.blocked,
                    "timeouts": self.timeouts, "late_dropped": self.late_dropped, "rejects": self.rejects,
                    "details_rejected": self.details_rejected, "errors": self.errors,
                    "p50_ms": round(statistics.median(self._recent_ms), 1) if self._recent_ms else None,
                    "last_ms": None if self.last_ms is None else round(self.last_ms, 1),
                    "cache_size": len(self._cache), "timeout_s": self.timeout_s,
                    "cache_file": None if self._file is None else str(self._file.path)}

    # --- internals -------------------------------------------------------------------------

    def _late(self) -> None:
        with self._lock:
            self.late_dropped += 1
        log.info("final reader: a late answer was dropped")

    def _from_cache(self, key, entry: Entry, snapshot: dict) -> dict | None:
        """A cached SONNET or PINNED result's operations name change ids; check they still fit this snapshot."""
        if entry.outcome not in (Outcome.SONNET, Outcome.PINNED):
            return entry.result()
        if all(validate_operation(dict(op), snapshot) == dict(op) for op in entry.operations):
            return entry.result()
        log.info("final reader: the cached operations for %r no longer fit the scene; dictionary path", _short(key[1]))
        return None

    def _call(self, text: str, snapshot: dict) -> Entry | None:
        """Worker thread: one Sonnet call, checked. None (a timeout or an error) or an Entry. Never raises."""
        from commands.composer import compose
        t0 = time.perf_counter()
        user = json.dumps({"scene": _scene_for_model(snapshot), "prompt": compose(snapshot), "sentence": text})
        try:
            reply_text = self._chat.complete(self.system_prompt, [{"role": "user", "content": user}])
            reply = json.loads(FENCE.sub("", reply_text))
            error = None
        except Exception as exc:  # the HTTP call or the JSON
            reply, error = None, exc
        ms = (time.perf_counter() - t0) * 1e3
        with self._lock:
            self.last_ms = ms
            self._recent_ms.append(ms)
        if error is not None:
            with self._lock:
                self.errors += 1
            log.warning("final reader: call failed for %r after %.0f ms: %s", _short(text), ms, repr(error)[:200])
            return Entry(Outcome.FALLBACK)
        entry, why = self._check(reply, snapshot)
        if entry.outcome is Outcome.FALLBACK:
            with self._lock:
                self.rejects += 1
            log.info("final reader: rejected the reply for %r (%s, %.0f ms): %s", _short(text), why, ms,
                     _short(json.dumps(reply)))
        elif entry.outcome is Outcome.BLOCKED:
            log.info("final reader: the model blocked %r (%.0f ms)", _short(text), ms)
        else:
            log.info("final reader: %r -> %d operations, prompt %r (%.0f ms)", _short(text), len(entry.operations),
                     _short(entry.prompt), ms)
        return entry

    def _check(self, reply, snapshot: dict) -> tuple[Entry, str]:
        if not isinstance(reply, dict):
            return Entry(Outcome.FALLBACK), "not an object"
        if reply.get("blocked") is True:
            return Entry(Outcome.BLOCKED), "blocked"
        ops = reply.get("operations")
        if not isinstance(ops, list) or not 1 <= len(ops) <= MAX_OPERATIONS:
            return Entry(Outcome.FALLBACK), "no operations"
        try:
            clean = expand_operations(ops, snapshot)
            after = scene_after(snapshot, clean) if clean is not None else None
        except (ValueError, TypeError, KeyError) as exc:
            return Entry(Outcome.FALLBACK), f"operations do not apply: {exc!r}"
        if clean is None:
            return Entry(Outcome.FALLBACK), "an invalid operation (undo and reset stay with the dictionary)"
        from commands.composer import compose
        try:
            canonical = compose(after)  # with PR #88 a world the base cannot take raises ValueError
        except (ValueError, TypeError, KeyError) as exc:
            return Entry(Outcome.FALLBACK), f"the new scene does not compose: {exc!r}"
        details = reply.get("details")
        try:  # only {"details"}: commands.rewriter.validate_rewrite rejects any other key
            prompt = self.spec.validate(canonical, {"details": details if isinstance(details, str) else ""})
        except Exception as exc:
            prompt = None
            log.info("final reader: validate raised %r", exc)
        if not isinstance(prompt, str) or not prompt.strip():
            # The operations are still the model's reading of the sentence; only its wording is dropped.
            with self._lock:
                self.details_rejected += 1
            log.info("final reader: details rejected, using the composed prompt: %s", _short(details))
            prompt = canonical
        return Entry(Outcome.SONNET, tuple(clean), prompt), "ok"


BATCH_LIMIT = 16  # commands.state: a batch holds 1 to 16 operations


def expand_operations(ops: list, snapshot: dict) -> list[dict] | None:
    """The model's operations as operations of commands/state.py, or None if one is invalid.

    {"op": "edit_base", "base": B} becomes new_scene B followed by one add per change the scene has at that
    point (label, kind, phrase and strength copied), so a base edit keeps the changes and, applied as one
    batch, is undone in one step. commands/state.py has no operation that rewrites only the base (a world
    change replaces the place, not an object of the base such as the tree), so this expansion stays.
    An operation naming a change id after a base edit or a new scene is invalid (those ids are gone), and undo
    and reset stay with the dictionary.
    """
    out: list[dict] = []
    rebuilt = False  # an edit_base or new_scene came first: the old ids no longer exist
    for op in ops:
        if isinstance(op, dict) and op.get("op") == "edit_base":
            clean = validate_operation({"op": "new_scene", "base": op.get("base")}, snapshot)
            if clean is None:
                return None
            current = scene_after(snapshot, out)["changes"]
            out.append(clean)
            out += [{"op": "add", "label": c["label"], "kind": c["kind"], "phrase": c["phrase"],
                     "strength": c["strength"]} for c in current]
            rebuilt = True
            continue
        clean = validate_operation(op, snapshot)
        if clean is None or clean["op"] in ("undo", "reset") or (rebuilt and "id" in clean):
            return None
        rebuilt = rebuilt or clean["op"] == "new_scene"
        out.append(canonical_world(clean))
    return out if len(out) <= BATCH_LIMIT else None


def canonical_world(op: dict) -> dict:
    """A world add for a place commands.world knows ("desert", "the city") gets its label and tested phrase,
    the same change the keyword table makes for that place. Any other operation is returned as it is."""
    if op.get("op") != "add" or op.get("kind") != "world":
        return op
    try:
        from commands.dictionary import normalize
    except ImportError:
        return op
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

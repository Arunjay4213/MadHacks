"""Named people the video model cannot draw (it has no real faces): a name is turned into a visual description.

    "add jensen huang" -> Sonnet reads "add a middle-aged man with short swept-back grey-black hair and glasses,
    wearing his signature black leather jacket over a black t-shirt", and the engine sets the add itself:
    label "jensen huang" (the user's words, shown on the chip, so "remove jensen huang" and undo find it) and the
    phrase PERSON.description + ", a large figure standing <spot>, facing the camera".

The spot is where the sentence puts the person ("next to the car"), else the first free one of SPOTS: an object
whose phrase already says "left" holds the left foreground (the lion), and so on.

Only adds are rewritten (Intent.ADD). Any other sentence naming the person ("remove jensen huang") keeps its words,
so the label of the chip matches it.
"""

import dataclasses
import enum
import re


@dataclasses.dataclass(frozen=True)
class Person:
    label: str                 # the chip label: the user's words
    aliases: tuple[str, ...]   # normalized, longest first when matching
    description: str


PEOPLE = (
    Person("jensen huang", ("jensen huang", "the nvidia ceo", "nvidia ceo", "jensen"),
           "a middle-aged man with short swept-back grey-black hair and glasses, wearing his signature black "
           "leather jacket over a black t-shirt"),
)
# (spot, the word an existing object's phrase has when it holds that spot)
SPOTS = (("in the foreground at the left of the frame", "left"),
         ("in the foreground at the right of the frame", "right"),
         ("in the middle ground at the centre of the frame", "centre"))
_ADD = re.compile(r"^(?:please\s+)?(?:add|put|place|bring|show|insert|include|drop)\b")
_RELATIVE = re.compile(r"\b(?P<spot>(?:next to|beside|near|in front of|behind|on top of) (?:the|a|an) [a-z ]+?)\s*$")


class Intent(enum.Enum):
    ADD = "add"      # the sentence adds the person: rewritten for the model, the add is the engine's
    OTHER = "other"  # it names the person another way (remove, move, ...): the words stay
    NONE = "none"    # no known person


@dataclasses.dataclass(frozen=True)
class Mention:
    intent: Intent
    person: Person | None = None
    alias: str | None = None


def _normalize(text: str) -> str:
    from commands.dictionary import normalize
    return normalize(text)


def _alias_rx(alias: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(alias)}\b")


def mention(text: str) -> Mention:
    t = _normalize(text)
    for person in PEOPLE:
        for alias in sorted(person.aliases, key=len, reverse=True):
            if _alias_rx(alias).search(t):
                return Mention(Intent.ADD if _ADD.match(t) else Intent.OTHER, person, alias)
    return Mention(Intent.NONE)


def for_model(text: str) -> str:
    """The sentence as the model reads it: an add of a known person names the description instead."""
    m = mention(text)
    if m.intent is not Intent.ADD:
        return text
    return _alias_rx(m.alias).sub(m.person.description, _normalize(text), count=1)


def spot(text: str, snapshot: dict) -> str:
    """Where the person stands: the sentence's own place ("next to the car"), else the first free SPOT."""
    m = _RELATIVE.search(_normalize(text))
    if m is not None:
        return m.group("spot")
    taken = " ".join(c["phrase"].lower() for c in snapshot["changes"] if c["kind"] == "extra" and c["strength"] > 0)
    return next((s for s, word in SPOTS if not re.search(rf"\b{word}\b", taken)), SPOTS[-1][0])


def person_add(text: str, snapshot: dict) -> dict | None:
    """The engine's add for a sentence of Intent.ADD, else None."""
    m = mention(text)
    if m.intent is not Intent.ADD:
        return None
    phrase = f"{m.person.description}, a large figure standing {spot(text, snapshot)}, facing the camera"
    return {"op": "add", "label": m.person.label, "kind": "extra", "phrase": phrase, "strength": 1.0}


_REMOVE = re.compile(r"^(?:please\s+)?(?:remove|delete|erase|take away|take out|get rid of|lose|drop)\b")


def person_remove(text: str, snapshot: dict) -> dict | None:
    """The remove of the person's chip for "remove jensen" (any alias) when the person is in the scene, else None.
    The keyword table knows only the chip label ("jensen huang"), so without this a short alias is unmatched when
    the model is unavailable."""
    m = mention(text)
    if m.intent is not Intent.OTHER or not _REMOVE.match(_normalize(text)):
        return None
    chip = next((c for c in snapshot["changes"] if c["label"] == m.person.label), None)
    return None if chip is None else {"op": "remove", "id": chip["id"]}


def with_person(text: str, ops, snapshot: dict):
    """The model's operations with its add of the person replaced by the engine's (the first add of kind extra),
    or the engine's add appended when the model gave none. Unchanged for a sentence that adds no known person."""
    add = person_add(text, snapshot)
    if add is None or not isinstance(ops, list):
        return ops
    i = next((n for n, op in enumerate(ops) if isinstance(op, dict) and op.get("op") == "add"
              and op.get("kind") == "extra"), None)
    return [*ops[:i], add, *ops[i + 1:]] if i is not None else [*ops, add]

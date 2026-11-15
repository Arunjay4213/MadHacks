"""Named people the video model cannot draw (it has no real faces): a name is turned into a visual description.

    "add jensen huang" -> Sonnet reads "add a middle-aged man with short swept-back grey-black hair and glasses,
    wearing his signature black leather jacket over a black t-shirt" and is told person_chip "jensen huang", so the
    chip keeps the user's words ("remove jensen huang" and undo find it). When the model gives nothing usable on a
    composed scene, the engine sets the add itself (person_add): label "jensen huang" and the phrase
    PERSON.description + ", a large figure standing <spot>, facing the camera".

The spot is where the sentence puts the person ("next to the car"), else the first free one of SPOTS: an object
whose phrase already says "left" holds the left foreground (the lion), and so on.

A sentence that is only the name ("jensen huang", "i want jensen") is an add too: a bare noun phrase means "add it".
Only adds are rewritten (Intent.ADD). Any other sentence naming the person ("remove jensen huang") keeps its words,
so the label of the chip matches it.

whole_sentence(text, snapshot) is the engine's fast path (engine/final_reader.py FinalReader.lookup): the add or
remove when the sentence is nothing but that ("jensen", "add jensen huang next to the car", "remove jensen"), so a
composed scene needs no Sonnet call for it.
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
    # Speech-to-text spellings of his name ("jensen wong", "jenson huang") are aliases too, so a spoken add still finds him.
    Person("jensen huang", ("jensen huang", "jensen wong", "jensen hwang", "jenson huang", "jensen hwong", "the nvidia ceo",
                            "nvidia ceo", "jensen"),
           "a middle-aged man with short swept-back grey-black hair and glasses, wearing his signature black "
           "leather jacket over a black t-shirt"),
)
# (spot, the word an existing object's phrase has when it holds that spot)
SPOTS = (("in the foreground at the left of the frame", "left"),
         ("in the foreground at the right of the frame", "right"),
         ("in the middle ground at the centre of the frame", "centre"))
_ADD = re.compile(r"^(?:please\s+)?(?:add|put|place|bring|show|insert|include|drop)\b")
# A place in the sentence ("next to the big red car"): at most four words after the article, none of them a word
# that starts another edit. So in "add jensen next to the car and make it winter" (or ", delete the dog": the comma
# is gone after normalize) the spot is "next to the car" and the rest stays in the sentence, where whole_sentence
# sees it and leaves the sentence to the model (Greptile, #273). A doubtful sentence goes to the model: that costs
# one call, a wrong shortcut drops the user's edit.
_CLAUSE_WORDS = ("and|then|also|plus|but|while|with|make|turn|change|set|switch|add|put|place|bring|insert|include|"
                 "remove|delete|erase|get|take|lose|drop|stop|no|without|undo|reset|clear|go|show|give|let|"
                 "don't|dont|never|more|less|so")
_RELATIVE = re.compile(r"\b(?P<spot>(?:next to|beside|near|in front of|behind|on top of) (?:the|a|an)"
                       rf"(?: (?!(?:{_CLAUSE_WORDS})\b)[a-z']+){{1,4}})\b")


# Words a bare mention may have besides the name: "i want jensen", "show me jensen huang please".
_BARE_WORDS = frozenset("the a an please now i i'd id we want would like to see me show give us let's lets also "
                        "here comes".split())
# Words an add may have besides its verb, the name and its spot, for the add to be the whole sentence.
_ADD_WORDS = _BARE_WORDS | {"add", "put", "place", "bring", "insert", "include", "drop", "in", "into", "scene",
                            "picture", "frame", "video", "too"}


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


def _rest(t: str, alias: str) -> list[str]:
    """The words of normalized `t` without the first match of `alias`."""
    return _alias_rx(alias).sub(" ", t, count=1).split()


def _bare(t: str, alias: str) -> bool:
    """True when normalized `t` is only the name, with filler words ("jensen huang", "i want jensen")."""
    return all(w in _BARE_WORDS for w in _rest(t, alias))


def mention(text: str) -> Mention:
    t = _normalize(text)
    for person in PEOPLE:
        for alias in sorted(person.aliases, key=len, reverse=True):
            if _alias_rx(alias).search(t):
                return Mention(Intent.ADD if _ADD.match(t) or _bare(t, alias) else Intent.OTHER, person, alias)
    return Mention(Intent.NONE)


def for_model(text: str) -> str:
    """The sentence as the model reads it: an add of a known person names the description instead ("add <the
    description>" for a bare name)."""
    m = mention(text)
    if m.intent is not Intent.ADD:
        return text
    t = _normalize(text)
    if not _ADD.match(t):
        return f"add {m.person.description}"
    return _alias_rx(m.alias).sub(m.person.description, t, count=1)


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


_FILLER = re.compile(r"\b(?:the|please|now|from the scene|from the picture)\b")
_REMOVE = re.compile(r"^(?:please\s+)?(?:remove|delete|erase|take away|take out|get rid of|lose|drop)\b")


def person_remove(text: str, snapshot: dict) -> dict | None:
    """The remove of the person's chip for "remove jensen" (any alias) when the person is in the scene, else None.
    The keyword table knows only the chip label ("jensen huang"), so without this a short alias is unmatched when
    the model is unavailable."""
    m = mention(text)
    if m.intent is not Intent.OTHER:
        return None
    # Only a sentence that is nothing but this removal: "remove the dog near jensen" removes the dog, and a compound
    # ("remove jensen huang and add a dog") goes to the keyword table whole, which reads every clause.
    target = _REMOVE.sub("", _normalize(text), count=1)
    if not _REMOVE.match(_normalize(text)) or " ".join(_FILLER.sub(" ", target).split()) != " ".join(_FILLER.sub(" ", m.alias).split()):
        return None
    chip = next((c for c in snapshot["changes"] if c["label"] == m.person.label), None)
    return None if chip is None else {"op": "remove", "id": chip["id"]}


def whole_sentence(text: str, snapshot: dict) -> dict | None:
    """The engine's add or remove of a known person when the sentence is nothing but that (the module docstring),
    else None: "add jensen huang and a dog" is not, so the model or the keyword table reads every part."""
    m = mention(text)
    if m.intent is Intent.ADD:
        from commands.reader import split_commands
        if len(split_commands(text)) != 1:  # "add jensen huang, then make it winter": more than the add
            return None
        t = _normalize(text)
        spot_m = _RELATIVE.search(t)
        if spot_m is not None:  # every word besides the spot must still be part of the add
            t = t[:spot_m.start()] + " " + t[spot_m.end():]
        if not all(w in _ADD_WORDS for w in _rest(t, m.alias)):
            return None
        return person_add(text, snapshot)
    if m.intent is Intent.OTHER:
        return person_remove(text, snapshot)
    return None

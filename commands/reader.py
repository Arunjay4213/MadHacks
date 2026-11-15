"""Fast reader: rules pick the operation and pace, the embedded dictionary picks the "what" (PLAN.md section 1).

Vocabulary lives in keywords.json (loaded by dictionary.py). Unknown objects after an add verb pass straight
through into the prompt; negation and transforms stay unmatched for the slow reader.
"""

import difflib
import re
import unicodedata

from .dictionary import KEYWORD_RX, KEYWORDS, normalize
from .state import MAX_BATCH
from .world import WORLDS, world_op

__all__ = ["KEYWORDS", "MAX_CLAUSES", "RAMP", "is_blocked", "ramp_for", "read", "remove_target", "resolve", "split_commands", "spot",
           "style_fallback"]

MAX_CLAUSES = MAX_BATCH  # callers reject a sentence that splits into more

RAMP = {"default": 1, "slow": 12, "sudden": 1, "provisional": 4}  # anchor fires at full target

_STRENGTH_LOW = 0.3
_STRENGTH_HIGH = 1.3  # "a lot": heavy wording in the composer
_STEP = 0.3


def _rx(words):
    return re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")\b")


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
# Pass-through: "add a lion" -> an extra change named by the spoken noun phrase.
_ADD_RX = re.compile(r"\b(?:add|put|place|bring|include|build|draw|give me|i want|i'd like)\s+(?:(a|an|some|the|another)\s+)?(.+)")
_PREP_RX = re.compile(r"\s+\b(?:in|on|at|near|by|over|under|to|with|beside|behind|across|next to|around|above|below)\b\s+")
_FILLER_TAIL_RX = re.compile(r"(?:\s+\b(?:please|too|as well|now|here|there|for me|in the scene|to the scene)\b)+$")
_NOT_AN_OBJECT = {"a", "an", "the", "another", "it", "that", "this", "them", "these", "those", "something", "anything", "one", "please", "more", "some"}
_MOTION_FORMS = {"drive": "driving", "move": "moving", "walk": "walking", "run": "running",
                 "fly": "flying", "swim": "swimming", "reverse": "reversing", "stop": "stopped"}
_MOTION_FORMS.update({verb + "s": participle for verb, participle in list(_MOTION_FORMS.items()) if verb != "fly"})
_MOTION_FORMS["flies"] = "flying"
_MOTION_RX = re.compile(r"\b(?:driv(?:es?|ing)|mov(?:es?|ing)|walk(?:s|ing)?|run(?:s|ning)?|fly(?:ing)?|flies|swim(?:s|ming)?|revers(?:es?|ing)|stop(?:s|ped)?|parked|standing|sitting)\b")


def _motion_tail(text):
    tail = re.sub(r"^(?:is|are|should be|must be)\s+", "", text.strip())
    if re.match(r"^(?:near|beside|by|behind|on|at|in front of)\b", tail):
        motion = _MOTION_RX.search(tail)
        if motion:
            return tail[motion.start():]
    return tail


def _additional(phrase, entry, keyword, snapshot):
    if not any(KEYWORDS[m.group(0)]["label"] == entry["label"]
               for m in KEYWORD_RX.finditer(normalize(snapshot.get("base", "")))):
        return phrase
    # Distinguish a requested addition from an object already present in the base scene.
    if phrase.startswith(("a ", "an ")):
        phrase = "a second " + phrase.split(" ", 1)[1]
    else:
        phrase = "additional " + phrase
    position = "alongside" if entry["label"] == "car" else "beside"
    return f"{phrase}, {position} the existing {keyword}"

_POSITION_RX = re.compile(r"^(?:please\s+)?(?:move|position|reposition)\b")
_MAX_LABEL_WORDS = 5
_ING_NOUNS = {"building", "buildings", "painting", "paintings", "ceiling", "evening", "morning", "string", "wedding", "clothing", "swing", "swings", "ring", "rings", "king", "kings", "wing", "wings", "thing", "things"}
# Words that never pick a change by themselves: articles, pronouns, colours, sizes and generic nouns.
_STOP = {
    "a", "an", "the", "of", "with", "on", "in", "and", "to", "it", "its", "that", "this", "those", "these", "them", "then",
    "some", "all", "one", "please", "thing", "things", "stuff", "bit", "there", "here", "now", "too", "very", "really",
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black", "grey", "gray", "brown", "golden", "gold",
    "silver", "big", "small", "little", "large", "tall", "short", "huge", "tiny", "soft", "bright", "dark", "strong", "thick",
    "calm", "harsh", "clear", "steady", "fresh", "lush", "colorful", "colourful", "glowing", "wooden", "stone",
    "whole", "entire", "everything", "something", "anything", "sky", "air", "ground", "weather", "atmosphere",
    "mood", "vibe", "feel", "feeling", "look", "effect", "lot", "lots", "scene", "picture", "image", "view", "place", "world",
}


def _norm(text: str) -> str:
    return normalize(text)


def _pace(t: str):
    return "slow" if _SLOW_RX.search(t) else "sudden" if _SUDDEN_RX.search(t) else None


def ramp_for(text: str) -> int:
    """The ramp of a sentence from its pace words: RAMP["slow"] for "slowly", else RAMP["default"]."""
    return RAMP.get(_pace(normalize(text)) or "default", RAMP["default"])


def strength_for(text, current=0.0):
    """Shared intensity rules for dictionary, vector and style paths."""
    if _HIGH_RX.search(text):
        return _STRENGTH_HIGH
    if current and _MORE_RX.search(text):
        return min(_STRENGTH_HIGH, round(current + _STEP, 2))
    if current and _LESS_RX.search(text):
        return max(_STRENGTH_LOW / 3, round(current - _STEP, 2))
    if _LOW_RX.search(text):
        return _STRENGTH_LOW
    return 1.0


def _pick(t: str):
    """The keyword match to act on: the first object noun if there is one, else the first match.

    Adjective-like aliases ("bright", "dark", "snowy") name a time or weather change on their own,
    but in "add a bright red car" or "add dark clouds" the object is what the speaker means.
    Words after the first preposition are a location: in "make it snow near the river" the change is snow,
    not the river. Without an add verb the phrases between prepositions are searched in order and the first one
    that names a change wins: "turn it to night near the river" is night ("turn it" names nothing, "night" does).
    """
    add = _ADD_RX.search(t)
    if add:
        end = add.start(2) + len(_PREP_RX.split(add.group(2), maxsplit=1)[0])
        matches = list(KEYWORD_RX.finditer(t, add.start(2), end))
    else:
        cuts = [0] + [i for p in _PREP_RX.finditer(t) for i in p.span()] + [len(t)]
        matches = []
        for start, end in zip(cuts[::2], cuts[1::2]):
            matches = list(KEYWORD_RX.finditer(t, start, end))
            if matches:
                break
    for m in matches:
        if KEYWORDS[m.group(0)]["kind"] == "extra":
            return m
    return matches[0] if matches else None


def resolve(query: str, snapshot: dict):
    """Change id for a spoken name. Most recent wins.

    A dictionary noun in the query decides alone: if that change is not in the scene the answer is None,
    never a guess from adjectives or phrase words ("the red car" must not remove the red autumn leaves).
    Without one: exact id, free-text label, phrase word (stopwords and colours excluded), then a close
    match on present labels for words of 4+ letters. Every step reads only the words before the first
    preposition.
    """
    changes = snapshot.get("changes", [])
    if not changes:
        return None
    raw = query.strip()
    for c in reversed(changes):
        if c["id"] == raw:
            return c["id"]
    # Only the words before the first preposition name the target. The rest is a location
    # ("remove the car near the meadow"), so an absent car removes nothing rather than the meadow.
    full = _norm(query)
    q = _PREP_RX.split(full, maxsplit=1)[0]
    # Present labels as whole phrases, longest first: "dark knight" before the "dark" alias of night.
    aliases = list(KEYWORD_RX.finditer(q))
    named = []
    for change in changes:
        label = _norm(change["label"])
        # A label with its own preposition ("city on mars") is matched in the whole query, but it must
        # start before the cut: in "remove the car near the city on mars" it is still only the location.
        for match in re.finditer(r"\b" + re.escape(label) + r"\b", full if _PREP_RX.search(label) else q):
            if match.start() >= len(q):
                continue
            # A named world is the target even where its name is also an alias ("forest" -> trees).
            if change["kind"] == "world" or not any(a.start() <= match.start() and match.end() <= a.end()
                       and KEYWORDS[a.group()]["label"] != change["label"] for a in aliases):
                named.append(change)
                break
    if named:
        longest = max(len(_norm(c["label"])) for c in named)
        return [c for c in named if len(_norm(c["label"])) == longest][-1]["id"]
    known = {KEYWORDS[m.group(0)]["label"] for m in KEYWORD_RX.finditer(q)}
    known |= {entry["label"] for alias, entry in WORLDS.items()
              if re.search(r"\b" + re.escape(alias) + r"\b", q)}
    if known:
        if _names_unknown_object(q) and not any(c["label"] in known and c["kind"] == "world" for c in changes):  # "the dark knight": the alias is an adjective on an absent object
            return None
        hits = [c for c in changes if c["label"] in known]
        return hits[-1]["id"] if hits else None
    words = {w for w in q.split() if w not in _STOP and len(w) >= 3}
    hits = [c for c in changes if words & (set(_norm(c["phrase"]).split()) - _STOP)]
    if hits:
        return hits[-1]["id"]
    labels = [_norm(c["label"]) for c in changes]
    for w in sorted(words):
        if len(w) < 4:
            continue
        close = difflib.get_close_matches(w, labels, n=1, cutoff=0.8)
        if close:
            return [c for c in changes if _norm(c["label"]) == close[0]][-1]["id"]
    return None


def _find(snapshot, label):
    hits = [c for c in snapshot.get("changes", []) if c["label"] == label]
    return hits[-1] if hits else None


def _pass_through(t: str):
    """An add op for an object the dictionary does not know, or None."""
    m = _ADD_RX.search(t)
    if not m:
        return None
    article, rest = m.group(1), m.group(2)
    rest = _FILLER_TAIL_RX.sub("", _SLOW_RX.sub(" ", _SUDDEN_RX.sub(" ", rest)))
    rest = " ".join(rest.split())
    label = _PREP_RX.split(rest, maxsplit=1)[0].strip()
    words = label.split()
    if not KEYWORD_RX.search(label):  # never split a known alias ("early morning")
        for i, w in enumerate(words[1:], 1):  # "dragon flying over..." -> "dragon"
            if w.endswith("ing") and len(w) > 4 and w not in _ING_NOUNS:
                label = " ".join(words[:i])
                break
    if not label or label in _NOT_AN_OBJECT or len(label.split()) > _MAX_LABEL_WORDS:
        return None
    if not set(label.split()) - _STOP:  # "the whole thing", "something big": no real object named
        return None
    phrase = f"{article} {rest}" if article in ("a", "an", "some") else rest
    return {"op": "add", "label": label, "kind": "extra", "phrase": phrase, "strength": 1.0}


def _names_unknown_object(label: str) -> bool:
    """True when an added object is a real noun the dictionary does not know ("bright unicorn", "lion near the river").

    A known concrete object in it wins ("bright red car" -> car). Otherwise drop alias, pace, strength and
    stopwords: if a real word is left it is the object ("unicorn"); if not, the alias is the meaning ("a little rain").
    """
    if any(KEYWORDS[m.group(0)]["kind"] == "extra" for m in KEYWORD_RX.finditer(label)):
        return False
    rest = KEYWORD_RX.sub(" ", label)
    for rx in (_SLOW_RX, _SUDDEN_RX, _LOW_RX, _HIGH_RX, _MORE_RX, _LESS_RX):
        rest = rx.sub(" ", rest)
    return any(w not in _STOP and w not in _NOT_AN_OBJECT for w in rest.split())


def read(text: str, snapshot: dict) -> dict:
    """{"op": operation dict or None, "keyword": spoken key word or None, "pace": "slow" | "sudden" | None}."""
    t = _norm(text)
    pace = _pace(t)
    out = lambda op, keyword=None: {"op": op, "keyword": keyword, "pace": pace}  # noqa: E731
    setting = world_op(t)  # before the transform check: "turn it into a beach" is a world change (world_op blocks negation)
    if setting:
        return out(setting, setting["label"])
    if _NEGATION_RX.search(t) or _TRANSFORM_RX.search(t):
        return out(None)
    if _POSITION_RX.search(t):
        return out(None)
    m = _UNDO_RX.search(t)
    if m:
        tail = t[m.end():].strip()
        if m.group(0) == "undo" and tail and tail not in ("it", "that", "please", "that change", "the last change"):
            target = resolve(tail, snapshot)
            return out({"op": "remove", "id": target} if target else None, m.group(0))
        return out({"op": "undo"}, m.group(0))
    m = _RESET_RX.search(t)
    if m:
        return out({"op": "reset"}, m.group(0))
    m = _SCENE_RX.search(t)
    if m and m.group(1).strip(" .'"):
        return out({"op": "new_scene", "base": m.group(1).strip(" .'")})
    kw = _pick(t)
    keyword = kw.group(0) if kw else None
    m = _REMOVE_RX.search(t)
    if m:
        target = resolve(t[m.end():] or t, snapshot)
        return out({"op": "remove", "id": target} if target else None, keyword)
    added = _pass_through(t)
    if added and _names_unknown_object(added["label"]):
        # The object itself is unknown ("a lion near the river"): known words in its location do not replace it.
        return out(added, added["label"])
    if not kw:
        m = _SHOW_RX.search(t)
        if m and m.group(1).strip(" .'"):
            return out({"op": "new_scene", "base": m.group(1).strip(" .'")})
        return out(None)
    entry = KEYWORDS[keyword]
    existing = _find(snapshot, entry["label"])
    current = existing["strength"] if existing else 0.0
    motion = _MOTION_RX.match(_motion_tail(t[kw.end():]))
    if (entry["kind"] == "extra" and motion and not _ADD_RX.search(t)
            and (kw.start() == 0 or _ACTION_RX.match(t) or re.match(r"^(?:now\s+)?the\b", t))):
        if existing is None:
            return out(None, keyword)
        old_phrase = normalize(existing["phrase"])
        noun = next((m for m in KEYWORD_RX.finditer(old_phrase)
                     if KEYWORDS[m.group(0)]["label"] == entry["label"]), None)
        old_motion = _MOTION_RX.search(old_phrase, noun.end() if noun else 0)
        subject = old_phrase[:old_motion.start()].strip() if old_motion else old_phrase
        subject = " ".join(_MOTION_RX.sub(" ", subject).split())
        action = _motion_tail(t[kw.end():])
        action = _MOTION_FORMS.get(motion.group(0), motion.group(0)) + action[motion.end():]
        if old_motion:
            tail = old_phrase[old_motion.end():].strip()
            location = _PREP_RX.search(" " + tail)
            split = max(0, location.start()) if location else len(tail)
            manner, placement = tail[:split], tail[split:].strip()
            manner = re.sub(r"\b(?:forwards?|backwards?|ahead|in reverse)\b", " ", manner)
            if re.search(r"\b(?:fast|slow|slowly|quickly|rapidly)\b", action):
                manner = re.sub(r"\b(?:fast|slow|slowly|quickly|rapidly)\b", " ", manner)
            if re.search(r"\b(?:towards?|away from)\b", action):
                manner = re.sub(r"\b(?:towards?|away from)\b.*$", "", manner)
            if _PREP_RX.search(" " + action):
                placement = ""
            action = " ".join((action + " " + manner + " " + placement).split())
        return out({"op": "add", **entry, "phrase": f"{subject} {action}",
                    "strength": existing["strength"]}, keyword)
    rest = t[: kw.start()] + " " + t[kw.end():]  # strength words outside the noun itself ("light house" is not "light")
    add_verb = _ADD_RX.search(t)
    bare = (entry["kind"] == "extra" and not _ACTION_RX.match(t)
            and not re.match(r"^(?:what|why|how|when|where|who|can|could|would|should|is|are|do|does)\b", t))
    description_match = add_verb or (_ADD_RX.search("add " + t) if bare else None)
    if entry["kind"] == "extra" and description_match:
        rest = t[:add_verb.start()] if add_verb else ""  # noun descriptions carry attributes, not strength
        if _MORE_RX.search(description_match.group(2)):
            rest += " more"
        elif _LESS_RX.search(description_match.group(2)):
            rest += " less"
    strength = strength_for(rest, current)
    if entry["kind"] == "extra":
        if description_match:
            article, description = description_match.groups()
            description = _FILLER_TAIL_RX.sub("", description).strip()
            # Bare aliases keep their stable dictionary phrase; explicit visual details survive.
            detail = KEYWORD_RX.sub(lambda m: " " if KEYWORDS[m.group()]["label"] == entry["label"] else m.group(), description)
            for rx in (_LOW_RX, _HIGH_RX, _MORE_RX, _LESS_RX):
                detail = rx.sub(" ", detail)
            if detail.strip() and detail.strip() not in _NOT_AN_OBJECT:
                phrase = f"{article} {description}" if article in ("a", "an", "some") else description
                if existing is None:
                    phrase = _additional(phrase, entry, keyword, snapshot)
                return out({"op": "add", **entry, "phrase": phrase, "strength": strength}, keyword)
    if existing:
        return out({"op": "set_strength", "id": existing["id"], "strength": strength}, keyword)
    return out({"op": "add", "label": entry["label"], "kind": entry["kind"], "phrase": _additional(entry["phrase"], entry, keyword, snapshot) if entry["kind"] == "extra" else entry["phrase"], "strength": strength}, keyword)


# A clause starts with an explicit action (optionally after "also/then/please/now" and a pace word).
_LEAD = r"^(?:(?:also|then|please|now|and)\s+)*(?:(?:slowly|gradually|gently|suddenly|instantly|quickly|immediately)\s+)?"
_ACTION_RX = re.compile(_LEAD + r"(?:add|put|place|bring|include|build|draw|give me|i want|i'd like|let's|lets|let it|remove|delete"
                        r"|get rid of|take away|drop|lose|no more|stop the|without|make|turn|change|set|switch|show me|take me to|go to|go back"
                        r"|undo|revert|reset|start over|start again|clear|new scene|never mind|nevermind|take that back"
                        r"|don't|dont|do not|never|a lot more|much more|more|less|a little more|a little less)\b")
# Verbs that may be repeated over a list of recognized nouns: "add a dog and a cat".
_VERB_PREFIX_RX = re.compile(_LEAD + r"(?:add|put|place|bring|include|build|give me|remove|delete|get rid of|take away|no more"
                             r"|make it|turn it to|turn it|turn to)\b")
_SOFT_SPLIT_RX = re.compile(r"\s+(?:and then|then|and|also)\s+(?=\S)")
# "add a dog without the rain" is two edits: add the dog, remove the rain. Only after an add verb,
# so a scene description ("take me to a beach without people") stays whole.
_ADD_START_RX = re.compile(_LEAD + r"(?:add|put|place|bring|include|build|draw|give me|i want|i'd like)\b")
_WITHOUT_SPLIT_RX = re.compile(r"\s+(?=without\s+\S)")
_ARTICLE_RX = re.compile(r"^(?:(?:a|an|the|some|another)\s+)")


def _is_alias_list_item(chunk: str) -> bool:
    """True if a chunk is nothing but one dictionary alias (after an article): 'a cat', 'the lighthouse', 'night'."""
    return bool(KEYWORD_RX.fullmatch(_ARTICLE_RX.sub("", chunk)))


def split_commands(text: str) -> list[str]:
    """Split one spoken sentence into normalized clauses, one operation each.

    Hard breaks: ; ! ? always. Soft breaks: commas, "and", "then", "also" only when an explicit action follows,
    so "black and white" or "rock and roll" stay inside their clause. A list of recognized nouns after one verb
    repeats that verb: "add a dog and a cat" -> ["add a dog", "add a cat"]. Callers reject > MAX_CLAUSES.
    """
    clauses: list[str] = []
    text = unicodedata.normalize("NFKC", text)  # full-width ；！？ and the ellipsis character become ASCII
    for segment in re.split(r"[;!?]+", text):
        pieces = []  # (soft connector before it, normalized text)
        # Commas and full stops are soft: "Make it... winter", "add a St. Bernard" stay one clause.
        for i, comma_part in enumerate(re.split(r"[,.]+", segment)):
            part = re.sub(r"^(?:and then|then|and)\s+", "", _norm(comma_part))  # ", then add ..." / ", and add ..."
            if not part:
                continue
            words = _SOFT_SPLIT_RX.split(part)
            connectors = _SOFT_SPLIT_RX.findall(part)
            pieces.append((None if i == 0 or not pieces else " ", words[0]))
            for conn, w in zip(connectors, words[1:]):
                pieces.append((" " + conn.strip() + " ", w))
        current: list[str] = []  # clause texts of this segment
        verb = None  # repeatable verb prefix of the clause that started the current list
        for conn, chunk in pieces:
            if not current:
                current.append(chunk)
            elif _ACTION_RX.match(chunk):
                current.append(chunk)
            elif verb and current[-1] == verb:  # "make it... winter": the verb had no object yet
                current[-1] = f"{current[-1]} {chunk}"
                continue
            elif verb and _is_alias_list_item(chunk):
                current.append(f"{verb} {chunk}")
                continue
            else:
                current[-1] = f"{current[-1]}{conn or ' '}{chunk}"
                continue
            m = _VERB_PREFIX_RX.match(chunk)
            verb = m.group(0).strip() if m else None
        for c in current:
            c = " ".join(c.split())
            if not c:
                continue
            clauses.extend(_WITHOUT_SPLIT_RX.split(c, maxsplit=1) if _ADD_START_RX.match(c) else [c])
    return clauses


def remove_target(text: str):
    """The words after a remove verb ("drop the dog" -> "dog"), or None if the text has no remove intent.
    Shared with the semantic fallback so a remove can never be read back as an add."""
    t = _norm(text)
    m = _REMOVE_RX.search(t)
    if not m:
        return None
    return t[m.end():].strip() or None


_STYLE_RX = re.compile(r"^(?:please\s+)?(?:make|turn)\s+(?:it|everything|the scene|the whole scene|the world|the picture)\s+"
                       r"(?:(?:look|feel|seem)\s+)?(?P<more>more\s+)?(?P<like>like\s+)?(?P<what>.+)$")
_MAX_STYLE_WORDS = 4
# Praise and comparison words describe no look; "make it nice" stays unmatched (red) instead of guessing.
_VAGUE = {"nice", "good", "better", "best", "cool", "great", "awesome", "amazing", "different", "interesting", "fun",
          "okay", "ok", "fine", "bad", "worse", "pretty", "beautiful", "work", "happen", "so", "like that", "that way"}


def style_fallback(text: str, snapshot: dict | None = None):
    """Last resort for a look or mood nothing else understood: "make it spooky" -> an extra change
    "the whole scene looks spooky". None for negation, transforms, dictionary words and long phrases.

    Strength words use the same strength_for as the dictionary and vector paths and never stay in the label:
    "make it a little spooky" adds "spooky" at 0.3, and "much more" or "less" on an existing "spooky" sets its strength."""
    t = _norm(text)
    if is_blocked(t) or KEYWORD_RX.search(t):
        return None
    # Pace words set the ramp (the service reads them with _pace), never the look or the strength.
    m = _STYLE_RX.match(" ".join(_SUDDEN_RX.sub(" ", _SLOW_RX.sub(" ", t)).split()))
    if not m:
        return None
    like, spoken = m.group("like"), _FILLER_TAIL_RX.sub("", m.group("what")).strip()
    if len(spoken.split()) > _MAX_STYLE_WORDS:
        return None
    what = spoken
    for rx in (_LOW_RX, _HIGH_RX, _MORE_RX, _LESS_RX):
        what = rx.sub(" ", what)
    what = " ".join(what.split())
    if not what or what in _NOT_AN_OBJECT or len(what.split()) > _MAX_STYLE_WORDS:
        return None
    if set(what.split()) <= _VAGUE | _STOP - {"black", "white", "red", "orange", "yellow", "green", "blue", "purple", "pink",
                                              "grey", "gray", "brown", "golden", "gold", "silver", "dark", "bright"}:
        return None
    strength_words = (m.group("more") or "") + spoken
    existing = _find(snapshot or {}, what)
    if existing:
        return {"op": "set_strength", "id": existing["id"], "strength": strength_for(strength_words, existing["strength"])}
    return {"op": "add", "label": what, "kind": "extra", "phrase": "the whole scene looks " + ("like " if like else "") + what,
            "strength": strength_for(strength_words)}


def is_blocked(text: str) -> bool:
    """True when read() left the text unmatched ON PURPOSE (negation or a transform it cannot express).
    A local fallback must not turn these into an add; the slow LLM can interpret transforms."""
    t = _norm(text)
    kw = _pick(t)
    motion_edit = kw and KEYWORDS[kw.group(0)]["kind"] == "extra" and _MOTION_RX.match(_motion_tail(t[kw.end():])) and not _ADD_RX.search(t)
    return bool(_NEGATION_RX.search(t) or _TRANSFORM_RX.search(t) or motion_edit or _POSITION_RX.search(t))


def spot(partial_text: str):
    """The KEYWORDS key in partial text, for early steering: whole aliases only, never prefixes or pass-through.
    None for undo/reset/remove/negation/transform/new-scene wording: never guessed."""
    t = _norm(partial_text)
    if is_blocked(t) or world_op(t):
        return None
    if any(rx.search(t) for rx in (_UNDO_RX, _RESET_RX, _REMOVE_RX, _NEGATION_RX, _TRANSFORM_RX, _SCENE_RX)):
        return None
    m = _pick(t)
    return m.group(0) if m else None

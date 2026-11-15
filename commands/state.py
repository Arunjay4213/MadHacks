"""Scene operations and bounded undo history; no model or network work.

A scene is in one of two named states (Origin, the scene's "origin" field):

- COMPOSED: a base sentence plus changes; commands.composer writes the prompt from them.
- FREE: a full prompt written by the engine's Sonnet final reader ("prompt"), with the chips it named as
  changes of kind "free" ({"id", "label", "kind": "free", "phrase": "", "strength": 1.0}). The base is kept as it
  was, unused while free, so every snapshot has the same shape.

{"op": "free_scene", "prompt": str, "chips": [str, ...]} makes the scene FREE (one undo step) from either state.
On a FREE scene only undo, reset, new_scene (which makes it COMPOSED again) and another free_scene apply; add,
remove and set_strength raise ValueError, so the dictionary can never edit a prompt it did not write.
"""

import enum
from copy import deepcopy


DEFAULT_BASE = "a wide shot of a meadow with a single oak tree"
KINDS = ("world", "season", "weather", "time", "extra")
WORLD_CLEARS = {"world", "season", "weather"}
MAX_BATCH = 16  # operations in one batch (one undo step); also the most clauses in one sentence
MAX_LABEL = 24  # characters of a chip label
FREE_KIND = "free"
MAX_FREE_CHIPS = 16
MAX_FREE_PROMPT_CHARS = 1900  # under umT5's 512 tokens at the worst 3.9 characters a token measured on Sonnet's prompts


class Origin(enum.Enum):
    """Who writes the scene's prompt."""
    COMPOSED = "composed"  # commands.composer, from the base and the changes
    FREE = "free"          # the engine's final reader wrote the whole prompt


FREE_ONLY = "a free scene takes only undo, reset, new_scene or another free scene"


def origin(snapshot: dict) -> Origin:
    """The Origin of a snapshot; a hand-built snapshot with no "origin" field is COMPOSED."""
    return Origin(snapshot.get("origin", Origin.COMPOSED.value))


def is_free(snapshot: dict) -> bool:
    return origin(snapshot) is Origin.FREE


def free_chips(chips) -> list[str]:
    """The chip labels of a free scene: 0 to MAX_FREE_CHIPS nonempty labels of at most MAX_LABEL characters,
    lower case, spaces collapsed, duplicates dropped (first kept). Raises ValueError otherwise."""
    if not isinstance(chips, list) or len(chips) > MAX_FREE_CHIPS:
        raise ValueError(f"chips must be a list of at most {MAX_FREE_CHIPS} labels")
    out = []
    for chip in chips:
        label = " ".join(_text(chip, "chip").lower().split())
        if len(label) > MAX_LABEL:
            raise ValueError(f"a chip label has at most {MAX_LABEL} characters: {label!r}")
        if label not in out:
            out.append(label)
    return out


def free_prompt(prompt) -> str:
    """A free scene's prompt: nonempty text, one paragraph, at most MAX_FREE_PROMPT_CHARS characters."""
    text = " ".join(_text(prompt, "prompt").split())
    if len(text) > MAX_FREE_PROMPT_CHARS:
        raise ValueError(f"a free prompt has at most {MAX_FREE_PROMPT_CHARS} characters")
    return text


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _strength(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1.3:
        raise ValueError("strength must be a finite number between 0 and 1.3")
    return value


class SceneState:
    def __init__(self, base=DEFAULT_BASE):
        self._initial_base = _text(base, "base")
        self._scene = self._composed(self._initial_base)
        self._history = []
        self._next_id = 1

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "SceneState":
        """A state holding the scene of `snapshot` (ids kept, no undo history), for trial readings on a copy."""
        state = cls(snapshot["base"])
        state._scene = {"base": snapshot["base"], "origin": origin(snapshot).value, "prompt": snapshot.get("prompt"),
                        "changes": deepcopy(snapshot["changes"])}
        ids = [int(c["id"][1:]) for c in snapshot["changes"] if str(c.get("id", ""))[1:].isdigit()]
        state._next_id = max(ids, default=0) + 1
        return state

    @staticmethod
    def _composed(base):
        return {"base": base, "origin": Origin.COMPOSED.value, "prompt": None, "changes": []}

    def snapshot(self):
        return {**deepcopy(self._scene), "can_undo": bool(self._history)}

    @property
    def origin(self) -> Origin:
        return Origin(self._scene["origin"])

    def apply(self, operation):
        """Apply one validated operation atomically. No-ops do not add history."""
        if not isinstance(operation, dict):
            raise ValueError("operation must be an object")
        op = operation.get("op")
        if op == "batch":
            operations = operation.get("operations")
            if (not isinstance(operations, list) or not 1 <= len(operations) <= MAX_BATCH
                    or any(not isinstance(item, dict) or item.get("op") == "batch" for item in operations)):
                raise ValueError(f"batch needs 1 to {MAX_BATCH} non-nested operations")
            trial = deepcopy(self)
            for item in operations:
                trial.apply(item)
            if trial._scene == self._scene:
                return False
            self._history.append(self._scene)
            del self._history[:-100]
            self._scene = trial._scene
            self._next_id = trial._next_id
            return True
        if op == "undo":
            if not self._history:
                return False
            self._scene = self._history.pop()
            return True

        next_scene = deepcopy(self._scene)
        changes = next_scene["changes"]
        new_id = False
        if self.origin is Origin.FREE and op in ("add", "remove", "set_strength"):
            raise ValueError(FREE_ONLY)
        if op == "free_scene":
            prompt, labels = free_prompt(operation.get("prompt")), free_chips(operation.get("chips"))
            # A chip that keeps its label keeps its id, so the page does not redraw it; every other chip is new.
            old = {c["label"]: c["id"] for c in changes}
            chips, next_id = [], self._next_id
            for label in labels:
                chip_id = old.get(label)
                if chip_id is None:
                    chip_id, next_id = f"c{next_id}", next_id + 1
                chips.append({"id": chip_id, "label": label, "kind": FREE_KIND, "phrase": "", "strength": 1.0})
            next_scene = {"base": next_scene["base"], "origin": Origin.FREE.value, "prompt": prompt, "changes": chips}
            if next_scene == self._scene:
                return False
            self._history.append(self._scene)
            del self._history[:-100]
            self._scene = next_scene
            self._next_id = next_id
            return True
        elif op == "add":
            label = _text(operation.get("label"), "label").lower()
            phrase = _text(operation.get("phrase"), "phrase")
            kind = operation.get("kind")
            if kind not in KINDS:
                raise ValueError("unknown change kind")
            strength = _strength(operation.get("strength", 1.0))
            existing = next((c for c in changes if c["label"].casefold() == label.casefold()), None)
            clears_more = kind == "world" and any(c["kind"] in WORLD_CLEARS - {"world"} for c in changes)
            if (existing and existing["kind"] == kind and existing["phrase"].casefold() == phrase.casefold()
                    and existing["strength"] == strength and not clears_more):  # a repeated world still clears a season
                return False
            item = dict(id=existing["id"] if existing else f"c{self._next_id}",
                        label=label, kind=kind, phrase=phrase, strength=strength)
            new_id = existing is None
            if kind != "extra":
                # A new world brings its own season and weather, so it also clears those; time and objects stay.
                cleared = WORLD_CLEARS if kind == "world" else {kind}
                changes[:] = [c for c in changes if c["kind"] not in cleared or c is existing]
            if existing:
                changes[changes.index(existing)] = item
            else:
                changes.append(item)
        elif op in ("remove", "set_strength"):
            change_id = _text(operation.get("id"), "id")
            strength = _strength(operation.get("strength")) if op == "set_strength" else None
            existing = next((c for c in changes if c["id"] == change_id), None)
            if existing is None:
                return False
            if op == "remove":
                changes.remove(existing)
            else:
                existing["strength"] = strength
        elif op == "new_scene":
            next_scene = self._composed(_text(operation.get("base"), "base"))
        elif op == "reset":
            next_scene = self._composed(self._initial_base)
        else:
            raise ValueError("unknown operation")

        if next_scene == self._scene:
            return False
        self._history.append(self._scene)
        # shortcut: keep the last 100 edits, use a persistent log for longer sessions.
        del self._history[:-100]
        self._scene = next_scene
        if new_id:
            self._next_id += 1
        return True

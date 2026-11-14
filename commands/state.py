"""Scene operations and bounded undo history; no model or network work."""

from copy import deepcopy


DEFAULT_BASE = "a wide shot of a meadow with a single oak tree"
KINDS = ("world", "season", "weather", "time", "extra")


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
        self._scene = {"base": self._initial_base, "changes": []}
        self._history = []
        self._next_id = 1

    def snapshot(self):
        return {**deepcopy(self._scene), "can_undo": bool(self._history)}

    def apply(self, operation):
        """Apply one validated operation atomically. No-ops do not add history."""
        if not isinstance(operation, dict):
            raise ValueError("operation must be an object")
        op = operation.get("op")
        if op == "batch":
            operations = operation.get("operations")
            if (not isinstance(operations, list) or not 1 <= len(operations) <= 16
                    or any(not isinstance(item, dict) or item.get("op") == "batch" for item in operations)):
                raise ValueError("batch needs 1 to 16 non-nested operations")
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
        if op == "add":
            label = _text(operation.get("label"), "label").lower()
            phrase = _text(operation.get("phrase"), "phrase")
            kind = operation.get("kind")
            if kind not in KINDS:
                raise ValueError("unknown change kind")
            strength = _strength(operation.get("strength", 1.0))
            existing = next((c for c in changes if c["label"].casefold() == label.casefold()), None)
            if existing and existing["kind"] == kind and existing["phrase"].casefold() == phrase.casefold() and existing["strength"] == strength:
                return False
            item = dict(id=existing["id"] if existing else f"c{self._next_id}",
                        label=label, kind=kind, phrase=phrase, strength=strength)
            new_id = existing is None
            if kind != "extra":
                changes[:] = [c for c in changes if c["kind"] != kind or c is existing]
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
            next_scene = {"base": _text(operation.get("base"), "base"), "changes": []}
        elif op == "reset":
            next_scene = {"base": self._initial_base, "changes": []}
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

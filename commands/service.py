"""CommandService: contract 1 in, contract 2 out, plus the contract-4 state fields. See PLAN.md section 6."""

import copy
import enum
import re
import threading
import time
import unicodedata

from .composer import compose
from .dictionary import normalize
from .reader import KEYWORDS, MAX_CLAUSES, RAMP, is_blocked, ramp_for, read, split_commands, spot, style_fallback
from .state import Origin, SceneState, is_free
from .world import WORLDS, world_op

PROVISIONAL_CAP = 0.4
SEMANTIC_WAIT_S = 0.05  # the local vector match is ~10 ms; never let it eat the slow LLM's budget


class Applied(enum.Enum):
    """What applying one operation (or one batch) to the scene did."""
    CHANGED = "changed"      # the scene changed: steer to the new composed prompt
    UNCHANGED = "unchanged"  # valid, but the scene is the same (a repeated add, undo with no history)
    REJECTED = "rejected"    # invalid or not understood: nothing applied, the words show as unmatched


def _has_non_latin_letters(text: str) -> bool:
    """True if normalization would drop letters (Chinese, Cyrillic, accented...): the reader cannot see them."""
    return any(ch.isalpha() and not ch.isascii() for ch in unicodedata.normalize("NFKC", text))


# Words a clause may have besides the keyword it was read by, for the keyword table's reading to be the whole
# clause: articles, edit and add verbs, pace and strength words. "add a dog" and "make it slowly snow" are read
# whole; "add a red car" (red), "kill the tree's leaves" and "now i want a savannah" are not.
FILLER_WORDS = frozenset("""
    a an the some it its it's this that make add put place bring give show me us turn into set let lets let's have
    get now please also then and just can could you i we want like there be is to of whole scene picture video
    more less little bit lot very slightly much really
    slowly gradually gently suddenly instantly quickly immediately by over time
    start again everything all from scratch
""".split())


# The explicit new scene words; "take me to X" is a place, which keeps the objects, so it is not one.
_NEW_SCENE_RX = re.compile(r"\bnew scene (?:of|with)\b")


def _words(text: str) -> list[str]:
    return normalize(text).split()


def fully_read(clause: str, reading: dict) -> bool:
    """True when `reading` (commands.reader.read(clause, ...)) accounts for every word of `clause`:

    - a tested place sentence (commands.world.world_op), an explicit "new scene of ...", a remove or strength
      change of an existing chip (read already refuses "make the dog black");
    - undo or reset with only filler words besides ("undo that", "start over please", not "start over with a beach");
    - an add of a keyword-table entry whose clause has only filler words besides the keyword ("add a dog").

    An add of an object the table does not know (the pass-through "add a glowing unicorn"), the "show me ..." guess,
    and any reading that leaves a word unused ("add a red car", "kill the tree's leaves") are not read whole."""
    op, keyword = reading["op"], reading["keyword"]
    if op is None:
        return False
    t = normalize(clause)
    kind = op["op"]
    if kind in ("remove", "set_strength"):
        return True
    if kind == "new_scene":
        return bool(_NEW_SCENE_RX.search(t))
    if world_op(t) is not None:
        return True
    if kind in ("undo", "reset") or (kind == "add" and keyword in KEYWORDS):
        rest = _words(t)
        key = _words(keyword or "")
        for i in range(len(rest) - len(key) + 1):  # drop the first match of the keyword itself
            if key and rest[i:i + len(key)] == key:
                rest = rest[:i] + rest[i + len(key):]
                break
        return all(w in FILLER_WORDS for w in rest)
    return False


def dictionary_reading(text: str, snapshot: dict) -> list[dict] | None:
    """The keyword table's operations for `text` on `snapshot` when it reads the whole sentence, else None.

    The rule both sides use for the fast path (engine/reader.py): a composed scene, Latin text, 1 to MAX_CLAUSES
    clauses, every clause read whole by the keyword table alone (fully_read: no vector matcher, no style fallback,
    no pass-through guess), and every operation applies and composes on a trial copy. A free scene gives None: only
    the final reader writes its prompt. Pure: `snapshot` is not changed."""
    if not isinstance(text, str) or is_free(snapshot) or _has_non_latin_letters(text):
        return None
    clauses = split_commands(text)
    if not 1 <= len(clauses) <= MAX_CLAUSES:
        return None
    trial = SceneState.from_snapshot(snapshot)
    operations = []
    for clause in clauses:
        reading = read(clause, trial.snapshot())
        if not fully_read(clause, reading):
            return None
        op = reading["op"]
        try:
            trial.apply(op)
            compose(trial.snapshot())
        except (ValueError, TypeError, KeyError):
            return None
        operations.extend(op["operations"] if op.get("op") == "batch" else [op])
    return operations


def _validate(msg) -> None:
    if not isinstance(msg, dict):
        raise ValueError("message must be an object")
    kind = msg.get("type")
    if kind in ("partial", "final"):
        if not isinstance(msg.get("text"), str):
            raise ValueError("text must be a string")
        t = msg.get("t_client")
        if isinstance(t, bool) or not isinstance(t, int):
            raise ValueError("t_client must be an integer")
    elif kind == "tap":
        action = msg.get("action")
        if action not in ("remove", "undo", "reset"):
            raise ValueError(f"unknown tap action {action!r}")
        if action == "remove" and not (isinstance(msg.get("id"), str) and msg["id"]):
            raise ValueError("remove needs an id")
    else:
        raise ValueError(f"unknown type {kind!r}")


class CommandService:
    def __init__(self, base: str | None = None, slow_reader=None, slow_timeout_s: float = 0.4, semantic_reader=None):
        self._state = SceneState() if base is None else SceneState(base)
        self._slow_reader = slow_reader
        if semantic_reader is None:
            from .semantic import configured_reader
            semantic_reader = configured_reader()
        self._semantic_reader = semantic_reader
        self._slow_timeout = slow_timeout_s
        self._slow_thread: threading.Thread | None = None
        self._semantic_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._committed = compose(self._state.snapshot())
        self._guess_label: str | None = None  # label of the provisional move in flight; None: no guess in flight
        self._heard = None
        self._keyword = None
        self._unmatched = False

    @property
    def committed_prompt(self) -> str:
        return self._committed

    def state_fields(self) -> dict:
        with self._lock:
            return self._fields()

    def likely_prompts(self) -> list[str]:
        with self._lock:
            if self._state.origin is Origin.FREE:  # the dictionary cannot edit a free scene: nothing to encode ahead
                return [self._committed]
            labels = {e["label"]: e for e in KEYWORDS.values()}
            candidates = {self._committed}
            labels.update({e["label"]: e for e in WORLDS.values()})
            for entry in labels.values():
                try:
                    candidates.add(self._hypothesis(entry))
                except ValueError:  # an unstructured base may not support a safe world replacement
                    continue
            return sorted(candidates)

    def snapshot(self) -> dict:
        """A deep copy of the committed scene state, for the engine's LLM final reader (with its "origin")."""
        with self._lock:
            return self._state.snapshot()

    def apply_final(self, text: str, operations) -> dict:
        """Apply the engine's LLM decision for one final sentence: 0..16 operations, applied atomically, with the
        same steer and state shape as handle(). The steer is the composed prompt of the new scene; the engine maps
        it to its rich prompt.

        [] means blocked or not understood: nothing applied, unmatched, a provisional guess is rolled back.
        Invalid operations reject the whole final the same way.
        """
        with self._lock:
            self._heard = text
            self._keyword = None
            valid = (isinstance(operations, list) and len(operations) <= MAX_CLAUSES
                     and all(isinstance(o, dict) for o in operations))
            op = None
            if valid and operations:
                op = operations[0] if len(operations) == 1 else {"op": "batch", "operations": operations}
                trial = copy.deepcopy(self._state)
                try:
                    trial.apply(op)
                    compose(trial.snapshot())  # a world/base conflict must fail here, before the live state changes
                except (ValueError, TypeError, KeyError):
                    op = None
            if op is None:
                self._unmatched = True
                return {"steer": self._commit(Applied.REJECTED, RAMP["default"]), "state": self._fields()}
            applied = self._apply(op)
            self._unmatched = False
            ramp = ramp_for(text) if isinstance(text, str) else RAMP["default"]  # "slowly ..." = 12
            return {"steer": self._commit(applied, ramp), "state": self._fields()}

    def handle(self, msg: dict) -> dict:
        _validate(msg)
        with self._lock:
            kind = msg["type"]
            if kind == "partial":
                steer = self._partial(msg["text"])
            elif kind == "final":
                steer = self._final(msg["text"])
            else:
                self._heard = self._keyword = None
                self._unmatched = False
                op = {"op": "remove", "id": msg["id"]} if msg["action"] == "remove" else {"op": msg["action"]}
                steer = self._commit(self._apply(op), RAMP["default"])
                if msg["action"] == "reset":
                    # The engine restarts the video on this prompt with no ramp (board #6884241134): always send it.
                    steer = self._steer(self._committed, RAMP["sudden"])
            return {"steer": steer, "state": self._fields()}

    # ---------- internals (lock held) ----------

    def _fields(self) -> dict:
        return {
            "changes": [{"id": c["id"], "label": c["label"], "strength": c["strength"]} for c in self._state.snapshot()["changes"]],
            "heard": self._heard,
            "keyword": self._keyword,
            "unmatched": self._unmatched,
        }

    def _hypothesis(self, entry: dict) -> str:
        """Prompt if this keyword were added, computed on a copy: the live state is never touched."""
        if any(c["label"] == entry["label"] for c in self._state.snapshot()["changes"]):
            return self._committed
        trial = copy.deepcopy(self._state)
        trial.apply({"op": "add", "label": entry["label"], "kind": entry["kind"], "phrase": entry["phrase"], "strength": 1.0})
        return compose(trial.snapshot())

    def _steer(self, prompt: str, ramp: int, provisional: bool = False) -> dict:
        return {"prompt": prompt, "ramp_chunks": ramp, "provisional": provisional, "cap": PROVISIONAL_CAP if provisional else 1.0}

    def _partial(self, text: str):
        self._heard = text
        self._unmatched = False
        if self._state.origin is Origin.FREE:
            # Only the final reader edits a free scene, so a partial guesses nothing (a guess in flight, from before
            # the scene became free, cannot exist: every final ends it).
            self._keyword = None
            return None
        word = spot(text)
        self._keyword = word
        label = KEYWORDS[word]["label"] if word else None
        if self._guess_label is not None and label == self._guess_label:
            return None
        prompt = self._hypothesis(KEYWORDS[word]) if word else self._committed
        if prompt == self._committed:  # word dropped by a revision, or already in the scene: move back if guessing
            had_guess = self._guess_label is not None
            self._guess_label = None
            return self._steer(self._committed, RAMP["default"]) if had_guess else None
        self._guess_label = label
        return self._steer(prompt, RAMP["provisional"], provisional=True)

    def _final(self, text: str):
        self._heard = text
        if _has_non_latin_letters(text):
            return self._final_original(text)
        clauses = split_commands(text)
        if not 1 <= len(clauses) <= MAX_CLAUSES:
            self._keyword = None
            self._unmatched = True
            return self._commit(Applied.REJECTED, RAMP["default"])
        trial = copy.deepcopy(self._state)
        operations = []
        pace = None
        self._keyword = None
        deadline = time.monotonic() + self._slow_timeout
        for clause in clauses:
            snapshot = trial.snapshot()
            result = read(clause, snapshot)
            pace = result["pace"] or pace
            self._keyword = result["keyword"] or self._keyword
            op = result["op"]
            if op is None:
                if self._semantic_reader is not None and not is_blocked(clause):
                    op = self._ask_semantic(clause, snapshot, min(SEMANTIC_WAIT_S, max(0, deadline - time.monotonic())))
                if op is None and self._slow_reader is not None:
                    op = self._ask_slow(clause, snapshot, max(0, deadline - time.monotonic()))
                if op is None:
                    op = style_fallback(clause, snapshot)  # last resort: "make it spooky" -> a "spooky" look
            if op is None:
                operations = []
                break
            try:
                trial.apply(op)
                compose(trial.snapshot())
            except (ValueError, TypeError, KeyError):
                operations = []
                break
            # A world-with-time phrase is itself a batch: flatten it so the sentence stays one non-nested batch.
            operations.extend(op["operations"] if op.get("op") == "batch" else [op])
        applied = (self._apply(operations[0] if len(operations) == 1 else {"op": "batch", "operations": operations})
                   if operations else Applied.REJECTED)
        self._unmatched = applied is Applied.REJECTED
        return self._commit(applied, RAMP.get(pace or "default", RAMP["default"]))

    def _final_original(self, text: str):
        """Text the reader cannot see (non-Latin letters): the slow reader gets the ORIGINAL text, one operation.
        No answer means unmatched; the dictionary never makes a partial edit from the Latin leftovers."""
        self._keyword = None
        op = self._ask_slow(text, self._state.snapshot()) if self._slow_reader is not None else None
        applied = Applied.REJECTED
        if op is not None:
            trial = copy.deepcopy(self._state)
            try:
                trial.apply(op)
                compose(trial.snapshot())
                applied = self._apply(op)
            except (ValueError, TypeError, KeyError):
                applied = Applied.REJECTED
        self._unmatched = applied is Applied.REJECTED
        return self._commit(applied, ramp_for(text))

    def _ask_semantic(self, text: str, snapshot: dict, timeout: float):
        """One bounded call to the local vector matcher (returns an op, never mutates). At most one in flight;
        on timeout the caller moves on to the slow LLM and a late answer is discarded."""
        if timeout <= 0 or self._semantic_thread is not None and self._semantic_thread.is_alive():
            return None
        box = {}

        def run():
            try:
                box["op"] = self._semantic_reader(text, snapshot)
            except Exception:  # noqa: BLE001 - an injected/model failure falls through to the slow reader
                box["op"] = None

        self._semantic_thread = threading.Thread(target=run, daemon=True, name="commands-semantic-reader")
        self._semantic_thread.start()
        self._semantic_thread.join(timeout)
        op = box.get("op")
        return op if isinstance(op, dict) else None

    def _apply(self, op) -> Applied:
        try:
            return Applied.CHANGED if self._state.apply(op) else Applied.UNCHANGED
        except (ValueError, TypeError, KeyError):
            return Applied.REJECTED

    def _commit(self, applied: Applied, ramp: int):
        """After a final or a tap: steer to the new state, or roll back a guess in flight."""
        had_guess = self._guess_label is not None
        self._guess_label = None
        if applied is Applied.CHANGED:
            self._committed = compose(self._state.snapshot())
            return self._steer(self._committed, ramp)
        return self._steer(self._committed, RAMP["default"]) if had_guess else None

    def _ask_slow(self, text: str, snapshot: dict, timeout=None):
        """One bounded call to the injected slow reader. At most one in flight; a hung one blocks further calls."""
        if timeout == 0 or self._slow_thread is not None and self._slow_thread.is_alive():
            return None
        box = {}

        def run():
            try:
                box["op"] = self._slow_reader(text, snapshot)
            except Exception:  # noqa: BLE001 - any failure of the injected callback is just "unmatched"
                box["op"] = None

        self._slow_thread = threading.Thread(target=run, daemon=True, name="commands-slow-reader")
        self._slow_thread.start()
        self._slow_thread.join(self._slow_timeout if timeout is None else timeout)
        op = box.get("op")
        return op if isinstance(op, dict) else None

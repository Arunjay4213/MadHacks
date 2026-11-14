"""CommandService: contract 1 in, contract 2 out, plus the contract-4 state fields. See PLAN.md section 6."""

import copy
import threading
import time
import unicodedata

from .composer import compose
from .reader import KEYWORDS, MAX_CLAUSES, RAMP, is_blocked, read, split_commands, spot, style_fallback
from .state import SceneState
from .world import WORLDS

PROVISIONAL_CAP = 0.4
MAX_FINAL_PROMPT_CHARS = 4000
SEMANTIC_WAIT_S = 0.05  # the local vector match is ~10 ms; never let it eat the slow LLM's budget


def _has_non_latin_letters(text: str) -> bool:
    """True if normalization would drop letters (Chinese, Cyrillic, accented...): the reader cannot see them."""
    return any(ch.isalpha() and not ch.isascii() for ch in unicodedata.normalize("NFKC", text))


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
        self._guess: str | None = None  # keyword of the provisional move in flight
        self._guess_prompt: str | None = None
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
        """A deep copy of the committed scene state, for the engine's LLM final reader."""
        with self._lock:
            return self._state.snapshot()

    def apply_final(self, text: str, operations, prompt=None) -> dict:
        """Apply the engine's LLM decision for one final sentence: 0..16 operations, applied atomically.

        [] means blocked or not understood: nothing applied, unmatched, a provisional guess is rolled back.
        Invalid operations or an invalid prompt (non-string, empty, > MAX_FINAL_PROMPT_CHARS) reject the whole
        final the same way. A valid prompt replaces the composed prompt as the committed prompt and the steer,
        so later rollbacks return to it; undo returns to the canonical composed prompt of the earlier state.
        """
        with self._lock:
            self._heard = text
            self._keyword = None
            if prompt is not None:
                prompt = prompt.strip() if isinstance(prompt, str) else ""
            valid = (isinstance(operations, list) and len(operations) <= MAX_CLAUSES
                     and all(isinstance(o, dict) for o in operations)
                     and (prompt is None or 0 < len(prompt) <= MAX_FINAL_PROMPT_CHARS))
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
                return {"steer": self._commit(None, RAMP["default"]), "state": self._fields()}
            changed = self._apply(op)
            self._unmatched = False
            pace = read(text, self._state.snapshot())["pace"] if isinstance(text, str) else None  # "slowly ..." = 12
            ramp = RAMP.get(pace or "default", RAMP["default"])
            if prompt is None:
                return {"steer": self._commit(changed, ramp), "state": self._fields()}
            guess_prompt = self._guess_prompt
            self._guess = self._guess_prompt = None
            if prompt == self._committed and guess_prompt is None:
                steer = None  # same scene, same words: nothing to send
            else:
                self._committed = prompt
                steer = self._steer(prompt, RAMP["confirm"] if prompt == guess_prompt and ramp == RAMP["default"] else ramp)
            return {"steer": steer, "state": self._fields()}

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
        word = spot(text)
        self._keyword = word
        label = KEYWORDS[word]["label"] if word else None
        if label == self._guess:
            return None
        prompt = self._hypothesis(KEYWORDS[word]) if word else self._committed
        if prompt == self._committed:  # word dropped by a revision, or already in the scene: move back if guessing
            had_guess = self._guess_prompt is not None
            self._guess = self._guess_prompt = None
            return self._steer(self._committed, RAMP["default"]) if had_guess else None
        self._guess, self._guess_prompt = label, prompt
        return self._steer(prompt, RAMP["provisional"], provisional=True)

    def _final(self, text: str):
        self._heard = text
        if _has_non_latin_letters(text):
            return self._final_original(text)
        clauses = split_commands(text)
        if not 1 <= len(clauses) <= MAX_CLAUSES:
            self._keyword = None
            self._unmatched = True
            return self._commit(None, RAMP["default"])
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
                    op = style_fallback(clause)  # last resort: "make it spooky" -> a "spooky" look
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
        changed = self._apply(operations[0] if len(operations) == 1 else {"op": "batch", "operations": operations}) if operations else None
        self._unmatched = changed is None
        return self._commit(changed, RAMP.get(pace or "default", RAMP["default"]))

    def _final_original(self, text: str):
        """Text the reader cannot see (non-Latin letters): the slow reader gets the ORIGINAL text, one operation.
        No answer means unmatched; the dictionary never makes a partial edit from the Latin leftovers."""
        self._keyword = None
        op = self._ask_slow(text, self._state.snapshot()) if self._slow_reader is not None else None
        changed = None
        if op is not None:
            trial = copy.deepcopy(self._state)
            try:
                trial.apply(op)
                compose(trial.snapshot())
                changed = self._apply(op)
            except (ValueError, TypeError, KeyError):
                changed = None
        self._unmatched = changed is None
        return self._commit(changed, RAMP["default"])

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

    def _apply(self, op):
        """True/False from state.apply, or None if the op was rejected as invalid."""
        try:
            return self._state.apply(op)
        except (ValueError, TypeError, KeyError):
            return None

    def _commit(self, changed, ramp: int):
        """After a final or a tap: steer to the new state, or roll back a guess in flight."""
        guess_prompt = self._guess_prompt
        self._guess = self._guess_prompt = None
        if changed:
            self._committed = compose(self._state.snapshot())
            if guess_prompt == self._committed and ramp == RAMP["default"]:
                ramp = RAMP["confirm"]  # the guess already moved part way
            return self._steer(self._committed, ramp)
        return self._steer(self._committed, RAMP["default"]) if guess_prompt is not None else None

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

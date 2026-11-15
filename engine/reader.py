"""The command reader boundary, and loading it by name (--reader standin|commands).

A reader is an object with this API (agreed with the commands/ owner):

    svc.handle(contract1_msg) -> {"steer": contract 2 dict or None,
                                  "state": {"changes": [{"id", "label", "strength"}], "heard", "keyword", "unmatched"}}
        Raises ValueError on an invalid contract 1 message. May block up to about 1.5 s (the slow reader's LLM call),
        or up to --final-timeout-s with --final-reader sonnet.
    svc.state_fields() -> the same state dict; used for every contract 4 message, from the streamer thread.
    svc.likely_prompts() -> list[str]; prompts worth encoding ahead of time. Asked after every final and tap,
        not after a partial: a partial must not change it (it never changes the committed scene).
    svc.committed_prompt -> str or None; the current scene's prompt.

The engine calls `handle` on one worker thread, in arrival order, never on the generation thread or event loop.

With --final-reader sonnet, FinalReadingService wraps CommandService and decides who reads each FINAL, in this order:
undo and reset sentences (the dictionary), the content block, a pinned or cached answer, a measured car world, the
dictionary when it reads the whole sentence (commands.service.dictionary_reading) and the result is one the keyword
table writes well (fast_path_ok: no object adds, no partial strengths, nothing added on a vehicle scene), and otherwise
one Sonnet call that
writes the whole new prompt (a free scene, final_reader.py). The answer is applied to the service's own scene, so
chips, heard and undo work as for any other edit.
"""

import enum
import logging
import re

from commands.service import dictionary_reading
from commands.state import is_free

log = logging.getLogger("engine.reader")

# Kinds of change the keyword table writes as well as Sonnet: a tested place, season, weather or time of day at full
# strength. Its object phrases change the count or the thing ("add an eagle" -> "a flock of birds", "add a human" ->
# "a small group of people") and carry no size or position, so the model draws them small or not at all; a partial
# strength reads "a subtle hint of a thick fog"; and on a vehicle scene the car template keeps the world's own light and
# weather words beside the new ones ("clear blue summer sky" with "a thick fog"). Regression sweep on #273.
FAST_PATH_KINDS = frozenset({"season", "weather", "time", "world"})


def fast_path_ok(operations: list[dict], snapshot: dict) -> bool:
    """True when the dictionary's `operations` for a whole sentence may skip Sonnet: undo, reset, remove, strength
    changes and explicit new scenes, and adds of FAST_PATH_KINDS at full strength on a scene without a vehicle."""
    from .car_template import vehicle_subject
    for op in operations:
        if op["op"] != "add":
            continue
        if op.get("kind") not in FAST_PATH_KINDS or op.get("strength", 1.0) != 1.0 \
                or vehicle_subject(snapshot.get("base")) is not None:
            return False
    return True

READERS = ("standin", "commands")


def load_reader(name: str, final_reader=None):
    """The reader `name`. `final_reader` (a final_reader.FinalReader, ON or not) is used only with "commands";
    with any other reader an ON final reader is marked UNAVAILABLE."""
    from .final_reader import FinalReaderState
    on = final_reader is not None and final_reader.state is FinalReaderState.ON
    if name == "standin":
        if on:
            final_reader.mark_unavailable("--final-reader sonnet needs --reader commands")
        from .standin_reader import StandinService
        return StandinService()
    if name == "commands":
        from commands.service import CommandService  # the commands/ owner's reader; imported only when chosen
        from .slow_reader import SERVICE_TIMEOUT_S, StateCachingReader, from_env
        if on:  # the Sonnet final reader replaces the nano slow reader
            return StateCachingReader(FinalReadingService(CommandService(), final_reader))
        return StateCachingReader(CommandService(slow_reader=from_env(), slow_timeout_s=SERVICE_TIMEOUT_S))
    raise ValueError(f"unknown reader {name!r}; known: {', '.join(READERS)}")


class FinalReadingService:
    """CommandService with the final reader around the dictionary for every FINAL (the order is in the module
    docstring).

    final_reader.result(...) gives None (the service reads the sentence itself, as without this wrapper, and on a
    composed vehicle scene final_reader.fallback_prompt gives the car template for the result), {"blocked": True}
    or {"not_understood": True} (no steer: unmatched, a guess in flight is rolled back), or {"operations", "prompt"}
    (applied to the service's scene; for Sonnet one free_scene operation).

    The model sees one prompt per scene state: the first rich prompt chosen for a composed prompt is kept in
    `_model_prompts` and replaces that composed prompt in every later steer (an undo back to it, a chip removal,
    a partial's guess) and in likely_prompts and committed_prompt, so a state always gives the same embedding.
    A free scene's composed prompt is its own prompt, so it needs no mapping.

    A tap on the x of a chip of a free scene is read as the sentence "remove the <label>": only the final reader can
    take a thing out of a prompt it wrote.
    """

    def __init__(self, service, final_reader) -> None:
        self._svc = service
        self._final_reader = final_reader
        self._model_prompts: dict[str, str] = {}  # composed prompt -> the prompt the model sees for it

    def handle(self, msg: dict) -> dict:
        if isinstance(msg, dict) and msg.get("type") == "final" and isinstance(msg.get("text"), str) \
                and isinstance(msg.get("t_client"), int) and not isinstance(msg.get("t_client"), bool):
            return self._final(msg)
        if isinstance(msg, dict) and msg.get("type") == "tap" and msg.get("action") == "remove":
            snapshot = self._svc.snapshot()
            chip = next((c for c in snapshot["changes"] if c["id"] == msg.get("id")), None)
            if is_free(snapshot) and chip is not None:
                text = f"remove the {chip['label']}"
                log.info("final reader: the x on free chip %r is read as %r", chip["label"], text)
                return self._final({"type": "final", "text": text, "t_client": 0})
        return self._mapped(self._svc.handle(msg))

    def _final(self, msg: dict) -> dict:
        text = msg["text"]
        snapshot = self._svc.snapshot()
        kind = sentence_kind(text, snapshot)
        if kind is SentenceKind.CONTROL:  # the dictionary knows these and they add nothing; no call
            return self._mapped(self._svc.handle(msg))
        fr = self._final_reader
        entry = fr.lookup(text, snapshot)  # the content block, pinned, cached or measured: no call
        reading = dictionary_reading(text, snapshot) if entry is None else None
        whole = reading is not None and fast_path_ok(reading, snapshot)
        if kind is SentenceKind.MIXED:
            if entry is None and not whole:
                split = leading_controls(text, snapshot)
                if split is not None:  # "undo and add a dog": the edits are read on the scene the controls give
                    return self._controls_then_edits(msg, *split)
            # The final reader leaves undo and reset to the dictionary, which reads the whole sentence. When it does
            # not read every clause by itself, Sonnet is still asked first, as the content check.
            if entry is None and not whole:
                entry = fr.ask(text, snapshot, self.committed_prompt)
            if entry is not None and fr.result(entry, text, snapshot) == {"blocked": True}:
                return self._blocked(text)
            return self._dictionary(msg, snapshot)
        if whole:
            return self._dictionary(msg, snapshot)  # the fast path: the keyword table reads every word
        if entry is None:
            entry = fr.ask(text, snapshot, self.committed_prompt)
        result = fr.result(entry, text, snapshot)
        if result is None:
            return self._dictionary(msg, snapshot)  # no answer: a timeout, an error or a reply that failed the check
        if result.get("blocked"):
            return self._blocked(text)
        if result.get("not_understood"):
            log.info("final reader: %r not understood, no steer", text[:100])
            return self._mapped(self._svc.apply_final(text, []))
        # The service steers to the composed prompt and _mapped gives the rich one, so an undo back to this state,
        # or another sentence reaching it, gives the same words (the same embedding).
        out = self._svc.apply_final(text, result["operations"])
        if out["steer"] is not None:
            self._model_prompts.setdefault(out["steer"]["prompt"], result["prompt"])
        return self._mapped(out)

    def _controls_then_edits(self, msg: dict, controls: list[dict], edits: str) -> dict:
        """A sentence of bare undo or reset clauses followed by edit clauses that the dictionary does not read
        whole. The controls run first on a trial copy of the scene (with its undo history), and the edits are read
        on the scene they give, as one EDIT sentence: the dictionary when it reads them whole, else the final reader
        (so "undo and add a dog" on a free scene reaches Sonnet with the scene before the last edit). The controls
        and the edits' operations then apply as one batch: one undo step, and nothing applies when the edits are
        blocked, not understood or unread."""
        from commands.composer import compose
        text = msg["text"]
        before = self._svc.snapshot()
        after = self._svc_trial(controls)
        fr = self._final_reader
        entry = fr.lookup(edits, after)
        if entry is None and dictionary_reading(edits, after) is not None:
            return self._dictionary(msg, before)  # the keyword table reads the whole sentence with the real undo
        if entry is None:
            entry = fr.ask(edits, after, self._model_prompts.get(compose(after), compose(after)))
        result = fr.result(entry, edits, after)
        if result is None:
            return self._dictionary(msg, before)  # no answer: as for any sentence (not understood on a free scene)
        if result.get("blocked"):
            return self._blocked(text)
        if result.get("not_understood"):
            log.info("final reader: %r not understood, no steer", text[:100])
            return self._mapped(self._svc.apply_final(text, []))
        out = self._svc.apply_final(text, controls + result["operations"])
        if out["steer"] is not None:
            self._model_prompts.setdefault(out["steer"]["prompt"], result["prompt"])
        return self._mapped(out)

    def _svc_trial(self, operations: list[dict]) -> dict:
        """The snapshot after `operations` on a copy of the service's scene with its undo history (an undo needs the
        history, which a snapshot does not carry). The live scene is not changed."""
        import copy
        with self._svc._lock:  # CommandService has no public trial with history; this reads it under its lock
            trial = copy.deepcopy(self._svc._state)
        for op in operations:
            trial.apply(op)
        return trial.snapshot()

    def _blocked(self, text: str) -> dict:
        log.info("final reader: %r blocked, no steer", text[:100])
        return self._mapped(self._svc.apply_final(text, []))

    def _dictionary(self, msg: dict, snapshot: dict) -> dict:
        """A final the dictionary reads (the fast path, no usable answer, or an undo or reset with edits). On a
        composed vehicle scene the final reader still gives the car template for the new state, so the car keeps
        driving forward. On a free scene the dictionary applies only a new scene; anything else shows as not
        understood (commands.state refuses edits of a free scene), never a silent redraw."""
        out = self._svc.handle(msg)
        if out["steer"] is not None and not out["state"]["unmatched"] \
                and out["steer"]["prompt"] not in self._model_prompts:
            prompt = self._final_reader.fallback_prompt(snapshot, self._svc.snapshot())
            if prompt is not None:
                self._model_prompts[out["steer"]["prompt"]] = prompt
        return self._mapped(out)

    def state_fields(self) -> dict:
        return self._svc.state_fields()

    def likely_prompts(self) -> list[str]:
        return [self._model_prompts.get(p, p) for p in self._svc.likely_prompts()]

    @property
    def committed_prompt(self):
        p = self._svc.committed_prompt
        return self._model_prompts.get(p, p)

    def _mapped(self, out: dict) -> dict:
        steer = out.get("steer")
        if steer is not None and steer["prompt"] in self._model_prompts:
            out = {**out, "steer": {**steer, "prompt": self._model_prompts[steer["prompt"]]}}
        return out


class SentenceKind(enum.Enum):
    """What a final sentence asks for, which decides whether it may skip the final reader's content check."""
    CONTROL = "control"  # every clause is a bare undo or reset: the dictionary alone, no call (nothing new enters)
    MIXED = "mixed"      # bare undo or reset plus edit clauses: content check first, then the dictionary
    EDIT = "edit"        # no bare undo or reset, or a reset that names a scene: the final reader decides


class ClauseKind(enum.Enum):
    """What one clause of a final sentence is."""
    BARE_CONTROL = "bare_control"  # undo, or a reset that names nothing ("start over please")
    SCENE_RESET = "scene_reset"    # a reset that names a scene ("start over with a beach"): a new scene
    EDIT = "edit"                  # anything else


# Words that may follow a reset ("start over please", "reset everything", "reset immediately") without naming a
# scene: filler, and the pace words of commands/reader.py. Any other word ("start over with a beach") names content.
RESET_FILLER = frozenset("""
    please now everything it all the scene video whole thing again from scratch completely entirely this that
    slowly gradually gently little by over time suddenly instantly immediately right quickly at once
""".split())
_WORDS = re.compile(r"[a-z']+")


def sentence_kind(text: str, snapshot: dict) -> SentenceKind:
    """Classify each clause the way CommandService splits the sentence, so a clause like "add ..." after
    "reset and" is seen and cannot ride past the content check on the reset.

    A scene reset with no undo or reset after it makes the whole sentence EDIT: the keyword table would read it
    as a plain reset and drop the scene, and the final reader's new_scene already clears the changes, so a bare
    reset before it adds nothing. An undo or reset after it ("start over with a beach and reset") must still run
    last, which only the keyword table does, so that sentence is MIXED."""
    from commands.reader import read, split_commands
    kinds = [clause_kind(clause, read(clause, snapshot)) for clause in split_commands(text) or [text]]
    if ClauseKind.SCENE_RESET in kinds:
        last_scene = len(kinds) - 1 - kinds[::-1].index(ClauseKind.SCENE_RESET)
        if ClauseKind.BARE_CONTROL not in kinds[last_scene + 1:]:
            return SentenceKind.EDIT
        return SentenceKind.MIXED
    if all(k is ClauseKind.BARE_CONTROL for k in kinds):
        return SentenceKind.CONTROL
    return SentenceKind.MIXED if ClauseKind.BARE_CONTROL in kinds else SentenceKind.EDIT


def leading_controls(text: str, snapshot: dict) -> tuple[list[dict], str] | None:
    """(the undo and reset operations, the rest of the sentence) when `text` is one or more bare undo or reset
    clauses followed only by edit clauses ("undo and add a dog", "reset, then make the grass dry"); else None
    (no control first, a control after an edit, or nothing after the controls)."""
    from commands.reader import read, split_commands
    clauses = split_commands(text) or [text]
    readings = [read(clause, snapshot) for clause in clauses]
    kinds = [clause_kind(clause, reading) for clause, reading in zip(clauses, readings)]
    n = next((i for i, k in enumerate(kinds) if k is not ClauseKind.BARE_CONTROL), len(kinds))
    if n == 0 or n == len(kinds) or any(k is not ClauseKind.EDIT for k in kinds[n:]):
        return None
    return [readings[i]["op"] for i in range(n)], " and ".join(clauses[n:])


def clause_kind(clause: str, reading: dict) -> ClauseKind:
    """`reading` is commands.reader.read(clause, ...). The keyword table reads "start over with a beach" as a
    plain reset; the words left after the reset words tell a scene reset from a bare one."""
    op = reading["op"]
    if op is None or op["op"] not in ("undo", "reset"):
        return ClauseKind.EDIT
    if op["op"] == "undo":
        return ClauseKind.BARE_CONTROL
    words = _WORDS.findall(clause.lower())
    keyword = _WORDS.findall((reading["keyword"] or "").lower())
    for i in range(len(words) - len(keyword) + 1):  # drop the first match of the reset words themselves
        if keyword and words[i:i + len(keyword)] == keyword:
            words = words[:i] + words[i + len(keyword):]
            break
    return ClauseKind.BARE_CONTROL if all(w in RESET_FILLER for w in words) else ClauseKind.SCENE_RESET

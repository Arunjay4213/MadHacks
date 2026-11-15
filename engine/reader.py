"""The command reader boundary, and loading it by name (--reader standin|commands).

A reader is an object with this API (agreed with the commands/ owner):

    svc.handle(contract1_msg) -> {"steer": contract 2 dict or None,
                                  "state": {"changes": [{"id", "label", "strength"}], "heard", "keyword", "unmatched"}}
        Raises ValueError on an invalid contract 1 message. May block up to about 1.5 s (the slow reader's LLM call),
        or up to --final-timeout-s with --final-reader sonnet.
    svc.state_fields() -> the same state dict; used for every contract 4 message, from the streamer thread.
    svc.likely_prompts() -> list[str]; prompts worth encoding ahead of time.
    svc.committed_prompt -> str or None; the current scene's prompt.

The engine calls `handle` on one worker thread, in arrival order, never on the generation thread or event loop.

With --final-reader sonnet, FinalReadingService wraps CommandService: every FINAL goes to the final reader
(final_reader.py) before the dictionary. Its operations are applied to the service's own scene, so chips, heard
and undo work as for any other edit, and the steer's prompt becomes the final reader's rich prompt.
"""

import enum
import logging
import re

log = logging.getLogger("engine.reader")

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
    """CommandService with the final reader in front of the dictionary for every FINAL.

    final_reader(text, snapshot) gives None (the service reads the sentence itself, as without this wrapper),
    {"blocked": True} (no steer: unmatched), or {"operations", "prompt"} (applied to the service's scene).

    The model sees one prompt per scene state: the first rich prompt chosen for a composed prompt is kept in
    `_model_prompts` and replaces that composed prompt in every later steer (an undo back to it, a chip removal,
    a partial's guess) and in likely_prompts and committed_prompt, so a state always gives the same embedding.
    """

    def __init__(self, service, final_reader) -> None:
        self._svc = service
        self._final_reader = final_reader
        self._model_prompts: dict[str, str] = {}  # composed prompt -> the prompt the model sees for it

    def handle(self, msg: dict) -> dict:
        if isinstance(msg, dict) and msg.get("type") == "final" and isinstance(msg.get("text"), str) \
                and isinstance(msg.get("t_client"), int) and not isinstance(msg.get("t_client"), bool):
            snapshot = self._svc.snapshot()
            kind = sentence_kind(msg["text"], snapshot)
            if kind is SentenceKind.CONTROL:  # the dictionary knows these and they add nothing; no call
                return self._mapped(self._svc.handle(msg))
            result = self._final_reader(msg["text"], snapshot)  # the content check, for every other sentence
            if result is not None and result.get("blocked"):
                log.info("final reader: %r blocked, no steer", msg["text"][:100])
                return self._mapped(self._svc.apply_final(msg["text"], []))
            if kind is SentenceKind.MIXED:
                # Not blocked. The final reader leaves undo and reset to the dictionary, so its operations would
                # drop them: the dictionary reads the whole sentence, as it would without this wrapper.
                return self._mapped(self._svc.handle(msg))
            if result is not None:
                # The service steers to the composed prompt and _mapped gives the rich one, so an undo back to
                # this state, or another sentence reaching it, gives the same words (the same embedding).
                out = self._svc.apply_final(msg["text"], result["operations"])
                if out["steer"] is not None:
                    self._model_prompts.setdefault(out["steer"]["prompt"], result["prompt"])
                return self._mapped(out)
        return self._mapped(self._svc.handle(msg))

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

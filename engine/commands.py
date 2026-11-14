"""One record per command, its progress as an explicit stage, and the five delay stamps.

Stage order for a command that reaches the screen:
    HEARD -> DECIDED -> EMBEDDED -> SCHEDULED -> GENERATED -> SHOWN
Terminal stages: SHOWN, DROPPED (the reader found no operation, an error, or a reset heard after it), SUPERSEDED (a newer
command's target was set before this one reached the model, or the steering Controller called it stale),
UNCHANGED (the steering Controller found the target already applied).

Stamps (monotonic clock, reported as integer ms after `heard`):
    heard      the server received the contract 1 message (or the POST /target body)
    decided    the command reader returned an operation (contract 2)
    embedded   the embedding was ready and the steering Controller applied it (it calls steerer.set_target)
    generated  the first chunk whose get_cond came after set_target finished generating
    shown      the first frame of that chunk was sent on the /frames socket
A command the reader found no operation for (DROPPED from HEARD) has only `heard`.

The most recent command, whose t_client, stamps and applies_from_chunk contract 4 carries, is the newest one the
reader read: a final, a tap or a POST /target, with or without an operation, and a partial only when it got a steer.
A reader rejection or error, and a partial with no steer, do not replace it.

The CommandBook lock is also the lock that orders Controller.handle (and so set_target) against get_cond, so
"the first get_cond after set_target" is exact. handle is only called under this lock once the embedding is in
the engine's encoder cache, so it does not wait for encoding while holding it.
"""

import collections
import enum
import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger("engine.commands")

STAMP_NAMES = ("heard", "decided", "embedded", "generated", "shown")


class CommandStage(enum.Enum):
    HEARD = "HEARD"
    DECIDED = "DECIDED"
    EMBEDDED = "EMBEDDED"
    SCHEDULED = "SCHEDULED"  # get_cond for a chunk included this target; that chunk is being generated
    GENERATED = "GENERATED"
    SHOWN = "SHOWN"
    DROPPED = "DROPPED"
    SUPERSEDED = "SUPERSEDED"
    UNCHANGED = "UNCHANGED"


TERMINAL = frozenset({CommandStage.SHOWN, CommandStage.DROPPED, CommandStage.SUPERSEDED, CommandStage.UNCHANGED})
IN_FLIGHT = frozenset(CommandStage) - TERMINAL

NEXT: dict[CommandStage, frozenset[CommandStage]] = {
    CommandStage.HEARD: frozenset({CommandStage.DECIDED, CommandStage.DROPPED}),
    CommandStage.DECIDED: frozenset({CommandStage.EMBEDDED, CommandStage.DROPPED, CommandStage.SUPERSEDED,
                                     CommandStage.UNCHANGED}),
    CommandStage.EMBEDDED: frozenset({CommandStage.SCHEDULED, CommandStage.DROPPED, CommandStage.SUPERSEDED}),
    CommandStage.SCHEDULED: frozenset({CommandStage.GENERATED, CommandStage.DROPPED}),
    CommandStage.GENERATED: frozenset({CommandStage.SHOWN, CommandStage.DROPPED}),
    CommandStage.SHOWN: frozenset(),
    CommandStage.DROPPED: frozenset(),
    CommandStage.SUPERSEDED: frozenset(),
    CommandStage.UNCHANGED: frozenset(),
}

# The stamp written when a record enters a stage.
STAMP_FOR_STAGE = {
    CommandStage.DECIDED: "decided",
    CommandStage.EMBEDDED: "embedded",
    CommandStage.GENERATED: "generated",
    CommandStage.SHOWN: "shown",
}


@dataclass
class CommandRecord:
    seq: int
    source: str                   # "control" (contract 1) or "target" (POST /target)
    text: str | None              # heard text; None for POST /target
    t_client: int | None
    stage: CommandStage = CommandStage.HEARD
    stamps_ns: dict[str, int] = field(default_factory=dict)
    target: dict | None = None    # the contract 2 message
    applies_from_chunk: int | None = None

    def stamps_ms(self) -> dict[str, int]:
        heard = self.stamps_ns["heard"]
        return {name: max(0, round((self.stamps_ns[name] - heard) / 1e6))
                for name in STAMP_NAMES if name in self.stamps_ns}


class CommandBook:
    def __init__(self, keep: int = 200) -> None:
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._records: collections.deque[CommandRecord] = collections.deque(maxlen=keep)
        self._latest: CommandRecord | None = None

    # --- transitions -------------------------------------------------------------------------

    def _move(self, rec: CommandRecord, new: CommandStage, now_ns: int, reason: str = "") -> None:
        if new not in NEXT[rec.stage]:
            raise RuntimeError(f"command {rec.seq}: {rec.stage.value} -> {new.value} is not allowed")
        old, rec.stage = rec.stage, new
        stamp = STAMP_FOR_STAGE.get(new)
        if stamp:
            rec.stamps_ns[stamp] = now_ns
        log.info("command %d: %s -> %s%s%s", rec.seq, old.value, new.value,
                 f" at +{rec.stamps_ms()[stamp]} ms" if stamp else "", f" ({reason})" if reason else "")

    def heard(self, source: str, text: str | None, t_client: int | None, heard_ns: int) -> CommandRecord:
        with self._lock:
            rec = CommandRecord(seq=next(self._seq), source=source, text=text, t_client=t_client,
                                stamps_ns={"heard": heard_ns})
            self._records.append(rec)
        log.info("command %d: HEARD %s %r", rec.seq, source, text)
        return rec

    def decided(self, rec: CommandRecord, target: dict) -> None:
        """The reader returned a steer (contract 2). This record becomes the most recent command."""
        with self._lock:
            rec.target = target
            self._move(rec, CommandStage.DECIDED, time.monotonic_ns())
            self._latest = rec

    def no_operation(self, rec: CommandRecord, reason: str) -> None:
        """The reader read the command and found no operation (an unmatched final, a tap with nothing to do, a reset
        to the starting prompt). It is still the most recent command, so contract 4 carries its t_client and its
        stamps, which are only `heard`: nothing was decided, embedded, generated or shown for it."""
        with self._lock:
            self._move(rec, CommandStage.DROPPED, time.monotonic_ns(), reason)
            self._latest = rec

    def drop(self, rec: CommandRecord, reason: str) -> None:
        """Drop a command without making it the most recent one (a reader rejection or error, a partial with no
        steer, a reset), or drop the most recent one where it stands."""
        with self._lock:
            if rec.stage not in TERMINAL:
                self._move(rec, CommandStage.DROPPED, time.monotonic_ns(), reason)

    def embedded(self, rec: CommandRecord, apply: Callable[[], dict]) -> dict | None:
        """The embedding is ready: apply the target unless a newer command's target is already set.

        `apply` is the steering Controller's handle for this command; it returns
        {"applied", "kind", "reason", "cache_hit"}. Returns that result, or None if `apply` was not called.
        """
        with self._lock:
            if rec.stage is not CommandStage.DECIDED:  # dropped by a reset while encoding
                return None
            newer_set = any(r.seq > rec.seq and r.stage in (CommandStage.EMBEDDED, CommandStage.SCHEDULED,
                                                              CommandStage.GENERATED, CommandStage.SHOWN)
                            for r in self._records)
            now = time.monotonic_ns()
            if newer_set:
                self._move(rec, CommandStage.SUPERSEDED, now, "a newer target was already set")
                return None
            result = apply()
            now = time.monotonic_ns()
            if not result["applied"]:
                reason = result["reason"]
                new = CommandStage.UNCHANGED if reason == "unchanged" else CommandStage.SUPERSEDED
                self._move(rec, new, now, f"controller: {reason}")
                return result
            self._move(rec, CommandStage.EMBEDDED, now, f"controller: {result['kind']}")
            for older in self._records:
                if older.seq < rec.seq and older.stage in (CommandStage.DECIDED, CommandStage.EMBEDDED):
                    self._move(older, CommandStage.SUPERSEDED, now, f"by command {rec.seq}")
            return result

    def take_cond(self, get_cond: Callable[[], object], chunk: int) -> object:
        """Call get_cond for `chunk` and schedule every EMBEDDED command onto that chunk."""
        with self._lock:
            cond = get_cond()
            for rec in self._records:
                if rec.stage is CommandStage.EMBEDDED:
                    rec.applies_from_chunk = chunk
                    self._move(rec, CommandStage.SCHEDULED, time.monotonic_ns(), f"chunk {chunk}")
            return cond

    def chunk_generated(self, chunk: int) -> None:
        with self._lock:
            now = time.monotonic_ns()
            for rec in self._records:
                if rec.stage is CommandStage.SCHEDULED and rec.applies_from_chunk == chunk:
                    self._move(rec, CommandStage.GENERATED, now)

    def chunk_shown(self, chunk: int) -> None:
        with self._lock:
            now = time.monotonic_ns()
            for rec in self._records:
                if rec.stage is CommandStage.GENERATED and rec.applies_from_chunk is not None \
                        and rec.applies_from_chunk <= chunk:
                    self._move(rec, CommandStage.SHOWN, now)

    def reset(self, replace_steerer: Callable[[], None], rec: CommandRecord | None = None,
              heard_up_to: int | None = None) -> None:
        """Drop the commands still in flight that the reset replaces, and replace the steerer and controller,
        atomically with respect to get_cond and handle.

        `rec` is the reset command itself when it carries a prompt: the new steerer starts on its embedding, so it
        moves to EMBEDDED here and the first chunk of the new video counts as its generated chunk.
        `heard_up_to` is the seq of the newest command the reset replaces (the reset's own command): commands heard
        after it are kept, because the engine holds them until the reset is done and then applies them to the new
        video. None drops every command in flight (recovery).
        """
        with self._lock:
            now = time.monotonic_ns()
            for other in self._records:
                if other is rec or other.stage in TERMINAL:
                    continue
                if heard_up_to is None or other.seq <= heard_up_to:
                    self._move(other, CommandStage.DROPPED, now, "reset")
            replace_steerer()
            if rec is not None and rec.stage is CommandStage.DECIDED:
                self._move(rec, CommandStage.EMBEDDED, time.monotonic_ns(), "new video starts on it")

    # --- reading -----------------------------------------------------------------------------

    def in_flight(self) -> bool:
        """True while any command is between HEARD and SHOWN: it still needs the reader, the encoder or the GPU."""
        with self._lock:
            return any(r.stage in IN_FLIGHT for r in self._records)

    def latest_fields(self) -> dict:
        """The contract 4 fields the engine fills from the most recent command: the newest one the reader read
        (a final, a tap, a partial that got a steer, or a POST /target), with or without an operation."""
        with self._lock:
            rec = self._latest
            if rec is None:
                return {"stamps": {}, "t_client": None, "applies_from_chunk": None}
            return {"stamps": rec.stamps_ms(), "t_client": rec.t_client, "applies_from_chunk": rec.applies_from_chunk}

    def newest_seq(self) -> int:
        """The seq of the newest command heard so far, 0 before the first."""
        with self._lock:
            return self._records[-1].seq if self._records else 0

    def latest(self) -> CommandRecord | None:
        with self._lock:
            return self._latest

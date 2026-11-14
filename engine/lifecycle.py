"""The engine's lifecycle as an explicit state machine."""

import enum
import logging
import threading

log = logging.getLogger("engine.lifecycle")


class Lifecycle(enum.Enum):
    LOADING = "LOADING"        # backend.load() is running
    WARMING = "WARMING"        # encoding the base prompt and the precompute file
    GENERATING = "GENERATING"  # the loop is producing chunks
    RESETTING = "RESETTING"    # between chunks: backend.reset() and a new steerer
    RECOVERING = "RECOVERING"  # a generation or decode error: clear the queues, backend.reset(), base prompt
    STOPPED = "STOPPED"        # clean shutdown finished or in progress
    FAILED = "FAILED"          # an unrecoverable error; see the log


ALLOWED: dict[Lifecycle, frozenset[Lifecycle]] = {
    Lifecycle.LOADING: frozenset({Lifecycle.WARMING, Lifecycle.STOPPED, Lifecycle.FAILED}),
    Lifecycle.WARMING: frozenset({Lifecycle.GENERATING, Lifecycle.STOPPED, Lifecycle.FAILED}),
    Lifecycle.GENERATING: frozenset({Lifecycle.RESETTING, Lifecycle.RECOVERING, Lifecycle.STOPPED, Lifecycle.FAILED}),
    Lifecycle.RESETTING: frozenset({Lifecycle.GENERATING, Lifecycle.RECOVERING, Lifecycle.STOPPED, Lifecycle.FAILED}),
    Lifecycle.RECOVERING: frozenset({Lifecycle.GENERATING, Lifecycle.STOPPED, Lifecycle.FAILED}),
    Lifecycle.FAILED: frozenset({Lifecycle.STOPPED}),
    Lifecycle.STOPPED: frozenset(),
}


class IllegalTransition(RuntimeError):
    pass


class LifecycleMachine:
    """Holds the current state; `transition` is the only way to change it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = Lifecycle.LOADING
        log.info("lifecycle: start in %s", self._state.value)

    @property
    def state(self) -> Lifecycle:
        with self._lock:
            return self._state

    def transition(self, new: Lifecycle, reason: str = "") -> None:
        """Move to `new`, logging it. Raises IllegalTransition if not allowed from the current state."""
        with self._lock:
            old = self._state
            if new not in ALLOWED[old]:
                raise IllegalTransition(f"{old.value} -> {new.value} is not allowed")
            self._state = new
        log.info("lifecycle: %s -> %s%s", old.value, new.value, f" ({reason})" if reason else "")

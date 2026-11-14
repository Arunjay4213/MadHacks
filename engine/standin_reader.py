"""Stand-in command reader, used until the commands/ owner's `commands.service.CommandService` is merged.

It has the same API (see engine/reader.py). A `final` text is taken as the complete new prompt with a 4-chunk ramp;
`partial` and `tap` give no steer. The changes list stays empty and there is no keyword.
"""

import logging
import threading

log = logging.getLogger("engine.standin_reader")

DEFAULT_RAMP_CHUNKS = 4


class StandinService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._heard: str | None = None
        self.committed_prompt: str | None = None  # None: the engine starts from --base-prompt

    def handle(self, message: dict) -> dict:
        kind = message.get("type")
        if kind not in ("partial", "final", "tap"):
            raise ValueError(f"unknown message type {kind!r}")
        steer = None
        with self._lock:
            if kind == "final":
                text = str(message.get("text", "")).strip()
                self._heard = text
                if text:
                    steer = {"prompt": text, "ramp_chunks": DEFAULT_RAMP_CHUNKS, "provisional": False, "cap": 1.0}
                    self.committed_prompt = text
            elif kind == "tap":
                log.info("stand-in reader: no operation for tap %s", message.get("action"))
            return {"steer": steer, "state": self._state()}

    def state_fields(self) -> dict:
        with self._lock:
            return self._state()

    def likely_prompts(self) -> list[str]:
        return []

    def _state(self) -> dict:
        return {"changes": [], "heard": self._heard, "keyword": None, "unmatched": False}

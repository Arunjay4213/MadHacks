"""Stand-in command reader, used until the commands/ owner's reader is merged.

The boundary is one function, `read_command(message, scene) -> contract 2 dict or None`.
Here a `final` text is taken as the complete new prompt; `partial` and `tap` give no operation.
The scene keeps an empty changes list and no keyword.
"""

import logging

log = logging.getLogger("engine.standin_reader")

DEFAULT_RAMP_CHUNKS = 4


class StandinScene:
    """The part of the scene state the engine reads for contract 4."""

    def __init__(self) -> None:
        self.changes: list[dict] = []    # each {"id", "label", "strength"}
        self.keyword: str | None = None  # the key word of the last command, for highlighting


def read_command(message: dict, scene: StandinScene) -> dict | None:
    kind = message["type"]
    if kind == "final":
        text = message["text"].strip()
        if not text:
            return None
        return {"prompt": text, "ramp_chunks": DEFAULT_RAMP_CHUNKS, "provisional": False, "cap": 1.0}
    if kind == "tap":
        log.info("stand-in reader: ignoring tap %s", message.get("action"))
    return None

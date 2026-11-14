"""Stable phrase order for the text encoder's token alignment."""

from .state import KINDS


def compose(snapshot):
    parts = [snapshot["base"]]
    for change in sorted(snapshot["changes"], key=lambda c: KINDS.index(c["kind"])):
        strength = change["strength"]
        if strength == 0:
            continue
        phrase = change["phrase"]
        if strength < 0.5:
            phrase = "a subtle hint of " + phrase
        elif strength < 1:
            phrase = "a moderate amount of " + phrase
        elif strength > 1:
            phrase = "intense " + phrase
        parts.append(phrase)
    return "; ".join(parts)

"""Validation of contract messages against contracts/schemas."""

import json

from contracts.validate import errors


class BadMessage(ValueError):
    pass


def parse_json(text: str | bytes) -> object:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BadMessage(f"not valid JSON: {exc}") from None


def check(contract: int, message: object) -> dict:
    """Return `message` if it is valid for `contract`, else raise BadMessage listing the problems."""
    problems = errors(contract, message)
    if problems:
        raise BadMessage(f"invalid contract {contract} message: " + "; ".join(problems))
    assert isinstance(message, dict)
    return message

"""Embedded English scene vocabulary, loaded and compiled once; no network/model call."""

import json
from pathlib import Path
import re
import unicodedata

from .state import KINDS


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return " ".join(text.split())


def load_dictionary(path: Path = Path(__file__).with_name("keywords.json")) -> dict[str, dict]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("keyword dictionary must be a nonempty array")
    keywords = {}
    labels = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("dictionary entry must be an object")
        label, kind, phrase = (item.get(key) for key in ("label", "kind", "phrase"))
        aliases = item.get("aliases", [])
        if not isinstance(label, str) or not normalize(label) or label != normalize(label) or label in labels:
            raise ValueError(f"invalid or repeated canonical label: {label!r}")
        if kind not in KINDS or not isinstance(phrase, str) or not phrase.strip():
            raise ValueError(f"invalid kind/phrase for {label}")
        if not isinstance(aliases, list) or any(not isinstance(alias, str) or not normalize(alias) for alias in aliases):
            raise ValueError(f"aliases must be nonempty strings for {label}")
        entry = dict(label=label, kind=kind, phrase=phrase.strip())
        for alias in [label, *aliases]:
            key = normalize(alias)
            if key in keywords and keywords[key]["label"] != label:
                raise ValueError(f"alias {key!r} names both {keywords[key]['label']} and {label}")
            keywords[key] = entry
        labels.add(label)
    return keywords


KEYWORDS = load_dictionary()
KEYWORD_RX = re.compile(r"\b(?:" + "|".join(re.escape(key) for key in sorted(KEYWORDS, key=lambda key: (-len(key), key))) + r")\b")

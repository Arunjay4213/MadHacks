"""Local semantic retrieval for final-command misses, with open-vocabulary additions."""

from collections import OrderedDict
from functools import lru_cache
import logging
import math
import os
import re
import threading
from pathlib import Path

from .dictionary import KEYWORD_RX, KEYWORDS, normalize
from .reader import remove_target, strength_for

SINGLE_CANDIDATE_MINIMUM = 0.70  # with one active change the margin test is meaningless

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = Path.home() / ".cache" / "madhacks" / "command-minilm"
log = logging.getLogger(__name__)


def _unit(vector):
    values = tuple(float(value) for value in vector)
    length = math.sqrt(sum(value * value for value in values))
    if not values or not math.isfinite(length) or length == 0:
        raise ValueError("embedding must be a finite, nonzero vector")
    return tuple(value / length for value in values)


class SimilarityIndex:
    def __init__(self, texts, labels, encode):
        import numpy as np
        if not texts or len(texts) != len(labels):
            raise ValueError("index needs one label per reference text")
        self._encode = encode
        self._vectors = [_unit(vector) for vector in encode(texts)]
        if len(self._vectors) != len(labels) or len({len(vector) for vector in self._vectors}) != 1:
            raise ValueError("reference embeddings have inconsistent dimensions")
        self._labels = labels
        self._matrix = np.asarray(self._vectors, dtype=np.float32)
        self._cache = OrderedDict()

    def match(self, text, minimum=0.55, margin=0.05):
        vector = self._cache.get(text)
        if vector is None:
            vector = _unit(self._encode([text])[0])
            if len(vector) != len(self._vectors[0]):
                raise ValueError("query embedding has the wrong dimension")
            self._cache[text] = vector
            if len(self._cache) > 256:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(text)
        scores = {}
        for label, score in zip(self._labels, self._matrix @ vector):
            scores[label] = max(scores.get(label, -1), score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        label, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1
        return label if best >= minimum and best - second >= margin else None


class SemanticReader:
    def __init__(self, encode, entries=None):
        self._encode = encode
        entries = KEYWORDS if entries is None else entries
        self._entries = {entry["label"]: entry for entry in entries.values()}
        texts, labels = [], []
        for alias, entry in entries.items():
            texts.append(alias)
            labels.append(entry["label"])
        for label, entry in self._entries.items():
            texts.append(entry["phrase"])
            labels.append(label)
        self._index = SimilarityIndex(texts, labels, encode)
        self._lock = threading.Lock()
        self._active_key = None
        self._active_index = None

    def __call__(self, text, snapshot):
        t = normalize(text)
        if not t or re.search(r"\b(don't|dont|do not|not|never|instead of|into|replace|swap)\b", t):
            return None
        target = remove_target(t)  # the reader's own remove verbs, so a remove is never read back as an add
        with self._lock:
            if target is not None:
                # A known-but-absent noun was deliberately rejected by the fast reader.
                if KEYWORD_RX.search(target):
                    return None
                changes = snapshot.get("changes", [])
                if not changes:
                    return None
                key = tuple((change["id"], change["label"], change["phrase"]) for change in changes)
                if key != self._active_key:
                    self._active_index = SimilarityIndex([f"{c['label']}: {c['phrase']}" for c in changes],
                                                         [c["id"] for c in changes], self._encode)
                    self._active_key = key
                minimum = SINGLE_CANDIDATE_MINIMUM if len(changes) == 1 else 0.60
                change_id = self._active_index.match(target, minimum=minimum, margin=0.10)
                return {"op": "remove", "id": change_id} if change_id else None
            label = self._index.match(t)
        if label is None:
            return None
        entry = self._entries[label]
        existing = next((c for c in snapshot.get("changes", []) if c["label"] == label), None)
        if existing:
            return {"op": "set_strength", "id": existing["id"], "strength": strength_for(t, existing["strength"])}
        return {"op": "add", **entry, "strength": strength_for(t)}


def local_reader(download=False):
    """Warm on CPU at startup. Live input never downloads a model or runs remote code."""
    from sentence_transformers import SentenceTransformer
    source = MODEL if download else str(MODEL_DIR)
    if not download and not (MODEL_DIR / "modules.json").is_file():
        raise FileNotFoundError(f"Prepare {MODEL_DIR} with python -m commands.semantic --prepare")
    model = SentenceTransformer(source, device="cpu", local_files_only=not download, trust_remote_code=False)
    if download:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(MODEL_DIR))

    def encode(texts):
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    return SemanticReader(encode)


@lru_cache(maxsize=1)
def configured_reader():
    if os.environ.get("COMMAND_SEMANTIC") != "1":
        return None
    try:
        return local_reader()
    except Exception as exc:
        log.warning("Semantic matcher disabled (%s); dictionary and slow reader remain available", exc)
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="download/cache and warm the CPU model before the demo")
    args = parser.parse_args()
    reader = local_reader(download=args.prepare)
    print(f"Ready: {MODEL}; {len(KEYWORDS)} dictionary references, CPU only")

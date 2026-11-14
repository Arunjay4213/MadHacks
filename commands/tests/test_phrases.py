"""Runs Codex's phrase fixtures (fixtures/phrases.json) through the fast reader. Format: commands/PLAN.md."""

import json
import pathlib
import unittest

from commands.reader import KEYWORDS, read
from commands.state import SceneState

CASES = json.loads((pathlib.Path(__file__).parent / "fixtures" / "phrases.json").read_text())
BY_LABEL = {e["label"]: e for e in KEYWORDS.values()}


class PhraseTest(unittest.TestCase):
    def test_fixture_is_big_enough(self):
        self.assertGreaterEqual(len(CASES), 40)

    def test_phrases(self):
        for case in CASES:
            with self.subTest(text=case["text"], before=case["before"]):
                state = SceneState()
                for label in case["before"]:
                    e = BY_LABEL[label]
                    self.assertTrue(state.apply({"op": "add", "label": label, "kind": e["kind"], "phrase": e["phrase"]}))
                snap = state.snapshot()
                r = read(case["text"], snap)
                expect = case["expect"]
                if expect is None:
                    self.assertIsNone(r["op"])
                    continue
                self.assertIsNotNone(r["op"])
                expect = dict(expect)
                if expect["op"] in ("remove", "set_strength"):  # label names the change whose id is expected
                    label = expect.pop("label")
                    expect["id"] = next(c["id"] for c in snap["changes"] if c["label"] == label)
                self.assertEqual({k: r["op"].get(k) for k in expect}, expect)
                if "pace" in case:
                    self.assertEqual(r["pace"], case["pace"])
                # every op the reader produces must be one the state accepts
                state.apply(r["op"])


if __name__ == "__main__":
    unittest.main()

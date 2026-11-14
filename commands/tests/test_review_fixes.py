"""Review fixes for the semantic fallback and compound service (fake encoder: no model download)."""

import unittest
from importlib.util import find_spec
import hashlib

from commands.semantic import SemanticReader
from commands.service import CommandService

# A tiny deterministic "embedding": bag of character trigrams hashed into 64 dims.
def fake_encode(texts):
    out = []
    for text in texts:
        v = [0.0] * 64
        t = f"  {text.lower()}  "
        for i in range(len(t) - 2):
            v[hashlib.sha256(t[i:i + 3].encode()).digest()[0] % 64] += 1.0
        out.append(v)
    return out


def scene(*changes):
    return {"base": "b", "can_undo": False, "changes": [
        {"id": f"c{i}", "label": l, "kind": "extra", "phrase": p, "strength": 1.0} for i, (l, p) in enumerate(changes, 1)]}


@unittest.skipUnless(find_spec("numpy"), "optional vector tests need numpy")
class SemanticSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = SemanticReader(fake_encode)

    def test_remove_wording_never_becomes_an_add(self):
        for text in ["without the river", "drop the dog", "lose the clouds", "stop the rain", "take away the boat"]:
            op = self.reader(text, scene())
            self.assertTrue(op is None or op["op"] == "remove", (text, op))

    def test_single_candidate_removal_needs_a_strong_match(self):
        # One active change: the margin is meaningless, so a weak phrase must not remove it.
        self.assertIsNone(self.reader("remove the thing over there", scene(("sunset", "a golden sunset sky"))))


class CompoundServiceTest(unittest.TestCase):
    def test_model_exception_falls_through_to_llm(self):
        def broken(text, snapshot):
            raise RuntimeError("encoder unavailable")
        service = CommandService(semantic_reader=broken, slow_reader=lambda text, snapshot: {
            "op": "add", "label": "rain", "kind": "weather", "phrase": "pouring rain"})
        result = service.handle({"type": "final", "text": "let it pour", "t_client": 1})
        self.assertEqual([c["label"] for c in result["state"]["changes"]], ["rain"])


    def test_style_fallback_is_last_resort_after_semantic_and_slow(self):
        svc = CommandService(semantic_reader=lambda text, snap: None)
        r = svc.handle({"type": "final", "text": "make it spooky and add a dog", "t_client": 1})
        self.assertEqual([c["label"] for c in r["state"]["changes"]], ["spooky", "dog"])
        self.assertIn("the whole scene looks spooky", r["steer"]["prompt"])
        # A semantic answer wins over the style fallback.
        svc = CommandService(semantic_reader=lambda text, snap: {"op": "add", "label": "fog", "kind": "weather",
                                                                  "phrase": "a thick fog in the air", "strength": 1.0})
        r = svc.handle({"type": "final", "text": "make it gloomy", "t_client": 1})
        self.assertEqual([c["label"] for c in r["state"]["changes"]], ["fog"])

    def test_negated_clause_still_rolls_back_the_sentence(self):
        svc = CommandService(semantic_reader=lambda text, snap: None)
        r = svc.handle({"type": "final", "text": "add a dog and don't make it spooky", "t_client": 1})
        self.assertTrue(r["state"]["unmatched"])
        self.assertEqual(r["state"]["changes"], [])


if __name__ == "__main__":
    unittest.main()

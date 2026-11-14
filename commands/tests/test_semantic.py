import unittest
from importlib.util import find_spec
from unittest.mock import patch

from commands.semantic import configured_reader

from commands.semantic import SemanticReader, SimilarityIndex
from commands.service import CommandService


def encode(texts):
    vectors = []
    for text in texts:
        if any(term in text.lower() for term in ("winter", "freezing", "snow", "subzero")):
            vectors.append([1, 0, 0])
        elif any(term in text.lower() for term in ("dog", "puppy", "fluffy companion")):
            vectors.append([0, 1, 0])
        else:
            vectors.append([0, 0, 1])
    return vectors


class ConfigurationTests(unittest.TestCase):
    def tearDown(self):
        configured_reader.cache_clear()

    def test_disabled_never_loads_dependencies(self):
        with patch.dict("os.environ", {"COMMAND_SEMANTIC": "0"}), patch("commands.semantic.local_reader") as load:
            self.assertIsNone(configured_reader())
            load.assert_not_called()

    def test_missing_model_keeps_dictionary_and_slow_reader_available(self):
        configured_reader.cache_clear()
        with patch.dict("os.environ", {"COMMAND_SEMANTIC": "1"}), patch("commands.semantic.local_reader", side_effect=FileNotFoundError("not prepared")):
            with self.assertLogs("commands.semantic", level="WARNING"):
                self.assertIsNone(configured_reader())
            service = CommandService(slow_reader=lambda text, snapshot: {"op": "add", "label": "lion", "kind": "extra", "phrase": "a lion"})
            self.assertFalse(service.handle(dict(type="final", text="make it winter", t_client=1))["state"]["unmatched"])
            result = service.handle(dict(type="final", text="turn the tree into a lion", t_client=2))
            self.assertEqual([c["label"] for c in result["state"]["changes"]], ["winter", "lion"])


@unittest.skipUnless(find_spec("numpy"), "optional vector tests need numpy")
class VectorIndexTests(unittest.TestCase):
    def test_cosine_match_rejects_ties_and_low_confidence(self):
        index = SimilarityIndex(["winter", "dog"], ["winter", "dog"], encode)
        self.assertEqual(index.match("subzero weather"), "winter")
        self.assertIsNone(index.match("account password"))
        tie = SimilarityIndex(["winter", "freezing"], ["a", "b"], encode)
        self.assertIsNone(tie.match("subzero"))

    def test_bad_vectors_fail_instead_of_selecting_a_random_target(self):
        for vector in ([0, 0], [float("nan"), 1], [float("inf"), 1]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                SimilarityIndex(["a"], ["a"], lambda texts: [vector])


@unittest.skipUnless(find_spec("numpy"), "optional vector tests need numpy")
class SemanticReaderTests(unittest.TestCase):
    def setUp(self):
        self.reader = SemanticReader(encode, entries={
            "winter": dict(label="winter", kind="season", phrase="winter snow"),
            "dog": dict(label="dog", kind="extra", phrase="a dog"),
        })

    def test_paraphrase_maps_to_a_supported_meaning(self):
        op = self.reader("make the weather subzero", {"base": "a meadow", "changes": []})
        self.assertEqual(op, dict(op="add", label="winter", kind="season", phrase="winter snow", strength=1.0))

    def test_new_object_is_preserved_rather_than_forced_to_a_known_neighbor(self):
        service = CommandService(semantic_reader=self.reader)
        result = service.handle(dict(type="final", text="add a glowing unicorn here", t_client=1))
        self.assertEqual(result["state"]["changes"][0]["label"], "glowing unicorn")
        self.assertIn("glowing unicorn", result["steer"]["prompt"])

    def test_semantic_removal_can_only_select_an_active_change(self):
        snapshot = {"base": "a meadow", "changes": [dict(id="c1", label="dog", phrase="a dog", kind="extra", strength=1)]}
        self.assertEqual(self.reader("remove the fluffy companion", snapshot), dict(op="remove", id="c1"))
        self.assertIsNone(self.reader("remove the fluffy companion", {"base": "a meadow", "changes": []}))
        self.assertIsNone(self.reader("remove my account password", snapshot))

    def test_negation_and_unrelated_text_never_become_an_edit(self):
        for text in ("don't make it freezing", "do not add a unicorn", "account password", "not subzero"):
            self.assertIsNone(self.reader(text, {"base": "b", "changes": []}), text)

    def test_service_uses_semantics_after_fast_miss_and_keeps_undo(self):
        service = CommandService(base="meadow", semantic_reader=self.reader)
        result = service.handle(dict(type="final", text="make the weather subzero", t_client=1))
        self.assertFalse(result["state"]["unmatched"])
        self.assertEqual(result["state"]["changes"][0]["label"], "winter")
        undone = service.handle(dict(type="tap", action="undo"))
        self.assertEqual(undone["state"]["changes"], [])


if __name__ == "__main__":
    unittest.main()

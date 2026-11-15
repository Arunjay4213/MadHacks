import unittest
from importlib.util import find_spec
from unittest.mock import patch

from commands.semantic import configured_reader

from commands.semantic import SemanticReader, SimilarityIndex
from commands.reader import RAMP
from commands.service import CommandService
from commands.world import CAR_BASE


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


def nocturnal_encode(texts):
    """'nocturnal' sits next to the dictionary's night entry; every other text is far from it and from each other."""
    from commands.dictionary import KEYWORDS
    night = {alias for alias, entry in KEYWORDS.items() if entry["label"] == "night"} | {KEYWORDS["night"]["phrase"]}
    return [[1, 0, 0] if text in night or "nocturnal" in text else [0, 1 + len(text) % 7, 1 + len(text) % 5] for text in texts]


class StyleWordsTests(unittest.TestCase):
    """Strength and pace words on the style path follow reader.strength_for, as on the dictionary path.

    The style path is plain Python, so these run without NumPy, with no semantic reader.
    The vector subclass below runs them again with a semantic reader in front of the style path.
    """

    def make_semantic_reader(self):
        return lambda *a: None  # "no semantic reader"; None would load the configured model

    def service(self, base=None):
        return CommandService(base, slow_reader=lambda *a: None, semantic_reader=self.make_semantic_reader())

    def run_sentences(self, *sentences):
        service = self.service()
        for i, text in enumerate(sentences):
            result = service.handle(dict(type="final", text=text, t_client=i + 1))
            self.assertFalse(result["state"]["unmatched"], text)
        return [(c["label"], c["strength"]) for c in result["state"]["changes"]]

    def check_pace(self, cases):
        # Gate 4: "slowly make it winter" must reach full winter, only more slowly (12 chunks instead of 1).
        for base in (None, CAR_BASE):
            for text, label in cases:
                with self.subTest(base=base, text=text):
                    result = self.service(base).handle(dict(type="final", text=text, t_client=1))
                    self.assertEqual([(c["label"], c["strength"]) for c in result["state"]["changes"]], [(label, 1.0)])
                    self.assertEqual(result["steer"]["ramp_chunks"], RAMP["slow"])
                    self.assertEqual(RAMP["slow"], 12)
                    self.assertNotIn("moderate amount", result["steer"]["prompt"])

    def test_style_path_strips_strength_words_and_sets_strength(self):
        self.assertEqual(self.run_sentences("make it a little spooky"), [("spooky", 0.3)])
        self.assertEqual(self.run_sentences("make it a lot spooky"), [("spooky", 1.3)])
        self.assertEqual(self.run_sentences("make it less spooky"), [("spooky", 1.0)])  # no chip named "less spooky"
        self.assertEqual(self.run_sentences("make it spooky", "make it much more spooky"), [("spooky", 1.3)])
        self.assertEqual(self.run_sentences("make it spooky", "make it more spooky"), [("spooky", 1.3)])
        self.assertEqual(self.run_sentences("make it spooky", "make it less spooky"), [("spooky", 0.7)])
        self.assertEqual(self.run_sentences("make it spooky", "make it a little less spooky"), [("spooky", 0.7)])

    def test_pace_words_set_the_ramp_and_never_the_strength(self):
        self.check_pace([("slowly make it winter", "winter"), ("gradually make it winter", "winter"),
                         ("gently make it winter", "winter"), ("little by little make it winter", "winter"),
                         ("make it winter slowly", "winter"),
                         ("slowly make it spooky", "spooky"), ("make it spooky slowly", "spooky")])

    def test_style_strength_change_is_one_undo(self):
        service = self.service()
        service.handle(dict(type="final", text="make it spooky", t_client=1))
        service.handle(dict(type="final", text="make it less spooky", t_client=2))
        undone = service.handle(dict(type="tap", action="undo"))
        self.assertEqual([(c["label"], c["strength"]) for c in undone["state"]["changes"]], [("spooky", 1.0)])


@unittest.skipUnless(find_spec("numpy"), "optional vector tests need numpy")
class StrengthWordsTests(StyleWordsTests):
    """Strength and pace words on the vector path; the inherited style tests run again with the vector reader."""

    def make_semantic_reader(self):
        return SemanticReader(nocturnal_encode)

    def test_vector_path_reads_strength_words(self):
        self.assertEqual(self.run_sentences("make it a little nocturnal"), [("night", 0.3)])
        self.assertEqual(self.run_sentences("make it a little night"), [("night", 0.3)])  # the dictionary path agrees
        self.assertEqual(self.run_sentences("make it a lot nocturnal"), [("night", 1.3)])
        self.assertEqual(self.run_sentences("make it nocturnal", "make it much more nocturnal"), [("night", 1.3)])
        self.assertEqual(self.run_sentences("make it nocturnal", "make it less nocturnal"), [("night", 0.7)])

    def test_vector_pace_words_set_the_ramp_and_never_the_strength(self):
        self.check_pace([("slowly make it nocturnal", "night")])


if __name__ == "__main__":
    unittest.main()

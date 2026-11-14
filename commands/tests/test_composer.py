import copy
import unittest

from commands.composer import compose


class ComposerTests(unittest.TestCase):
    def test_fixed_order_and_wording_survive_middle_removal(self):
        state = {"base": "a meadow", "changes": [
            dict(kind="extra", phrase="a river", strength=1),
            dict(kind="time", phrase="moonlit night", strength=1),
            dict(kind="season", phrase="winter scenery", strength=1),
        ]}
        before = copy.deepcopy(state)
        self.assertEqual(compose(state), "a meadow; winter scenery; moonlit night; a river")
        self.assertEqual(state, before)
        state["changes"].pop(1)
        self.assertEqual(compose(state), "a meadow; winter scenery; a river")

    def test_strength_wording_and_zero_effect(self):
        for strength, phrase in [(0, "a meadow"), (0.3, "a meadow; a subtle hint of snow"),
                                 (0.6, "a meadow; a moderate amount of snow"),
                                 (1, "a meadow; snow"), (1.3, "a meadow; intense snow")]:
            with self.subTest(strength=strength):
                self.assertEqual(compose({"base": "a meadow", "changes": [
                    dict(kind="weather", phrase="snow", strength=strength)]}), phrase)

    def test_extras_keep_their_insertion_order(self):
        self.assertEqual(compose({"base": "a meadow", "changes": [
            dict(kind="extra", phrase="a river", strength=1),
            dict(kind="extra", phrase="a bridge", strength=1)]}), "a meadow; a river; a bridge")


if __name__ == "__main__":
    unittest.main()

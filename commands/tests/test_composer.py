import copy
import unittest

from commands.composer import compose
from commands.state import DEFAULT_BASE
from commands.world import WORLDS


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

    def test_intense_goes_after_the_article(self):
        # "intense a large lion" is not English: the adjective goes after the article, which agrees with it.
        for phrase, worded in [("a large lion at the left", "an intense large lion at the left"),
                               ("an old bridge", "an intense old bridge"), ("the river", "the intense river")]:
            with self.subTest(phrase=phrase):
                self.assertEqual(compose({"base": "a meadow", "changes": [
                    dict(kind="extra", phrase=phrase, strength=1.3)]}), f"a meadow; {worded}")

    def test_extras_keep_their_insertion_order(self):
        self.assertEqual(compose({"base": "a meadow", "changes": [
            dict(kind="extra", phrase="a river", strength=1),
            dict(kind="extra", phrase="a bridge", strength=1)]}), "a meadow; a river; a bridge")


def world(name, strength=1.0):
    return dict(kind="world", label=WORLDS[name]["label"], phrase=WORLDS[name]["phrase"], strength=strength)


ROAD_WORDS = ("road", "highway", "asphalt", "lane")


class WorldFormTests(unittest.TestCase):
    """A known world has a road only when the scene has a vehicle (pinning run, 2026-11-15: a highway on the meadow)."""

    def test_a_scene_with_no_vehicle_gets_the_landscape(self):
        for name, says in [("desert", "a vast open desert"), ("beach", "a wide sandy beach"),
                           ("snowy mountains", "snowy mountains"), ("forest", "a dense green pine forest"),
                           ("city", "a big city at night"), ("mars", "the red rocky surface of Mars")]:
            with self.subTest(name=name):
                prompt = compose({"base": DEFAULT_BASE, "changes": [world(name)]})
                self.assertIn(says, prompt)
                self.assertFalse([w for w in ROAD_WORDS if w in prompt.lower()], prompt)

    def test_a_vehicle_object_or_base_keeps_the_road(self):
        car = dict(kind="extra", label="red car", phrase="a red car parked beside the oak tree", strength=1.0)
        self.assertIn("a long straight desert highway", compose({"base": DEFAULT_BASE, "changes": [car, world("desert")]}))
        self.assertEqual(compose({"base": "a red sports car", "changes": [world("desert")]}),
                         "a red sports car; " + WORLDS["desert"]["phrase"].rstrip("."))

    def test_the_landscape_drops_the_same_built_in_sky(self):
        night = dict(kind="time", label="night", phrase="at night", strength=1.0)
        prompt = compose({"base": DEFAULT_BASE, "changes": [world("desert"), night]})
        self.assertNotIn("afternoon sun", prompt)
        self.assertIn("heat haze shimmering above the sand", prompt)

    def test_a_world_strength_is_worded(self):
        # "a little more snow" on the car in winter (the winter world at 1.3) must not compose to the same prompt as
        # winter at 1.0: the engine keeps one rich prompt per composed prompt.
        winter = {"base": "a red sports car", "changes": [world("snowy mountain road")]}
        more = {"base": "a red sports car", "changes": [world("snowy mountain road", 1.3)]}
        self.assertNotEqual(compose(winter), compose(more))
        self.assertIn("an intense snowy mountain road", compose(more))


if __name__ == "__main__":
    unittest.main()

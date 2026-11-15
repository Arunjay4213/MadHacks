"""Greptile PR #70 leftovers: a location word never replaces the asked-for change, and "without" is a removal."""

import unittest

from commands.reader import split_commands
from commands.service import CommandService


def say(*lines):
    s = CommandService(semantic_reader=lambda t, x: None)
    for t in lines:
        r = s.handle({"type": "final", "text": t, "t_client": 1})
    return [c["label"] for c in r["state"]["changes"]]


class LocationDoesNotReplaceTheEditTest(unittest.TestCase):
    def test_snow_near_a_present_river_adds_snow(self):
        self.assertEqual(say("add a river", "make it snow near the river"), ["river", "winter"])
        self.assertEqual(say("add a river", "add snow near the river"), ["river", "winter"])

    def test_snow_near_a_river_with_no_river_adds_only_snow(self):
        self.assertEqual(say("make it snow near the river"), ["winter"])

    def test_change_named_only_after_a_preposition_still_works(self):
        self.assertEqual(say("turn it to night"), ["night"])

    def test_change_after_a_preposition_beats_a_later_location(self):
        self.assertEqual(say("turn it to night near the river"), ["night"])
        self.assertEqual(say("add a river", "turn it to night near the river"), ["river", "night"])
        self.assertEqual(say("add a dog", "set it to night with the dog"), ["dog", "night"])


class WithoutIsARemovalTest(unittest.TestCase):
    def test_add_a_dog_without_the_rain_does_both(self):
        self.assertEqual(say("add rain", "add a dog without the rain"), ["dog"])

    def test_without_an_absent_change_rejects_the_whole_sentence(self):
        # Same as "add a dog and remove the rain" with no rain: all or nothing, never a partial edit.
        self.assertEqual(say("add a dog without the rain"), [])
        self.assertEqual(say("add a dog and remove the rain"), [])

    def test_a_scene_description_with_without_stays_one_clause(self):
        self.assertEqual(split_commands("take me to a beach without people"), ["take me to a beach without people"])


class AliasRemovalTest(unittest.TestCase):
    """Already right on main before this file; kept so a future reader change cannot bring the bug back."""

    def test_light_house_removes_the_lighthouse_not_the_house(self):
        self.assertEqual(say("add a house", "add a lighthouse", "remove the light house"), ["house"])

    def test_rain_bow_removes_the_rainbow_not_the_rain(self):
        self.assertEqual(say("add rain", "add a rainbow", "remove the rain bow"), ["rain"])


if __name__ == "__main__":
    unittest.main()

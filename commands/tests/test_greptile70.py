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
    def test_a_location_never_replaces_the_asked_for_change(self):
        for lines, want in [
            (["add a river", "make it snow near the river"], ["river", "winter"]),
            (["add a river", "add snow near the river"], ["river", "winter"]),
            (["make it snow near the river"], ["winter"]),  # no river: only the snow
            (["turn it to night"], ["night"]),  # the change is named only after a preposition
            (["turn it to night near the river"], ["night"]),  # and it beats a later location
            (["add a river", "turn it to night near the river"], ["river", "night"]),
            (["add a dog", "set it to night with the dog"], ["dog", "night"]),
        ]:
            with self.subTest(lines=lines):
                self.assertEqual(say(*lines), want)


class WithoutIsARemovalTest(unittest.TestCase):
    def test_without_removes_and_is_all_or_nothing(self):
        # Same as "add a dog and remove the rain" with no rain: all or nothing, never a partial edit.
        for lines, want in [
            (["add rain", "add a dog without the rain"], ["dog"]),
            (["add a dog without the rain"], []),
            (["add a dog and remove the rain"], []),
        ]:
            with self.subTest(lines=lines):
                self.assertEqual(say(*lines), want)

    def test_a_scene_description_with_without_stays_one_clause(self):
        self.assertEqual(split_commands("take me to a beach without people"), ["take me to a beach without people"])


class AliasRemovalTest(unittest.TestCase):
    """Already right on main before this file; kept so a future reader change cannot bring the bug back."""

    def test_a_split_compound_word_removes_the_compound_not_its_part(self):
        for lines, want in [
            (["add a house", "add a lighthouse", "remove the light house"], ["house"]),
            (["add rain", "add a rainbow", "remove the rain bow"], ["rain"]),
        ]:
            with self.subTest(lines=lines):
                self.assertEqual(say(*lines), want)


if __name__ == "__main__":
    unittest.main()

"""Greptile PR #252: with Sonnet down, a pace word or a place on an existing weather or season change still adjusts it."""

import unittest

from commands.reader import RAMP
from commands.service import CommandService


def say(*lines):
    """Chips (label, strength) and the ramp of the last sentence, on the dictionary path (no Sonnet)."""
    s = CommandService(semantic_reader=lambda t, x: None)
    for t in lines:
        r = s.handle({"type": "final", "text": t, "t_client": 1})
    chips = [(c["label"], c["strength"]) for c in r["state"]["changes"]]
    return chips, (r["steer"] or {}).get("ramp_chunks")


class PaceOnAnExistingChangeTest(unittest.TestCase):
    def test_slowly_lowers_the_strength_over_a_slow_ramp(self):
        for lines, chips in [
            (["make it winter", "slowly add a little snow"], [("winter", 0.3)]),
            (["make it winter", "slowly make it a little snowy"], [("winter", 0.3)]),
            (["make it winter", "gradually less snow"], [("winter", 0.7)]),
            (["add rain", "slowly add a little rain"], [("rain", 0.3)]),
        ]:
            with self.subTest(lines=lines):
                self.assertEqual(say(*lines), (chips, RAMP["slow"]))

    def test_the_same_sentence_without_slowly_gives_the_same_strength_at_once(self):
        self.assertEqual(say("make it winter", "add a little snow"), ([("winter", 0.3)], RAMP["default"]))

    def test_a_place_on_a_whole_scene_change_is_not_a_rejection(self):
        self.assertEqual(say("add a river", "make it winter", "make it a little snowy near the river"),
                         ([("river", 1.0), ("winter", 0.3)], RAMP["default"]))

    def test_every_request_form_adjusts_an_existing_change(self):
        for lines, chips in [
            (["make it winter", "give me a little snow"], [("winter", 0.3)]),
            (["make it winter", "i want more snow"], [("winter", 1.3)]),
            (["make it winter", "i’d like less snow"], [("winter", 0.7)]),
            (["make it winter", "add a little snow near the river"], [("winter", 0.3)]),
        ]:
            with self.subTest(lines=lines):
                self.assertEqual(say(*lines), (chips, RAMP["default"]))

    def test_an_unsupported_description_is_still_unmatched(self):
        # "black" and "red" are changes the table cannot write: nothing changes, slowly or not, strength kept.
        for lines, chips in [
            (["add a dog", "make the dog black"], [("dog", 1.0)]),
            (["add a dog", "slowly make the dog black"], [("dog", 1.0)]),
            (["add rain", "a little rain", "turn it to red rain"], [("rain", 0.3)]),
        ]:
            with self.subTest(lines=lines):
                self.assertEqual(say(*lines), (chips, None))


if __name__ == "__main__":
    unittest.main()

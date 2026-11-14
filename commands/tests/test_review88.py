"""Greptile PR #88 findings: mood words are not places, zero-strength worlds/times, apply_final pace."""

import unittest

from commands.composer import compose
from commands.reader import RAMP
from commands.service import CommandService
from commands.world import WORLDS, world_op, world_with_overrides

DESERT = WORLDS["desert"]


def svc():
    return CommandService(semantic_reader=lambda t, s: None)


def labels(result):
    return [c["label"] for c in result["state"]["changes"]]


class MoodIsNotAPlaceTest(unittest.TestCase):
    def test_make_it_a_mood_is_not_a_world(self):
        for text in ["make it a little spooky", "make it a bit eerie", "make it an odd one"]:
            self.assertIsNone(world_op(text), text)
        self.assertEqual(world_op("make it a forest")["label"], "forest")  # known places still work
        self.assertIsNone(world_op("make it a winter wonderland"))           # season meaning kept
        self.assertEqual(world_op("take me to an abandoned train station")["label"], "abandoned train station")

    def test_known_place_with_time_is_world_plus_time_and_keeps_objects(self):
        s = svc()
        s.handle({"type": "final", "text": "add a red car driving forward", "t_client": 1})
        r = s.handle({"type": "final", "text": "take me to the beach at night", "t_client": 2})
        self.assertEqual(labels(r), ["car", "ocean coast", "night"])
        self.assertIn("red car driving forward", s.committed_prompt)
        u = s.handle({"type": "tap", "action": "undo"})
        self.assertEqual(labels(u), ["car"])  # one undo step

    def test_known_place_with_time_inside_a_compound_sentence(self):
        s = svc()
        r = s.handle({"type": "final", "text": "take me to the desert at night and add a dog", "t_client": 1})
        self.assertEqual(labels(r), ["desert", "night", "dog"])


class ZeroStrengthTest(unittest.TestCase):
    def test_zero_strength_world_does_not_replace_the_base(self):
        snap = {"base": "a wide shot of a meadow with a single oak tree", "changes": [
            {"id": "c1", "label": "desert", "kind": "world", "phrase": DESERT["phrase"], "strength": 0}]}
        self.assertEqual(compose(snap), "a wide shot of a meadow with a single oak tree")

    def test_zero_strength_time_does_not_strip_the_world_sky(self):
        desert = {"id": "c1", "label": "desert", "kind": "world", "phrase": DESERT["phrase"], "strength": 1.0}
        night0 = {"id": "c2", "label": "night", "kind": "time", "phrase": "night", "strength": 0}
        self.assertEqual(world_with_overrides(desert, [desert, night0]), DESERT["phrase"])


class ApplyFinalPaceTest(unittest.TestCase):
    DOG = {"op": "add", "label": "dog", "kind": "extra", "phrase": "a dog standing nearby", "strength": 1.0}

    def test_pace_from_text_without_prompt(self):
        self.assertEqual(svc().apply_final("slowly add a dog", [self.DOG])["steer"]["ramp_chunks"], RAMP["slow"])
        self.assertEqual(svc().apply_final("add a dog", [self.DOG])["steer"]["ramp_chunks"], RAMP["default"])

    def test_pace_from_text_with_prompt_override(self):
        r = svc().apply_final("slowly add a dog", [self.DOG], prompt="A dog arrives in the meadow.")
        self.assertEqual((r["steer"]["prompt"], r["steer"]["ramp_chunks"]), ("A dog arrives in the meadow.", RAMP["slow"]))


if __name__ == "__main__":
    unittest.main()

import unittest

from commands.state import DEFAULT_BASE, KINDS
from commands.world import (DEFAULT_WORLD, STYLE, SUBJECT, WORLDS, compose_base, world_op, world_phrase,
                            world_with_overrides)

ISHAAN = {  # exact tested strings, board #6885673339
    "country road": "a two-lane country road through green fields and trees under a clear blue summer sky, bright sunshine.",
    "desert": "a long straight desert highway between red sandstone mesas and sand dunes, heat haze, orange sunset sky.",
    "snowy mountain road": "a snowy mountain road between pine forests covered in thick white snow, snow falling, cold blue overcast light.",
    "neon city": "a wet city street at night in the rain, tall skyscrapers with glowing neon signs on both sides, reflections on the asphalt.",
    "ocean coast": "a coastal cliff road beside a bright turquoise ocean with waves and palm trees, sunny tropical day.",
    "mars": "a dusty road across the red rocky surface of Mars, craters and red mountains, a pale pink sky with two small moons.",
}


def label(text):
    op = world_op(text)
    return op and op["label"]


class TableTest(unittest.TestCase):
    def test_phrases_are_ishaans_verbatim(self):
        by_label = {e["label"]: e["phrase"] for e in WORLDS.values()}
        self.assertEqual({k: v for k, v in by_label.items() if k in ISHAAN}, ISHAAN)
        self.assertEqual(set(by_label) - set(ISHAAN), {"forest", "meadow"})  # added per lead / oak default
        self.assertEqual(DEFAULT_WORLD["phrase"], ISHAAN["country road"])
        self.assertTrue(SUBJECT.startswith("A red sports car") and SUBJECT.endswith(", "))
        self.assertTrue(STYLE.startswith(" Cinematic") and STYLE.endswith("."))
        self.assertEqual(SUBJECT + ISHAAN["desert"] + STYLE,
                         "A red sports car driving fast along a road, seen from behind in a smooth tracking shot, the car in the "
                         "centre of the frame, a long straight desert highway between red sandstone mesas and sand dunes, heat "
                         "haze, orange sunset sky. Cinematic, highly detailed, realistic, sharp focus, motion blur on the road.")

    def test_entries_have_the_change_shape(self):
        for alias, entry in WORLDS.items():
            self.assertEqual(set(entry), {"label", "kind", "phrase"}, alias)
            self.assertEqual(entry["kind"], "world")


class RecognitionTest(unittest.TestCase):
    def test_setting_verbs_with_known_places(self):
        for text, want in [("make it a forest", "forest"), ("take me back to the meadow", "meadow"), ("make it a desert", "desert"), ("Take me to the beach!", "ocean coast"), ("go to mars", "mars"),
                           ("set it in a neon city", "neon city"), ("move us to the snowy mountains", "snowy mountain road"),
                           ("change the setting to the coast", "ocean coast"), ("switch to the city", "neon city"),
                           ("take me back to the country road", "country road"), ("make it a desert please", "desert")]:
            self.assertEqual(label(text), want, text)

    def test_op_shape(self):
        self.assertEqual(world_op("make it a desert"),
                         {"op": "add", "label": "desert", "kind": "world", "phrase": ISHAAN["desert"], "strength": 1.0})

    def test_only_setting_verbs_fire(self):
        for text in ["add a desert", "the desert is nice", "desert", "add a city", "remove the desert", "a beach towel"]:
            self.assertIsNone(world_op(text), text)

    def test_season_weather_time_and_objects_are_not_hijacked(self):
        for text in ["make it winter", "make it a winter wonderland", "make it night", "make it a dog", "make it a red car",
                     "make it rainy", "make it a desert at night", "make it a sunny beach"]:
            self.assertIsNone(world_op(text), text)

    def test_arbitrary_places_fall_back_to_the_spoken_place(self):
        self.assertEqual(world_op("take me to an abandoned train station"),
                         {"op": "add", "label": "abandoned train station", "kind": "world",
                          "phrase": "an abandoned train station.", "strength": 1.0})
        self.assertEqual(label("set it in a misty bamboo grove"), None)  # "misty" is a weather word: not a pure place
        self.assertEqual(label("move us to the old harbour"), "old harbour")

    def test_bounds_negation_transform_and_vague(self):
        for text in ["don't take me to the desert", "make it a desert instead of a city", "take me to",
                     "take me to a very long and winding and complicated sort of place", "make it a nice one", "go to it"]:
            self.assertIsNone(world_op(text), text)


def change(label, kind, phrase="p"):
    return {"id": label, "label": label, "kind": kind, "phrase": phrase, "strength": 1.0}


class OverridesTest(unittest.TestCase):
    def test_no_conflict_keeps_the_tested_phrase_verbatim(self):
        desert = change("desert", "world", ISHAAN["desert"])
        self.assertEqual(world_with_overrides(desert, [desert]), ISHAAN["desert"])
        self.assertEqual(world_with_overrides(desert, [desert, change("dog", "extra"), change("sunset", "time")]), ISHAAN["desert"])

    def test_explicit_time_weather_season_drop_only_conflicting_builtin_sky(self):
        desert = change("desert", "world", ISHAAN["desert"])
        self.assertEqual(world_with_overrides(desert, [desert, change("night", "time")]),
                         "a long straight desert highway between red sandstone mesas and sand dunes, heat haze.")
        self.assertEqual(world_with_overrides(desert, [desert, change("winter", "season")]),
                         "a long straight desert highway between red sandstone mesas and sand dunes, orange sunset sky.")
        city = change("neon city", "world", ISHAAN["neon city"])
        self.assertEqual(world_with_overrides(city, [city, change("day", "time"), change("fog", "weather")]),
                         "a wet city street, tall skyscrapers with glowing neon signs on both sides, reflections on the asphalt.")
        snowy = change("snowy mountain road", "world", ISHAAN["snowy mountain road"])
        self.assertEqual(world_with_overrides(snowy, [snowy, change("night", "time")]),
                         "a snowy mountain road between pine forests covered in thick white snow, snow falling.")

    def test_unknown_world_phrase_is_left_literal(self):
        custom = change("old harbour", "world", "the old harbour.")
        self.assertEqual(world_with_overrides(custom, [custom, change("night", "time")]), "the old harbour.")
        self.assertIsNone(world_with_overrides(None, []))


class ComposeBaseTest(unittest.TestCase):
    def test_default_oak_base_becomes_subject_plus_world(self):
        self.assertEqual(compose_base(DEFAULT_BASE, ISHAAN["desert"]), "a wide shot of a single oak tree; " + ISHAAN["desert"])

    def test_template_base_swaps_only_the_world(self):
        start = SUBJECT + ISHAAN["country road"] + STYLE
        self.assertEqual(compose_base(start, ISHAAN["mars"]), SUBJECT + ISHAAN["mars"] + STYLE)
        self.assertEqual(compose_base(SUBJECT + STYLE, ISHAAN["mars"]), SUBJECT + ISHAAN["mars"] + STYLE)

    def test_subject_only_or_generic_base_appends(self):
        self.assertEqual(compose_base("a red car driving along a road", ISHAAN["desert"]),
                         "a red car driving along a road; " + ISHAAN["desert"])

    def test_unknown_base_with_its_own_world_is_refused(self):
        for base in ["a wide shot of a busy city street", "a quiet beach at dawn", "a dense forest clearing"]:
            with self.assertRaises(ValueError):
                compose_base(base, ISHAAN["desert"])


class PhraseTest(unittest.TestCase):
    def test_default_and_active_world(self):
        self.assertEqual(world_phrase({"changes": []}), ISHAAN["country road"])
        snap = {"changes": [{"id": "c1", "label": "desert", "kind": "world", "phrase": ISHAAN["desert"], "strength": 1.0},
                            {"id": "c2", "label": "dog", "kind": "extra", "phrase": "a dog", "strength": 1.0}]}
        self.assertEqual(world_phrase(snap), ISHAAN["desert"])

    def test_world_kind_is_known_to_state_once_wired(self):
        # Pending Codex: state.KINDS gains "world". Until then this documents the dependency.
        if "world" not in KINDS:
            self.skipTest("state.KINDS does not include 'world' yet (Codex wires state/composer)")
        self.assertIn("world", KINDS)


if __name__ == "__main__":
    unittest.main()

import unittest

from commands.state import DEFAULT_BASE, KINDS
from commands.world import (CAMERA, CAR_BASE, DEFAULT_WORLD, STYLE, WORLDS, car_prompt, compose_base, is_road_scene,
                            world_op, world_phrase, world_with_overrides)

# Ishaan's measured worlds as road | scene | light, copied from board #6886548029 ("winter" is our snowy mountain road).
MEASURED = {
    # Not on the board: the world of Ishaan's earlier tested car prompt (engine CAR_PROMPT), our start world.
    "country road": ("a straight two-lane country road",
                     "Green fields and rows of trees line both sides of the road under a clear blue summer sky.",
                     "Bright midday sunshine, crisp shadows."),
    "snowy mountain road": ("a snowy mountain road",
                            "Pine forests covered in thick white snow line both sides, high snowbanks along the edges of the road, snowflakes falling.",
                            "Cold blue overcast winter light, soft diffuse shadows."),
    "desert": ("a long straight desert highway",
               "Red sandstone mesas and golden sand dunes stretch to the horizon on both sides, dry scrub along the roadside, heat haze shimmering above the asphalt.",
               "Hot afternoon sun in a cloudless pale blue sky, strong hard shadows."),
    "neon city": ("a wide city street at night",
                  "Tall skyscrapers covered in glowing pink, blue and purple neon signs rise on both sides, the wet asphalt reflects the colourful lights, light rain falling.",
                  "Dark night sky, vivid neon glow, cyberpunk atmosphere."),
    "ocean coast": ("a coastal cliff road",
                    "A bright turquoise ocean with white waves breaking on the rocks below on one side, palm trees and green hills on the other side.",
                    "Sunny tropical day, clear blue sky, sparkling light on the water."),
}
ISHAAN = {  # exact tested strings, board #6885673339, and the measured worlds as "{road}. {scene} {light}"
    "mars": "a dusty road across the red rocky surface of Mars, craters and red mountains, a pale pink sky with two small moons.",
    **{place: f"{road}. {scene} {light}" for place, (road, scene, light) in MEASURED.items()},
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
        # Ishaan's template, word for word (board #6886548029).
        self.assertEqual(car_prompt(ISHAAN["desert"]),
                         "A red sports car, seen from directly behind with its rear lights facing the camera, drives fast "
                         "along a long straight desert highway, moving forward away from the camera. The camera follows "
                         "directly behind the car in a smooth, low tracking shot at the same speed, keeping the car centred "
                         "in the frame while the white dashed centre line rushes toward the viewer. Red sandstone mesas and "
                         "golden sand dunes stretch to the horizon on both sides, dry scrub along the roadside, heat haze "
                         "shimmering above the asphalt. Hot afternoon sun in a cloudless pale blue sky, strong hard shadows. "
                         "Cinematic, highly detailed, realistic, sharp focus, motion blur on the road.")
        self.assertEqual(CAR_BASE, car_prompt(ISHAAN["country road"]))

    def test_car_template_puts_objects_after_the_camera_and_modifiers_after_the_light(self):
        road, scene, light = MEASURED["desert"]
        self.assertEqual(car_prompt(ISHAAN["desert"], ["a dog running beside the car"], ["a subtle hint of fog"]),
                         f"A red sports car, seen from directly behind with its rear lights facing the camera, drives fast "
                         f"along {road}, moving forward away from the camera. {CAMERA} A dog running beside the car. "
                         f"{scene} {light} A subtle hint of fog. {STYLE}")
        self.assertEqual(car_prompt("the old harbour."), car_prompt("the old harbour").replace("..", "."))
        self.assertIn("drives fast along the old harbour, moving forward", car_prompt("the old harbour."))

    def test_road_scenes_are_the_template_bases(self):
        self.assertTrue(is_road_scene(CAR_BASE))
        self.assertTrue(is_road_scene(car_prompt(ISHAAN["mars"])))
        self.assertTrue(is_road_scene("a red sports car driving along a country road"))  # the spoken car scene
        for base in [DEFAULT_BASE, "a red car parked in a meadow", "a red sports car driving along the old harbour",
                     CAR_BASE + " "]:
            self.assertFalse(is_road_scene(base), base)

    def test_no_night_highway_world(self):  # Ishaan: a plain night highway turns to orange streaks
        self.assertNotIn("night highway", WORLDS)

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
                     "make it rainy", "make it a sunny beach"]:
            self.assertIsNone(world_op(text), text)
        # A known place plus time words is one world + time batch (PR #88), not None.
        combo = world_op("make it a desert at night")
        self.assertEqual([o["label"] for o in combo["operations"]], ["desert", "night"])

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
        self.assertEqual(world_with_overrides(desert, [desert, change("dog", "extra"), change("day", "time")]), ISHAAN["desert"])

    def test_explicit_time_weather_season_drop_only_conflicting_builtin_sky(self):
        desert = change("desert", "world", ISHAAN["desert"])
        self.assertEqual(world_with_overrides(desert, [desert, change("night", "time")]),
                         "a long straight desert highway. Red sandstone mesas and golden sand dunes stretch to the horizon on "
                         "both sides, dry scrub along the roadside, heat haze shimmering above the asphalt.")
        self.assertEqual(world_with_overrides(desert, [desert, change("winter", "season")]),
                         "a long straight desert highway. Red sandstone mesas and golden sand dunes stretch to the horizon on "
                         "both sides, dry scrub along the roadside. Hot afternoon sun in a cloudless pale blue sky, strong hard "
                         "shadows.")
        city = change("neon city", "world", ISHAAN["neon city"])
        self.assertEqual(world_with_overrides(city, [city, change("day", "time"), change("fog", "weather")]),
                         "a wide city street. Tall skyscrapers covered in glowing pink, blue and purple neon signs rise on both "
                         "sides, the wet asphalt reflects the colourful lights, vivid neon glow, cyberpunk atmosphere.")
        snowy = change("snowy mountain road", "world", ISHAAN["snowy mountain road"])
        self.assertEqual(world_with_overrides(snowy, [snowy, change("night", "time")]),
                         "a snowy mountain road. Pine forests covered in thick white snow line both sides, high snowbanks "
                         "along the edges of the road, snowflakes falling.")
        self.assertEqual(world_with_overrides(snowy, [snowy, change("summer", "season")]),  # no winter light in summer
                         "a snowy mountain road. Pine forests covered in thick white snow line both sides, high snowbanks "
                         "along the edges of the road.")
        coast = change("ocean coast", "world", ISHAAN["ocean coast"])
        self.assertEqual(world_with_overrides(coast, [coast, change("rain", "weather")]),
                         "a coastal cliff road. A bright turquoise ocean with white waves breaking on the rocks below on one "
                         "side, palm trees and green hills on the other side.")

    def test_unknown_world_phrase_is_left_literal(self):
        custom = change("old harbour", "world", "the old harbour.")
        self.assertEqual(world_with_overrides(custom, [custom, change("night", "time")]), "the old harbour.")
        self.assertIsNone(world_with_overrides(None, []))


class ComposeBaseTest(unittest.TestCase):
    def test_default_oak_base_becomes_subject_plus_world(self):
        self.assertEqual(compose_base(DEFAULT_BASE, ISHAAN["desert"]), "a wide shot of a single oak tree; " + ISHAAN["desert"])

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

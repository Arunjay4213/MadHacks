import json
import unittest
from pathlib import Path

from commands.rewriter import MAX_TOTAL_WORDS, SYSTEM_PROMPT, validate_rewrite

CAR = "a wide shot of a meadow with a single oak tree; a red car driving forward, away from the camera"
WINTER = CAR + "; winter scenery with snow on the ground"
NIGHT = CAR + "; night time under a full moon"
DEMO_PROMPTS = json.loads((Path(__file__).resolve().parents[2] / "engine" / "demo_prompts.json").read_text())
GOOD = ("fine powder snow texture, pale sky, diffuse cool illumination, crisp surface detail, gentle tonal gradients, "
        "photorealistic detail, cinematic color rendering")


def reply(details):
    return json.dumps({"details": details})


class SystemPromptTest(unittest.TestCase):
    def test_target_length_is_40_to_50_words(self):
        self.assertIn("40 to 50 words", SYSTEM_PROMPT)
        self.assertNotIn("temperature", SYSTEM_PROMPT.lower())

    def test_system_prompt_states_the_rules(self):
        for rule in ['{"details"', "camera", "motion", "objects", "earlier frames", "time of day", "scene"]:
            self.assertIn(rule, SYSTEM_PROMPT)


class AcceptTest(unittest.TestCase):
    def test_details_are_appended_after_the_verbatim_prompt(self):
        out = validate_rewrite(WINTER, reply(GOOD + "."))
        self.assertEqual(out, WINTER + "; " + GOOD)

    def test_dict_reply_and_whitespace(self):
        self.assertEqual(validate_rewrite(WINTER, {"details": "  " + GOOD + "  "}), WINTER + "; " + GOOD)

    def test_empty_details_return_the_canonical_prompt(self):
        self.assertEqual(validate_rewrite(WINTER, reply("")), WINTER)
        self.assertEqual(validate_rewrite(WINTER, {"details": "   "}), WINTER)

    def test_own_world_words_are_allowed(self):
        self.assertIsNotNone(validate_rewrite(NIGHT, reply("silver moonlight on the grass, deep blue shadows, crisp cold air, photorealistic")))
        self.assertIsNotNone(validate_rewrite(CAR, reply("golden grass texture, clear air, crisp bark detail, photorealistic")))

    def test_subject_prefix_is_byte_identical_across_world_edits(self):
        a = validate_rewrite(WINTER, reply(GOOD))
        b = validate_rewrite(NIGHT, reply("silver moonlight across the meadow, deep blue shadows, photorealistic"))
        self.assertTrue(a.startswith(CAR + "; ") and b.startswith(CAR + "; "))

    def test_deterministic(self):
        self.assertEqual(validate_rewrite(WINTER, reply(GOOD)), validate_rewrite(WINTER, reply(GOOD)))


class AdjectiveAliasTest(unittest.TestCase):
    STARS = "night time under a full moon; bright stars scattered across the sky"

    def test_generic_bright_or_dark_never_establishes_a_time_of_day(self):
        self.assertIsNone(validate_rewrite(self.STARS, reply("daylight and warm sunlight")))
        self.assertIsNone(validate_rewrite("bright summer scenery with lush green grass", reply("clear midday sunlight")))
        self.assertIsNone(validate_rewrite("a dark river; summer scenery", reply("silver moonlight on the water")))

    def test_bright_or_dark_as_plain_adjectives_in_details_are_fine(self):
        self.assertIsNotNone(validate_rewrite(self.STARS, reply("bright silver moonlight, deep dark blue shadows, crisp air")))


class RejectTest(unittest.TestCase):
    def assertRejected(self, prompt, details):
        self.assertIsNone(validate_rewrite(prompt, reply(details)), details)

    def test_malformed_replies(self):
        for bad in [None, 3, [], "not json", "[1]", json.dumps({"detail": GOOD}), json.dumps({"details": 5}),
                    json.dumps({"details": GOOD, "extra": 1}), {"details": None}]:
            self.assertIsNone(validate_rewrite(WINTER, bad), bad)
        self.assertIsNone(validate_rewrite("", reply(GOOD)))
        self.assertIsNone(validate_rewrite(None, reply(GOOD)))

    def test_length_and_lines(self):
        self.assertRejected(WINTER, " ".join(["frosty"] * 61))
        self.assertRejected(WINTER, GOOD + "\nsecond line")
        long_prompt = WINTER + " " + " ".join(["detail"] * (MAX_TOTAL_WORDS - 2 - len(WINTER.split())))
        self.assertEqual(len(long_prompt.split()), MAX_TOTAL_WORDS - 2)
        self.assertRejected(long_prompt, "cold blue tones, crisp air")  # 3 words over the total cap
        self.assertEqual(validate_rewrite(long_prompt, reply("")), long_prompt)

    def test_long_demo_scene_keeps_its_details(self):
        # Real prompts from engine/demo_prompts.json: the meadow chain's fixed part reaches 95 to 120 words, so
        # the old 100-word cap left nothing for details. 40 words of details after it must now pass.
        entries = {e["sentence"]: e for e in DEMO_PROMPTS[:9]}
        details = ", ".join(["warm ochre tones across the open ground", "fine dry grit on every surface",
                             "pale dusty air", "soft golden light low on the dunes", "faint heat shimmer",
                             "long soft shadows", "sun-bleached textures", "muted sand and amber colours",
                             "photorealistic, cinematic, sharp detail"]).split()[:40]
        details = " ".join(details)
        self.assertEqual(len(details.split()), 40)
        for sentence in ["make it a desert", "add a red car racing on a road"]:
            prompt = entries[sentence]["canonical"]
            self.assertGreaterEqual(len(prompt.split()), 95, sentence)
            self.assertEqual(validate_rewrite(prompt, reply(details)), f"{prompt}; {details}", sentence)
        desert = entries["make it a desert"]["canonical"]
        filler = " ".join(["dusty"] * (200 - len(desert.split()) - 40))
        self.assertRejected(desert + " " + filler, details)  # 200 words in total

    def test_markup_and_urls(self):
        for details in ["cold light {glow}", "see https://example.com", "<b>bold</b> snow", "cold `code`", "# heading"]:
            self.assertRejected(WINTER, details)

    def test_camera_motion_events(self):
        for details in ["the camera slowly pans across the snow", "zoom in on the frosty grass", "aerial view of the snowfield",
                        "the car is driving backwards", "something approaching through the snow", "suddenly an explosion of light"]:
            self.assertRejected(WINTER, details)

    def test_history_and_frame_claims(self):
        for details in ["the snow is still falling", "as in the previous frame", "snow falls again", "the meadow now white"]:
            self.assertRejected(WINTER, details)

    def test_negations(self):
        for details in ["no people around", "a scene without colour", "not a cloud in sight"]:
            self.assertRejected(WINTER, details)

    def test_objects_cannot_be_added_or_restated(self):
        for details in ["a dog sleeping in the snow", "a small wooden cabin nearby", "snow on the car roof", "a flock of birds"]:
            self.assertRejected(WINTER, details)

    def test_world_contradictions(self):
        self.assertRejected(NIGHT, "bright midday sunlight on the snow")  # day light in an active night
        self.assertRejected(NIGHT, "warm golden hour glow")               # sunset at night
        self.assertRejected(WINTER, "lush summer grass")                   # another season
        self.assertRejected(WINTER, "steady rain soaking the ground")      # weather not in the prompt
        self.assertRejected(CAR, "dark night sky full of moonlight")      # no time of day in the prompt: never add one
        self.assertRejected(CAR, "warm midday sunlight")
        self.assertRejected(CAR, "soft winter daylight")                   # neither season nor time requested


if __name__ == "__main__":
    unittest.main()

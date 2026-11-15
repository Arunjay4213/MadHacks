"""Greptile PR #215: a later "make it day" never brings back a season the user replaced."""

import unittest

from commands.composer import compose
from commands.service import CommandService

CAR_ON_SNOW = "a car driving on a snowy mountain road"
WINTER_SUN = "winter sunshine"


def prompt(*lines, base=None):
    s = CommandService(base, semantic_reader=lambda t, x: None)
    for t in lines:
        s.handle({"type": "final", "text": t, "t_client": 1})
    return compose(s.snapshot())


class DayAfterSummerStaysSummerTest(unittest.TestCase):
    def test_snowy_mountains_then_summer_then_day_has_no_winter_sunshine(self):
        p = prompt("move us to the snowy mountains", "make it summer", "make it day")
        self.assertIn("bright summer scenery", p)
        self.assertIn("bright daytime sunlight", p)
        self.assertNotIn(WINTER_SUN, p)
        self.assertNotIn("sparkling snow", p)

    def test_on_the_car_scene_too(self):
        p = prompt("move us to the snowy mountains", "make it summer", "make it day", base=CAR_ON_SNOW)
        self.assertIn("Bright summer scenery", p)
        self.assertNotIn(WINTER_SUN, p.lower())


class SnowyRoadDayWithoutSeasonChangeTest(unittest.TestCase):
    def test_snowy_mountains_then_day_is_winter_sunshine(self):
        self.assertIn(WINTER_SUN, prompt("move us to the snowy mountains", "make it day"))

    def test_car_scene_winter_then_day_is_winter_sunshine(self):
        self.assertIn(WINTER_SUN, prompt("make it winter", "make it day", base=CAR_ON_SNOW).lower())

    def test_undoing_summer_brings_the_snowy_day_back(self):
        p = prompt("move us to the snowy mountains", "make it summer", "make it day", "remove the summer")
        self.assertIn(WINTER_SUN, p)


if __name__ == "__main__":
    unittest.main()

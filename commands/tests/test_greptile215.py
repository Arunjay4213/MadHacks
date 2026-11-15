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
        for base in (None, CAR_ON_SNOW):
            with self.subTest(base=base):
                p = prompt("move us to the snowy mountains", "make it summer", "make it day", base=base)
                self.assertIn("bright summer scenery", p.lower())
                self.assertNotIn(WINTER_SUN, p.lower())
                if base is None:
                    self.assertIn("bright daytime sunlight", p)
                    self.assertNotIn("sparkling snow", p)


class SnowyRoadDayWithoutSeasonChangeTest(unittest.TestCase):
    def test_day_on_the_snowy_road_is_winter_sunshine(self):
        for lines, base in [
            (["move us to the snowy mountains", "make it day"], None),
            (["make it winter", "make it day"], CAR_ON_SNOW),
            (["move us to the snowy mountains", "make it summer", "make it day", "remove the summer"], None),
        ]:
            with self.subTest(lines=lines, base=base):
                self.assertIn(WINTER_SUN, prompt(*lines, base=base).lower())


if __name__ == "__main__":
    unittest.main()

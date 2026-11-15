"""The car scene through CommandService, with expectations copied from Ishaan's measured text (board #6886548029)."""

import unittest

from commands.service import CommandService
from commands.state import DEFAULT_BASE
from commands.world import CAR_BASE

HEAD = "A red sports car, seen from directly behind with its rear lights facing the camera, drives fast along "
CAMERA = (", moving forward away from the camera. The camera follows directly behind the car in a smooth, low tracking "
          "shot at the same speed, keeping the car centred in the frame while the white dashed centre line rushes toward "
          "the viewer. ")
STYLE = " Cinematic, highly detailed, realistic, sharp focus, motion blur on the road."
COUNTRY = (HEAD + "a straight two-lane country road" + CAMERA + "Green fields and rows of trees line both sides of the road "
           "under a clear blue summer sky. Bright midday sunshine, crisp shadows." + STYLE)
WINTER = (HEAD + "a snowy mountain road" + CAMERA + "Pine forests covered in thick white snow line both sides, high "
          "snowbanks along the edges of the road, snowflakes falling. Cold blue overcast winter light, soft diffuse "
          "shadows." + STYLE)
DESERT = (HEAD + "a long straight desert highway" + CAMERA + "Red sandstone mesas and golden sand dunes stretch to the "
          "horizon on both sides, dry scrub along the roadside, heat haze shimmering above the asphalt. Hot afternoon sun "
          "in a cloudless pale blue sky, strong hard shadows." + STYLE)


def final(service, text):
    return service.handle(dict(type="final", text=text, t_client=1))


def tap(service, action):
    return service.handle(dict(type="tap", action=action, t_client=1))


def car_service():
    service = CommandService(CAR_BASE, semantic_reader=lambda *a: None)
    final(service, "make it a desert")
    tap(service, "reset")  # reset to the car scene
    return service


class CarSceneTest(unittest.TestCase):
    def test_reset_gives_the_country_road_template(self):
        service = car_service()
        self.assertEqual(service.committed_prompt, COUNTRY)
        self.assertEqual(service.state_fields()["changes"], [])

    def test_winter_switches_to_the_snowy_mountain_road_and_one_undo_restores(self):
        for sentence in ["make it winter", "make it snowy"]:
            service = car_service()
            result = final(service, sentence)
            self.assertFalse(result["state"]["unmatched"], sentence)
            self.assertEqual(service.committed_prompt, WINTER, sentence)
            self.assertEqual(result["steer"]["prompt"], WINTER, sentence)
            self.assertNotIn("winter scenery with snow on the ground", service.committed_prompt)
            final(service, "undo")
            self.assertEqual(service.committed_prompt, COUNTRY, sentence)

    def test_winter_after_desert_replaces_the_desert(self):
        service = car_service()
        final(service, "make it a desert")
        self.assertEqual(service.committed_prompt, DESERT)
        final(service, "make it winter")
        self.assertEqual(service.committed_prompt, WINTER)  # the snowy road, not desert plus snow
        final(service, "undo")
        self.assertEqual(service.committed_prompt, DESERT)
        final(service, "make it winter")
        final(service, "make it a desert")  # a new place clears the winter again
        self.assertEqual(service.committed_prompt, DESERT)

    def test_partial_guess_is_the_snowy_road_too(self):
        service = car_service()
        steer = service.handle(dict(type="partial", text="make it winter", t_client=1))["steer"]
        self.assertEqual(steer["prompt"], WINTER)
        self.assertIn(WINTER, service.likely_prompts())

    def test_night_on_the_snowy_road_drops_only_the_daylight(self):
        service = car_service()
        final(service, "make it winter")
        final(service, "make it night")
        prompt = service.committed_prompt
        self.assertTrue(prompt.startswith(HEAD + "a snowy mountain road" + CAMERA), prompt)
        self.assertIn("snowflakes falling.", prompt)
        self.assertNotIn("Cold blue overcast winter light", prompt)
        self.assertTrue(prompt.endswith(STYLE))

    def test_meadow_winter_is_unchanged(self):
        service = CommandService(semantic_reader=lambda *a: None)
        final(service, "make it winter")
        self.assertEqual(service.committed_prompt, DEFAULT_BASE + "; winter scenery with snow on the ground")
        self.assertNotIn("motion blur", service.committed_prompt)


if __name__ == "__main__":
    unittest.main()

import unittest

from commands.reader import read
from commands.service import CommandService


def send(service, text):
    return service.handle(dict(type="final", text=text, t_client=1))


class ContinuityTests(unittest.TestCase):
    def test_explicit_appearance_and_motion_survive_world_edits(self):
        service = CommandService()
        send(service, "add a red car driving forward away from the camera")
        for command in ("make it winter", "also add a river", "make it night"):
            result = send(service, command)
            self.assertIn("a red car driving forward away from the camera", service.committed_prompt)
            self.assertNotIn("parked", service.committed_prompt)
        self.assertEqual([c["label"] for c in result["state"]["changes"]], ["car", "winter", "river", "night"])

    def test_generic_repeat_retains_specific_description(self):
        service = CommandService()
        send(service, "add a red car driving forward")
        send(service, "add a car")
        self.assertIn("a red car driving forward", service.committed_prompt)

    def test_explicit_motion_change_preserves_subject_and_weather(self):
        service = CommandService()
        send(service, "add a red car driving forward")
        send(service, "make it winter")
        result = send(service, "make the car drive backwards")
        self.assertFalse(result["state"]["unmatched"])
        self.assertIn("a red car driving backwards", service.committed_prompt)
        self.assertNotIn("driving forward", service.committed_prompt)
        self.assertIn("snow", service.committed_prompt)
        send(service, "undo")
        self.assertIn("a red car driving forward", service.committed_prompt)

    def test_partial_does_not_replace_a_detailed_subject_with_dictionary_default(self):
        service = CommandService()
        send(service, "add a red car driving forward")
        self.assertIsNone(service.handle(dict(type="partial", text="add a car", t_client=1))["steer"])
        self.assertIsNone(service.handle(dict(type="partial", text="make the car drive backwards", t_client=2))["steer"])

    def test_absent_motion_does_not_fall_through_to_nearest_embedding(self):
        calls = []
        service = CommandService(semantic_reader=lambda text, snapshot: calls.append(text))
        self.assertTrue(send(service, "make the car drive backwards")["state"]["unmatched"])
        self.assertEqual(calls, [])

    def test_motion_does_not_create_absent_subject(self):
        self.assertIsNone(read("make the car drive backwards", {"changes": []})["op"])

    def test_details_do_not_return_after_remove_or_reset(self):
        service = CommandService()
        send(service, "add a red car driving forward")
        send(service, "remove the car")
        self.assertNotIn("car", service.committed_prompt)
        send(service, "undo")
        self.assertIn("red car driving forward", service.committed_prompt)
        send(service, "reset")
        self.assertNotIn("car", service.committed_prompt)

    def test_motion_and_weather_in_one_sentence(self):
        service = CommandService()
        send(service, "add a red car driving forward")
        result = send(service, "make the car drive backwards and add a river")
        self.assertFalse(result["state"]["unmatched"])
        self.assertIn("red car driving backwards", service.committed_prompt)
        self.assertIn("river", service.committed_prompt)

    def test_strength_word_does_not_discard_real_motion_speed(self):
        service = CommandService()
        send(service, "add a red car driving slowly forward")
        self.assertIn("driving slowly forward", service.committed_prompt)

class RampTests(unittest.TestCase):
    def test_confirmed_edit_is_fast_while_slow_and_guess_remain_gentle(self):
        service = CommandService()
        guessed = service.handle(dict(type="partial", text="make it winter", t_client=1))
        self.assertEqual(guessed["steer"]["ramp_chunks"], 4)
        self.assertEqual(send(service, "make it winter")["steer"]["ramp_chunks"], 1)
        self.assertEqual(send(service, "slowly add a river")["steer"]["ramp_chunks"], 12)
        self.assertEqual(send(service, "make it night")["steer"]["ramp_chunks"], 1)

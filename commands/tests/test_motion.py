import unittest
from commands.service import CommandService


def send(service, text):
    return service.handle(dict(type='final', text=text, t_client=1))


class MotionFormsTests(unittest.TestCase):
    def test_natural_motion_forms_replace_the_existing_direction(self):
        for command in ('the car drives backwards', 'make the car move backwards', 'the car is driving backwards', 'make the car near the lake drive backwards'):
            service = CommandService()
            send(service, 'add a red car driving forward')
            result = send(service, command)
            self.assertFalse(result['state']['unmatched'])
            self.assertIn('backwards', service.committed_prompt, command)
            self.assertNotIn('driving forward', service.committed_prompt, command)

    def test_bare_descriptions_keep_all_explicit_visual_details(self):
        for text in ('car on road', 'a red car driving fast on a road', 'a little red car', 'a light blue boat'):
            service = CommandService()
            send(service, text)
            self.assertIn(text, service.committed_prompt)
            self.assertEqual(service.state_fields()['changes'][0]['strength'], 1.0)

    def test_bare_car_does_not_force_parked_but_explicit_parked_is_kept(self):
        service = CommandService()
        send(service, 'add a car')
        self.assertNotIn('parked', service.committed_prompt)
        self.assertIn('driving forward', service.committed_prompt)
        send(service, 'add a red car parked beside the road')
        self.assertIn('red car parked beside the road', service.committed_prompt)

    def test_second_object_is_distinct_from_subject_in_base(self):
        service = CommandService(base='a red car driving forward away from the camera')
        send(service, 'add a blue car')
        self.assertIn('a red car driving forward away from the camera', service.committed_prompt)
        self.assertIn('a second blue car', service.committed_prompt)
        self.assertIn('alongside', service.committed_prompt)

    def test_base_tree_and_added_tree_are_distinct(self):
        service = CommandService()
        send(service, 'add a tree')
        self.assertIn('additional', service.committed_prompt)
        self.assertIn('beside', service.committed_prompt)

import unittest
from commands.service import CommandService


def send(service, text):
    return service.handle(dict(type='final', text=text, t_client=1))


class WorldIntegrationTests(unittest.TestCase):
    def test_world_replaces_meadow_but_keeps_subject_and_user_details(self):
        service = CommandService()
        send(service, 'add a red car driving forward on a road')
        result = send(service, 'make it a desert')
        self.assertFalse(result['state']['unmatched'])
        self.assertIn('desert', service.committed_prompt)
        self.assertNotIn('meadow', service.committed_prompt)
        self.assertIn('oak tree', service.committed_prompt)
        self.assertIn('red car driving forward on a road', service.committed_prompt)
        self.assertNotIn('looks a desert', service.committed_prompt)
        send(service, 'undo')
        self.assertIn('meadow', service.committed_prompt)
        self.assertNotIn('desert', service.committed_prompt)

    def test_second_world_replaces_first_and_removal_returns_original(self):
        service = CommandService()
        send(service, 'make it a desert')
        send(service, 'make it a forest')
        self.assertNotIn('desert', service.committed_prompt)
        self.assertIn('forest', service.committed_prompt)
        self.assertEqual(len(service.state_fields()['changes']), 1)
        send(service, 'remove the forest')
        self.assertIn('meadow', service.committed_prompt)

    def test_world_change_keeps_objects_and_replaces_season(self):
        service = CommandService()
        send(service, 'make it winter')
        send(service, 'add a river')
        send(service, 'make it a desert')
        self.assertNotIn('winter', service.committed_prompt)  # the new world brings its own season (issue #112)
        self.assertIn('river', service.committed_prompt)
        self.assertNotIn('summer', service.committed_prompt)
        send(service, 'make it night')
        self.assertNotIn('sunset', service.committed_prompt)
        self.assertIn('night', service.committed_prompt)

    def test_compound_place_and_subject_preserve_one_undo(self):
        service = CommandService()
        result = send(service, 'make it a desert and add a red car')
        self.assertFalse(result['state']['unmatched'])
        self.assertIn('red car', service.committed_prompt)
        self.assertNotIn('meadow', service.committed_prompt)
        send(service, 'undo')
        self.assertEqual(service.state_fields()['changes'], [])
        self.assertIn('meadow', service.committed_prompt)

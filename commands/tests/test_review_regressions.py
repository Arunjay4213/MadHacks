import unittest
from commands.reader import read, is_blocked, spot
from commands.service import CommandService


def send(service, text):
    return service.handle(dict(type='final', text=text, t_client=1))


class ReviewRegressions(unittest.TestCase):
    def test_setting_commands_preserve_subjects_instead_of_resetting_scene(self):
        for command in ('take me to the desert', 'change the scene to a desert', 'go back to the desert'):
            service = CommandService()
            send(service, 'add a red car')
            send(service, 'make it night')
            result = send(service, command)
            self.assertEqual([c['label'] for c in result['state']['changes']], ['car', 'night', 'desert'], command)
            self.assertNotIn('meadow', service.committed_prompt)
            self.assertIn('red car', service.committed_prompt)

    def test_setting_move_is_not_an_object_position_edit(self):
        service = CommandService()
        send(service, 'add a red car')
        result = send(service, 'move us to the desert')
        self.assertFalse(result['state']['unmatched'])
        self.assertIn('red car', service.committed_prompt)
        self.assertNotIn('meadow', service.committed_prompt)

    def test_adjective_aliases_are_preserved_in_object_descriptions(self):
        for text, fragment in [('add a dark car', 'dark car'), ('add dark clouds', 'dark clouds')]:
            service = CommandService()
            send(service, text)
            self.assertIn(fragment, service.committed_prompt)

    def test_place_partial_does_not_guess_an_object_addition(self):
        self.assertIsNone(spot('make it a forest'))
        self.assertIsNone(spot('take me to the beach'))

    def test_position_edit_is_deferred_not_stored_as_a_noun(self):
        service = CommandService()
        send(service, 'add a red car on the left')
        result = send(service, 'move the car to the right')
        self.assertTrue(result['state']['unmatched'])
        self.assertIn('red car on the left', service.committed_prompt)
        self.assertTrue(is_blocked('move the car to the right'))

    def test_main_noun_beats_a_location_noun(self):
        for text, label in [('add snow near the river', 'winter'), ('add a bright red car near the lake','car'),
                            ('add dark clouds above a house','clouds'), ('add a lion near the river','lion')]:
            op = read(text, {'changes': []})['op']
            self.assertEqual(op['label'], label, text)

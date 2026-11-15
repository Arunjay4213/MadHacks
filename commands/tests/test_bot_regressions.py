import unittest
from importlib.util import find_spec
from commands.service import CommandService
from commands.reader import resolve, style_fallback
from commands.semantic import SemanticReader


class BotRegressions(unittest.TestCase):
    def test_stop_and_without_are_separate_actions(self):
        for clause in ('stop the rain', 'without the rain'):
            service = CommandService()
            service.handle(dict(type='final', text='make it rain', t_client=1))
            result = service.handle(dict(type='final', text='add a dog and '+clause, t_client=2))
            self.assertFalse(result['state']['unmatched'])
            self.assertEqual([c['label'] for c in result['state']['changes']], ['dog'])

    def test_multiword_alias_wins_over_label_inside_it(self):
        service = CommandService()
        for text in ('add a house', 'add a lighthouse'):
            service.handle(dict(type='final', text=text, t_client=1))
        for query in ('the light house', 'the white light house'):
            self.assertEqual(resolve(query, service.snapshot()), 'c2')

    def test_rainbow_alias_does_not_remove_rain(self):
        service = CommandService()
        for text in ('make it rain', 'add a rainbow'):
            service.handle(dict(type='final', text=text, t_client=1))
        self.assertEqual(resolve('the rain bow', service.snapshot()), 'c2')

    def test_style_intensity_is_not_part_of_the_label(self):
        op = style_fallback('make it a little nocturnal')
        self.assertEqual((op['label'], op['strength']), ('nocturnal', .3))

    @unittest.skipUnless(find_spec("numpy"), "optional vector tests need numpy")
    def test_vector_intensity_uses_the_same_reader_rules(self):
        reader = SemanticReader(lambda texts: [[1, 0] for t in texts],
                                entries={'night':dict(label='night',kind='time',phrase='night time')})
        op = reader('make it a little nocturnal', {'changes': []})
        self.assertEqual((op['label'], op['strength']), ('night', .3))

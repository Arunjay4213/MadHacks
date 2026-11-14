"""User-visible prompt assertions for the actual demo script and reported sentences."""
import json
from pathlib import Path
import unittest
from commands.service import CommandService


def send(service, text):
    return service.handle(dict(type='final', text=text, t_client=1))


class DemoPromptTests(unittest.TestCase):
    def test_each_line_of_the_demo_preserves_its_visible_content(self):
        script = json.loads((Path(__file__).resolve().parents[2] / 'engine/tools/demo_scripts/draft.json').read_text())
        service = CommandService()
        checks = {
            'make it winter': ('snow on the ground',),
            'add a red car': ('snow on the ground', 'red car'),
            'make it night': ('snow on the ground', 'red car', 'night time'),
            'add a lion': ('snow on the ground', 'red car', 'night time', 'lion'),
            'sunny': ('snow on the ground', 'red car', 'lion', 'daytime sunlight'),
        }
        for step in script['steps']:
            if step.get('reset'):
                service.handle(dict(type='tap',action='reset'))
                self.assertEqual(service.state_fields()['changes'], [])
                continue
            text = step['say']
            with self.subTest(text=text):
                self.assertFalse(send(service, text)['state']['unmatched'])
                for fragment in checks[text]:
                    self.assertIn(fragment, service.committed_prompt)
                self.assertNotIn('parked', service.committed_prompt)
        self.assertNotIn('night time', service.committed_prompt)

    def test_reported_description_variations_reach_the_prompt(self):
        for text, fragment in [
            ('add a bright red car', 'bright red car'),
            ('put a boat on the lake', 'boat on the lake'),
            ('add a dog on the beach', 'dog on the beach'),
            ('Add a red car, racing on a road!', 'red car racing on a road'),
            ('car on road', 'car on road'),
        ]:
            with self.subTest(text=text):
                service = CommandService()
                send(service, text)
                self.assertIn(fragment, service.committed_prompt)

    def test_a_little_more_increases_and_less_decreases(self):
        service = CommandService()
        send(service, 'make it winter')
        send(service, 'a little more snow')
        self.assertEqual(service.state_fields()['changes'][0]['strength'], 1.3)
        send(service, 'a little less snow')
        self.assertEqual(service.state_fields()['changes'][0]['strength'], 1.0)

    def test_named_undo_targets_the_named_modifier(self):
        service = CommandService()
        for text in ('make it winter','add a river','add a red car'):
            send(service, text)
        result = send(service,'undo the winter')
        self.assertEqual([c['label'] for c in result['state']['changes']], ['river','car'])
        self.assertIn('red car', service.committed_prompt)

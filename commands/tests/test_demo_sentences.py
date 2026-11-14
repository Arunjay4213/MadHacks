"""Sentence to composed prompt, through CommandService, for the demo script and sentences users reported.

Each expectation says what the user asked for, so it must be visible in the prompt the engine steers to.
"""

import json
import unittest
from pathlib import Path

from commands.service import CommandService

DRAFT = Path(__file__).resolve().parents[2] / "engine" / "tools" / "demo_scripts" / "draft.json"


def send(service, text):
    return service.handle({"type": "final", "text": text, "t_client": 1})


def labels(service):
    return [c["label"] for c in service.state_fields()["changes"]]


def strength(service, label):
    return next(c["strength"] for c in service.state_fields()["changes"] if c["label"] == label)


class DemoScriptTest(unittest.TestCase):
    def test_every_line_of_the_draft_script_keeps_what_came_before(self):
        # Fragments the prompt must hold after each line. A line without an entry here fails, so a new script
        # line needs its own expectation.
        must_have = {
            "make it winter": ("snow on the ground",),
            "add a red car": ("snow on the ground", "red car"),
            "make it night": ("snow on the ground", "red car", "night time"),
            "add a lion": ("snow on the ground", "red car", "night time", "lion"),
            "sunny": ("snow on the ground", "red car", "lion", "daytime sunlight"),
        }
        service = CommandService()
        for step in json.loads(DRAFT.read_text())["steps"]:
            if step.get("reset"):
                service.handle({"type": "tap", "action": "reset"})
                self.assertEqual(service.state_fields()["changes"], [])
                continue
            text = step["say"]
            with self.subTest(text=text):
                self.assertFalse(send(service, text)["state"]["unmatched"])
                for fragment in must_have[text]:
                    self.assertIn(fragment, service.committed_prompt)
        self.assertNotIn("night time", service.committed_prompt)  # sunny replaced the night


class ReportedSentencesTest(unittest.TestCase):
    def test_descriptions_reach_the_prompt(self):
        for text, fragment in [
            ("add a bright red car", "bright red car"),
            ("put a boat on the lake", "boat on the lake"),
            ("add a dog on the beach", "dog on the beach"),
            ("Add a red car, racing on a road!", "red car racing on a road"),
            ("car on road", "car on road"),
        ]:
            with self.subTest(text=text):
                service = CommandService()
                send(service, text)
                self.assertIn(fragment, service.committed_prompt)

    def test_a_little_more_raises_and_a_little_less_lowers(self):
        service = CommandService()
        send(service, "make it winter")
        before = strength(service, "winter")
        send(service, "a little more snow")
        more = strength(service, "winter")
        self.assertGreater(more, before)
        send(service, "a little less snow")
        self.assertLess(strength(service, "winter"), more)

    def test_named_undo_removes_the_named_change(self):
        service = CommandService()
        for text in ("make it winter", "add a river", "add a red car"):
            send(service, text)
        send(service, "undo the winter")
        self.assertEqual(labels(service), ["river", "car"])
        self.assertNotIn("snow", service.committed_prompt)
        self.assertIn("red car", service.committed_prompt)

    def test_snow_near_the_river_adds_snow_not_another_river(self):
        service = CommandService()
        send(service, "add a river")
        send(service, "add snow near the river")
        self.assertEqual(sorted(labels(service)), ["river", "winter"])
        self.assertIn("snow", service.committed_prompt)

    def test_add_and_stop_in_one_sentence_does_both(self):
        service = CommandService()
        send(service, "make it rain")
        send(service, "add a dog and stop the rain")
        self.assertEqual(labels(service), ["dog"])
        self.assertIn("dog", service.committed_prompt)
        self.assertNotIn("rain", service.committed_prompt)


if __name__ == "__main__":
    unittest.main()

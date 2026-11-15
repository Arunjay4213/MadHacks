"""An object the sentence only mentions as context is never added (live, 2026-11-15 07:42: "i want to see the car's
side profile" on the car scene added a second car, which drove off the road)."""

import unittest

from commands.reader import read

CAR = {"base": "a red sports car", "changes": []}
MEADOW = {"base": "a wide shot of a meadow with a single oak tree", "changes": []}


class ContextObjectTest(unittest.TestCase):
    def test_context_mentions_add_nothing(self):
        for text, snap in [("i want to see the car's side profile", CAR), ("change the view of the car", CAR),
                           ("show me the car from the side", CAR), ("i'd like to see the dog", MEADOW)]:
            with self.subTest(text=text):
                self.assertIsNone(read(text, snap)["op"])

    def test_requests_to_add_still_add(self):
        for text in ["add a dog", "a dog", "i want a dog", "put a red car next to the tree", "add another car"]:
            with self.subTest(text=text):
                op = read(text, MEADOW if "another" not in text else CAR)["op"]
                self.assertEqual(op and op["op"], "add")


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

from commands.dictionary import KEYWORDS, KEYWORD_RX, load_dictionary, normalize
from commands.reader import read, spot
from commands.service import CommandService
from commands.state import SceneState


class DictionaryTests(unittest.TestCase):
    def test_every_alias_can_add_and_remove_its_canonical_change(self):
        for alias, entry in KEYWORDS.items():
            with self.subTest(alias=alias):
                state = SceneState("an empty scene")
                result = read(f"please add {alias}", state.snapshot())
                self.assertIsNotNone(result["op"])
                self.assertEqual(result["op"]["label"], entry["label"])
                self.assertEqual(result["op"]["phrase"], entry["phrase"])
                self.assertEqual(spot(f"add {alias}"), alias)
                state.apply(result["op"])
                removed = read(f"remove the {alias}", state.snapshot())
                self.assertEqual(removed["op"], {"op": "remove", "id": state.snapshot()["changes"][0]["id"]})

    def test_longest_multiword_match_wins_over_shorter_alias(self):
        for phrase, label in [("snow man", "snowman"), ("night time", "night"),
                              ("early morning", "sunrise"), ("water fall", "waterfall")]:
            with self.subTest(phrase=phrase):
                key = KEYWORD_RX.search(normalize(phrase)).group()
                self.assertEqual(KEYWORDS[key]["label"], label)

    def test_unicode_punctuation_hyphens_and_spacing_are_normalized(self):
        self.assertEqual(normalize("  Don’t…\tmake—IT  NIGHT! "), "don't make it night")
        self.assertEqual(normalize("ＭＡＫＥ ＩＴ ＤＡＹ"), "make it day")
        self.assertEqual(read("go...   back", SceneState().snapshot())["op"], {"op": "undo"})
        self.assertIsNone(read("do   not make it night", SceneState().snapshot())["op"])

    def test_day_replaces_night_and_objects_stack_in_service(self):
        service = CommandService()
        for index, text in enumerate(["make it night", "make it sunny", "add a dog", "also add a car"]):
            result = service.handle(dict(type="final", text=text, t_client=index))
            self.assertFalse(result["state"]["unmatched"], text)
        self.assertEqual([c["label"] for c in result["state"]["changes"]], ["day", "dog", "car"])
        removed = service.handle(dict(type="final", text="remove the automobile", t_client=9))
        self.assertEqual([c["label"] for c in removed["state"]["changes"]], ["day", "dog"])

    def test_bad_or_conflicting_dictionary_fails_at_load(self):
        valid = dict(label="river", kind="extra", phrase="a river", aliases=["stream"])
        cases = [[], {}, [dict(valid, phrase=" ")], [dict(valid, kind="madeup")],
                 [dict(valid, aliases="stream")], [dict(valid, aliases=[""])],
                 [valid, dict(label="lake", kind="extra", phrase="a lake", aliases=["stream"])]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "keywords.json"
            for data in cases:
                with self.subTest(data=data):
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        load_dictionary(path)


if __name__ == "__main__":
    unittest.main()

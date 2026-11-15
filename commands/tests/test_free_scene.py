"""Free scenes (commands.state.Origin.FREE, issue #271): a full prompt written by the engine's final reader, its chips,
and the dictionary fast-path rule (commands.service.dictionary_reading)."""

import unittest

from commands.composer import compose
from commands.service import CommandService, dictionary_reading
from commands.state import FREE_ONLY, Origin, SceneState, is_free

LION = dict(op="add", label="lion", kind="extra", phrase="a lion", strength=1.0)
ARID = {"op": "free_scene", "prompt": "A wide shot of cracked dry earth with a bare oak tree.", "chips": ["arid ground"]}


def free(prompt="A wide shot of cracked dry earth with a bare oak tree.", chips=("arid ground",)):
    return {"op": "free_scene", "prompt": prompt, "chips": list(chips)}


class FreeSceneStateTests(unittest.TestCase):
    def test_a_free_scene_is_its_own_prompt_with_its_chips(self):
        state = SceneState()
        self.assertIs(state.origin, Origin.COMPOSED)
        self.assertTrue(state.apply(free(chips=["Arid  Ground", "lion", "arid ground"])))
        snap = state.snapshot()
        self.assertIs(state.origin, Origin.FREE)
        self.assertTrue(is_free(snap))
        self.assertEqual(compose(snap), "A wide shot of cracked dry earth with a bare oak tree.")
        self.assertEqual([(c["label"], c["kind"], c["strength"]) for c in snap["changes"]],
                         [("arid ground", "free", 1.0), ("lion", "free", 1.0)])
        self.assertEqual(snap["base"], "a wide shot of a meadow with a single oak tree")  # kept, unused while free

    def test_the_dictionary_cannot_edit_a_free_scene(self):
        state = SceneState()
        state.apply(free())
        chip = state.snapshot()["changes"][0]["id"]
        for op in (LION, {"op": "remove", "id": chip}, {"op": "set_strength", "id": chip, "strength": 0.5},
                   {"op": "batch", "operations": [LION]}):
            with self.assertRaises(ValueError) as raised:
                state.apply(op)
            if op["op"] != "batch":
                self.assertEqual(str(raised.exception), FREE_ONLY)
        self.assertTrue(is_free(state.snapshot()))

    def test_new_scene_reset_and_undo_leave_a_free_scene(self):
        state = SceneState()
        state.apply(LION)
        state.apply(free())
        self.assertTrue(state.apply({"op": "batch", "operations": [{"op": "new_scene", "base": "a beach"}, LION]}))
        self.assertEqual((state.origin, compose(state.snapshot())), (Origin.COMPOSED, "a beach; a lion"))
        self.assertTrue(state.apply({"op": "undo"}))
        self.assertIs(state.origin, Origin.FREE)
        self.assertTrue(state.apply({"op": "undo"}))  # back to the composed meadow with the lion
        self.assertEqual(compose(state.snapshot()), "a wide shot of a meadow with a single oak tree; a lion")
        state.apply(free())
        self.assertTrue(state.apply({"op": "reset"}))
        self.assertEqual((state.origin, state.snapshot()["changes"]), (Origin.COMPOSED, []))

    def test_a_chip_that_keeps_its_label_keeps_its_id(self):
        state = SceneState()
        state.apply(free(chips=["arid ground"]))
        first = state.snapshot()["changes"][0]["id"]
        state.apply(free("A lion on cracked dry earth.", ["arid ground", "lion"]))
        ids = [c["id"] for c in state.snapshot()["changes"]]
        self.assertEqual(ids[0], first)
        self.assertNotEqual(ids[1], first)
        self.assertFalse(state.apply(free("A lion on cracked dry earth.", ["arid ground", "lion"])))  # no change

    def test_invalid_free_scenes_are_refused(self):
        state = SceneState()
        for op in (free(prompt=""), free(prompt="x" * 2000), free(chips=["a" * 25]), free(chips=["x"] * 17),
                   {"op": "free_scene", "prompt": "p", "chips": "lion"}):
            with self.assertRaises(ValueError):
                state.apply(op)
        self.assertIs(state.origin, Origin.COMPOSED)

    def test_from_snapshot_keeps_ids_and_origin(self):
        state = SceneState()
        state.apply(LION)
        state.apply(free())
        copy = SceneState.from_snapshot(state.snapshot())
        self.assertEqual(copy.snapshot()["changes"], state.snapshot()["changes"])
        self.assertIs(copy.origin, Origin.FREE)


class DictionaryReadingTests(unittest.TestCase):
    def setUp(self):
        self.meadow = SceneState().snapshot()

    def test_sentences_the_keyword_table_reads_whole(self):
        for text, labels in (("make it winter", ["winter"]), ("add a dog and a cat", ["dog", "cat"]),
                             ("make it slowly rain", ["rain"]), ("a little more snow", ["winter"]),
                             ("take me to the beach", ["ocean coast"])):
            ops = dictionary_reading(text, self.meadow)
            self.assertEqual([o.get("label") for o in ops], labels, text)

    def test_the_leads_live_sequence_is_not_read_whole(self):
        # Live 07:27 UTC: the keyword table "read" these as an add of "trees", a pass-through object, or night only.
        for text in ("kill the tree's leaves", "cut the grass", "i want no grass. arid environment", "add a lion",
                     "now i want a savannah", "i now want a city in the night with a car driving on the road",
                     "city with road and red car driving", "add a red car", "add a glowing unicorn",
                     "add jensen huang", "make it spooky", "start over with a beach", "zoom out"):
            self.assertIsNone(dictionary_reading(text, self.meadow), text)

    def test_a_free_scene_is_never_read_by_the_dictionary(self):
        state = SceneState()
        state.apply(free())
        self.assertIsNone(dictionary_reading("make it winter", state.snapshot()))

    def test_removes_strength_and_explicit_new_scenes_are_read_whole(self):
        state = SceneState()
        state.apply(dict(op="add", label="dog", kind="extra", phrase="a dog", strength=1.0))
        snap = state.snapshot()
        self.assertEqual(dictionary_reading("remove the dog", snap), [{"op": "remove", "id": "c1"}])
        self.assertEqual(dictionary_reading("new scene of a lighthouse on a cliff", snap)[0]["op"], "new_scene")
        self.assertEqual(dictionary_reading("undo that", snap), [{"op": "undo"}])
        self.assertEqual(snap, state.snapshot())  # pure

    def test_a_remove_that_leaves_a_named_thing_behind_is_not_read_whole(self):
        # Greptile on #273: "lion" is not in the table, so the clause is not split and only one of them would go.
        state = SceneState()
        state.apply(dict(op="add", label="dog", kind="extra", phrase="a dog", strength=1.0))
        state.apply(dict(op="add", label="lion", kind="extra", phrase="a lion", strength=1.0))
        snap = state.snapshot()
        self.assertIsNone(dictionary_reading("remove the dog and the lion", snap))
        self.assertEqual(dictionary_reading("remove the puppy", snap), [{"op": "remove", "id": "c1"}])
        self.assertEqual(dictionary_reading("undo the dog", snap), [{"op": "remove", "id": "c1"}])

    def test_a_repeated_add_that_changes_nothing_goes_to_the_final_reader(self):
        # Greptile on #273: the second "add a dog" changed nothing and skipped Sonnet, which makes it stronger.
        state = SceneState()
        state.apply(dict(op="add", label="dog", kind="extra", phrase="a dog", strength=1.0))
        self.assertIsNone(dictionary_reading("add a dog", state.snapshot()))
        self.assertEqual(dictionary_reading("undo", SceneState().snapshot()), [{"op": "undo"}])


class ServiceOnFreeScenesTests(unittest.TestCase):
    def setUp(self):
        self.svc = CommandService(semantic_reader=lambda t, s: None)
        self.svc.apply_final("i want no grass. arid environment", [ARID])

    def test_partials_on_a_free_scene_guess_nothing(self):
        out = self.svc.handle({"type": "partial", "text": "add a dog", "t_client": 1})
        self.assertIsNone(out["steer"])
        self.assertEqual(self.svc.likely_prompts(), [ARID["prompt"]])

    def test_a_dictionary_edit_of_a_free_scene_is_not_understood_and_changes_nothing(self):
        out = self.svc.handle({"type": "final", "text": "add a dog", "t_client": 1})
        self.assertIsNone(out["steer"])
        self.assertTrue(out["state"]["unmatched"])
        self.assertEqual(self.svc.committed_prompt, ARID["prompt"])

    def test_undo_goes_back_to_the_composed_meadow(self):
        out = self.svc.handle({"type": "final", "text": "undo", "t_client": 1})
        self.assertEqual(out["steer"]["prompt"], "a wide shot of a meadow with a single oak tree")
        self.assertEqual(out["state"]["changes"], [])


if __name__ == "__main__":
    unittest.main()

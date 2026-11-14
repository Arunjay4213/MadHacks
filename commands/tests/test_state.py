import unittest

from commands.state import SceneState


def change(label, kind="extra", strength=1.0):
    return dict(op="add", label=label, kind=kind, phrase=f"a {label}", strength=strength)


class StateTests(unittest.TestCase):
    def test_stack_remove_middle_and_three_undos(self):
        state = SceneState("meadow")
        for label in ("winter", "river", "night"):
            self.assertTrue(state.apply(change(label)))
        river = state.snapshot()["changes"][1]["id"]
        state.apply(dict(op="remove", id=river))
        self.assertEqual([c["label"] for c in state.snapshot()["changes"]], ["winter", "night"])
        state.apply(dict(op="undo"))
        self.assertEqual([c["label"] for c in state.snapshot()["changes"]], ["winter", "river", "night"])
        for _ in range(3):
            self.assertTrue(state.apply(dict(op="undo")))
        self.assertEqual(state.snapshot(), dict(base="meadow", changes=[], can_undo=False))
        self.assertFalse(state.apply(dict(op="undo")))

    def test_singleton_replaces_while_extras_stack(self):
        state = SceneState("meadow")
        state.apply(change("winter", "season"))
        state.apply(change("river"))
        state.apply(change("summer", "season"))
        self.assertEqual([c["label"] for c in state.snapshot()["changes"]], ["river", "summer"])
        state.apply(dict(op="undo"))
        self.assertEqual([c["label"] for c in state.snapshot()["changes"]], ["winter", "river"])

    def test_duplicate_and_unknown_remove_do_not_consume_history(self):
        state = SceneState("meadow")
        state.apply(change("river"))
        self.assertFalse(state.apply(change("RIVER")))
        self.assertFalse(state.apply(dict(op="remove", id="unknown")))
        state.apply(dict(op="undo"))
        self.assertFalse(state.snapshot()["can_undo"])

    def test_strength_update_preserves_id_and_is_undoable(self):
        state = SceneState("meadow")
        state.apply(change("river"))
        original = state.snapshot()["changes"][0]
        state.apply(dict(op="set_strength", id=original["id"], strength=0.3))
        self.assertEqual(state.snapshot()["changes"][0]["strength"], 0.3)
        state.apply(dict(op="undo"))
        self.assertEqual(state.snapshot()["changes"][0], original)
        state.apply(change("river", strength=1.3))
        self.assertEqual(state.snapshot()["changes"][0]["id"], original["id"])

    def test_reset_restores_original_base_and_can_be_undone(self):
        state = SceneState("meadow")
        state.apply(dict(op="new_scene", base="beach"))
        state.apply(change("river"))
        before = state.snapshot()
        state.apply(dict(op="reset"))
        self.assertEqual(state.snapshot()["base"], "meadow")
        self.assertEqual(state.snapshot()["changes"], [])
        state.apply(dict(op="undo"))
        self.assertEqual(state.snapshot(), before)

    def test_snapshot_is_detached_and_ids_are_never_reused(self):
        state = SceneState("meadow")
        state.apply(change("river"))
        snapshot = state.snapshot()
        old_id = snapshot["changes"][0]["id"]
        snapshot["changes"][0]["label"] = "wrong"
        self.assertEqual(state.snapshot()["changes"][0]["label"], "river")
        state.apply(dict(op="undo"))
        state.apply(change("lake"))
        self.assertNotEqual(state.snapshot()["changes"][0]["id"], old_id)

    def test_invalid_operations_are_atomic(self):
        state = SceneState("meadow")
        state.apply(change("river"))
        before = state.snapshot()
        invalid = [None, {}, {"op": "wrong"}, {"op": "add", "label": "x"},
                   change("x", kind="unknown"), change(""), change("x", strength=True),
                   change("x", strength=float("nan")), change("x", strength=1.4),
                   {"op": "new_scene", "base": " "}, {"op": "remove", "id": 1},
                   {"op": "set_strength", "id": "c1", "strength": -1}]
        for op in invalid:
            with self.subTest(op=op), self.assertRaises(ValueError):
                state.apply(op)
            self.assertEqual(state.snapshot(), before)

    def test_history_is_bounded_but_recent_undo_still_works(self):
        state = SceneState("meadow")
        for n in range(105):
            state.apply(dict(op="new_scene", base=f"scene {n}"))
        count = 0
        while state.apply(dict(op="undo")):
            count += 1
        self.assertEqual(count, 100)
        self.assertEqual(state.snapshot()["base"], "scene 4")

    def test_updating_kind_still_replaces_the_previous_singleton(self):
        state = SceneState("meadow")
        state.apply(change("winter", "season"))
        state.apply(change("summer", "extra"))
        summer_id = state.snapshot()["changes"][1]["id"]
        state.apply(change("summer", "season"))
        self.assertEqual(state.snapshot()["changes"], [
            dict(id=summer_id, label="summer", kind="season", phrase="a summer", strength=1.0)])


if __name__ == "__main__":
    unittest.main()

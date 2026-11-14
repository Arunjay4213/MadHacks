import unittest

from commands.service import CommandService
from commands.state import SceneState


def send(service, text):
    return service.handle(dict(type="final", text=text, t_client=1))


class CompoundTests(unittest.TestCase):
    def test_add_and_remove_share_one_target_and_one_undo_step(self):
        service = CommandService()
        send(service, "add a lighthouse")
        result = send(service, "add a dog and remove the lighthouse")
        self.assertFalse(result["state"]["unmatched"])
        self.assertEqual([c["label"] for c in result["state"]["changes"]], ["dog"])
        self.assertIn("dog", result["steer"]["prompt"])
        self.assertNotIn("lighthouse", result["steer"]["prompt"])
        undo = send(service, "go back")
        self.assertEqual([c["label"] for c in undo["state"]["changes"]], ["lighthouse"])

    def test_sequential_clauses_see_the_previous_clause(self):
        service = CommandService()
        send(service, "make it winter")
        result = send(service, "add a dog and remove the dog")
        self.assertFalse(result["state"]["unmatched"])
        self.assertEqual([c["label"] for c in result["state"]["changes"]], ["winter"])

    def test_noun_lists_and_pace(self):
        service = CommandService()
        result = send(service, "slowly add a dog and a cat then make it night")
        self.assertEqual([c["label"] for c in result["state"]["changes"]], ["dog", "cat", "night"])
        self.assertEqual(result["steer"]["ramp_chunks"], 12)

    def test_a_rejected_clause_rolls_back_the_whole_sentence(self):
        service = CommandService()
        send(service, "make it winter")
        result = send(service, "add a dog and do not add a river")
        self.assertTrue(result["state"]["unmatched"])
        self.assertEqual([c["label"] for c in result["state"]["changes"]], ["winter"])

    def test_invalid_batch_is_atomic_and_nested_batch_is_rejected(self):
        state = SceneState()
        before = state.snapshot()
        valid = dict(op="add", label="dog", kind="extra", phrase="a dog", strength=1)
        with self.assertRaises(ValueError):
            state.apply(dict(op="batch", operations=[valid, {"op": "invalid"}]))
        self.assertEqual(state.snapshot(), before)
        for operations in ([], [valid] * 17, [dict(op="batch", operations=[valid])]):
            with self.subTest(operations=operations), self.assertRaises(ValueError):
                state.apply(dict(op="batch", operations=operations))
        self.assertEqual(state.snapshot(), before)


if __name__ == "__main__":
    unittest.main()

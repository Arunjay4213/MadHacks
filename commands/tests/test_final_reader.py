"""Engine-facing methods for its async LLM final reader: snapshot() and apply_final() (board #6885800061)."""

import unittest

from commands.service import CommandService

DOG = {"op": "add", "label": "dog", "kind": "extra", "phrase": "a dog standing nearby", "strength": 1.0}
RIVER = {"op": "add", "label": "river", "kind": "extra", "phrase": "a river running through the scene", "strength": 1.0}


def labels(result):
    return [c["label"] for c in result["state"]["changes"]]


def svc():
    return CommandService(semantic_reader=lambda t, s: None)


class SnapshotTest(unittest.TestCase):
    def test_snapshot_is_an_independent_copy(self):
        s = svc()
        s.apply_final("add a dog", [DOG])
        snap = s.snapshot()
        snap["changes"][0]["label"] = "mutated"
        snap["changes"].clear()
        self.assertEqual([c["label"] for c in s.snapshot()["changes"]], ["dog"])


class ApplyFinalTest(unittest.TestCase):
    def test_operations_apply_atomically_with_canonical_prompt(self):
        s = svc()
        r = s.apply_final("add a dog and a river", [DOG, RIVER])
        self.assertEqual((labels(r), r["state"]["unmatched"], r["state"]["heard"], r["state"]["keyword"]),
                         (["dog", "river"], False, "add a dog and a river", None))
        self.assertEqual(r["steer"]["prompt"], s.committed_prompt)
        self.assertIn("a dog standing nearby", s.committed_prompt)
        u = s.handle({"type": "tap", "action": "undo"})
        self.assertEqual(labels(u), [])  # one undo step for the whole sentence

    def test_invalid_operations_roll_back_everything(self):
        for ops in [[DOG, {"op": "explode"}], [DOG] * 17, "x", [3], [{"op": "add", "label": "x", "kind": "planet", "phrase": "p"}]]:
            s = svc()
            r = s.apply_final("bad", ops)
            self.assertEqual((labels(r), r["state"]["unmatched"], r["steer"]), ([], True, None), ops)

    def test_world_base_conflict_is_caught_before_state_changes(self):
        s = CommandService(base="a quiet beach at dawn", semantic_reader=lambda t, s: None)
        desert = {"op": "add", "label": "desert", "kind": "world", "phrase": "a long straight desert highway.", "strength": 1.0}
        r = s.apply_final("make it a desert", [desert])
        self.assertEqual((labels(r), r["state"]["unmatched"]), ([], True))
        self.assertEqual(s.committed_prompt, "a quiet beach at dawn")

    def test_blocked_is_empty_operations(self):
        s = svc()
        base = s.committed_prompt
        s.handle({"type": "partial", "text": "make it winter", "t_client": 1})
        r = s.apply_final("don't make it winter", [])
        self.assertEqual((labels(r), r["state"]["unmatched"]), ([], True))
        self.assertEqual(r["steer"]["prompt"], base)  # the provisional guess is rolled back
        self.assertIsNone(svc().apply_final("nope", [])["steer"])  # no guess: nothing to send

    def test_valid_noop(self):
        s = svc()
        r = s.apply_final("undo", [{"op": "undo"}])
        self.assertEqual((r["steer"], r["state"]["unmatched"]), (None, False))  # valid, nothing changed, same prompt

    def test_constructor_hook_is_gone(self):
        with self.assertRaises(TypeError):
            CommandService(final_reader=lambda t, s: None)


if __name__ == "__main__":
    unittest.main()


class NonLatinTextTest(unittest.TestCase):
    """Greptile: normalization drops non-Latin letters, so the slow reader must get the original text."""

    def slow_service(self, reply):
        self.seen = []

        def slow(text, snapshot):
            self.seen.append(text)
            return reply
        return CommandService(slow_reader=slow, slow_timeout_s=1.0, semantic_reader=lambda t, s: None)

    def test_chinese_and_mixed_text_reach_the_slow_reader_intact(self):
        for text in ["把天空变成夜晚", "make it 下雪", "добавь собаку"]:
            s = self.slow_service(DOG)
            r = s.handle({"type": "final", "text": text, "t_client": 1})
            self.assertEqual(self.seen, [text], text)
            self.assertEqual((labels(r), r["state"]["unmatched"], r["state"]["heard"]), (["dog"], False, text))

    def test_no_answer_means_no_partial_edit(self):
        s = self.slow_service(None)
        r = s.handle({"type": "final", "text": "make it 下雪", "t_client": 1})
        self.assertEqual((labels(r), r["state"]["unmatched"], r["steer"]), ([], True, None))
        offline = CommandService(semantic_reader=lambda t, s: None)  # no LLM configured at all
        r = offline.handle({"type": "final", "text": "make it 下雪", "t_client": 1})
        self.assertEqual((labels(r), r["state"]["unmatched"]), ([], True))

    def test_symbols_and_fullwidth_latin_still_use_the_dictionary(self):
        s = self.slow_service(None)
        for text in ["make it winter ❄", "ｍａｋｅ ｉｔ ｗｉｎｔｅｒ", "make it winter — please"]:
            s = self.slow_service(None)
            self.assertEqual(labels(s.handle({"type": "final", "text": text, "t_client": 1})), ["winter"], text)
            self.assertEqual(self.seen, [])


class BoundedSemanticTest(unittest.TestCase):
    """Greptile: the vector call ran unbounded under the lock and ate the LLM budget."""

    def test_slow_vector_call_times_out_to_the_llm_and_never_applies_late(self):
        import threading
        import time
        release, slow_seen, vector_calls = threading.Event(), [], []

        def vector(text, snapshot):
            vector_calls.append(text)
            release.wait(5)
            return {"op": "add", "label": "fog", "kind": "weather", "phrase": "a thick fog in the air", "strength": 1.0}

        def slow(text, snapshot):
            slow_seen.append(text)
            return DOG
        s = CommandService(semantic_reader=vector, slow_reader=slow, slow_timeout_s=1.5)
        t0 = time.monotonic()
        r = s.handle({"type": "final", "text": "it feels gloomy and grey", "t_client": 1})
        self.assertLess(time.monotonic() - t0, 0.5)  # waited ~50 ms for the vector, then asked the LLM
        self.assertEqual((labels(r), slow_seen), (["dog"], ["it feels gloomy and grey"]))
        r = s.handle({"type": "final", "text": "it feels moody", "t_client": 2})
        self.assertEqual(len(vector_calls), 1)  # the hung vector call blocks a second vector thread
        release.set()
        s._semantic_thread.join(1)
        self.assertNotIn("fog", [c["label"] for c in s.snapshot()["changes"]])  # the late answer is discarded

    def test_fast_vector_answer_is_still_used(self):
        fog = {"op": "add", "label": "fog", "kind": "weather", "phrase": "a thick fog in the air", "strength": 1.0}
        s = CommandService(semantic_reader=lambda t, snap: fog)
        self.assertEqual(labels(s.handle({"type": "final", "text": "it feels gloomy and grey", "t_client": 1})), ["fog"])

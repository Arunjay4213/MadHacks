import threading
import time
import unittest

from commands.reader import RAMP
from commands.service import CommandService
from commands.state import DEFAULT_BASE

try:
    from contracts.validate import errors as contract_errors
except Exception:  # jsonschema not installed: the shape check below still runs
    contract_errors = None


def final(text):
    return {"type": "final", "text": text, "t_client": 1}


def partial(text):
    return {"type": "partial", "text": text, "t_client": 1}


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.svc = CommandService()

    def check_steer(self, steer):
        self.assertEqual(set(steer), {"prompt", "ramp_chunks", "provisional", "cap"})
        if contract_errors:
            self.assertEqual(contract_errors(2, steer), [])

    def test_final_add_stack_remove_undo(self):
        r = self.svc.handle(final("make it winter"))
        self.check_steer(r["steer"])
        self.assertEqual(r["steer"]["ramp_chunks"], RAMP["default"])
        self.assertFalse(r["steer"]["provisional"])
        self.assertIn("snow on the ground", r["steer"]["prompt"])
        self.assertEqual(r["state"], {"changes": [{"id": "c1", "label": "winter", "strength": 1.0}],
                                      "heard": "make it winter", "keyword": "winter", "unmatched": False})
        self.svc.handle(final("slowly add a river"))
        self.assertEqual([c["label"] for c in self.svc.state_fields()["changes"]], ["winter", "river"])
        r = self.svc.handle(final("remove the river"))
        self.assertNotIn("river", r["steer"]["prompt"])
        r = self.svc.handle(final("go back"))
        self.assertIn("river", r["steer"]["prompt"])
        self.assertEqual(r["steer"]["prompt"], self.svc.committed_prompt)

    def test_curly_apostrophe_negation_is_unmatched(self):
        self.assertIsNone(self.svc.handle(partial("don\u2019t make it winter"))["steer"])  # no early guess
        r = self.svc.handle(final("don\u2019t make it winter"))
        self.assertIsNone(r["steer"])
        self.assertTrue(r["state"]["unmatched"])
        self.assertEqual(r["state"]["changes"], [])

    def test_pace_sets_ramp(self):
        self.assertEqual(self.svc.handle(final("slowly turn to night"))["steer"]["ramp_chunks"], RAMP["slow"])
        self.assertEqual(self.svc.handle(final("suddenly rain"))["steer"]["ramp_chunks"], RAMP["sudden"])

    def test_unmatched_and_valid_noop(self):
        r = self.svc.handle(final("turn the tree into a dragon"))
        self.assertIsNone(r["steer"])
        self.assertTrue(r["state"]["unmatched"])
        r = self.svc.handle(final("undo"))  # valid, nothing to undo
        self.assertIsNone(r["steer"])
        self.assertFalse(r["state"]["unmatched"])

    def test_early_guess_confirmed_uses_short_ramp(self):
        r = self.svc.handle(partial("make it winter and"))
        self.check_steer(r["steer"])
        self.assertTrue(r["steer"]["provisional"])
        self.assertEqual(r["steer"]["cap"], 0.4)
        self.assertEqual(self.svc.state_fields()["changes"], [])  # a guess never touches the state
        self.assertIsNone(self.svc.handle(partial("make it winter and cold"))["steer"])  # same guess
        r = self.svc.handle(final("make it winter"))
        self.assertEqual((r["steer"]["ramp_chunks"], r["steer"]["provisional"]), (RAMP["confirm"], False))

    def test_wrong_guess_rolls_back_on_unmatched_or_noop_final(self):
        base_prompt = self.svc.committed_prompt
        self.svc.handle(partial("make it night"))
        r = self.svc.handle(final("make it nice"))
        self.assertEqual(r["steer"], {"prompt": base_prompt, "ramp_chunks": RAMP["default"], "provisional": False, "cap": 1.0})
        self.svc.handle(partial("rain"))
        r = self.svc.handle(final("undo"))  # valid no-op still rolls back
        self.assertEqual(r["steer"]["prompt"], base_prompt)

    def test_revised_partial_rolls_back_and_new_word_reguesses(self):
        self.svc.handle(partial("make it night"))
        r = self.svc.handle(partial("make it nice"))
        self.assertEqual((r["steer"]["prompt"], r["steer"]["provisional"]), (self.svc.committed_prompt, False))
        self.assertIsNone(self.svc.handle(partial("make it nicer"))["steer"])  # nothing in flight now
        self.svc.handle(partial("night"))
        r = self.svc.handle(partial("fog"))
        self.assertTrue(r["steer"]["provisional"])
        self.assertIn("fog", r["steer"]["prompt"])

    def test_taps(self):
        self.svc.handle(final("make it winter"))
        self.svc.handle(final("add a river"))
        r = self.svc.handle({"type": "tap", "action": "remove", "id": "c1"})
        self.assertEqual([c["label"] for c in r["state"]["changes"]], ["river"])
        self.assertIsNone(r["state"]["heard"])
        self.assertIsNone(self.svc.handle({"type": "tap", "action": "remove", "id": "c99"})["steer"])
        self.svc.handle(partial("night"))
        r = self.svc.handle({"type": "tap", "action": "remove", "id": "c99"})  # tap no-op rolls back the guess
        self.assertEqual(r["steer"]["prompt"], self.svc.committed_prompt)
        r = self.svc.handle({"type": "tap", "action": "reset"})
        self.assertEqual(r["steer"]["prompt"], DEFAULT_BASE)
        r = self.svc.handle({"type": "tap", "action": "undo"})
        self.assertIn("river", r["steer"]["prompt"])

    def test_reset_always_steers_to_base_even_when_already_there(self):
        # The engine clears the frame cache and restarts on this prompt (board #6884241134), so it must never be None.
        svc = CommandService(base="a quiet beach")
        r = svc.handle({"type": "tap", "action": "reset"})  # already at base: no state change
        self.check_steer(r["steer"])
        self.assertEqual(r["steer"], {"prompt": "a quiet beach", "ramp_chunks": RAMP["sudden"], "provisional": False, "cap": 1.0})
        svc.handle(final("make it winter"))
        svc.handle(partial("night"))  # a guess in flight is dropped by the reset
        r = svc.handle({"type": "tap", "action": "reset"})
        self.assertEqual(r["steer"]["prompt"], "a quiet beach")
        self.assertEqual(svc.committed_prompt, "a quiet beach")
        self.assertTrue(svc.handle(partial("night and"))["steer"]["provisional"])  # the old guess was cleared, so this is a new one
        self.assertEqual(svc.state_fields()["changes"], [])

    def test_voice_reset_keeps_normal_ramp_and_base_wording(self):
        self.svc.handle(final("make it winter"))
        r = self.svc.handle(final("start over"))
        self.assertEqual(r["steer"], {"prompt": DEFAULT_BASE, "ramp_chunks": RAMP["default"], "provisional": False, "cap": 1.0})
        self.assertIsNone(self.svc.handle(final("start over"))["steer"])  # voice reset at base is a plain no-op

    def test_invalid_contract1_raises(self):
        for bad in [None, [], {}, {"type": "shout"}, {"type": "final", "text": "x"}, {"type": "final", "text": 3, "t_client": 1},
                    {"type": "final", "text": "x", "t_client": True}, {"type": "tap"}, {"type": "tap", "action": "remove"},
                    {"type": "tap", "action": "explode"}]:
            with self.assertRaises(ValueError, msg=bad):
                self.svc.handle(bad)

    def test_likely_prompts_do_not_touch_state(self):
        prompts = self.svc.likely_prompts()
        self.assertIn(self.svc.committed_prompt, prompts)
        self.assertTrue(any("river" in p for p in prompts))
        self.assertEqual(self.svc.state_fields()["changes"], [])


class SlowReaderTest(unittest.TestCase):
    def test_used_on_fast_miss_and_validated(self):
        svc = CommandService(slow_reader=lambda text, snap: {"op": "new_scene", "base": "a lighthouse on a cliff"})
        r = svc.handle(final("put the whole thing on a cliff edge"))
        self.assertEqual(r["steer"]["prompt"], "a lighthouse on a cliff")
        self.assertFalse(r["state"]["unmatched"])
        bad = CommandService(slow_reader=lambda text, snap: {"op": "set_strength", "id": "c1", "strength": 9})
        self.assertTrue(bad.handle(final("blorp"))["state"]["unmatched"])
        junk = CommandService(slow_reader=lambda text, snap: "not a dict")
        self.assertTrue(junk.handle(final("blorp"))["state"]["unmatched"])

    def test_exception_is_unmatched(self):
        def boom(text, snap):
            raise RuntimeError("network down")
        self.assertTrue(CommandService(slow_reader=boom).handle(final("blorp"))["state"]["unmatched"])

    def test_timeout_and_at_most_one_in_flight(self):
        release = threading.Event()
        calls = []

        def hang(text, snap):
            calls.append(text)
            release.wait(5)
            return {"op": "new_scene", "base": "late"}

        svc = CommandService(slow_reader=hang, slow_timeout_s=0.05)
        t0 = time.monotonic()
        r = svc.handle(final("blorp one"))
        self.assertLess(time.monotonic() - t0, 1)
        self.assertTrue(r["state"]["unmatched"])
        self.assertTrue(svc.handle(final("blorp two"))["state"]["unmatched"])
        self.assertEqual(calls, ["blorp one"])  # second miss did not start another thread
        release.set()
        svc._slow_thread.join(1)
        self.assertEqual(svc.state_fields()["changes"], [])  # the late answer is discarded
        self.assertNotEqual(svc.committed_prompt, "late")


if __name__ == "__main__":
    unittest.main()

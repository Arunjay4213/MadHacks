"""Exercise service targets with the real steering controller and CPU fake backend."""

import unittest


class PipelineTest(unittest.TestCase):
    def test_guesses_commits_and_undo_reach_decoded_frames(self):
        try:
            import torch
            from engine.backends.fake import load
            from steering import Controller, Steerer
        except ImportError as exc:
            self.skipTest(f"engine verification dependencies are optional: {exc}")
        from commands.service import CommandService
        from contracts.validate import errors

        service = CommandService(base="a meadow")
        backend = load(64, 48, 2, gen_ms=0, decode_ms=0, encode_ms=0)
        initial = backend.encode_prompt(service.committed_prompt)
        steerer = Steerer(initial)
        controller = Controller(steerer, backend.encode_prompt)
        controller.precompute(service.likely_prompts())

        def command(kind, text):
            result = service.handle(dict(type=kind, text=text, t_client=1))
            target = result["steer"]
            self.assertEqual(errors(2, target), [])
            controller.handle(target)
            for _ in range(target["ramp_chunks"]):
                cond = steerer.get_cond()
            frames = backend.decode_chunk(backend.generate_chunk(cond))
            self.assertEqual(frames.shape[-3:], (48, 64, 3))
            return result, cond

        result, provisional = command("partial", "make it winter")
        self.assertEqual(result["state"]["changes"], [])
        self.assertFalse(torch.equal(provisional, initial))
        _, restored = command("final", "make it nice")
        self.assertTrue(torch.allclose(restored, initial, atol=1e-5))
        for text in ("make it winter", "add a river", "turn to night"):
            result, _ = command("final", text)
        self.assertEqual([c["label"] for c in result["state"]["changes"]], ["winter", "river", "night"])
        result, _ = command("final", "remove the river")
        self.assertEqual([c["label"] for c in result["state"]["changes"]], ["winter", "night"])
        result, _ = command("final", "go back")
        self.assertEqual([c["label"] for c in result["state"]["changes"]], ["winter", "river", "night"])


if __name__ == "__main__":
    unittest.main()

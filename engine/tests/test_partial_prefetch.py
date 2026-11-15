"""Typing sends a partial per keystroke. A burst of partials must not redo the prefetch work or queue encodes.

The live engine is CPU bound on one core, so the reader thread's work per keystroke slows generation directly.
Asking the reader for its likely prompts after every partial (one scene copy per keyword) raised gen time per
chunk from about 195 to 280 ms and stalled the stream while someone typed (live journal, 2026-11-15 07:27).
"""

import threading
import time

from commands.service import CommandService
from engine.engine import Engine, EngineConfig
from engine.lifecycle import Lifecycle


def wait(pred, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


class CountingReader:
    """The real commands/ reader, counting how often the engine asks for its likely prompts."""

    def __init__(self):
        self._svc = CommandService()
        self.likely_calls = 0

    def handle(self, msg):
        return self._svc.handle(msg)

    def state_fields(self):
        return self._svc.state_fields()

    def likely_prompts(self):
        self.likely_calls += 1
        return self._svc.likely_prompts()

    @property
    def committed_prompt(self):
        return self._svc.committed_prompt


def test_a_burst_of_partials_causes_at_most_one_encode_and_no_prefetch_work():
    reader = CountingReader()
    cfg = EngineConfig(backend="fake", width=64, height=48, backend_options=dict(
        gen_ms=20, decode_ms=5, encode_ms=5, frames_per_chunk=3, fps=50))
    e = Engine(cfg, lambda b: None, lambda m: None, reader=reader)
    e.start()
    try:
        assert wait(lambda: e.lifecycle.state is Lifecycle.GENERATING)
        encodes, lock = [], threading.Lock()
        real_encode = e.encoder._encode

        def counting_encode(prompt):
            with lock:
                encodes.append(prompt)
            return real_encode(prompt)
        e.encoder._encode = counting_encode
        likely_before = reader.likely_calls

        sentence = "make it winter with snow on the old oak tree"
        futures = [e.handle_control({"type": "partial", "text": sentence[:k].strip(), "t_client": k},
                                    time.monotonic_ns())
                   for k in range(1, len(sentence) + 1) if sentence[:k].strip()]
        for fut in futures:
            fut.result(5)
        assert wait(lambda: e.encoder.pending == 0 and not e.book.in_flight())
        assert reader.likely_calls == likely_before  # a partial never changes the scene: nothing to prefetch
        assert len(encodes) <= 1, encodes  # at most the provisional guess, which is usually already cached

        e.handle_control({"type": "final", "text": sentence, "t_client": 999}, time.monotonic_ns()).result(5)
        assert reader.likely_calls == likely_before + 1  # a final changes the scene: prefetch for the new one
    finally:
        e.stop()

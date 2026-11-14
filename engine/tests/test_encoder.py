import threading
import time

import pytest

from engine.encoder import EncoderService


def make(delay=0.05, size=512):
    calls = []

    def enc(prompt):
        calls.append(prompt)
        time.sleep(delay)
        return f"emb:{prompt}", [len(prompt)]

    svc = EncoderService(enc, cache_size=size)
    svc.start()
    return svc, calls


def test_miss_then_hit_is_immediate():
    svc, calls = make()
    assert svc.encode("a") == "emb:a"
    t0 = time.perf_counter()
    fut = svc.request("a")
    assert fut.done() and (time.perf_counter() - t0) < 0.01
    assert svc.encode_with_ids("a") == ("emb:a", [1])
    assert calls == ["a"] and svc.hits == 2 and svc.misses == 1
    svc.stop()


def test_concurrent_identical_requests_share_one_encode():
    svc, calls = make(delay=0.1)
    futs = [svc.request("same") for _ in range(5)]
    assert all(f.result() == ("emb:same", [4]) for f in futs)
    assert calls == ["same"]
    svc.stop()


def test_runs_off_the_calling_thread():
    seen = []
    svc = EncoderService(lambda p: (seen.append(threading.current_thread().name), (p, []))[1])
    svc.start()
    svc.encode("x")
    assert seen == ["encoder"]
    svc.stop()


def test_lru_bound():
    svc, calls = make(delay=0, size=2)
    for p in ("a", "b", "c"):
        svc.encode(p)
    assert svc.cache_size == 2
    svc.encode("a")  # evicted, so encoded again
    assert calls == ["a", "b", "c", "a"]
    svc.stop()


def test_error_reaches_caller_and_service_survives():
    def enc(prompt):
        if prompt == "bad":
            raise RuntimeError("boom")
        return prompt, []

    svc = EncoderService(enc)
    svc.start()
    with pytest.raises(RuntimeError):
        svc.encode("bad")
    assert svc.encode("good") == "good"
    svc.stop()


def test_prefetch_runs_behind_urgent_requests():
    order = []
    gate = threading.Event()

    def enc(prompt):
        if prompt == "first":
            gate.wait(2)
        order.append(prompt)
        return prompt, []

    svc = EncoderService(enc)
    svc.start()
    first = svc.request("first")  # occupies the encoder thread
    time.sleep(0.05)
    assert svc.prefetch(["bg1", "bg2", "first"]) == 2  # "first" is already in flight
    urgent = svc.request("now")
    promoted = svc.request("bg2")  # was queued for prefetch, becomes urgent
    gate.set()
    for f in (first, urgent, promoted):
        f.result(2)
    svc.encode("bg1")
    assert order == ["first", "now", "bg2", "bg1"]
    assert svc.prefetch(["now"]) == 0
    svc.stop()

"""PrefetchPacer: background encodes wait while a command is in flight and use only generation's idle time."""

import threading

from engine.engine import PrefetchPacer, PrefetchState


def test_held_while_a_command_is_in_flight():
    p = PrefetchPacer()
    assert p.state is PrefetchState.HELD
    assert not p.chunk_done(idle_ms=500, in_flight=lambda: True, cost_ms=50)
    assert p.state is PrefetchState.HELD and not p.take()


def test_saves_idle_time_until_one_encode_fits():
    p = PrefetchPacer()
    assert not p.chunk_done(idle_ms=30, in_flight=lambda: False, cost_ms=50)
    assert p.state is PrefetchState.SAVING and not p.take()
    assert p.chunk_done(idle_ms=30, in_flight=lambda: False, cost_ms=50)  # 60 saved >= 50
    assert p.state is PrefetchState.READY
    assert not p.chunk_done(idle_ms=30, in_flight=lambda: False, cost_ms=50)  # one turn at a time
    assert p.take() and p.state is PrefetchState.SAVING
    assert not p.take()
    assert p.chunk_done(idle_ms=40, in_flight=lambda: False, cost_ms=50)  # 10 left over + 40


def test_a_command_takes_back_a_ready_turn():
    p = PrefetchPacer()
    assert p.chunk_done(idle_ms=200, in_flight=lambda: False, cost_ms=50)
    p.hold()
    assert p.state is PrefetchState.HELD and not p.take()
    assert not p.chunk_done(idle_ms=30, in_flight=lambda: False, cost_ms=50)  # saving starts again from zero
    assert p.state is PrefetchState.SAVING


def test_default_cost_before_any_encode():
    p = PrefetchPacer()
    assert not p.chunk_done(idle_ms=PrefetchPacer.DEFAULT_COST_MS - 1, in_flight=lambda: False, cost_ms=None)
    assert p.chunk_done(idle_ms=1, in_flight=lambda: False, cost_ms=None)



def test_a_hold_during_chunk_done_is_not_overwritten():
    """A command thread holds while the generation thread is in chunk_done, just after in_flight() said no."""
    p = PrefetchPacer()
    holder = threading.Thread(target=p.hold)

    def in_flight() -> bool:
        holder.start()  # the command arrives now; its hold waits for the pacer's lock
        holder.join(0.05)
        return False

    assert p.chunk_done(idle_ms=200, in_flight=in_flight, cost_ms=50)
    holder.join(2)
    assert p.state is PrefetchState.HELD and not p.take()

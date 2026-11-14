"""PrefetchPacer: background encodes wait while a command is in flight and use only generation's idle time."""

from engine.engine import PrefetchPacer, PrefetchState


def test_held_while_a_command_is_in_flight():
    p = PrefetchPacer()
    assert p.state is PrefetchState.HELD
    assert not p.chunk_done(idle_ms=500, in_flight=True, cost_ms=50)
    assert p.state is PrefetchState.HELD and not p.take()


def test_saves_idle_time_until_one_encode_fits():
    p = PrefetchPacer()
    assert not p.chunk_done(idle_ms=30, in_flight=False, cost_ms=50)
    assert p.state is PrefetchState.SAVING and not p.take()
    assert p.chunk_done(idle_ms=30, in_flight=False, cost_ms=50)  # 60 saved >= 50
    assert p.state is PrefetchState.READY
    assert not p.chunk_done(idle_ms=30, in_flight=False, cost_ms=50)  # one turn at a time
    assert p.take() and p.state is PrefetchState.SAVING
    assert not p.take()
    assert p.chunk_done(idle_ms=40, in_flight=False, cost_ms=50)  # 10 left over + 40


def test_a_command_takes_back_a_ready_turn():
    p = PrefetchPacer()
    assert p.chunk_done(idle_ms=200, in_flight=False, cost_ms=50)
    p.hold()
    assert p.state is PrefetchState.HELD and not p.take()
    assert not p.chunk_done(idle_ms=30, in_flight=False, cost_ms=50)  # saving starts again from zero
    assert p.state is PrefetchState.SAVING


def test_default_cost_before_any_encode():
    p = PrefetchPacer()
    assert not p.chunk_done(idle_ms=PrefetchPacer.DEFAULT_COST_MS - 1, in_flight=False, cost_ms=None)
    assert p.chunk_done(idle_ms=1, in_flight=False, cost_ms=None)


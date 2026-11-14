import pytest

from engine.lifecycle import ALLOWED, IllegalTransition, Lifecycle, LifecycleMachine


def test_normal_path_and_reset():
    m = LifecycleMachine()
    assert m.state is Lifecycle.LOADING
    for s in (Lifecycle.WARMING, Lifecycle.GENERATING, Lifecycle.RESETTING, Lifecycle.GENERATING, Lifecycle.STOPPED):
        m.transition(s)
        assert m.state is s


@pytest.mark.parametrize("start,bad", [
    (Lifecycle.LOADING, Lifecycle.GENERATING),
    (Lifecycle.WARMING, Lifecycle.RESETTING),
    (Lifecycle.GENERATING, Lifecycle.WARMING),
    (Lifecycle.FAILED, Lifecycle.GENERATING),
])
def test_illegal_transitions_rejected(start, bad):
    m = LifecycleMachine()
    path = {Lifecycle.LOADING: [], Lifecycle.WARMING: [Lifecycle.WARMING],
            Lifecycle.GENERATING: [Lifecycle.WARMING, Lifecycle.GENERATING], Lifecycle.FAILED: [Lifecycle.FAILED]}
    for s in path[start]:
        m.transition(s)
    with pytest.raises(IllegalTransition):
        m.transition(bad)
    assert m.state is start


def test_stopped_is_final_and_every_state_listed():
    assert ALLOWED[Lifecycle.STOPPED] == frozenset()
    assert set(ALLOWED) == set(Lifecycle)

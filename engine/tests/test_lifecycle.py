import pytest

from engine.lifecycle import ALLOWED, IllegalTransition, Lifecycle, LifecycleMachine

L = Lifecycle


def test_legal_paths_through_reset_and_recovery():
    for path in ([L.WARMING, L.GENERATING, L.RESETTING, L.GENERATING, L.STOPPED],
                 [L.WARMING, L.GENERATING, L.RECOVERING, L.GENERATING, L.RESETTING, L.RECOVERING, L.FAILED,
                  L.STOPPED]):
        m = LifecycleMachine()
        assert m.state is L.LOADING
        for s in path:
            m.transition(s)
            assert m.state is s


def test_illegal_transitions_are_rejected_and_leave_the_state_alone():
    for before, bad in [([], L.GENERATING), ([L.WARMING], L.RESETTING), ([L.WARMING], L.RECOVERING),
                        ([L.WARMING, L.GENERATING], L.WARMING), ([L.FAILED], L.GENERATING)]:
        m = LifecycleMachine()
        for s in before:
            m.transition(s)
        start = m.state
        with pytest.raises(IllegalTransition):
            m.transition(bad)
        assert m.state is start
    assert ALLOWED[L.STOPPED] == frozenset()
    assert set(ALLOWED) == set(Lifecycle)

import pytest

from steering import ease


def test_endpoints_and_midpoint():
    assert ease(0.0) == 0.0
    assert ease(1.0) == 1.0
    assert ease(0.5) == pytest.approx(0.5)


def test_clamps_outside_unit_interval():
    assert ease(-0.5) == 0.0
    assert ease(-100.0) == 0.0
    assert ease(1.5) == 1.0
    assert ease(100.0) == 1.0


def test_monotonic_non_decreasing():
    grid = [i / 1000 for i in range(1001)]
    values = [ease(r) for r in grid]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_symmetric():
    for i in range(101):
        r = i / 100
        assert ease(r) + ease(1 - r) == pytest.approx(1.0, abs=1e-12)

"""DataGenerator output checks: check_series + drain_generator."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from metronome.interface.generator import DataGenerator, check_series, drain_generator


class _Good(DataGenerator):
    def __init__(self, config_dir: str = ".", *, seed: int = 0) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "good"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._seed)
        for _ in range(n_series):
            yield rng.standard_normal(100)


def test_check_series_accepts_valid():
    check_series(np.zeros(100), min_length=10, max_length=200)


@pytest.mark.parametrize(
    "arr",
    [
        [1.0, 2.0],                       # not ndarray
        np.zeros((2, 2)),                 # 2-D
        np.arange(100),                   # int dtype
        np.zeros(5),                      # too short
        np.zeros(500),                    # too long
        np.array([np.nan, 1.0, 2.0] * 10),  # non-finite
    ],
)
def test_check_series_rejects(arr):
    with pytest.raises(ValueError):
        check_series(arr, min_length=10, max_length=200)


def test_drain_generator_returns_exact_count():
    out = drain_generator(_Good(seed=1), 5, min_length=10, max_length=200, max_total_points=10_000)
    assert len(out) == 5
    assert all(a.dtype == np.float64 for a in out)


def test_drain_generator_determinism():
    a = drain_generator(_Good(seed=7), 4, min_length=10, max_length=200, max_total_points=10_000)
    b = drain_generator(_Good(seed=7), 4, min_length=10, max_length=200, max_total_points=10_000)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_drain_generator_enforces_point_cap():
    with pytest.raises(ValueError):
        drain_generator(_Good(seed=1), 5, min_length=10, max_length=200, max_total_points=150)


class _WrongCount(DataGenerator):
    def __init__(self, config_dir: str = ".", *, seed: int = 0) -> None:
        pass

    @property
    def name(self) -> str:
        return "wrong"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        yield np.zeros(50)  # yields 1 regardless of n_series


def test_drain_generator_rejects_wrong_count():
    with pytest.raises(ValueError):
        drain_generator(_WrongCount(), 5, min_length=10, max_length=200, max_total_points=10_000)

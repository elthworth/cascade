from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "cascade_v18_generator", REPO / "generator.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_config_matches_complete_default_mixture():
    module = _module()
    config = json.loads((REPO / "config.json").read_text())
    assert set(module._FAMILIES) == set(module._DEFAULT_WEIGHTS)
    assert config["family_weights"] == module._DEFAULT_WEIGHTS
    assert np.isclose(sum(module._DEFAULT_WEIGHTS.values()), 1.0)
    assert module._DEFAULT_WEIGHTS["weekly_demand"] == 0.08


def test_weekly_demand_is_deterministic_nonnegative_and_finite():
    module = _module()
    left = module._weekly_demand(np.random.default_rng(1818), 512, 512)
    right = module._weekly_demand(np.random.default_rng(1818), 512, 512)
    assert np.array_equal(left, right)
    assert left.shape == (512, 512)
    assert np.isfinite(left).all()
    assert (left >= 0.0).all()
    assert np.all(left.std(axis=1) > 1e-9)


def test_weekly_demand_contains_period_seven_and_count_rows():
    module = _module()
    rows = module._weekly_demand(np.random.default_rng(1919), 1024, 512)
    lag_seven = np.asarray(
        [np.corrcoef(row[:-7], row[7:])[0, 1] for row in rows]
    )
    assert np.median(lag_seven) > 0.5

    integer_rows = np.all(rows == np.rint(rows), axis=1)
    assert 0.25 < integer_rows.mean() < 0.45

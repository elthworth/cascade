from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def _generator():
    spec = importlib.util.spec_from_file_location("cascade_v2_generator", REPO / "generator.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Generator(str(REPO), seed=123)


def _module():
    spec = importlib.util.spec_from_file_location("cascade_v2_module", REPO / "generator.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_exact_count_full_context_and_finite():
    rows = list(_generator().generate(1031))
    assert len(rows) == 1031
    assert all(row.shape == (4096,) for row in rows)
    assert all(row.dtype == np.float64 for row in rows)
    assert all(np.isfinite(row).all() and row.std() > 1e-9 for row in rows)


def test_deterministic_across_instances():
    left = list(_generator().generate(1030))
    right = list(_generator().generate(1030))
    assert all(np.array_equal(a, b) for a, b in zip(left, right, strict=True))


def test_requested_size_does_not_change_prefix():
    short = list(_generator().generate(17))
    long = list(_generator().generate(1030))
    assert all(np.array_equal(a, b) for a, b in zip(short, long[:17], strict=True))


def test_nonpositive_request_is_empty():
    assert list(_generator().generate(0)) == []


def test_ou_stochastic_vol_is_deterministic_finite_and_active():
    module = _module()
    left = module._ou_stochastic_vol(np.random.default_rng(77), 16, 512)
    right = module._ou_stochastic_vol(np.random.default_rng(77), 16, 512)
    assert np.array_equal(left, right)
    assert np.isfinite(left).all()
    assert (left.std(axis=1) > 1e-9).all()
    assert "ou_stochastic_vol" in module._FAMILIES
    assert module._DEFAULT_WEIGHTS["ou_stochastic_vol"] > 0

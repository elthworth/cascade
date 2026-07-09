"""CRPS (MWSQL) + MASE sanity and the paired bootstrap."""

from __future__ import annotations

import numpy as np

from cascade.eval.bootstrap import paired_bootstrap_lcb, paired_bootstrap_lcb_aggregated
from cascade.eval.crps import mwsql_components, mwsql_from_components
from cascade.eval.mase import in_sample_naive_mae, mase


def test_mwsql_perfect_forecast_is_zero():
    obs = np.array([[1.0, 2.0, 3.0]])
    samples = np.broadcast_to(obs[:, None, :], (1, 50, 3)).copy()
    qloss, abs_t = mwsql_components(samples, obs)
    assert mwsql_from_components(qloss, abs_t) < 1e-9


def test_mwsql_worse_forecast_scores_higher():
    obs = np.ones((1, 4))
    good = np.broadcast_to(obs[:, None, :], (1, 50, 4)).copy() + 0.05 * np.random.default_rng(0).standard_normal((1, 50, 4))
    bad = good + 5.0
    g = mwsql_from_components(*mwsql_components(good, obs))
    b = mwsql_from_components(*mwsql_components(bad, obs))
    assert b > g


def test_mase_perfect_is_zero_and_scale_floor():
    hist = np.arange(50, dtype=np.float64)
    obs = np.array([50.0, 51.0, 52.0])
    assert mase(obs.copy(), obs, hist, seasonal_period=1) == 0.0
    assert in_sample_naive_mae(np.zeros(50), 1) >= 1e-9


def test_paired_bootstrap_lcb_detects_clear_winner():
    rng = np.random.default_rng(0)
    king = rng.uniform(1.0, 2.0, size=400)
    chal = king * 0.7  # challenger 30% better on every window
    lcb = paired_bootstrap_lcb(king, chal, alpha=0.05, B=2000, seed="x")
    assert lcb > 0.2


def test_paired_bootstrap_lcb_no_improvement_is_nonpositive():
    rng = np.random.default_rng(1)
    king = rng.uniform(1.0, 2.0, size=400)
    chal = king.copy()
    lcb = paired_bootstrap_lcb(king, chal, alpha=0.05, B=2000, seed="x")
    assert lcb <= 1e-6


def test_paired_bootstrap_is_deterministic_in_seed():
    rng = np.random.default_rng(2)
    king = rng.uniform(1.0, 2.0, size=200)
    chal = king * 0.9
    a = paired_bootstrap_lcb(king, chal, seed="block-hash-abc", B=1000)
    b = paired_bootstrap_lcb(king, chal, seed="block-hash-abc", B=1000)
    assert a == b


def _components(n, num_q, scale, seed):
    rng = np.random.default_rng(seed)
    qloss = rng.uniform(0.1, 1.0, size=(n, num_q)) * scale
    abs_t = rng.uniform(5.0, 10.0, size=n)
    mase_a = rng.uniform(0.5, 1.5, size=n) * scale
    return qloss, abs_t, mase_a


def test_aggregated_lcb_requires_paired_abs_target():
    k_q, k_a, k_m = _components(50, 9, 1.0, 0)
    c_q, _, c_m = _components(50, 9, 0.5, 1)
    # Same abs_target (paired windows) → fine.
    lcb = paired_bootstrap_lcb_aggregated(k_q, k_a, k_m, c_q, k_a, c_m, B=500, seed="s")
    assert lcb > 0.0  # challenger has lower qloss/mase scale


def test_aggregated_lcb_rejects_unpaired_windows():
    import pytest

    k_q, k_a, k_m = _components(50, 9, 1.0, 0)
    c_q, c_a, c_m = _components(50, 9, 0.5, 1)  # different abs_target
    with pytest.raises(ValueError):
        paired_bootstrap_lcb_aggregated(k_q, k_a, k_m, c_q, c_a, c_m, B=200, seed="s")


def test_singleton_clusters_match_default_bootstrap():
    rng = np.random.default_rng(7)
    n, nq = 60, 9
    k_q = rng.uniform(0.1, 1.0, size=(n, nq))
    abs_t = rng.uniform(5.0, 10.0, size=n)
    k_m = rng.uniform(0.5, 1.5, size=n)
    c_q, c_m = k_q * 0.9, k_m * 0.9
    base = paired_bootstrap_lcb_aggregated(k_q, abs_t, k_m, c_q, abs_t, c_m, B=500, seed="s")
    singletons = paired_bootstrap_lcb_aggregated(
        k_q, abs_t, k_m, c_q, abs_t, c_m, B=500, seed="s", clusters=list(range(n))
    )
    assert base == singletons


def test_cluster_bootstrap_is_wider_under_cluster_correlation():
    """Windows that move together must not be counted as independent: with a
    strong per-cluster effect the cluster LCB sits below the i.i.d. LCB."""
    rng = np.random.default_rng(11)
    n_clusters, per = 8, 40
    n = n_clusters * per
    labels = [f"feed{i // per}" for i in range(n)]
    k_q = rng.uniform(0.4, 0.6, size=(n, 9))
    abs_t = rng.uniform(5.0, 10.0, size=n)
    k_m = rng.uniform(0.9, 1.1, size=n)
    # Challenger improvement varies BY CLUSTER (correlated), not by window.
    effect = np.repeat(rng.uniform(0.7, 1.1, size=n_clusters), per)
    c_q = k_q * effect[:, None]
    c_m = k_m * effect
    iid = paired_bootstrap_lcb_aggregated(k_q, abs_t, k_m, c_q, abs_t, c_m, B=2000, seed="s")
    clustered = paired_bootstrap_lcb_aggregated(
        k_q, abs_t, k_m, c_q, abs_t, c_m, B=2000, seed="s", clusters=labels
    )
    assert clustered < iid


def test_cluster_bootstrap_deterministic_in_seed_and_labels():
    rng = np.random.default_rng(3)
    n = 40
    k_q = rng.uniform(0.1, 1.0, size=(n, 9))
    abs_t = rng.uniform(5.0, 10.0, size=n)
    k_m = rng.uniform(0.5, 1.5, size=n)
    # Heterogeneous improvement so the bag distribution isn't a point mass
    # (a constant factor cancels in every bag, making all seeds agree).
    factor = rng.uniform(0.6, 1.0, size=n)
    c_q, c_m = k_q * factor[:, None], k_m * factor
    labels = [f"s{i % 5}" for i in range(n)]
    a = paired_bootstrap_lcb_aggregated(k_q, abs_t, k_m, c_q, abs_t, c_m, B=500, seed="x", clusters=labels)
    b = paired_bootstrap_lcb_aggregated(k_q, abs_t, k_m, c_q, abs_t, c_m, B=500, seed="x", clusters=labels)
    assert a == b
    c = paired_bootstrap_lcb_aggregated(k_q, abs_t, k_m, c_q, abs_t, c_m, B=500, seed="y", clusters=labels)
    assert a != c

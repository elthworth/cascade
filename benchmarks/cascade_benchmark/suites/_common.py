"""Shared gluonts evaluation loop for the GIFT-Eval and BOOM runners.

Both expose datasets through gift-eval's ``Dataset`` class, so scoring is
identical and matches gift-eval's own runner (``notebooks/naive.ipynb``):
``evaluate_model`` with the same kwargs and seasonality derived from the
dataset frequency. We report CRPS (= ``mean_weighted_sum_quantile_loss``, the
GIFT-Eval headline) and ``MASE[0.5]`` — the two cascade also scores in-protocol.
"""

from __future__ import annotations

import numpy as np

from ..predictor import CheckpointPredictor

# 9-level grid — matches GIFT-Eval and cascade's DEFAULT_QUANTILE_LEVELS.
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def build_dataset(name: str, term: str, *, storage_env_var: str = "GIFT_EVAL"):
    """Construct a gift-eval ``Dataset``, following the reference runner's
    ``to_univariate`` rule (multivariate series are split into univariate ones,
    matching the leaderboard). Returns ``None`` if the dataset fails to load so
    the caller can skip it rather than abort the sweep.
    """
    from gift_eval.data import Dataset

    try:
        probe = Dataset(name=name, term=term, to_univariate=False, storage_env_var=storage_env_var)
        to_univariate = probe.target_dim != 1
        return Dataset(
            name=name, term=term, to_univariate=to_univariate, storage_env_var=storage_env_var
        )
    except Exception:  # noqa: BLE001 — unknown/invalid (name, term) combo → skip
        return None


def evaluate_datasets(
    datasets,
    checkpoint_dir: str,
    *,
    num_samples: int,
    device: str,
) -> tuple[dict, int]:
    """Score the checkpoint on an iterable of gift-eval ``Dataset`` objects.

    Returns ``(metrics, n_scored)`` with the cross-dataset mean CRPS and MASE.
    A dataset that errors is skipped (not fatal). The ``evaluate_model`` call
    mirrors gift-eval's reference runner exactly so the numbers are comparable.
    """
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    metrics = [MASE(), MeanWeightedSumQuantileLoss(quantile_levels=list(QUANTILE_LEVELS))]

    crps_vals: list[float] = []
    mase_vals: list[float] = []
    scored = 0
    for ds in datasets:
        if ds is None:
            continue
        season_length = get_seasonality(ds.freq)
        predictor = CheckpointPredictor(
            checkpoint_dir,
            prediction_length=ds.prediction_length,
            num_samples=num_samples,
            device=device,
        )
        res = evaluate_model(
            predictor,
            test_data=ds.test_data,
            metrics=metrics,
            batch_size=512,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )
        crps = float(res["mean_weighted_sum_quantile_loss"][0])
        mase = float(res["MASE[0.5]"][0])
        if np.isfinite(crps):
            crps_vals.append(crps)
        if np.isfinite(mase):
            mase_vals.append(mase)
        scored += 1

    out = {
        "crps": float(np.mean(crps_vals)) if crps_vals else float("nan"),
        "mase": float(np.mean(mase_vals)) if mase_vals else float("nan"),
    }
    return out, scored

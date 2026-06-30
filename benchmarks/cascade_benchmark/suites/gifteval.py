"""GIFT-Eval runner — Salesforce's general TS benchmark.

gift-eval does *not* expose the benchmark's dataset list as an importable
constant — its reference runner (``notebooks/naive.ipynb``) hardcodes two
space-separated strings and derives the per-dataset terms from them. We embed
those exact strings (verbatim from the pinned commit) and replicate the term
logic: every dataset is scored on ``short``; only datasets in
``MED_LONG_DATASETS`` are also scored on ``medium`` and ``long``.

Override with ``CASCADE_BENCH_GIFTEVAL_DATASETS`` (comma-separated ``name`` or
``name/freq`` — terms are still auto-applied). Point ``GIFT_EVAL`` at the
downloaded benchmark data (gift-eval's own env var).
"""

from __future__ import annotations

import os
import traceback

from ..results import SuiteResult
from ._common import build_dataset, evaluate_datasets

# Verbatim from gift-eval notebooks/naive.ipynb @ 1527c415 (the full, commented
# lists — the notebook ships them commented for a fast 2-dataset demo).
SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W hospital covid_deaths "
    "us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing kdd_cup_2018_with_missing/H "
    "kdd_cup_2018_with_missing/D car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D "
    "ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W "
    "jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T "
    "bitbrains_rnd/H bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)
MED_LONG_DATASETS = (
    "electricity/15T electricity/H solar/10T solar/H "
    "kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T "
    "M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application "
    "bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
)


def _name_term_pairs(max_tasks: int | None):
    override = os.environ.get("CASCADE_BENCH_GIFTEVAL_DATASETS", "").strip()
    names = override.replace(",", " ").split() if override else SHORT_DATASETS.split()
    med_long = set(MED_LONG_DATASETS.split())

    n = 0
    for ds_name in names:
        for term in ("short", "medium", "long"):
            if term in ("medium", "long") and ds_name not in med_long:
                continue
            yield ds_name, term
            n += 1
            if max_tasks and n >= max_tasks:
                return


def _iter_datasets(max_tasks: int | None):
    for ds_name, term in _name_term_pairs(max_tasks):
        ds = build_dataset(ds_name, term)  # None on load error → skipped downstream
        if ds is not None:
            yield ds


def run(
    checkpoint_dir: str,
    *,
    num_samples: int = 100,
    max_series: int | None = None,
    device: str = "cpu",
) -> SuiteResult:
    if not (os.environ.get("GIFT_EVAL") or os.environ.get("CASCADE_BENCH_GIFTEVAL_DATASETS")):
        return SuiteResult(
            suite="gift-eval",
            status="skipped",
            detail="GIFT_EVAL not set; point it at the downloaded gift-eval benchmark data.",
        )
    try:
        metrics, n = evaluate_datasets(
            _iter_datasets(max_series),
            checkpoint_dir,
            num_samples=num_samples,
            device=device,
        )
        if not n:
            return SuiteResult(suite="gift-eval", status="error", detail="no datasets scored")
        return SuiteResult(suite="gift-eval", status="ok", metrics=metrics, n_series=n)
    except ImportError as e:
        return SuiteResult(suite="gift-eval", status="skipped", detail=f"gift-eval not importable: {e}")
    except Exception as e:  # noqa: BLE001
        return SuiteResult(
            suite="gift-eval",
            status="error",
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )

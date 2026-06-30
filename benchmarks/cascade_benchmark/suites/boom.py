"""BOOM runner — DataDog's observability benchmark (2,807 real prod series).

BOOM is scored through gift-eval's ``Dataset`` (DataDog adapted gift-eval), but
its dataset manifest is *not* part of the installed gift-eval package — the
2,807 configs (``ds-<n>-<freq>``) and, crucially, each config's fixed ``term``
live in DataDog's ``boom_properties.json``. We vendor that file
(``data/boom_properties.json``, from DataDog/toto, Apache-2.0) and iterate it,
constructing one ``Dataset`` per config with its designated term.

Setup:
* ``BOOM`` (or ``CASCADE_BENCH_BOOM_PATH``) → the downloaded BOOM data dir
  (gift-eval layout). Required.
* ``CASCADE_BENCH_BOOM_PROPERTIES`` → override the vendored manifest path.
* ``CASCADE_BENCH_BOOM_DATASETS`` → comma-separated config names to restrict to.

BOOM is large (350M obs); use ``--max-series`` for anything but a full run.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from ..results import SuiteResult
from ._common import build_dataset, evaluate_datasets

_VENDORED_PROPERTIES = Path(__file__).resolve().parent.parent / "data" / "boom_properties.json"


def _load_properties() -> dict:
    path = Path(os.environ.get("CASCADE_BENCH_BOOM_PROPERTIES") or _VENDORED_PROPERTIES)
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_datasets(max_tasks: int | None):
    props = _load_properties()
    override = os.environ.get("CASCADE_BENCH_BOOM_DATASETS", "").strip()
    if override:
        wanted = [c.strip() for c in override.split(",") if c.strip()]
    else:
        wanted = list(props.keys())

    n = 0
    for cfg in wanted:
        # Each BOOM config has ONE designated term in the manifest (not the
        # short/medium/long sweep gift-eval uses); fall back to "short".
        term = props.get(cfg, {}).get("term", "short")
        ds = build_dataset(cfg, term, storage_env_var="BOOM")
        if ds is None:
            continue
        yield ds
        n += 1
        if max_tasks and n >= max_tasks:
            return


def run(
    checkpoint_dir: str,
    *,
    num_samples: int = 100,
    max_series: int | None = None,
    device: str = "cpu",
) -> SuiteResult:
    boom_path = os.environ.get("CASCADE_BENCH_BOOM_PATH") or os.environ.get("BOOM")
    if not boom_path:
        return SuiteResult(
            suite="boom",
            status="skipped",
            detail="BOOM (or CASCADE_BENCH_BOOM_PATH) not set; point it at the BOOM data dir.",
        )
    os.environ.setdefault("BOOM", boom_path)
    try:
        metrics, n = evaluate_datasets(
            _iter_datasets(max_series),
            checkpoint_dir,
            num_samples=num_samples,
            device=device,
        )
        if not n:
            return SuiteResult(suite="boom", status="error", detail="no BOOM configs scored")
        return SuiteResult(suite="boom", status="ok", metrics=metrics, n_series=n)
    except FileNotFoundError as e:
        return SuiteResult(suite="boom", status="skipped", detail=f"BOOM manifest missing: {e}")
    except ImportError as e:
        return SuiteResult(suite="boom", status="skipped", detail=f"gift-eval not importable: {e}")
    except Exception as e:  # noqa: BLE001
        return SuiteResult(
            suite="boom",
            status="error",
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )

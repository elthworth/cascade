"""Contract tests for aurora-mix-v1 (mirrors base_generator/tests).

Run from the cascade repo root:

    python -m pytest my_generator/tests -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cascade.interface.generator import check_series, drain_generator

REPO_DIR = Path(__file__).resolve().parents[1]

# Mirror chain.toml [generator] bounds.
MIN_LEN = 64
MAX_LEN = 2048
MAX_TOTAL_POINTS = 2_000_000_000
MAX_CHANNELS = 1


def _load_generator_cls():
    spec = importlib.util.spec_from_file_location(
        "aurora_generator_under_test", REPO_DIR / "generator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Generator


@pytest.fixture(scope="module")
def Generator():
    return _load_generator_cls()


def _drain(Generator, n, seed=0):
    gen = Generator(str(REPO_DIR), seed=seed)
    return drain_generator(
        gen, n,
        min_length=MIN_LEN, max_length=MAX_LEN,
        max_total_points=MAX_TOTAL_POINTS, max_channels=MAX_CHANNELS,
    )


def test_subclasses_datagenerator(Generator):
    from cascade.interface import DataGenerator

    gen = Generator(str(REPO_DIR), seed=0)
    assert isinstance(gen, DataGenerator)
    assert isinstance(gen.name, str) and gen.name


def test_yields_exact_count(Generator):
    out = _drain(Generator, 40, seed=3)
    assert len(out) == 40


def test_yields_exact_count_small_and_awkward(Generator):
    for n in (1, 2, 3, 7, 13):
        assert len(_drain(Generator, n, seed=9)) == n


def test_every_series_passes_check_series(Generator):
    gen = Generator(str(REPO_DIR), seed=5)
    n = 0
    for i, arr in enumerate(gen.generate(40)):
        check_series(arr, min_length=MIN_LEN, max_length=MAX_LEN,
                     max_channels=MAX_CHANNELS, index=i)
        n += 1
    assert n == 40


def test_canonicalised_shape_and_dtype(Generator):
    out = _drain(Generator, 24, seed=7)
    assert all(a.ndim == 2 and a.shape[0] == 1 for a in out)
    assert all(a.dtype == np.float64 for a in out)
    assert all(np.isfinite(a).all() for a in out)
    assert all(MIN_LEN <= a.shape[-1] <= MAX_LEN for a in out)


def test_deterministic_same_seed(Generator):
    a = _drain(Generator, 32, seed=11)
    b = _drain(Generator, 32, seed=11)
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_deterministic_huge_chain_seed(Generator):
    # Round seeds come from block hashes and can be enormous.
    seed = 2269901645662351552
    a = _drain(Generator, 12, seed=seed)
    b = _drain(Generator, 12, seed=seed)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_different_seed_differs(Generator):
    a = _drain(Generator, 16, seed=0)
    b = _drain(Generator, 16, seed=1)
    assert any(x.shape != y.shape or not np.array_equal(x, y) for x, y in zip(a, b))


def test_length_band_diversity(Generator):
    out = _drain(Generator, 64, seed=2)
    lengths = {a.shape[-1] for a in out}
    assert len(lengths) > 10
    assert min(a.shape[-1] for a in out) >= MIN_LEN
    assert max(a.shape[-1] for a in out) <= MAX_LEN


def test_point_cap_enforced(Generator):
    gen = Generator(str(REPO_DIR), seed=0)
    with pytest.raises(ValueError):
        drain_generator(
            gen, 64,
            min_length=MIN_LEN, max_length=MAX_LEN,
            max_total_points=100, max_channels=MAX_CHANNELS,
        )


def test_no_degenerate_constant_series(Generator):
    out = _drain(Generator, 96, seed=13)
    stds = np.array([float(a.std()) for a in out])
    assert (stds > 1e-9).all(), "flat series teach nothing"


# ── family_params (config tuning surface) ────────────────────────────────────

def _cfg_dir_with_family_params(tmp_path, fp):
    import json

    cfg = json.loads((REPO_DIR / "config.json").read_text(encoding="utf-8"))
    cfg["family_params"] = fp
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return tmp_path


def test_family_params_default_config_matches_hardcoded(Generator, tmp_path):
    # A config WITHOUT family_params must draw the same corpus as the shipped
    # config (whose family_params spell out the code defaults): knob reads
    # consume no RNG and the shipped values equal the hardcoded defaults.
    import json

    cfg = json.loads((REPO_DIR / "config.json").read_text(encoding="utf-8"))
    cfg.pop("family_params")
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    bare = drain_generator(Generator(str(tmp_path), seed=5), 24,
                           min_length=MIN_LEN, max_length=MAX_LEN,
                           max_total_points=MAX_TOTAL_POINTS, max_channels=MAX_CHANNELS)
    shipped = _drain(Generator, 24, seed=5)
    for a, b in zip(shipped, bare, strict=True):
        assert np.array_equal(a, b)


def test_family_params_override_changes_corpus(Generator, tmp_path):
    d = _cfg_dir_with_family_params(tmp_path, {
        "trend_seasonal": {"harmonics": [6, 10], "amp_seasonal": [1.5, 4.0]},
    })
    out = drain_generator(Generator(str(d), seed=5), 24,
                          min_length=MIN_LEN, max_length=MAX_LEN,
                          max_total_points=MAX_TOTAL_POINTS, max_channels=MAX_CHANNELS)
    base = _drain(Generator, 24, seed=5)
    assert any(a.shape != b.shape or not np.array_equal(a, b)
               for a, b in zip(base, out, strict=True))


def test_family_params_unknown_keys_ignored(Generator, tmp_path):
    d = _cfg_dir_with_family_params(tmp_path, {
        "trend_seasonal": {"no_such_knob": 123},
        "not_a_family": {"x": 1},
    })
    out = drain_generator(Generator(str(d), seed=3), 8,
                          min_length=MIN_LEN, max_length=MAX_LEN,
                          max_total_points=MAX_TOTAL_POINTS, max_channels=MAX_CHANNELS)
    assert len(out) == 8

"""Classical-econometrics families, reimplemented from a public testnet
submission (uid16's ``custom-mixture-of-priors-v1`` — published on the hub,
tier-1 auditable). Four of its ten families, chosen for orthogonality to the
vendored TempoPFN priors and to our existing ares families:

* ``trend_seasonal_ar`` — linear trend × multi-sinusoid seasonality + AR(1)
  residual (the classical decomposition prior; their heaviest weight, 0.26).
* ``multiplicative``    — positive exponential-drift level × seasonal factor ×
  noise factor (airline-passengers shape; pure multiplicative structure the
  additive priors never produce).
* ``regime_shift``      — piecewise-constant level (cumsum of sparse jumps)
  with a piecewise volatility regime + mild seasonality.
* ``intermittent``      — zero-inflated sparse demand (Croston-style), gamma
  spike magnitudes on a low baseline.

Pure numpy; deterministic per (seed + series index) via np.random.default_rng —
no global RNG state, safe under the cross-process determinism contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams

# Hard magnitude backstop (repo-level Generator._sanitize also applies the
# config max_abs_value; this keeps a lone family draw finite and bounded).
_ABS_MAX = 1.0e6


# ── shared primitives (batched over n; here always n == 1 per series) ────────


def _ar1_batch(innov: np.ndarray, phi: np.ndarray) -> np.ndarray:
    n, L = innov.shape
    x = np.empty((n, L), dtype=np.float64)
    x[:, 0] = innov[:, 0]
    p = phi.reshape(n)
    for t in range(1, L):
        x[:, t] = p * x[:, t - 1] + innov[:, t]
    return x


def _seasonal(rng: np.random.Generator, n: int, L: int, k_max: int = 3) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    periods = np.array([4, 7, 12, 24, 30, 52, 96, 144, 168, 336], dtype=np.float64)
    k = rng.integers(1, k_max + 1, size=n)
    out = np.zeros((n, L), dtype=np.float64)
    for j in range(k_max):
        active = (k > j).astype(np.float64)[:, None]
        per = rng.choice(periods, size=n)[:, None]
        amp = rng.uniform(0.2, 2.0, size=n)[:, None]
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n)[:, None]
        out += active * amp * np.sin(2.0 * np.pi * t / per + phase)
    return out


def _sparse_jumps(rng: np.random.Generator, n: int, L: int, rate: float, scale) -> np.ndarray:
    mask = rng.random((n, L)) < rate
    mag = rng.normal(0.0, 1.0, size=(n, L))
    s = np.asarray(scale, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    jumps = mask * mag * s
    jumps[:, 0] = 0.0
    return jumps


# ── family builders: each returns a (n, L) float64 block ────────────────────


def _trend_seasonal_ar(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    level = rng.normal(0.0, 1.0, size=(n, 1))
    slope = rng.normal(0.0, 0.01, size=(n, 1))
    series = level + slope * t + _seasonal(rng, n, L)
    phi = rng.uniform(0.0, 0.85, size=n)
    sigma = rng.uniform(0.1, 0.6, size=(n, 1))
    innov = rng.normal(0.0, 1.0, size=(n, L)) * sigma
    return series + _ar1_batch(innov, phi)


def _multiplicative(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    growth = rng.normal(0.0, 0.003, size=(n, 1))
    base_level = np.exp(growth * t + rng.normal(0.0, 0.3, size=(n, 1)))
    amp = rng.uniform(0.1, 0.6, size=(n, 1))
    seas = 1.0 + amp * np.sin(
        2.0 * np.pi * t / rng.choice([7.0, 12.0, 24.0, 52.0], size=n)[:, None]
        + rng.uniform(0.0, 2 * np.pi, size=(n, 1))
    )
    noise = 1.0 + rng.normal(0.0, 1.0, size=(n, L)) * rng.uniform(0.02, 0.15, size=(n, 1))
    scale = rng.uniform(1.0, 50.0, size=(n, 1))
    return scale * base_level * np.clip(seas, 0.05, None) * np.clip(noise, 0.05, None)


def _regime_shift(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    level = np.cumsum(_sparse_jumps(rng, n, L, rate=3.0 / L, scale=2.0), axis=1)
    log_vol = np.cumsum(_sparse_jumps(rng, n, L, rate=3.0 / L, scale=0.5), axis=1)
    vol = np.exp(np.clip(log_vol, -3.0, 3.0)) * rng.uniform(0.1, 0.5, size=(n, 1))
    noise = rng.normal(0.0, 1.0, size=(n, L)) * vol
    seas = _seasonal(rng, n, L, k_max=2) * rng.uniform(0.0, 1.0, size=(n, 1))
    return level + seas + noise


def _intermittent(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    p = rng.uniform(0.05, 0.4, size=(n, 1))
    occur = (rng.random((n, L)) < p).astype(np.float64)
    magnitude = rng.gamma(shape=2.0, scale=1.0, size=(n, L)) * rng.uniform(1.0, 10.0, size=(n, 1))
    baseline = rng.uniform(0.0, 0.5, size=(n, 1))
    return baseline + occur * magnitude


_BUILDERS = {
    "trend_seasonal_ar": _trend_seasonal_ar,
    "multiplicative": _multiplicative,
    "regime_shift": _regime_shift,
    "intermittent": _intermittent,
}


@dataclass
class JTestGeneratorParams(GeneratorParams):
    """Which classical family this key draws (set per-key via family_params)."""

    family: str = "trend_seasonal_ar"


class JTestGeneratorWrapper(GeneratorWrapper):
    """Batch generator for one classical family (mirrors the vendored wrapper API)."""

    def __init__(self, params: JTestGeneratorParams):
        super().__init__(params)

    def generate_batch(self, batch_size: int, seed: int | None = None) -> TimeSeriesContainer:
        if seed is None:
            seed = int(self.params.global_seed)
        self._set_random_seeds(seed)  # base-class metadata sampling below
        length = int(self.params.length)
        build = _BUILDERS[str(self.params.family)]
        values = np.empty((batch_size, length), dtype=np.float64)
        for i in range(batch_size):
            rng = np.random.default_rng((int(seed) + i) % (2**31))
            row = build(rng, 1, length)[0]
            row = np.nan_to_num(row, nan=0.0, posinf=_ABS_MAX, neginf=-_ABS_MAX)
            values[i] = np.clip(row, -_ABS_MAX, _ABS_MAX)
        sampled = self._sample_parameters(batch_size)
        return TimeSeriesContainer(
            values=values,
            start=sampled["start"],
            frequency=sampled["frequency"],
        )

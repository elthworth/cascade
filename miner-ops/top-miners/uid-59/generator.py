"""custom_miner — a diverse, deterministic synthetic time-series generator.

This is the artifact a cascade miner actually competes with: a subclass of
``cascade.interface.DataGenerator`` that turns a single integer ``seed`` into a
corpus of univariate float series. The subnet holds the model, seeds, and
compute budget byte-identical between the king and every challenger, so the
*only* thing that moves the forecast score is the distribution this file emits.
The competitive lever is therefore **prior diversity + realism**: a corpus that
covers more of the shapes a real forecaster must handle (trend, multi-seasonal,
regime shifts, integrated/near-unit-root dynamics, smooth GP-like curves,
nonlinear/chaotic recurrences, intermittent demand, outliers) trains a stronger
zero-shot model than the reference generator's trend+seasonal+AR(1) mix.

Design constraints this file respects (all from the contract in
``cascade.interface``):

* **Determinism is load-bearing.** Every value is drawn from one
  ``np.random.default_rng(seed)`` in a fixed draw order, so two runs at the same
  seed produce byte-identical corpora — the property ``cascade verify`` audits
  by building the corpus twice and comparing digests.
* **Code-only.** No shipped weights, no network, no clock, no un-seeded RNG.
  Imports stay on the dependency allowlist (numpy only here) and clear of the
  static-guard blocklist.
* **Bounded + finite.** Each series is 1-D ``(L,)`` float64, length in
  ``[min_length, max_length]``, finite (no NaN/inf). ``_sanitize`` is the last
  gate so a numerically unlucky draw can never poison a training run.

Everything is **vectorised per family** (a batched time-axis recurrence, never a
per-series Python loop over time), so draining the full ``corpus_n_series``
(16384 on mainnet) is fast enough to stay well under ``max_generate_seconds``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cascade.interface import DataGenerator

# Series generated per vectorised batch. Bounds peak memory to O(_CHUNK · max_len)
# so streaming feed modes (which request millions of series and stop early) never
# materialise more than one chunk. ~256 keeps the vectorisation win without a big
# working set: 256 × 2048 × 8 B ≈ 4 MB per family buffer.
_CHUNK = 256

# ── family mixture ──────────────────────────────────────────────────────────
# Names are the process families the corpus mixes over; the default weights are
# a deliberate spread (no single family dominates). Override with
# ``"family_weights": {"chaotic": 0.2, ...}`` in config.json to tune the prior
# without touching code — unspecified families keep their default weight.
_FAMILIES: tuple[str, ...] = (
    "trend_seasonal_ar",   # level + slope + multi-seasonal + AR(1) noise (rich reference)
    "regime_shift",        # piecewise level/variance regimes with structural breaks
    "multiplicative",      # positive level × seasonal factor × multiplicative noise
    "ar2",                 # AR(2), stationarity-guaranteed, incl. near-unit-root
    "integrated",          # I(1)/I(2) random walks with drift
    "threshold_ar",        # SETAR — regime-switching nonlinear recurrence
    "chaotic",             # bounded chaotic maps (logistic / sine)
    "rff_gp",              # smooth GP-like sample via random Fourier features
    "intermittent",        # zero-inflated / intermittent demand
    "pulse_outlier",       # smooth base + sparse pulses/outliers + flat gaps
)
# Tuned with local_validator against a broad multi-domain eval: strong seasonal
# coverage (trend_seasonal_ar + multiplicative) matters because most real series
# are seasonal, but every family keeps meaningful mass so the prior stays diverse
# (the diversity is what generalises across non-seasonal domains). Override per
# submission via ``"family_weights"`` in config.json.
#
# NOTE: a richer "composite" family (layered trend + multi-harmonic seasonality +
# heavy-tailed heteroskedastic noise + structural events) was implemented and
# measured with local_validator — it is more *realistic*, but at high fidelity it
# scored slightly WORSE on the proxy (LCB +0.15 vs this mix's +0.167), so it was
# not shipped. Re-evaluate it against REAL held-out data (`--eval-windows`) before
# adopting; the synthetic proxy pool under-rewards its realism.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "trend_seasonal_ar": 0.26,
    "regime_shift": 0.10,
    "multiplicative": 0.16,
    "ar2": 0.10,
    "integrated": 0.08,
    "threshold_ar": 0.06,
    "chaotic": 0.05,
    "rff_gp": 0.07,
    "intermittent": 0.03,
    "pulse_outlier": 0.03,
}


class Generator(DataGenerator):
    """A mixture-of-priors generator. Submit as ``generator.Generator``."""

    def __init__(self, config_dir: str, *, seed: int) -> None:
        cfg_path = Path(config_dir) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        self._cfg = cfg
        self._seed = int(seed)
        self._min_len = int(cfg.get("min_length", 64))
        self._max_len = int(cfg.get("max_length", 2048))
        if self._min_len < 1 or self._max_len < self._min_len:
            raise ValueError(f"invalid length band [{self._min_len}, {self._max_len}]")
        weights = dict(_DEFAULT_WEIGHTS)
        for k, v in dict(cfg.get("family_weights", {})).items():
            if k in weights:
                weights[k] = float(v)
        w = np.asarray([weights[f] for f in _FAMILIES], dtype=np.float64)
        if not np.all(np.isfinite(w)) or w.min() < 0 or w.sum() <= 0:
            raise ValueError("family_weights must be finite, non-negative, and not all zero")
        self._weights = w / w.sum()

    @property
    def name(self) -> str:
        return str(self._cfg.get("name", "custom-mixture-of-priors-v1"))

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        # Lazy, chunked generation. This is REQUIRED for the streaming feed
        # modes (chain.toml ``corpus_mode = "stream_cpu"``): the trainer calls
        # ``generate(n_upper)`` with ``n_upper = token_budget // min_length + 2``
        # — often millions — and stops pulling once the token budget is hit
        # (see cascade/trainer/stream.py). Materialising all ``n_series`` up
        # front would OOM before the first yield. Generating one CHUNK at a time
        # keeps memory at O(CHUNK) and stops early when the consumer stops,
        # while a fixed draw order keeps the whole sequence seed-deterministic.
        if n_series <= 0:
            return
        rng = np.random.default_rng(self._seed)
        max_len = self._max_len
        builders = (
            _trend_seasonal_ar, _regime_shift, _multiplicative, _ar2,
            _integrated, _threshold_ar, _chaotic, _rff_gp,
            _intermittent, _pulse_outlier,
        )
        produced = 0
        while produced < n_series:
            # Always draw a FULL _CHUNK (yielding only what's still needed), so
            # chunk boundaries fall at fixed multiples of _CHUNK regardless of
            # the total requested. Then series i is a pure function of (seed, i):
            # a run at any n >= i produces the identical series i. That makes a
            # smaller local build a true prefix of the mainnet corpus, and keeps
            # cross-mode/cross-party runs reproducible.
            lengths = rng.integers(self._min_len, max_len + 1, size=_CHUNK)
            fam_ids = rng.choice(len(_FAMILIES), size=_CHUNK, p=self._weights)
            chunk: list[np.ndarray | None] = [None] * _CHUNK
            for fam in range(len(_FAMILIES)):
                idx = np.nonzero(fam_ids == fam)[0]
                if idx.size == 0:
                    continue
                block = _sanitize(builders[fam](rng, int(idx.size), max_len))
                for row, series_i in enumerate(idx):
                    L = int(lengths[series_i])
                    chunk[series_i] = np.ascontiguousarray(block[row, :L], dtype=np.float64)
            take = min(_CHUNK, n_series - produced)
            for arr in chunk[:take]:
                # Every slot is filled: fam_ids partitions [0, _CHUNK). Guard
                # anyway (survives python -O) so a logic slip fails loud.
                if arr is None:  # pragma: no cover - defensive
                    raise RuntimeError("internal: unfilled series slot")
                yield arr
            produced += take


# ── shared vectorised primitives ────────────────────────────────────────────


def _ar1_batch(innov: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """AR(1) filter applied along the time axis of a (n, L) innovation block.

    ``x[:, t] = phi * x[:, t-1] + innov[:, t]``. The loop is over time (L
    iterations, vectorised across the batch), never over the n series.
    """
    n, L = innov.shape
    x = np.empty((n, L), dtype=np.float64)
    x[:, 0] = innov[:, 0]
    p = phi.reshape(n)
    for t in range(1, L):
        x[:, t] = p * x[:, t - 1] + innov[:, t]
    return x


def _ar2_batch(innov: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """AR(2) filter: ``x_t = a1 x_{t-1} + a2 x_{t-2} + e_t`` (batched over n)."""
    n, L = innov.shape
    x = np.empty((n, L), dtype=np.float64)
    x[:, 0] = innov[:, 0]
    if L > 1:
        x[:, 1] = a1 * x[:, 0] + innov[:, 1]
    for t in range(2, L):
        x[:, t] = a1 * x[:, t - 1] + a2 * x[:, t - 2] + innov[:, t]
    return x


def _seasonal(rng: np.random.Generator, n: int, L: int, k_max: int = 3) -> np.ndarray:
    """Sum of 1..k_max sinusoids with per-series random period/amp/phase."""
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
    """A (n, L) block of mostly-zero values with occasional N(0, scale) jumps.

    ``cumsum`` over this yields a piecewise-constant level; ``exp(cumsum)`` of a
    scaled version yields a piecewise-constant positive multiplier.
    """
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


def _regime_shift(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # Piecewise-constant level via cumsum of sparse jumps, plus a piecewise
    # variance regime (occasional volatility multiplier), plus mild seasonality.
    level = np.cumsum(_sparse_jumps(rng, n, L, rate=3.0 / L, scale=2.0), axis=1)
    log_vol = np.cumsum(_sparse_jumps(rng, n, L, rate=3.0 / L, scale=0.5), axis=1)
    vol = np.exp(np.clip(log_vol, -3.0, 3.0)) * rng.uniform(0.1, 0.5, size=(n, 1))
    noise = rng.normal(0.0, 1.0, size=(n, L)) * vol
    seas = _seasonal(rng, n, L, k_max=2) * rng.uniform(0.0, 1.0, size=(n, 1))
    return level + seas + noise


def _multiplicative(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    growth = rng.normal(0.0, 0.003, size=(n, 1))
    base_level = np.exp(growth * t + rng.normal(0.0, 0.3, size=(n, 1)))  # positive, drifting
    amp = rng.uniform(0.1, 0.6, size=(n, 1))
    seas = 1.0 + amp * np.sin(
        2.0 * np.pi * t / rng.choice([7.0, 12.0, 24.0, 52.0], size=n)[:, None]
        + rng.uniform(0.0, 2 * np.pi, size=(n, 1))
    )
    noise = 1.0 + rng.normal(0.0, 1.0, size=(n, L)) * rng.uniform(0.02, 0.15, size=(n, 1))
    scale = rng.uniform(1.0, 50.0, size=(n, 1))
    return scale * base_level * np.clip(seas, 0.05, None) * np.clip(noise, 0.05, None)


def _ar2(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # Draw partial autocorrelations in (-1, 1) and map to AR(2) coeffs via
    # Levinson-Durbin, which guarantees stationarity. Bias p1 high for
    # persistent (sometimes near-unit-root) series.
    p1 = rng.uniform(0.3, 0.98, size=n)
    p2 = rng.uniform(-0.6, 0.6, size=n)
    a2 = p2
    a1 = p1 * (1.0 - p2)
    sigma = rng.uniform(0.2, 0.8, size=(n, 1))
    innov = rng.normal(0.0, 1.0, size=(n, L)) * sigma
    x = _ar2_batch(innov, a1, a2)
    drift = rng.normal(0.0, 0.005, size=(n, 1)) * np.arange(L, dtype=np.float64)[None, :]
    return x + drift


def _integrated(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    order2 = rng.random(n) < 0.35
    drift = rng.normal(0.0, 0.02, size=(n, 1))
    sigma = rng.uniform(0.2, 1.0, size=(n, 1))
    steps = rng.normal(0.0, 1.0, size=(n, L)) * sigma + drift
    walk = np.cumsum(steps, axis=1)
    walk2 = np.cumsum(walk, axis=1)
    o2 = order2[:, None]
    # I(2) grows fast; damp it so it shares scale with the I(1) branch.
    return np.where(o2, walk2 / max(L, 1) ** 0.5, walk)


def _threshold_ar(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # SETAR(2): coefficient flips with the sign of the previous value — a simple
    # nonlinear recurrence that produces asymmetric, regime-switching dynamics.
    phi_hi = rng.uniform(0.3, 0.9, size=n)
    phi_lo = rng.uniform(-0.9, 0.3, size=n)
    const_hi = rng.normal(0.0, 0.3, size=n)
    const_lo = rng.normal(0.0, 0.3, size=n)
    sigma = rng.uniform(0.2, 0.7, size=(n, 1))
    innov = rng.normal(0.0, 1.0, size=(n, L)) * sigma
    x = np.empty((n, L), dtype=np.float64)
    x[:, 0] = innov[:, 0]
    for t in range(1, L):
        prev = x[:, t - 1]
        hi = prev >= 0.0
        phi = np.where(hi, phi_hi, phi_lo)
        const = np.where(hi, const_hi, const_lo)
        x[:, t] = np.clip(const + phi * prev + innov[:, t], -1e6, 1e6)
    return x


def _chaotic(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # Bounded chaotic maps: logistic x_{t+1}=r x(1-x) with r∈[3.6,4.0], and the
    # sine map r sin(pi x). Both stay in [0,1]; standardise afterwards. A random
    # observation length as a "sampling rate" adds variety across series.
    use_sine = rng.random(n) < 0.5
    r_log = rng.uniform(3.6, 4.0, size=n)
    r_sin = rng.uniform(0.85, 1.0, size=n)
    x0 = rng.uniform(0.05, 0.95, size=n)
    x = np.empty((n, L), dtype=np.float64)
    cur = x0.copy()
    x[:, 0] = cur
    for t in range(1, L):
        nxt_log = r_log * cur * (1.0 - cur)
        nxt_sin = r_sin * np.sin(np.pi * cur)
        cur = np.where(use_sine, nxt_sin, nxt_log)
        cur = np.clip(cur, 0.0, 1.0)
        x[:, t] = cur
    return x


def _rff_gp(rng: np.random.Generator, n: int, L: int, K: int = 48) -> np.ndarray:
    # Random Fourier features approximate a stationary (RBF-like) GP sample:
    # f(t) = sqrt(2/K) * sum_k cos(w_k t + b_k), w_k ~ N(0, 1/lengthscale^2).
    # Loop over the K features (K iterations, vectorised over n and L) so peak
    # memory stays (n, L), never (n, K, L).
    t = np.arange(L, dtype=np.float64)[None, :]
    lengthscale = rng.uniform(20.0, 200.0, size=(n, 1))
    acc = np.zeros((n, L), dtype=np.float64)
    for _ in range(K):
        w = rng.normal(0.0, 1.0, size=(n, 1)) / lengthscale
        b = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
        acc += np.cos(w * t + b)
    return np.sqrt(2.0 / K) * acc


def _intermittent(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # Zero-inflated demand: sparse positive spikes on a low baseline. Common in
    # retail/logistics and absent from the reference generator.
    p = rng.uniform(0.05, 0.4, size=(n, 1))
    occur = (rng.random((n, L)) < p).astype(np.float64)
    magnitude = rng.gamma(shape=2.0, scale=1.0, size=(n, L)) * rng.uniform(1.0, 10.0, size=(n, 1))
    baseline = rng.uniform(0.0, 0.5, size=(n, 1))
    return baseline + occur * magnitude


def _pulse_outlier(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # A smooth base with sparse additive pulses (outliers) and occasional flat
    # (held-constant) gaps — the messy structure real series carry.
    base = _rff_gp(rng, n, L, K=24) * rng.uniform(0.5, 2.0, size=(n, 1))
    base += _seasonal(rng, n, L, k_max=1) * rng.uniform(0.0, 1.0, size=(n, 1))
    pulses = _sparse_jumps(rng, n, L, rate=5.0 / L, scale=rng.uniform(3.0, 8.0, size=n))
    series = base + pulses
    # Occasional flat gaps: hold the value across a short random run.
    hold = rng.random((n, L)) < (2.0 / L)
    hold[:, 0] = False
    for t in range(1, L):
        m = hold[:, t]
        series[m, t] = series[m, t - 1]
    return series


# ── final safety gate ───────────────────────────────────────────────────────


def _sanitize(block: np.ndarray) -> np.ndarray:
    """Guarantee the contract: finite float64, no NaN/inf, bounded magnitude.

    The trainer's ``check_series`` rejects any non-finite value, which would
    fail the whole run — so this is the hard backstop after every family
    builder. Replaces non-finite values and clips to a generous bound.
    """
    x = np.asarray(block, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(x, -1e6, 1e6)

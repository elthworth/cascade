"""Four structural axes absent from both the vendored TempoPFN priors and our
existing ares families — long-range dependence, multifractal volatility,
deterministic chaos, and calendar load-curves. Original implementations from
the published literature (no rival code):

* ``fgn``            — fractional Gaussian noise via Davies–Harte circulant
  embedding (exact spectral synthesis), Hurst H ~ U(0.15, 0.95); half the
  draws are integrated to fBm so both stationary long-memory noise and
  H-self-similar level paths appear.
* ``fractal_multi``  — multifractal random walk: a dyadic lognormal cascade
  builds a log-volatility field, Gaussian returns are modulated by it, and
  draws are emitted as returns / integrated level / activity (|.|) — the
  volatility-clustering + fat-tail regime none of the additive priors reach.
* ``chaotic``        — deterministic chaos observed through noise: Lorenz and
  Roessler flows (plain-float RK4), Mackey–Glass delay dynamics, and the
  logistic and Henon maps, with parameter jitter, burn-in, random observable
  and observation noise.
* ``rhythm``         — human-activity load curves: harmonic daily profile with
  a random period, weekday/weekend modulation, slow amplitude drift, level +
  trend, AR(1) residual; mostly softplus-positive like real utilisation data.

Pure numpy + math; deterministic per (seed + series index) via
np.random.default_rng — no global RNG state, matching the jtest idiom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams

_ABS_MAX = 1.0e6


# ── fgn: fractional Gaussian noise / fractional Brownian motion ──────────────


def _fgn_davies_harte(rng: np.random.Generator, L: int, H: float) -> np.ndarray:
    """Exact fGn of length L via circulant embedding of the autocovariance."""
    n = 1 << max(1, int(math.ceil(math.log2(max(L, 2)))))
    k = np.arange(n + 1, dtype=np.float64)
    gamma = 0.5 * ((k + 1.0) ** (2 * H) - 2.0 * k ** (2 * H) + np.abs(k - 1.0) ** (2 * H))
    c = np.concatenate([gamma, gamma[-2:0:-1]])  # circulant row, length 2n
    lam = np.fft.fft(c).real
    lam = np.clip(lam, 0.0, None)  # numeric guard; theory gives lam >= 0
    m = 2 * n
    w = np.zeros(m, dtype=np.complex128)
    w[0] = math.sqrt(lam[0] / m) * rng.standard_normal()
    w[n] = math.sqrt(lam[n] / m) * rng.standard_normal()
    u = rng.standard_normal(n - 1)
    v = rng.standard_normal(n - 1)
    half = np.sqrt(lam[1:n] / (2.0 * m)) * (u + 1j * v)
    w[1:n] = half
    w[n + 1:] = np.conj(half[::-1])
    x = np.fft.fft(w).real[:L]
    return x


def _fgn(rng: np.random.Generator, L: int) -> np.ndarray:
    H = float(rng.uniform(0.15, 0.95))
    x = _fgn_davies_harte(rng, L, H)
    sd = x.std()
    if sd > 1e-12:
        x = x / sd
    if rng.random() < 0.5:  # integrate to fBm: H-self-similar level path
        x = np.cumsum(x)
    scale = float(rng.lognormal(0.0, 1.0))
    level = float(rng.normal(0.0, 2.0))
    trend = float(rng.normal(0.0, 0.002)) * np.arange(L, dtype=np.float64)
    return level + trend + scale * x


# ── fractal_multi: multifractal random walk via lognormal cascade ────────────


def _fractal_multi(rng: np.random.Generator, L: int) -> np.ndarray:
    levels = max(1, int(math.ceil(math.log2(max(L, 2)))))
    lam2 = float(rng.uniform(0.01, 0.12))  # intermittency (log-variance per octave)
    omega = np.zeros(1, dtype=np.float64)
    for _ in range(levels):
        omega = np.repeat(omega, 2)
        omega = omega + rng.normal(-lam2, math.sqrt(2.0 * lam2), size=omega.size)
    omega = omega[:L]
    omega -= omega.mean()  # center so intensity is O(1)
    vol = np.exp(np.clip(omega, -8.0, 8.0))
    ret = rng.standard_normal(L) * vol
    sd = ret.std()
    if sd > 1e-12:
        ret = ret / sd
    mode = rng.random()
    if mode < 0.45:  # price-like level path
        x = np.cumsum(ret)
    elif mode < 0.75:  # stationary volatility-clustered returns
        x = ret
    else:  # activity/traffic-like positive intensity
        x = np.abs(ret) + float(rng.uniform(0.0, 0.5))
    scale = float(rng.lognormal(0.0, 1.0))
    return float(rng.normal(0.0, 2.0)) + scale * x


# ── chaotic: strange attractors + chaotic maps under observation noise ───────


def _lorenz(rng: np.random.Generator, L: int) -> np.ndarray:
    s = 10.0 * float(rng.uniform(0.9, 1.1))
    r = 28.0 * float(rng.uniform(0.9, 1.15))
    b = (8.0 / 3.0) * float(rng.uniform(0.9, 1.1))
    dt = float(rng.uniform(0.006, 0.02))
    x, y, z = (float(rng.normal(0.0, 5.0)) + 1e-3, float(rng.normal(0.0, 5.0)), 20.0 + float(rng.normal(0.0, 5.0)))
    obs = int(rng.integers(0, 3))
    out = np.empty(L, dtype=np.float64)

    def deriv(x: float, y: float, z: float) -> tuple[float, float, float]:
        return s * (y - x), x * (r - z) - y, x * y - b * z

    def rk4(x: float, y: float, z: float) -> tuple[float, float, float]:
        k1 = deriv(x, y, z)
        k2 = deriv(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1], z + 0.5 * dt * k1[2])
        k3 = deriv(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1], z + 0.5 * dt * k2[2])
        k4 = deriv(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
        return (
            x + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0,
            y + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0,
            z + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0,
        )

    for _ in range(300):  # burn-in onto the attractor
        x, y, z = rk4(x, y, z)
    for t in range(L):
        x, y, z = rk4(x, y, z)
        out[t] = (x, y, z)[obs]
    return out


def _roessler(rng: np.random.Generator, L: int) -> np.ndarray:
    a = 0.2 * float(rng.uniform(0.85, 1.15))
    b = 0.2 * float(rng.uniform(0.85, 1.15))
    c = 5.7 * float(rng.uniform(0.9, 1.1))
    dt = float(rng.uniform(0.05, 0.15))
    x, y, z = float(rng.normal(0.0, 2.0)) + 1e-3, float(rng.normal(0.0, 2.0)), 0.1
    obs = int(rng.integers(0, 2))  # x or y (z is spiky-degenerate)
    out = np.empty(L, dtype=np.float64)

    def rk4(x: float, y: float, z: float) -> tuple[float, float, float]:
        def d(x: float, y: float, z: float) -> tuple[float, float, float]:
            return -y - z, x + a * y, b + z * (x - c)

        k1 = d(x, y, z)
        k2 = d(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1], z + 0.5 * dt * k1[2])
        k3 = d(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1], z + 0.5 * dt * k2[2])
        k4 = d(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
        return (
            x + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0,
            y + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0,
            z + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0,
        )

    for _ in range(300):
        x, y, z = rk4(x, y, z)
    for t in range(L):
        x, y, z = rk4(x, y, z)
        out[t] = (x, y)[obs]
    return out


def _mackey_glass(rng: np.random.Generator, L: int) -> np.ndarray:
    beta = 0.2 * float(rng.uniform(0.9, 1.1))
    gam = 0.1 * float(rng.uniform(0.9, 1.1))
    tau = int(rng.integers(17, 31))
    n_exp = 10.0
    sub = 2  # Euler substeps per emitted sample (dt = 0.5)
    hist_len = tau * sub
    hist = [1.2 + float(rng.normal(0.0, 0.1))] * hist_len
    x = hist[-1]
    out = np.empty(L, dtype=np.float64)
    dt = 1.0 / sub
    total = 300 + L
    for t in range(total * sub):
        x_tau = hist[0]
        x = x + dt * (beta * x_tau / (1.0 + x_tau ** n_exp) - gam * x)
        hist.pop(0)
        hist.append(x)
        if t % sub == sub - 1:
            i = t // sub
            if i >= 300:
                out[i - 300] = x
    return out


def _logistic_map(rng: np.random.Generator, L: int) -> np.ndarray:
    r = float(rng.uniform(3.7, 3.999))
    x = float(rng.uniform(0.05, 0.95))
    out = np.empty(L, dtype=np.float64)
    for _ in range(200):
        x = r * x * (1.0 - x)
    for t in range(L):
        x = r * x * (1.0 - x)
        out[t] = x
    return out


def _henon_map(rng: np.random.Generator, L: int) -> np.ndarray:
    a = 1.4 * float(rng.uniform(0.95, 1.02))
    b = 0.3 * float(rng.uniform(0.95, 1.05))
    x, y = float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5))
    out = np.empty(L, dtype=np.float64)
    for _ in range(200):
        x, y = 1.0 - a * x * x + y, b * x
        if abs(x) > 5.0:  # escaped (bad param draw) — reset into the basin
            x, y = 0.1, 0.1
    for t in range(L):
        x, y = 1.0 - a * x * x + y, b * x
        if abs(x) > 5.0:
            x, y = 0.1, 0.1
        out[t] = x
    return out


_CHAOS = (_lorenz, _roessler, _mackey_glass, _logistic_map, _henon_map)


def _chaotic(rng: np.random.Generator, L: int) -> np.ndarray:
    sysf = _CHAOS[int(rng.integers(0, len(_CHAOS)))]
    x = sysf(rng, L)
    sd = x.std()
    if sd > 1e-12:
        x = (x - x.mean()) / sd
    noise_sd = float(rng.uniform(0.0, 0.15))  # observation noise
    x = x + rng.standard_normal(L) * noise_sd
    scale = float(rng.lognormal(0.0, 1.0))
    return float(rng.normal(0.0, 2.0)) + scale * x


# ── rhythm: calendar / human-activity load curves ────────────────────────────


def _rhythm(rng: np.random.Generator, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)
    period = float(rng.choice([24.0, 48.0, 96.0, 144.0, 168.0, 288.0]))
    # Harmonic daily profile (decaying amplitudes -> smooth realistic bumps).
    k_max = int(rng.integers(2, 6))
    profile = np.zeros(L, dtype=np.float64)
    for k in range(1, k_max + 1):
        amp = float(rng.uniform(0.3, 1.0)) / k
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        profile += amp * np.sin(2.0 * math.pi * k * t / period + phase)
    # Weekday/weekend modulation when the base period is sub-weekly.
    if period <= 96.0 and rng.random() < 0.7:
        week = 7.0 * period
        day_idx = np.floor((t % week) / period)
        weekend_factor = float(rng.uniform(0.2, 0.8))
        weekly = np.where(day_idx >= 5.0, weekend_factor, 1.0)
    else:
        weekly = 1.0
    # Slow amplitude drift + level/trend.
    drift = 1.0 + float(rng.uniform(0.0, 0.5)) * np.sin(
        2.0 * math.pi * t / float(rng.uniform(4.0, 20.0) * period) + float(rng.uniform(0.0, 2.0 * math.pi))
    )
    level = float(rng.uniform(0.5, 5.0))
    trend = float(rng.normal(0.0, 0.0015)) * t
    # AR(1) residual.
    phi = float(rng.uniform(0.3, 0.9))
    innov = rng.standard_normal(L) * float(rng.uniform(0.05, 0.3))
    resid = np.empty(L, dtype=np.float64)
    acc = 0.0
    for i in range(L):
        acc = phi * acc + innov[i]
        resid[i] = acc
    x = level + trend + drift * profile * weekly + resid
    if rng.random() < 0.7:  # utilisation-like data is non-negative
        x = np.logaddexp(0.0, 3.0 * x) / 3.0  # softplus, slope-preserving
    scale = float(rng.lognormal(0.0, 0.8))
    return scale * x


_BUILDERS = {
    "fgn": _fgn,
    "fractal_multi": _fractal_multi,
    "chaotic": _chaotic,
    "rhythm": _rhythm,
}


# ── params + wrapper (mirrors the jtest idiom) ───────────────────────────────


@dataclass
class FgnGeneratorParams(GeneratorParams):
    family: str = "fgn"


@dataclass
class FractalMultiGeneratorParams(GeneratorParams):
    family: str = "fractal_multi"


@dataclass
class ChaoticGeneratorParams(GeneratorParams):
    family: str = "chaotic"


@dataclass
class RhythmGeneratorParams(GeneratorParams):
    family: str = "rhythm"


class AxesGeneratorWrapper(GeneratorWrapper):
    """Batch generator for one axes family (mirrors the vendored wrapper API)."""

    def __init__(self, params: GeneratorParams):
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
            row = build(rng, length)
            row = np.nan_to_num(row, nan=0.0, posinf=_ABS_MAX, neginf=-_ABS_MAX)
            values[i] = np.clip(row, -_ABS_MAX, _ABS_MAX)
        sampled = self._sample_parameters(batch_size)
        return TimeSeriesContainer(
            values=values,
            start=sampled["start"],
            frequency=sampled["frequency"],
        )

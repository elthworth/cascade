"""Signature families — priors calibrated to the eval pool's dominant real clusters.

Three families, each targeting a measured statistical fingerprint that the
vendored TempoPFN priors cover poorly:

* ``web_counts``  — npm-downloads-like: strongly weekly (seasonal-lag ACF ≈ 0.87),
  light-tailed integer counts, adoption trends, holiday dips, CV ≈ 0.6,
  essentially never zero.
* ``pageviews``   — wikimedia-like: weakly weekly (ACF ≈ 0.43), heavy-tailed
  (tail index ≈ 2.5), news-event spikes with build-up/decay (≈ 3% of points
  beyond 8 MADs), level shifts, integer counts, CV ≈ 1.4.
* ``ksynth_cal``  — KernelSynth (Chronos, arXiv:2403.07815) reproduced exactly:
  33-kernel scikit-learn bank (21 calendar ExpSineSquared periodicities +
  DotProduct/RBF/RationalQuadratic/White/Constant), j ~ U{1,5} kernels with
  replacement folded by random +/×, zero-mean GP draw — plus the
  Mean-KernelSynth extension (CauKer, arXiv:2508.02879 ablation: non-zero mean
  functions improve it): half the draws add a linear/exponential/step mean.

Pure numpy/scikit-learn; deterministic per (seed + series index) via
np.random.default_rng — no global RNG state, safe under the cross-process
determinism contract.
"""

from __future__ import annotations

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import GeneratorParams

_LOG_LAM_MAX = float(np.log(5e6))


def _ar1(rng: np.random.Generator, length: int, phi: float, sig: float,
         innov: np.ndarray | None = None) -> np.ndarray:
    e = innov if innov is not None else rng.normal(0.0, sig, length)
    out = np.empty(length)
    a = 0.0
    for j in range(length):
        a = phi * a + e[j]
        out[j] = a
    return out


def _poissonize(rng: np.random.Generator, log_lam: np.ndarray) -> np.ndarray:
    lam = np.exp(np.clip(log_lam, -20.0, _LOG_LAM_MAX))
    big = lam > 1e5
    out = np.empty(lam.shape)
    if big.any():  # normal approximation keeps sampling fast and overflow-safe
        out[big] = np.round(lam[big] + rng.normal(0.0, np.sqrt(lam[big])))
    if (~big).any():
        out[~big] = rng.poisson(lam[~big])
    return np.maximum(out, 0.0)


# ── web_counts: npm-downloads-like ───────────────────────────────────────────


class WebCountsGeneratorParams(GeneratorParams):
    """Strongly-weekly integer count series (package-download shape)."""

    log_base_range: tuple[float, float] = (np.log(50.0), np.log(5e5))
    weekend_dip_prob: float = 0.85     # most download series dip on weekends
    p_saturating: float = 0.35         # logistic adoption trend
    holiday_rate: float = 2.0          # holiday dips per 365 steps
    ar_sigma_range: tuple[float, float] = (0.04, 0.16)


class WebCountsGeneratorWrapper(GeneratorWrapper):
    def __init__(self, params: WebCountsGeneratorParams):
        super().__init__(params)

    def generate_batch(self, batch_size: int, seed: int | None = None) -> TimeSeriesContainer:
        if seed is None:
            seed = int(self.params.global_seed)
        self._set_random_seeds(seed)
        length = int(self.params.length)
        values = np.empty((batch_size, length), dtype=np.float64)
        for i in range(batch_size):
            rng = np.random.default_rng((int(seed) + i) % (2**31))
            values[i] = self._one_series(rng, length)
        sampled = self._sample_parameters(batch_size)
        return TimeSeriesContainer(values=values, start=sampled["start"],
                                   frequency=sampled["frequency"])

    def _one_series(self, rng: np.random.Generator, length: int) -> np.ndarray:
        p = self.params
        t = np.arange(length, dtype=np.float64)
        log_base = rng.uniform(*p.log_base_range)

        # Trend: adoption S-curve, or gentle exponential growth/decay.
        if rng.random() < p.p_saturating:
            k = rng.uniform(0.8, 3.0)
            x0 = rng.uniform(0.1, 0.9) * length
            width = length * rng.uniform(0.05, 0.25)
            trend = k / (1.0 + np.exp(-(t - x0) / width))
            trend -= trend[0]
        else:
            trend = rng.normal(0.0, 0.6) * t / length

        # Strong weekly profile: per-weekday effects + a usually-present weekend
        # dip. Amplitude high relative to noise → seasonal-lag ACF ≈ 0.85-0.9.
        weekday = rng.normal(0.0, 0.18, 7)
        if rng.random() < p.weekend_dip_prob:
            weekday[5:] -= rng.uniform(0.35, 1.1)
        weekday -= weekday.mean()
        weekday *= rng.uniform(0.8, 1.6)
        phase = int(rng.integers(0, 7))
        seas = weekday[(t.astype(np.int64) + phase) % 7]

        # Holiday dips: short multi-day multiplicative drops.
        holidays = np.zeros(length)
        for _ in range(rng.poisson(p.holiday_rate * length / 365.0)):
            start = int(rng.integers(0, max(1, length - 5)))
            dur = int(rng.integers(1, 6))
            holidays[start:start + dur] -= rng.uniform(0.2, 0.9)

        phi = rng.uniform(0.6, 0.95)
        sig = rng.uniform(*p.ar_sigma_range)
        noise = _ar1(rng, length, phi, sig)

        log_lam = log_base + trend + seas + holidays + noise
        return _poissonize(rng, log_lam)


# ── pageviews: wikimedia-like ────────────────────────────────────────────────


class PageviewsGeneratorParams(GeneratorParams):
    """Weakly-weekly heavy-tailed count series with news-event spikes."""

    log_base_range: tuple[float, float] = (np.log(20.0), np.log(3e4))
    student_df_range: tuple[float, float] = (2.0, 4.0)   # tail index target ~2.5
    event_rate: float = 4.5            # news events per 365 steps
    level_shift_prob: float = 0.5
    weekly_amp_range: tuple[float, float] = (0.18, 0.50)  # weak-moderate → ACF7 ~0.4


class PageviewsGeneratorWrapper(GeneratorWrapper):
    def __init__(self, params: PageviewsGeneratorParams):
        super().__init__(params)

    def generate_batch(self, batch_size: int, seed: int | None = None) -> TimeSeriesContainer:
        if seed is None:
            seed = int(self.params.global_seed)
        self._set_random_seeds(seed)
        length = int(self.params.length)
        values = np.empty((batch_size, length), dtype=np.float64)
        for i in range(batch_size):
            rng = np.random.default_rng((int(seed) + i) % (2**31))
            values[i] = self._one_series(rng, length)
        sampled = self._sample_parameters(batch_size)
        return TimeSeriesContainer(values=values, start=sampled["start"],
                                   frequency=sampled["frequency"])

    def _one_series(self, rng: np.random.Generator, length: int) -> np.ndarray:
        p = self.params
        t = np.arange(length, dtype=np.float64)
        log_base = rng.uniform(*p.log_base_range)

        # Weak weekly structure (readers browse a bit less on weekends).
        weekday = rng.normal(0.0, rng.uniform(*p.weekly_amp_range), 7)
        weekday -= weekday.mean()
        phase = int(rng.integers(0, 7))
        seas = weekday[(t.astype(np.int64) + phase) % 7]

        # Slow popularity drift + occasional regime (level) shifts.
        drift = _ar1(rng, length, rng.uniform(0.95, 0.995), rng.uniform(0.008, 0.025))
        shifts = np.zeros(length)
        while rng.random() < p.level_shift_prob:
            at = int(rng.integers(length // 10, length))
            shifts[at:] += rng.normal(0.0, 0.35)

        # Heavy-tailed day-to-day noise: Student-t innovations in log space.
        df = rng.uniform(*p.student_df_range)
        scale = rng.uniform(0.05, 0.16)
        heavy = rng.standard_t(df, length) * scale

        # News events: sudden jump, geometric decay, occasional 1-day build-up.
        events = np.zeros(length)
        for _ in range(rng.poisson(p.event_rate * length / 365.0)):
            at = int(rng.integers(1, length))
            mag = rng.pareto(rng.uniform(1.6, 3.0)) + 0.35    # log-space jump
            mag = min(mag, 3.2)
            decay = rng.uniform(0.5, 0.92)
            span = min(length - at, 30)
            events[at:at + span] += mag * (decay ** np.arange(span))
            if at >= 1 and rng.random() < 0.3:
                events[at - 1] += mag * rng.uniform(0.1, 0.4)

        log_lam = log_base + seas + drift + shifts + heavy + events
        return _poissonize(rng, log_lam)


# ── ksynth_cal: KernelSynth (exact) + Mean-KernelSynth extension ─────────────


class KSynthCalGeneratorParams(GeneratorParams):
    """KernelSynth 33-kernel bank, random +/× composition, optional mean fn."""

    max_kernels: int = 5
    mean_fn_prob: float = 0.5      # Mean-KernelSynth (CauKer ablation) share
    scale_range: tuple[float, float] = (0.5, 3.0)


class KSynthCalGeneratorWrapper(GeneratorWrapper):
    def __init__(self, params: KSynthCalGeneratorParams):
        super().__init__(params)
        self._bank = None

    def _kernel_bank(self):
        if self._bank is None:
            from sklearn.gaussian_process.kernels import (
                RBF,
                ConstantKernel,
                DotProduct,
                ExpSineSquared,
                RationalQuadratic,
                WhiteKernel,
            )
            length = float(self.params.length)
            periods = [24, 48, 96, 24 * 7, 48 * 7, 96 * 7, 7, 14, 30, 60,
                       365, 365 * 2, 4, 4, 4, 26, 52, 6, 12, 40, 10]
            bank = [ExpSineSquared(periodicity=p / length) for p in periods]
            bank += [DotProduct(sigma_0=s) for s in (0.0, 1.0, 10.0)]
            bank += [RBF(length_scale=s) for s in (0.1, 1.0, 10.0)]
            bank += [RationalQuadratic(alpha=s) for s in (0.1, 1.0, 10.0)]
            bank += [WhiteKernel(noise_level=s) for s in (0.1, 1.0)]
            bank += [ConstantKernel()]
            self._bank = bank
        return self._bank

    def generate_batch(self, batch_size: int, seed: int | None = None) -> TimeSeriesContainer:
        if seed is None:
            seed = int(self.params.global_seed)
        self._set_random_seeds(seed)
        length = int(self.params.length)
        values = np.empty((batch_size, length), dtype=np.float64)
        for i in range(batch_size):
            rng = np.random.default_rng((int(seed) + i) % (2**31))
            values[i] = self._one_series(rng, length)
        sampled = self._sample_parameters(batch_size)
        return TimeSeriesContainer(values=values, start=sampled["start"],
                                   frequency=sampled["frequency"])

    def _one_series(self, rng: np.random.Generator, length: int) -> np.ndarray:
        import functools

        bank = self._kernel_bank()
        x = np.linspace(0.0, 1.0, length)[:, None]
        for _attempt in range(10):
            j = int(rng.integers(1, self.params.max_kernels + 1))
            picks = [bank[int(k)] for k in rng.integers(0, len(bank), j)]
            kernel = functools.reduce(
                lambda a, b: a + b if rng.random() < 0.5 else a * b, picks)
            try:
                cov = kernel(x)
                cov[np.diag_indices_from(cov)] += 1e-8
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                continue  # KernelSynth behavior: resample a fresh kernel
            y = chol @ rng.standard_normal(length)
            break
        else:
            y = rng.standard_normal(length)

        # Mean-KernelSynth: add a non-zero mean function half the time.
        if rng.random() < self.params.mean_fn_prob:
            t = np.linspace(0.0, 1.0, length)
            kind = rng.random()
            if kind < 0.4:
                mean = rng.normal(0.0, 2.0) * t
            elif kind < 0.7:
                mean = rng.uniform(0.5, 3.0) * np.exp(rng.normal(0.0, 1.0) * t)
                mean -= mean[0]
            else:
                step_at = int(rng.integers(length // 5, 4 * length // 5))
                mean = np.zeros(length)
                mean[step_at:] += rng.normal(0.0, 2.0)
            y = y + mean

        return y * rng.uniform(*self.params.scale_range)

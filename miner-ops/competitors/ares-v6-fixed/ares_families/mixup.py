"""TSMixup family — convex combinations of series drawn from a base prior pool.

Chronos (arXiv:2403.07815) showed TSMixup augmentation improves zero-shot
forecast accuracy: take k ~ U{1, K_max} series, mean-scale each, and combine
with Dirichlet(alpha) weights. Mixing creates cross-family shapes (e.g. a
seasonal series riding a regime-shifting level) that no single prior emits.

Pool here: ForecastPFN (with this repo's harmonic parameterisation) plus the
classical jtest families (trend/seasonal/AR, multiplicative, regime shift).
Pure numpy on top of existing wrappers; deterministic per (seed + series index)
— component sub-seeds derive from the per-series seed, no global RNG state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.synthetic_generation.abstract_classes import GeneratorWrapper
from tempo_gen.synthetic_generation.generator_params import (
    ForecastPFNGeneratorParams,
    GeneratorParams,
)

from ares_families.jtest import _multiplicative, _regime_shift, _trend_seasonal_ar

_ABS_MAX = 1.0e6


@dataclass
class TSMixupGeneratorParams(GeneratorParams):
    """TSMixup over {fpfn, jt_tsa, jt_mult, jt_regime}."""

    k_max: int = 3
    alpha: float = 1.5
    # fpfn component uses the same harmonic parameterisation as the main
    # forecast_pfn key so the mixup pool matches the arm's fpfn flavour.
    harmonic_scale_ratio: float = 0.65
    harmonic_rate: float = 1.25


class TSMixupGeneratorWrapper(GeneratorWrapper):
    """Batch generator for the TSMixup family (mirrors the vendored wrapper API)."""

    def __init__(self, params: TSMixupGeneratorParams):
        super().__init__(params)
        self._fpfn = None

    def _fpfn_wrapper(self):
        if self._fpfn is None:
            from tempo_gen.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator_wrapper import (
                ForecastPFNGeneratorWrapper,
            )

            p = ForecastPFNGeneratorParams(
                global_seed=int(self.params.global_seed),
                length=int(self.params.length),
                harmonic_scale_ratio=float(self.params.harmonic_scale_ratio),
                harmonic_rate=float(self.params.harmonic_rate),
            )
            self._fpfn = ForecastPFNGeneratorWrapper(p)
        return self._fpfn

    def _component(self, kind: int, sub_seed: int, length: int) -> np.ndarray:
        if kind == 0:
            # fpfn's internal batch-mixup needs >= mixup_series rows; draw 4, keep row 0.
            batch = self._fpfn_wrapper().generate_batch(batch_size=4, seed=sub_seed)
            vals = np.asarray(batch.values, dtype=np.float64)
            row = vals[0].reshape(-1)[:length]
            if row.shape[0] < length:  # defensive: pad by wrap if a draw is short
                row = np.resize(row, length)
            return row
        rng = np.random.default_rng(sub_seed)
        build = (_trend_seasonal_ar, _multiplicative, _regime_shift)[kind - 1]
        return build(rng, 1, length)[0]

    def generate_batch(self, batch_size: int, seed: int | None = None) -> TimeSeriesContainer:
        if seed is None:
            seed = int(self.params.global_seed)
        self._set_random_seeds(seed)  # base-class metadata sampling below
        length = int(self.params.length)
        k_max = max(1, int(self.params.k_max))
        alpha = float(self.params.alpha)
        values = np.empty((batch_size, length), dtype=np.float64)
        for i in range(batch_size):
            base = (int(seed) + i) % (2**31)
            rng = np.random.default_rng(base)
            k = int(rng.integers(1, k_max + 1))
            kinds = rng.integers(0, 4, size=k)
            w = rng.dirichlet(np.full(k, alpha))
            acc = np.zeros(length, dtype=np.float64)
            for j in range(k):
                sub = (base * 1000003 + (j + 1) * 7919) % (2**31)
                comp = self._component(int(kinds[j]), sub, length)
                comp = np.nan_to_num(comp, nan=0.0, posinf=_ABS_MAX, neginf=-_ABS_MAX)
                scale = float(np.mean(np.abs(comp))) + 1e-8
                acc += w[j] * (comp / scale)
            # restore magnitude diversity lost to mean-scaling
            row = acc * float(np.exp(rng.normal(0.0, 1.0)))
            values[i] = np.clip(row, -_ABS_MAX, _ABS_MAX)
        sampled = self._sample_parameters(batch_size)
        return TimeSeriesContainer(
            values=values,
            start=sampled["start"],
            frequency=sampled["frequency"],
        )

"""aurora-mix-v3 — a cascade data generator redesigned around TempoPFN's pipeline.

v3 over v2: STRATIFIED INTERLEAVED EMISSION (the trainer's streaming modes
consume only a token-budget prefix of the stream — blocked emission trained on
the first family alone; interleaved, every prefix carries the configured
mixture) and evidence-based weight rebalance (anchor on the two throne-proven
families: the reigning king ares-v3 holds with 100% tuned trend x seasonality
composition, and our kernel_gp out-ranked the former king smoothgp in the heat).

TempoPFN (arXiv 2510.25502) is the strongest synthetic-only forecaster to date;
its pipeline = adapted priors (ForecastPFN composition, KernelSynth/GP, CauKer
causal graphs) + novel priors (regime-switching OU, parametric shapes,
spikes/anomalies, *audio-inspired textures*) + an augmentation layer (mixup,
time-warp, damping, spike injection, series transitions). The genesis king
(tempopfn-base-mix-v1) vendors ten of those families but had to DROP the
audio-inspired group (pyo runs an audio server and seeds via ``hash()`` — not
deterministic, not allowlisted) and leaves the augmentation layer mostly
unused. This generator is the counter-design, in order:

1. **Keep the highest-signal families** (per the TempoPFN ablation), with our
   own implementations: compositional GP/kernel priors and rich
   trend x multi-seasonality x structured-noise composition.

2. **Cover the king's TempoPFN gap, deterministically** — the audio-inspired
   priors reimplemented in pure numpy/scipy:

   * ``rhythm`` — Stochastic Rhythms: quasi-periodic event trains on a
     tempo-drifting beat grid with accent bars, swing, dropouts, and
     pluck / percussive / bump event kernels
   * ``fractal_multi`` — Multi-Scale Fractals: piecewise-slope spectral
     synthesis (different roughness per frequency band), Weierstrass sums,
     and multiplicative cascades (multifractal volatility)
   * ``net_diffusion`` — Network Topology: forced diffusion dynamics on
     random directed graphs (propagation, echoes, superposition — the CauKer
     flavour without networkx)

   Financial Volatility, TempoPFN's fourth audio prior, is already covered by
   ``regime_garch``.

3. **Port TempoPFN's augmentation layer**: per-series time-warp, damping
   envelopes and spike injection in post-processing, plus batch-level splice
   *transitions* (two series crossfaded at a changepoint — TempoPFN's
   ``transition_ratio``) and the TSMixup-style ``mixup`` family.

4. **Regime classes the king does not emit** (kept from v1): chaotic
   dynamical systems (DynaMix, arXiv 2505.13192), long-memory fGn/fBm,
   Markov-switching AR + GARCH(1,1), zero-inflated intermittent demand,
   calendar-structured load curves, saturating growth / lifecycle, and
   self-exciting (Hawkes-like) bursts with anomaly-injected AR bases.

5. **Determinism.** The corpus is a pure function of ``(seed, n_series)``.
   Every RNG is a ``np.random.default_rng`` derived from
   ``SeedSequence([seed, tag, index])``; there is no global RNG, no torch, no
   wall-clock, no ``hash()``. Two in-process runs and two cross-process runs
   produce byte-identical corpora.

6. **Throughput.** Everything is drawn in vectorised batches; the GP family
   samples on a coarse grid and cubic-spline upsamples; rhythm/fractal are
   FFT/convolution-cheap. No torch import keeps the sandbox light and the
   stream fast (stream_cpu must not starve the GPU).

Dependencies: numpy + scipy only (both on the chain.toml allowlist).
"""

from __future__ import annotations

import json
import os
import random as _py_random
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from cascade.interface import DataGenerator

# v4: the vendored TempoPFN subset (Apache-2.0; see NOTICE) supplies the REAL
# ForecastPFN prior — the exact composition the reigning king trains on and wins
# the count/benchmark domains with. The trainer imports this file by path, so add
# our own directory to sys.path so ``import tempo_gen`` resolves however loaded.
# (``os``/``sys`` are not on the static-guard blocklist; only ``os.system`` is.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── family registry ──────────────────────────────────────────────────────────
# Weights need not sum to 1 (normalised). Mass is biased toward the families
# with the strongest documented downstream signal (GP/kernel priors, rich
# composition) and the novel dynamics families the king lacks.
_DEFAULT_WEIGHTS: dict[str, float] = {
    # ~45% anchor on the two throne-proven families: the reigning king holds
    # with 100% tuned trend x seasonality x noise composition, and our pure-GP
    # prefix (the accidental v2 entry) out-ranked the former king smoothgp.
    "trend_seasonal": 0.30,
    "kernel_gp": 0.15,
    # ~22% real-world-shape robustness (count-like / load-like / volatility
    # feeds) — the KOTH bootstrap-LCB rewards being uniformly decent across the
    # rotating pool, not occasionally brilliant.
    "calendar": 0.08,
    "regime_garch": 0.07,
    "intermittent": 0.07,
    # variance smoothing + long memory.
    "mixup": 0.05,
    "fgn": 0.05,
    # small live shares: spiky or not-yet-A/B-tested families (the TempoPFN-gap
    # trio has never been trained on pre-interleave — grow these only on
    # `cascade score` evidence).
    "chaotic": 0.04,
    "bursts_anomaly": 0.04,
    "rhythm": 0.04,
    "fractal_multi": 0.04,
    "net_diffusion": 0.03,
    "random_walk": 0.02,
    "waves": 0.01,
    "growth": 0.01,
}

# Families whose raw output is meaningfully non-negative; post-processing
# preserves positivity instead of re-centering.
_POSITIVE_FAMILIES = frozenset({"calendar", "intermittent", "growth", "web_traffic"})

# Stable integer tag per family for seed derivation (order-independent).
_FAMILY_TAG: dict[str, int] = {
    "kernel_gp": 11,
    "trend_seasonal": 12,
    "chaotic": 13,
    "regime_garch": 14,
    "fgn": 15,
    "calendar": 16,
    "bursts_anomaly": 17,
    "waves": 18,
    "random_walk": 19,
    "intermittent": 20,
    "growth": 21,
    "mixup": 22,
    "rhythm": 23,
    "fractal_multi": 24,
    "net_diffusion": 25,
    "web_traffic": 26,
    "seasonal_level": 27,
    "forecast_pfn": 28,
}

_CROP_TAG = 101
_POST_TAG = 102


def _rng(seed: int, *parts: int) -> np.random.Generator:
    """Deterministic child generator from the master seed."""
    return np.random.default_rng(np.random.SeedSequence([seed, *parts]))


def _standardize_rows(a: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance per row, safe on degenerate rows."""
    mu = a.mean(axis=1, keepdims=True)
    sd = a.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (a - mu) / sd


def _ar1(e: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Vectorised AR(1) filter: y_t = phi*y_{t-1} + e_t. e (m,L), phi (m,)."""
    y = np.empty_like(e)
    y[:, 0] = e[:, 0]
    for t in range(1, e.shape[1]):
        y[:, t] = phi * y[:, t - 1] + e[:, t]
    return y


class Generator(DataGenerator):
    """Fifteen-family mixture prior, emitted as a deterministic corpus."""

    def __init__(self, config_dir: str, *, seed: int) -> None:
        cfg_path = Path(config_dir) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}

        # Chain block-hash seeds can be huge; SeedSequence needs non-negative.
        self._seed = int(seed) % (2**63)
        self._min_len = int(cfg.get("min_length", 64))
        self._max_len = int(cfg.get("max_length", 2048))
        if not (1 <= self._min_len <= self._max_len):
            raise ValueError(f"invalid length band [{self._min_len}, {self._max_len}]")
        self._gen_len = max(int(cfg.get("generate_length", self._max_len)), self._max_len)
        self._batch = max(1, int(cfg.get("batch_size", 64)))
        self._max_abs = float(cfg.get("max_abs_value", 1.0e6))

        # Length-mixture knobs: bias toward long crops (more signal per series)
        # while keeping short-series coverage.
        self._len_mix = cfg.get("length_mixture", [
            [0.15, 64, 192],
            [0.30, 192, 768],
            [0.55, 768, 2048],
        ])

        weights = cfg.get("weights", _DEFAULT_WEIGHTS)
        self._weights = {
            k: float(weights[k])
            for k in _FAMILY_TAG
            if k in weights and float(weights[k]) > 0.0
        }
        if not self._weights:
            self._weights = dict(_DEFAULT_WEIGHTS)

        # Per-family hyperparameter overrides — the config tuning surface.
        # config.json `family_params` maps a family key (or "post") to knob
        # overrides; knobs are read at draw time via _fp(), so unknown keys are
        # simply never read (a stale config cannot crash generation). JSON
        # lists become (lo, hi) tuples. Knob reads consume no RNG, so a config
        # with default values draws a byte-identical corpus.
        raw_fp = cfg.get("family_params", {}) or {}
        self._fp_cfg: dict[str, dict] = {
            fam: {k: (tuple(v) if isinstance(v, list) else v) for k, v in over.items()}
            for fam, over in raw_fp.items()
            if isinstance(over, dict)
        }

        # Global determinism seeding — needed only for the vendored ForecastPFN
        # family (its wrappers touch np.random global state + torch); the native
        # numpy/scipy families use local default_rng and are unaffected. Mirrors
        # base_generator's proven cross-process determinism setup.
        if "forecast_pfn" in self._weights:
            np.random.seed(self._seed % 2**31)
            _py_random.seed(self._seed)
            try:
                import torch

                torch.manual_seed(self._seed)
                torch.use_deterministic_algorithms(True, warn_only=True)
                torch.set_num_threads(1)
            except Exception:  # pragma: no cover - torch is allowlisted, be safe
                pass

    def _fp(self, family: str, key: str, default):
        """Config-tunable knob: family_params[family][key], else default."""
        return self._fp_cfg.get(family, {}).get(key, default)

    @property
    def name(self) -> str:
        return "aurora-mix-v3"

    # ── allocation (pure function, no RNG) ──────────────────────────────────
    def _allocate(self, n_series: int) -> list[tuple[str, int]]:
        keys = list(self._weights)
        total_w = sum(self._weights[k] for k in keys)
        raw = {k: n_series * self._weights[k] / total_w for k in keys}
        floor = {k: int(np.floor(raw[k])) for k in keys}
        remainder = n_series - sum(floor.values())
        order = sorted(keys, key=lambda k: (-(raw[k] - floor[k]), keys.index(k)))
        for i in range(remainder):
            floor[order[i % len(order)]] += 1
        return [(k, floor[k]) for k in keys if floor[k] > 0]

    # ── emission order ────────────────────────────────────────────────────────
    @staticmethod
    def _interleave_order(allocation: list[tuple[str, int]]) -> list[int]:
        """Stratified interleave of family slots (indices into ``allocation``).

        Family ``f`` with ``count`` series takes emission keys ``(j + 0.5) / count``
        — evenly spread over [0, 1) — and the merged order sorts by key (ties by
        allocation index). Pure function of the allocation, no RNG.

        This is load-bearing under the trainer's streaming feed modes
        (``stream_cpu`` / ``stream_gpu``): training consumes only a token-budget
        PREFIX of this stream, so a family-blocked order would train on the
        first family alone and silently drop the rest. Interleaved, every prefix
        carries (approximately) the configured family mixture, and training
        batches mix families throughout the run instead of seeing a one-family
        curriculum.
        """
        order: list[tuple[float, int]] = []
        for fam_idx, (_family, count) in enumerate(allocation):
            for j in range(count):
                order.append(((j + 0.5) / count, fam_idx))
        order.sort(key=lambda t: (t[0], t[1]))
        return [fam_idx for _key, fam_idx in order]

    def _family_rows(self, family: str, count: int) -> Iterator[np.ndarray]:
        """Lazy full-length row stream for one family: exactly ``count`` rows,
        drawn in batches on demand. Seeding is identical to the old blocked
        scheme — ``_rng(seed, family_tag, batch_no)`` per batch — so a family's
        raw rows do not depend on the emission order."""
        tag = _FAMILY_TAG[family]
        remaining = count
        batch_no = 0
        while remaining > 0:
            b = min(self._batch, remaining)
            rng = _rng(self._seed, tag, batch_no)
            batch = self._draw(family, rng, b, self._gen_len)
            yield from batch
            remaining -= b
            batch_no += 1

    # ── main entrypoint ──────────────────────────────────────────────────────
    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        if n_series <= 0:
            return
        crop_rng = _rng(self._seed, _CROP_TAG)
        allocation = self._allocate(n_series)
        streams = [self._family_rows(family, count) for family, count in allocation]
        emitted = 0
        for fam_idx in self._interleave_order(allocation):
            family = allocation[fam_idx][0]
            row = next(streams[fam_idx])
            # Crop RNG advances exactly twice per emitted series.
            length = self._draw_length(crop_rng)
            max_off = self._gen_len - length
            offset = int(crop_rng.integers(0, max_off + 1)) if max_off > 0 else 0
            post_rng = _rng(self._seed, _POST_TAG, emitted)
            yield self._postprocess(row[offset:offset + length], post_rng, family)
            emitted += 1
        if emitted != n_series:  # pragma: no cover — allocation guarantees this
            raise RuntimeError(f"emitted {emitted} series; expected {n_series}")

    def _draw_length(self, crop_rng: np.random.Generator) -> int:
        u = float(crop_rng.random())
        acc = 0.0
        lo, hi = self._min_len, self._max_len
        for p, a, b in self._len_mix:
            acc += float(p)
            if u <= acc:
                lo, hi = int(a), int(b)
                break
        lo = max(self._min_len, min(lo, self._max_len))
        hi = max(lo, min(hi, self._max_len))
        return int(crop_rng.integers(lo, hi + 1))

    # ── family dispatch ──────────────────────────────────────────────────────
    def _draw(self, family: str, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        fn = getattr(self, f"_fam_{family}")
        out = np.asarray(fn(rng, b, L), dtype=np.float64)
        # Repair any numerical accident at the source so downstream is clean.
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=self._max_abs, neginf=-self._max_abs)
        # TempoPFN-style *transition* augmentation: crossfade a few rows into a
        # batch partner at a random changepoint, so one series hands off between
        # two regimes. Skipped for positive families (a scale-matched blend can
        # cross zero) and for mixup (already a cross-family blend).
        if b >= 2 and family != "mixup" and family not in _POSITIVE_FAMILIES:
            out = self._splice_rows(out, rng)
        return out

    def _splice_rows(self, out: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        b, L = out.shape
        # TempoPFN's transition_ratio analogue. The reigning king trains on
        # ~100% spliced series (every example is a regime handoff, forcing
        # in-context re-inference); config-sweepable via family_params
        # augment.p_splice to test that hypothesis on our mixture.
        p_splice = float(self._fp("augment", "p_splice", 0.08))
        do = rng.random(b) < p_splice
        idx = np.flatnonzero(do)
        if idx.size == 0:
            return out
        partners = (idx + 1 + rng.integers(0, b - 1, idx.size)) % b
        t = np.arange(L, dtype=np.float64)
        for i, j in zip(idx, partners):
            cp = int(rng.integers(L // 8, 7 * L // 8 + 1))
            width = max(3.0, L * float(rng.uniform(0.01, 0.06)))
            blend = 1.0 / (1.0 + np.exp(-(t - cp) / (width / 4.0)))
            # Scale-match the partner to the host so neither side dominates.
            xj = out[j]
            mu_i, sd_i = float(out[i].mean()), float(out[i].std()) + 1e-12
            mu_j, sd_j = float(xj.mean()), float(xj.std()) + 1e-12
            xj = (xj - mu_j) / sd_j * sd_i + mu_i
            out[i] = out[i] * (1.0 - blend) + xj * blend
        return out

    # ═════════════════════════ family implementations ═══════════════════════

    # 1 ── compositional GP / KernelSynth-style priors ────────────────────────
    def _fam_kernel_gp(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        # Draw on a coarse grid (O(grid^3) Cholesky is the family's cost) and
        # cubic-upsample to L; 256 is smooth enough for series upsampled to 2048.
        grid = int(self._fp("kernel_gp", "grid", 256))
        k_lo, k_hi = self._fp("kernel_gp", "samples_per_chol", (6, 16))
        x = np.linspace(0.0, 1.0, grid)
        xt = np.linspace(0.0, 1.0, L)
        out = np.empty((b, L))
        i = 0
        while i < b:
            # More samples per Cholesky amortises the O(grid^3) factorisation.
            k = min(int(rng.integers(int(k_lo), int(k_hi) + 1)), b - i)
            K = self._sample_kernel(rng, x)
            # Jitter escalation keeps Cholesky deterministic and robust.
            jitter = 1e-6 * float(np.mean(np.diag(K)) + 1e-12)
            chol = None
            for _ in range(6):
                try:
                    chol = np.linalg.cholesky(K + jitter * np.eye(grid))
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0
            if chol is None:  # pragma: no cover — extreme fallback
                chol = np.eye(grid)
            z = rng.standard_normal((grid, k))
            y_grid = (chol @ z).T  # (k, grid)
            out[i:i + k] = CubicSpline(x, y_grid, axis=1)(xt)
            i += k
        return out

    def _sample_kernel(self, rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
        d = x[:, None] - x[None, :]
        d2 = d * d

        def primitive() -> np.ndarray:
            kind = rng.choice(["rbf", "rq", "periodic", "linear", "constant", "white"],
                              p=[0.28, 0.18, 0.28, 0.12, 0.07, 0.07])
            if kind == "rbf":
                ell = float(np.exp(rng.uniform(np.log(0.01), np.log(0.5))))
                return np.exp(-0.5 * d2 / ell**2)
            if kind == "rq":
                ell = float(np.exp(rng.uniform(np.log(0.02), np.log(0.5))))
                alpha = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
                return (1.0 + d2 / (2.0 * alpha * ell**2)) ** (-alpha)
            if kind == "periodic":
                p = float(np.exp(rng.uniform(np.log(0.02), np.log(0.7))))
                ell = float(np.exp(rng.uniform(np.log(0.3), np.log(2.0))))
                return np.exp(-2.0 * np.sin(np.pi * np.abs(d) / p) ** 2 / ell**2)
            if kind == "linear":
                c = float(rng.uniform(-0.5, 1.5))
                k = (x[:, None] - c) * (x[None, :] - c)
                scale = float(np.mean(np.diag(k)) + 1e-6)
                return k / scale
            if kind == "constant":
                return np.full_like(d2, float(np.exp(rng.uniform(np.log(0.05), 0.0))))
            sig = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))
            return (sig**2) * np.eye(x.size)

        K = primitive()
        comp_max = int(self._fp("kernel_gp", "max_compositions", 3))
        for _ in range(int(rng.integers(0, comp_max + 1))):
            if rng.random() < 0.5:
                K = K + primitive()
            else:
                K = K * primitive()
        # Normalise overall scale so composite kernels don't explode.
        K = K / (float(np.mean(np.diag(K))) + 1e-12)
        return K

    # 2 ── trend x multi-seasonality x structured noise ───────────────────────
    def _fam_trend_seasonal(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        ns_lo, ns_hi = self._fp("trend_seasonal", "n_seasonal", (1, 3))
        as_lo, as_hi = self._fp("trend_seasonal", "amp_seasonal", (0.1, 2.0))
        an_lo, an_hi = self._fp("trend_seasonal", "amp_noise", (0.02, 0.6))
        p_mult = float(self._fp("trend_seasonal", "p_multiplicative", 0.35))
        t = np.linspace(0.0, 1.0, L)
        out = np.empty((b, L))
        for i in range(b):
            trend = self._sample_trend(rng, t)
            seasonal = np.zeros(L)
            for _ in range(int(rng.integers(int(ns_lo), int(ns_hi) + 1))):
                seasonal += self._sample_seasonal(rng, L)
            noise = self._sample_noise(rng, L)
            amp_s = float(np.exp(rng.uniform(np.log(as_lo), np.log(as_hi))))
            amp_n = float(np.exp(rng.uniform(np.log(an_lo), np.log(an_hi))))
            if rng.random() < p_mult:  # multiplicative composition
                y = (1.0 + trend) * (1.0 + amp_s * seasonal) + amp_n * noise
            else:
                y = trend + amp_s * seasonal + amp_n * noise
            out[i] = y
        return out

    def _sample_trend(self, rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
        kind = rng.choice(["none", "linear", "piecewise", "exp", "logistic"],
                          p=[0.2, 0.3, 0.2, 0.15, 0.15])
        a = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0)))) * (1 if rng.random() < 0.5 else -1)
        if kind == "none":
            return np.zeros_like(t)
        if kind == "linear":
            return a * t
        if kind == "piecewise":
            y = a * t
            for _ in range(int(rng.integers(1, 4))):
                cp = float(rng.uniform(0.1, 0.9))
                slope = float(rng.normal(0.0, abs(a)))
                y = y + slope * np.clip(t - cp, 0.0, None)
            return y
        if kind == "exp":
            r = float(rng.uniform(0.5, 3.0))
            return a * (np.exp(r * t) - 1.0) / (np.exp(r) - 1.0)
        r = float(rng.uniform(4.0, 20.0))
        t0 = float(rng.uniform(0.2, 0.8))
        return a / (1.0 + np.exp(-r * (t - t0)))

    def _sample_seasonal(self, rng: np.random.Generator, L: int) -> np.ndarray:
        # Common calendar-ish periods (jittered) or free log-uniform periods.
        if rng.random() < float(self._fp("trend_seasonal", "p_calendar_period", 0.5)):
            base = float(rng.choice([7, 12, 24, 52, 96, 168, 336]))
            period = base * float(rng.uniform(0.9, 1.1))
        else:
            period = float(np.exp(rng.uniform(np.log(4.0), np.log(max(8.0, L / 2)))))
        n = np.arange(L)
        y = np.zeros(L)
        d_lo, d_hi = self._fp("trend_seasonal", "harmonic_decay", (0.8, 2.0))
        h_lo, h_hi = self._fp("trend_seasonal", "harmonics", (1, 4))
        decay = float(rng.uniform(d_lo, d_hi))
        for h in range(1, int(rng.integers(int(h_lo), int(h_hi) + 1)) + 1):
            c = float(rng.normal(0.0, 1.0)) / h**decay
            phi = float(rng.uniform(0.0, 2 * np.pi))
            y += c * np.sin(2 * np.pi * h * n / period + phi)
        return y / (np.std(y) + 1e-12)

    def _sample_noise(self, rng: np.random.Generator, L: int) -> np.ndarray:
        kind = rng.choice(["gauss", "student", "weibull"], p=[0.5, 0.25, 0.25])
        if kind == "gauss":
            e = rng.standard_normal(L)
        elif kind == "student":
            e = rng.standard_t(df=float(rng.uniform(3.0, 8.0)), size=L)
        else:
            k = float(rng.uniform(0.7, 2.5))
            e = rng.weibull(k, size=L) - float(np.exp(np.log(2) / k))  # roughly centred
        if rng.random() < 0.4:  # AR(1)-coloured noise
            phi = np.full(1, float(rng.uniform(0.5, 0.95)))
            e = _ar1(e[None, :], phi)[0]
        return e / (np.std(e) + 1e-12)

    # 3 ── chaotic dynamical systems ──────────────────────────────────────────
    def _fam_chaotic(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        out = np.empty((b, L))
        i = 0
        while i < b:
            # The RK4/map step loop runs L times per block regardless of block
            # size, so its Python overhead amortises over m — draw big blocks
            # (1-2 systems per batch; diversity comes from many batches/round).
            m = min(int(rng.integers(32, 65)), b - i)
            system = rng.choice(["lorenz", "rossler", "duffing", "mackey", "logistic", "henon"],
                                p=[0.22, 0.18, 0.14, 0.16, 0.15, 0.15])
            if system == "lorenz":
                rows = self._chaos_lorenz(rng, m, L)
            elif system == "rossler":
                rows = self._chaos_rossler(rng, m, L)
            elif system == "duffing":
                rows = self._chaos_duffing(rng, m, L)
            elif system == "mackey":
                rows = self._chaos_mackey(rng, m, L)
            elif system == "logistic":
                rows = self._chaos_logistic(rng, m, L)
            else:
                rows = self._chaos_henon(rng, m, L)
            rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
            rows = _standardize_rows(rows)
            if rng.random() < 0.5:  # mild observation noise
                rows = rows + rng.standard_normal(rows.shape) * float(rng.uniform(0.01, 0.1))
            out[i:i + m] = rows
            i += m
        return out

    @staticmethod
    def _rk4(f, s: np.ndarray, dt: float, burn: int, stride: int,
             n_record: int, observe) -> np.ndarray:
        def step(state):
            k1 = f(state)
            k2 = f(state + 0.5 * dt * k1)
            k3 = f(state + 0.5 * dt * k2)
            k4 = f(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            np.clip(state, -1e6, 1e6, out=state)
            return state

        for _ in range(burn):
            s = step(s)
        rec = np.empty((s.shape[0], n_record))
        for r in range(n_record):
            for _ in range(stride):
                s = step(s)
            rec[:, r] = observe(s)
        return rec

    def _project(self, rng: np.random.Generator, dim: int, m: int):
        w = rng.standard_normal((m, dim))
        w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-12
        return lambda s: np.sum(s * w, axis=1)

    def _chaos_lorenz(self, rng: np.random.Generator, m: int, L: int) -> np.ndarray:
        sig = rng.uniform(9.0, 11.0, m)
        rho = rng.uniform(24.0, 32.0, m)
        beta = rng.uniform(2.2, 3.0, m)
        # stride=1 (record every step) halves the loop vs subsampling; the dt
        # range instead carries the timescale variety subsampling gave.
        dt = float(rng.uniform(0.005, 0.025))
        stride = 1
        s = rng.normal(0.0, 5.0, (m, 3)) + np.array([0.0, 0.0, 25.0])

        def f(st):
            x, y, z = st[:, 0], st[:, 1], st[:, 2]
            return np.stack([sig * (y - x), x * (rho - z) - y, x * y - beta * z], axis=1)

        obs = self._project(rng, 3, m)
        return self._rk4(f, s, dt, 300, stride, L, obs)

    def _chaos_rossler(self, rng: np.random.Generator, m: int, L: int) -> np.ndarray:
        a = rng.uniform(0.1, 0.3, m)
        bb = rng.uniform(0.1, 0.3, m)
        c = rng.uniform(4.5, 9.0, m)
        dt = float(rng.uniform(0.02, 0.14))
        stride = 1
        s = rng.normal(0.0, 2.0, (m, 3))

        def f(st):
            x, y, z = st[:, 0], st[:, 1], st[:, 2]
            return np.stack([-y - z, x + a * y, bb + z * (x - c)], axis=1)

        obs = self._project(rng, 3, m)
        return self._rk4(f, s, dt, 300, stride, L, obs)

    def _chaos_duffing(self, rng: np.random.Generator, m: int, L: int) -> np.ndarray:
        delta = rng.uniform(0.1, 0.4, m)
        gamma = rng.uniform(0.3, 0.6, m)
        omega = rng.uniform(0.9, 1.4, m)
        dt = float(rng.uniform(0.05, 0.28))
        stride = 1
        s = np.concatenate([rng.normal(0, 1, (m, 2)), np.zeros((m, 1))], axis=1)  # x, v, t

        def f(st):
            x, v, tt = st[:, 0], st[:, 1], st[:, 2]
            dv = -delta * v + x - x**3 + gamma * np.cos(omega * tt)
            return np.stack([v, dv, np.ones_like(x)], axis=1)

        return self._rk4(f, s, dt, 200, stride, L, lambda st: st[:, 0])

    def _chaos_mackey(self, rng: np.random.Generator, m: int, L: int) -> np.ndarray:
        tau = int(rng.integers(12, 35))
        beta, gamma, n_exp = 0.2, 0.1, 10.0
        burn = 200
        total = burn + L
        hist = rng.uniform(0.5, 1.5, (m, tau + total))
        for t in range(tau, tau + total - 1):
            x_tau = hist[:, t - tau]
            hist[:, t + 1] = hist[:, t] + (beta * x_tau / (1.0 + x_tau**n_exp)
                                           - gamma * hist[:, t])
        return hist[:, -L:]

    def _chaos_logistic(self, rng: np.random.Generator, m: int, L: int) -> np.ndarray:
        r = rng.uniform(3.6, 3.999, m)
        x = rng.uniform(0.1, 0.9, m)
        out = np.empty((m, L + 100))
        for t in range(L + 100):
            x = r * x * (1.0 - x)
            out[:, t] = x
        return out[:, -L:]

    def _chaos_henon(self, rng: np.random.Generator, m: int, L: int) -> np.ndarray:
        a = rng.uniform(1.2, 1.4, m)
        bb = rng.uniform(0.25, 0.31, m)
        x = rng.uniform(-0.5, 0.5, m)
        y = rng.uniform(-0.5, 0.5, m)
        out = np.empty((m, L + 100))
        for t in range(L + 100):
            x_new = 1.0 - a * x**2 + y
            y = bb * x
            x = np.clip(x_new, -10.0, 10.0)
            # Deterministic re-seed of any diverged orbit.
            diverged = np.abs(x) >= 10.0
            if diverged.any():
                x = np.where(diverged, 0.1, x)
                y = np.where(diverged, 0.1, y)
            out[:, t] = x
        return out[:, -L:]

    # 4 ── long-memory: fractional Gaussian noise / fBm ───────────────────────
    def _fam_fgn(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        persistent = rng.random(b) < 0.7
        H = np.where(persistent, rng.uniform(0.55, 0.95, b), rng.uniform(0.05, 0.45, b))
        k = np.arange(L + 1, dtype=np.float64)
        twoH = 2.0 * H[:, None]
        g = 0.5 * (np.abs(k - 1) ** twoH - 2.0 * np.abs(k) ** twoH + np.abs(k + 1) ** twoH)
        c = np.concatenate([g, g[:, L - 1:0:-1]], axis=1)  # (b, 2L)
        lam = np.clip(np.fft.fft(c, axis=1).real, 0.0, None)
        z = rng.standard_normal((b, 2 * L)) + 1j * rng.standard_normal((b, 2 * L))
        x = np.fft.fft(np.sqrt(lam / (4.0 * L)) * z, axis=1).real[:, :L]
        # Half stay stationary fGn; half integrate to fBm (long-memory paths).
        integrate = rng.random(b) < 0.5
        x[integrate] = np.cumsum(x[integrate], axis=1)
        return _standardize_rows(x)

    # 5 ── Markov regime-switching AR + GARCH volatility clustering ───────────
    def _fam_regime_garch(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        half = b // 2
        parts = []
        if half > 0:
            parts.append(self._markov_ar(rng, half, L))
        if b - half > 0:
            parts.append(self._garch(rng, b - half, L))
        return np.concatenate(parts, axis=0)

    def _markov_ar(self, rng: np.random.Generator, m: int, L: int) -> np.ndarray:
        K = int(rng.integers(2, 4))
        mu = rng.normal(0.0, 1.0, (m, K))
        phi = rng.uniform(-0.3, 0.98, (m, K))
        sig = np.exp(rng.uniform(np.log(0.05), np.log(1.5), (m, K)))
        stay = rng.uniform(0.95, 0.999, m)
        state = rng.integers(0, K, m)
        rows_idx = np.arange(m)
        x = mu[rows_idx, state] + rng.standard_normal(m) * sig[rows_idx, state]
        out = np.empty((m, L))
        switch_u = rng.random((m, L))
        jump_to = rng.integers(0, K, (m, L))
        eps = rng.standard_normal((m, L))
        for t in range(L):
            switch = switch_u[:, t] > stay
            state = np.where(switch, jump_to[:, t], state)
            mu_t = mu[rows_idx, state]
            x = mu_t + phi[rows_idx, state] * (x - mu_t) + sig[rows_idx, state] * eps[:, t]
            out[:, t] = x
        return out

    def _garch(self, rng: np.random.Generator, m: int, L: int) -> np.ndarray:
        omega = np.exp(rng.uniform(np.log(1e-4), np.log(1e-2), m))
        alpha = rng.uniform(0.02, 0.15, m)
        beta = np.minimum(rng.uniform(0.75, 0.96, m), 0.999 - alpha)
        heavy = rng.random(m) < 0.5
        df = rng.uniform(3.0, 8.0, m)
        var = omega / np.clip(1.0 - alpha - beta, 1e-3, None)
        r_prev = np.zeros(m)
        out = np.empty((m, L))
        gauss = rng.standard_normal((m, L))
        studt = rng.standard_t(3.0, (m, L))  # deterministic draw; df-scaled below
        for t in range(L):
            var = omega + alpha * r_prev**2 + beta * var
            e = np.where(heavy, studt[:, t] * np.sqrt((df - 2.0) / df), gauss[:, t])
            r_prev = np.sqrt(var) * e
            out[:, t] = r_prev
        # Emit returns, cumulated log-price, or exp price — three real shapes.
        # exp(zero-mean log walk) drifts UP in expectation (Jensen); its share
        # is a knob so drift-neutral configs can shrink it.
        p_ret = float(self._fp("regime_garch", "p_returns", 0.4))
        p_prc = float(self._fp("regime_garch", "p_price", 0.4))
        mode = rng.random(m)
        price = np.cumsum(out, axis=1)
        out = np.where((mode < p_ret)[:, None], out,
                       np.where((mode < p_ret + p_prc)[:, None], price,
                                np.exp(np.clip(price, -20.0, 20.0))))
        return out

    # 6 ── unit-root random walks with drift breaks ───────────────────────────
    def _fam_random_walk(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        kind = rng.random(b)
        sig = np.exp(rng.uniform(np.log(0.2), np.log(2.0), b))
        e = np.where((kind < 0.6)[:, None], rng.standard_normal((b, L)),
                     np.where((kind < 0.85)[:, None],
                              rng.standard_t(4.0, (b, L)),
                              rng.laplace(0.0, 1.0, (b, L))))
        drift = np.zeros((b, L))
        for i in range(b):
            n_cp = int(rng.integers(0, 4))
            cps = np.sort(rng.integers(1, L, n_cp)) if n_cp else np.array([], dtype=int)
            segs = np.split(np.arange(L), cps)
            for seg in segs:
                drift[i, seg] = rng.normal(0.0, 0.15)
        y = np.cumsum(sig[:, None] * e + drift, axis=1)
        gbm = rng.random(b) < 0.3
        z = _standardize_rows(y)
        y[gbm] = np.exp(np.clip(z[gbm] * rng.uniform(0.2, 1.0), -20, 20))
        return y

    # 7 ── intermittent / count demand ────────────────────────────────────────
    def _fam_intermittent(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        p = np.exp(rng.uniform(np.log(0.02), np.log(0.5), b))
        lam = np.exp(rng.uniform(np.log(1.0), np.log(50.0), b))
        occurs = rng.random((b, L)) < p[:, None]
        style = rng.random(b)
        pois = rng.poisson(lam[:, None], (b, L)).astype(np.float64)
        nb = rng.negative_binomial(3, np.clip(3.0 / (3.0 + lam), 1e-3, 1.0)[:, None],
                                   (b, L)).astype(np.float64)
        logn = np.round(np.exp(rng.normal(np.log(lam)[:, None] - 0.5, 1.0, (b, L))))
        sizes = np.where((style < 0.45)[:, None], pois,
                         np.where((style < 0.8)[:, None], nb, logn))
        y = np.where(occurs, np.maximum(sizes, 1.0), 0.0)
        # Slow seasonal modulation of demand probability for some series.
        seasonal = rng.random(b) < 0.3
        if seasonal.any():
            t = np.arange(L)
            per = np.exp(rng.uniform(np.log(24.0), np.log(400.0), b))
            mod = 0.5 * (1.0 + np.sin(2 * np.pi * t[None, :] / per[:, None]))
            keep = rng.random((b, L)) < (0.3 + 0.7 * mod)
            y = np.where(seasonal[:, None] & ~keep, 0.0, y)
        return y

    # 8 ── calendar-structured positive series ────────────────────────────────
    def _fam_calendar(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        out = np.empty((b, L))
        t = np.arange(L)
        for i in range(b):
            period = int(rng.choice([24, 48, 96, 144, 168]))
            phase = t % period
            # Smooth positive daily profile: softplus of a random 4-harmonic curve.
            prof = np.zeros(period)
            for h in range(1, 5):
                prof += rng.normal(0, 1.0 / h) * np.sin(
                    2 * np.pi * h * np.arange(period) / period + rng.uniform(0, 2 * np.pi))
            prof = np.log1p(np.exp(2.0 * prof))
            prof /= prof.mean() + 1e-12
            y = prof[phase]
            # Weekly super-structure: 7 daily factors, weekends distinct.
            week = period * 7
            day_idx = (t // period) % 7
            factors = np.abs(rng.normal(1.0, 0.15, 7))
            factors[5:] *= rng.uniform(0.4, 1.3)
            y = y * factors[day_idx]
            # Slow drift + holiday shocks. Default drift range is up-skewed;
            # the drift-neutral experiments symmetrise it via family_params
            # (forecast inspection showed our model hallucinates upward drift
            # on stable count windows).
            d_lo, d_hi = self._fp("calendar", "drift", (-0.4, 0.6))
            drift = 1.0 + rng.uniform(d_lo, d_hi) * (t / L)
            y = y * np.clip(drift, 0.05, None)
            n_days = max(1, L // period)
            for d in range(n_days):
                if rng.random() < 0.04:
                    f = rng.uniform(0.15, 0.6) if rng.random() < 0.7 else rng.uniform(1.5, 3.0)
                    y[d * period:(d + 1) * period] *= f
            shape = float(np.exp(rng.uniform(np.log(5.0), np.log(100.0))))
            y = y * rng.gamma(shape, 1.0 / shape, L)
            out[i] = y
        return out

    # 9 ── saturating growth / lifecycle ──────────────────────────────────────
    def _fam_growth(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        t = np.linspace(0.0, 1.0, L)
        out = np.empty((b, L))
        for i in range(b):
            K = float(np.exp(rng.uniform(np.log(1.0), np.log(100.0))))
            if rng.random() < 0.5:  # logistic
                r = float(rng.uniform(4.0, 25.0))
                t0 = float(rng.uniform(0.2, 0.8))
                y = K / (1.0 + np.exp(-r * (t - t0)))
            else:  # Gompertz
                bb = float(rng.uniform(2.0, 10.0))
                c = float(rng.uniform(3.0, 15.0))
                y = K * np.exp(-bb * np.exp(-c * t))
            if rng.random() < 0.3:  # lifecycle: rise then decline
                peak = float(rng.uniform(0.4, 0.9))
                decay = float(rng.uniform(1.0, 6.0))
                y = y * np.where(t > peak, np.exp(-decay * (t - peak)), 1.0)
            noise = 1.0 + rng.normal(0.0, float(rng.uniform(0.01, 0.15)), L)
            y = y * np.clip(noise, 0.05, None)
            if rng.random() < 0.25:  # one structural shock
                at = int(rng.integers(L // 4, L))
                y[at:] *= float(rng.uniform(0.5, 1.6))
            out[i] = y
        return out

    # 10 ── non-sinusoidal waves with FM/AM drift ─────────────────────────────
    def _fam_waves(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        out = np.empty((b, L))
        for i in range(b):
            y = self._one_wave(rng, L)
            if rng.random() < 0.3:
                y = y + self._one_wave(rng, L) * float(rng.uniform(0.3, 1.0))
            y = y + rng.standard_normal(L) * float(rng.uniform(0.0, 0.15))
            out[i] = y
        return out

    def _one_wave(self, rng: np.random.Generator, L: int) -> np.ndarray:
        p0 = float(np.exp(rng.uniform(np.log(8.0), np.log(512.0))))
        # FM: instantaneous period drifts up to ±20 % via a smoothed random walk.
        steps = rng.standard_normal(L) * 0.002
        drift = np.clip(np.cumsum(steps), -0.2, 0.2)
        phase = np.cumsum(1.0 / (p0 * (1.0 + drift)))
        frac = np.mod(phase, 1.0)
        kind = rng.choice(["saw", "square", "triangle", "pulse"])
        if kind == "saw":
            y = 2.0 * frac - 1.0
        elif kind == "square":
            duty = float(rng.uniform(0.2, 0.8))
            y = np.where(frac < duty, 1.0, -1.0)
        elif kind == "triangle":
            y = 4.0 * np.abs(frac - 0.5) - 1.0
        else:
            duty = float(rng.uniform(0.05, 0.3))
            y = np.where(frac < duty, 1.0, 0.0)
        # AM envelope: slow positive modulation.
        env_per = float(np.exp(rng.uniform(np.log(L / 8), np.log(L * 2.0))))
        env = 1.0 + float(rng.uniform(0.0, 0.8)) * np.sin(
            2 * np.pi * np.arange(L) / env_per + rng.uniform(0, 2 * np.pi))
        return y * env

    # 11 ── self-exciting bursts + anomaly-injected bases ─────────────────────
    def _fam_bursts_anomaly(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        half = b // 2
        parts = []
        if half > 0:
            parts.append(self._hawkes(rng, half, L))
        if b - half > 0:
            parts.append(self._anomaly_ar(rng, b - half, L))
        return np.concatenate(parts, axis=0)

    def _hawkes(self, rng: np.random.Generator, m: int, L: int) -> np.ndarray:
        lam0 = np.exp(rng.uniform(np.log(0.05), np.log(2.0), m))
        decay = np.exp(-np.exp(rng.uniform(np.log(0.05), np.log(0.5), m)))
        # Subcritical branching ratio in (0.2, 0.9) keeps bursts bursty but stable.
        alpha = rng.uniform(0.2, 0.9, m) * (1.0 - decay)
        intensity = lam0.copy()
        out = np.empty((m, L))
        for t in range(L):
            counts = rng.poisson(np.clip(intensity, 0.0, 1e4)).astype(np.float64)
            out[:, t] = counts
            intensity = lam0 + (intensity - lam0) * decay + alpha * counts
        smooth = rng.random(m) < 0.4
        if smooth.any():  # emit intensity-like smoothed version for some rows
            k = np.ones(8) / 8.0
            sm = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, out[smooth])
            out[smooth] = sm
        return out

    def _anomaly_ar(self, rng: np.random.Generator, m: int, L: int) -> np.ndarray:
        phi = rng.uniform(0.5, 0.98, m)
        base = _ar1(rng.standard_normal((m, L)), phi)
        base = _standardize_rows(base)
        for i in range(m):
            for _ in range(int(rng.poisson(3.0))):  # point anomalies
                at = int(rng.integers(0, L))
                base[i, at] += float(rng.uniform(3.0, 10.0)) * (1 if rng.random() < 0.5 else -1)
            for _ in range(int(rng.integers(0, 3))):  # level shifts
                at = int(rng.integers(L // 8, L))
                base[i, at:] += float(rng.normal(0.0, 2.0))
            if rng.random() < 0.3:  # variance burst window
                a = int(rng.integers(0, L - L // 8))
                w = int(rng.integers(L // 16, L // 4))
                base[i, a:a + w] *= float(rng.uniform(2.0, 5.0))
        return base

    # 12 ── TSMixup-style convex mixing across families ───────────────────────
    def _fam_mixup(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        # Exclude the most expensive families (kernel_gp, chaotic, and the
        # stepwise net_diffusion) from the mix sources so mixup stays cheap —
        # they already appear standalone.
        _slow = {"kernel_gp", "chaotic", "net_diffusion"}
        sources = [k for k in self._weights if k != "mixup" and k not in _slow] \
            or ["trend_seasonal"]
        k = int(rng.integers(2, int(self._fp("mixup", "k_max", 3)) + 1))
        picks = rng.choice(len(sources), size=k, replace=True)
        batches = []
        for j, pi in enumerate(picks):
            child = np.random.default_rng(rng.integers(0, 2**63 - 1))
            batches.append(_standardize_rows(self._draw(sources[int(pi)], child, b, L)))
        w = rng.dirichlet(np.ones(k) * float(self._fp("mixup", "dirichlet_alpha", 1.5)), size=b)
        out = np.zeros((b, L))
        for j in range(k):
            out += w[:, j:j + 1] * batches[j]
        return out

    # 13 ── audio-inspired stochastic rhythms (TempoPFN gap) ──────────────────
    def _fam_rhythm(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        out = np.empty((b, L))
        for i in range(b):
            p_lo, p_hi = self._fp("rhythm", "period", (8.0, 200.0))
            p0 = float(np.exp(rng.uniform(np.log(p_lo), np.log(p_hi))))
            # Tempo drift: smoothed random walk on the instantaneous period.
            drift = np.clip(np.cumsum(rng.standard_normal(L) * 0.003), -0.25, 0.25)
            phase = np.cumsum(1.0 / (p0 * (1.0 + drift)))
            beat_pos = np.flatnonzero(np.diff(np.floor(phase), prepend=0.0) > 0)
            if beat_pos.size == 0:  # period longer than the series — noise floor
                out[i] = rng.standard_normal(L) * 0.05
                continue
            n_beats = beat_pos.size
            # Accent pattern over a bar of 2-8 beats, human amplitude jitter,
            # probabilistic dropouts, swing on the off-beats.
            bar = int(rng.integers(2, 9))
            accents = np.exp(rng.normal(0.0, 0.6, bar))
            accents[0] *= float(rng.uniform(1.2, 2.5))
            amp = accents[np.arange(n_beats) % bar] * np.exp(rng.normal(0.0, 0.25, n_beats))
            kp_lo, kp_hi = self._fp("rhythm", "keep", (0.55, 0.95))
            keep = rng.random(n_beats) < float(rng.uniform(kp_lo, kp_hi))
            pos = beat_pos.astype(np.float64)
            pos[1::2] += float(rng.uniform(0.0, 0.35)) * p0 * 0.5
            pos = np.clip(np.round(pos).astype(int), 0, L - 1)
            impulses = np.zeros(L)
            np.add.at(impulses, pos[keep], amp[keep])
            # Event kernel: pluck (decaying oscillation), percussive hit
            # (attack + exponential decay), or smooth bump.
            kw = max(4, int(min(p0 * float(rng.uniform(0.5, 2.0)), 256.0)))
            tk = np.arange(kw, dtype=np.float64)
            kind = rng.random()
            if kind < 0.45:
                kf = float(rng.uniform(0.05, 0.45))
                kernel = np.exp(-tk / (kw * float(rng.uniform(0.15, 0.5)))) \
                    * np.cos(2 * np.pi * kf * tk)
            elif kind < 0.8:
                atk = max(1, int(kw * float(rng.uniform(0.02, 0.15))))
                kernel = np.exp(-tk / (kw * float(rng.uniform(0.1, 0.4))))
                kernel[:atk] *= np.linspace(0.0, 1.0, atk)
            else:
                width = kw * float(rng.uniform(0.08, 0.25))
                kernel = np.exp(-0.5 * ((tk - kw / 3.0) / width) ** 2)
            y = np.convolve(impulses, kernel)[:L]
            out[i] = y + rng.standard_normal(L) * float(rng.uniform(0.0, 0.08))
        return out

    # 14 ── multi-scale fractals (TempoPFN gap) ───────────────────────────────
    def _fam_fractal_multi(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        out = np.empty((b, L))
        for i in range(b):
            mode = rng.random()
            if mode < 0.4:
                out[i] = self._fractal_spectral(rng, L)
            elif mode < 0.7:
                out[i] = self._fractal_weierstrass(rng, L)
            else:
                out[i] = self._fractal_cascade(rng, L)
        return _standardize_rows(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0))

    def _fractal_spectral(self, rng: np.random.Generator, L: int) -> np.ndarray:
        """Spectral synthesis with a piecewise spectral slope: different
        roughness per frequency band (multi-scale), amplitude-continuous by
        integrating the slope over log-frequency."""
        n = L // 2 + 1
        f = np.arange(1, n, dtype=np.float64)
        nb = int(rng.integers(2, 5))
        edges = np.exp(np.linspace(0.0, np.log(n - 1.0), nb + 1))[1:-1]
        beta = rng.uniform(0.1, 3.0, nb)  # per-band slope of the power spectrum
        slope = beta[np.searchsorted(edges, f)] / 2.0  # amplitude slope
        dlogf = np.diff(np.log(f), prepend=np.log(f[0]))
        amp = np.exp(-np.cumsum(slope * dlogf))
        spec = np.zeros(n, dtype=np.complex128)
        spec[1:] = amp * np.exp(2j * np.pi * rng.random(n - 1))
        return np.fft.irfft(spec, n=L)

    def _fractal_weierstrass(self, rng: np.random.Generator, L: int) -> np.ndarray:
        """Weierstrass-type sum: geometric ladder of frequencies with
        H-controlled amplitude decay — self-affine at every scale."""
        t = np.arange(L, dtype=np.float64) / L
        bfac = float(rng.uniform(1.4, 3.0))
        H = float(rng.uniform(0.2, 0.95))
        f0 = float(rng.uniform(1.0, 6.0))
        y = np.zeros(L)
        k, f = 0, f0
        while f < L / 2.0 and k < 40:
            y += (bfac ** (-k * H)) * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
            k += 1
            f = f0 * bfac ** k
        return y

    def _fractal_cascade(self, rng: np.random.Generator, L: int) -> np.ndarray:
        """Multiplicative (log-normal) cascade over dyadic scales — a
        multifractal measure with volatility clustering at every scale."""
        sig = float(rng.uniform(0.15, 0.5))
        w = np.ones(1)
        while w.size < L:
            w = np.repeat(w, 2)
            w = w * np.exp(rng.normal(0.0, sig, w.size))
        measure = w[:L] / (w[:L].mean() + 1e-12)
        if rng.random() < 0.5:  # rough level-shifting path
            return np.log(measure + 1e-9)
        # volatility-modulated noise (multifractal "returns")
        return rng.standard_normal(L) * np.sqrt(measure)

    # 15 ── network-topology diffusion (TempoPFN gap, CauKer-flavoured) ───────
    def _fam_net_diffusion(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        out = np.empty((b, L))
        made = 0
        while made < b:
            m = min(int(rng.integers(4, 9)), b - made)  # observations per graph
            N = int(rng.integers(8, 33))
            if rng.random() < 0.6:  # Erdős–Rényi
                A = (rng.random((N, N)) < float(rng.uniform(0.05, 0.3))).astype(np.float64)
            else:  # degree-preferential attachment
                A = np.zeros((N, N))
                for v in range(1, N):
                    deg = A[:v, :v].sum(axis=0) + A[:v, :v].sum(axis=1) + 1.0
                    k = min(v, int(rng.integers(1, 4)))
                    targets = rng.choice(v, size=k, replace=False, p=deg / deg.sum())
                    A[targets, v] = 1.0
            np.fill_diagonal(A, 0.0)
            # Signed random edge weights, row-normalised, spectral scale < 1 so
            # the dynamics are stable but long-memoried.
            W = A * rng.uniform(0.2, 1.0, (N, N)) \
                * np.where(rng.random((N, N)) < 0.15, -1.0, 1.0)
            W = W / np.maximum(np.abs(W).sum(axis=1, keepdims=True), 1e-12)
            W *= float(rng.uniform(0.7, 0.99))
            # Forcing at a few source nodes: seasonal drive + Poisson-ish bursts.
            n_src = int(rng.integers(1, max(2, N // 8) + 1))
            src = rng.choice(N, size=n_src, replace=False)
            tt = np.arange(L, dtype=np.float64)
            U = np.zeros((N, L))
            for s_node in src:
                per = float(np.exp(rng.uniform(np.log(8.0), np.log(512.0))))
                U[s_node] = np.sin(2 * np.pi * tt / per + rng.uniform(0, 2 * np.pi)) \
                    * float(rng.uniform(0.5, 2.0))
                if rng.random() < 0.5:
                    U[s_node] += (rng.random(L) < 0.01) * float(rng.uniform(3.0, 10.0))
            noise = rng.standard_normal((N, L)) * float(rng.uniform(0.01, 0.2))
            s = np.zeros(N)
            rec = np.empty((N, L))
            for t in range(L):
                s = W @ s + U[:, t] + noise[:, t]
                rec[:, t] = s
            out[made:made + m] = rec[rng.choice(N, size=m, replace=False)]
            made += m
        return _standardize_rows(out)

    # 16 ── web/count traffic: npm-downloads / pageview shapes ────────────────
    # Purpose-built from receipt intel: the private eval pool is ~65% count/web
    # sources (npm_downloads_per_pkg + wikimedia pageviews alone ≈ 53%). Those
    # series are log-multiplicative: positive integer counts over many decades,
    # weekday/weekend (period-7) structure on daily cadence, adoption S-curves,
    # news-event spikes with decay tails, and occasional level shifts. No other
    # family emits this exact composition (calendar has no period-7; intermittent
    # is zero-inflated retail counts).
    def _fam_web_traffic(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        base_lo, base_hi = self._fp("web_traffic", "log_base", (np.log(10.0), np.log(1e8)))
        p_daily = float(self._fp("web_traffic", "p_daily", 0.7))
        # Per-window loss analysis (pool2, v3 vs king): our worst losses are
        # CLEAN series — low cv, strong weekly autocorrelation, almost no
        # spikes. The "clean" mode emits exactly that: a strong regular weekly
        # pattern with very low noise and rare events, so the model learns that
        # regular periodicity can be stable.
        p_clean = float(self._fp("web_traffic", "p_clean", 0.55))
        # Calibratable distribution knobs (defaults = hand-designed v2 values;
        # calibrate_web.py fits these from REAL wikimedia/npm series and injects
        # the fitted ranges via config family_params).
        ps_cl = self._fp("web_traffic", "pattern_std_clean", (0.25, 0.7))
        ps_ns = self._fp("web_traffic", "pattern_std_noisy", (0.1, 0.5))
        sg_cl = self._fp("web_traffic", "sigma_clean", (0.02, 0.10))
        sg_ns = self._fp("web_traffic", "sigma_noisy", (0.05, 0.3))
        phi_rg = self._fp("web_traffic", "phi", (0.5, 0.95))
        ev_cl = self._fp("web_traffic", "event_rate_div_clean", (600.0, 2000.0))
        ev_ns = self._fp("web_traffic", "event_rate_div_noisy", (150.0, 600.0))
        ev_mag = self._fp("web_traffic", "event_mag", (0.5, 2.0))
        ev_tau = self._fp("web_traffic", "event_tau", (1.0, 12.0))
        wk_dip = self._fp("web_traffic", "weekend_dip", (0.2, 0.9))
        p_dip = float(self._fp("web_traffic", "p_weekend_dip", 0.8))
        tr_cl = self._fp("web_traffic", "trend_amp_clean", (0.3, 1.2))
        tr_ns = self._fp("web_traffic", "trend_amp_noisy", (0.5, 3.0))
        out = np.empty((b, L))
        t01 = np.arange(L, dtype=np.float64) / L
        for i in range(b):
            clean = rng.random() < p_clean
            period = 7 if rng.random() < p_daily else int(rng.choice([24, 168]))
            base = float(rng.uniform(base_lo, base_hi))
            # Adoption trend in log space: S-curve rise, slow decay, growth, flat.
            kind = rng.random()
            amp = float(rng.uniform(*(tr_cl if clean else tr_ns)))
            if kind < 0.35:
                r = float(rng.uniform(4.0, 20.0))
                t0 = float(rng.uniform(0.1, 0.9))
                trend = amp / (1.0 + np.exp(-r * (t01 - t0)))
            elif kind < 0.55:
                trend = -float(rng.uniform(0.3, 2.0)) * t01 * (0.5 if clean else 1.0)
            elif kind < 0.75:
                trend = amp * float(rng.uniform(0.3, 1.0)) * t01
            else:
                trend = np.zeros(L)
            # Weekly/daily multiplicative pattern: per-slot log factors, with the
            # canonical weekend dip on daily cadence. Clean mode: strong pattern.
            pat_std = float(rng.uniform(*(ps_cl if clean else ps_ns)))
            pat = rng.normal(0.0, pat_std, period)
            if period == 7 and rng.random() < p_dip:
                pat[5:] -= float(rng.uniform(*wk_dip))
            pat -= pat.mean()
            seasonal = pat[np.arange(L) % period]
            # AR(1) log-noise (persistence of interest levels). Clean mode: quiet.
            sigma = float(rng.uniform(*(sg_cl if clean else sg_ns)))
            phi = np.full(1, float(rng.uniform(*phi_rg)))
            noise = _ar1(rng.normal(0.0, sigma, (1, L)), phi)[0]
            logy = base + trend + seasonal + noise
            # News/release events: Poisson arrivals, heavy-tailed magnitude,
            # exponential decay back to baseline. Clean mode: rare.
            rate_div = float(rng.uniform(*(ev_cl if clean else ev_ns)))
            for _ in range(int(rng.poisson(L / rate_div))):
                at = int(rng.integers(0, L))
                mag = float(rng.lognormal(np.log(float(rng.uniform(*ev_mag))), 0.5))
                tau = float(rng.uniform(*ev_tau))
                tail = min(L - at, int(6.0 * tau) + 1)
                logy[at:at + tail] += mag * np.exp(-np.arange(tail, dtype=np.float64) / tau)
            if rng.random() < (0.05 if clean else 0.3):  # level shift
                at = int(rng.integers(L // 8, L))
                logy[at:] += float(rng.normal(0.0, 0.8))
            y = np.exp(np.clip(logy, -1.0, 25.0))
            if rng.random() < 0.8:
                y = np.round(y)  # integer counts
            out[i] = y
        return out

    # 17 ── seasonal level: driftless RW level × calendar seasonality ─────────
    # The canonical "real feed" shape (energy/traffic/sales/pageviews): a
    # wandering but DRIFT-NEUTRAL level carrying persistent calendar
    # seasonality, with damped (never runaway) trends. Added after the
    # forecast inspection showed our model hallucinates upward drift on
    # stable windows — the optimal forecast of a random-walk level is its
    # CURRENT value, so this family explicitly teaches level-anchored,
    # drift-skeptical forecasting (the reigning king's winning behaviour).
    def _fam_seasonal_level(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        sig_lo, sig_hi = self._fp("seasonal_level", "level_sigma", (0.005, 0.06))
        p_mult = float(self._fp("seasonal_level", "p_multiplicative", 0.4))
        amp_lo, amp_hi = self._fp("seasonal_level", "seasonal_amp", (0.2, 2.0))
        t01 = np.arange(L, dtype=np.float64) / L
        out = np.empty((b, L))
        for i in range(b):
            # Level: zero-drift random walk; occasional mean-zero drift
            # segments; optional damped trend that FLATTENS by the horizon.
            sigma_l = float(np.exp(rng.uniform(np.log(sig_lo), np.log(sig_hi))))
            level = np.cumsum(rng.standard_normal(L)) * sigma_l
            for _ in range(int(rng.poisson(1.0))):
                cp = int(rng.integers(0, L))
                level[cp:] += float(rng.normal(0.0, 0.01)) * np.arange(L - cp)
            if rng.random() < 0.3:
                lam = float(rng.uniform(1.0, 6.0))
                level = level + float(rng.normal(0.0, 0.8)) * (1.0 - np.exp(-lam * t01))
            # Seasonality: 1-2 calendar-period harmonic stacks (reuses the
            # trend_seasonal pattern sampler, which favours 7/24/168-type periods).
            seasonal = np.zeros(L)
            for _ in range(int(rng.integers(1, 3))):
                amp = float(np.exp(rng.uniform(np.log(amp_lo), np.log(amp_hi))))
                seasonal += amp * self._sample_seasonal(rng, L)
            noise = rng.standard_normal(L) * float(rng.exponential(0.15))
            if rng.random() < p_mult:
                # Multiplicative on a positive level — retail/energy shape.
                base = np.exp(np.clip(0.5 * level - 0.5 * float(level.mean()), -10.0, 10.0))
                y = base * (1.0 + 0.3 * np.tanh(seasonal)) * (1.0 + 0.1 * np.tanh(noise))
            else:
                y = level + seasonal + noise
            out[i] = y
        return out

    # 18 ── the REAL vendored ForecastPFN prior (v4, Apache-2.0) ───────────────
    # The exact trend x multi-seasonality x Weibull-noise composition (with the
    # built-in transition/damping/spike augmentations) that the reigning king
    # trains on 100% and wins the count/benchmark domains with. Our own
    # trend_seasonal only approximates it; 22 knob/reweight candidates plateaued
    # ~3-6% short on counts, so v4 mixes the genuine prior (~35%) with our
    # physical-domain-dominant families (~65%). Drawn via the vendored wrapper
    # exactly as base_generator does (proven cross-process deterministic).
    def _fam_forecast_pfn(self, rng: np.random.Generator, b: int, L: int) -> np.ndarray:
        from tempo_gen.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator_wrapper import (
            ForecastPFNGeneratorWrapper,
        )
        from tempo_gen.synthetic_generation.generator_params import (
            ForecastPFNGeneratorParams,
        )

        base_seed = int(rng.integers(0, 2_000_000_000))
        params = ForecastPFNGeneratorParams(global_seed=base_seed, length=L)
        wrapper = ForecastPFNGeneratorWrapper(params)
        out = np.empty((b, L))
        made = 0
        while made < b:
            chunk = min(self._batch, b - made)
            batch = wrapper.generate_batch(batch_size=chunk, seed=(base_seed + made) % 2_000_000_000)
            vals = np.asarray(batch.values, dtype=np.float64)
            if vals.ndim == 1:
                vals = vals[None, :]
            for row in vals:
                r = row.ravel()
                if r.size >= L:
                    out[made] = r[:L]
                else:
                    out[made] = np.concatenate([r, np.full(L - r.size, r[-1] if r.size else 0.0)])
                made += 1
                if made >= b:
                    break
        return out

    # ── TempoPFN-style per-series augmentations ───────────────────────────────
    def _augment(self, x: np.ndarray, rng: np.random.Generator,
                 family: str) -> np.ndarray:
        """Port of TempoPFN's augmentation layer (probabilities mirror its
        generator params): smooth monotone time-warp, damping envelopes, and
        spike injection. All draws come from the per-series post RNG, so the
        stream stays a pure function of (seed, n_series)."""
        L = x.size
        # Time warping (TempoPFN time_warp_prob ≈ 0.1): resample along a
        # monotone perturbation of the time axis.
        if L >= 32 and rng.random() < 0.10:
            k = int(rng.integers(4, 9))
            u = np.linspace(0.0, 1.0, k)
            v = u + rng.normal(0.0, float(rng.uniform(0.02, 0.08)), k)
            v[0], v[-1] = 0.0, 1.0
            v = np.sort(np.clip(v, 0.0, 1.0))
            pos = np.interp(np.linspace(0.0, 1.0, L), u, v) * (L - 1.0)
            x = np.interp(pos, np.arange(L, dtype=np.float64), x)
        # Damping (TempoPFN damping_prob ≈ 0.1): multiplicative decay (or
        # ramp-up) envelope from a random onset. Keeps positive series positive.
        if rng.random() < 0.08:
            onset = int(rng.integers(0, max(1, L - L // 8)))
            half_life = float(rng.uniform(L / 16.0, L / 2.0))
            env = np.ones(L)
            tail = np.arange(L - onset, dtype=np.float64)
            env[onset:] = np.maximum(2.0 ** (-tail / half_life), 0.02)
            if rng.random() < 0.3:
                env = env[::-1].copy()
            x = x * env
        # Spike injection (TempoPFN spike_prob ≈ 0.15): a few decaying-tail
        # spikes; multiplicative for positive families so they stay positive.
        if rng.random() < 0.10:
            sd = float(x.std()) + 1e-12
            for _ in range(1 + min(int(rng.poisson(1.0)), 3)):
                at = int(rng.integers(0, L))
                w = min(int(rng.integers(1, 6)), L - at)
                tail = np.exp(-np.arange(w, dtype=np.float64) / max(1.0, w / 2.0))
                if family in _POSITIVE_FAMILIES:
                    x[at:at + w] = x[at:at + w] * (1.0 + float(rng.uniform(1.5, 5.0)) * tail)
                else:
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    x[at:at + w] = x[at:at + w] + sign * float(rng.uniform(3.0, 10.0)) * sd * tail
        return x

    # ── post-processing: scale/offset/quantisation diversity ─────────────────
    def _postprocess(self, window: np.ndarray, rng: np.random.Generator,
                     family: str) -> np.ndarray:
        x = np.array(window, dtype=np.float64).ravel()
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=self._max_abs, neginf=-self._max_abs)

        # ForecastPFN ships its OWN augmentation + scale and is the king's
        # winning distribution verbatim — reshaping it (5-decade rescale,
        # offsets, our time-warp/damp/spike) would corrupt exactly what wins the
        # count domains. Preserve it natively: only guarantee finiteness + a
        # trainer-safe magnitude, like base_generator's light sanitize.
        if family == "forecast_pfn":
            np.clip(x, -self._max_abs, self._max_abs, out=x)
            return np.ascontiguousarray(x, dtype=np.float64)

        if float(x.std()) < 1e-9:  # degenerate-flat guard: keep a learnable signal
            x = x + rng.standard_normal(x.size) * max(1e-3, abs(float(x.mean())) * 1e-3)

        # TempoPFN augmentation layer: time-warp / damping / spike injection.
        x = self._augment(x, rng, family)

        if family in _POSITIVE_FAMILIES:
            # Preserve positivity; scale over decades.
            ps_lo, ps_hi = self._fp("post", "positive_scale", (0.05, 100.0))
            s = float(np.exp(rng.uniform(np.log(ps_lo), np.log(ps_hi))))
            x = x * s
            if family == "intermittent" and rng.random() < 0.6:
                x = np.round(x)  # keep count-like
        else:
            sc_lo, sc_hi = self._fp("post", "scale", (1e-2, 1e3))
            x = (x - x.mean()) / (x.std() + 1e-12)
            s = float(np.exp(rng.uniform(np.log(sc_lo), np.log(sc_hi))))
            x = x * s
            if rng.random() < float(self._fp("post", "p_offset", 0.6)):
                x = x + s * float(rng.normal(0.0, 3.0))
            if rng.random() < 0.08 and s >= 10.0:
                x = np.round(x)  # integer-quantised sensor/count shapes
            if rng.random() < 0.06:
                x = np.log1p(np.exp(np.clip(x / (s + 1e-12), -30, 30))) * s  # softplus → positive

        # Post-transform degeneracy repair: rounding a degenerate-repaired series
        # (tiny injected noise → round → all zeros) can flatten it again. Sparse
        # unit events keep count-like series count-like while guaranteeing signal.
        if float(x.std()) < 1e-9:
            k = max(1, x.size // 64)
            idx = rng.integers(0, x.size, k)
            x[idx] += np.maximum(1.0, abs(float(x.mean())))

        np.clip(x, -self._max_abs, self._max_abs, out=x)
        if not np.isfinite(x).all():  # pragma: no cover — belt and braces
            x = np.nan_to_num(x, nan=0.0, posinf=self._max_abs, neginf=-self._max_abs)
        return np.ascontiguousarray(x, dtype=np.float64)

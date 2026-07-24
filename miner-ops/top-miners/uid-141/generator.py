"""cascade-fullctx-research-v18 — v16 plus weekly non-negative demand.

This is the artifact a cascade miner actually competes with: a subclass of
``cascade.interface.DataGenerator`` that turns a single integer ``seed`` into a
corpus of univariate float series. The subnet holds the model, seeds, and
compute budget byte-identical between the king and every challenger, so the
*only* thing that moves the forecast score is the distribution this file emits.
The competitive lever is therefore **prior diversity + realism**: a corpus that
covers more of the shapes a real forecaster must handle (trend, multi-seasonal,
regime shifts, integrated/near-unit-root dynamics, smooth GP-like curves,
nonlinear/chaotic recurrences, mean-reverting stochastic volatility,
intermittent demand, event recovery, and measurement artifacts) trains a
stronger zero-shot model than the reference generator's trend+seasonal+AR(1)
mix.

Design constraints this file respects (all from the contract in
``cascade.interface``):

* **Determinism is load-bearing.** Every value is drawn from one
  ``np.random.default_rng(seed)`` in a fixed draw order, so two runs at the same
  seed produce byte-identical corpora — the property ``cascade verify`` audits
  by building the corpus twice and comparing digests.
* **Code-only.** No shipped weights, no network, no clock, no un-seeded RNG.
  Imports stay on the dependency allowlist (NumPy/SciPy only) and clear of the
  static-guard blocklist.
* **Bounded + finite.** Each series is 1-D ``(L,)`` float64, length in
  ``[min_length, max_length]``, finite (no NaN/inf). ``_sanitize`` is the last
  gate so a numerically unlucky draw can never poison a training run.

Everything is **vectorised per family** (a batched time-axis recurrence, never a
per-series Python loop over time). Compared with custom-fullctx-v4, the slow
random-Fourier GP is replaced by FFT spectral sampling, a long-memory spectral
family is added, and larger chunks amortise dispatch while staying far below the
sandbox memory limit.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import lru_cache, partial
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread

import numpy as np
from scipy.signal import lfilter
from scipy.special import gammaln

from cascade.interface import DataGenerator

# Series generated per vectorised batch. Bounds peak memory to O(_CHUNK · max_len)
# so streaming feed modes (which request millions of series and stop early) never
# materialise the full corpus. Prefetching holds at most two completed chunks
# (current + queued) while the producer may build the next. The base block is
# 2048 × 4096 × 8 B = 64 MiB per base family block, plus temporary arrays.
# This remains comfortably below the 4 GiB sandbox cap. On the reference local
# A100 environment, 2048 rows generated ~6% more points/s than 1024 while 4096
# regressed slightly, so 2048 is the measured throughput sweet spot.
_CHUNK = 2048

# Multi-cadence seasonal bank. Full 4096-point contexts can identify several
# cycles even at 365/672/730-step periods, unlike short-crop generators.
_SEASONAL_PERIODS = np.array(
    [4, 7, 12, 24, 30, 48, 52, 90, 96, 144, 168, 183, 288, 336, 365, 672, 730],
    dtype=np.float64,
)
_SEASONAL_PROBS = np.array(
    [0.04, 0.12, 0.04, 0.16, 0.05, 0.06, 0.04, 0.03, 0.07, 0.03,
     0.13, 0.04, 0.04, 0.06, 0.07, 0.04, 0.05],
    dtype=np.float64,
)
_SEASONAL_PROBS /= _SEASONAL_PROBS.sum()

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
    "spectral_gp",         # smooth GP-like paths via batched FFT sampling
    "long_memory",         # persistent/anti-persistent power-law spectra
    "ou_stochastic_vol",   # mean-reverting regimes + clustered/heavy-tailed volatility
    "physical_sensors",    # bounded/skewed/smooth physical measurement archetypes
    "seasonal_counts",     # seasonal Poisson/NB web and demand counts with bursts
    "intermittent",        # zero-inflated / intermittent demand
    "pulse_outlier",       # sharp/decaying events, outliers, and true flat runs
    "weekly_demand",       # period-7 retail/web demand with promotions and dips
)
# Dynamics-heavy composition selected by a controlled local A/B: it beat the
# prior baseline on all three validation seeds (mean geomean 0.18431 vs
# 0.19097). Config may still override these defaults, but copying generator.py
# without config now retains the measured mixture.
_DEFAULT_WEIGHTS: dict[str, float] = {
    # Preserve v16's relative composition at 92% and reserve the externally
    # validated sweet-spot weight for dedicated weekly demand.
    "trend_seasonal_ar": 0.1104,
    "regime_shift": 0.1104,
    "multiplicative": 0.0736,
    "ar2": 0.1380,
    "integrated": 0.1104,
    "threshold_ar": 0.0736,
    "chaotic": 0.0368,
    "spectral_gp": 0.0644,
    "long_memory": 0.0552,
    "ou_stochastic_vol": 0.0920,
    "physical_sensors": 0.0184,
    "seasonal_counts": 0.0184,
    "intermittent": 0.0092,
    "pulse_outlier": 0.0092,
    "weekly_demand": 0.08,
}


class Generator(DataGenerator):
    """A mixture-of-priors generator. Submit as ``generator.Generator``."""

    def __init__(self, config_dir: str, *, seed: int) -> None:
        cfg_path = Path(config_dir) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        self._cfg = cfg
        self._seed = int(seed)
        self._min_len = int(cfg.get("min_length", 64))
        self._max_len = int(cfg.get("max_length", 4096))  # = [training] context_length (train on full context)
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
        # v3.9 length-NORMALIZED bimodal trend knobs (trend excursion is length-invariant;
        # real trend-strength is ~0.02 and length-invariant, but v2's slope*t grows with L).
        self._tr_hi_frac = float(cfg.get("tr_hi_frac", 0.25))
        self._tr_exc_lo = float(cfg.get("tr_exc_lo", 0.4))
        self._tr_exc_hi = float(cfg.get("tr_exc_hi", 3.0))
        self._gr_exc_lo = float(cfg.get("gr_exc_lo", 0.3))
        self._gr_exc_hi = float(cfg.get("gr_exc_hi", 2.0))
        self._sa_clean_frac = float(cfg.get("sa_clean_frac", 0.4))
        self._sa_clean_lo = float(cfg.get("sa_clean_lo", 0.02))
        self._sa_clean_hi = float(cfg.get("sa_clean_hi", 0.12))

    @property
    def name(self) -> str:
        return str(self._cfg.get("name", "cascade-fullctx-research-v18"))

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
        # Bind the trend-excursion knobs as explicit builder arguments (no shared
        # module state) so the corpus is a pure function of (seed, config).
        builders = (
            partial(_trend_seasonal_ar, hi_frac=self._tr_hi_frac,
                    exc_lo=self._tr_exc_lo, exc_hi=self._tr_exc_hi,
                    clean_frac=self._sa_clean_frac,
                    clean_lo=self._sa_clean_lo, clean_hi=self._sa_clean_hi),
            _regime_shift,
            partial(_multiplicative, hi_frac=self._tr_hi_frac,
                    exc_lo=self._gr_exc_lo, exc_hi=self._gr_exc_hi),
            _ar2, _integrated, _threshold_ar, _chaotic, _spectral_gp,
            _long_memory, _ou_stochastic_vol, _physical_sensors,
            _seasonal_counts, _intermittent, _pulse_outlier, _weekly_demand,
        )
        # Generate one chunk ahead on a CPU thread while the consumer trains on
        # the current chunk. The isolation benchmark measured 21.9% of training
        # wall blocked in next(); a one-slot queue overlaps NumPy/SciPy work
        # (which releases the GIL) without changing the RNG owner or draw order.
        queue: Queue[object] = Queue(maxsize=1)
        stop = Event()
        done = object()

        def put(item: object) -> bool:
            while not stop.is_set():
                try:
                    queue.put(item, timeout=0.1)
                    return True
                except Full:
                    continue
            return False

        def produce() -> None:
            try:
                produced = 0
                while produced < n_series and not stop.is_set():
                    # Always draw a FULL _CHUNK (yielding only what's still
                    # needed), so series i remains a pure function of (seed, i).
                    lengths = rng.integers(
                        self._min_len, max_len + 1, size=_CHUNK
                    )
                    fam_ids = rng.choice(
                        len(_FAMILIES), size=_CHUNK, p=self._weights
                    )
                    chunk: list[np.ndarray | None] = [None] * _CHUNK
                    for fam in range(len(_FAMILIES)):
                        idx = np.nonzero(fam_ids == fam)[0]
                        if idx.size == 0:
                            continue
                        block = builders[fam](rng, int(idx.size), max_len)
                        # Preserve positivity for count/magnitude families and
                        # exact integer/causal structure for count processes.
                        preserve_nonnegative = fam in (2, 10, 11, 12, 14)
                        preserve_integers = fam in (11, 12)
                        block = _sanitize(
                            _measurement_artifacts(
                                rng,
                                block,
                                preserve_nonnegative=preserve_nonnegative,
                                preserve_integers=preserve_integers,
                                # Reverse only families whose laws remain valid
                                # under time reversal. Reversing causal regimes,
                                # SETAR maps, OU recovery, or shock paths creates
                                # anti-causal precursors absent from the model.
                                allow_reverse=fam in (0, 2, 7, 8),
                                # A prefix-calibrated hard sensor bound turns an
                                # unbounded random walk into an absorbing flat
                                # line. Keep range artifacts on bounded families,
                                # but preserve integrated dynamics.
                                allow_range_artifacts=fam != 4,
                            )
                        )
                        for row, series_i in enumerate(idx):
                            length = int(lengths[series_i])
                            chunk[series_i] = np.ascontiguousarray(
                                block[row, :length], dtype=np.float64
                            )
                    take = min(_CHUNK, n_series - produced)
                    if not put((chunk, take)):
                        return
                    produced += take
            except BaseException as exc:  # propagate producer failures
                put(exc)
            finally:
                put(done)

        producer = Thread(target=produce, name="cascade-generator", daemon=True)
        producer.start()
        try:
            while True:
                item = queue.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                chunk, take = item
                for arr in chunk[:take]:
                    # fam_ids partitions [0, _CHUNK); fail loud if that changes.
                    if arr is None:  # pragma: no cover - defensive
                        raise RuntimeError("internal: unfilled series slot")
                    yield arr
        finally:
            stop.set()
            producer.join(timeout=1.0)


# ── shared vectorised primitives ────────────────────────────────────────────


def _ar1_batch(innov: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """AR(1) filter applied along the time axis of a (n, L) innovation block.

    ``x[:, t] = phi * x[:, t-1] + innov[:, t]``. The loop is over time (L
    iterations, vectorised across the batch), never over the n series.
    """
    n, L = innov.shape
    x = np.empty((n, L), dtype=np.float64)
    p = phi.reshape(n)
    for i in range(n):
        x[i] = lfilter([1.0], [1.0, -float(p[i])], innov[i])
    return x


def _ar2_batch(innov: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """AR(2) filter: ``x_t = a1 x_{t-1} + a2 x_{t-2} + e_t`` (batched over n)."""
    n, L = innov.shape
    x = np.empty((n, L), dtype=np.float64)
    for i in range(n):
        x[i] = lfilter(
            [1.0], [1.0, -float(a1[i]), -float(a2[i])], innov[i]
        )
    return x


def _prefix_mean_std(
    x: np.ndarray, *, calibration_points: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Location/scale estimated from an initial calibration prefix only.

    Whole-path normalization makes an emitted prefix depend on unseen future
    values and leaks the evaluation target into the synthetic process. A fixed
    early calibration interval keeps subsequent transformations causal.
    """
    prefix = x[:, : min(x.shape[1], calibration_points)]
    mean = prefix.mean(axis=1, keepdims=True)
    std = prefix.std(axis=1, keepdims=True)
    return mean, np.where(std < 1e-12, 1.0, std)


def _prefix_standardize(
    x: np.ndarray, *, center: bool = True, calibration_points: int = 512
) -> np.ndarray:
    mean, std = _prefix_mean_std(
        x, calibration_points=calibration_points
    )
    return (x - mean) / std if center else x / std


@lru_cache(maxsize=4)
def _seasonal_basis(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached unit sine/cosine waves for the fixed cadence bank."""
    angle = (
        2.0
        * np.pi
        * np.arange(L, dtype=np.float64)[None, :]
        / _SEASONAL_PERIODS[:, None]
    )
    return np.sin(angle), np.cos(angle)


def _seasonal(rng: np.random.Generator, n: int, L: int, k_max: int = 3) -> np.ndarray:
    """Sum of 1..k_max stationary or slowly modulated seasonal components."""
    t = np.arange(L, dtype=np.float64)[None, :]
    sin_basis, cos_basis = _seasonal_basis(L)
    k = rng.integers(1, k_max + 1, size=n)
    out = np.zeros((n, L), dtype=np.float64)
    for j in range(k_max):
        active = np.nonzero(k > j)[0]
        per = rng.choice(_SEASONAL_PERIODS, size=n, p=_SEASONAL_PROBS)[:, None]
        amp = rng.uniform(0.2, 2.0, size=n)[:, None]
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n)[:, None]
        # Draw parameters for every row to preserve the fixed RNG sequence, but
        # evaluate only active rows. Stationary components reuse the cadence
        # bank via sin(a+b), avoiding a fresh transcendental pass over n×L.
        basis_idx = np.searchsorted(_SEASONAL_PERIODS, per[active, 0])
        component = amp[active] * (
            sin_basis[basis_idx] * np.cos(phase[active])
            + cos_basis[basis_idx] * np.sin(phase[active])
        )
        # Real seasonal strength and timing drift. TempoPFN's strongest
        # non-SDE ablation was its complex-seasonality prior, so a minority of
        # components receive slow amplitude and phase modulation while the
        # stationary baseline remains well represented.
        modulated = np.nonzero((k > j) & (rng.random(n) < 0.35))[0]
        if modulated.size:
            # Map global row indices into the active component block.
            modulated_local = np.searchsorted(active, modulated)
            jittered_period = per[modulated] * rng.uniform(
                0.95, 1.05, size=(modulated.size, 1)
            )
            modulated_arg = (
                2.0 * np.pi * t / jittered_period + phase[modulated]
            )
            m_per = np.clip(
                per[modulated] * rng.uniform(
                    4.0, 12.0, size=(modulated.size, 1)
                ),
                32.0,
                2.0 * L,
            )
            m_phase = rng.uniform(
                0.0, 2.0 * np.pi, size=(modulated.size, 1)
            )
            slow = np.sin(2.0 * np.pi * t / m_per + m_phase)
            amp_mod = 1.0 + rng.uniform(
                0.05, 0.45, size=(modulated.size, 1)
            ) * slow
            phase_mod = rng.uniform(
                0.05, 0.75, size=(modulated.size, 1)
            ) * np.sin(2.0 * np.pi * t / (1.7 * m_per) - m_phase)
            component[modulated_local] = (
                amp[modulated]
                * amp_mod
                * np.sin(modulated_arg + phase_mod)
            )
        out[active] += component
    return out


def _sparse_jumps(rng: np.random.Generator, n: int, L: int, rate: float, scale) -> np.ndarray:
    """A (n, L) block of mostly-zero values with occasional N(0, scale) jumps.

    ``cumsum`` over this yields a piecewise-constant level; ``exp(cumsum)`` of a
    scaled version yields a piecewise-constant positive multiplier.
    """
    mask = rng.random((n, L)) < rate
    mask[:, 0] = False
    rows, cols = np.nonzero(mask)
    jumps = np.zeros((n, L), dtype=np.float64)
    if rows.size == 0:
        return jumps
    # Rates are O(1/L), so draw magnitudes only for actual events rather than
    # allocating and filling a second dense n×L normal array.
    s = np.asarray(scale, dtype=np.float64)
    event_scale = s if s.ndim == 0 else s.reshape(n)[rows]
    jumps[rows, cols] = rng.normal(0.0, 1.0, size=rows.size) * event_scale
    return jumps


def _measurement_artifacts(
    rng: np.random.Generator,
    block: np.ndarray,
    *,
    preserve_nonnegative: bool,
    preserve_integers: bool = False,
    allow_reverse: bool = True,
    allow_range_artifacts: bool = True,
) -> np.ndarray:
    """Apply sparse, cheap real-measurement effects to a generated block.

    TempoPFN reports a 5.4% aggregate CRPS gain from its complete augmentation
    pipeline, but does not isolate optimal probabilities for Toto2. These rates
    are deliberately conservative: most rows remain untouched, and a selected
    row receives only plausible reversal/sign, censoring, quantization, or
    sample-and-hold behavior.
    """
    original = np.asarray(block, dtype=np.float64)
    out = original.copy()
    n, L = out.shape

    reverse = (rng.random(n) < 0.06) if allow_reverse else np.zeros(n, dtype=bool)
    out[reverse] = out[reverse, ::-1]

    if not preserve_nonnegative:
        invert = rng.random(n) < 0.04
        out[invert] *= -1.0

    calibration_len = min(L, 512)

    # Calibrate sensor thresholds from the initial observed prefix. Using a
    # whole-path quantile would make history depend on the unseen target.
    # Selections and parameter draws happen even when range artifacts are
    # disabled, preserving the RNG sequence for controlled family comparisons.
    for row in np.nonzero(rng.random(n) < 0.06)[0]:
        q = float(rng.uniform(0.03, 0.18))
        upper = rng.random() < 0.5
        if not allow_range_artifacts:
            continue
        calibration = out[row, :calibration_len]
        if upper:
            threshold = np.quantile(calibration, 1.0 - q)
            out[row] = np.minimum(out[row], threshold)
        else:
            threshold = np.quantile(calibration, q)
            out[row] = np.maximum(out[row], threshold)

    quantized = np.nonzero(rng.random(n) < 0.07)[0]
    if quantized.size:
        levels = rng.integers(16, 257, size=(quantized.size, 1))
        if allow_range_artifacts:
            x = out[quantized]
            calibration = x[:, :calibration_len]
            lo = calibration.min(axis=1, keepdims=True)
            hi = calibration.max(axis=1, keepdims=True)
            step = (hi - lo) / np.maximum(levels - 1, 1)
            safe_step = np.where(step < 1e-12, 1.0, step)
            clipped = np.clip(x, lo, hi)
            out[quantized] = (
                lo + np.rint((clipped - lo) / safe_step) * safe_step
            )

    # Zero-order-hold resampling approximates telemetry gathered at a lower
    # cadence and forwarded at the nominal cadence.
    held = np.nonzero(rng.random(n) < 0.04)[0]
    if held.size:
        factors = rng.choice([2, 4, 8], size=held.size, p=[0.55, 0.30, 0.15])
        for factor in (2, 4, 8):
            rows = held[factors == factor]
            if rows.size:
                out[rows] = np.repeat(
                    out[rows, ::factor], factor, axis=1
                )[:, :L]
    if preserve_integers:
        out = np.maximum(np.rint(out), 0.0)
    # Heavy zero inflation plus upper censoring can otherwise collapse a sparse
    # row to its baseline. Such a row carries no forecasting signal.
    degenerate = out[:, :calibration_len].std(axis=1) < 1e-9
    out[degenerate] = original[degenerate]
    return out


# ── family builders: each returns a (n, L) float64 block ────────────────────


def _trend_seasonal_ar(rng: np.random.Generator, n: int, L: int, *,
                       hi_frac: float = 0.25, exc_lo: float = 0.4,
                       exc_hi: float = 3.0, clean_frac: float = 0.4,
                       clean_lo: float = 0.02, clean_hi: float = 0.12) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    level = rng.normal(0.0, 1.0, size=(n, 1))
    # v3: bimodal trend. The total trend EXCURSION over the series is drawn directly
    # (0..exc across t/(L-1)), so the trend sits ~16x below v2's slope*t — v2's linear
    # trend was a measured ~16x too strong vs real data at production lengths.
    _hi = rng.random((n, 1)) < hi_frac
    exc = np.where(_hi, rng.normal(0.0, exc_hi, size=(n, 1)),
                   rng.normal(0.0, exc_lo, size=(n, 1)))
    tn = t / max(L - 1, 1)
    series = level + exc * tn + _seasonal(rng, n, L)
    phi = rng.uniform(0.0, 0.85, size=n)
    clean = rng.random((n, 1)) < clean_frac
    sigma = np.where(
        clean,
        rng.uniform(clean_lo, clean_hi, size=(n, 1)),
        rng.uniform(0.1, 0.6, size=(n, 1)),
    )
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
    # Piecewise-affine drift complements abrupt level jumps. Sparse slope
    # changes create ramps and recoveries without the explosive scale of an I(2)
    # process, covering TempoPFN's high-impact Step/Sawtooth structures.
    slope = rng.normal(0.0, 1.0 / L, size=(n, 1)) + np.cumsum(
        _sparse_jumps(rng, n, L, rate=2.0 / L, scale=4.0 / L), axis=1
    )
    piecewise_trend = np.cumsum(slope, axis=1)
    return level + piecewise_trend + seas + noise


def _multiplicative(rng: np.random.Generator, n: int, L: int, *,
                    hi_frac: float = 0.25, exc_lo: float = 0.3, exc_hi: float = 2.0) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    # v3: bimodal log-growth excursion (drawn directly), same rationale as the linear trend.
    _hg = rng.random((n, 1)) < hi_frac
    gexc = np.where(_hg, rng.normal(0.0, exc_hi, size=(n, 1)),
                    rng.normal(0.0, exc_lo, size=(n, 1)))
    tn = t / max(L - 1, 1)
    base_level = np.exp(gexc * tn + rng.normal(0.0, 0.3, size=(n, 1)))  # positive, drifting
    amp = rng.uniform(0.1, 0.6, size=(n, 1))
    seasonal_shape = _seasonal(rng, n, L, k_max=1)
    seasonal_shape = _prefix_standardize(seasonal_shape, center=False)
    seas = 1.0 + amp * seasonal_shape
    # ForecastPFN uses multiplicative Weibull noise so skew varies without
    # making signal-to-noise depend on the base level. Center on the exact
    # Weibull expectation to preserve trend and seasonality in expectation.
    shape = np.exp(rng.uniform(np.log(1.2), np.log(8.0), size=(n, 1)))
    uniform = np.maximum(rng.random((n, L)), np.finfo(np.float64).tiny)
    weibull = (-np.log(uniform)) ** (1.0 / shape)
    weibull_mean = np.exp(gammaln(1.0 + 1.0 / shape))
    noise = 1.0 + rng.uniform(0.02, 0.15, size=(n, 1)) * (
        weibull - weibull_mean
    )
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
    burn = 512
    innov = rng.normal(0.0, 1.0, size=(n, L + burn)) * sigma
    # Keep AR(2) genuinely stationary. The old 0.005-per-step drift accumulated
    # to roughly 20 units at full context and overwhelmed the AR covariance.
    return _ar2_batch(innov, a1, a2)[:, burn:]


def _integrated(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Integrated paths with forecastable differenced dynamics.

    A corpus of pure iid random walks mostly teaches persistence because future
    increments are irreducible noise. Real integrated series more often have
    autocorrelated increments, recurring changes, or a persistent local drift.
    Retain an iid minority, but give most rows structure in first differences
    that a forecaster can identify from context.
    """
    branch = rng.random(n)
    # Preserve a hard, genuine I(2) minority for coverage, but do not let it
    # dominate a family whose tiny local slope is poorly represented by the
    # fixed model's expanding global scaler. A separate near-I(2) branch uses
    # highly persistent (but stationary) velocity, retaining local-linear
    # forecast structure without cubic variance growth.
    order2 = branch < 0.10
    persistent_velocity = (branch >= 0.10) & (branch < 0.45)
    drift = rng.normal(0.0, 0.02, size=(n, 1))
    sigma = rng.uniform(0.2, 1.0, size=(n, 1))
    raw = rng.normal(0.0, 1.0, size=(n, L)) * sigma

    # Correlated increments turn the family into a broad ARIMA prior instead of
    # almost exclusively an unpredictable random walk. Scaling innovations by
    # sqrt(1-phi²) keeps the marginal increment variance comparable across phi.
    correlated = (rng.random((n, 1)) < 0.70) | persistent_velocity[:, None]
    phi = rng.uniform(-0.35, 0.85, size=n)
    phi[persistent_velocity] = rng.uniform(
        0.97, 0.999, size=persistent_velocity.sum()
    )
    ar_steps = _ar1_batch(
        raw * np.sqrt(np.maximum(1.0 - phi[:, None] ** 2, 1e-3)),
        phi,
    )
    steps = np.where(correlated, ar_steps, raw)

    # A minority has periodic first differences, as in seasonal ARIMA and
    # accumulated demand/sensor totals. Integrating a zero-mean periodic signal
    # remains bounded around the stochastic trend rather than exploding.
    seasonal_on = rng.random((n, 1)) < 0.30
    seasonal_steps = _seasonal(rng, n, L, k_max=1)
    seasonal_steps = _prefix_standardize(seasonal_steps)
    steps += seasonal_on * seasonal_steps * sigma * rng.uniform(0.05, 0.35, size=(n, 1))
    steps += drift

    walk = np.cumsum(steps, axis=1)
    walk2 = np.cumsum(walk, axis=1)
    o2 = order2[:, None]
    # I(2) variance grows cubically with length versus linearly for I(1).
    # Dividing by L aligns their standard-deviation order; sqrt(L) left the
    # double-integrated branch about sqrt(L) too large.
    return np.where(o2, walk2 / max(L, 1), walk)


def _threshold_ar(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # SETAR(2): coefficient flips with the sign of the previous value — a simple
    # nonlinear recurrence that produces asymmetric, regime-switching dynamics.
    phi_hi = rng.uniform(0.3, 0.9, size=n)
    phi_lo = rng.uniform(-0.9, 0.3, size=n)
    const_hi = rng.normal(0.0, 0.3, size=n)
    const_lo = rng.normal(0.0, 0.3, size=n)
    sigma = rng.uniform(0.2, 0.7, size=(n, 1))
    burn = 256
    total = L + burn
    innov = rng.normal(0.0, 1.0, size=(n, total)) * sigma
    x = np.empty((n, total), dtype=np.float64)
    x[:, 0] = innov[:, 0]
    for t in range(1, total):
        prev = x[:, t - 1]
        hi = prev >= 0.0
        phi = np.where(hi, phi_hi, phi_lo)
        const = np.where(hi, const_hi, const_lo)
        x[:, t] = np.clip(const + phi * prev + innov[:, t], -1e6, 1e6)
    return x[:, burn:]


def _chaotic(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # Bounded chaotic maps: logistic x_{t+1}=r x(1-x) with r∈[3.6,4.0], and the
    # sine map r sin(pi x). Both stay in [0,1]; standardise afterwards. A random
    # observation length as a "sampling rate" adds variety across series.
    use_sine = rng.random(n) < 0.5
    r_log = rng.uniform(3.6, 4.0, size=n)
    r_sin = rng.uniform(0.85, 1.0, size=n)
    x0 = rng.uniform(0.05, 0.95, size=n)
    cur = x0.copy()
    # Remove initial-condition transients before emitting observations.
    for _ in range(64):
        nxt_log = r_log * cur * (1.0 - cur)
        nxt_sin = r_sin * np.sin(np.pi * cur)
        cur = np.clip(np.where(use_sine, nxt_sin, nxt_log), 0.0, 1.0)
    x = np.empty((n, L), dtype=np.float64)
    x[:, 0] = cur
    for t in range(1, L):
        nxt_log = r_log * cur * (1.0 - cur)
        nxt_sin = r_sin * np.sin(np.pi * cur)
        cur = np.where(use_sine, nxt_sin, nxt_log)
        cur = np.clip(cur, 0.0, 1.0)
        x[:, t] = cur
    x = _prefix_standardize(x)
    # Most real observations of nonlinear systems include measurement noise;
    # retain a clean minority to preserve the exact dynamical prior.
    noisy = rng.random((n, 1)) < 0.7
    x += noisy * rng.normal(0.0, 1.0, size=(n, L)) * rng.uniform(
        0.01, 0.15, size=(n, 1)
    )
    return x * np.exp(rng.uniform(np.log(0.2), np.log(5.0), size=(n, 1))) \
        + rng.normal(0.0, 2.0, size=(n, 1))


def _spectral_gp(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Composite RBF/Rational-Quadratic GP paths in O(n L log L).

    Chronos KernelSynth uses both kernels. We use a 2L circulant embedding and
    retain its first L samples. This preserves the requested kernel covariance
    over the emitted interval without making its two endpoints artificial
    neighbours, which an L-periodic inverse FFT would do.
    """
    embed_len = 2 * L
    lag = np.minimum(np.arange(embed_len), embed_len - np.arange(embed_len))[None, :]
    lengthscale = np.exp(rng.uniform(np.log(8.0), np.log(256.0), size=(n, 1)))
    scaled_lag2 = (lag / lengthscale) ** 2
    rbf_cov = np.exp(-0.5 * scaled_lag2)
    alpha = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=(n, 1)))
    rq_cov = (1.0 + scaled_lag2 / (2.0 * alpha)) ** (-alpha)
    blend = rng.beta(0.7, 0.7, size=(n, 1))
    covariance = blend * rbf_cov + (1.0 - blend) * rq_cov
    spectrum = np.maximum(np.fft.rfft(covariance, axis=1).real, 0.0)
    z = rng.standard_normal(spectrum.shape) + 1j * rng.standard_normal(spectrum.shape)
    z[:, 0] = 0.0
    x = np.fft.irfft(z * np.sqrt(spectrum), n=embed_len, axis=1)[:, :L]
    return _prefix_standardize(x)


def _davies_harte_fgn(
    rng: np.random.Generator, hurst: np.ndarray, L: int
) -> np.ndarray:
    """Exact fractional Gaussian noise via Davies-Harte embedding.

    The covariance is
    ``γ(k)=0.5[(k+1)^(2H)-2k^(2H)+|k-1|^(2H)]``. Embedding it in a
    ``2L`` circulant matrix gives a real Gaussian sample with the requested
    finite-lag covariance, unlike a generic ``1/f^β`` envelope.
    """
    h = np.asarray(hurst, dtype=np.float64).reshape(-1, 1)
    n = h.shape[0]
    if n == 0:
        return np.empty((0, L), dtype=np.float64)

    k = np.arange(L, dtype=np.float64)[None, :]
    power = 2.0 * h
    covariance = 0.5 * (
        (k + 1.0) ** power
        - 2.0 * k ** power
        + np.abs(k - 1.0) ** power
    )
    circulant = np.concatenate(
        [covariance, np.zeros((n, 1)), covariance[:, 1:][:, ::-1]], axis=1
    )
    eigenvalues = np.maximum(
        np.fft.rfft(circulant, axis=1).real, 0.0
    )
    z = (
        rng.standard_normal(eigenvalues.shape)
        + 1j * rng.standard_normal(eigenvalues.shape)
    ) / np.sqrt(2.0)
    z[:, 0] = rng.standard_normal(n)
    z[:, -1] = rng.standard_normal(n)
    return np.fft.irfft(
        z * np.sqrt(eigenvalues),
        n=2 * L,
        axis=1,
        norm="ortho",
    )[:, :L]


def _long_memory(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Fractional power-law paths with both persistent and rough regimes.

    For Hurst H, fractional Gaussian noise has beta=2H-1. We sample that
    stationary increment process, then cumulatively sum selected rows to obtain
    mathematically consistent fractional Brownian motion paths.
    """
    # Generate on a 2L embedding and retain only the first L samples. Directly
    # inverse-FFTing an L-point spectrum makes the emitted path circular, so an
    # evaluation target immediately after a long context sits near an
    # artificial wrap boundary.
    embed_len = 2 * L
    f = np.fft.rfftfreq(embed_len)
    safe_f = np.maximum(f, 1.0 / embed_len)[None, :]
    hurst = rng.uniform(0.3, 0.85, size=(n, 1))
    level_path = rng.random((n, 1)) < 0.40
    # Generate stationary fGn first (beta=2H-1), then integrate selected rows
    # to obtain actual non-stationary fBm. A steep beta=2H+1 spectrum sampled
    # directly by an inverse FFT is periodic coloured noise, not true fBm.
    beta = 2.0 * hurst - 1.0
    amp = safe_f ** (-0.5 * beta)
    # Some rows change roughness above a random frequency, giving smooth
    # large-scale structure and rough local variation (or the reverse) without
    # another FFT. Match amplitudes at the split to avoid a spectral jump.
    multiscale = rng.random((n, 1)) < 0.4
    split_idx = rng.integers(8, max(9, f.size // 3), size=(n, 1))
    split_f = np.maximum(split_idx / embed_len, 1.0 / embed_len)
    hurst_hi = rng.uniform(0.3, 0.8, size=(n, 1))
    beta_hi = 2.0 * hurst_hi - 1.0
    above = np.arange(f.size)[None, :] > split_idx
    amp_hi = split_f ** (-0.5 * beta) \
        * (safe_f / split_f) ** (-0.5 * beta_hi)
    amp = np.where(multiscale & above, amp_hi, amp)
    amp[:, 0] = 0.0
    z = rng.standard_normal((n, f.size)) + 1j * rng.standard_normal((n, f.size))
    x = np.fft.irfft(z * amp, n=embed_len, axis=1)[:, :L]
    # Most rows use exact fGn covariance. Retain a minority of the approximate
    # piecewise-spectrum paths because scale-dependent roughness is useful
    # real-world coverage not represented by a single Hurst exponent.
    exact_core = rng.random(n) < 0.65
    exact_rows = np.nonzero(exact_core)[0]
    if exact_rows.size:
        x[exact_rows] = _davies_harte_fgn(
            rng, hurst[exact_rows], L
        )
    # fBm is the cumulative sum of fGn. Remove the initial level so scale and
    # shift augmentation elsewhere do not depend on an arbitrary FFT endpoint.
    if np.any(level_path):
        level_rows = np.nonzero(level_path.reshape(-1))[0]
        x[level_rows] = np.cumsum(x[level_rows], axis=1)
        x[level_rows] -= x[level_rows, :1]
    return _prefix_standardize(x)


def _ou_stochastic_vol(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Regime-switching mean reversion with bounded stochastic volatility.

    This is a CPU-cheap discrete Euler/AR analogue of TempoPFN's highest-impact
    OU SDE prior. Regime paths, seasonal means, volatility envelopes, and
    heavy-tail masks are sampled in whole blocks; only the state recurrence
    scans time, vectorised across all rows.
    """
    # Toggle between a fast/quiet and a slow/volatile regime. A cumulative XOR
    # builds persistent Markov-like paths without a per-row Python loop.
    # Most regimes persist long enough to infer from context. Retain a small
    # rapid-switch minority rather than making the synthetic task uniformly
    # easier and deleting realistic hard cases.
    rapid_switch = rng.random((n, 1)) < 0.15
    persistent_rate = np.exp(
        rng.uniform(np.log(0.0005), np.log(0.03), size=(n, 1))
    )
    rapid_rate = np.exp(rng.uniform(np.log(0.03), np.log(0.15), size=(n, 1)))
    switch_rate = np.where(rapid_switch, rapid_rate, persistent_rate)
    switches = rng.random((n, L)) < switch_rate
    switches[:, 0] = rng.random(n) < 0.5
    regime = np.bitwise_and(np.cumsum(switches, axis=1), 1).astype(np.int8)

    # One mean-reversion speed per row lets SciPy execute the recurrence in
    # compiled code. Regime paths still switch equilibrium mean and volatility;
    # rows span both fast/quiet and slow/persistent reversion rates.
    speed = rng.random((n, 1))
    ultra_slow = speed < 0.20
    slow = (speed >= 0.20) & (speed < 0.65)
    fast_phi = rng.uniform(0.900, 0.980, size=(n, 1))
    slow_phi = rng.uniform(0.985, 0.9975, size=(n, 1))
    ultra_slow_phi = rng.uniform(0.9975, 0.9995, size=(n, 1))
    phi = np.where(
        ultra_slow,
        ultra_slow_phi,
        np.where(slow, slow_phi, fast_phi),
    )
    mu0 = rng.normal(-2.0, 1.0, size=(n, 1))
    mu1 = rng.normal(2.0, 1.0, size=(n, 1))
    mean = np.where(regime == 0, mu0, mu1)
    seasonal_on = rng.random((n, 1)) < 0.6
    sigma_seasonal_on = rng.random((n, 1)) < 0.3
    seasonal_component = np.zeros((n, L), dtype=np.float64)
    seasonal_rows = np.nonzero(
        (seasonal_on | sigma_seasonal_on).reshape(-1)
    )[0]
    if seasonal_rows.size:
        seasonal_component[seasonal_rows] = _seasonal(
            rng, int(seasonal_rows.size), L, k_max=3
        )
    mean += seasonal_on * seasonal_component \
        * rng.uniform(0.5, 3.0, size=(n, 1))

    # Mean-reverting log volatility gives clustered but bounded uncertainty.
    # The persistence range maps TempoPFN's kappa_v=[0.5,5] at dt=0.01 into
    # exp(-kappa_v*dt)≈[0.951,0.995].
    log_sigma0 = rng.normal(np.log(0.3), 0.3, size=(n, 1))
    log_sigma1 = rng.normal(np.log(1.5), 0.5, size=(n, 1))
    log_sigma_mean = np.where(regime == 0, log_sigma0, log_sigma1)
    vol_rho = rng.uniform(0.951, 0.995, size=(n, 1))
    vol_eta = rng.uniform(0.03, 0.20, size=(n, 1))
    vol_eps = rng.standard_normal((n, L))
    vol_drive = (1.0 - vol_rho) * log_sigma_mean \
        + np.sqrt(1.0 - vol_rho * vol_rho) * vol_eta * vol_eps
    log_vol = np.empty((n, L), dtype=np.float64)
    log_vol[:, 0] = log_sigma_mean[:, 0]
    for i in range(n):
        rho = float(vol_rho[i, 0])
        log_vol[i, 1:] = lfilter(
            [1.0],
            [1.0, -rho],
            vol_drive[i, 1:],
            zi=[rho * log_vol[i, 0]],
        )[0]
    vol = np.exp(np.clip(log_vol, -5.0, 5.0))
    # TempoPFN independently applies seasonality to sigma in 30% of paths.
    sigma_seasonal = sigma_seasonal_on * seasonal_component * rng.uniform(
        0.03, 0.18, size=(n, 1)
    )
    vol *= np.exp(np.clip(sigma_seasonal, -0.7, 0.7))

    eps = rng.standard_normal((n, L))
    heavy = np.nonzero(rng.random(n) < 0.35)[0]
    if heavy.size:
        # Replace only heavy-tailed rows; drawing Student-t noise for every row
        # previously discarded 65% of that relatively expensive work.
        eps[heavy] = (
            rng.standard_t(4.0, size=(heavy.size, L)) / np.sqrt(2.0)
        )
    shocks = rng.random((n, L)) < (3.0 / L)
    shock_rows, shock_cols = np.nonzero(shocks)
    # As with sparse jumps, draw shock magnitudes only at the O(n) events.
    eps[shock_rows, shock_cols] += rng.normal(
        0.0, 5.0, size=shock_rows.size
    )

    innovation_scale = np.sqrt(np.maximum(1.0 - phi * phi, 1e-6))
    drive = (1.0 - phi) * mean + innovation_scale * vol * eps
    out = np.empty((n, L), dtype=np.float64)
    out[:, 0] = mean[:, 0] + vol[:, 0] * eps[:, 0]
    for i in range(n):
        p = float(phi[i, 0])
        out[i, 1:] = lfilter(
            [1.0], [1.0, -p], drive[i, 1:], zi=[p * out[i, 0]]
        )[0]

    scale = np.exp(rng.uniform(np.log(0.1), np.log(50.0), size=(n, 1)))
    shift = rng.uniform(-100.0, 100.0, size=(n, 1))
    return out * scale + shift


def _physical_sensors(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Generic physical measurements without matching one private dataset.

    Four row-level archetypes cover smooth signed measurements, bounded
    percentages, pressure-like wandering levels, and non-negative skewed
    magnitudes. All share multi-cadence seasonality, smooth synoptic variation,
    and sparse fronts/gusts.
    """
    seasonal = _seasonal(rng, n, L, k_max=2)
    smooth = _spectral_gp(rng, n, L)
    fronts = np.cumsum(
        _sparse_jumps(rng, n, L, rate=5.0 / L, scale=1.0), axis=1
    )
    base = (
        seasonal * rng.uniform(0.3, 2.0, size=(n, 1))
        + smooth * rng.uniform(0.2, 1.2, size=(n, 1))
        + fronts * rng.uniform(0.2, 1.0, size=(n, 1))
    )

    kind = rng.integers(0, 4, size=n)
    out = base.copy()

    bounded = kind == 1
    if bounded.any():
        gain = rng.uniform(0.8, 3.5, size=(int(bounded.sum()), 1))
        midpoint = rng.uniform(-0.8, 0.8, size=(int(bounded.sum()), 1))
        out[bounded] = 100.0 / (1.0 + np.exp(-gain * (base[bounded] - midpoint)))

    pressure = kind == 2
    if pressure.any():
        count = int(pressure.sum())
        # Sample a per-step diffusion coefficient. Dividing by sqrt(L) made the
        # process depend on the requested total length, so prefixes generated
        # under different horizons did not share one stochastic law.
        diffusion = np.exp(
            rng.uniform(np.log(0.03), np.log(0.20), size=(count, 1))
        )
        walk = np.cumsum(
            rng.standard_normal((count, L)) * diffusion, axis=1
        )
        level = rng.uniform(900.0, 1100.0, size=(count, 1))
        out[pressure] = level + walk \
            + 2.0 * fronts[pressure] + 0.5 * seasonal[pressure]

    magnitude = kind == 3
    if magnitude.any():
        count = int(magnitude.sum())
        gusts = (rng.random((count, L)) < (8.0 / L)) \
            * rng.lognormal(0.0, 0.8, size=(count, L))
        power = rng.uniform(1.0, 1.6, size=(count, 1))
        out[magnitude] = np.abs(base[magnitude]) ** power + gusts

    return out


def _seasonal_counts(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    """Seasonal Poisson/negative-binomial counts with decaying bursts.

    This keeps count positivity and discreteness intact while covering
    overdispersion, cadence-linked rate variation, slow signed growth, and
    release/news-like bursts. Computation remains batched across rows.
    """
    t = np.arange(L, dtype=np.float64)[None, :]
    period = rng.choice(
        _SEASONAL_PERIODS, size=(n, 1), p=_SEASONAL_PROBS
    )
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    amp = rng.uniform(0.15, 0.8, size=(n, 1))
    log_rate = amp * np.sin(2.0 * np.pi * t / period + phase)
    second = rng.random((n, 1)) < 0.55
    log_rate += second * (0.5 * amp) * np.sin(
        4.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    )
    # A minority carry explicit calendar interaction: intraday cadence plus
    # seven day-specific factors, with a randomized weekend dip or lift.
    calendar = rng.random((n, 1)) < 0.35
    sampled_day_period = rng.choice([24, 48, 96, 144], size=(n, 1))
    # A seven-step primary period represents daily observations, so each sample
    # is one day. Sub-daily rows retain a cadence-scaled day length.
    day_period = np.where(period <= 7.0, 1, sampled_day_period)
    day_idx = (np.floor_divide(np.arange(L)[None, :], day_period) % 7).astype(np.int64)
    day_factors = rng.normal(0.0, 0.12, size=(n, 7))
    day_factors[:, 5:] += rng.uniform(-0.8, 0.3, size=(n, 1))
    calendar_effect = np.take_along_axis(day_factors, day_idx, axis=1)
    log_rate += calendar * calendar_effect
    excursion = rng.uniform(-0.5, 0.5, size=(n, 1))
    log_rate += excursion * t / max(L - 1, 1)

    # Sparse positive impulses filtered by row-specific decay create bursts
    # without a Python loop over timesteps.
    impulses = (
        (rng.random((n, L)) < (2.0 / L))
        * rng.uniform(1.0, 10.0, size=(n, L))
    )
    burst = _ar1_batch(impulses, rng.uniform(0.85, 0.995, size=(n, 1)))
    base = np.exp(rng.uniform(np.log(3.0), np.log(3000.0), size=(n, 1)))
    lam = base * np.exp(np.clip(log_rate, -5.0, 5.0)) * (1.0 + burst)
    np.clip(lam, 0.0, 1.0e7, out=lam)

    # A gamma-mixed Poisson is negative-binomial marginally. A separate
    # lognormal-AR intensity branch makes overdispersion persistent through
    # time rather than redrawing an unrelated multiplier at every step.
    overdispersion_kind = rng.random((n, 1))
    gamma_mixed = overdispersion_kind < 0.35
    persistent_mixed = (overdispersion_kind >= 0.35) & (
        overdispersion_kind < 0.70
    )
    shape = rng.uniform(0.5, 4.0, size=(n, 1))
    gamma_intensity = lam * rng.gamma(shape, 1.0 / shape, size=(n, L))
    intensity_state = _ar1_batch(
        rng.standard_normal((n, L)),
        rng.uniform(0.70, 0.995, size=n),
    )
    intensity_state = _prefix_standardize(intensity_state)
    eta = rng.uniform(0.10, 0.60, size=(n, 1))
    # Center the lognormal multiplier at expectation one.
    persistent_intensity = lam * np.exp(
        np.clip(eta * intensity_state - 0.5 * eta * eta, -3.0, 3.0)
    )
    mixed = np.where(
        gamma_mixed,
        gamma_intensity,
        np.where(persistent_mixed, persistent_intensity, lam),
    )
    return rng.poisson(mixed).astype(np.float64)


def _intermittent(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # Seasonal zero-inflated demand. Occurrence probabilities vary by cadence
    # instead of being iid, teaching the model forecastable sparse structure.
    t = np.arange(L, dtype=np.float64)[None, :]
    base_p = rng.uniform(0.03, 0.35, size=(n, 1))
    period = rng.choice([7.0, 12.0, 24.0, 48.0, 168.0], size=(n, 1))
    season = rng.uniform(0.2, 1.2, size=(n, 1)) * np.sin(
        2.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    )
    # Intermittent-demand models separate occurrence from positive size. A
    # persistent latent state makes occurrence probability evolve over time
    # instead of producing iid Bernoulli zeros.
    occurrence_state = _ar1_batch(
        rng.normal(0.0, 1.0, size=(n, L)),
        rng.uniform(0.0, 0.95, size=n),
    )
    occurrence_state = _prefix_standardize(occurrence_state)
    logit = np.log(base_p / (1.0 - base_p)) + season \
        + rng.uniform(0.0, 1.2, size=(n, 1)) * occurrence_state
    p = 1.0 / (1.0 + np.exp(-logit))
    occur = (rng.random((n, L)) < p).astype(np.float64)

    # Positive demand sizes are rarely iid in practice: customer/product scale
    # and local demand intensity persist. Couple a smooth latent size state
    # weakly to the occurrence state, while retaining gamma observation noise.
    independent_size_state = _ar1_batch(
        rng.normal(0.0, 1.0, size=(n, L)),
        rng.uniform(0.5, 0.98, size=n),
    )
    independent_size_state = _prefix_standardize(independent_size_state)
    coupling = rng.uniform(0.15, 0.55, size=(n, 1))
    size_state = (
        coupling * occurrence_state
        + np.sqrt(1.0 - coupling * coupling) * independent_size_state
    )
    size_factor = np.exp(np.clip(
        rng.uniform(0.15, 0.45, size=(n, 1)) * size_state, -1.5, 1.5
    ))
    magnitude = np.maximum(1.0, np.rint(
        rng.gamma(shape=2.0, scale=1.0, size=(n, L))
        * rng.uniform(1.0, 10.0, size=(n, 1))
        * np.exp(0.25 * season)
        * size_factor
    ))
    return occur * magnitude


def _pulse_event_mask(
    rng: np.random.Generator, n: int, L: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample independent and history-dependent pulse occurrence processes.

    Kind 0 is a small calibration-only Poisson/Bernoulli branch. Kinds 1--3
    carry forecastable timing information through repeated cadence, seasonal
    conditional intensity, or self-excitation respectively.
    """
    kind = rng.choice(4, size=n, p=[0.15, 0.35, 0.30, 0.20])
    events = np.zeros((n, L), dtype=bool)

    # Independent innovations: history identifies only the marginal event rate,
    # never the exact next event. Keep this branch small but nonzero so quantile
    # forecasts still learn honest tail mass.
    independent = np.nonzero(kind == 0)[0]
    if independent.size:
        rate = rng.uniform(2.0, 6.0, size=(independent.size, 1)) / max(L, 1)
        events[independent] = rng.random((independent.size, L)) < rate

    # Repeated events with modest timing jitter, following the learnable
    # periodic/clustered spike construction used by synthetic forecasting
    # priors. At least several cycles occur in a full-context series.
    periodic = np.nonzero(kind == 1)[0]
    periods = rng.choice(
        np.asarray([24, 48, 96, 168, 256, 336, 512]),
        size=periodic.size,
        p=np.asarray([0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.10]),
    )
    phases = np.asarray(
        [rng.integers(0, max(int(period), 1)) for period in periods]
    )
    for row, period, phase in zip(
        periodic, periods, phases, strict=True
    ):
        nominal = np.arange(int(phase), L, int(period))
        jitter = np.rint(
            rng.normal(0.0, max(1.0, 0.04 * period), size=nominal.size)
        ).astype(np.int64)
        starts = np.clip(nominal + jitter, 1, L - 1)
        events[row, starts] = True

    # A cyclic conditional intensity makes event probability forecastable while
    # retaining irreducible Bernoulli timing uncertainty.
    seasonal = np.nonzero(kind == 2)[0]
    if seasonal.size:
        t = np.arange(L, dtype=np.float64)[None, :]
        period = rng.choice(
            np.asarray([24.0, 48.0, 96.0, 168.0, 336.0]),
            size=(seasonal.size, 1),
        )
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(seasonal.size, 1))
        base_rate = rng.uniform(4.0, 16.0, size=(seasonal.size, 1)) / max(L, 1)
        modulation = 0.15 + 1.70 * (
            0.5 + 0.5 * np.sin(2.0 * np.pi * t / period + phase)
        )
        events[seasonal] = (
            rng.random((seasonal.size, L)) < base_rate * modulation
        )

    # Discrete Hawkes analogue:
    #   p_t = mu + s_t,
    #   s_{t+1} = decay*s_t + (1-decay)*branching*event_t.
    # Its expected offspring count is ``branching < 1``, so it is stable, and
    # observed events raise the near-future conditional event probability.
    hawkes = np.nonzero(kind == 3)[0]
    if hawkes.size:
        baseline = rng.uniform(2.0, 8.0, size=hawkes.size) / max(L, 1)
        decay = rng.uniform(0.70, 0.96, size=hawkes.size)
        branching = rng.uniform(0.30, 0.80, size=hawkes.size)
        excitation = np.zeros(hawkes.size, dtype=np.float64)
        uniforms = rng.random((hawkes.size, L))
        for step in range(L):
            occurred = uniforms[:, step] < np.minimum(
                baseline + excitation, 0.35
            )
            events[hawkes, step] = occurred
            excitation = (
                decay * excitation
                + (1.0 - decay) * branching * occurred
            )

    events[:, 0] = False
    return events, kind


def _pulse_outlier(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    # A smooth base with isolated innovations, predictable event processes,
    # persistent shock/recovery responses, and genuine held-constant runs.
    base = _spectral_gp(rng, n, L) * rng.uniform(0.5, 2.0, size=(n, 1))
    base += _seasonal(rng, n, L, k_max=1) * rng.uniform(0.0, 1.0, size=(n, 1))

    events, kind = _pulse_event_mask(rng, n, L)
    magnitude_phi = rng.uniform(0.70, 0.98, size=n)
    magnitude_state = _ar1_batch(
        rng.normal(0.0, 1.0, size=(n, L))
        * np.sqrt(1.0 - magnitude_phi[:, None] ** 2),
        magnitude_phi,
    )
    magnitude_state = _prefix_standardize(magnitude_state)
    magnitude = rng.uniform(2.0, 8.0, size=(n, 1)) * np.exp(
        np.clip(
            rng.uniform(0.10, 0.40, size=(n, 1)) * magnitude_state,
            -1.0,
            1.0,
        )
    )
    # Periodic and seasonal rows also receive slowly evolving event magnitude;
    # independent pulses retain iid timing and Hawkes rows derive predictability
    # from occurrence clustering rather than a fabricated deterministic trend.
    learnable_magnitude = ((kind == 1) | (kind == 2))[:, None]
    magnitude_cycle = 1.0 + 0.25 * np.sin(
        2.0
        * np.pi
        * np.arange(L, dtype=np.float64)[None, :]
        / rng.choice(
            np.asarray([96.0, 168.0, 336.0, 672.0]), size=(n, 1)
        )
        + rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    )
    magnitude *= np.where(learnable_magnitude, magnitude_cycle, 1.0)
    sign = rng.choice(np.asarray([-1.0, 1.0]), size=(n, 1))
    impulses = events * sign * magnitude

    recovery = _ar1_batch(impulses, rng.uniform(0.75, 0.995, size=n))
    sharp_shape = rng.random((n, 1)) < 0.45
    series = base + np.where(sharp_shape, impulses, recovery)

    # Sparse event loops, not a time-axis scan: typically two starts per row.
    starts = rng.random((n, L)) < (2.0 / L)
    starts[:, 0] = False
    for row in range(n):
        for start in np.nonzero(starts[row])[0]:
            run = int(rng.integers(3, 65))
            end = min(int(start) + run, L)
            series[row, start:end] = series[row, start - 1]
    return series


def _weekly_demand(
    rng: np.random.Generator, n: int, L: int
) -> np.ndarray:
    """Non-negative period-7 demand with promotions, dips, and count rows.

    Adapted from the public ``j-test/dasadas_v18`` generator. Its dedicated 8%
    demand prior beat that generator's base model on a multi-domain pool, while
    10% overshot. The useful process is integrated here without replacing
    cascade-v16's richer GP, long-memory, OU, sensor, and count families.
    """
    time = np.arange(L, dtype=np.float64)[None, :]
    normalized_time = time / max(L - 1, 1)

    # A learned day-of-week profile, with a weekend-dip branch whose two-day
    # phase is randomized so it remains a generic weekly demand prior.
    seasonal_amplitude = rng.uniform(0.03, 0.5, size=(n, 1))
    profile = rng.normal(0.0, 1.0, size=(n, 7))
    profile -= profile.mean(axis=1, keepdims=True)
    has_weekend_dip = rng.random(n) < 0.5
    dip_start = rng.integers(0, 7, size=n)
    dip_depth = rng.uniform(0.4, 1.6, size=n)
    weekend_profile = np.zeros((n, 7), dtype=np.float64)
    rows = np.arange(n)
    weekend_profile[rows, dip_start] -= dip_depth
    weekend_profile[rows, (dip_start + 1) % 7] -= dip_depth
    weekend_profile -= weekend_profile.mean(axis=1, keepdims=True)
    profile += np.where(
        has_weekend_dip[:, None], weekend_profile, 0.0
    )
    profile -= profile.mean(axis=1, keepdims=True)
    phase = rng.integers(0, 7, size=(n, 1))
    weekday_index = (np.arange(L)[None, :] + phase) % 7
    weekly_log = seasonal_amplitude * np.take_along_axis(
        profile, weekday_index, axis=1
    )

    excursion = (
        rng.normal(0.0, 1.0, size=(n, 1))
        * rng.uniform(0.3, 2.5, size=(n, 1))
    )
    trend = excursion * normalized_time
    step_scale = rng.uniform(0.005, 0.05, size=(n, 1))
    random_walk = np.clip(
        np.cumsum(
            rng.normal(0.0, 1.0, size=(n, L)) * step_scale,
            axis=1,
        ),
        -3.0,
        3.0,
    )

    # Sparse promotions have a one-step echo; independent negative events
    # represent holidays, outages, or temporary stock constraints.
    promotion_mask = rng.random((n, L)) < (
        rng.uniform(1.0, 8.0, size=(n, 1)) / L
    )
    promotions = (
        promotion_mask
        * np.abs(rng.normal(0.0, 1.0, size=(n, L)))
        * rng.uniform(0.5, 2.5, size=(n, 1))
    )
    echo = np.zeros_like(promotions)
    echo[:, 1:] = (
        promotions[:, :-1] * rng.uniform(0.2, 0.6, size=(n, 1))
    )
    promotions += echo
    holiday_mask = rng.random((n, L)) < (
        rng.uniform(0.0, 4.0, size=(n, 1)) / L
    )
    holiday_dips = (
        holiday_mask
        * np.abs(rng.normal(0.0, 1.0, size=(n, L)))
        * rng.uniform(0.3, 1.5, size=(n, 1))
    )

    noise = (
        rng.normal(0.0, 1.0, size=(n, L))
        * rng.uniform(0.02, 0.25, size=(n, 1))
    )
    base = rng.uniform(0.0, 8.0, size=(n, 1))
    log_mean = np.clip(
        base
        + trend
        + random_walk
        + weekly_log
        + promotions
        - holiday_dips
        + noise,
        -8.0,
        13.0,
    )
    level = np.exp(log_mean)

    # A minority is emitted as exact Poisson demand; the rest remains positive
    # continuous magnitude data such as traffic, revenue, or energy load.
    is_count = rng.random(n) < 0.35
    count_scale = rng.uniform(1.0, 60.0, size=(n, 1)) / np.clip(
        level.mean(axis=1, keepdims=True), 1e-9, None
    )
    counts = rng.poisson(
        np.clip(level * count_scale, 0.0, 1e6)
    ).astype(np.float64)
    return np.where(is_count[:, None], counts, level)


# ── final safety gate ───────────────────────────────────────────────────────


def _sanitize(block: np.ndarray) -> np.ndarray:
    """Guarantee the contract: finite float64, no NaN/inf, bounded magnitude.

    The trainer's ``check_series`` rejects any non-finite value, which would
    fail the whole run — so this is the hard backstop after every family
    builder. Replaces non-finite values and clips to a generous bound.
    """
    x = np.asarray(block, dtype=np.float64)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    np.clip(x, -1e6, 1e6, out=x)
    return x

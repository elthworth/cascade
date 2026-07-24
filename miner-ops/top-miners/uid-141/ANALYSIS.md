# v18 weekly-demand integration

The public
[`j-test/dasadas_v18`](https://hub.hippius.com/models/j-test/dasadas_v18/main)
reports that a dedicated 8% period-7 demand family beat its base generator on a
multi-domain GIFT-Eval pool with a `+0.0639` paired cluster-bootstrap LCB. Its
10% variant overshot and failed, while increasing generic intermittent demand
also failed. The transferable contribution is therefore the focused process and
measured 8% weight, not the external generator's older base mixture.

Cascade v18 starts from the exact archived v16 source and adds that useful
section as `weekly_demand`. It emits non-negative continuous or Poisson-count
series with:

- randomized day-of-week profiles and optional two-day weekend dips;
- slow trend and clipped random-walk level evolution;
- sparse positive promotions with one-step echoes;
- sparse negative holiday/outage/stock-constraint events; and
- observation noise in log space.

The full v16 mixture is scaled uniformly by `0.92`; no existing family is
selectively removed or retuned. This preserves v16's relative prior while
reserving the externally measured sweet-spot mass for weekly demand. The
external result is not treated as proof of transfer: v18 must pass a matched
v16 A/B under this repository's model, token budget, and paired scoring path.

## Validation status

- Trainer contract verification passes with deterministic seed-0 digest
  `b6e7de66d08538a4…`.
- Three focused tests pass: config/default-mixture equality, deterministic
  finite non-negative generation, and the presence of period-7 structure plus
  the expected count-row share.
- An 8,192-series local generation check measured `8.30M` points/s versus
  `8.59M` for the archived v16 source (3.4% slower, still well above the
  contract reference in isolation).
- On 512 held-out weekly-demand series, the existing v16 checkpoint already
  beat persistence by 9.8% live-style (`0.2859` versus `0.3169`) despite never
  training on the dedicated family, so the added law is compatible rather than
  adversarial. This does not estimate the benefit of retraining on it.
- Forecast quality has not yet been claimed. The next gate is a matched,
  equal-budget v16/v18 GPU A/B on identical pool windows and seeds.

## Inherited v16 integrated-family correction

The matched v15 checkpoint was evaluated on 4,096 held-out v15 series per
family. Its integrated model score remained close to v14 (`0.5607` versus
`0.5526`), but the persistence score fell artificially from `0.5857` to
`0.2133`. The cause was not improved random-walk forecasting: approximately 8%
of a diagnostic sample received exactly zero persistence MASE.

V15 correctly stopped using future targets to estimate measurement ranges, but
applying a hard range calibrated from the first 512 observations to an
unbounded integrated path introduced a different error. Once a random walk
crossed that early range, censoring or clipped quantization could hold it at a
constant boundary through the entire 64-step target. This changed the process
from integrated dynamics into an artificial absorbing plateau.

V16 disables censoring and finite-range quantization for the integrated family
only. It retains sign inversion and causal sample-and-hold, and preserves all
range artifacts for bounded families where a finite sensor range is coherent.
Family weights and the underlying I(1), persistent-velocity, seasonal-
increment, and I(2) mixture are unchanged.

## Inherited v15 family-by-family rationale

The v14 checkpoint was evaluated on 4,096 held-out series per family with the
live-style diagnostic `sqrt(MWSQL * geometric_mean(MASE))`. This analysis uses
the score decomposition, not absolute geomean alone: smooth processes naturally
have large MASE because their one-step scale is tiny, while stochastic-volatility
and intermittent processes can improve MWSQL more than their conditional median.

The generator weights are unchanged. A family was modified only when the
evaluation and process definition identified a transferable mathematical issue.

## Family decisions

- **trend_seasonal_ar** — no change. It improved 77.4% over persistence, with
  strong gains in both MWSQL and MASE. Its stationary/modulated cadence mix and
  clean/noisy branches already provide useful diversity.
- **regime_shift** — no change. Level, volatility, and slope regimes are
  mathematically distinct and the model improved both score components.
- **multiplicative** — no change. This was the strongest family (86.9%); the
  expectation-centered Weibull construction preserves its latent signal.
- **ar2** — its partial-autorrelation parameterization guarantees stationarity,
  but filtering from an all-zero state did not sample that stationary law. V15
  now discards a 512-step burn-in.
- **integrated** — retains all v14 subtypes while reducing I(2) mass from 15% to
  10% and expanding the persistent-velocity branch from 25% to 35%. Its AR
  coefficient range is tightened to 0.97–0.999. Seasonal increments are
  calibrated on an initial prefix instead of the complete future path.
- **threshold_ar** — the SETAR recurrence is bounded/stable and gains are
  balanced across metrics, but its arbitrary initial state created a transient.
  V15 now discards a 256-step burn-in.
- **chaotic** — retains its burn-in, noisy/clean branches, and affine variation.
  Standardization now uses only an initial calibration prefix.
- **spectral_gp** — corrected. V14 generated an L-periodic path, making the first
  and last points almost identical (measured endpoint correlation `0.998`).
  V15 samples a `2L` circulant embedding and emits the first `L` points, retaining
  local kernel smoothness while removing the artificial endpoint adjacency
  (measured endpoint correlation approximately `-0.06`).
- **long_memory** — corrected further. V14 fixed pseudo-fBm and the L-point
  wrap, but its `1/f^beta` envelope remained only an approximation to fGn.
  V15 uses exact Davies-Harte covariance for 65% of rows and retains 35% of the
  piecewise-spectrum branch for realistic scale-dependent roughness.
- **ou_stochastic_vol** — no change. The recurrence and variance scaling are
  sound. Low point-score gain is expected from near-unit-root rows, latent
  switching, stochastic volatility, heavy tails, and unpredictable shocks.
  V14 already balances identifiable regimes with hard ultra-slow/rapid cases.
- **physical_sensors** — corrected. It inherits the non-circular GP component,
  and pressure diffusion now uses a per-step coefficient independent of total
  requested length. Its high absolute MASE remains a smooth-series scale effect.
- **seasonal_counts** — refined without changing its strong deterministic
  structure. V15 retains Poisson and iid gamma-mixed Poisson rows and adds a
  persistent lognormal-AR intensity branch so overdispersion is temporally
  coherent instead of always redrawn independently.
- **intermittent** — updated. V14 made occurrence stateful but positive sizes
  remained conditionally iid. V15 adds a persistent latent size process weakly
  coupled to occurrence intensity, while retaining gamma observation noise and
  integer outputs.
- **pulse_outlier** — remains only 1% of the corpus, but is now an explicit
  mixture of occurrence laws: 15% independent events for honest tail
  calibration, 35% jittered periodic events, 30% cyclic conditional intensity,
  and 20% stable self-exciting Hawkes-like clusters. The latter three provide
  context-dependent timing information without pretending that random
  innovations are exactly predictable. Magnitudes evolve persistently, and
  periodic/seasonal rows also carry a slow magnitude cycle.

## Cross-family corrections

- Whole-path mean, scale, range, and quantile estimates made emitted history
  depend on unseen targets. Normalization, censoring, and quantization now
  calibrate from the first 512 values and then hold those parameters fixed.
- Quantization now clips to its calibrated sensor range before rounding, so it
  has the requested finite number of output levels.
- Time reversal is restricted to trend/seasonal, multiplicative, spectral-GP,
  and long-memory laws. It is disabled for causal regime, SETAR, OU, sensor,
  count, intermittent, and shock-recovery paths, where reversal creates an
  invalid anti-causal process.

## Validation status

- Contract verification passes and generation is deterministic.
- All 16 v16 generator tests pass, including a regression test that requires
  active integrated innovations throughout the final 64 points.
- A controlled 512-series-per-family screen with the v15 checkpoint changed
  integrated performance from `-123.0%` to `+9.1%` versus persistence.
  Persistence geometric MASE returned from the artifactually low `1.879` to
  `12.053`; model geometric MASE was `12.682`.
- Every other family's screen result was unchanged, confirming that the RNG
  sequence and emitted distributions outside integrated range artifacts were
  preserved.
- This compatibility screen validates removal of the metric artifact, not v16
  training quality. A matched-seed v16 training run is required for a definitive
  comparison.

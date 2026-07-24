# cascade-fullctx-research-v18

A differentiated NumPy/SciPy full-context generator derived from the public
`custom-fullctx-v4` design.

Version 18 preserves cascade-v16's broad process coverage and adds one targeted
family from the public
[`j-test/dasadas_v18`](https://hub.hippius.com/models/j-test/dasadas_v18/main):
period-7 non-negative demand with weekday profiles, weekend dips, slow trend,
random-walk level, promotions with echoes, holiday dips, and a Poisson-count
branch.

The external generator measured an 8% demand weight as its exact sweet spot on
a multi-domain pool (`+6.39%` paired-cluster-bootstrap LCB); 10% overshot and
failed. This integration keeps that 8% family but does **not** copy the older
external base. Cascade-v16's stronger GP, exact fGn, OU, physical-sensor,
stateful count/intermittent, and conditional-pulse families retain their
relative composition, scaled proportionally into the remaining 92%.

The refinement retains hard subtypes rather than optimizing for an easy
synthetic self-score: integrated paths keep genuine I(1) and I(2) branches while
placing more mass on learnable persistent velocity, and 20% of OU rows retain
an ultra-slow near-unit-root rate while 15% retain rapid hidden switching. GP
and long-memory paths use 2L embeddings so emitted segments no longer end at an
artificial circular wrap boundary.

## Design

- Emits only 4096-point series, matching the 128-patch training/eval geometry
  and minimizing the one non-target context patch paid per series.
- Samples trend as total end-to-end excursion, so trend strength is independent
  of series length.
- Mixes fifteen families: trend/seasonality, structural regimes,
  multiplicative series, AR/integrated/nonlinear dynamics, smooth spectral GP,
  power-law long memory, physical sensors, seasonal counts, intermittent
  demand, outliers, and dedicated weekly non-negative demand.
- Adds an 8% period-7 demand family with randomized weekday phase/profile,
  optional two-day weekend depression, slow level evolution, sparse promotions
  and holiday dips, and a 35% Poisson-count branch.
- Replaces the predecessor's slow 48-pass random-Fourier GP with one batched
  inverse FFT and adds persistent/anti-persistent spectral paths.
- Adds a small regime-switching mean-reverting family with bounded clustered
  volatility, heavy-tailed innovations, transient shocks, and seasonal means.
- Executes the OU recurrence with SciPy's compiled linear filter instead of a
  4095-step Python loop.
- Executes AR(1) and AR(2) recurrences with compiled SciPy filters; local
  microbenchmarks were 4–12× faster than scanning time in Python.
- Adds batched cadence-seasonal Poisson and gamma-mixed Poisson counts with
  signed trends and decaying bursts, preserving positive integer structure.
- Adds weekday/weekend interactions to a minority of count rows and piecewise
  spectral slopes to long-memory rows, covering calendar effects and
  scale-dependent roughness without another FFT.
- Gives 40% of trend-seasonal rows a low-innovation mode, teaching sharp,
  stable periodic reconstruction while retaining noisy seasonal coverage.
- Anchors most mixture mass on the five-seed-tested full-context core
  (trend/seasonal, regimes, multiplicative, AR2, and integrated paths), while
  retaining each newer prior at a conservative share.
- Uses published TempoPFN ablations to strengthen OU/SDE-like dynamics,
  spectral/long-memory paths, and transient events without letting one family
  dominate the corpus.
- Adds slowly modulated amplitude and phase to a minority of seasonal
  components; stationary cycles remain the majority. Multiplicative paths use
  the same full cadence bank instead of a four-period subset.
- Extends structural/event coverage with piecewise-affine regime trends,
  decaying shock recovery, event plateaus, and genuine held-constant runs.
- Keeps independent outliers as only 15% of the 1%-weight pulse family; most
  pulse rows instead use jittered cadence, seasonal event intensity, or a
  stable Hawkes-like recurrence whose near-future hazard responds to history.
- Applies low-rate causal censoring, finite-range quantization, and
  sample-and-hold artifacts to bridge clean priors to real measurement
  pipelines. Time reversal is limited to laws valid under reversal.
- Extends seasonality through 365/672/730-step cycles and adds a small generic
  physical-sensor family (smooth, bounded, pressure-like, and skewed-positive)
  without adopting the competitor's private-pool-shaped weather weighting.
- Generates lazy 2048-row random-family chunks, keeping every stream prefix
  mixed while amortizing Python dispatch. Local profiling found this about 6%
  faster than 1024 rows; 4096 rows regressed slightly.
- Evaluates optional seasonal components only for active rows while preserving
  the fixed RNG draw sequence, and caches the fixed cadence sine/cosine basis,
  reducing trigonometric work without narrowing the prior.
- Draws jump, shock, and heavy-tail values only where those sparse branches are
  active instead of allocating dense arrays whose values are mostly discarded.
- Corrects the AR(2) family to remain stationary instead of adding a
  20-unit full-context drift, and scales I(2) paths by `L` rather than
  `sqrt(L)` so they no longer dominate I(1) paths.
- Burns in stationary AR(2) and SETAR recurrences rather than emitting their
  arbitrary zero-state transient.
- Calibrates standardization and measurement thresholds on the first 512
  observations, preventing generated history from depending on unseen targets.
- Excludes hard range artifacts from integrated paths so an unbounded process
  cannot become an artificial absorbing plateau. Sign inversion and causal
  sample-and-hold remain available.
- Uses ForecastPFN-style expectation-centered Weibull multiplicative noise,
  chaotic-map burn-in with optional observation noise, and ±5% period jitter
  on slowly evolving seasonal components.
- Samples an actual RBF/Rational-Quadratic covariance mixture through
  circulant FFT embedding, rather than labeling a generic heavy-tailed
  frequency envelope as Rational Quadratic.
- Samples stationary fGn with `beta=2H-1`, then cumulatively sums selected
  paths into mathematically consistent fBm.
- Replaces permanent volatility jumps with bounded mean-reverting log
  stochastic volatility calibrated to TempoPFN's OU ranges.
- Makes intermittent demand genuinely zero-inflated with a correlated
  occurrence state and positive integer sizes. Count artifacts preserve
  integer values and causal direction.

The v13 predecessor generated a five-run median
of 6.59M points/s versus 7.23M for the archived v12 baseline. The exact
composite covariance and richer observation models cost about 8.8% throughput,
but the candidate remains 78% above the mainnet contract's 3.7M reference in
isolation. End-to-end token completion also includes model training and stream
handoff. Re-measure v18 before treating those inherited numbers as current.

An end-to-end isolation run found that synchronous generation left training
blocked on data for 21.9% of its wall (`2.13M` point-passes/s). A deterministic
one-chunk producer thread now overlaps NumPy/SciPy generation with GPU work,
cutting data wait to 3.9% and raising training throughput to `2.43M`
point-passes/s (+14.4%) on the A100. The same short contract budget then
completed without a deadline hit. Cached rows reached `2.74M`, confirming the
remaining gap to the live L40S reference is mostly model/device throughput.

A controlled 120-second parameter screen then compared the baseline mixture
with seasonal-, spectral-, and dynamics-heavy variants under the same model,
pool, budget, and seeds. Dynamics-heavy won all three validation seeds, reducing
mean local synthetic-pool geomean from `0.19097` to `0.18431` (3.5%; lower is
better). The applied weights increase AR(2), integrated, threshold-AR, chaotic,
regime-shift, and OU coverage while reducing stationary seasonal, spectral, and
sparse/count families. This remains a directional local result, not a live
validator verdict.

## Local training result

V18 has not yet completed a full matched train. The external 8% family result
is strong evidence for the process and weight on its own base, not proof that
the same gain transfers unchanged to cascade-v16. Run an equal-budget paired
v16/v18 screen before deployment.

The v10 corpus was trained under the mainnet `chain.toml` contract on an A100
for the full 3-hour wall. It scored `0.13679` on the 64-window local synthetic
smoke pool (lower is better), improving from `0.15429` at the 30-minute heat
budget, while reaching 55% of the token budget. The optimized dynamics-heavy
v11 heat reached 59% (`3.90B / 6.66B`) and scored `0.15424`. The v12 prefetch
isolation test then cut data wait from 21.9% to 3.9% and raised end-to-end
throughput from `2.13M` to `2.43M` point-passes/s. These scores are directional
and are not live-validator verdicts; the A100 remains below the contract's
L40S-calibrated `3.7M` reference.

## Research basis

- TempoPFN supplies the regime-switching OU ranges, seasonal jitter, bounded
  stochastic-volatility rationale, and evidence for diverse step/spike priors:
  <https://arxiv.org/html/2510.25502v1>.
- ForecastPFN motivates multiplicative Weibull noise whose center does not
  bias the underlying signal: <https://arxiv.org/abs/2311.01933>.
- Chronos KernelSynth motivates RBF and Rational-Quadratic covariance
  composition: <https://arxiv.org/html/2403.07815v1>.
- Fractional-process literature gives `beta=2H-1` for fGn and `beta=2H+1`
  for fBm: <https://pmc.ncbi.nlm.nih.gov/articles/PMC3947294/>.
- Intermittent-demand state-space work separates occurrence probability from
  positive size: <https://mpra.ub.uni-muenchen.de/82487/>.
- Hawkes-process literature motivates a conditional event intensity increased
  by recent arrivals: <https://arxiv.org/abs/2405.10527>.
- Heavy-tail forecasting work motivates retaining unpredictable innovations
  for likelihood and tail calibration rather than exact timing prediction:
  <https://arxiv.org/abs/2106.10952>.
- Proper-scoring-rule theory motivates honest predictive distributions for
  irreducible event uncertainty: <https://doi.org/10.1198/016214506000001437>.

The trend, regime-step, SETAR, physical-sensor, seasonal-count, and pulse
families retain their prior broad parameter ranges where no source establishes
a universal Toto2-optimal distribution. Their implementations were checked for
stability and structural validity; changing every number would be false
precision. V18 scales the complete v16 mixture uniformly to fund the externally
validated 8% weekly-demand prior.

## Validate

```bash
python -m cascade.miner.cli verify ./generators/cascade-v18 --chain-toml chain.toml
```

Contract validity and CPU throughput do not establish forecasting quality.
Run a production-faithful GPU A/B score against the current king before
deploying this candidate.

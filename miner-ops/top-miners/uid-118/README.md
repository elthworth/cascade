# cascade-fullctx-spectral-v7

A differentiated NumPy/SciPy full-context generator derived from the public
`custom-fullctx-v4` design.

## Design

- Emits only 4096-point series, matching the 128-patch training/eval geometry
  and minimizing the one non-target context patch paid per series.
- Samples trend as total end-to-end excursion, so trend strength is independent
  of series length.
- Mixes fifteen families: trend/seasonality, structural regimes,
  multiplicative series, AR/integrated/nonlinear dynamics, smooth spectral GP,
  power-law long memory, physical sensors, seasonal counts, intermittent
  demand, and outliers.
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
- Extends seasonality through 365/672/730-step cycles and adds a small generic
  physical-sensor family (smooth, bounded, pressure-like, and skewed-positive)
  without adopting the competitor's private-pool-shaped weather weighting.
- Generates lazy 1024-row random-family chunks, keeping every stream prefix
  mixed while amortizing Python dispatch.

## Validate

```bash
python -m cascade.miner.cli verify ./cascade-v2 --chain-toml chain.toml
```

Contract validity and CPU throughput do not establish forecasting quality.
Run a production-faithful GPU A/B score against the current king before
deploying this candidate.

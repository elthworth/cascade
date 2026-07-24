# cascade-fullctx-spectral-v4

A differentiated NumPy/SciPy full-context generator derived from the public
`custom-fullctx-v4` design.

## Design

- Emits only 4096-point series, matching the 128-patch training/eval geometry
  and minimizing the one non-target context patch paid per series.
- Samples trend as total end-to-end excursion, so trend strength is independent
  of series length.
- Mixes fourteen families: trend/seasonality, structural regimes,
  multiplicative series, AR/integrated/nonlinear dynamics, smooth spectral GP,
  power-law long memory, physical sensors, intermittent demand, and outliers.
- Replaces the predecessor's slow 48-pass random-Fourier GP with one batched
  inverse FFT and adds persistent/anti-persistent spectral paths.
- Adds a small regime-switching mean-reverting family with bounded clustered
  volatility, heavy-tailed innovations, transient shocks, and seasonal means.
- Executes the OU recurrence with SciPy's compiled linear filter instead of a
  4095-step Python loop.
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

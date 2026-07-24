# custom_miner — the generator you submit

A cascade `DataGenerator` (`generator.Generator`) that turns one integer `seed`
into a corpus of univariate float series. The subnet holds the model, seeds, and
compute identical between king and challenger, so **the only thing that moves
your score is the distribution this generator emits.**

## The idea: compete on prior *diversity + realism*

The reference/genesis generators win by covering the shapes a forecaster must
handle. This one mixes **10 process families**, each fully seed-deterministic and
vectorised:

| family | what it contributes |
|--------|--------------------|
| `trend_seasonal_ar` | level + slope + multi-seasonal sinusoids + AR(1) noise |
| `regime_shift` | piecewise level & variance regimes (structural breaks) |
| `multiplicative` | positive level × seasonal factor × multiplicative noise |
| `ar2` | AR(2), stationarity-guaranteed (Levinson-Durbin), incl. near-unit-root |
| `integrated` | I(1)/I(2) random walks with drift |
| `threshold_ar` | SETAR — regime-switching nonlinear recurrence |
| `chaotic` | bounded chaotic maps (logistic / sine) |
| `rff_gp` | smooth GP-like samples via random Fourier features |
| `intermittent` | zero-inflated / intermittent demand |
| `pulse_outlier` | smooth base + sparse pulses/outliers + flat gaps |

The mixture weights (`family_weights` in [`config.json`](config.json)) were tuned
with `local_validator` against a broad multi-domain eval: strong seasonal
coverage matters (most real series are seasonal) while every family keeps
meaningful mass so the prior generalises across non-seasonal domains too.

## Contract compliance (what `cascade verify` checks)

* **Determinism** — every value comes from one `np.random.default_rng(seed)` in a
  fixed draw order → byte-identical corpus at a fixed seed (the property the
  trainer audits by building twice).
* **Code-only** — no shipped weights, no network, no clock; imports are numpy +
  `cascade.interface` only (on the allowlist, clear of the static-guard blocklist).
* **Bounded + finite** — 1-D float64 series, length in `[min_length, max_length]`,
  finite; `_sanitize` is the hard backstop.
* **Fast** — vectorised per family (a batched time-axis recurrence, never a
  per-series Python loop), so draining the full `corpus_n_series` (16384) stays
  well under `max_generate_seconds`.

Verify it yourself:

```bash
python -m cascade.miner.cli verify custom/custom_miner
# OK: generator would be accepted by the trainer.
#   corpus_digest (seed=0): 8ecf44e7ebbb601f…  [deterministic]
```

## How to make it yours

1. **Tune the mixture** — edit `family_weights` in `config.json` (no code change),
   then re-run `python -m custom.local_validator` and watch the LCB / per-domain
   win-rate move. This is the cheapest, safest lever.
2. **Add/replace a family** — add a `_yourfamily(rng, n, L) -> (n, L)` builder,
   register it in `_FAMILIES` / `_DEFAULT_WEIGHTS` / the `builders` tuple. Keep it
   vectorised and seed-deterministic; `_sanitize` guarantees finiteness.
3. **Re-verify + re-A/B** every change: `verify` must stay green and you want the
   local KOTH verdict trending up before you deploy.

## Files

```
custom_miner/
  generator.py      class Generator(DataGenerator) — the mixture-of-priors
  config.json       name/description, length band, family_weights
  requirements.txt  hash-locked deps (numpy only). The trainer only FORMAT-checks
                    this file (it does not reinstall — the sandbox ships the
                    allowlisted stack), so the placeholder zero-hash is accepted,
                    as in the shipped reference generators. Real hashes optional.
```

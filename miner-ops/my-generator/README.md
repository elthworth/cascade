# aurora-mix-v3 — cascade challenger generator

A fifteen-family synthetic time-series prior redesigned around **TempoPFN's
full pipeline** (arXiv 2510.25502). numpy + scipy only — no torch, no vendored
package, one self-contained `generator.py`.

**v3 over v2 — the streaming-prefix fix and the evidence rebalance:**

1. **Stratified interleaved emission.** The trainer's streaming feed modes
   (`stream_cpu`/`stream_gpu`) consume only a token-budget *prefix* of the
   stream. v1/v2 emitted family-blocked, so only the first family
   (`kernel_gp`) ever trained — the deployed `valor/aurora-mix` was
   effectively a pure-GP generator (heat rank 3, rel 1.143). v3 interleaves
   (the same stratified scheme the reigning king `ares-v3` uses), so every
   prefix carries the configured mixture — verified: all 15 families present
   at their configured shares in a 5% prefix.
2. **Evidence-based weights** (`config.json`, no code change needed to
   iterate): ~45% anchored on the two throne-proven families —
   `trend_seasonal` 0.30 (the king holds 4 rounds with 100% tuned
   trend×seasonality composition) and `kernel_gp` 0.15 (our GP out-ranked the
   former king smoothgp) — ~22% on real-world-shape robustness
   (calendar/GARCH/intermittent), small live shares for the untested
   TempoPFN-gap trio. Rationale: the KOTH bootstrap-LCB rewards per-window
   *consistency* (this round's challenger was 14% better on mean and still
   lost with LCB −0.356), so uniform decency beats occasional brilliance.
3. Post-transform degeneracy repair: count-rounding can no longer flatten a
   degenerate-repaired series back to zero-std.

## Study the live king first (`cascade fetch`)

Every committed generator is public and content-addressed. Pull the reigning
king and diff your prior against *the actual code that is winning right now*:

```bash
cascade fetch king --network test --chain-toml chain.testnet.toml   # → ./fetched-king-uidN
python local_validator/validate.py --repo my_generator --king ./fetched-king-uidN
```

`--king` accepts any generator dir, so the `compare` stage scores your
feature-space coverage against the *live* king, not just the shipped
`base_generator`. You win by improving on the visible best — a byte-identical
copy is dropped before it trains, so you must genuinely beat it.

## Strategy vs the king (the TempoPFN redesign)

The king vendors ten TempoPFN families but **drops the audio-inspired group**
(pyo runs an audio server and seeds via `hash()` — non-deterministic, not on
the allowlist) and leaves TempoPFN's augmentation layer mostly unused. v2
attacks exactly that gap:

1. **Keeps the highest-signal families** (per the TempoPFN ablation), with its
   own implementations: compositional GP/kernel priors (KernelSynth-style,
   coarse-grid Cholesky + cubic upsampling) and rich trend × multi-seasonality
   × structured-noise composition.

2. **Reimplements the king's missing TempoPFN audio priors deterministically**
   (pure numpy/scipy):

   | family | TempoPFN prior it restores | what it teaches |
   |---|---|---|
   | `rhythm` | Stochastic Rhythms | quasi-periodic event trains: tempo drift, accent bars, swing, dropouts, pluck/percussive/bump kernels |
   | `fractal_multi` | Multi-Scale Fractals | piecewise-slope spectra, Weierstrass sums, multiplicative cascades (multifractal volatility) |
   | `net_diffusion` | Network Topology | forced diffusion on random directed graphs — propagation, echoes, superposition (CauKer flavour, no networkx) |

   Financial Volatility (TempoPFN's fourth audio prior) is already covered by
   `regime_garch`.

3. **Ports TempoPFN's augmentation layer** (the vendored king has the params
   but doesn't use most of it): per-series **time-warp**, **damping
   envelopes**, and **spike injection** in post-processing, plus batch-level
   **splice transitions** (two series crossfaded at a changepoint —
   TempoPFN's `transition_ratio`) and the TSMixup-style `mixup` family.

4. **Keeps the seven regime classes the king does not emit** (from v1):

   | family | real-world shape it teaches | literature |
   |---|---|---|
   | `chaotic` (Lorenz/Rössler/Duffing/Mackey-Glass/logistic/Hénon) | nonlinear dynamics, traffic/weather transfer | DynaMix (arXiv 2505.13192) |
   | `fgn` (fractional Gaussian noise / fBm) | long-range dependence | classic Hurst |
   | `regime_garch` (Markov-switching AR + GARCH) | volatility clustering, regime shifts | econ/finance |
   | `intermittent` (zero-inflated counts) | retail demand, sparse events | Croston |
   | `calendar` (daily profile × weekly factors × holidays) | energy/traffic load curves | — |
   | `growth` (logistic/Gompertz/lifecycle) | adoption curves, saturation | Bass |
   | `bursts_anomaly` (subcritical Hawkes + anomaly-injected AR) | bursty events, robustness to outliers/level shifts | — |

5. **Scale/offset/quantisation diversity** in post-processing: log-uniform
   scales over five decades, offsets, integer quantisation (count-like series),
   softplus positivity — the from-scratch model must learn scale-robustness
   from the corpus alone.

6. **Length mixture biased long** (55 % of crops in [768, 2048]) — more signal
   per series while keeping short-series coverage.

## Determinism

The corpus is a pure function of `(seed, n_series)`. Every RNG is a
`np.random.default_rng(SeedSequence([seed, tag, index]))`; there is no global
RNG, no torch, no wall-clock, no `hash()`. Verified by
`cascade verify` (two full draws, digest-compared) and by a cross-process
digest smoke (~650k points/s on a CPU box — well above the trainer's 185k
tokens/s reference rate, so `stream_cpu` never starves the GPU).

## Verify / test / score

```bash
# from the cascade repo root
python local_validator/validate.py --repo my_generator          # CPU gauntlet (valid + regime coverage vs king)
python -m pytest my_generator/tests -q                          # contract tests
cascade verify ./my_generator                                   # the real pre-deploy check

# the REAL signal (GPU, upstream `cascade score`): does it actually beat the king?
bash local_validator/vps_score.sh                               # on a GPU VPS — A/B train+score vs the fetched king
```

`cascade score` trains the fixed Toto2-4M on your corpus at the heat budget and
scores it (lower geomean = better), so you learn whether you clear the 0.02 win
margin *before* spending a hotkey. See `local_validator/README.md` for the GPU
VPS setup + a sufficient Vast.ai spec.

## Deploy (when ready)

```bash
cascade deploy ./my_generator --hub-repo <namespace/aurora-mix> \
    --wallet-name <coldkey> --wallet-hotkey <hotkey>
# optional outage fallback (Hub is always tried first; --hf-repo alone is refused):
#   --hf-repo <hf-namespace/aurora-mix>   # needs HF_TOKEN; commits repo@hf:<sha>
```

Remember: one hotkey = one submission, for life. Deploy only after the local
gauntlet and a GPU `cascade score` A/B that clears the 0.02 margin vs the
current king (`bash local_validator/vps_score.sh`).

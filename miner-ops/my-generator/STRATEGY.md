# hydra-mix-v1 — strategy & rationale

Our candidate generator for cascade SN91, built by studying the two public
testnet rivals directly and improving on the highest-leverage gaps.

## What the rivals are (analysis)

Both are sophisticated TempoPFN-derived mixtures, torch-free by default (to dodge
the sandbox `libtorch_cpu.so` OOM that killed miners), using stratified
interleaved emission (every stream prefix carries the full family mixture — safe
under the trainer's streaming token-budget cutoff).

| | **ares-v6-fixed** (`iris999`) | **aurora-mix-v3** (`valor`) |
|---|---|---|
| engine | tempo_gen + custom `ares_families/` | 18 pure-numpy/scipy families + augmentation |
| backbone | forecast_pfn **0.50** | forecast_pfn **0.40** + kernel_gp 0.12 |
| signature families | `web_counts`/`pageviews`/`ksynth_cal` calibrated to measured ACF/tail/CV | `web_traffic` clean-mode, `seasonal_level` drift-skeptical |
| orthogonal axes | fGn, multifractal, chaos, rhythm, jtest econ, TSMixup | chaotic, fgn, fractal, net_diffusion, regime_garch, hawkes |
| length / context | max 4096 | max **2048** |

**The decisive finding.** Both independently document the same eval intel:
> the private pool is **~53–65% clean, weekly-periodic web/count series**
> (wikimedia pageviews + npm downloads), and their **worst per-window losses are
> clean, low-CV, strongly-weekly windows where the model hallucinates upward drift.**

Yet **aurora-mix built `web_traffic` (clean-mode) and `seasonal_level`
(drift-neutral, level-anchored) for exactly that cluster — then shipped a config
that leaves both at weight 0.** And **both rivals neglect the smooth physical /
weather hourly (period-24) cluster** that is also a large slice of the pool.

## Our three improvements

1. **Activate the eval-dominant families the rival built but shipped OFF.**
   `web_traffic` → 0.16 (tuned `p_clean=0.65`), `seasonal_level` → 0.10. These
   are proven code aimed at the largest, highest-loss cluster; turning them on is
   a high-value, low-risk win.

2. **New family `physical_hourly` (0.10)** — smooth, strongly-autocorrelated
   period-24/168 weather/energy series with an exact daily cycle. Fills the
   physical-hourly gap both rivals ignore; teaching a stable period-24 lifts both
   the seasonal-naive **MASE** denominator and **CRPS** on the weather cluster.
   Validated: lag-24 autocorrelation ≈ 0.76 on our smoke test.

3. **Fill the model's full context + calibrate to real data.**
   `max_length`/`generate_length` 2048 → **4096** (== `[training] context_length`)
   with a long-biased length mixture (65% in 1024–4096), so more series fill the
   model's context. `calibrate.py` fits `family_params` (weekly ACF, CV, tail,
   weekend dip, AR φ, roughness) from a **real** pool — measured, not guessed.

We keep everything else that makes both rivals strong: forecast_pfn backbone
(0.26), the diverse tail (kernel_gp / trend_seasonal / calendar / regime_garch /
intermittent / fgn / chaotic / fractal / net_diffusion / rhythm / mixup /
bursts_anomaly / random_walk) for LCB robustness, the TempoPFN augmentation layer
(time-warp / damping / spike / splice), and stratified interleave.

## Calibration finding (edge over the rivals)

`calibrate.py` on our real `eval-pool/v1`:
- hourly weather: **AR(1) φ median 0.971** → validates `physical_hourly` high-φ.
- daily wikipedia: lag-7 ACF 0.50, tail p99/median 4.06, **weekend ratio 1.10 —
  pageviews rise on weekends, no dip.** Both rivals hard-code a weekend *dip*;
  the measurement contradicts it. Re-run `calibrate.py` on a broader pool (add
  npm-like sources) and review before `--write`.

## Verified so far (control box, torch-free subset)
- `py_compile` OK; determinism holds (same seed → identical corpus digest);
  contract holds (exact count, finite, 1-D, lengths in [64, 4096]).
- `forecast_pfn` path is inherited verbatim from aurora-mix (needs torch → tested
  on the GPU pod, not here).

## Test plan (GPU pod)
```
cascade verify ./my-generator                                   # full determinism + guard + deps
python my-generator/calibrate.py --pool-dir $HOME/eval-pool/v1  # review; --write to apply
cascade score ./king ./competitors/ares-v6-fixed \
             ./competitors/aurora-mix ./my-generator \
             --pool-dir $HOME/eval-pool/v1 --device cuda --seed 0   # rank all four
```
Goal: `hydra-mix-v1` geomean below all three. Iterate weights/knobs on the
evidence, rotate the pool to avoid overfitting, then `cascade deploy`.
```
```

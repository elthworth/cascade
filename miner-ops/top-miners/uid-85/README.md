# custom_miner v3 CANDIDATE — de-trended (worker_2)

A calibrated evolution of v2 that fixes v2's largest measured divergence from real data. It is a
**strong lead, not a confirmed ship** — read the caveats.

## The insight (data-driven)
A statistical fingerprint (`scratchpad/statgap.py`) showed v2's corpus has **far too much linear
trend**, and the excess **grows with series length** (v2's `slope*t`): trend-strength median ~0.42
at length 1536 vs real weather ~0.026. v2 was teaching the model to over-extrapolate trends.

## The change (vs v2 — a clean 2-family diff)
`_trend_seasonal_ar` and `_multiplicative` draw the **total trend excursion** over the series
directly (`exc * t/(L-1)`) instead of `slope*t`. Effect: the trend is **~16x weaker than v2's and
bimodal** (~75% low-trend + ~25% trending). *(Note: because builders run at max_len then crop, this
is v2-with-smaller-bimodal-slope, not true per-series normalization.)* Persistence, noise, and all
family weights are **identical to v2**. Tunable via config (`tr_exc_lo/hi`, `gr_exc_lo/hi`,
`tr_hi_frac`). Deterministic; `cascade verify` passes.

## Measured results (real Toto2, multi-seed pooled, honest feed-level clustering, real weather)
| regime | v3 vs v2 |
|---|---|
| ctx 1024 / gen-max 2048 | **+12.9%, LCB +0.077 — robust WIN** |
| ctx 1536 / gen-max 4096 (production length) | **+8.4%, LCB +0.056 — robust WIN** |

Fingerprint: v3 trend-strength median at length 1536 is **0.008** (near real 0.026) vs v2's 0.42.
Unlike a first slope-based variant (which won +14% at ctx1024 but collapsed to a tie at gen-max
4096), this holds across both regimes tested.

## Caveats (from adversarial review — DO NOT ship without resolving)
1. **Not tested at production CONTEXT 4096.** The win **shrinks as context grows** (+12.9% → +8.4%
   from ctx 1024 → 1536); extrapolated to 4096 it could narrow toward the +2% margin. Production
   uses ctx 4096; I could only reach ctx 1536.
2. **It is a LOW-TREND SPECIALIST — it LOSES multi-domain pools.** On the synthetic 8-domain
   pool at ctx 1536 it scores **−22%**: it wins low-trend domains (energy +23%, traffic +21%,
   weather +17%) but loses trending ones (web −47%, econ −10%, retail −2%). The real subnet pool
   is multi-domain; v3 nets positive only if that pool is low-trend-dominated (plausible — it is
   weather + pageviews heavy — but unconfirmed). Its big wins are all on genuinely low-trend REAL
   data (weather); its losses are on trending data.
3. **Decisive test:** v3 at generation & eval context **4096** on the **full mixed real pool**,
   per-seed win rate. Needs the operator's real private pool / more GPU than this run had.

**Recommendation:** keep v2 (worker_1) live; treat v3 as the leading candidate and A/B it against
the real pool at ctx 4096 before any deploy. Found via a rigorous, adversarially-reviewed search
that also rejected TempoPFN+CauKer, CauKer/GP injections, and 6 reweights (none beat v2).

## UPDATE — production-geometry re-test (ctx 4096, 2026-07-13)
Re-tested at the operator's new production geometry (context 4096, gen-max 4096, after the
max_length 2048→4096 change). The big short-context wins FADE: v3 vs v2 context trajectory =
+12.9% (ctx1024) → +8.4% (1536) → +1.9% (2048) → **+3.4% (ctx4096, 5/5 seeds positive but
LCB +0.005 < the 0.02 margin = TIE)** on real weather. Per-domain at ctx4096: still a LOW-TREND
SPECIALIST — wins energy +16%/traffic +15%/weather +3%, loses web −47%/finance −6%/retail −6%
(synthetic multi-domain OVERALL −30%). VERDICT: v3 does NOT robustly beat v2 at production; it's a
consistent hair-better-on-low-trend, worse-on-trending trade-off. **v2 (worker_1) remains the
multi-domain choice.** v3's only unambiguous edge: numpy generation at 3.16M tok/s (17x the 185k
wall-clock cap), immune to the new deadline_hit truncation that threatens the GP/CauKer base king.

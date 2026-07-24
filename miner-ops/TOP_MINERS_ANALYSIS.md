# Top Miners Strategy Analysis — CASCADE SN91

**Analysis Date**: 2026-07-24  
**Mainnet Dashboard**: https://dashboard.cascadesub.net/

## Executive Summary

Analyzed the top 2 generators on mainnet by emission:
- **UID 131** (jan/cascade9) — CURRENT KING — 103.58 TAO
- **UID 64** (j-test/2_1_v5) — 44.02 TAO

Both generators converge on the same **core winning strategy**:
1. **All-4096 length** (no variable length) = perfect target efficiency
2. **Dynamics-heavy mixture** (~57% weight on persistent/structural families)
3. **Length-invariant drift** (excursion-based, not slope·t)
4. **Heavy near-unit-root AR** (0.15 AR2 + 0.12-0.15 integrated = 0.27-0.30 total)

The king adds: 4 extra families, FFT optimizations, prefetching thread, measurement artifacts.

---

## 1. UID 131: jan/cascade9 (CURRENT KING)

**Digest**: `sha256:c2d0d6b1b0eccafbc47930c1e86f973daeea2aeeda62529329c05608a1bdef1d`  
**Name**: cascade-fullctx-spectral-v12  
**Emission**: 103.58 TAO

### Architecture

```
14 families | all-4096 | NumPy/SciPy | 2048-row chunks | prefetching thread
```

### Family Weights (dynamics-heavy A/B winner)

| Family | Weight | Category |
|--------|--------|----------|
| **ar2** | 0.15 | Persistent dynamics |
| **regime_shift** | 0.12 | Structural |
| **integrated** | 0.12 | Random walk |
| **trend_seasonal_ar** | 0.12 | Baseline seasonal |
| **ou_stochastic_vol** | 0.10 | Mean-reverting SDE |
| **multiplicative** | 0.08 | Seasonal multiplicative |
| **threshold_ar** | 0.08 | Nonlinear regime |
| **spectral_gp** | 0.07 | Smooth GP (FFT-based) |
| **long_memory** | 0.06 | Power-law spectra |
| **chaotic** | 0.04 | Chaotic maps |
| **physical_sensors** | 0.02 | Bounded/skewed |
| **seasonal_counts** | 0.02 | Poisson/gamma counts |
| **intermittent** | 0.01 | Zero-inflated |
| **pulse_outlier** | 0.01 | Events/outliers |

**Dynamics total**: ar2 + integrated + threshold_ar + regime_shift + ou_stochastic_vol = **0.57**

### Key Innovations

1. **FFT Spectral GP**: Replaced 48-pass RFF with one batched inverse FFT
   - Added persistent/anti-persistent spectral paths
   - Faster + power-law long-memory coverage

2. **SciPy lfilter for AR recurrences**: 4–12× faster than Python loops
   - AR(1) and AR(2) use compiled filters
   - OU recurrence uses scipy.signal.lfilter

3. **Prefetching thread**: Overlaps NumPy/SciPy generation with GPU training
   - Reduced data blocking: 21.9% → 3.9%
   - Raised throughput: 2.13M → 2.43M point-passes/s (+14.4%)
   - One-slot queue, deterministic draw order preserved

4. **Measurement artifacts layer**: Applied to all families
   - Reversal (6%), sign inversion (4%), censoring (6%), quantization (7%), sample-and-hold (4%)
   - Bridges clean priors to real measurement pipelines
   - Preserves positivity for count/magnitude families

5. **Extended seasonality**: 365/672/730-step periods
   - Full 4096-point contexts can identify long cycles
   - 17 cadence options vs j-test's 10

6. **Modulated seasonality**: 35% of seasonal components get slow amplitude/phase drift
   - TempoPFN's strongest non-SDE ablation was complex-seasonality
   - Stationary baseline remains well-represented

7. **Cached seasonal basis**: `@lru_cache` on sine/cosine waves
   - Reduces trigonometric work without narrowing prior
   - Draws parameters for all rows, evaluates only active rows

8. **2048-row chunks**: Local profiling sweet spot
   - ~6% faster than 1024 rows
   - 4096 rows regressed slightly

### A/B Testing Discipline

**Controlled 120-second parameter screen**:
- Compared baseline vs seasonal-heavy vs spectral-heavy vs **dynamics-heavy**
- Same model, pool, budget, seeds
- **Dynamics-heavy won on all 3 validation seeds**
- Mean geomean: 0.18431 vs baseline 0.19097 (3.5% improvement)

Applied weights increase: AR(2), integrated, threshold-AR, chaotic, regime-shift, OU  
Reduced: stationary seasonal, spectral, sparse/count families

**Note**: "A richer 'composite' family was implemented and measured — more realistic, but scored WORSE on proxy pool. Did not ship."

### Throughput

- Isolated generation: **11.38M points/s** (v12 after optimization)
- End-to-end with prefetch: **2.43M point-passes/s** (A100)
- Well above mainnet contract: **3.7M reference** (L40S-calibrated)

### Length-Invariant Drift

```python
# v3: bimodal trend EXCURSION (not slope)
exc = np.where(_hi, rng.normal(0.0, exc_hi), rng.normal(0.0, exc_lo))
tn = t / max(L - 1, 1)  # normalized time [0, 1]
series = level + exc * tn + seasonal
```

Trend strength is **independent of series length** (real trend ~0.02, length-invariant).  
v2's slope·t was measured **~16× too strong** vs real data at production lengths.

### Config Knobs

```json
{
  "tr_exc_lo": 0.4, "tr_exc_hi": 2.5,
  "gr_exc_lo": 0.3, "gr_exc_hi": 1.5,
  "sa_clean_frac": 0.4,  // 40% of trend_seasonal_ar get low noise
  "sa_clean_lo": 0.02, "sa_clean_hi": 0.12
}
```

---

## 2. UID 64: j-test/2_1_v5

**Digest**: `sha256:38dda53cff4a65f883b2686a4a27fd4597c69f580f20c1d4254e323cfbff11e4`  
**Name**: custom-longctx-lrfam-v5  
**Emission**: 44.02 TAO

### Architecture

```
10 families | all-4096 | NumPy-only | 256-row chunks | no prefetch
```

### Family Weights (long-range reweight)

| Family | Weight | Category |
|--------|--------|----------|
| **trend_seasonal_ar** | 0.20 | Baseline seasonal |
| **regime_shift** | 0.16 | Structural |
| **ar2** | 0.15 | Persistent dynamics |
| **integrated** | 0.15 | Random walk |
| **multiplicative** | 0.13 | Seasonal multiplicative |
| **rff_gp** | 0.11 | Smooth GP |
| **threshold_ar** | 0.04 | Nonlinear regime |
| **chaotic** | 0.03 | Chaotic maps |
| **intermittent** | 0.02 | Zero-inflated |
| **pulse_outlier** | 0.01 | Events/outliers |

**Long-range total**: integrated + ar2 + regime_shift + rff_gp = **0.57**

### Strategy: Long-Range Structure Hypothesis

From README:
> "v4's all-4096 target-efficiency PLUS a family reweight toward long-range-structure priors (integrated random-walk, regime-shift, persistent AR2, GP, multi-seasonal) and **away from choppy short-range ones** (chaotic/intermittent/pulse) that waste the 128-patch context."

**Hypothesis**: At a 4096-token context, the model benefits from **long-range learnable structure**.

### Comparison to Default Weights (custom-miner baseline)

```
Long-range families INCREASED:
  integrated:      0.08 → 0.15  (+88%)
  ar2:             0.10 → 0.15  (+50%)
  regime_shift:    0.10 → 0.16  (+60%)
  rff_gp:          0.07 → 0.11  (+57%)

Short-range families DECREASED:
  chaotic:         0.05 → 0.03  (-40%)
  intermittent:    0.03 → 0.02  (-33%)
  pulse_outlier:   0.03 → 0.01  (-67%)
```

### Minimalist Design

- **NumPy-only**: No SciPy, no torch
- **10 families**: 4 fewer than jan/cascade9
- **No measurement artifacts**
- **No prefetching**
- **256-row chunks** (vs 2048)
- **Simpler AR loops**: Pure Python time-stepping (not lfilter)

### Detrended

Config note: "De-trended (from v3)"  
Same length-invariant excursion approach as jan/cascade9:

```python
tn = t / max(L - 1, 1)
series = level + exc * tn + seasonal
```

---

## 3. Convergent Winning Strategy

Both generators independently converged on:

### (A) All-4096 Length

```json
"min_length": 4096,
"max_length": 4096
```

**Why it wins**:
- Training uses 128 patches × 32 tokens/patch = 4096 context
- Variable-length series pay 1 non-target patch per series (wasted context)
- All-4096 = 128 target patches = **perfect token efficiency**
- Full context enables long-cycle identification (365/672/730 periods)

### (B) Dynamics-Heavy Mixture (~57% weight)

| Generator | ar2 | integrated | regime_shift | threshold_ar | OU/other | **Total** |
|-----------|-----|------------|--------------|--------------|----------|-----------|
| jan/cascade9 | 0.15 | 0.12 | 0.12 | 0.08 | 0.10 (OU) | **0.57** |
| j-test/2_1_v5 | 0.15 | 0.15 | 0.16 | 0.04 | — | **0.50** |

Including long-range GP:
- jan: +0.07 spectral_gp +0.06 long_memory = **0.70**
- j-test: +0.11 rff_gp = **0.61**

**Both generators put majority mass on persistent/structural families.**

### (C) Heavy Near-Unit-Root AR

| Generator | AR2 | Integrated | **Total** |
|-----------|-----|------------|-----------|
| jan/cascade9 | 0.15 | 0.12 | **0.27** |
| j-test/2_1_v5 | 0.15 | 0.15 | **0.30** |

This is the **finance/econ coverage** that earlier analysis identified as critical.

j-test's earlier comment (custom-miner code):
> "Motivated per weak domain:  
> finance (45%) → ar2 0.10→0.12, integrated 0.08→0.10 (near-unit-root/I(1))"

j-test/2_1_v5 pushes this **even further**: ar2 0.12→0.15, integrated 0.10→0.15

### (D) Length-Invariant Drift

Both use **excursion-based trend**, not slope·t:

```python
exc = rng.normal(0.0, exc_range)
tn = t / max(L - 1, 1)  # normalized [0, 1]
trend = exc * tn
```

**Why**: Real trend strength is ~0.02 and length-invariant.  
v2's `slope * t` grew with L and was measured **~16× too strong**.

---

## 4. Key Differences: King vs Challenger

| Aspect | jan/cascade9 (King) | j-test/2_1_v5 | Winner? |
|--------|---------------------|----------------|---------|
| Families | 14 | 10 | King (diversity) |
| Dependencies | NumPy + SciPy | NumPy-only | j-test (simpler) |
| AR filters | SciPy lfilter (4–12× faster) | Python loops | King (throughput) |
| GP method | FFT spectral (1 pass) | RFF (48 passes) | King (faster + long-memory) |
| Prefetching | Yes (+14.4% throughput) | No | King (GPU overlap) |
| Chunks | 2048 rows | 256 rows | King (measured sweet spot) |
| Measurement artifacts | Yes (6 types) | No | King (realism) |
| Seasonal periods | 17 options (up to 730) | 10 options (up to 336) | King (long-cycle coverage) |
| Modulated seasonality | 35% of components | All stationary | King (TempoPFN ablation) |
| A/B testing | Documented 3-seed screen | Not documented | King (rigor) |
| Emission | **103.58 TAO** | 44.02 TAO | **King wins** |

**Conclusion**: jan/cascade9 wins through:
1. **More families** (14 vs 10) = broader prior coverage
2. **Performance optimizations** (SciPy, FFT, prefetch, caching)
3. **Measurement artifacts** (bridges to real pipelines)
4. **A/B-tested mixture** (dynamics-heavy validated on 3 seeds)

j-test's minimalism (NumPy-only, 10 families) keeps it competitive but falls behind on diversity and throughput.

---

## 5. What Both Generators Do NOT Have

**Neither generator uses**:
- TempoPFN augmentation (unlike aurora-mix/ares-v6)
- Torch-based families
- Variable-length mixture
- Web_traffic / seasonal_level families (our advantage!)
- Physical_hourly (our calibrated 24/168 family)

**Opportunity**: Our my-generator has families they lack, but we're missing their core strength — **dynamics-heavy weight** and **all-4096 efficiency**.

---

## 6. Insights for Our Generator

### What We Should Adopt

1. **All-4096 length** (no variable)
   ```json
   "min_length": 4096,
   "max_length": 4096
   ```
   - Perfect token efficiency (128 target patches)
   - Our current 64–4096 wastes context on short series

2. **Increase dynamics weight to ~0.55–0.60**
   - Current: ar2 (?) + integrated (?) = need to check our weights
   - Target: ar2 0.15 + integrated 0.12–0.15 + regime_shift 0.12–0.16

3. **Reduce TempoPFN weight** (currently 0.26)
   - King uses 0 TempoPFN, j-test uses 0
   - TempoPFN families already covered by custom implementations
   - Rebalance that 0.26 to dynamics families

4. **Keep our unique families** (competitive advantage)
   - web_traffic (0.16) — neither king nor j-test has this
   - seasonal_level (0.10) — drift-neutral seasonal
   - physical_hourly (0.10) — calibrated 24/168 weather

5. **Consider adding**:
   - OU stochastic volatility (king has 0.10)
   - Long-memory spectral (king has 0.06)
   - Seasonal counts (king has 0.02)

6. **Consider measurement artifacts** (king's layer)
   - Reversal, censoring, quantization, sample-and-hold
   - Bridges clean priors to real pipelines
   - Low probability (4–7%) so most rows stay clean

### What We Should Keep

- **Web_traffic family**: Neither competitor has clean weekly counts
- **Physical_hourly**: Calibrated to real weather (period-24/168, high AR phi)
- **Seasonal_level**: Drift-neutral seasonal (missing from both)
- **TempoPFN engine**: If we can make it work without sandbox OOM

### Proposed Rebalance

```json
{
  "min_length": 4096,
  "max_length": 4096,
  "family_weights": {
    "ar2": 0.15,                  // ← INCREASE (match king/j-test)
    "integrated": 0.13,           // ← INCREASE
    "regime_shift": 0.12,         // ← INCREASE
    "web_traffic": 0.14,          // ← KEEP (our advantage)
    "seasonal_level": 0.10,       // ← KEEP
    "physical_hourly": 0.10,      // ← KEEP
    "multiplicative": 0.08,
    "threshold_ar": 0.06,
    "kernel_gp": 0.06,            // ← reduce from 0.08
    "forecast_pfn": 0.06,         // ← REDUCE from 0.26 (king uses 0)
    "ou_stochastic_vol": 0.00     // ← CONSIDER ADDING (king 0.10)
  }
}
```

**Dynamics total**: 0.15 + 0.13 + 0.12 + 0.06 = **0.46** (+ ou 0.10 = 0.56)

---

## 7. Performance Optimization Opportunities

### From jan/cascade9

1. **SciPy lfilter for AR recurrences** (4–12× faster)
   - Our current AR uses Python loops
   - Replace with scipy.signal.lfilter

2. **FFT spectral GP** (1 pass vs 48)
   - Our kernel_gp might be using RFF
   - Replace with numpy.fft.irfft

3. **Prefetching thread** (+14.4% throughput)
   - Overlap generation with training
   - One-slot queue, deterministic draw order

4. **Cached seasonal basis** (@lru_cache)
   - Our _seasonal recomputes sin/cos every call
   - Cache basis, reuse with sin(a+b)

5. **2048-row chunks** (vs our current ?)
   - King measured this as sweet spot
   - Larger than j-test's 256

6. **Sparse event draws** (king's optimization)
   - Draw jump/shock/heavy-tail values only for active events
   - Not dense arrays mostly discarded

### From j-test minimalism

- **NumPy-only**: Faster import, simpler sandbox
- **10 families**: Easier to reason about, less code
- **No augmentation layer**: Simpler pipeline

---

## 8. A/B Testing Lessons

jan/cascade9 documented:
> "A richer 'composite' family (layered trend + multi-harmonic seasonality + heavy-tailed heteroskedastic noise + structural events) was implemented and measured — it is more **realistic**, but at high fidelity it scored slightly **WORSE** on the proxy (LCB +0.15 vs this mix's +0.167), so it was **not shipped**."

**Lesson**: More realistic ≠ better score.  
The validator's synthetic pool may under-reward realism.  
**Ship what wins on the proxy, not what looks more realistic.**

Also:
> "Dynamics-heavy composition selected by a controlled local A/B: it beat the prior baseline on all three validation seeds."

**Lesson**: Don't guess — measure. Run controlled 120-second A/B with same model/pool/budget/seeds.

---

## 9. Recommendations

### Immediate Actions

1. **Switch to all-4096**
   - Drop variable-length entirely
   - Perfect token efficiency

2. **Rebalance to dynamics-heavy**
   - ar2: 0.15
   - integrated: 0.13–0.15
   - regime_shift: 0.12
   - Reduce forecast_pfn: 0.26 → 0.06 or 0

3. **Keep our unique families**
   - web_traffic, seasonal_level, physical_hourly
   - This is our competitive advantage

### Performance Wins

4. **Add SciPy lfilter for AR** (4–12× faster)
5. **Add prefetching thread** (+14% throughput)
6. **Cache seasonal basis** (@lru_cache)
7. **Increase chunk size** to 2048

### Consider Adding

8. **Measurement artifacts layer** (king's innovation)
9. **OU stochastic volatility** (king 0.10)
10. **Long-memory spectral** (king 0.06)

### A/B Validation

11. **Run local 120-second A/B** before deploying
    - New rebalance vs current
    - Same model/pool/budget/seeds
    - Ship only if it wins on all 3 seeds

---

## 10. File Locations

```
miner-ops/top-miners/
├── uid-131/              # jan/cascade9 (KING)
│   ├── generator.py      # 14 families, SciPy, prefetch, artifacts
│   ├── config.json       # dynamics-heavy weights
│   └── README.md         # A/B testing notes
└── uid-64/               # j-test/2_1_v5
    ├── generator.py      # 10 families, NumPy-only, long-range focus
    ├── config.json       # all-4096, de-trended
    └── README.md         # long-range hypothesis
```

---

## Conclusion

The mainnet king (jan/cascade9) and top challenger (j-test/2_1_v5) **independently converged** on:
- **All-4096 length** (perfect token efficiency)
- **Dynamics-heavy mixture** (~57% weight on persistent/structural families)
- **Heavy near-unit-root AR** (ar2 0.15 + integrated 0.12–0.15)
- **Length-invariant drift** (excursion-based)

The king wins through **broader prior coverage** (14 families), **performance optimizations** (SciPy, FFT, prefetch), **measurement artifacts**, and **rigorous A/B testing**.

**Our path forward**: Adopt their all-4096 + dynamics-heavy core, keep our unique web_traffic/physical_hourly advantage, add their performance wins, and A/B validate before deploying.

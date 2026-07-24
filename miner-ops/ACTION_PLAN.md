# Action Plan: Upgrade Our Generator Based on Top Miner Analysis

**Status**: Ready to implement  
**Target**: Beat jan/cascade9 (current king, 103.58 TAO)

---

## Critical Changes (MUST DO)

### 1. Switch to All-4096 Length

**Current**: `min_length: 64, max_length: 4096` (variable)  
**Target**: `min_length: 4096, max_length: 4096` (fixed)

**Why**: 
- Training uses 128 patches × 32 tokens = 4096 context
- Variable-length wastes 1 non-target patch per series
- All-4096 = **perfect token efficiency** (128 target patches)
- **Both king and j-test use this**

**Impact**: Immediate throughput gain, better long-cycle learning

```json
{
  "min_length": 4096,
  "max_length": 4096
}
```

---

### 2. Rebalance to Dynamics-Heavy Mixture

**Current weights** (my-generator):
```json
{
  "forecast_pfn": 0.26,        // ← TOO HIGH (king uses 0)
  "web_traffic": 0.16,         // ✓ KEEP (our advantage)
  "seasonal_level": 0.10,      // ✓ KEEP
  "physical_hourly": 0.10,     // ✓ KEEP
  "kernel_gp": 0.08,
  // ar2, integrated, regime_shift: TOO LOW
}
```

**Target weights** (jan/cascade9 inspired):
```json
{
  "ar2": 0.15,                 // ← INCREASE (was ~0.05?)
  "integrated": 0.13,          // ← INCREASE (was ~0.05?)
  "regime_shift": 0.12,        // ← INCREASE (was ~0.05?)
  "web_traffic": 0.14,         // ← KEEP (unique advantage)
  "seasonal_level": 0.10,      // ← KEEP (drift-neutral)
  "physical_hourly": 0.10,     // ← KEEP (calibrated weather)
  "multiplicative": 0.08,
  "threshold_ar": 0.06,
  "kernel_gp": 0.06,           // reduce
  "forecast_pfn": 0.06,        // ← CUT from 0.26 (king uses 0!)
  // Consider: ou_stochastic_vol 0.10 (king has this)
}
```

**Dynamics total**: 0.15 + 0.13 + 0.12 + 0.06 = **0.46** (vs king's 0.57)

**Why**:
- King and j-test both put ~57% weight on persistent/structural families
- Our forecast_pfn overlap is redundant (covered by custom families)
- ar2 + integrated = near-unit-root coverage (finance/econ domains)
- Keep our unique web_traffic/physical_hourly advantage

---

## Performance Optimizations (HIGH VALUE)

### 3. Add SciPy lfilter for AR Recurrences

**Current**: Python loops over time (slow)
```python
for t in range(1, L):
    x[:, t] = phi * x[:, t - 1] + innov[:, t]
```

**Target**: SciPy compiled filter (king's approach)
```python
from scipy.signal import lfilter

def _ar1_batch(innov, phi):
    n, L = innov.shape
    x = np.empty((n, L), dtype=np.float64)
    for i in range(n):
        x[i] = lfilter([1.0], [1.0, -float(phi[i])], innov[i])
    return x
```

**Impact**: King measured **4–12× faster** for AR recurrences

---

### 4. Add Prefetching Thread

**Current**: Synchronous generation (GPU waits for data)  
**Target**: One-chunk prefetch (overlaps NumPy with GPU)

**Why**:
- King measured: 21.9% → 3.9% data blocking
- Throughput gain: +14.4% (2.13M → 2.43M point-passes/s)
- Deterministic draw order preserved

**Implementation**: See jan/cascade9 generator.py lines 183-254 (Queue + Thread)

---

### 5. Cache Seasonal Basis

**Current**: Recompute sin/cos every call
```python
def _seasonal(rng, n, L):
    t = np.arange(L)[None, :]
    out = amp * np.sin(2.0 * np.pi * t / per + phase)  # ← recomputed
```

**Target**: Cached basis (king's optimization)
```python
from functools import lru_cache

@lru_cache(maxsize=4)
def _seasonal_basis(L):
    angle = 2.0 * np.pi * np.arange(L)[None, :] / periods[:, None]
    return np.sin(angle), np.cos(angle)

def _seasonal(rng, n, L):
    sin_basis, cos_basis = _seasonal_basis(L)
    # Use sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
    component = amp * (sin_basis * np.cos(phase) + cos_basis * np.sin(phase))
```

**Impact**: Reduces trigonometric work without narrowing prior

---

### 6. Increase Chunk Size to 2048

**Current**: Unknown (check our code)  
**Target**: 2048 rows per chunk

**Why**: King measured this as throughput sweet spot (~6% faster than 1024)

---

## Consider Adding (MEDIUM VALUE)

### 7. Measurement Artifacts Layer

King applies low-rate artifacts to all families:
- Reversal (6%)
- Sign inversion (4%, non-count families)
- Censoring (6%)
- Quantization (7%)
- Sample-and-hold (4%)

**Why**: Bridges clean priors to real measurement pipelines  
**Preserves**: Positivity for count/magnitude families

See jan/cascade9 `_measurement_artifacts()` function (lines 373-432)

---

### 8. Add OU Stochastic Volatility Family

King has this at 0.10 weight. It's:
- Mean-reverting regimes
- Clustered/heavy-tailed volatility
- Bounded dynamics

**Gap**: We don't have a stochastic-volatility family

---

### 9. Add Long-Memory Spectral Family

King has this at 0.06 weight:
- Persistent/anti-persistent power-law spectra
- fGn-like paths

**Covers**: Domains with scale-dependent roughness

---

## DO NOT DO

### ❌ Don't Increase Realism Without A/B Validation

King's note:
> "A richer 'composite' family was implemented — more **realistic**, but scored **WORSE** on the proxy. Did not ship."

**Lesson**: More realistic ≠ better score. Ship what wins on validation, not what looks better.

### ❌ Don't Guess — Measure

King ran controlled 120-second A/B with 3 validation seeds before deploying dynamics-heavy.

**Our process**:
1. Implement changes
2. Run local A/B: new vs current (same model/pool/budget/seeds)
3. Deploy **only if new wins on all 3 seeds**

---

## Implementation Checklist

- [ ] 1. Set `min_length: 4096, max_length: 4096` in config.json
- [ ] 2. Rebalance family_weights (cut forecast_pfn 0.26→0.06, boost ar2/integrated/regime_shift)
- [ ] 3. Replace AR loops with scipy.signal.lfilter
- [ ] 4. Add prefetching thread (copy king's Queue + Thread pattern)
- [ ] 5. Cache seasonal basis with @lru_cache
- [ ] 6. Set _CHUNK = 2048
- [ ] 7. (Optional) Add measurement_artifacts layer
- [ ] 8. (Optional) Add ou_stochastic_vol family (0.10)
- [ ] 9. (Optional) Add long_memory spectral family (0.06)
- [ ] 10. Verify determinism: `cascade verify ./my-generator`
- [ ] 11. Run local 120-second A/B (3 seeds): new vs current
- [ ] 12. Deploy **only if new wins all 3 seeds**

---

## Expected Gains

| Change | Impact |
|--------|--------|
| All-4096 length | Immediate token efficiency (no wasted patches) |
| Dynamics-heavy rebalance | Better finance/econ/regime coverage (king's A/B winner) |
| SciPy lfilter | 4–12× faster AR recurrences |
| Prefetching thread | +14% throughput, less GPU blocking |
| Cached seasonal basis | Reduced trig work |
| 2048 chunks | ~6% throughput gain |

**Conservative estimate**: 20–30% throughput gain + better score from dynamics-heavy mixture

---

## Risk Mitigation

1. **Keep our unique families** (web_traffic, physical_hourly, seasonal_level)
   - This is our competitive advantage
   - Neither king nor j-test has clean weekly counts or calibrated weather

2. **A/B validate before deploying**
   - Don't ship without 3-seed local validation
   - King's discipline: measure, don't guess

3. **Verify determinism after every change**
   - `cascade verify` must pass
   - Byte-identical corpora required

---

## Timeline

1. **Day 1**: Implement critical changes (1–2, all-4096 + rebalance)
2. **Day 2**: Add performance optimizations (3–6, SciPy + prefetch + cache)
3. **Day 3**: Optional additions (7–9, artifacts + OU + long-memory)
4. **Day 4**: Local A/B validation (3 seeds, 120 seconds each)
5. **Day 5**: Deploy if validated, otherwise iterate

---

## Success Metrics

- [ ] `cascade verify` passes (determinism)
- [ ] Local throughput > 11M points/s (match king's isolated benchmark)
- [ ] A/B score beats current on all 3 validation seeds
- [ ] Mainnet emission > 44 TAO (beat j-test)
- [ ] **Target**: Dethrone jan/cascade9 (103.58 TAO)

---

## Files to Modify

```
miner-ops/my-generator/
├── config.json           # min/max length, family_weights rebalance
├── generator.py          # SciPy lfilter, prefetch, cache, chunk size
└── (optional) families/  # Add ou_stochastic_vol, long_memory if needed
```

Reference implementations:
```
miner-ops/top-miners/
├── uid-131/generator.py  # jan/cascade9 (copy prefetch, lfilter, cache patterns)
└── uid-64/generator.py   # j-test/2_1_v5 (minimalist reference)
```

---

## Next Steps

1. **Read current my-generator config** to audit existing weights
2. **Implement critical changes** (all-4096 + rebalance)
3. **Verify determinism** after each change
4. **Run local A/B** before deploying

Ready to proceed?

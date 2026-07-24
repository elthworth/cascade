# Conservative Rebalance — Convergent Winner Strategy

**File**: `my-generator/config.CONSERVATIVE.json`  
**Strategy**: Adopt validated convergent core, keep unique families modest  
**Target**: Beat j-test (44 TAO), challenge jan (103 TAO with performance stack)

---

## Philosophy

**90% convergent winners, 10% unique advantage**

Both jan/cascade9 and j-test/2_1_v5 independently discovered the same core strategy through 12 total iterations. This is **strong validation** that we should copy, not reinvent.

Our unique families (web_traffic, physical_hourly, seasonal_level) are kept at **modest weight** (0.04–0.06 each) to avoid over-investing in unvalidated hypotheses.

---

## Critical Changes

### 1. All-4096 Length (Perfect Token Efficiency)

**Before**:
```json
{
  "min_length": 64,
  "max_length": 4096,
  "length_mixture": [
    [0.10, 64, 256],
    [0.25, 256, 1024],
    [0.65, 1024, 4096]
  ]
}
```

**After**:
```json
{
  "min_length": 4096,
  "max_length": 4096
}
```

**Why**: 
- Training uses 128 patches × 32 tokens = 4096 context
- Variable-length wastes 1 non-target patch per series
- **Both winners use all-4096** (jan from day 1, j-test from v4)

**Impact**: Immediate throughput gain + better long-cycle learning

---

### 2. Dynamics-Heavy Rebalance (0.55 target)

#### Mapping Our Families → Convergent Winners

| Our Family | Maps To | Winner Weight | Our New Weight |
|------------|---------|---------------|----------------|
| `random_walk` | integrated | 0.12–0.15 | **0.15** |
| `regime_garch` | regime_shift | 0.12–0.16 | **0.13** |
| `trend_seasonal` | trend_seasonal_ar | 0.12–0.20 | **0.12** |
| `fgn` | long_memory / AR2 | 0.06–0.15 | **0.10** |
| `kernel_gp` | spectral_gp | 0.07–0.08 | **0.08** |
| **Dynamics Total** | | **0.55–0.57** | **0.58** ✓ |

**Rationale**:
- `random_walk` = I(1)/I(2) random walks (same as integrated)
- `regime_garch` = structural breaks + volatility regimes (same as regime_shift)
- `fgn` = fractional Gaussian noise (persistent/anti-persistent, maps to long_memory + AR2)
- `trend_seasonal` = baseline seasonal (same as trend_seasonal_ar)

---

### 3. Weight Changes — Before vs After

| Family | BEFORE | AFTER | Change | Rationale |
|--------|--------|-------|--------|-----------|
| **random_walk** | 0.02 | **0.15** | +650% | ← Integrated (both winners 0.12–0.15) |
| **regime_garch** | 0.04 | **0.13** | +225% | ← Regime shift (both winners 0.12–0.16) |
| **trend_seasonal** | 0.05 | **0.12** | +140% | ← Baseline seasonal (winners 0.12–0.20) |
| **fgn** | 0.03 | **0.10** | +233% | ← Long memory + AR2 (winners 0.06–0.15) |
| **forecast_pfn** | 0.26 | **0.06** | **-77%** | ← Neither winner uses TempoPFN! |
| **web_traffic** | 0.16 | **0.06** | **-63%** | ← Conservative (jan's counts: 0.02) |
| **seasonal_level** | 0.10 | **0.06** | **-40%** | ← Conservative (unvalidated) |
| **physical_hourly** | 0.10 | **0.04** | **-60%** | ← Conservative (jan's sensors: 0.02) |
| **kernel_gp** | 0.08 | 0.08 | 0% | ← Kept (matches winners 0.07–0.08) |
| **calendar** | 0.05 | **0.04** | -20% | ← Minor trim |
| **chaotic** | 0.02 | **0.04** | +100% | ← Match winners (0.03–0.04) |
| **rhythm** | 0.02 | **0.03** | +50% | ← Slight boost (threshold_ar analog) |
| **intermittent** | 0.03 | **0.02** | -33% | ← Cut (winners 0.01–0.02) |
| **fractal_multi** | 0.02 | **0.03** | +50% | ← Multiplicative analog |
| **net_diffusion** | 0.02 | 0.02 | 0% | ← Kept small |
| **bursts_anomaly** | 0.02 | **0.01** | -50% | ← Cut (winners pulse 0.01) |
| **mixup** | 0.02 | **0.01** | -50% | ← Cut (augmentation) |

---

### 4. Dynamics Breakdown

**BEFORE** (current):
```
random_walk:     0.02
regime_garch:    0.04
fgn:             0.03
trend_seasonal:  0.05
---------------------------
Dynamics total:  0.14  ← WAY TOO LOW
```

**AFTER** (conservative):
```
random_walk:     0.15  (integrated analog)
regime_garch:    0.13  (regime_shift analog)
trend_seasonal:  0.12  (trend_seasonal_ar)
fgn:             0.10  (long_memory + AR2)
chaotic:         0.04
rhythm:          0.03  (threshold_ar analog)
---------------------------
Dynamics total:  0.57  ← MATCHES WINNERS ✓
```

**Convergent validation**:
- jan/cascade9 (KING): 0.57 dynamics
- j-test/2_1_v5: 0.57 long-range
- **Our conservative**: 0.58 ✓

---

### 5. Unique Family Strategy (Modest Weight)

**Our unique advantage** (neither winner has):
```
web_traffic:     0.06  (was 0.16)  ← Clean weekly counts
seasonal_level:  0.06  (was 0.10)  ← Drift-neutral seasonal
physical_hourly: 0.04  (was 0.10)  ← Calibrated weather
---------------------------
Unique total:    0.16  (was 0.36)
```

**Why cut them**?

1. **jan tested similar families at low weight**:
   - `seasonal_counts: 0.02` (vs our web_traffic 0.16)
   - `physical_sensors: 0.02` (vs our physical_hourly 0.10)

2. **jan removed retail_demand** after testing (v16 had 0.06, v12 removed)

3. **"Realistic" ≠ better score** (jan's own lesson)

**Conservative approach**: Keep them at 0.04–0.06 (2–5% advantage), not 0.36 (betting the farm).

---

### 6. forecast_pfn Dramatic Cut (0.26 → 0.06)

**Why cut TempoPFN**?

1. **Neither winner uses it**:
   - jan/cascade9: 0.00
   - j-test/2_1_v5: 0.00

2. **TempoPFN families already covered** by custom implementations:
   - Trend/seasonal → `trend_seasonal`
   - Regime shifts → `regime_garch`
   - Random walk → `random_walk`
   - GP/smooth → `kernel_gp`

3. **TempoPFN augmentation layer** might conflict with deterministic requirement or add overhead

**Conservative**: Keep 0.06 (not 0) in case augmentation helps, but don't over-invest.

---

## Expected Impact

### Throughput Gains

| Change | Impact |
|--------|--------|
| All-4096 length | +15–25% (no wasted patches) |
| Cut TempoPFN overhead | +5–10% (simpler pipeline) |
| **Total immediate** | **+20–35%** |

### Score Gains (Hypothesis)

| Change | Impact |
|--------|--------|
| Dynamics 0.14 → 0.58 | **Primary driver** (validated by both winners) |
| Finance/econ coverage | random_walk 0.15 + fgn 0.10 = 0.25 ← near-unit-root |
| Structural coverage | regime_garch 0.13 ← regime shifts |
| Unique families | +2–5% (modest weight, unvalidated) |
| **Conservative target** | **Beat j-test (44 TAO)** |

---

## Risk Mitigation

### What We're Keeping

1. **All unique families** (web_traffic, physical_hourly, seasonal_level)
   - Just at modest weight (0.04–0.06)
   - Can boost if A/B validates them

2. **All existing families** (none removed)
   - Just rebalanced to match winners
   - Trimmed low-performers (mixup, bursts)

3. **All calibration params** (family_params)
   - web_traffic: p_clean, p_weekend_dip
   - physical_hourly: ar_phi, period24
   - No changes to family internals

### What We're Not Doing

❌ **Over-investing in unique families** (was 0.36, now 0.16)  
❌ **Betting against convergent winners** (adopting their 0.57 core)  
❌ **Removing TempoPFN completely** (keeping 0.06 as insurance)  
❌ **Changing family implementations** (only weights)

---

## A/B Validation Plan

**Before deploying**, run local 120-second A/B (jan's discipline):

### Setup
```bash
# 1. Score current config (baseline)
cascade score my-generator --pool-dir eval-pool/v1 --seed 42 --heat-budget 120

# 2. Score conservative config
cp config.CONSERVATIVE.json config.json
cascade score my-generator --pool-dir eval-pool/v1 --seed 42 --heat-budget 120

# 3. Repeat with seeds 43, 44 (3-seed validation)
```

### Decision Rule

**Ship conservative IF**:
- Wins on **all 3 seeds** (jan's standard)
- Geomean score < current (lower is better)

**Otherwise**: Iterate on weights, don't guess.

---

## Implementation Checklist

- [ ] 1. Backup current config: `cp config.json config.ORIGINAL.json`
- [ ] 2. Copy conservative: `cp config.CONSERVATIVE.json config.json`
- [ ] 3. Verify determinism: `cascade verify my-generator`
- [ ] 4. Run A/B validation (3 seeds, 120 seconds each)
- [ ] 5. **If conservative wins all 3**: Deploy
- [ ] 6. **If current wins any**: Analyze gap, iterate

---

## Next Phase: Performance Stack (After A/B Validates)

Once conservative config wins A/B:

1. **Add SciPy lfilter** for AR recurrences (4–12× faster)
2. **Replace RFF with FFT** for GP (1 pass vs 48)
3. **Add prefetching thread** (+14% throughput)
4. **Increase chunk size** to 2048 rows
5. **Cache seasonal basis** (@lru_cache)

**Expected**: Match jan's throughput advantage (2.4× j-test)

---

## Success Metrics

### Phase 1 (Config Only)
- [ ] `cascade verify` passes (determinism preserved)
- [ ] A/B wins on all 3 validation seeds
- [ ] Local throughput > current (all-4096 efficiency)

### Phase 2 (Mainnet)
- [ ] Emission > 10 TAO (proof of competitiveness)
- [ ] **Target**: Emission > 44 TAO (beat j-test)
- [ ] **Stretch**: Emission > 50 TAO (top 3)

### Phase 3 (Performance Stack)
- [ ] Throughput > 8M points/s (2× current estimate)
- [ ] **Target**: Emission > 60 TAO
- [ ] **Stretch**: Challenge jan (103 TAO)

---

## Rationale Summary

**Why this will work**:

1. **Validated by convergent evolution** (2 independent miners, 12 iterations)
2. **Conservative on unvalidated hypotheses** (unique families modest, not dominant)
3. **Immediate wins** (all-4096, dynamics boost)
4. **Low risk** (keeps all families, just rebalances)
5. **A/B validated before deploy** (ship what measures, not what feels right)

**Why this beats j-test (44 TAO)**:
- Same core strategy (0.57 dynamics, all-4096)
- Plus unique families (small edge)
- Plus future performance stack (SciPy, FFT, prefetch)

**Why this can challenge jan (103 TAO)**:
- Same dynamics core (0.57)
- Add performance stack (SciPy, FFT, prefetch → 2.4× multiplier)
- Unique families (2–5% edge if validated)

---

## Files

- **Conservative config**: `my-generator/config.CONSERVATIVE.json`
- **Original backup**: `my-generator/config.ORIGINAL.json` (create before applying)
- **Analysis**: `miner-ops/EVOLUTION_ANALYSIS.md` (full 13-version trace)
- **This document**: `miner-ops/CONSERVATIVE_REBALANCE.md`

---

## Ready to Deploy?

**NO — A/B validate first!**

1. Run 3-seed local validation
2. **Ship only if conservative wins all 3**
3. Then add performance stack
4. Challenge the king

**The winners didn't guess — neither should we.**

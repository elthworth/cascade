# Strategy Evolution Analysis — CASCADE Top Miners

**Analysis of 13 generator versions** across 2 winning families  
**Date**: 2026-07-24

---

## Executive Summary

Traced the complete evolution of both winning families:
- **jan/cascade**: 7 versions (v2→v4→v7→v8→v16→v16b→v12 KING)
- **j-test**: 5 versions (v3→v3.1→v3-detrended→v4→v5)

### Critical Evolution Insights

1. **jan discovered all-4096 FIRST** (cascade1/v2 already all-4096)
2. **j-test discovered it at v4** after 3 variable-length iterations
3. **Both converged on dynamics-heavy** through independent iteration
4. **jan's winning move**: Return to spectral (v12) after research detour (v16)
5. **j-test's winning move**: Long-range rebalance (v5) with dramatic weight shifts

---

## 1. jan/cascade Evolution (7 tracked versions)

### Version Timeline

```
cascade1/v2 → v4 → v7 → v8 → v16 (research) → v16b → v12 KING
  ↓          ↓     ↓     ↓       ↓              ↓       ↓
all-4096   +phys +seas  BOOST  +retail       test    WINNER
                        dynam   demand
```

### Length Strategy: All-4096 From Day One

**ALL versions**: `[4096, 4096]`

jan **never experimented** with variable length. They committed to perfect token efficiency from the start.

---

### Weight Evolution Table

| Family | v2 (cascade1) | v4 | v7 | v8 | v16 (research) | v16b | **v12 (KING)** |
|--------|---------------|----|----|----|----|-------|---------|
| **ar2** | 0.10 | 0.09 | 0.09 | **0.11** | **0.15** | 0.13 | **0.15** ✓ |
| **integrated** | 0.08 | 0.07 | 0.07 | **0.09** | **0.12** | 0.11 | **0.12** ✓ |
| **regime_shift** | 0.12 | 0.11 | 0.10 | 0.11 | 0.11 | 0.11 | **0.12** |
| **threshold_ar** | 0.06 | 0.06 | 0.06 | 0.05 | 0.07 | 0.06 | **0.08** |
| **ou_stochastic_vol** | 0.07 | 0.07 | 0.07 | 0.05 | **0.10** | **0.10** | **0.10** ✓ |
| **Dynamics Total** | **0.43** | **0.40** | **0.39** | **0.41** | **0.55** | **0.51** | **0.57** ✓ |
| | | | | | | | |
| **trend_seasonal_ar** | 0.18 | 0.17 | 0.16 | 0.18 | 0.11 | 0.11 | **0.12** |
| **multiplicative** | 0.12 | 0.11 | 0.10 | 0.12 | 0.07 | 0.07 | **0.08** |
| **spectral_gp** | 0.07 | 0.07 | 0.07 | 0.08 | 0.06 | 0.06 | **0.07** |
| **long_memory** | 0.06 | 0.06 | 0.06 | 0.05 | 0.06 | 0.06 | **0.06** |
| **chaotic** | 0.06 | 0.06 | 0.06 | 0.04 | 0.03 | 0.02 | **0.04** |
| **intermittent** | 0.05 | 0.05 | 0.03 | 0.02 | 0.01 | 0.01 | **0.01** |
| **pulse_outlier** | 0.03 | 0.03 | 0.03 | 0.02 | 0.01 | 0.02 | **0.01** |
| **physical_sensors** | — | **0.05** | 0.05 | 0.04 | 0.02 | 0.04 | **0.02** |
| **seasonal_counts** | — | — | **0.05** | 0.04 | 0.02 | 0.05 | **0.02** |
| **retail_demand** | — | — | — | — | **0.06** | 0.05 | — |

---

### Strategic Shifts Across Versions

#### v2 → v4 (cascade1 → cascade2)
- **Added**: physical_sensors (0.05)
- **Reduced**: ar2 (0.10→0.09), integrated (0.08→0.07)
- **Strategy**: Exploration — test physical sensor coverage

#### v4 → v7
- **Added**: seasonal_counts (0.05)
- **Reduced**: multiplicative (0.11→0.10), trend_seasonal (0.17→0.16), intermittent (0.05→0.03)
- **Strategy**: Add count/discrete-demand coverage

#### v7 → v8 (First dynamics boost)
- **BOOSTED**: ar2 (0.09→**0.11**), integrated (0.07→**0.09**)
- **Reduced**: ou_stochastic_vol (0.07→0.05), long_memory (0.06→0.05), threshold_ar (0.06→0.05)
- **Reduced**: chaotic, intermittent, pulse_outlier (moved toward long-range)
- **Strategy**: **First signal toward dynamics-heavy**

#### v8 → v16 (MAJOR PIVOT — "research" branch)
- **MAJOR BOOST**: ar2 (0.11→**0.15**), integrated (0.09→**0.12**), ou_stochastic_vol (0.05→**0.10**)
- **Added**: retail_demand (0.06)
- **CUT**: trend_seasonal (0.18→0.11), multiplicative (0.12→0.07), chaotic (0.04→0.03)
- **Dynamics total**: 0.41 → **0.55**
- **Strategy**: **Committed to dynamics-heavy hypothesis**
- **Branch name change**: "spectral" → "research"

#### v16 → v16b (Test variant)
- **Minor rebalance**: ar2 (0.15→0.13), integrated (0.12→0.11)
- **Added back**: chaotic (0.03→0.02→back), seasonal_counts weight
- **Strategy**: Testing slight variations of dynamics-heavy formula

#### v16 → v12 (WINNING PIVOT — back to "spectral")
- **Kept**: ar2 (0.15), integrated (0.12), ou_stochastic_vol (0.10) ← dynamics core
- **BOOSTED**: threshold_ar (0.07→**0.08**), regime_shift (0.11→**0.12**)
- **Refined**: trend_seasonal (0.11→0.12), multiplicative (0.07→0.08), spectral_gp (0.06→0.07)
- **REMOVED**: retail_demand
- **Dynamics total**: **0.57** (highest ever)
- **Branch name**: "research" → **"spectral"** (returned to roots)
- **Strategy**: **Dynamics-heavy validated, return to spectral GP focus, remove experimental retail family**

---

### The Winning Formula (v12)

jan/cascade **tried 7 iterations**, and the winner:

1. **Kept** the v16 dynamics boost (ar2 0.15, integrated 0.12, OU 0.10)
2. **Boosted** threshold_ar and regime_shift further
3. **Removed** experimental retail_demand family
4. **Returned to "spectral"** branding (confidence in FFT GP approach)
5. **Achieved** 0.57 dynamics weight (highest in the lineage)

**Key insight**: v16 (research) discovered the dynamics formula, but **overshot** by cutting seasonality too much. v12 **rebalanced** back toward seasonal coverage while keeping dynamics core.

---

## 2. j-test Evolution (5 versions)

### Version Timeline

```
v3 → v3.1 → v3-detrended → v4 → v5 CHALLENGER
 ↓     ↓          ↓         ↓     ↓
512-  SAME    64-4096    ALL-  LONG-
2048                     4096   RANGE
```

### The All-4096 Discovery (v3 → v4)

| Version | min | max | Name | Status |
|---------|-----|-----|------|--------|
| v3 (UID 59) | 512 | 2048 | custom-mixture-of-priors-v3 | Variable |
| v3.1 (UID 60) | 512 | 2048 | custom-mixture-of-priors-v3.1 | Variable |
| v3-detrended (UID 85) | 64 | 4096 | custom-detrended-v3 | Variable |
| **v4** (UID 63) | **4096** | **4096** | **custom-fullctx-v4** | **ALL-4096** ✓ |
| **v5** (UID 64) | **4096** | **4096** | custom-longctx-lrfam-v5 | **ALL-4096** ✓ |

**Critical discovery**: j-test found all-4096 at **v4** (after 3 variable-length attempts).

---

### Weight Evolution Table

| Family | v3 | v3.1 | v3-detrend | v4 (all-4096) | **v5 (long-range)** |
|--------|-------|------|------------|---------------|---------------------|
| **ar2** | 0.10 | 0.10 | **0.12** | 0.12 | **0.15** ✓ |
| **integrated** | 0.08 | 0.08 | **0.10** | 0.10 | **0.15** ✓ |
| **regime_shift** | 0.10 | 0.10 | **0.12** | 0.12 | **0.16** ✓ |
| **rff_gp** | 0.07 | 0.07 | **0.08** | 0.08 | **0.11** ✓ |
| **Long-range Total** | **0.35** | **0.35** | **0.42** | **0.42** | **0.57** ✓ |
| | | | | | |
| **trend_seasonal_ar** | 0.26 | 0.26 | **0.22** | 0.22 | **0.20** |
| **multiplicative** | 0.16 | 0.16 | 0.14 | 0.14 | 0.13 |
| **chaotic** | 0.05 | 0.05 | **0.07** | 0.07 | **0.03** |
| **threshold_ar** | 0.06 | 0.06 | 0.06 | 0.06 | **0.04** |
| **intermittent** | 0.03 | 0.03 | **0.06** | 0.06 | **0.02** |
| **pulse_outlier** | 0.03 | 0.03 | 0.03 | 0.03 | **0.01** |

---

### Strategic Shifts Across Versions

#### v3 → v3.1 (NO CHANGES)
- **Identical weights** (experimental validation or bug fix)

#### v3.1 → v3-detrended (First rebalance)
- **Length**: 512-2048 → **64-4096** (moved toward long context)
- **BOOSTED**: ar2 (0.10→0.12), integrated (0.08→0.10), regime_shift (0.10→0.12), rff_gp (0.07→0.08), chaotic (0.05→0.07), intermittent (0.03→0.06)
- **CUT**: trend_seasonal (0.26→0.22)
- **Long-range total**: 0.35 → **0.42**
- **Strategy**: "Detrend" + move toward long-context families

#### v3-detrend → v4 (ALL-4096 BREAKTHROUGH)
- **Length**: 64-4096 → **[4096, 4096]** ← **CRITICAL DISCOVERY**
- **Weights**: UNCHANGED from v3-detrend
- **Strategy**: Test all-4096 efficiency with same mixture

#### v4 → v5 (LONG-RANGE REBALANCE — final iteration)
- **MAJOR BOOST**: ar2 (0.12→**0.15**), integrated (0.10→**0.15**), regime_shift (0.12→**0.16**), rff_gp (0.08→**0.11**)
- **MAJOR CUT**: chaotic (0.07→**0.03**), intermittent (0.06→**0.02**), pulse_outlier (0.03→**0.01**), threshold_ar (0.06→**0.04**)
- **Long-range total**: 0.42 → **0.57**
- **Strategy**: "Away from choppy short-range families that waste the 128-patch context"
- **README quote**: "Long-range-structure priors... at a 4096-token context the model benefits from long-range learnable structure."

---

### The j-test Winning Formula (v5)

j-test's **breakthrough came in 2 steps**:

1. **v3-detrend → v4**: Discovered all-4096 (kept weights, tested length)
2. **v4 → v5**: Committed to "long-range hypothesis" with dramatic rebalance

**v5 strategy**:
- **Maximize**: ar2 + integrated + regime_shift + GP (long-range families)
- **Minimize**: chaotic + intermittent + pulse_outlier (short-range "chop")
- **Hypothesis**: 128-patch context rewards long-range learnable structure

---

## 3. Convergent Evolution

### Both Families Independently Converged On:

| Strategy | jan/cascade | j-test | Convergence? |
|----------|-------------|--------|--------------|
| All-4096 | v2 (day 1) | v4 (iteration 4) | ✓ |
| Dynamics-heavy ~0.55–0.57 | v16 → v12 | v5 | ✓ |
| ar2 weight 0.15 | v16, v12 | v5 | ✓ |
| integrated 0.12–0.15 | v16, v12 | v5 | ✓ |
| Cut intermittent to 0.01–0.02 | v8 → v12 | v5 | ✓ |
| Cut pulse_outlier to 0.01 | v16 → v12 | v5 | ✓ |
| Reduce chaotic | v8: 0.04, v12: 0.04 | v5: 0.03 | ✓ |

**Identical final ar2 weight**: Both settled on **0.15** independently.

---

## 4. What They Tried and Abandoned

### jan/cascade Experiments

#### Kept (worked):
- **physical_sensors** (v4+): Kept at 0.02–0.05
- **seasonal_counts** (v7+): Kept at 0.02–0.05
- **ou_stochastic_vol boost** (v16+): 0.05 → 0.10 (kept in winner)
- **Dynamics-heavy core** (v16+): ar2 0.15, integrated 0.12 (kept)

#### Abandoned (didn't work or overcorrected):
- **retail_demand** (v16 only): Added 0.06, then **removed** in v12
  - **Lesson**: Experimental family didn't improve score
- **Over-cutting seasonality** (v16): trend_seasonal 0.11, multiplicative 0.07
  - **v12 corrected**: 0.12, 0.08 (rebalanced back)
- **Low chaotic** (v16: 0.03)
  - **v12 corrected**: 0.04 (small but meaningful)

### j-test Experiments

#### Kept (worked):
- **All-4096** (v4+): Perfect token efficiency
- **Long-range boost** (v5): ar2/integrated/regime_shift high
- **GP boost** (v5): rff_gp 0.08 → 0.11

#### Abandoned (overcorrected or didn't help):
- **Variable length** (v3, v3.1, v3-detrend): Tried 512-2048, 64-4096 → settled all-4096
- **High intermittent** (v3-detrend: 0.06) → **cut to 0.02** in v5
  - **Lesson**: Zero-inflated demand at high weight wastes context
- **High chaotic** (v3-detrend: 0.07) → **cut to 0.03** in v5
  - **Lesson**: Chaotic maps are short-range, don't leverage 128 patches

---

## 5. The Race to Dynamics-Heavy

### Timeline of Discovery

```
jan v2  (cascade1): 0.43 dynamics
jan v4:             0.40 (slight drop)
jan v7:             0.39 (continued drop)
jan v8:             0.41 (REVERSAL — first boost)
jan v16:            0.55 (MAJOR PIVOT)
jan v12 (KING):     0.57 (refined, WINNER)

j-test v3:          0.35 long-range (includes GP)
j-test v3-detrend:  0.42 (first boost)
j-test v4:          0.42 (kept, tested all-4096)
j-test v5:          0.57 (MATCHED jan's peak)
```

**Both hit 0.57 independently** through different paths:
- **jan**: Started 0.43, dropped to 0.39, then **reversed** at v8, **committed** at v16, **refined** at v12
- **j-test**: Started 0.35, **steady climb** to 0.57 at v5

---

## 6. Performance Optimizations (jan only)

j-test stayed **NumPy-only** across all versions.  
jan added **SciPy + optimizations** (not visible in config, requires code analysis):

From v12 README:
- **SciPy lfilter** for AR recurrences (4–12× faster)
- **FFT spectral GP** (1 pass vs 48-pass RFF)
- **Prefetching thread** (+14.4% throughput, 21.9% → 3.9% data blocking)
- **Cached seasonal basis** (@lru_cache)
- **2048-row chunks** (measured sweet spot, ~6% faster than 1024)
- **Sparse event draws** (only allocate where active)

**These optimizations don't appear in weights** but are critical to jan's 103 TAO vs j-test's 44 TAO.

---

## 7. Key Strategic Lessons

### From jan/cascade Evolution

1. **All-4096 from day 1** (never wavered)
2. **A/B test major shifts** (v16 dynamics-heavy validated on 3 seeds)
3. **Don't overfit to experiments** (retail_demand added, then removed)
4. **Rebalance after breakthroughs** (v16 over-cut seasonality, v12 corrected)
5. **Performance matters** (SciPy, FFT, prefetch → 2.4× j-test's emission)

### From j-test Evolution

1. **Iterate on length first** (512-2048 → 64-4096 → all-4096)
2. **Keep weights stable during length tests** (v3-detrend → v4 unchanged)
3. **Commit to hypothesis** (v5 "long-range" rebalance was dramatic, not timid)
4. **Simplicity wins** (NumPy-only, 10 families, no augmentation)
5. **Clear naming** ("long-range-structure priors" in README)

### Convergent Lessons (both families)

1. **All-4096 is non-negotiable** (perfect token efficiency)
2. **Dynamics-heavy wins** (~0.57 is the magic number)
3. **ar2 + integrated = 0.27–0.30** (finance/econ coverage)
4. **Cut short-range families** (intermittent, pulse_outlier, chaotic to ~0.01–0.04)
5. **Don't ship experiments without validation** (jan removed retail_demand, j-test cut intermittent)

---

## 8. What NEITHER Family Has

Opportunities for our generator:

1. **TempoPFN augmentation** (neither uses forecast_pfn)
2. **Web_traffic family** (clean weekly counts)
3. **Physical_hourly** (calibrated 24/168 weather)
4. **Seasonal_level** (drift-neutral seasonal)

jan has `physical_sensors` (0.02) and `seasonal_counts` (0.02), but these are different from our calibrated families.

---

## 9. Recommended Strategy for Our Generator

### Phase 1: Adopt Convergent Winners (IMMEDIATE)

1. **All-4096 length** (both families, day 1)
   ```json
   {"min_length": 4096, "max_length": 4096}
   ```

2. **Dynamics-heavy rebalance to 0.55–0.57**
   - ar2: **0.15** (both winners)
   - integrated: **0.13–0.15** (both winners)
   - regime_shift: **0.12–0.16** (both winners)
   - threshold_ar: **0.06–0.08**
   - ou_stochastic_vol: **0.10** (jan only, consider adding)

3. **Cut short-range families**
   - intermittent: **0.01–0.02**
   - pulse_outlier: **0.01**
   - chaotic: **0.03–0.04**

4. **Cut forecast_pfn** from 0.26 to **0.06 or 0** (neither winner uses TempoPFN)

### Phase 2: Keep Our Advantages

5. **web_traffic: 0.14** (neither has clean weekly counts)
6. **physical_hourly: 0.10** (jan's physical_sensors is 0.02, ours is calibrated)
7. **seasonal_level: 0.10** (neither has drift-neutral seasonal)

### Phase 3: Add jan's Performance Wins

8. **SciPy lfilter** for AR (4–12× faster)
9. **Prefetching thread** (+14% throughput)
10. **Cached seasonal basis**
11. **2048-row chunks**

### Phase 4: A/B Validate (jan's discipline)

12. **Local 120-second A/B** (3 seeds)
13. **Ship only if new wins all 3** (don't guess, measure)

---

## 10. Expected Outcome

### If we adopt all convergent strategies:

**Length efficiency**: all-4096 = +X% (immediate throughput gain)  
**Dynamics-heavy**: 0.57 = better finance/econ/regime coverage (jan & j-test validated)  
**Performance**: SciPy + prefetch = +20–30% throughput  
**Unique families**: web_traffic + physical_hourly = edge over both winners

**Conservative target**: **Beat j-test (44 TAO)** in round 1  
**Stretch target**: **Challenge jan (103 TAO)** if our unique families + performance match theirs

---

## Conclusion

Both winning families **independently discovered**:
- All-4096 length (perfect token efficiency)
- Dynamics-heavy mixture (~0.57)
- ar2 + integrated = 0.27–0.30 (near-unit-root coverage)

**jan won** through:
- All-4096 from day 1 (never wavered)
- 7 iterations of refinement (v2 → v12)
- Performance optimizations (SciPy, FFT, prefetch)
- A/B-tested rebalancing (v16 dynamics boost, v12 seasonal correction)

**j-test competed** through:
- All-4096 discovery at v4 (critical breakthrough)
- Long-range hypothesis (v5 dramatic rebalance)
- Minimalist simplicity (NumPy-only, 10 families)

**Our path**: Adopt their convergent winners + keep our unique families + add jan's performance stack = **new king**.

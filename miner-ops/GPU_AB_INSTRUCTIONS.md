# GPU A/B Validation Instructions

**Status**: Conservative config verified deterministic ✓  
**Ready**: Transfer to GPU pod and run validation

---

## Files to Transfer to GPU Pod

```bash
# From your local machine, transfer these to GPU pod:
miner-ops/
├── my-generator/
│   ├── config.ORIGINAL.json       # Backup of current config
│   ├── config.CONSERVATIVE.json   # New conservative config
│   ├── generator.py               # Your generator code
│   └── (all other generator files)
├── eval-pool/v1/                  # Your 453-window eval pool
├── run_ab_validation.sh           # A/B validation script
└── gpu_pod_bootstrap.sh           # Environment setup (if needed)
```

---

## On GPU Pod: Run A/B Validation

### Step 1: Setup Environment (if fresh pod)

```bash
# If starting from clean GPU pod, bootstrap first:
cd /workspace
bash gpu_pod_bootstrap.sh

# Or minimal setup:
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
cd cascade
uv sync --all-extras
```

### Step 2: Run A/B Validation

```bash
cd miner-ops

# Run full 3-seed A/B (takes ~15-20 minutes)
bash run_ab_validation.sh
```

**What it does**:
1. Tests CONSERVATIVE config on seeds 42, 43, 44 (120s each)
2. Tests ORIGINAL config on seeds 42, 43, 44 (120s each)
3. Compares results (lower geomean is better)
4. Auto-decides: deploy conservative, keep original, or iterate

---

## Expected Output

### During Run

```
=========================================
A/B Validation: Conservative vs Original
=========================================

Generator: my-generator
Pool: eval-pool/v1
Heat budget: 120s per run
Seeds: [42, 43, 44]

=== Phase 1: Testing CONSERVATIVE config ===

--- Conservative config, seed=42 ---
[... training progress ...]
geomean 0.1456
✓ Conservative seed 42 done: 0.1456

--- Conservative config, seed=43 ---
[... training progress ...]
geomean 0.1423
✓ Conservative seed 43 done: 0.1423

--- Conservative config, seed=44 ---
[... training progress ...]
geomean 0.1489
✓ Conservative seed 44 done: 0.1489

=== Phase 2: Testing ORIGINAL config ===

--- Original config, seed=42 ---
[... training progress ...]
geomean 0.1678
✓ Original seed 42 done: 0.1678

[... etc for seeds 43, 44 ...]

=========================================
RESULTS SUMMARY
=========================================

CONSERVATIVE seed 42: 0.1456
CONSERVATIVE seed 43: 0.1423
CONSERVATIVE seed 44: 0.1489
ORIGINAL seed 42: 0.1678
ORIGINAL seed 43: 0.1701
ORIGINAL seed 44: 0.1645

Seed 42: CONSERVATIVE wins ✓
Seed 43: CONSERVATIVE wins ✓
Seed 44: CONSERVATIVE wins ✓

=========================================
FINAL VERDICT
=========================================

CONSERVATIVE wins: 3 / 3

✅ DEPLOY CONSERVATIVE CONFIG

Conservative config won on ALL 3 seeds (jan's standard).
Ready to deploy to mainnet.

Next steps:
1. cd my-generator && cp config.CONSERVATIVE.json config.json
2. cascade deploy . --wallet-name h --wallet-hotkey h03 --hub-repo ramsey/suker-miner
```

---

## Interpreting Results

### ✅ Conservative Wins All 3 Seeds

**Action**: DEPLOY
```bash
cd my-generator
cp config.CONSERVATIVE.json config.json
cascade deploy . --wallet-name h --wallet-hotkey h03 --hub-repo ramsey/suker-miner
```

**Expected mainnet impact**:
- Beat j-test (44 TAO) with convergent strategy
- Challenge jan (103 TAO) after adding performance stack

---

### ❌ Original Wins All 3 Seeds

**Action**: DO NOT DEPLOY, iterate

**Possible reasons**:
1. Our family mapping was wrong (random_walk ≠ integrated?)
2. TempoPFN at 0.26 was actually helping
3. Our unique families at 0.36 were load-bearing
4. Eval pool doesn't match mainnet distribution

**Next steps**:
1. Analyze logs: which domains did conservative lose on?
2. Try intermediate weights (50% between original and conservative)
3. Test one change at a time:
   - Just all-4096 (no weight changes)
   - Just dynamics boost (keep TempoPFN)
   - Just cut TempoPFN (keep other weights)

---

### ⚠️ Mixed Results (each wins some seeds)

**Action**: ITERATE, do not deploy

**Analysis**:
1. Check which seeds conservative won/lost
2. Compare domain breakdown (weather vs web vs finance)
3. Look for patterns (e.g., conservative wins on long-context windows?)

**Next iteration**:
- Try 75% conservative, 25% original (blend weights)
- Add performance stack to original (SciPy, FFT) without changing weights
- Test on more seeds (5-10 instead of 3)

---

## Quick Local Test (Optional — Before GPU)

If you want to sanity-check before GPU time:

```bash
cd miner-ops

# Quick 30-second smoke test (local CPU)
uv run cascade score my-generator \
  --pool-dir eval-pool/v1 \
  --seed 42 \
  --heat-budget 30

# Should complete in ~5 minutes
# Won't match GPU scores, but catches major breaks
```

---

## Performance Notes

### GPU Requirements
- **Minimum**: L40S or A100 (mainnet reference)
- **VRAM**: 24GB+ for Toto2-4M training
- **Time**: ~2.5 minutes per seed (120s heat + overhead)
- **Total**: ~15-20 minutes for full 6-run A/B

### CPU Alternative (Slow)
- Can run on CPU but 10-20× slower
- 120s heat budget becomes ~40 minutes per seed
- Only use for smoke tests, not final validation

---

## After A/B Completes

### If Conservative Wins → Add Performance Stack

Once conservative validates, add jan's performance optimizations:

1. **SciPy lfilter** for AR recurrences (4–12× faster)
2. **FFT spectral GP** (1 pass vs 48-pass RFF)
3. **Prefetching thread** (+14% throughput)
4. **2048-row chunks** (6% faster than 1024)
5. **Cached seasonal basis** (@lru_cache)

**Expected**: Match jan's 2.4× throughput edge over j-test

### If Original Wins → Performance Stack First

Try adding optimizations to original config without weight changes:
- Prove performance stack works independently
- Then retry conservative rebalance

---

## Troubleshooting

### "cuda out of memory"
- Reduce batch size in config (if configurable)
- Use smaller heat budget (60s instead of 120s)
- Restart pod to clear VRAM

### "cascade: command not found"
- Run bootstrap script: `bash gpu_pod_bootstrap.sh`
- Or minimal: `uv sync --all-extras`

### "eval-pool not found"
- Transfer eval-pool/v1/ to GPU pod
- Or rebuild: `cascade-pool build --sources openmeteo,wikimedia ...`

### Scores look wrong (very high/low)
- Check you're using geomean (lower is better)
- Verify eval pool has real windows (not synthetic)
- Compare to known baselines (king ~0.137 on local pool)

---

## Next Steps After Validation

1. **If conservative wins**: Deploy to testnet first, then mainnet
2. **Add performance stack**: SciPy, FFT, prefetch
3. **Monitor emissions**: Target >44 TAO (beat j-test)
4. **Iterate**: Once live, continue A/B testing improvements

---

## Files Generated

After A/B completes:
```
miner-ops/
└── ab_results_YYYYMMDD_HHMMSS/
    ├── summary.txt                  # Quick results
    ├── conservative_seed42.log      # Full training logs
    ├── conservative_seed43.log
    ├── conservative_seed44.log
    ├── original_seed42.log
    ├── original_seed43.log
    └── original_seed44.log
```

Save these logs — they're proof of A/B discipline (jan's standard).

---

## Ready?

Transfer files to GPU pod and run:
```bash
bash run_ab_validation.sh
```

Then report back results!

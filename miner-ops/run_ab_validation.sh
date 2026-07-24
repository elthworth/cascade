#!/bin/bash
# A/B Validation: Conservative vs Original Config
# Run on GPU pod with cascade environment

set -e

GENERATOR_DIR="my-generator"
POOL_DIR="eval-pool/v1"
HEAT_BUDGET=120  # 120 seconds per run (jan's standard)
SEEDS=(42 43 44)  # 3-seed validation

echo "========================================="
echo "A/B Validation: Conservative vs Original"
echo "========================================="
echo ""
echo "Generator: $GENERATOR_DIR"
echo "Pool: $POOL_DIR"
echo "Heat budget: ${HEAT_BUDGET}s per run"
echo "Seeds: ${SEEDS[@]}"
echo ""

# Check if we're in the right directory
if [ ! -d "$GENERATOR_DIR" ]; then
    echo "ERROR: $GENERATOR_DIR not found. Run from miner-ops/ directory."
    exit 1
fi

if [ ! -d "$POOL_DIR" ]; then
    echo "ERROR: $POOL_DIR not found. Run from miner-ops/ directory."
    exit 1
fi

# Create results directory
RESULTS_DIR="ab_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# ============================================
# Phase 1: Score CONSERVATIVE config (3 seeds)
# ============================================

echo "=== Phase 1: Testing CONSERVATIVE config ==="
echo ""

cd "$GENERATOR_DIR"

# Ensure conservative config is active
if [ ! -f "config.CONSERVATIVE.json" ]; then
    echo "ERROR: config.CONSERVATIVE.json not found"
    exit 1
fi

cp config.CONSERVATIVE.json config.json
echo "✓ Applied conservative config"

cd ..

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- Conservative config, seed=$seed ---"

    OUTPUT_FILE="$RESULTS_DIR/conservative_seed${seed}.log"

    uv run cascade score "$GENERATOR_DIR" \
        --pool-dir "$POOL_DIR" \
        --seed "$seed" \
        --heat-budget "$HEAT_BUDGET" \
        2>&1 | tee "$OUTPUT_FILE"

    # Extract final score
    SCORE=$(grep -o 'geomean [0-9.]*' "$OUTPUT_FILE" | tail -1 | awk '{print $2}')
    echo "CONSERVATIVE seed $seed: $SCORE" >> "$RESULTS_DIR/summary.txt"

    echo "✓ Conservative seed $seed done: $SCORE"
done

echo ""
echo "=== Conservative config complete ==="
echo ""

# ============================================
# Phase 2: Score ORIGINAL config (3 seeds)
# ============================================

echo "=== Phase 2: Testing ORIGINAL config ==="
echo ""

cd "$GENERATOR_DIR"

# Switch to original config
if [ ! -f "config.ORIGINAL.json" ]; then
    echo "ERROR: config.ORIGINAL.json not found"
    exit 1
fi

cp config.ORIGINAL.json config.json
echo "✓ Applied original config"

cd ..

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- Original config, seed=$seed ---"

    OUTPUT_FILE="$RESULTS_DIR/original_seed${seed}.log"

    uv run cascade score "$GENERATOR_DIR" \
        --pool-dir "$POOL_DIR" \
        --seed "$seed" \
        --heat-budget "$HEAT_BUDGET" \
        2>&1 | tee "$OUTPUT_FILE"

    # Extract final score
    SCORE=$(grep -o 'geomean [0-9.]*' "$OUTPUT_FILE" | tail -1 | awk '{print $2}')
    echo "ORIGINAL seed $seed: $SCORE" >> "$RESULTS_DIR/summary.txt"

    echo "✓ Original seed $seed done: $SCORE"
done

echo ""
echo "=== Original config complete ==="
echo ""

# ============================================
# Phase 3: Compare Results
# ============================================

echo "========================================="
echo "RESULTS SUMMARY"
echo "========================================="
echo ""

cat "$RESULTS_DIR/summary.txt"

echo ""
echo "--- Detailed Analysis ---"
echo ""

# Parse scores
CONS_42=$(grep "CONSERVATIVE seed 42" "$RESULTS_DIR/summary.txt" | awk '{print $4}')
CONS_43=$(grep "CONSERVATIVE seed 43" "$RESULTS_DIR/summary.txt" | awk '{print $4}')
CONS_44=$(grep "CONSERVATIVE seed 44" "$RESULTS_DIR/summary.txt" | awk '{print $4}')

ORIG_42=$(grep "ORIGINAL seed 42" "$RESULTS_DIR/summary.txt" | awk '{print $4}')
ORIG_43=$(grep "ORIGINAL seed 43" "$RESULTS_DIR/summary.txt" | awk '{print $4}')
ORIG_44=$(grep "ORIGINAL seed 44" "$RESULTS_DIR/summary.txt" | awk '{print $4}')

echo "Seed 42: CONSERVATIVE $CONS_42 vs ORIGINAL $ORIG_42"
echo "Seed 43: CONSERVATIVE $CONS_43 vs ORIGINAL $ORIG_43"
echo "Seed 44: CONSERVATIVE $CONS_44 vs ORIGINAL $ORIG_44"
echo ""

# Determine winner (lower is better)
CONS_WINS=0
ORIG_WINS=0

if (( $(echo "$CONS_42 < $ORIG_42" | bc -l) )); then
    echo "Seed 42: CONSERVATIVE wins ✓"
    CONS_WINS=$((CONS_WINS + 1))
else
    echo "Seed 42: ORIGINAL wins"
    ORIG_WINS=$((ORIG_WINS + 1))
fi

if (( $(echo "$CONS_43 < $ORIG_43" | bc -l) )); then
    echo "Seed 43: CONSERVATIVE wins ✓"
    CONS_WINS=$((CONS_WINS + 1))
else
    echo "Seed 43: ORIGINAL wins"
    ORIG_WINS=$((ORIG_WINS + 1))
fi

if (( $(echo "$CONS_44 < $ORIG_44" | bc -l) )); then
    echo "Seed 44: CONSERVATIVE wins ✓"
    CONS_WINS=$((CONS_WINS + 1))
else
    echo "Seed 44: ORIGINAL wins"
    ORIG_WINS=$((ORIG_WINS + 1))
fi

echo ""
echo "========================================="
echo "FINAL VERDICT"
echo "========================================="
echo ""
echo "CONSERVATIVE wins: $CONS_WINS / 3"
echo "ORIGINAL wins: $ORIG_WINS / 3"
echo ""

if [ "$CONS_WINS" -eq 3 ]; then
    echo "✅ DEPLOY CONSERVATIVE CONFIG"
    echo ""
    echo "Conservative config won on ALL 3 seeds (jan's standard)."
    echo "Ready to deploy to mainnet."
    echo ""
    echo "Next steps:"
    echo "1. cd my-generator && cp config.CONSERVATIVE.json config.json"
    echo "2. cascade deploy . --wallet-name h --wallet-hotkey h03 --hub-repo ramsey/suker-miner"

    # Auto-apply conservative (optional, commented out for safety)
    # cd "$GENERATOR_DIR"
    # cp config.CONSERVATIVE.json config.json
    # echo "✓ Auto-applied conservative config"

elif [ "$ORIG_WINS" -eq 3 ]; then
    echo "❌ KEEP ORIGINAL CONFIG"
    echo ""
    echo "Original config won on all 3 seeds."
    echo "Conservative rebalance did not improve score."
    echo ""
    echo "Recommendations:"
    echo "1. Analyze what went wrong (check logs in $RESULTS_DIR)"
    echo "2. Try intermediate rebalance (less aggressive dynamics boost)"
    echo "3. Test performance stack without weight changes"

    # Restore original
    cd "$GENERATOR_DIR"
    cp config.ORIGINAL.json config.json
    echo "✓ Restored original config"

else
    echo "⚠️  MIXED RESULTS - ITERATE"
    echo ""
    echo "Neither config won all 3 seeds."
    echo ""
    echo "Recommendations:"
    echo "1. Analyze per-seed differences (which domains diverge?)"
    echo "2. Try intermediate weights between original and conservative"
    echo "3. Consider domain-specific rebalancing"
    echo "4. DO NOT deploy until clear winner emerges"

    # Keep original by default
    cd "$GENERATOR_DIR"
    cp config.ORIGINAL.json config.json
    echo "✓ Restored original config (default for mixed results)"
fi

echo ""
echo "========================================="
echo "Full logs saved to: $RESULTS_DIR"
echo "========================================="

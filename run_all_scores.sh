#!/usr/bin/env bash
# Run scoring for all three generators and save results
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""  # Force CPU to avoid CUDA determinism issues

POOL_DIR="./miner-ops/eval-pool/v1"
DEVICE="cpu"
SEED="0"
CHAIN_TOML="chain.toml"

echo "=================================================="
echo "Scoring All Generators on Local Eval Pool"
echo "=================================================="
echo ""

# Create results directory
mkdir -p /tmp/cascade_scoring_results
RESULTS_FILE="/tmp/cascade_scoring_results/comparison.txt"

# Clear previous results
> "$RESULTS_FILE"

echo "Starting scoring runs..."
echo "This will take some time (training + evaluation for each generator)"
echo ""

# Function to extract and format results
extract_score() {
    local log_file="$1"
    local generator_name="$2"
    
    if grep -q "score: geomean=" "$log_file"; then
        geomean=$(grep "score: geomean=" "$log_file" | sed 's/.*geomean=\([0-9.]*\).*/\1/')
        echo "$generator_name: geomean=$geomean" | tee -a "$RESULTS_FILE"
        cat "$log_file" | grep -A 10 "score: geomean=" | tee -a "$RESULTS_FILE"
        echo "" | tee -a "$RESULTS_FILE"
    else
        echo "$generator_name: FAILED" | tee -a "$RESULTS_FILE"
        tail -20 "$log_file" | tee -a "$RESULTS_FILE"
        echo "" | tee -a "$RESULTS_FILE"
    fi
}

# Score 1: my-generator (hydra-mix-v1)
echo "=========================================="
echo "1/3: Scoring my-generator (hydra-mix-v1)"
echo "=========================================="
LOG_FILE="/tmp/cascade_scoring_results/my_generator.log"
uv run cascade score ./miner-ops/my-generator \
    --pool-dir "$POOL_DIR" --device "$DEVICE" \
    --seed "$SEED" --chain-toml "$CHAIN_TOML" \
    2>&1 | tee "$LOG_FILE"
extract_score "$LOG_FILE" "my-generator"

# Score 2: ares-v6-fixed
echo "=========================================="
echo "2/3: Scoring ares-v6-fixed"
echo "=========================================="
LOG_FILE="/tmp/cascade_scoring_results/ares.log"
uv run cascade score ./miner-ops/competitors/ares-v6-fixed \
    --pool-dir "$POOL_DIR" --device "$DEVICE" \
    --seed "$SEED" --chain-toml "$CHAIN_TOML" \
    2>&1 | tee "$LOG_FILE"
extract_score "$LOG_FILE" "ares-v6-fixed"

# Score 3: aurora-mix
echo "=========================================="
echo "3/3: Scoring aurora-mix"
echo "=========================================="
LOG_FILE="/tmp/cascade_scoring_results/aurora.log"
uv run cascade score ./miner-ops/competitors/aurora-mix \
    --pool-dir "$POOL_DIR" --device "$DEVICE" \
    --seed "$SEED" --chain-toml "$CHAIN_TOML" \
    2>&1 | tee "$LOG_FILE"
extract_score "$LOG_FILE" "aurora-mix"

# Final summary
echo "=================================================="
echo "FINAL COMPARISON (lower geomean = better)"
echo "=================================================="
cat "$RESULTS_FILE"

echo ""
echo "Full logs saved in: /tmp/cascade_scoring_results/"
echo "=================================================="

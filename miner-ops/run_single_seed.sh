#!/bin/bash
# Run a single seed scoring
set -eo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <config_name> <seed> <results_dir>"
    echo "Example: $0 CONSERVATIVE 42 ab_results_20260724_032307"
    exit 1
fi

CONFIG_NAME=$1
SEED=$2
RESULTS_DIR=$3

GENERATOR_DIR="my-generator"
POOL_DIR="eval-pool/v1"
HEAT_BUDGET_SECONDS=120
TRAIN_HOURS=$(echo "scale=4; $HEAT_BUDGET_SECONDS / 3600" | bc)

echo "========================================="
echo "Running: ${CONFIG_NAME} seed ${SEED}"
echo "========================================="
echo ""

# Apply config
cd "$GENERATOR_DIR"
cp "config.${CONFIG_NAME}.json" config.json
echo "✓ Applied ${CONFIG_NAME} config"
cd ..

# Run scoring
CONFIG_LOWER=$(echo "$CONFIG_NAME" | tr '[:upper:]' '[:lower:]')
OUTPUT_FILE="$RESULTS_DIR/${CONFIG_LOWER}_seed${SEED}.log"

echo "--- ${CONFIG_NAME} config, seed=$SEED ---"

~/.local/bin/uv run cascade score "$GENERATOR_DIR" \
    --pool-dir "$POOL_DIR" \
    --seed "$SEED" \
    --train-hours "$TRAIN_HOURS" \
    2>&1 | tee "$OUTPUT_FILE"

# Extract and save score
SCORE=$(grep -o 'geomean [0-9.]*' "$OUTPUT_FILE" | tail -1 | awk '{print $2}')
echo "${CONFIG_NAME} seed $SEED: $SCORE" >> "$RESULTS_DIR/summary.txt"

echo ""
echo "✓ ${CONFIG_NAME} seed $SEED done: $SCORE"
echo ""

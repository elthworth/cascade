#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# cascade (Bittensor SN91) — GPU-pod bootstrap for the `cascade score` dev loop.
#
# WHERE TO RUN: on a rented GPU pod (L40S / A100 / 4090 all fine for the CHEAP
# heat-budget local scoring). The miner SUBMISSION needs no GPU; this box exists
# only to run the fast iterate-and-score loop against your held-out pool.
#
# WHAT IT DOES: installs uv, syncs the repo with ALL extras (this is the pinned
# cu124 torch==2.4.1 + gpytorch/sklearn/networkx that base_generator and the
# evaluator need — matches the trainer/validator numerics), and verifies the GPU
# is visible to torch.
#
# USAGE:
#   1) Get the repo onto the pod (git clone OR rsync your local tree):
#        git clone https://github.com/TensorLink-AI/cascade && cd cascade
#      (Pods are rsync'd trees in production; a git clone is fine for dev.)
#   2) bash /path/to/gpu_pod_bootstrap.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="${1:-$PWD}"
cd "$REPO_DIR"

echo "== [1/5] install uv (if missing) =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== [2/5] pin interpreter (3.11, matches .python-version) =="
uv python pin 3.11

echo "== [3/5] uv sync --all-extras (cu124 torch 2.4.1 + gpytorch/sklearn/networkx + hippius + chain) =="
# NOTE: --all-extras is REQUIRED — torch lives behind the [train] extra, and
# base_generator's GP/kernel priors need gpytorch/scikit-learn/networkx.
uv sync --all-extras

echo "== [4/5] confirm GPU visible to torch =="
uv run python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no CUDA device — scoring will fall back to CPU (slow but works).")
PY

echo "== [5/5] ready. The score loop: =="
cat <<'EOF'

  # pull the current king and score it on your pool (the baseline to beat)
  uv run cascade fetch king --network finney --out ./king
  uv run cascade score ./king        --pool-dir /path/to/eval-pool --device cuda

  # score your candidate; edit ./my-generator and repeat until you beat the king
  uv run cascade score ./my-generator --pool-dir /path/to/eval-pool --device cuda

  # lower geomean is better. Rotate/expand the pool so you don't overfit it.
EOF
echo "bootstrap complete."

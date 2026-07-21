#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# cascade SN91 — get a REAL score on a clean GPU pod (king vs your candidate).
# Copy-paste these blocks on the pod. Fill the two Hippius Hub creds in step 4.
# Runtime: ~env 5-10 min, ~pool 1-2 min, ~each score 30 min (heat budget).
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── 1. Get the repo ─────────────────────────────────────────────────────────
git clone https://github.com/TensorLink-AI/cascade
cd cascade

# ── 2. Install uv + the interpreter pin ─────────────────────────────────────
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv python pin 3.11

# ── 3. Sync env, THEN add the generator compute libs ────────────────────────
#   --all-extras gives cu124 torch 2.4.1 + hippius + chain + pool tooling, but
#   NOT gpytorch/sklearn/networkx (those are on the generator allowlist, not in
#   cascade's own extras). Local scoring runs the generator in-process under
#   this venv (no sandbox pip-install), so its imports must be present here.
uv sync --all-extras
uv pip install gpytorch scikit-learn networkx        # base_generator's GP/kernel/causal libs
# If a fetched king later errors ModuleNotFoundError, add: statsmodels numba

# confirm the GPU is visible to torch (must print cuda available: True)
uv run python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# ── 4. Hippius Hub READ creds (needed to fetch the king) ────────────────────
export HIPPIUS_HUB_USERNAME="REPLACE_ME"
export HIPPIUS_HUB_PASSWORD="REPLACE_ME"
# (or a single token instead of the two above: export HIPPIUS_HUB_TOKEN=...)

# ── 5. Build a fresh real-world held-out pool ON THE POD ────────────────────
#   Self-contained (no transfer). King + candidate get scored on THIS pool, so
#   the comparison is apples-to-apples. ~453 windows (weather H + web_traffic D).
uv run cascade-pool build --out "$HOME/eval-pool/v1" \
  --sources openmeteo,wikimedia --span-days 210 \
  --max-series-per-source 400 --max-series-total 500 \
  --overwrite --chain-toml chain.toml

# ── 6. Pull the reigning king (the baseline to beat) ────────────────────────
uv run cascade fetch king --network finney --chain-toml chain.toml --out ./king

# ── 7. Our candidate: hydra-mix-v1 (built locally, transfer it to the pod) ──
#   my-generator is our fork of aurora-mix's engine with the three evidence-based
#   improvements (see miner-ops/my-generator/STRATEGY.md). It is NOT on any
#   registry, so copy it from the control box to the pod, e.g.:
#     scp -r <controlbox>:/home/ubuntu/workspace/miner-ops/my-generator ~/cascade/my-generator
#   (or rsync the whole miner-ops/ tree). Then optionally calibrate its knobs to
#   the real pool before scoring (review the print; --write to apply):
uv run python my-generator/calibrate.py --pool-dir "$HOME/eval-pool/v1"
# uv run python my-generator/calibrate.py --pool-dir "$HOME/eval-pool/v1" --write

# ── 8. SCORE both on the same pool (same --seed so it's a fair contract) ────
#   lower geomean = required. --device cuda is required (default is cpu).
uv run cascade score ./king         --pool-dir "$HOME/eval-pool/v1" --device cuda --seed 0 --chain-toml chain.toml
uv run cascade score ./my-generator --pool-dir "$HOME/eval-pool/v1" --device cuda --seed 0 --chain-toml chain.toml

# ── 9. COMPETITORS (testnet) — fetch by pinned digest (anonymous, reproducible) ──
#   Resolved 2026-07-20 from the /main tag of each Hippius Hub repo. Same deps as
#   base_generator (gpytorch/sklearn/networkx already installed in step 3), same
#   pool + seed → directly comparable to king / my-generator above.
ARES="iris999/ares-v6-fixed@sha256:2684974cfc0aa9bfe8517c68a310c6dc6cdf2ebcfcbc1c4ac51a2a748ea74dc9"
AURORA="valor/aurora-mix@sha256:1aa043f1e2c2e7ccaf73924130c22abb85acd5ca6561fbda642896049b3e7279"

uv run cascade fetch "$ARES"   --out ./competitors/ares-v6-fixed
uv run cascade fetch "$AURORA" --out ./competitors/aurora-mix

uv run cascade score ./competitors/ares-v6-fixed --pool-dir "$HOME/eval-pool/v1" --device cuda --seed 0 --chain-toml chain.toml
uv run cascade score ./competitors/aurora-mix    --pool-dir "$HOME/eval-pool/v1" --device cuda --seed 0 --chain-toml chain.toml

# ── read every `score: geomean=...` line (lower = better). Ranking these four
#    (king / my-generator / ares / aurora) on YOUR pool tells you the target to
#    beat. Then iterate my-generator and re-run step 8 (± step 9 to re-rank).
#    NOTE: directional only — your pool ≠ the validator's private rotating pool.

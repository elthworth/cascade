# miner-ops — cascade (Bittensor SN91) mining assets

Operator working directory for competing on SN91 "cascade" (data-generator KOTH).
Kept OUTSIDE the subnet git repo (`../cascade`) so the checkout stays clean.

## What's set up

### (a) Control-box environment  — `../cascade`
A `uv` venv (Python 3.11.14) synced with the **control/dev/pool** extras
(`hippius,chain,dev,deploy,pool-forge`) — everything the miner needs EXCEPT
torch. torch is deliberately absent: this box does wallet / deploy / round /
fetch / pool work (all CPU), and the GPU-only `cascade score` loop runs on a
rented pod.

Run any CLI from the repo with `uv run` (from `../cascade`):
```bash
uv run cascade verify ./my-generator --chain-toml chain.toml
uv run cascade round  --network finney
uv run cascade fetch  king --network finney --out ./king
uv run cascade deploy ./my-generator --network finney \
    --wallet-name <cold> --wallet-hotkey <hot> --hub-repo <ns>/<name>
```

### GPU-pod bootstrap — `gpu_pod_bootstrap.sh`
Run on a rented GPU pod (L40S / A100 / 4090 all fine for cheap heat-budget local
scoring). Installs uv + `uv sync --all-extras` (pinned cu124 torch 2.4.1 +
gpytorch/sklearn/networkx) so `base_generator` dev and `cascade score` both work.
```bash
# on the pod, after cloning the repo:
bash gpu_pod_bootstrap.sh
```

### (b) Held-out eval pool — `eval-pool/v1/`
Real-world series harvested with the subnet's own `cascade-pool build`, in the
exact `<id>.npy` + `metadata.json` format the validator's pool loader consumes
(verified: loads as 453 `EvalWindow`s via `window_source_from_dir`).

| property | value |
|---|---|
| series / windows | 453 (weather=368 hourly, web_traffic=85 daily) |
| geometry | context_length=4096, horizon=64 (matches `[eval]`) |
| seasonal periods | 24 (H), 7 (D) |
| scale spread (\|median\|) | p5≈2.2 · p50≈57 · p95≈1.4e4 (realistic, varied) |
| windows vs mainnet quorum | 453 ≥ `min_windows`=200 ✅ |

Use it as the baseline-to-beat signal (on the GPU pod):
```bash
uv run cascade score ./king        --pool-dir <path>/miner-ops/eval-pool/v1 --device cuda
uv run cascade score ./my-generator --pool-dir <path>/miner-ops/eval-pool/v1 --device cuda
# lower geomean = better
```

## ⚠️ Known limitations / next improvements to the pool
- **Only 2 domains + 1 cluster.** The metadata `source` cluster key is uniform
  (1 cluster). Fine locally (mainnet `min_clusters=0`), but the private eval is
  broader. **Rotate and widen** before trusting the number — add domains/sources.
- **Don't overfit.** A generator tuned to ace this fixed set is exactly what the
  private ROTATING eval punishes. Rebuild periodically:
  ```bash
  # rotate: a fresh cut on a later as-of date → v2 (keep v1 to compare)
  uv run cascade-pool build --out <path>/miner-ops/eval-pool/v2 \
     --sources openmeteo,wikimedia --span-days 210 \
     --max-series-per-source 400 --max-series-total 500 \
     --overwrite --chain-toml chain.toml
  ```
- To add more feeds later: `tsbench_forge` source (needs the forge scraper
  output) or extend `cascade/pool/sources/` with new keyless APIs.

## Competitors (downloaded for study)
`competitors/ares-v6-fixed/` and `competitors/aurora-mix/` — two testnet rivals,
pulled by pinned digest. Both are torch-free TempoPFN mixtures with stratified
interleave; both target the ~53–65% clean weekly web/count eval cluster. See
`my-generator/STRATEGY.md` for the full analysis.

## Our candidate — `my-generator/` (hydra-mix-v1)
Fork of aurora-mix's engine with three evidence-based improvements: (1) activate
the eval-dominant families aurora built but shipped OFF (`web_traffic`,
`seasonal_level`); (2) new `physical_hourly` family for the smooth weather
period-24 cluster both rivals neglect; (3) full 4096 context + `calibrate.py`
data-driven knob fitting. **Verified on the control box** (torch-free subset):
deterministic, contract-valid. `forecast_pfn` path + full `cascade score` need
the GPU pod. Details: `my-generator/STRATEGY.md`.

## Status / not yet done
- [ ] No wallet created yet, no UID registered on netuid 91.
- [x] Candidate built (`my-generator/`, hydra-mix-v1) — needs pod verify + score.
- [ ] Transfer `my-generator/` to the GPU pod (scp/rsync) and score vs king +
      both competitors (pod_score_run.sh steps 7–9).
- [ ] Hippius Hub credentials not set (`HIPPIUS_HUB_USERNAME/PASSWORD`).

# metronome — synthetic time-series data subnet

A Bittensor subnet where miners compete on the quality of **training data**, not
models. It is the dual of [`horizon`](../horizon): horizon scores trained TSFMs
that miners submit; metronome holds the *training process* fixed and scores the
**data generators** that feed it.

## How it works

```
 miner          owner trainer                         validators
 ─────          ─────────────                         ──────────
 generator.py   ┌─ king's generator ─┐  train (fixed   ┌─ pull king ckpt ─┐
   (no weights) │                    │  contract: same │                  │
       │        │  challenger's gen ─┘  arch/seed/      └─ pull chal ckpt ─┘
   commit ──────►   draw corpus → train base model → push 2 ckpts → manifest
   metro-v1:gen          │                                      │
                         └──────────── manifest ────────────────► eval on shared
                                                                  held-out windows
                                                                       │
                                                  paired bootstrap LCB of
                                                  geomean(CRPS, MASE), king vs
                                                  challenger → KOTH decision →
                                                  winner-take-all weights
```

The **central invariant**: in a round, the king's generator and the
challenger's generator are trained into models under a *byte-identical* contract
— same base architecture, epochs, batch/lr, generation seed, and training seed.
The only thing that differs is the generator code. So the downstream eval is a
controlled measurement of **data quality**, not a confound of data + luck +
hyperparameters.

A challenger only takes the throne after winning **`dethrone_cp` consecutive
rounds** by a confidence-bounded margin (paired bootstrap LCB clears the
tenure-adjusted win margin). Weights are pure winner-takes-all.

## Three roles

| role | package | needs GPU | needs chain |
|------|---------|-----------|-------------|
| **miner** | `metronome.miner` | no | to deploy |
| **trainer** (owner) | `metronome.trainer` | yes | to read king / sign manifest |
| **validator** | `metronome.validator` | yes (eval) | to set weights |

## Layout

```
metronome/
  interface/   miner-facing contract (DataGenerator ABC, output checks, static guard)
  eval/        scoring math: CRPS (MWSQL), MASE, paired bootstrap, KOTH decision
  trainer/     owner GPU service: corpus build, fixed contract, train+upload, manifest
  validator/   manifest gate, checkpoint evaluator, KOTH state machine, weights
  miner/       miner CLI: verify, deploy
  shared/      config loader, HF fetch/upload, chain client, manifest schema

docs/
  ARCHITECTURE.md   end-to-end flow, trust model, the controlled-experiment invariant
  INTERFACE.md      the DataGenerator submission contract for miners
scripts/
  example_generator/   a forkable reference generator (also a test fixture)
```

## Console scripts

After `uv sync` / `pip install -e .`:

* `metronome verify <repo_dir>` — run every check the trainer runs (layout,
  static guard, hash-locked deps, **and the determinism check**: your generator
  must produce a byte-identical corpus at a fixed seed).
* `metronome deploy <hf_repo> --revision <40-char-sha> --wallet-name ... --wallet-hotkey ...`
  — commit `metro-v1:gen:hf:<repo>@<sha>` on-chain.
* `metronome-trainer` — the owner training service (`--offline` for a config/seed smoke).
* `metronome-validator` — the validator loop (`--offline` for a state smoke).

Before launching, set `chain.toml [subnet] netuid`, `[training] base_arch_digest`
(sha256 of the frozen base architecture), `[manifest] trainer_hotkey`, and the
repo identifiers.

## Quick start

```bash
pip install -e .                 # core (numpy/scipy/huggingface-hub)
pip install -e '.[dev]'          # + pytest/ruff
python -m pytest tests/unit -q   # pure-numpy tests, no torch/HF/chain needed
```

The trainer's actual base-model training is the **owner's** to implement behind
the `metronome.trainer.contract.BaseTrainer` protocol (the GPU boundary) — see
`docs/ARCHITECTURE.md`. Everything above that boundary is numpy/CPU and tested.

## License

MIT

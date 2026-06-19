# metronome architecture

## The thesis

A time-series foundation model is only as good as the data it was trained on.
metronome makes **synthetic training data** the competitive resource: miners
write data generators, the subnet owner trains a fixed model on each, and the
generator whose data yields the best forecaster wins. By holding the model
architecture and the entire training process constant, the subnet turns a noisy
question ("is this model good?") into a controlled one ("is this *data* good?").

## Roles and data flow

### 1. Miner — submits a generator

A miner writes `generator.py` exposing `Generator(DataGenerator)` and commits a
single on-chain pointer:

```
metro-v1:gen:hf:<org>/<repo>@<40-char-sha>
```

The git SHA pins the generator code, `config.json`, and `requirements.txt`
together. **No model weights** — that is the whole distinction from horizon.
See `docs/INTERFACE.md`.

### 2. Trainer — owner-operated, the GPU boundary

Once per round the trainer:

1. Resolves on-chain commitments to `(hotkey, uid, repo, revision)`.
2. Identifies the reigning **king** (highest-incentive UID on the metagraph) and
   selects **challenger(s)**.
3. Derives one `RoundSeeds` from the round's base seed (the chain block hash):
   a shared `generation_seed` and a shared `training_seed`.
4. For the king and each challenger, **under that one shared seed pair**:
   - materialises the generator repo, runs it in a sandbox, drains a validated
     corpus (`metronome.trainer.corpus`),
   - trains a fresh copy of the base model via the owner's `BaseTrainer`
     (`metronome.trainer.contract`),
   - uploads the checkpoint to HF.
5. Publishes a signed `TrainingManifest` listing both trained-model pointers and
   the corpus/contract digests.

`BaseTrainer` is a `Protocol` — the single GPU-dependent seam. Everything else
in the trainer is numpy/CPU and unit-tested. A reference implementation (e.g. a
Chronos-Bolt-style encoder fine-tune) is the operator's to provide; it must be
**stateless across the king and challenger calls** so no information leaks
between the two training runs.

### 3. Validator — reads the manifest, decides the throne

The validator never trains. Each round it:

1. Reads the current manifest, verifies its signature and that king and
   challenger share the **contract digest** and **base-arch digest** (the
   controlled-experiment gate — `ValidatorRunner.check_manifest`).
2. Pulls both trained checkpoints and scores them on the **same** held-out
   real-world eval windows (`metronome.validator.evaluator`).
3. Runs the paired-bootstrap KOTH verdict (`metronome.eval.koth.evaluate_round`)
   and folds it into the sticky champion state.
4. Sets winner-take-all weights on the reigning king's UID.

## The controlled-experiment invariant

For a round to be a fair measurement of data quality, the king's model and the
challenger's model must differ in **exactly one** thing: the corpus. metronome
enforces this on three sides:

* **Trainer:** one `RoundSeeds` instance is reused for both — identical weight
  initialisation (`training_seed`) and identical generation seed.
* **Manifest:** `contract_digest` (sha256 of the `TrainingContractConfig`) and
  `base_arch_digest` are recorded once and asserted equal for both entries.
* **Validator:** rejects any manifest whose digests don't match its own
  `chain.toml`, so a tampered or mismatched training run can't score.

Auditability: because both seeds derive deterministically from the chain block
hash and every corpus carries a `corpus_digest`, a second honest trainer (or a
suspicious validator) can re-draw the corpus and re-train to confirm the run.

## Scoring

Per window, per model: MASE (Hyndman seasonal-naive denominator) and the gluonts
`MeanWeightedSumQuantileLoss` components `(qloss_per_q, abs_target)`. The KOTH
decision is a **paired bootstrap LCB** on the relative improvement of
`geomean(MWSQL, mean MASE)`, challenger vs king, resampling window indices once
per bag and aggregating MWSQL numerator/denominator before dividing (robust to
near-zero-mean windows). The challenger wins a round iff that LCB clears the
tenure-adjusted win margin on at least `min_windows` common windows.

Dethroning is sticky: `dethrone_cp` consecutive round wins are required; a single
loss or inconclusive round resets the streak. An entrenched king's margin ramps
from `win_margin_start` to `win_margin_end` over `margin_warmup_rounds` of
tenure, so displacing a long-standing king takes a more decisive win.

## Trust model (v1) and the path to decentralisation

v1 centralises training in the owner's trainer and trust in `[manifest]
trainer_hotkey`. This is the pragmatic bootstrap: it makes the controlled
experiment trivially enforceable. The corpus/contract digests already make every
run *reproducible*, which is the hook for decentralising training later (have
validators or a trainer quorum re-derive and challenge a manifest). See
`OPEN_QUESTIONS.md` #1–#2.

## What's implemented vs. a boundary

Implemented and tested (numpy/CPU): the generator contract + output checks, the
static guard, commit/pointer parsing, config, the manifest schema + digests, the
full scoring + KOTH math, the champion state machine, corpus building from a
generator, and the trainer's pairing logic.

Boundaries left for the integrator (clearly marked TODO): the `BaseTrainer` GPU
implementation, the corpus sandbox subprocess, the held-out eval-window source,
manifest signing/verification, and the two live service loops
(`trainer/main.py`, `validator/main.py`).

# Open questions — metronome scaffold

Substantive design calls the initial spec left ambiguous. Each is implemented
with a clear default; the listed location is where to change it if a different
intent was meant. Same convention as horizon's `OPEN_QUESTIONS.md`.

## 1. Manifest trust / training centralisation

**Question.** Validators need to know which trained checkpoint corresponds to
which miner's generator. Who produces that mapping, and why should a validator
trust it?

**Default.** A single owner-operated trainer publishes a signed
`TrainingManifest` to an owner-controlled HF dataset repo (`[manifest]
hf_dataset_repo`); validators trust manifests signed by `[manifest]
trainer_hotkey` only. Training is centralised in v1 because it makes the
controlled-experiment invariant trivially enforceable.

**Flip point.** `metronome/shared/manifest.py::verify_signature` (currently a
presence check — wire to bittensor keypair verification over
`canonical_body()`). The decentralisation path: every corpus carries a
`corpus_digest` and every run a `contract_digest`, so a validator or a trainer
quorum can re-derive the corpus from the pinned generator + seed and re-train to
challenge a manifest. Moving to a re-derivation challenge protocol is the
milestone that removes the single trusted trainer.

## 2. Generation sandbox

**Question.** Generators are miner-controlled code. How isolated must their
execution be?

**Default.** Two layers: a cheap AST static guard at submit time
(`interface/static_guard.py`) and an intended network-isolated, rlimited
subprocess at run time. The subprocess is a TODO —
`trainer/corpus.py::run_in_sandbox` currently delegates to the in-process
`build_corpus`, which is fine for trusted offline runs but **not** for
adversarial mainnet use.

**Flip point.** `metronome/trainer/corpus.py::run_in_sandbox`. Mirror horizon's
`validator/scorer/sandbox.py` (network namespace, disk + rlimit, pipe the corpus
back). Until then the trainer must only be pointed at trusted generators.

## 3. King identity across rounds

**Question.** The trainer must train the reigning king, but the dethrone
decision is the validators'. How does the trainer learn who the king is without
re-deciding it?

**Default.** King identity flows validators → chain weights → trainer. The
trainer reads the highest-incentive UID on the metagraph as the reigning king
(`plan_round(..., king_hotkey=<highest incentive>)`); validators are the sole
authority for dethroning and set weights accordingly. On a vacant throne
(genesis or king deregistered) the lowest-UID resolvable generator is promoted
to interim king so there is always something to defend.

**Flip point.** `metronome/trainer/loop.py::plan_round` (interim-king choice) and
the live loop's king lookup (TODO in `trainer/main.py`). An alternative is an
authoritative owner-maintained king pointer alongside the manifest; that
re-centralises the decision and is not the default.

## 4. Challengers per round

**Question.** How many challengers does the trainer train and the validator
judge per round?

**Default.** `TrainerRunner.run_round(..., max_challengers=1)` — one challenger
per round, the lowest-UID non-king resolvable generator. Simple and cheap (two
trainings per round). Rotating fairly through the field, or batching multiple
challengers into one manifest, is a straightforward extension.

**Flip point.** `metronome/trainer/loop.py::plan_round` /
`TrainerRunner.run_round`, and `validator/loop.py::process_round` (which today
reads the single `king`/`challenger` pair from the manifest).

## 5. Shared training + generation seed

**Question.** Should the king and challenger share the generation seed and the
training seed, or get independent ones?

**Default.** Both seeds are **shared** across king and challenger in a round
(`trainer/contract.py::RoundSeeds.derive`). Shared `training_seed` means
identical weight init and data-order RNG (the controlled experiment); shared
`generation_seed` means neither generator draws a "luckier" data seed. Both
derive deterministically from the chain block hash.

**Flip point.** `metronome/trainer/contract.py::RoundSeeds.derive`. If you want
per-miner generation seeds (so a generator can't tune to one fixed seed), give
each its own `generation_seed` while keeping `training_seed` shared — but note
that weakens reproducibility unless the per-miner seed is also chain-derived.

## 6. Eval-window source

**Question.** Where do the held-out real-world eval windows come from?

**Default.** Left as an injectable boundary. `chain.toml [eval] eval_dataset`
names the dataset (default `gift-eval`); `validator/loop.py` takes the window
list as an argument so it's testable, and the concrete loader (pull GIFT-Eval /
a held-out corpus, slice into `EvalWindow`s seeded by the round) is the TODO. It
must produce the **same** windows for the king and challenger in a round.

**Flip point.** Add a `WindowSource` to `metronome/validator/` and wire it into
the live validator loop (TODO in `validator/main.py`). Reusing horizon's
`horizon-forge` data source is a reasonable option.

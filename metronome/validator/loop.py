"""Validator loop — manifest → eval → KOTH decision → weights.

The validator never trains. Each round it:

1. Reads the current :class:`TrainingManifest` from the owner dataset repo and
   verifies its signature + that king and challenger share the contract digest
   (the controlled-experiment guarantee).
2. Pulls the king's and challenger's trained checkpoints and scores both on the
   *same* held-out eval windows.
3. Runs the paired-bootstrap KOTH verdict and folds it into the sticky
   champion state (``dethrone_cp`` consecutive wins to take the throne).
4. Sets winner-take-all weights on the reigning king's UID.

The pure orchestration in :meth:`ValidatorRunner.process_round` is testable by
injecting ``evaluate_fn`` and ``windows``; HF + torch + chain are isolated
behind the defaults.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..eval.koth import RoundResult, evaluate_round
from ..eval.scoring import WindowScore
from ..eval.window import EvalWindow
from ..shared.config import ChainConfig
from ..shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    parse_trained_pointer,
)
from . import state as state_mod
from .state import ChampionState, StateTransition

log = logging.getLogger("metronome.validator")

# Resolve a trained entry to its per-window scores on the eval set.
EvaluateFn = Callable[[TrainedEntry, list[EvalWindow]], list[WindowScore]]


@dataclass(frozen=True)
class RoundOutcome:
    result: RoundResult
    transition: StateTransition


@dataclass
class ValidatorRunner:
    cfg: ChainConfig
    state: ChampionState = field(default_factory=ChampionState)
    evaluate_fn: EvaluateFn | None = None     # injected in tests; defaults to HF+torch
    hf_cache_dir: Path | None = None
    hf_token: str | None = None
    device: str = "cpu"

    # ── manifest gating ─────────────────────────────────────────────────────

    def check_manifest(self, manifest: TrainingManifest) -> str | None:
        """Return a rejection reason string, or None if the manifest is usable.

        Enforces the contract-digest match (king and challenger trained under
        the same terms) and that the manifest targets our configured base
        architecture and eval dataset. Signature verification is delegated to
        :func:`metronome.shared.manifest.verify_signature` (TODO).
        """
        want_contract = contract_digest(self.cfg.training)
        if manifest.contract_digest != want_contract:
            return f"contract_digest_mismatch: {manifest.contract_digest} != {want_contract}"
        if manifest.base_arch_digest != self.cfg.training.base_arch_digest:
            return "base_arch_digest_mismatch"
        if manifest.eval_dataset != self.cfg.eval.eval_dataset:
            return "eval_dataset_mismatch"
        return None

    # ── per-round decision ──────────────────────────────────────────────────

    def _evaluate(self, entry: TrainedEntry, windows: list[EvalWindow]) -> list[WindowScore]:
        if self.evaluate_fn is not None:
            return self.evaluate_fn(entry, windows)
        # Default path: fetch the checkpoint and score it (HF + torch).
        from ..shared.hf import fetch_revision
        from .evaluator import evaluate_checkpoint

        pointer = parse_trained_pointer(entry.trained_pointer)
        if pointer is None:
            raise ValueError(f"malformed trained_pointer: {entry.trained_pointer!r}")
        repo, revision = pointer
        fetched = fetch_revision(repo, revision, cache_dir=self.hf_cache_dir, token=self.hf_token)
        return evaluate_checkpoint(
            fetched.local_dir, windows, num_samples=self.cfg.eval.num_samples, device=self.device
        )

    def process_round(
        self,
        manifest: TrainingManifest,
        windows: list[EvalWindow],
        base_seed: int | str,
    ) -> RoundOutcome | None:
        """Evaluate one manifest against the eval windows and update state.

        Returns None (king holds, no state change) when the manifest carries no
        challenger or fails the contract gate. Otherwise returns the round
        outcome with the (already-applied) state transition.
        """
        reason = self.check_manifest(manifest)
        if reason is not None:
            log.warning("rejecting manifest round=%s: %s", manifest.round_id, reason)
            return None

        king_entry = manifest.entry_for_role("king")
        chal_entry = manifest.entry_for_role("challenger")
        if king_entry is None or chal_entry is None:
            log.info("manifest round=%s has no king/challenger pair; king holds", manifest.round_id)
            return None

        king_scores = self._evaluate(king_entry, windows)
        chal_scores = self._evaluate(chal_entry, windows)

        result = evaluate_round(
            king_scores,
            chal_scores,
            self.cfg.koth_params(),
            seed=base_seed,
            king_tenure_rounds=self.state.tenure_rounds,
        )
        transition = state_mod.apply_round(
            self.state,
            challenger_hotkey=chal_entry.miner_hotkey,
            challenger_uid=chal_entry.miner_uid,
            result=result,
            dethrone_cp=self.cfg.scoring.dethrone_cp,
        )
        self.state = transition.state
        log.info(
            "round=%s lcb=%.4f margin=%.4f win=%s %s king=%s tenure=%d",
            manifest.round_id, result.lcb, result.margin, result.challenger_wins_round,
            transition.note, self.state.king_hotkey, self.state.tenure_rounds,
        )
        return RoundOutcome(result=result, transition=transition)


def build_runner(
    *,
    chain_toml: Path | None = None,
    hf_cache_dir: Path | None = None,
    device: str = "cpu",
) -> ValidatorRunner:
    """Construct a runner from ``chain.toml``. Wallet/chain wiring for live
    weight-setting is attached by ``metronome-validator`` (see main.py)."""
    from ..shared.config import load_chain_config

    cfg = load_chain_config(chain_toml)
    return ValidatorRunner(cfg=cfg, hf_cache_dir=hf_cache_dir, device=device)

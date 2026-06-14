"""The fixed training contract and the pluggable base-trainer protocol.

metronome's central invariant: in a given round, the king's generator and the
challenger's generator are trained into models under *byte-identical* terms —
same base architecture, same epochs/batch/lr, same generation seed, same
training seed. The ONLY difference is the generator code that produced the
corpus. That is what turns the downstream eval into a controlled measurement of
data quality rather than a confound of data + luck + hyperparameters.

Seeds are shared across king and challenger on purpose:

* ``generation_seed`` — passed to each generator's ``__init__``. Same value for
  both so neither draws a "luckier" data seed; each generator is still
  deterministic in it.
* ``training_seed`` — model init + data-order RNG inside the base trainer. Same
  value for both so weight initialisation and shuffling are identical.

Both derive deterministically from the round's base seed (the chain block hash
at round start), so a second honest trainer reproduces the exact run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..shared.config import TrainingContractConfig


def _mix(base_seed: int, tag: str, salt: int = 0) -> int:
    """Deterministically mix a base seed, a string tag, and an int salt into a
    64-bit seed. Stable across platforms (blake2b)."""
    h = hashlib.blake2b(
        f"{base_seed}:{tag}:{salt}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(h, "big", signed=False)


@dataclass(frozen=True)
class RoundSeeds:
    """The two seeds for a round, shared identically across king and challenger."""

    base_seed: int
    generation_seed: int
    training_seed: int

    @classmethod
    def derive(cls, base_seed: int, contract: TrainingContractConfig) -> RoundSeeds:
        return cls(
            base_seed=base_seed,
            generation_seed=_mix(base_seed, "generation"),
            training_seed=_mix(base_seed, "training", contract.train_seed_salt),
        )


@dataclass(frozen=True)
class TrainResult:
    """What a :class:`BaseTrainer` returns after training one model."""

    local_dir: Path        # directory holding the trained checkpoint to upload
    param_count: int
    train_seconds: float


class BaseTrainer(Protocol):
    """Owner-supplied training backend.

    Implementations train a fresh copy of the base architecture on ``corpus``
    and write a complete, uploadable checkpoint into ``out_dir``. The same
    instance is reused for king and challenger within a round, so it MUST be
    stateless across calls beyond the (immutable) base architecture — any
    leakage from king's run into challenger's would break the controlled
    comparison.

    This Protocol is the GPU boundary: everything above it in metronome is
    numpy/CPU and unit-testable; the concrete trainer (e.g. a Chronos-style
    encoder fine-tune) is the operator's to provide. See
    ``docs/ARCHITECTURE.md`` for the reference shape.
    """

    def train(
        self,
        corpus: Sequence[np.ndarray],
        contract: TrainingContractConfig,
        *,
        training_seed: int,
        out_dir: Path,
    ) -> TrainResult:
        ...

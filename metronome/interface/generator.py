"""DataGenerator — the miner-facing contract.

Every submitted generator must subclass :class:`DataGenerator`. The generator
is the adapter between the trainer's standard corpus-building protocol and
whatever arbitrary synthetic-data process the miner has designed.

The contract is intentionally narrow:

* construction takes the local repo directory and a deterministic ``seed``
* ``generate(n_series)`` yields exactly ``n_series`` univariate float series
* output is bounded: per-series length in ``[min_length, max_length]`` and a
  global cap on total emitted points, both enforced by the trainer

Determinism is load-bearing. The trainer seeds every generator from the chain
block hash at the training round's start, so two honest trainers (or a
validator re-running the trainer to audit) draw the *same* corpus. A generator
whose output depends on wall-clock, process entropy, or un-seeded RNG breaks
auditability and is rejected by the determinism check in
``metronome verify``.

The on-chain submission is a single pointer string
``metro-v1:gen:hf:<repo>@<sha>``; the git SHA pins the full HF tree — generator
code, ``config.json``, and ``requirements.txt`` — together. No model weights
are part of a generator submission (the trainer produces the weights).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np


class DataGenerator(ABC):
    """Standard interface every submitted generator must implement.

    Implementations are loaded from a miner-controlled HF repo at a pinned git
    SHA. The subclass MUST be importable as ``generator.Generator`` from the
    cloned repo root. The trainer rejects generators whose code imports
    anything on the static-guard blocked list (see
    :mod:`metronome.interface.static_guard`) and runs them inside a
    network-isolated sandbox.
    """

    @abstractmethod
    def __init__(self, config_dir: str, *, seed: int) -> None:
        """Construct from a local repo directory and a deterministic seed.

        ``config_dir`` contains the materialised HF repo at the pinned
        revision: ``config.json``, ``generator.py``, and ``requirements.txt``
        (already installed by the trainer before this constructor runs).

        ``seed`` is the only source of randomness the generator is allowed to
        use. Derive every RNG from it (``np.random.default_rng(seed)``); do not
        read the system clock, ``os.urandom``, or any un-seeded global RNG. The
        constructor MUST NOT touch the network.
        """

    @abstractmethod
    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        """Yield exactly ``n_series`` univariate training series.

        Each yielded value is a 1-D ``float`` ``np.ndarray`` (finite, no NaN or
        inf). Series lengths may vary but must fall within the configured
        ``[min_length, max_length]`` band; the trainer validates every series
        with :func:`check_series` and aborts the round if any fails.

        The sequence MUST be a deterministic function of the ``seed`` passed to
        ``__init__`` and of ``n_series`` only.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable generator name, for operator logs."""


# ─────────────────────────────── output checks ─────────────────────────────


def check_series(
    arr: object,
    *,
    min_length: int,
    max_length: int,
    index: int | None = None,
) -> None:
    """Validate a single emitted series. Raises ``ValueError`` on any problem.

    Used by the trainer while draining ``generate`` and by ``metronome verify``
    on the miner side so a miner sees the same failure locally. The trainer
    catches the error, marks the generator's training run failed, and the
    challenger simply doesn't qualify this round — a bad generator can never
    poison the king.
    """
    where = "" if index is None else f" (series {index})"
    if not isinstance(arr, np.ndarray):
        raise ValueError(
            f"generate must yield np.ndarray{where}; got {type(arr).__name__}"
        )
    if arr.ndim != 1:
        raise ValueError(f"series must be 1-D{where}; got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"series dtype must be floating{where}; got {arr.dtype}")
    n = int(arr.shape[0])
    if n < min_length or n > max_length:
        raise ValueError(
            f"series length {n} outside [{min_length}, {max_length}]{where}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"series has non-finite values{where}")


def drain_generator(
    gen: DataGenerator,
    n_series: int,
    *,
    min_length: int,
    max_length: int,
    max_total_points: int,
) -> list[np.ndarray]:
    """Pull ``n_series`` series from ``gen``, validating each one.

    Enforces the per-series length band and a global cap on total emitted
    points (a memory / time guard against a generator that emits a few
    enormous series). Raises ``ValueError`` if the generator yields the wrong
    count, a malformed series, or blows the point budget.

    Returns the materialised list of series in yield order. The trainer hashes
    this list (see :func:`metronome.shared.manifest.corpus_digest`) so the
    corpus is reproducible and auditable.
    """
    if n_series <= 0:
        raise ValueError(f"n_series must be positive; got {n_series}")
    out: list[np.ndarray] = []
    total = 0
    for i, arr in enumerate(gen.generate(n_series)):
        if i >= n_series:
            raise ValueError(
                f"generate yielded more than n_series={n_series} series"
            )
        check_series(arr, min_length=min_length, max_length=max_length, index=i)
        total += int(arr.shape[0])
        if total > max_total_points:
            raise ValueError(
                f"total emitted points {total} exceeds cap {max_total_points}"
            )
        out.append(np.ascontiguousarray(arr, dtype=np.float64))
    if len(out) != n_series:
        raise ValueError(
            f"generate yielded {len(out)} series; expected exactly {n_series}"
        )
    return out

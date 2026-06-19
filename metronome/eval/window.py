"""EvalWindow — one held-out forecasting task.

Numpy-only by design: the eval/scoring path carries no torch dependency so the
statistics are unit-testable in a minimal environment. The validator's
evaluator (which loads the trained model and needs torch) converts model output
to numpy sample arrays and feeds them through :mod:`.scoring`.

A window is a single real-world series split into ``history`` (the context the
model sees) and ``target`` (the held-out continuation it must forecast). The
eval set is the *same* for the king's model and the challenger's model — that
shared, fixed, real-world set is what makes the KOTH comparison a controlled
measurement of data quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EvalWindow:
    """One context/target split of a held-out series.

    Attributes:
        series_id: stable identifier, used for logging and pairing.
        history: shape ``(L,)`` — the context the model conditions on.
        target: shape ``(H,)`` — the held-out continuation to forecast.
        metadata: free-form; ``freq`` / ``seasonal_period`` drive MASE
            seasonality (see :func:`metronome.eval.scoring`).
    """

    series_id: str
    history: np.ndarray
    target: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.history.ndim != 1:
            raise ValueError(f"history must be 1-D; got {self.history.shape}")
        if self.target.ndim != 1:
            raise ValueError(f"target must be 1-D; got {self.target.shape}")

    @property
    def horizon(self) -> int:
        return int(self.target.shape[0])

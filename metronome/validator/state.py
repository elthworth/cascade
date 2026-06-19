"""Champion state and the sticky dethrone machine.

The per-round statistical verdict lives in :mod:`metronome.eval.koth`; this
module owns the *stateful* part: a challenger must win ``dethrone_cp``
**consecutive** rounds before it takes the throne. A single lost or
inconclusive round resets its streak. The king's ``tenure_rounds`` feeds the
margin schedule (an entrenched king is harder to displace).

State is a small, JSON-serialisable record so it can be persisted to the
validator's state DB (``[validator] state_db_path``) and survive restarts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from ..eval.koth import RoundResult


@dataclass(frozen=True)
class ChampionState:
    """The validator's view of the throne.

    Attributes:
        king_hotkey / king_uid: the reigning champion. None before genesis.
        tenure_rounds: rounds the current king has held the throne.
        streaks: per-challenger-hotkey count of *consecutive* round wins.
        rounds_seen: total rounds processed (monotonic; used for logging).
    """

    king_hotkey: str | None = None
    king_uid: int | None = None
    tenure_rounds: int = 0
    streaks: dict[str, int] = field(default_factory=dict)
    rounds_seen: int = 0


@dataclass(frozen=True)
class StateTransition:
    state: ChampionState
    dethroned: bool
    new_king_hotkey: str | None
    note: str


def genesis(king_hotkey: str, king_uid: int) -> ChampionState:
    """Seed the throne with an initial king (the owner baseline or first miner)."""
    return ChampionState(king_hotkey=king_hotkey, king_uid=king_uid)


def apply_round(
    state: ChampionState,
    *,
    challenger_hotkey: str,
    challenger_uid: int,
    result: RoundResult,
    dethrone_cp: int,
) -> StateTransition:
    """Fold one round's result into the champion state.

    Pure: returns a new :class:`ChampionState`. The streak only advances on a
    conclusive win; an inconclusive round (too few windows) leaves the streak
    untouched but still counts the king's tenure.
    """
    rounds_seen = state.rounds_seen + 1

    if result.inconclusive:
        # No decision: king holds, tenure advances, streaks unchanged.
        return StateTransition(
            state=replace(state, tenure_rounds=state.tenure_rounds + 1, rounds_seen=rounds_seen),
            dethroned=False,
            new_king_hotkey=state.king_hotkey,
            note="inconclusive",
        )

    streaks = dict(state.streaks)
    if result.challenger_wins_round:
        streaks[challenger_hotkey] = streaks.get(challenger_hotkey, 0) + 1
    else:
        streaks.pop(challenger_hotkey, None)

    if streaks.get(challenger_hotkey, 0) >= dethrone_cp:
        # Dethrone: challenger becomes king; tenure and all streaks reset.
        return StateTransition(
            state=ChampionState(
                king_hotkey=challenger_hotkey,
                king_uid=challenger_uid,
                tenure_rounds=0,
                streaks={},
                rounds_seen=rounds_seen,
            ),
            dethroned=True,
            new_king_hotkey=challenger_hotkey,
            note=f"dethroned after {dethrone_cp} consecutive wins",
        )

    return StateTransition(
        state=replace(
            state,
            tenure_rounds=state.tenure_rounds + 1,
            streaks=streaks,
            rounds_seen=rounds_seen,
        ),
        dethroned=False,
        new_king_hotkey=state.king_hotkey,
        note=("win" if result.challenger_wins_round else "loss"),
    )


def dumps(state: ChampionState) -> str:
    return json.dumps(
        {
            "king_hotkey": state.king_hotkey,
            "king_uid": state.king_uid,
            "tenure_rounds": state.tenure_rounds,
            "streaks": state.streaks,
            "rounds_seen": state.rounds_seen,
        },
        sort_keys=True,
    )


def loads(text: str) -> ChampionState:
    obj = json.loads(text)
    return ChampionState(
        king_hotkey=obj.get("king_hotkey"),
        king_uid=obj.get("king_uid"),
        tenure_rounds=int(obj.get("tenure_rounds", 0)),
        streaks={str(k): int(v) for k, v in (obj.get("streaks") or {}).items()},
        rounds_seen=int(obj.get("rounds_seen", 0)),
    )

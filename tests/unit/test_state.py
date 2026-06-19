"""Champion state machine: sticky dethroning over consecutive wins."""

from __future__ import annotations

from metronome.eval.koth import RoundResult
from metronome.validator.state import apply_round, dumps, genesis, loads


def _win() -> RoundResult:
    return RoundResult(True, lcb=0.3, margin=0.05, n_windows=300,
                       king_geomean=1.0, chal_geomean=0.7, inconclusive=False)


def _loss() -> RoundResult:
    return RoundResult(False, lcb=0.0, margin=0.05, n_windows=300,
                       king_geomean=1.0, chal_geomean=1.0, inconclusive=False)


def _inconclusive() -> RoundResult:
    return RoundResult(False, lcb=float("nan"), margin=0.05, n_windows=5,
                       king_geomean=1.0, chal_geomean=0.9, inconclusive=True)


def test_three_consecutive_wins_dethrone():
    st = genesis("king", 0)
    for _ in range(2):
        t = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=3)
        st = t.state
        assert not t.dethroned
    t = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=3)
    assert t.dethroned
    assert t.state.king_hotkey == "chal"
    assert t.state.king_uid == 1
    assert t.state.tenure_rounds == 0
    assert t.state.streaks == {}


def test_loss_resets_streak():
    st = genesis("king", 0)
    st = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=3).state
    st = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_loss(), dethrone_cp=3).state
    assert st.streaks.get("chal", 0) == 0
    # Two more wins is not enough now — streak restarts.
    st = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=3).state
    t = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=3)
    assert not t.dethroned
    assert t.state.king_hotkey == "king"


def test_inconclusive_holds_throne_without_touching_streak():
    st = genesis("king", 0)
    st = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=3).state
    t = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_inconclusive(), dethrone_cp=3)
    assert not t.dethroned
    assert t.state.streaks.get("chal") == 1  # untouched
    assert t.state.tenure_rounds == 2


def test_tenure_increments_while_king_holds():
    st = genesis("king", 0)
    for _ in range(4):
        st = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_loss(), dethrone_cp=3).state
    assert st.tenure_rounds == 4


def test_state_round_trips_through_json():
    st = genesis("king", 3)
    st = apply_round(st, challenger_hotkey="c", challenger_uid=2, result=_win(), dethrone_cp=3).state
    again = loads(dumps(st))
    assert again == st

"""Tests for the standings-based competitiveness signal and the historical
upset forward pass (correctness + temporal integrity)."""

from __future__ import annotations

import pandas as pd

from src.core.features import _running_upset
from src.sports.standings import competitiveness


# ---------------------------------------------------------------- competitiveness
def test_competitiveness_close_vs_far() -> None:
    table = {
        "AAA": {"strength": 2.4, "games": 10},   # title pace
        "BBB": {"strength": 2.3, "games": 10},   # right behind
        "ZZZ": {"strength": 0.4, "games": 10},   # relegation pace
    }
    close = competitiveness("AAA", "BBB", table, "soccer")
    lopsided = competitiveness("AAA", "ZZZ", table, "soccer")
    assert close > lopsided
    assert 0.0 <= lopsided <= close <= 1.0


def test_competitiveness_insufficient_data_is_neutral() -> None:
    table = {"AAA": {"strength": 3.0, "games": 1}, "BBB": {"strength": 0.0, "games": 1}}
    assert competitiveness("AAA", "BBB", table, "soccer") == 0.5   # < 3 games -> neutral
    assert competitiveness("AAA", "MISSING", table, "soccer") == 0.5


# ------------------------------------------------------------------------ upset
def _match(iid, day, home, away, hs, as_, league="eng.1", season=2024) -> dict:
    return dict(item_id=iid, league=league, season=season,
                date=pd.Timestamp(f"2024-01-{day:02d}"),
                home=home, away=away, home_score=hs, away_score=as_)


def _soccer_upset(df: pd.DataFrame) -> pd.Series:
    return _running_upset(df, win_pts=3.0, draw_pts=1.0, scale=1.5,
                          min_games=3, group_cols=["league", "season"])


def _season_where_weak_beats_strong() -> pd.DataFrame:
    # STR wins its first 3, WEK loses its first 3, then WEK beats STR at home.
    rows = [
        _match("s1", 1, "STR", "C", 2, 0), _match("s2", 2, "STR", "D", 2, 0),
        _match("s3", 3, "STR", "E", 2, 0),                       # STR ppg 3.0 after 3
        _match("w1", 1, "F", "WEK", 2, 0), _match("w2", 2, "G", "WEK", 2, 0),
        _match("w3", 3, "H", "WEK", 2, 0),                       # WEK ppg 0.0 after 3
        _match("up", 10, "WEK", "STR", 1, 0),                    # the upset (home WEK wins)
    ]
    return pd.DataFrame(rows)


def test_upset_weak_beats_strong_is_high() -> None:
    df = _season_where_weak_beats_strong()
    up = _soccer_upset(df)
    assert up.loc[df.index[-1]] == 1.0        # gap 3.0/1.5 -> capped at 1.0
    assert up.iloc[:6].sum() == 0.0           # first-3-games rows never flag (min_games)


def test_favorite_win_and_draw_are_not_upsets() -> None:
    df = _season_where_weak_beats_strong()
    df.loc[df.index[-1], ["home_score", "away_score"]] = [0, 1]   # STR (away) wins as expected
    assert _soccer_upset(df).iloc[-1] == 0.0
    df.loc[df.index[-1], ["home_score", "away_score"]] = [1, 1]   # draw
    assert _soccer_upset(df).iloc[-1] == 0.0


def test_no_lookahead() -> None:
    """Perturbing a LATER match must not change any EARLIER match's upset."""
    df = _season_where_weak_beats_strong()
    # add a later fixture whose result we will flip
    later = pd.concat([df, pd.DataFrame([_match("late", 20, "STR", "WEK", 5, 0)])],
                      ignore_index=True)
    base = _soccer_upset(later)

    perturbed = later.copy()
    perturbed.loc[perturbed.index[-1], ["home_score", "away_score"]] = [0, 5]  # flip the last result
    after = _soccer_upset(perturbed)

    # every row BEFORE the perturbed one is byte-identical
    assert base.iloc[:-1].equals(after.iloc[:-1])

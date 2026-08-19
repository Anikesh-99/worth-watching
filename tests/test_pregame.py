"""Temporal-integrity tests for pre-game features (the model's credibility).

The load-bearing property: an event's features must depend ONLY on prior
events. `test_no_lookahead` proves it directly — perturbing a later game leaves
every earlier game's features byte-for-byte identical.
"""

from __future__ import annotations

import pandas as pd

from src.sports.pregame import build_f1_pregame, build_nba_pregame


def _nba() -> pd.DataFrame:
    # a small chronological slate within one season
    return pd.DataFrame([
        dict(item_id=f"g{i}", season=2024, date=pd.Timestamp("2024-01-01") + pd.Timedelta(f"{i}D"),
             home=h, away=a, home_score=hs, away_score=as_, is_playoff=0)
        for i, (h, a, hs, as_) in enumerate([
            ("BOS", "NYK", 110, 100),   # BOS beats NYK
            ("MIA", "BOS", 99, 105),    # BOS beats MIA (away)
            ("NYK", "MIA", 120, 118),   # NYK beats MIA
            ("BOS", "MIA", 101, 100),   # BOS v MIA again (h2h exists now)
        ])
    ])


def _ex(df: pd.DataFrame, val: float = 0.5) -> dict:
    return {i: val for i in df["item_id"]}


def test_first_game_uses_neutral_defaults() -> None:
    df = _nba()
    feats = build_nba_pregame(df, _ex(df)).set_index("item_id")
    r0 = feats.loc["g0"]
    assert r0["standings_gap"] == 0.0          # both 0.5 win% -> gap 0
    assert r0["combined_winpct"] == 1.0        # 0.5 + 0.5
    assert r0["home_form"] == 0.5 and r0["away_form"] == 0.5
    assert r0["h2h_excitement"] == 0.5         # no prior meeting


def test_state_updates_after_result() -> None:
    df = _nba()
    feats = build_nba_pregame(df, _ex(df)).set_index("item_id")
    # g1 is MIA(home) v BOS(away); BOS won g0, so BOS away_winpct should be 1.0
    assert feats.loc["g1", "away_form"] == 1.0
    # g3 BOS v MIA is the 2nd meeting -> h2h has one prior game's excitement
    assert feats.loc["g3", "h2h_excitement"] == 0.5   # prior meeting g1 had ex 0.5


def test_no_lookahead() -> None:
    df = _nba()
    base = build_nba_pregame(df, _ex(df)).set_index("item_id")

    # Perturb the LAST game's result AND its excitement, rebuild.
    df2 = df.copy()
    df2.loc[df2["item_id"] == "g3", ["home_score", "away_score"]] = [200, 1]
    ex2 = _ex(df); ex2["g3"] = 1.0
    after = build_nba_pregame(df2, ex2).set_index("item_id")

    # Every earlier game's feature row must be unchanged.
    cols = ["standings_gap", "combined_winpct", "home_form", "away_form", "h2h_excitement"]
    for g in ["g0", "g1", "g2"]:
        pd.testing.assert_series_equal(base.loc[g, cols], after.loc[g, cols])


def test_f1_circuit_history_builds_forward() -> None:
    f1 = pd.DataFrame([
        dict(item_id="r1", season=2023, round=1, country="Italy", date=pd.Timestamp("2023-04-01")),
        dict(item_id="r2", season=2023, round=2, country="Italy", date=pd.Timestamp("2023-05-01")),
    ])
    feats = build_f1_pregame(f1, {"r1": 0.9, "r2": 0.2}).set_index("item_id")
    assert feats.loc["r1", "circuit_hist"] == 0.5     # first visit -> neutral
    assert feats.loc["r2", "circuit_hist"] == 0.9     # sees only r1's excitement
    assert feats.loc["r2", "stakes"] == 1.0           # round 2 of 2

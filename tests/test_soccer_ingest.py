"""Tests for the soccer (PL + CL) vertical — network-free synthetic events."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.core.features import UNIFIED_FEATURES, normalize_soccer
from src.sports.soccer_ingest import MATCH_COLUMNS, SoccerIngest


def _event(home_goals: list[int], away_goals: list[int], reds: int = 0) -> dict:
    """Build a minimal ESPN-shaped completed match from goal minutes per side."""
    details = []
    for m in home_goals:
        details.append({"scoringPlay": True, "clock": {"displayValue": f"{m}'"},
                        "team": {"id": "H"}, "scoreValue": 1})
    for m in away_goals:
        details.append({"scoringPlay": True, "clock": {"displayValue": f"{m}'"},
                        "team": {"id": "A"}, "scoreValue": 1})
    for _ in range(reds):
        details.append({"redCard": True, "clock": {"displayValue": "70'"}})

    def comp(side, tid, n):
        return {"homeAway": side, "score": str(n), "team": {"id": tid, "abbreviation": tid, "logo": "x.png"}}
    return {
        "id": "m1", "date": "2024-04-07T14:00Z",
        "competitions": [{
            "status": {"type": {"state": "post"}},
            "details": details,
            "competitors": [comp("home", "H", len(home_goals)), comp("away", "A", len(away_goals))],
        }],
    }


def test_comeback_and_late_drama() -> None:
    # Away leads 0-2 early; home scores 60', 70', 85' to win 3-2: the lead flips.
    row = SoccerIngest._match_row(_event([60, 70, 85], [10, 20]), "eng.1", 2024, date(2024, 4, 7))
    assert row is not None
    assert set(row) == set(MATCH_COLUMNS)
    assert row["total_goals"] == 5 and row["final_margin"] == 1
    assert row["came_from_behind"] == 1    # home won after trailing
    assert row["late_drama"] == 1          # goal at 85'
    assert row["lead_changes"] == 1        # away led, then home took the lead


def test_red_cards_and_knockout_flag() -> None:
    row = SoccerIngest._match_row(_event([30], [], reds=2), "uefa.champions", 2024, date(2024, 3, 5))
    assert row["red_cards"] == 2
    assert row["is_knockout"] == 1         # CL in March

    grp = SoccerIngest._match_row(_event([30], []), "uefa.champions", 2024, date(2023, 10, 5))
    assert grp["is_knockout"] == 0         # CL in October (group stage)


def test_normalize_soccer_onto_unified() -> None:
    raw = pd.DataFrame([
        dict(item_id="s1", league="eng.1", season=2024, date="2024-04-07T14:00Z",
             away="ARS", home="LIV", away_score=2, home_score=2, away_logo="", home_logo="",
             final_margin=0, total_goals=4, red_cards=1, lead_changes=1,
             came_from_behind=1, late_drama=1, is_knockout=0),
    ])
    u = normalize_soccer(raw)
    assert set(UNIFIED_FEATURES).issubset(u.columns)
    vals = u[UNIFIED_FEATURES].to_numpy()
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    assert u.loc[0, "competitiveness"] == 1.0     # 0-goal margin (draw) -> perfectly close
    assert list(u.loc[0, "entities"]) == ["ARS", "LIV"]

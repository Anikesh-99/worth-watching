"""Feature-correctness tests for the NBA ingest.

One network-backed check against a game with a known result, plus a synthetic
game that pins the flow features (lead changes, come-from-behind, overtime)
deterministically without hitting the network.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.sports.nba_ingest import GAME_COLUMNS, NBAIngest


def _fake_event(home_q: list[int], away_q: list[int], playoff: bool = False) -> dict:
    """Build a minimal ESPN-shaped completed-game event from quarter scores."""
    def competitor(side: str, q: list[int]) -> dict:
        return {
            "homeAway": side,
            "score": str(sum(q)),
            "team": {"abbreviation": side.upper()[:3]},
            "linescores": [{"value": v} for v in q],
        }
    return {
        "id": "test1",
        "date": "2024-01-01T00:00Z",
        "season": {"type": 3 if playoff else 2},
        "competitions": [{
            "status": {"type": {"state": "post"}},
            "competitors": [competitor("home", home_q), competitor("away", away_q)],
        }],
    }


def test_synthetic_comeback_and_overtime() -> None:
    # Home trails every regulation quarter, forces OT, wins in the extra period.
    # home cum: 20,45,68,95(tie),108 ; away cum: 30,55,80,95(tie),101
    home_q = [20, 25, 23, 27, 13]
    away_q = [30, 25, 25, 15, 6]
    row = NBAIngest._game_row(_fake_event(home_q, away_q, playoff=True), season=2024)
    assert row is not None
    assert row["home_score"] == 108 and row["away_score"] == 101
    assert row["final_margin"] == 7
    assert row["overtime_periods"] == 1          # 5 periods -> 1 OT
    assert row["winner_came_from_behind"] == 1   # home trailed after Q3
    assert row["is_playoff"] == 1


def test_synthetic_wire_to_wire_has_no_comeback() -> None:
    home_q = [30, 30, 30, 30]   # home leads throughout
    away_q = [20, 20, 20, 20]
    row = NBAIngest._game_row(_fake_event(home_q, away_q), season=2024)
    assert row["winner_came_from_behind"] == 0
    assert row["overtime_periods"] == 0
    assert row["lead_changes"] == 0
    assert row["is_playoff"] == 0


def test_in_progress_game_is_skipped() -> None:
    ev = _fake_event([30, 30, 30, 30], [20, 20, 20, 20])
    ev["competitions"][0]["status"]["type"]["state"] = "in"
    assert NBAIngest._game_row(ev, season=2024) is None


@pytest.mark.parametrize("season,expect_days", [(2024, True)])
def test_season_dates_span(season: int, expect_days: bool) -> None:
    days = NBAIngest._season_dates(season)
    assert days[0] == date(season - 1, 10, 1)
    assert days[-1] == date(season, 6, 30)


def test_finals_g2_2024_network() -> None:
    # Network-backed (cached after first run): BOS 105-98 DAL, a playoff game.
    ing = NBAIngest()
    sb = ing._scoreboard(date(2024, 6, 9))
    rows = [ing._game_row(ev, 2024) for ev in sb.get("events", [])]
    rows = [r for r in rows if r]
    assert len(rows) == 1
    r = rows[0]
    assert set(r) == set(GAME_COLUMNS)
    assert {r["home"], r["away"]} == {"BOS", "DAL"}
    assert r["final_margin"] == 7
    assert r["is_playoff"] == 1

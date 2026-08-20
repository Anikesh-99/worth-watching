"""Tests for the Phase 3 recommender + personalization (network-free)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.core.features import unify
from src.core.interfaces import Item, UserProfile
from src.sports.personalize import (
    DEFAULT_WEIGHTS, SportTaste, TasteWeights, calibrate, calibrate_per_sport, personalize,
)
from src.sports.recommender import SportRecommender

# outcome words a spoiler-free reason must never contain
BANNED = ["won", "win", "beat", "comeback", "margin", "score", "dnf", "overtime", "lead change"]


def _unified() -> pd.DataFrame:
    f1 = pd.DataFrame([
        dict(item_id="f1-2023-13", season=2023, round=13, event_name="Dutch GP", country="NL",
             date="2023-08-27", n_classified=17, n_dnf=3, winner_margin_s=3.7,
             total_positions_moved=106, max_gain=9, podium_from_outside_top5=1, winner_grid=1),
    ])
    nba = pd.DataFrame([
        dict(item_id="nba-1", season=2024, date="2024-01-01T00:00Z", away="DAL", home="BOS",
             away_score=101, home_score=108, final_margin=7, overtime_periods=1,
             lead_changes=3, winner_came_from_behind=1, max_abs_lead=9, is_playoff=1),
        dict(item_id="nba-2", season=2024, date="2024-01-02T00:00Z", away="SAS", home="GSW",
             away_score=90, home_score=125, final_margin=35, overtime_periods=0,
             lead_changes=0, winner_came_from_behind=0, max_abs_lead=38, is_playoff=0),
    ])
    return unify(f1, nba)


def test_personalize_boosts_followed_team_and_stakes() -> None:
    item = Item(item_id="nba-1", vertical="nba", when=datetime(2024, 1, 1),
                features={"stakes": 1.0}, meta={"entities": ["DAL", "BOS"], "label": "DAL @ BOS"})
    mult, reasons = personalize(item, UserProfile(followed_entities={"BOS"}))
    assert mult > 1.5  # followed boost + playoff stakes
    assert any("BOS" in r for r in reasons)
    assert any("playoff" in r for r in reasons)


def test_personalize_neutral_without_follow_or_stakes() -> None:
    item = Item(item_id="nba-2", vertical="nba", when=datetime(2024, 1, 2),
                features={"stakes": 0.0}, meta={"entities": ["SAS", "GSW"], "label": "SAS @ GSW"})
    mult, reasons = personalize(item, UserProfile(followed_entities={"BOS"}))
    assert mult == 1.0 and reasons == []


def test_generate_candidates_filters_window() -> None:
    rec = SportRecommender(_unified())
    all_items = rec.generate_candidates(datetime(2023, 1, 1), datetime(2024, 12, 31))
    assert len(all_items) == 3
    jan = rec.generate_candidates(datetime(2024, 1, 1), datetime(2024, 1, 1))
    assert [i.item_id for i in jan] == ["nba-1"]


def test_scoring_is_spoiler_free_and_bounded() -> None:
    rec = SportRecommender(_unified())
    user = UserProfile(followed_entities={"BOS"})
    scored = rec.score(rec.generate_candidates(datetime(2023, 1, 1), datetime(2024, 12, 31)), user)
    for s in scored:
        assert 0.0 <= s.excitement <= 1.0
        assert s.personalization >= 1.0
        blob = " ".join(s.reasons).lower()
        assert not any(b in blob for b in BANNED), f"spoiler leaked: {s.reasons}"


def test_calibrate_clamps_refuted_signal_to_zero() -> None:
    # Followed team (BOS) is rated LOWER than others, stakes anti-correlate.
    df = pd.DataFrame({
        "entities": [["BOS", "X"], ["BOS", "Y"], ["A", "B"], ["C", "D"]],
        "stakes": [1.0, 1.0, 0.0, 0.0],
        "rating": [2, 2, 5, 4],  # followed/high-stakes = worse
    })
    w = calibrate(df, UserProfile(followed_entities={"BOS"}))
    assert w.followed_boost == 0.0
    assert w.stakes_boost == 0.0


def test_calibrate_rewards_supported_signal() -> None:
    # Here the followed team IS rated higher -> positive (clamped) boost.
    df = pd.DataFrame({
        "entities": [["BOS", "X"], ["BOS", "Y"], ["A", "B"], ["C", "D"]],
        "stakes": [0.0, 0.0, 0.0, 0.0],
        "rating": [5, 5, 2, 2],
    })
    w = calibrate(df, UserProfile(followed_entities={"BOS"}))
    assert w.followed_boost > 0.0


def test_sport_taste_falls_back_to_default() -> None:
    taste = SportTaste(per_sport={"nba": TasteWeights(followed_boost=0.1, stakes_boost=0.0)})
    assert taste.for_sport("nba").followed_boost == 0.1
    assert taste.for_sport("soccer") is DEFAULT_WEIGHTS   # unseen sport -> prior


def test_calibrate_per_sport_independent_and_sparse_fallback() -> None:
    # NBA has enough ratings and REFUTES the followed boost; soccer has only 2
    # ratings so it must keep the neutral prior rather than fit noise.
    rows = []
    for i in range(10):                                   # 10 NBA: followed = worse
        followed = i % 2 == 0
        rows.append(dict(sport="nba", entities=(["BOS", "X"] if followed else ["A", "B"]),
                         stakes=0.0, rating=2 if followed else 5))
    rows += [dict(sport="soccer", entities=["ARS", "X"], stakes=0.0, rating=5),
             dict(sport="soccer", entities=["A", "B"], stakes=0.0, rating=2)]
    taste = calibrate_per_sport(pd.DataFrame(rows), UserProfile(followed_entities={"BOS", "ARS"}))
    assert "nba" in taste.per_sport and taste.per_sport["nba"].followed_boost == 0.0  # refuted
    assert "soccer" not in taste.per_sport                # < MIN_SPORT_RATINGS -> prior
    assert taste.for_sport("soccer") is DEFAULT_WEIGHTS


def test_recommender_applies_per_sport_weights() -> None:
    # A followed-club boost for soccer must lift a soccer item even when nba is zeroed.
    taste = SportTaste(per_sport={"nba": TasteWeights(followed_boost=0.0, stakes_boost=0.0)},
                       default=TasteWeights(followed_boost=0.4, stakes_boost=0.0))
    df = _unified()
    rec = SportRecommender(df, weights=taste)
    items = rec.generate_candidates(datetime(2023, 1, 1), datetime(2024, 12, 31))
    user = UserProfile(followed_entities={"BOS"})
    by_id = {s.item.item_id: s for s in rec.score(items, user)}
    # nba items involving BOS get x1.0 (zeroed); a non-nba sport would use default 0.4
    nba_bos = [s for s in by_id.values() if s.item.vertical == "nba"
               and "BOS" in s.item.meta["entities"]]
    assert nba_bos and all(abs(s.personalization - 1.0) < 1e-9 for s in nba_bos)


def test_rank_orders_by_score_desc() -> None:
    rec = SportRecommender(_unified())
    ranked = rec.watchlist(datetime(2023, 1, 1), datetime(2024, 12, 31), UserProfile(followed_entities={"BOS"}))
    scores = [r.scored.score for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in ranked] == [1, 2, 3]

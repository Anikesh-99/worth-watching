"""Tests for the upcoming-events engine (watchability index + recommender)."""

from __future__ import annotations

import pandas as pd

from src.core.interfaces import Ranked, UserProfile
from src.sports.fixtures import f1_fixtures
from src.sports.upcoming import UpcomingRecommender
from src.sports.watchability import WATCH_FEATURES, WatchabilityIndex

# a result-revealing word must never appear in a pre-game reason
BANNED = ["won", "beat", "final", "comeback", "margin", "score", "lead change"]


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame([
        dict(item_id="hi", sport="nba", label="Big v Game", date=pd.Timestamp("2026-10-01"),
             entities=["OKC", "DEN"], stakes=1.0, pedigree=0.9, competitiveness=0.9, form=0.8),
        dict(item_id="lo", sport="nba", label="Meh v Dud", date=pd.Timestamp("2026-10-02"),
             entities=["XXX", "YYY"], stakes=0.1, pedigree=0.2, competitiveness=0.2, form=0.3),
    ])


def test_index_bounded_and_weighted() -> None:
    s = WatchabilityIndex().score(_fixtures())
    assert (s.between(0, 1)).all()
    assert s.iloc[0] > s.iloc[1]           # the high-signal fixture scores higher


def test_recommender_ranks_and_is_spoiler_free() -> None:
    rec = UpcomingRecommender(_fixtures())
    ranked = rec.recommend(UserProfile(followed_entities={"OKC"}))
    assert [r.scored.item.item_id for r in ranked] == ["hi", "lo"]
    assert all(isinstance(r, Ranked) for r in ranked)
    for r in ranked:
        blob = " ".join(r.scored.reasons).lower()
        assert not any(b in blob for b in BANNED), r.scored.reasons


def test_followed_entity_boosts_score() -> None:
    rec = UpcomingRecommender(_fixtures())
    followed = {s.item.item_id: s.score for s in
                (r.scored for r in rec.recommend(UserProfile(followed_entities={"OKC"})))}
    plain = {s.item.item_id: s.score for s in
             (r.scored for r in rec.recommend(UserProfile()))}
    assert followed["hi"] > plain["hi"]    # OKC involvement lifts it


def test_f1_fixtures_features() -> None:
    hist = pd.DataFrame([
        dict(item_id="f1-2023-01", country="Italy"),
        dict(item_id="f1-2023-02", country="Italy"),
    ])
    ex = {"f1-2023-01": 0.8, "f1-2023-02": 0.6}    # Italy pedigree = 0.7
    races = [dict(season=2026, round=12, event_name="Italian GP", country="Italy",
                  date="2026-09-06", season_len=24)]
    fx = f1_fixtures(races, hist, ex)
    row = fx.iloc[0]
    assert set(WATCH_FEATURES).issubset(fx.columns)
    assert abs(row["pedigree"] - 0.7) < 1e-9
    assert abs(row["stakes"] - 12 / 24) < 1e-9

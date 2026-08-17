"""Unit tests for the Phase 2 modeling layer (all network-free)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.evaluation import ndcg_grouped, spearman, temporal_split
from src.core.excitement import ExcitementIndex, ExcitementModel
from src.core.features import UNIFIED_FEATURES, normalize_f1, normalize_nba, unify


def _f1_raw() -> pd.DataFrame:
    return pd.DataFrame([
        # a thriller (close, chaotic) and a procession (dominant, clean)
        dict(item_id="f1-2023-13", season=2023, round=13, event_name="Dutch GP",
             country="NL", date="2023-08-27", n_classified=17, n_dnf=3,
             winner_margin_s=3.7, total_positions_moved=106, max_gain=9,
             podium_from_outside_top5=1, winner_grid=1),
        dict(item_id="f1-2023-12", season=2023, round=12, event_name="Belgian GP",
             country="BE", date="2023-07-30", n_classified=19, n_dnf=1,
             winner_margin_s=22.0, total_positions_moved=40, max_gain=4,
             podium_from_outside_top5=0, winner_grid=6),
    ])


def _nba_raw() -> pd.DataFrame:
    return pd.DataFrame([
        dict(item_id="nba-1", season=2024, date="2024-01-01T00:00Z", away="DAL",
             home="BOS", away_score=101, home_score=108, final_margin=7,
             overtime_periods=1, lead_changes=3, winner_came_from_behind=1,
             max_abs_lead=9, is_playoff=1),
        dict(item_id="nba-2", season=2024, date="2024-01-02T00:00Z", away="SAS",
             home="GSW", away_score=90, home_score=125, final_margin=35,
             overtime_periods=0, lead_changes=0, winner_came_from_behind=0,
             max_abs_lead=38, is_playoff=0),
    ])


def test_normalizers_produce_unified_scale() -> None:
    for norm, raw in [(normalize_f1, _f1_raw()), (normalize_nba, _nba_raw())]:
        u = norm(raw)
        assert set(UNIFIED_FEATURES).issubset(u.columns)
        vals = u[UNIFIED_FEATURES].to_numpy()
        assert vals.min() >= 0.0 and vals.max() <= 1.0


def test_thriller_scores_above_procession() -> None:
    idx = ExcitementIndex()
    f1 = normalize_f1(_f1_raw())
    scores = idx.score(f1)
    # Dutch GP (row 0) must out-excite the Belgian GP (row 1).
    assert scores.iloc[0] > scores.iloc[1]
    assert (scores.between(0, 1)).all()


def test_unify_mixes_sports_and_sorts_by_date() -> None:
    u = unify(_f1_raw(), _nba_raw())
    assert set(u["sport"]) == {"f1", "nba"}
    assert u["date"].is_monotonic_increasing  # tz-naive, sortable across sports


def test_model_fits_and_predicts() -> None:
    # Enough rows/queries for LambdaRank to train.
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({f: rng.random(n) for f in UNIFIED_FEATURES})
    df["sport"] = np.where(rng.random(n) < 0.5, "f1", "nba")
    df["season"] = rng.integers(2021, 2024, n)
    idx = ExcitementIndex()
    y = idx.grades(df)
    model = ExcitementModel().fit(df, y)
    preds = model.predict(df)
    assert preds.shape == (n,)
    assert model.feature_importance().sum() > 0


def test_temporal_split_isolates_test_season() -> None:
    df = pd.DataFrame({"season": [2021, 2022, 2023, 2024], "x": [1, 2, 3, 4]})
    train, test = temporal_split(df, {2024})
    assert set(train["season"]) == {2021, 2022, 2023}
    assert set(test["season"]) == {2024}


def test_spearman_perfect_on_monotone() -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    assert abs(spearman(df, "a", "b") - 1.0) < 1e-9


def test_ndcg_rewards_correct_order() -> None:
    df = pd.DataFrame({
        "sport": ["nba"] * 5, "season": [2024] * 5, "month": ["2024-01"] * 5,
        "grade": [4, 3, 2, 1, 0], "good": [5, 4, 3, 2, 1], "bad": [1, 2, 3, 4, 5],
    })
    g = ["sport", "season", "month"]
    assert ndcg_grouped(df, "good", "grade", g, k=5) > ndcg_grouped(df, "bad", "grade", g, k=5)

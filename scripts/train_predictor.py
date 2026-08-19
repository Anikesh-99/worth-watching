"""Train + evaluate the predictive excitement model (pre-game -> excitement).

Pipeline:
  1. compute realized excitement per historical event (the target);
  2. build lookahead-free pre-game features (src/sports/pregame.py);
  3. temporal split (train earlier seasons, test the latest);
  4. train per sport; evaluate how well predicted excitement ranks the ACTUAL
     excitement of held-out events, vs naive baselines;
  5. report feature importance; save models.

This is the honest, non-circular model: inputs (pre-game) and target (post-hoc)
are disjoint, so a real eval number means something.

Usage:
    python scripts/train_predictor.py            # test season defaults to 2024
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.evaluation import ndcg_grouped, spearman  # noqa: E402
from src.core.excitement import ExcitementIndex  # noqa: E402
from src.core.features import normalize_f1, normalize_nba  # noqa: E402
from src.sports.pregame import (F1_FEATURES, NBA_FEATURES,  # noqa: E402
                                build_f1_pregame, build_nba_pregame)
from src.sports.predict import PredictiveExcitementModel  # noqa: E402


def _excitement_map(raw: pd.DataFrame, normalizer) -> dict[str, float]:
    u = normalizer(raw)
    return dict(zip(u["item_id"], ExcitementIndex().score(u)))


def _evaluate(name: str, feats: pd.DataFrame, features: list[str], test_season: int,
              group_cols: list[str], baseline: str) -> None:
    train = feats[feats["season"] != test_season].reset_index(drop=True)
    test = feats[feats["season"] == test_season].reset_index(drop=True)
    print(f"\n=== {name} === train {len(train)} | test {len(test)} (season {test_season})")
    if len(train) < 30 or len(test) < 5:
        print("  too little data to train/evaluate honestly.")
        return

    model = PredictiveExcitementModel(features).fit(train)
    test = test.copy()
    test["pred"] = model.predict(test)
    test["chrono"] = range(len(test))                       # naive: no signal

    rankers = {
        "chronological (naive)": "chrono",
        f"single feature ({baseline})": baseline,
        "predictive model": "pred",
    }
    print(f"  {'ranker':34} {'Spearman':>9} {'NDCG@10':>9}   (vs ACTUAL excitement)")
    for label, col in rankers.items():
        sp = spearman(test, col, "excitement")
        nd = ndcg_grouped(test.assign(_gain=(test["excitement"] * 100).round()),
                          col, "_gain", group_cols, k=10)
        print(f"  {label:34} {sp:9.3f} {nd:9.3f}")
    print("  feature importance:", model.feature_importance().to_dict())
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, f"models/predictor_{name}.joblib")


def main(test_season: int = 2024) -> None:
    f1 = pd.read_csv("data/f1_events.csv", parse_dates=["date"])
    nba = pd.read_csv("data/nba_events.csv", parse_dates=["date"])

    nba_feats = build_nba_pregame(nba, _excitement_map(nba, normalize_nba))
    nba_feats["month"] = pd.to_datetime(nba_feats["date"]).dt.strftime("%Y-%m")
    _evaluate("nba", nba_feats, NBA_FEATURES, test_season,
              ["season", "month"], baseline="combined_winpct")

    f1_feats = build_f1_pregame(f1, _excitement_map(f1, normalize_f1))
    _evaluate("f1", f1_feats, F1_FEATURES, test_season,
              ["season"], baseline="circuit_hist")

    print("\nNote: NBA carries rich pre-game signal (standings, form, h2h). F1's "
          "race-aggregate table gives only stakes/circuit/season-form — driver "
          "championship tightness needs per-driver standings (FastF1 enrichment).")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)

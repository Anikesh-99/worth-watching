"""Train and evaluate the shared cross-sport excitement model.

Pipeline:
  1. load both sports' tidy CSVs and unify them onto the shared feature schema;
  2. label rows with index-derived relevance grades (bootstrap target);
  3. temporal split (train earlier seasons, test the latest);
  4. train one LightGBM LambdaRank model across both sports;
  5. evaluate on the held-out season vs naive baselines, per sport;
  6. report cross-sport feature importance; save the model.

Usage:
    python scripts/train_excitement.py            # test season defaults to 2024
    python scripts/train_excitement.py 2024
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.evaluation import mrr_grouped, ndcg_grouped, spearman, temporal_split  # noqa: E402
from src.core.excitement import ExcitementIndex, ExcitementModel  # noqa: E402
from src.core.features import unify  # noqa: E402


def _load(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def main(test_season: int = 2024) -> None:
    f1 = _load("data/f1_events.csv")
    nba = _load("data/nba_events.csv")
    df = unify(f1, nba)
    if df.empty:
        sys.exit("No data. Run scripts/build_dataset.py for f1 and nba first.")

    # Bootstrap relevance grades from the transparent index (see excitement.py).
    index = ExcitementIndex()
    df["index_score"] = index.score(df)
    df["grade"] = index.grades(df)
    df["month"] = df["date"].dt.to_period("M").astype(str)

    train, test = temporal_split(df, {test_season})
    print(f"unified events: {len(df)} | train: {len(train)} | test ({test_season}): {len(test)}")
    print("by sport (test):")
    print(test.groupby("sport").size().to_string(), "\n")

    # Train one model across both sports.
    model = ExcitementModel().fit(train, train["grade"].to_numpy(), group_key="season")

    # Score test rows with each ranker.
    test = test.copy()
    test["model_score"] = model.predict(test)
    test["competitiveness_only"] = test["competitiveness"]
    test["chronological"] = range(len(test))  # naive: no excitement info

    groups = ["sport", "season", "month"]
    rankers = {
        "chronological (naive)": "chronological",
        "single feature (competitiveness)": "competitiveness_only",
        "shared LightGBM model": "model_score",
    }

    def n_queries(sub: pd.DataFrame) -> int:
        return int((sub.groupby(groups).size() >= 2).sum())

    print("=== NDCG@10 on held-out season (weekly/monthly ranking queries) ===")
    print(f"ranking queries -> f1: {n_queries(test[test.sport=='f1'])} "
          f"(tiny: 1-3 races/month, NDCG unreliable) | nba: {n_queries(test[test.sport=='nba'])}\n")
    print(f"{'ranker':38} {'nba':>8} {'f1*':>8}")
    for name, col in rankers.items():
        nba_ndcg = ndcg_grouped(test[test.sport == "nba"], col, "grade", groups, k=10)
        f1_ndcg = ndcg_grouped(test[test.sport == "f1"], col, "grade", groups, k=10)
        print(f"{name:38} {nba_ndcg:8.3f} {f1_ndcg:8.3f}")
    print("  * F1 queries are 1-3 items, so its NDCG saturates near 1.0 regardless of ranker.")

    print("\n=== honesty check on the learned model ===")
    print(f"Spearman(model score, index):    {spearman(test, 'model_score', 'index_score'):.3f}"
          "   <- ~1.0 EXPECTED: it was bootstrapped on the index; not a quality claim.")
    print(f"NBA MRR (top-grade item found):  {mrr_grouped(test[test.sport=='nba'], 'model_score', 'grade', groups):.3f}")
    print("Meaningful result: on NBA (large queries), multi-signal ranking beats the")
    print("chronological baseline; real quality eval awaits the user's ratings (Phase 4).")

    print("\n=== cross-sport feature importance (what actually drives the excitement definition) ===")
    print(model.feature_importance().to_string())

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/excitement_model.joblib")
    print("\nsaved -> models/excitement_model.joblib")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)

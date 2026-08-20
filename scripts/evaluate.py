"""Evaluate every ranker against the user's REAL 1-5 ratings, and emit a table.

The honest quality test the whole design was built toward. For each rated event
we compute several rankers and measure how well each reproduces *your* ratings:

  * chronological (naive baseline)
  * popularity / "marquee" (rank by how often the teams appear — a real baseline)
  * single feature (competitiveness)
  * excitement index (objective layer)
  * personalized, assumed boosts (the intuitive +35%-for-your-team prior)
  * personalized, per-sport calibrated (learns the boosts; clamps refuted to 0)
  * small models trained on your ratings (5-fold out-of-fold, no leakage)

Metrics: Spearman (overall rank agreement), NDCG@10 and MAP@10 per sport
(gain = your rating; MAP relevance = rating >= 4). Plus a beyond-accuracy pass:
an MMR re-rank trading a little NDCG for intra-list diversity, because a
watch-list that's five games of one team is accurate and useless.

Outputs: prints a table, and writes docs/evaluation.md + docs/evaluation.json so
the numbers can be embedded in the README (never hand-transcribed).

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.evaluation import (  # noqa: E402
    intra_list_diversity, map_grouped, ndcg_grouped, spearman,
)
from src.core.features import UNIFIED_FEATURES, unify  # noqa: E402
from src.core.profile import load_user_profile  # noqa: E402
from src.core.rerank import event_similarity, mmr_rerank  # noqa: E402
from src.sports.personalize import calibrate_per_sport  # noqa: E402
from src.sports.recommender import SportRecommender  # noqa: E402

OUT = Path("docs/evaluation.md")
OUT_JSON = Path("docs/evaluation.json")


def _load(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def _entity_popularity(all_events: pd.DataFrame) -> dict[str, float]:
    """How often each team appears across ALL events -> a marquee/popularity prior."""
    freq: dict[str, int] = {}
    for ents in all_events["entities"]:
        for e in ents or []:
            freq[e] = freq.get(e, 0) + 1
    return freq


def main() -> None:
    if not Path("data/my_ratings.csv").exists():
        sys.exit("No data/my_ratings.csv. Fill the template first.")
    ratings = pd.read_csv("data/my_ratings.csv")[["item_id", "rating"]]

    all_events = unify(_load("data/f1_events.csv"), _load("data/nba_events.csv"),
                       _load("data/soccer_events.csv"))
    df = all_events.merge(ratings, on="item_id", how="inner").reset_index(drop=True)
    counts = df.groupby("sport").size().to_dict()
    print(f"rated events: {len(df)}  {counts}")

    user = load_user_profile(["configs/f1.yaml", "configs/nba.yaml", "configs/soccer.yaml"])

    # --- rankers straight from the pipeline (assumed prior vs per-sport calibrated)
    rec = SportRecommender(df)                                  # default prior
    items = rec.generate_candidates(datetime(2000, 1, 1), datetime(2100, 1, 1))
    scored = {s.item.item_id: s for s in rec.score(items, user)}
    df["excitement"] = df["item_id"].map(lambda i: scored[i].excitement)
    df["personalized_assumed"] = df["item_id"].map(lambda i: scored[i].score)

    taste = calibrate_per_sport(df, user)                       # per-sport, evidence-based
    rec_cal = SportRecommender(df, weights=taste)
    scored_cal = {s.item.item_id: s for s in rec_cal.score(items, user)}
    df["personalized_calibrated"] = df["item_id"].map(lambda i: scored_cal[i].score)

    # --- baselines
    df["competitiveness_only"] = df["competitiveness"]
    df["chronological"] = df["date"].rank()
    pop = _entity_popularity(all_events)
    gmean = (sum(pop.values()) / len(pop)) if pop else 0.0
    df["popularity"] = df["entities"].map(
        lambda es: (sum(pop.get(e, 0) for e in es) / len(es)) if es else gmean)
    df["followed"] = df["entities"].map(lambda e: int(bool(set(e) & user.followed_entities)))

    # --- small models trained on YOUR ratings (out-of-fold to avoid leakage)
    X = df[UNIFIED_FEATURES + ["followed"]].to_numpy()
    y = df["rating"].to_numpy()
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    df["ridge_oof"] = cross_val_predict(Ridge(alpha=1.0), X, y, cv=kf)
    df["gbm_oof"] = cross_val_predict(
        GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05, random_state=0),
        X, y, cv=kf)

    rankers = {
        "chronological (naive)": "chronological",
        "popularity / marquee": "popularity",
        "competitiveness only": "competitiveness_only",
        "excitement index": "excitement",
        "personalized (assumed boosts)": "personalized_assumed",
        "personalized (per-sport calibrated)": "personalized_calibrated",
        "Ridge on your ratings (OOF)": "ridge_oof",
        "GBM on your ratings (OOF)": "gbm_oof",
    }
    sports = [s for s in ("nba", "f1", "soccer") if counts.get(s, 0) >= 5]

    # --- accuracy table -----------------------------------------------------
    rows = []
    for name, col in rankers.items():
        row = {"ranker": name, "spearman": round(spearman(df, col, "rating"), 3)}
        for sp in sports:
            g = df[df.sport == sp]
            row[f"ndcg@10 {sp}"] = round(ndcg_grouped(g, col, "rating", ["sport"], k=10), 3)
            row[f"map@10 {sp}"] = round(map_grouped(g, col, "rating", ["sport"], k=10, rel_threshold=4), 3)
        rows.append(row)

    # --- beyond-accuracy: MMR diversity/relevance tradeoff ------------------
    smin, smax = df["personalized_calibrated"].min(), df["personalized_calibrated"].max()
    df["rel01"] = ((df["personalized_calibrated"] - smin) / (smax - smin)) if smax > smin else 0.5
    pool = [dict(item_id=r.item_id, sport=r.sport, entities=list(r.entities),
                 rating=r.rating, rel=r.rel01) for r in df.itertuples()]
    rel = {p["item_id"]: p["rel"] for p in pool}
    tradeoff = []
    for lam in (1.0, 0.8, 0.6, 0.4):
        order = mmr_rerank(pool, relevance=lambda p: rel[p["item_id"]],
                           sim=event_similarity, lam=lam, k=10)
        ndcg = ndcg_grouped(
            pd.DataFrame([{"g": 0, "score": len(order) - i, "rating": o["rating"]}
                          for i, o in enumerate(order)]),
            "score", "rating", ["g"], k=10)
        ild = intra_list_diversity(order, event_similarity)
        tradeoff.append({"lambda": lam, "ndcg@10": round(ndcg, 3), "diversity@10": round(ild, 3)})

    _report(df, rows, sports, taste, tradeoff, counts)


def _md_table(headers: list[str], rows: list[list]) -> str:
    def line(cells):
        return "| " + " | ".join(str(c) for c in cells) + " |"
    out = [line(headers), line(["---"] * len(headers))]
    out += [line(r) for r in rows]
    return "\n".join(out)


def _report(df, rows, sports, taste, tradeoff, counts) -> None:
    cols = ["ranker", "spearman"] + [f"{m}@10 {s}" for s in sports for m in ("ndcg", "map")]
    acc_rows = [[r.get(c, "—") for c in cols] for r in rows]

    # console
    print("\n=== agreement with YOUR ratings ===")
    print(_md_table(cols, acc_rows))
    print("\n=== beyond-accuracy: MMR relevance/diversity tradeoff (top-10) ===")
    print(_md_table(["lambda", "ndcg@10", "diversity@10"],
                    [[t["lambda"], t["ndcg@10"], t["diversity@10"]] for t in tradeoff]))
    ps = ", ".join(f"{s}: followed={w.followed_boost}/stakes={w.stakes_boost}"
                   for s, w in sorted(taste.per_sport.items()))
    print(f"\nper-sport calibrated weights -> {ps}")

    # markdown artifact
    n = len(df)
    md = [
        "# Evaluation results",
        "",
        f"_Auto-generated by `scripts/evaluate.py` — do not hand-edit._ "
        f"Rated events: **{n}** ({counts}).",
        "",
        "## Agreement with the author's own 1-5 ratings",
        "",
        "Temporal by construction (each rated event is scored by rankers that never "
        "saw its rating). Gain = your rating; MAP relevance = rating ≥ 4.",
        "",
        _md_table(cols, acc_rows),
        "",
        "## Beyond-accuracy: MMR relevance/diversity tradeoff (top-10)",
        "",
        "λ = 1.0 is the pure relevance ranking; lowering λ spreads the list across "
        "teams/sports. Diversity = 1 − mean pairwise (same-sport + shared-team) similarity.",
        "",
        _md_table(["λ", "NDCG@10", "diversity@10"],
                  [[t["lambda"], t["ndcg@10"], t["diversity@10"]] for t in tradeoff]),
        "",
        f"Per-sport calibrated taste weights: {ps or '(none — sports below the ratings threshold)'}.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md))
    OUT_JSON.write_text(json.dumps(
        {"n": n, "counts": counts, "accuracy": rows, "tradeoff": tradeoff,
         "weights_by_sport": {s: {"followed_boost": w.followed_boost, "stakes_boost": w.stakes_boost}
                              for s, w in taste.per_sport.items()}}, indent=2))
    print(f"\nwrote {OUT} and {OUT_JSON}")
    print("Note: n is small; treat single-decimal gaps as directional, not definitive.")


if __name__ == "__main__":
    main()

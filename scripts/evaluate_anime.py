"""Phase 6 evaluation — does content-based taste matching predict YOUR anime ratings?

Leave-one-out: for each rated anime, build the taste profile from all your OTHER
ratings and score the held-out title; measure how well those scores reproduce
your ratings (Spearman, NDCG@10). Reuses the same evaluation harness as sports,
proving the platform's shared machinery.

Usage:
    python scripts/evaluate_anime.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.evaluation import ndcg_grouped, spearman  # noqa: E402
from src.core.interfaces import UserProfile  # noqa: E402
from src.media.recommender import AnimeRecommender  # noqa: E402


def main() -> None:
    if not Path("data/my_anime_ratings.csv").exists():
        sys.exit("No data/my_anime_ratings.csv. Fill the template or import from MAL first.")
    cat = pd.read_csv("data/anime_catalog.csv")
    ratings = pd.read_csv("data/my_anime_ratings.csv")
    ratings["item_id"] = ratings["item_id"].astype(str)

    rec = AnimeRecommender(cat)
    in_cat = ratings[ratings["item_id"].isin(cat["item_id"])].reset_index(drop=True)
    print(f"rated anime in catalog: {len(in_cat)} / {len(ratings)} "
          f"(only in-catalog titles can be scored)")
    if len(in_cat) < 5:
        sys.exit("Need at least ~5 rated titles that are in the catalog to evaluate.")

    rows = []
    for i, row in in_cat.iterrows():
        others = in_cat.drop(index=i)
        user = UserProfile(ratings=dict(zip(others["item_id"], others["rating"])))
        scored = {s.item.item_id: s for s in
                  rec.score(rec.generate_candidates(_MIN, _MAX), user)}
        s = scored.get(row["item_id"])
        if s:
            rows.append({"item_id": row["item_id"], "rating": row["rating"],
                         "taste": s.personalization, "score": s.score,
                         "quality": s.excitement, "sport": "anime"})
    ev = pd.DataFrame(rows)

    print("\n=== agreement with YOUR anime ratings (leave-one-out) ===")
    print(f"{'ranker':34} {'Spearman':>9} {'NDCG@10':>9}")
    for name, col in [("popularity/quality prior", "quality"),
                      ("taste match (content-based)", "taste"),
                      ("quality × taste", "score")]:
        sp = spearman(ev, col, "rating")
        nd = ndcg_grouped(ev, col, "rating", ["sport"], k=10)
        print(f"{name:34} {sp:9.3f} {nd:9.3f}")
    print("\nContent-based taste matching is the media analogue of the sports")
    print("excitement model — same interface, same eval harness, different engine.")


from datetime import datetime  # noqa: E402
_MIN, _MAX = datetime(1960, 1, 1), datetime(2100, 1, 1)


if __name__ == "__main__":
    main()

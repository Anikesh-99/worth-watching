"""Evaluate content-based book recommendations against YOUR Goodreads ratings.

Leave-one-out: for each rated book, build the taste profile from all your OTHER
ratings and score the held-out book; measure how well those scores reproduce
your ratings (Spearman, NDCG@10). Same harness as sports and anime.

Usage:
    python scripts/evaluate_books.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.evaluation import ndcg_grouped, spearman  # noqa: E402
from src.core.interfaces import UserProfile  # noqa: E402
from src.media.book_recommender import BookRecommender  # noqa: E402

_MIN, _MAX = datetime(1400, 1, 1), datetime(2100, 1, 1)


def main() -> None:
    if not Path("data/my_books_ratings.csv").exists():
        sys.exit("No data/my_books_ratings.csv. Import from Goodreads or fill the template first.")
    ratings = pd.read_csv("data/my_books_ratings.csv")
    ratings["item_id"] = ratings["item_id"].astype(str)

    # A rated book contributes a taste signal only if it carries subject tags;
    # evaluate on the subset that has them (so the leave-one-out is meaningful).
    ratings = ratings[ratings.get("subjects").fillna("").str.len() > 0].reset_index(drop=True)
    if len(ratings) < 5:
        sys.exit("Need at least ~5 rated books with subject tags to evaluate.")

    # Build a catalog that INCLUDES the rated books (so they can be scored),
    # merging their subjects into the recommender's tag space.
    cat = pd.read_csv("data/books_catalog.csv")
    rated_as_catalog = ratings.rename(columns={}).assign(
        author="", year=None, editions=1)[["item_id", "title", "author", "year", "editions", "subjects"]]
    catalog = pd.concat([cat, rated_as_catalog]).drop_duplicates("item_id").reset_index(drop=True)
    rec = BookRecommender(catalog)

    rows = []
    for i, row in ratings.iterrows():
        others = ratings.drop(index=i)
        user = UserProfile(ratings=dict(zip(others["item_id"], others["rating"])))
        scored = {s.item.item_id: s for s in rec.score(rec.generate_candidates(_MIN, _MAX), user)}
        s = scored.get(row["item_id"])
        if s:
            rows.append({"item_id": row["item_id"], "rating": row["rating"],
                         "taste": s.personalization, "score": s.score,
                         "quality": s.excitement, "sport": "book"})
    ev = pd.DataFrame(rows)

    print(f"\nrated books with subjects: {len(ev)}")
    print("\n=== agreement with YOUR book ratings (leave-one-out) ===")
    print(f"{'ranker':34} {'Spearman':>9} {'NDCG@10':>9}")
    for name, col in [("renown / popularity prior", "quality"),
                      ("taste match (content-based)", "taste"),
                      ("renown × taste", "score")]:
        print(f"{name:34} {spearman(ev, col, 'rating'):9.3f} "
              f"{ndcg_grouped(ev, col, 'rating', ['sport'], k=10):9.3f}")


if __name__ == "__main__":
    main()

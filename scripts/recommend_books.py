"""Recommend unread books — the books vertical's reading list.

Uses your ratings (data/my_books_ratings.csv, from scripts/import_goodreads.py)
if present, else cold-starts from the subjects in configs/books.yaml.

Usage:
    python scripts/recommend_books.py [top=15]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.profile import load_user_profile  # noqa: E402
from src.media.book_recommender import BookRecommender  # noqa: E402


def main(top: int = 15) -> None:
    if not Path("data/books_catalog.csv").exists():
        sys.exit("Build the catalog first: scripts/build_dataset.py configs/books.yaml")
    cat = pd.read_csv("data/books_catalog.csv")
    user = load_user_profile(["configs/books.yaml"], "data/my_books_ratings.csv")

    rec = BookRecommender(cat)
    ranked = rec.recommend(user, top=top)

    mode = f"{len(user.ratings)} ratings" if user.ratings else f"cold-start: {sorted(user.followed_entities)}"
    print(f"\n  BOOKS FOR YOU  (from {mode})")
    print("  " + "-" * 68)
    for r in ranked:
        sc = r.scored
        why = " · ".join(sc.reasons[1:]) or "popular pick"
        author = sc.item.meta.get("author", "")
        print(f"  #{r.rank:<2} [{sc.score:4.2f}]  {sc.item.meta['label'][:42]:42} {('— ' + author)[:22]}")
        print(f"        {sc.reasons[0].split(' · ')[0]:11}  {why}  "
              f"(renown {sc.excitement:.2f} × taste {sc.personalization:.2f})")
    print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)

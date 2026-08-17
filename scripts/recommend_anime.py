"""Recommend unwatched anime — the media vertical's watch-list.

Uses your ratings (data/my_anime_ratings.csv) if present, else cold-starts from
the genres in configs/anime.yaml. Same two-stage Recommender as sports.

Usage:
    python scripts/recommend_anime.py [top=15]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.profile import load_user_profile  # noqa: E402
from src.media.content import augment_catalog  # noqa: E402
from src.media.recommender import AnimeRecommender  # noqa: E402


def main(top: int = 15) -> None:
    if not Path("data/anime_catalog.csv").exists():
        sys.exit("Build the catalog first: scripts/build_dataset.py configs/anime.yaml")
    cat = pd.read_csv("data/anime_catalog.csv")
    ratings_path = "data/my_anime_ratings.csv"
    user = load_user_profile(["configs/anime.yaml"], ratings_path)

    # Ground taste in your rated titles' own tags (from AniList), even if they
    # aren't in our catalog.
    if Path(ratings_path).exists():
        rated = pd.read_csv(ratings_path)
        if {"genres", "themes"} & set(rated.columns):
            cat = augment_catalog(cat, rated)

    rec = AnimeRecommender(cat)
    ranked = rec.recommend(user, top=top)

    mode = f"{len(user.ratings)} ratings" if user.ratings else f"cold-start: {sorted(user.followed_entities)}"
    print(f"\n  ANIME FOR YOU  (from {mode})")
    print("  " + "-" * 66)
    for r in ranked:
        sc = r.scored
        why = " · ".join(sc.reasons[1:]) or "popular pick"
        print(f"  #{r.rank:<2} [{sc.score:4.2f}]  {sc.item.meta['label'][:40]:40}")
        print(f"        {sc.reasons[0].split(' · ')[0]:11}  {why}  "
              f"(quality {sc.excitement:.2f} × taste {sc.personalization:.2f})")
    print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)

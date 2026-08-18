"""Recommend new-release albums matching your Spotify taste.

Taste = the genres of your top artists (data/my_music_ratings.csv); candidates =
new releases (data/music_catalog.csv). Same two-stage Recommender as everything
else. Run scripts/build_music.py first.

Usage:
    python scripts/recommend_music.py [top=15]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.profile import load_user_profile  # noqa: E402
from src.media.content import augment_catalog  # noqa: E402
from src.media.music_recommender import MusicRecommender  # noqa: E402


def main(top: int = 15) -> None:
    if not Path("data/music_catalog.csv").exists():
        sys.exit("Fetch data first: scripts/spotify_auth.py then scripts/build_music.py")
    cat = pd.read_csv("data/music_catalog.csv")
    ratings_path = "data/my_music_ratings.csv"
    user = load_user_profile(["configs/music.yaml"], ratings_path)

    # Your top artists aren't albums, so merge them (with their genres) into the
    # catalog to ground the taste vector.
    if Path(ratings_path).exists():
        rated = pd.read_csv(ratings_path)
        if "genres" in rated.columns:
            cat = augment_catalog(cat, rated)

    rec = MusicRecommender(cat)
    ranked = rec.recommend(user, top=top)

    mode = f"{len(user.ratings)} top artists" if user.ratings else f"cold-start: {sorted(user.followed_entities)}"
    print(f"\n  NEW MUSIC FOR YOU  (taste from {mode})")
    print("  " + "-" * 66)
    for r in ranked:
        sc = r.scored
        why = " · ".join(sc.reasons[1:]) or "new release"
        artist = sc.item.meta.get("artist", "")
        print(f"  #{r.rank:<2} [{sc.score:4.2f}]  {sc.item.meta['label'][:38]:38} {('— ' + artist)[:22]}")
        print(f"        {sc.reasons[0].split(' · ')[0]:12} {why}  "
              f"(popularity {sc.excitement:.2f} × taste {sc.personalization:.2f})")
    print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)

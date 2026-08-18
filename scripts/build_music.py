"""Fetch your Spotify data: new-release candidates + top-artist taste.

Run scripts/spotify_auth.py first. Writes:
  data/music_catalog.csv     - new-release albums (candidates)
  data/my_music_ratings.csv  - your top artists as pseudo-ratings (taste)

Usage:
    python scripts/build_music.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.media.spotify_ingest import SpotifyClient  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load(Path("configs/music.yaml").read_text())
    client = SpotifyClient()

    taste = client.top_artists(limit=50, time_range=cfg.get("top_artists_time_range", "medium_term"))
    catalog = client.new_releases(limit=int(cfg.get("new_releases_limit", 60)))

    Path("data").mkdir(exist_ok=True)
    taste.to_csv("data/my_music_ratings.csv", index=False)
    catalog.to_csv("data/music_catalog.csv", index=False)

    tagged = (catalog["genres"].str.len() > 0).sum()
    print(f"top artists (taste): {len(taste)} -> data/my_music_ratings.csv")
    print(f"new releases (candidates): {len(catalog)} ({tagged} with genres) -> data/music_catalog.csv")
    if taste.empty:
        print("note: no top artists returned — a brand-new Spotify account needs listening history.")


if __name__ == "__main__":
    main()

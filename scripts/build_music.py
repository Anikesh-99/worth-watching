"""Fetch your Spotify data and enrich with genres.

Spotify's restricted tier gives no genres, so taste genres come from MusicBrainz
(by artist name) and candidates are discovered via Spotify genre artist-search.
Run scripts/spotify_auth.py first. Writes:
  data/my_music_ratings.csv  - your top artists (taste), genre-enriched
  data/music_catalog.csv     - discovered candidate albums in your genres

Usage:
    python scripts/build_music.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.media.musicbrainz import artist_genres  # noqa: E402
from src.media.spotify_ingest import CATALOG_COLUMNS, SpotifyClient  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load(Path("configs/music.yaml").read_text())
    client = SpotifyClient()

    # 1) taste: your top artists, genre-enriched from MusicBrainz
    taste = client.top_artists(limit=50, time_range=cfg.get("top_artists_time_range", "medium_term"))
    print(f"enriching {len(taste)} top artists with MusicBrainz genres (cached; first run is slow)…")
    taste["genres"] = [("|".join(artist_genres(n))) for n in taste["name"]]
    taste = taste[taste["genres"].str.len() > 0].reset_index(drop=True)

    # 2) seed genres = your most common genres (lowercased for Spotify search)
    counts: Counter = Counter()
    for g in taste["genres"]:
        for tag in g.split("|"):
            counts[tag.lower()] += 1
    seeds = [g for g, _ in counts.most_common(6)]
    print(f"your top genres: {seeds}")

    # 3a) candidates: discover NEW artists in those genres
    exclude = {i.replace("music-", "") for i in taste["item_id"]}
    catalog = client.discover_candidates(seeds, per_genre=10, albums_per_artist=2, exclude_ids=exclude)

    # 3b) candidates: recent releases from YOUR top artists (known quality),
    #     tagged with their MusicBrainz genres and a high popularity prior.
    fav = []
    for _, art in taste.iterrows():
        aid = art["item_id"].replace("music-", "")
        pop = int(round(float(art["rating"]) * 10))          # rank -> 0..100 prior
        for alb in client.artist_recent_albums(aid, limit=2):
            fav.append({
                "item_id": f"music-album-{alb['id']}",
                "name": alb["name"],
                "artist": art["name"],
                "year": (alb.get("release_date") or "0")[:4],
                "popularity": pop,
                "genres": art["genres"],
            })
    catalog = pd.concat([catalog, pd.DataFrame(fav, columns=CATALOG_COLUMNS)]) \
                .drop_duplicates("item_id").reset_index(drop=True)

    Path("data").mkdir(exist_ok=True)
    taste.to_csv("data/my_music_ratings.csv", index=False)
    catalog.to_csv("data/music_catalog.csv", index=False)
    print(f"\ntop artists (taste): {len(taste)} genre-tagged -> data/my_music_ratings.csv")
    print(f"discovered candidates: {len(catalog)} albums -> data/music_catalog.csv")


if __name__ == "__main__":
    main()

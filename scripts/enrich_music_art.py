"""Add album art URLs to an existing data/music_catalog.csv.

Faster than a full build_music re-run: fetches each album's image from Spotify
by id (single /albums/{id} calls, which the restricted tier allows). Idempotent
— only fills rows missing image_url.

Usage:
    python scripts/enrich_music_art.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.media.spotify_ingest import SpotifyClient  # noqa: E402


def main() -> None:
    df = pd.read_csv("data/music_catalog.csv")
    if "image_url" not in df.columns:
        df["image_url"] = ""
    client = SpotifyClient()

    filled = 0
    for i, row in df.iterrows():
        if str(row.get("image_url") or ""):
            continue
        album_id = str(row["item_id"]).replace("music-album-", "")
        try:
            data = client._get(f"/albums/{album_id}")
            imgs = data.get("images") or []
            df.at[i, "image_url"] = imgs[0]["url"] if imgs else ""
            filled += int(bool(imgs))
        except Exception:
            df.at[i, "image_url"] = ""

    df.to_csv("data/music_catalog.csv", index=False)
    print(f"enriched {filled}/{len(df)} albums with art -> data/music_catalog.csv")


if __name__ == "__main__":
    main()

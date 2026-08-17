"""Import your real ratings from a MyAnimeList XML export.

On MAL: Profile -> List -> Export (gives a gzipped XML). Point this at it.
It keeps only scored entries and writes data/my_anime_ratings.csv with the
item_id form the recommender expects (anime-<mal_id>).

Usage:
    python scripts/import_mal.py ~/Downloads/animelist_export.xml.gz
    python scripts/import_mal.py animelist.xml
"""

from __future__ import annotations

import gzip
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def main(path: str) -> None:
    p = Path(path)
    if not p.exists():
        sys.exit(f"file not found: {path}")
    raw = gzip.open(p, "rb").read() if p.suffix == ".gz" else p.read_bytes()

    root = ET.fromstring(raw)
    rows = []
    for a in root.findall("anime"):
        def txt(tag: str) -> str:
            el = a.find(tag)
            return el.text if el is not None and el.text else ""
        mal_id, score, title = txt("series_animedb_id"), txt("my_score"), txt("series_title")
        if mal_id and score and score.isdigit() and int(score) > 0:
            rows.append({"item_id": f"anime-{mal_id}", "rating": int(score), "title": title})

    if not rows:
        sys.exit("No scored entries found in the export.")
    df = pd.DataFrame(rows)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/my_anime_ratings.csv", index=False)
    print(f"Imported {len(df)} scored anime -> data/my_anime_ratings.csv")
    print(df["rating"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/import_mal.py <export.xml[.gz]>")
    main(sys.argv[1])

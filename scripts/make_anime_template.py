"""Write a ratings template for anime — the most popular catalog titles.

Fill `rating` (1-10, MAL scale — how much YOU liked it) for anime you've seen,
delete the rest, save as data/my_anime_ratings.csv. Or skip this and use
scripts/import_mal.py if you have a MAL account.

Usage:
    python scripts/make_anime_template.py [n=60]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main(n: int = 60) -> None:
    cat = pd.read_csv("data/anime_catalog.csv")
    # Most popular by members = most likely you've seen them.
    top = cat.sort_values("members", ascending=False).head(n).copy()
    out = top[["item_id", "title", "type", "year", "genres"]].copy()
    out["rating"] = ""   # <- you fill: 1-10
    Path("data").mkdir(exist_ok=True)
    dest = Path("data/my_anime_ratings_template.csv")
    out.to_csv(dest, index=False)
    print(f"Wrote {len(out)} popular anime -> {dest}")
    print("Fill `rating` (1-10) for ones you've seen, delete the rest,")
    print("save as data/my_anime_ratings.csv. Aim for a spread (some low, some high).")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)

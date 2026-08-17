"""Ratings template for books — the most renowned catalog titles.

An alternative to the Goodreads import: fill `rating` (1-5) for books you've
read, delete the rest, save as data/my_books_ratings.csv.

Usage:
    python scripts/make_books_template.py [n=60]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main(n: int = 60) -> None:
    cat = pd.read_csv("data/books_catalog.csv")
    top = cat.sort_values("editions", ascending=False).head(n).copy()
    out = top[["item_id", "title", "author", "year", "subjects"]].copy()
    out["rating"] = ""   # <- you fill: 1-5
    Path("data").mkdir(exist_ok=True)
    dest = Path("data/my_books_ratings_template.csv")
    out.to_csv(dest, index=False)
    print(f"Wrote {len(out)} well-known books -> {dest}")
    print("Fill `rating` (1-5) for ones you've read, delete the rest,")
    print("save as data/my_books_ratings.csv. Aim for a spread.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)

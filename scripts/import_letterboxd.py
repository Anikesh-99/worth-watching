"""Import your real movie ratings from a Letterboxd export.

On Letterboxd: Settings -> Import & Export -> Export Your Data. The zip contains
`ratings.csv` (Name, Year, Rating 0.5-5). Point this at that file. Each rated
film is matched to TMDB (search by title + year) to recover its id and genres —
the taste signal the content recommender needs — then written to
data/my_movie_ratings.csv.

Letterboxd's half-star scale (0.5-5) is mapped to the platform's 1-5.
Needs TMDB_API_KEY in .env (see README).

Usage:
    python scripts/import_letterboxd.py ~/Downloads/letterboxd/ratings.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.media.tmdb_ingest import TMDBIngest  # noqa: E402

CACHE = Path("data/tmdb_cache/lb_match")
OUT = "data/my_movie_ratings.csv"


def _stars_to_5(x) -> int | None:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v) or v <= 0:
        return None
    return max(1, min(5, round(float(v))))          # 0.5-5 half-stars -> 1-5


def main(path: str) -> None:
    p = Path(path)
    if not p.exists():
        sys.exit(f"file not found: {path}")
    lb = pd.read_csv(p)
    if "Name" not in lb.columns or "Rating" not in lb.columns:
        sys.exit("Doesn't look like a Letterboxd ratings.csv (need Name, Rating columns).")
    lb["_r"] = lb["Rating"].map(_stars_to_5)
    lb = lb[lb["_r"].notna()].copy()
    if lb.empty:
        sys.exit("No rated films found in the export.")

    tmdb = TMDBIngest("movie")
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Matching {len(lb)} rated films to TMDB…")
    rows, matched = [], 0
    for i, (_, f) in enumerate(lb.iterrows(), 1):
        title, year = str(f["Name"]), f.get("Year")
        key = "".join(c for c in f"{title}-{year}"[:60] if c.isalnum())
        cf = CACHE / f"{key}.json"
        if cf.exists():
            hit = json.loads(cf.read_text())
        else:
            row = tmdb.search(title, int(year) if pd.notna(year) else None)
            hit = {"item_id": row["item_id"], "genres": row["genres"]} if row else {}
            cf.write_text(json.dumps(hit))
        if hit.get("item_id"):
            matched += 1
            rows.append({"item_id": hit["item_id"], "rating": int(f["_r"]),
                         "title": title, "genres": hit.get("genres", "")})
        if i % 25 == 0:
            print(f"  {i}/{len(lb)}…")

    if not rows:
        sys.exit("No films matched on TMDB — check the export or your TMDB key.")
    out = pd.DataFrame(rows).drop_duplicates("item_id")
    Path("data").mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Imported {len(out)} movie ratings ({matched}/{len(lb)} matched) -> {OUT}")
    print(out["rating"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/import_letterboxd.py <letterboxd ratings.csv>")
    main(sys.argv[1])

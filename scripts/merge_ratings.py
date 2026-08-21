"""Fold filled ratings from the template into the right ratings files.

Routes each row by vertical, non-destructively (upsert + backup):
  * f1 / nba / soccer -> data/my_ratings.csv       (sports personalization)
  * movie             -> data/my_movie_ratings.csv (content taste)
  * tv                -> data/my_tv_ratings.csv

Rows with a blank/invalid rating are ignored, so you can fill the template over
several sittings. Media titles come from their catalog, so item_id + rating is
enough (genres are read from the catalog).

Usage:
    python scripts/merge_ratings.py [template=data/my_ratings_template.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

FILES = {"movie": "data/my_movie_ratings.csv", "tv": "data/my_tv_ratings.csv"}
SPORTS_FILE = "data/my_ratings.csv"


def _upsert(path: str, new: pd.DataFrame, watched: bool) -> tuple[int, int]:
    """Merge `new` (item_id, rating) into `path` on item_id; keep other columns."""
    p = Path(path)
    if p.exists():
        p.with_suffix(".csv.bak").write_text(p.read_text())          # backup
        existing = pd.read_csv(p)
    else:
        existing = pd.DataFrame(columns=["item_id", "rating"])
    add = new.copy()
    if watched:
        add["watched"] = 1
    n_updates = existing["item_id"].isin(add["item_id"]).sum()
    merged = pd.concat([existing[~existing["item_id"].isin(add["item_id"])], add], ignore_index=True)
    Path("data").mkdir(exist_ok=True)
    merged.to_csv(p, index=False)
    return len(add) - n_updates, len(merged)                          # (new rows, total)


def main(template: str = "data/my_ratings_template.csv") -> None:
    tp = Path(template)
    if not tp.exists():
        sys.exit(f"No {template}. Run scripts/make_ratings_template.py first.")
    tmpl = pd.read_csv(tp)
    tmpl["rating"] = pd.to_numeric(tmpl.get("rating"), errors="coerce")
    valid = tmpl[tmpl["rating"].between(1, 5)].copy()
    valid["rating"] = valid["rating"].round().astype(int)
    if valid.empty:
        sys.exit("No valid 1-5 ratings filled in the template yet — nothing to merge.")

    vert = valid["vertical"] if "vertical" in valid.columns else valid.get("sport", "")
    sports = valid[vert.isin(["f1", "nba", "soccer"])][["item_id", "rating"]]
    if len(sports):
        added, total = _upsert(SPORTS_FILE, sports, watched=True)
        print(f"sports -> {SPORTS_FILE}: +{added} new ({len(sports)} rated), {total} total")
    for kind, path in FILES.items():
        rows = valid[vert == kind][["item_id", "rating"]]
        if len(rows):
            added, total = _upsert(path, rows, watched=False)
            print(f"{kind} -> {path}: +{added} new ({len(rows)} rated), {total} total")

    print("Backups saved as *.csv.bak. Recalibration picks these up next run.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/my_ratings_template.csv")

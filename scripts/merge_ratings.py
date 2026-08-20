"""Fold filled ratings from the template into data/my_ratings.csv.

Non-destructive: reads your existing ratings, upserts the newly-rated rows
(same item_id -> updated; new item_id -> added), and writes back. Rows in the
template with a blank/invalid rating are ignored, so you can fill it over
several sittings. A timestamped backup of the current ratings is kept first.

Usage:
    python scripts/merge_ratings.py [template=data/my_ratings_template.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RATINGS = Path("data/my_ratings.csv")


def main(template: str = "data/my_ratings_template.csv") -> None:
    tp = Path(template)
    if not tp.exists():
        sys.exit(f"No {template}. Run scripts/make_ratings_template.py first.")

    tmpl = pd.read_csv(tp)
    tmpl["rating"] = pd.to_numeric(tmpl.get("rating"), errors="coerce")
    new = tmpl[tmpl["rating"].between(1, 5)][["item_id", "rating"]].copy()
    new["rating"] = new["rating"].round().astype(int)
    new["watched"] = 1
    if new.empty:
        sys.exit("No valid 1-5 ratings filled in the template yet — nothing to merge.")

    existing = pd.read_csv(RATINGS) if RATINGS.exists() else pd.DataFrame(
        columns=["item_id", "rating", "watched"])
    if RATINGS.exists():                                   # keep a backup before writing
        (RATINGS.with_suffix(".csv.bak")).write_text(RATINGS.read_text())

    merged = (pd.concat([existing[~existing["item_id"].isin(new["item_id"])], new],
                        ignore_index=True))
    merged.to_csv(RATINGS, index=False)
    added = len(new) - existing["item_id"].isin(new["item_id"]).sum()
    print(f"Merged {len(new)} ratings ({added} new) -> {RATINGS} now has {len(merged)} total.")
    print("Backup saved to data/my_ratings.csv.bak")
    print("Recalibration picks them up next time you run the dashboard or scripts/evaluate.py.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/my_ratings_template.csv")

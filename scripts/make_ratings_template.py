"""Write a human-readable ratings template for you to fill in.

Includes the 2024 F1 races and the 2024 NBA playoff games — a memorable,
bounded set. Fill the `rating` column (1-5 = how much YOU enjoyed watching)
for events you actually saw, delete the rows you didn't, then save as
data/my_ratings.csv.

Usage:
    python scripts/make_ratings_template.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    f1 = pd.read_csv("data/f1_events.csv", parse_dates=["date"])
    nba = pd.read_csv("data/nba_events.csv", parse_dates=["date"])

    f1 = f1[f1["season"] == 2024].copy()
    f1["sport"] = "f1"
    f1["label"] = "R" + f1["round"].astype(str) + " " + f1["event_name"]

    nba = nba[(nba["season"] == 2024) & (nba["is_playoff"] == 1)].copy()
    nba["sport"] = "nba"
    nba["label"] = nba["away"] + " @ " + nba["home"] + " (playoff)"

    cols = ["item_id", "sport", "date", "label"]
    tmpl = pd.concat([f1[cols], nba[cols]], ignore_index=True)
    # F1 dates are tz-naive, NBA tz-aware; normalize before sorting.
    tmpl["date"] = pd.to_datetime(tmpl["date"], utc=True).dt.tz_localize(None)
    tmpl = tmpl.sort_values("date")
    tmpl["date"] = tmpl["date"].dt.strftime("%Y-%m-%d")
    tmpl["rating"] = ""    # <- you fill this: 1-5
    tmpl["watched"] = ""   # <- 1 if you watched it (optional)

    out = Path("data/my_ratings_template.csv")
    tmpl.to_csv(out, index=False)

    print(f"Wrote {len(tmpl)} events -> {out}")
    print(f"  F1 2024 races: {len(f1)} | NBA 2024 playoff games: {len(nba)}")
    print("\nNext: open it, fill `rating` (1-5) for events you watched, delete the rest,")
    print("save as data/my_ratings.csv. Aim for a SPREAD (some 1-2s, some 4-5s).")


if __name__ == "__main__":
    main()

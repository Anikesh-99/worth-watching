"""Rate a sports event you watched, by name — grows your taste profile.

Searches your event data for a match/race/game and appends the rating to
data/my_ratings.csv (owner-only, local). Run `make update` afterwards (or just
`make serve`) to have personalization pick it up.

Usage:
    python scripts/rate.py "Arsenal Liverpool" 5
    python scripts/rate.py "Dutch Grand Prix" 4
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.features import unify  # noqa: E402
from src.core.ratings import append_sports_rating  # noqa: E402


def _load(name: str):
    p = Path("data") / name
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def main(query: str, rating: int) -> None:
    df = unify(_load("f1_events.csv"), _load("nba_events.csv"), _load("soccer_events.csv"))
    if df.empty:
        sys.exit("No event data. Build datasets first.")

    terms = query.lower().split()
    hits = df[df["label"].str.lower().apply(lambda s: all(t in s for t in terms))]
    hits = hits.sort_values("date", ascending=False)
    if hits.empty:
        sys.exit(f"No event matched '{query}'. Try fewer / different words.")
    if len(hits) > 1:
        print(f"{len(hits)} matches — rating the most recent. Others:")
        for _, r in hits.head(5).iloc[1:].iterrows():
            print(f"  · {r['label']} ({r['sport'].upper()}, {r['date'].date()})  id={r['item_id']}")

    top = hits.iloc[0]
    total = append_sports_rating(top["item_id"], rating)
    print(f"\nRated {rating}/5: {top['label']} ({top['sport'].upper()}, {top['date'].date()})")
    print(f"You now have {total} ratings. Run `make update` (or restart `make serve`) to recalibrate.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('usage: python scripts/rate.py "<event name>" <1-5>')
    main(sys.argv[1], int(sys.argv[2]))

"""Append/update the owner's sports ratings — the feedback that grows the system.

You rate a match/race/game you watched (dashboard click or CLI); it's upserted
into data/my_ratings.csv, and the personalization layer re-calibrates on the
larger set. Owner-only by construction: this writes to local disk and is reached
only from the local FastAPI app or the CLI, never the public static demo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RATINGS_PATH = "data/my_ratings.csv"
_COLUMNS = ["item_id", "rating", "watched"]


def append_sports_rating(item_id: str, rating: int, path: str = RATINGS_PATH) -> int:
    """Upsert one sports rating (latest wins). Returns the new total count."""
    if not 1 <= int(rating) <= 5:
        raise ValueError("rating must be 1-5")
    p = Path(path)
    df = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=_COLUMNS)
    df = df[df["item_id"] != item_id]                       # drop prior rating for this item
    row = pd.DataFrame([{"item_id": item_id, "rating": int(rating), "watched": 1}])
    df = pd.concat([df[_COLUMNS] if set(_COLUMNS).issubset(df.columns) else df, row],
                   ignore_index=True)
    p.parent.mkdir(exist_ok=True)
    df.to_csv(p, index=False)
    return len(df)

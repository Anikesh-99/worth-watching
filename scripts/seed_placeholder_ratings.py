"""Seed a PLACEHOLDER ratings file so the pipeline runs end-to-end.

These are NOT real ratings. They are sampled to loosely track the excitement
index plus a bump for followed teams, purely so Phase 3/4 have signal to wire
against. Replace data/my_ratings.csv with your own 1-5 ratings and the
personalization + evaluation immediately use them instead.

Usage:
    python scripts/seed_placeholder_ratings.py [n=40]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.excitement import ExcitementIndex  # noqa: E402
from src.core.features import unify  # noqa: E402
from src.core.profile import load_user_profile  # noqa: E402


def main(n: int = 40) -> None:
    random.seed(7)
    f1 = pd.read_csv("data/f1_events.csv", parse_dates=["date"]) if Path("data/f1_events.csv").exists() else None
    nba = pd.read_csv("data/nba_events.csv", parse_dates=["date"]) if Path("data/nba_events.csv").exists() else None
    df = unify(f1, nba)
    if df.empty:
        sys.exit("Build datasets first (scripts/build_dataset.py).")

    df = df.copy()
    df["idx"] = ExcitementIndex().score(df)
    user = load_user_profile(["configs/f1.yaml", "configs/nba.yaml"])

    # Sample a spread of events, weighting a bit toward followed teams for realism.
    def weight(row) -> float:
        w = 1.0
        if set(row["entities"]) & user.followed_entities:
            w += 2.0
        return w

    sample = df.sample(n=min(n, len(df)), weights=df.apply(weight, axis=1), random_state=7).copy()

    # Rank-map the index within the sample to a 1-5 spread (so the placeholder
    # looks like real ratings), then nudge followed teams up and add noise.
    sample["pct"] = sample["idx"].rank(pct=True)
    rows = []
    for _, row in sample.iterrows():
        base = 1.0 + 4.0 * row["pct"]
        if set(row["entities"]) & user.followed_entities:
            base += 0.5  # you enjoy your teams more
        rating = int(min(5, max(1, round(base + random.uniform(-0.5, 0.5)))))
        rows.append({"item_id": row["item_id"], "rating": rating,
                     "watched": 1, "label": row["label"]})

    out = pd.DataFrame(rows).sort_values("item_id")
    Path("data").mkdir(exist_ok=True)
    out.to_csv("data/my_ratings.csv", index=False)
    print(f"Wrote {len(out)} PLACEHOLDER ratings -> data/my_ratings.csv")
    print(out["rating"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)

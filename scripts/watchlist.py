"""Generate your spoiler-free weekly watch-list across all sports.

Usage:
    python scripts/watchlist.py                      # default demo week
    python scripts/watchlist.py 2024-04-19 2024-04-25

End-to-end Phase 3 deliverable: candidate generation -> excitement x
personalization -> spoiler-free ranked list. Reasons never reveal a result.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.features import unify  # noqa: E402
from src.core.profile import load_user_profile  # noqa: E402
from src.sports.personalize import DEFAULT_WEIGHTS, calibrate  # noqa: E402
from src.sports.recommender import SportRecommender  # noqa: E402

DEFAULT_START, DEFAULT_END = "2024-04-19", "2024-04-25"  # an NBA-playoffs + F1 week


def _load(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def main(start: str = DEFAULT_START, end: str = DEFAULT_END) -> None:
    df = unify(_load("data/f1_events.csv"), _load("data/nba_events.csv"), _load("data/soccer_events.csv"))
    if df.empty:
        sys.exit("Build datasets first (scripts/build_dataset.py).")

    user = load_user_profile(["configs/f1.yaml", "configs/nba.yaml"])

    # Use taste weights calibrated from your ratings when available; else the prior.
    weights = DEFAULT_WEIGHTS
    if user.ratings:
        rated = df[df["item_id"].isin(user.ratings)].copy()
        rated["rating"] = rated["item_id"].map(user.ratings)
        weights = calibrate(rated, user)
        print(f"  (taste weights calibrated from {len(rated)} ratings: "
              f"followed={weights.followed_boost}, stakes={weights.stakes_boost})")

    rec = SportRecommender(df, weights=weights)
    s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
    ranked = rec.watchlist(s, e, user, top=15)

    print(f"\n  YOUR WATCH-LIST · {start} → {end}  (spoiler-free)")
    print(f"  following: {', '.join(sorted(user.followed_entities))}")
    print("  " + "-" * 74)
    if not ranked:
        print("  (no events in this window)")
    for r in ranked:
        sc = r.scored
        why = " · ".join(sc.reasons)
        print(f"  #{r.rank:<2} [{sc.score:4.2f}]  {why}")
        print(f"        excitement {sc.excitement:.2f} × taste {sc.personalization:.2f}")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(args if len(args) == 2 else ()))

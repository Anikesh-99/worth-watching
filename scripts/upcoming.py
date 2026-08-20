"""Rank UPCOMING events by watchability x personalization (spoiler-free).

Fetches live schedules (best-effort) and ranks not-yet-played fixtures across
sports: F1 races (Jolpica) + Premier League / Champions League matches this week
(ESPN), interleaved chronologically. Uses the transparent WatchabilityIndex, not
an excitement forecast (which the predictor showed is unreliable pre-game).

Usage:
    python scripts/upcoming.py [top=15]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.profile import load_user_profile  # noqa: E402
from src.sports.fixtures import collect_upcoming_fixtures  # noqa: E402
from src.sports.upcoming import UpcomingRecommender  # noqa: E402


def _load(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def main(top: int = 15) -> None:
    fixtures, _ = collect_upcoming_fixtures(
        _load("data/f1_events.csv"), _load("data/soccer_events.csv"))
    if fixtures.empty:
        print("No upcoming fixtures from the live schedules (off-season or fetch failed).")
        print("The engine is ready; fixtures populate when a season is live.")
        return

    user = load_user_profile(["configs/f1.yaml", "configs/nba.yaml", "configs/soccer.yaml"],
                             "data/__no_ratings__.csv")
    ranked = UpcomingRecommender(fixtures).recommend(user, top=top)

    print(f"\n  WORTH WATCHING — UPCOMING  ({len(fixtures)} fixtures)")
    print("  " + "-" * 66)
    for r in ranked:
        sc = r.scored
        why = " · ".join(sc.reasons[1:]) or "on the calendar"
        print(f"  #{r.rank:<2} [{sc.score:4.2f}]  {sc.item.meta['label'][:34]:34} {sc.item.when.strftime('%b %d')}")
        print(f"        {sc.reasons[0].split(' · ')[0]:11} {why}  "
              f"(watchability {sc.excitement:.2f} × taste {sc.personalization:.2f})")
    print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)

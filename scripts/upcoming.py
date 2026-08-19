"""Rank UPCOMING events by watchability x personalization (spoiler-free).

Fetches the current F1 schedule (best-effort) and ranks the not-yet-run races.
NBA/soccer upcoming join automatically once those seasons are live and their
fixtures are supplied. Uses the transparent WatchabilityIndex, not an excitement
forecast (which the predictor showed is unreliable pre-game).

Usage:
    python scripts/upcoming.py [top=15]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.excitement import ExcitementIndex  # noqa: E402
from src.core.features import normalize_f1  # noqa: E402
from src.core.profile import load_user_profile  # noqa: E402
from src.sports.fixtures import f1_fixtures, fetch_f1_schedule  # noqa: E402
from src.sports.upcoming import UpcomingRecommender  # noqa: E402


def main(top: int = 15) -> None:
    f1 = pd.read_csv("data/f1_events.csv", parse_dates=["date"])
    u = normalize_f1(f1)
    excitement = dict(zip(u["item_id"], ExcitementIndex().score(u)))

    races = fetch_f1_schedule()
    if not races:
        print("No upcoming F1 races from the live schedule (off-season or fetch failed).")
        print("The engine is ready; fixtures populate when a season is live.")
        return
    fixtures = f1_fixtures(races, f1, excitement)

    user = load_user_profile(["configs/f1.yaml", "configs/nba.yaml"])
    ranked = UpcomingRecommender(fixtures).recommend(user, top=top)

    print(f"\n  WORTH WATCHING THIS SEASON — UPCOMING  ({len(fixtures)} races)")
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

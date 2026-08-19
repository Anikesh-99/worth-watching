"""Refresh only web_static/data/upcoming.json from live fixtures.

Used by the scheduled cron so the demo's "Coming up" stays current without
rebuilding (and possibly regressing) the rest of the bundle. Needs only the
committed data/f1_events.csv + the live F1 schedule.

Usage:
    python scripts/refresh_upcoming.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.excitement import ExcitementIndex  # noqa: E402
from src.core.features import normalize_f1  # noqa: E402
from src.core.profile import load_user_profile  # noqa: E402
from src.sports.fixtures import f1_fixtures, fetch_f1_schedule  # noqa: E402
from src.sports.upcoming import UpcomingRecommender  # noqa: E402


def main() -> None:
    f1 = pd.read_csv("data/f1_events.csv", parse_dates=["date"])
    races = fetch_f1_schedule()
    u = normalize_f1(f1)
    excitement = dict(zip(u["item_id"], ExcitementIndex().score(u)))
    fixtures = f1_fixtures(races, f1, excitement)

    user = load_user_profile(["configs/f1.yaml", "configs/nba.yaml", "configs/soccer.yaml"],
                             "data/__no_ratings__.csv")
    ranked = UpcomingRecommender(fixtures).recommend(user, top=20)

    def to_dict(r) -> dict:
        sc = r.scored
        return {"rank": r.rank, "item_id": sc.item.item_id, "sport": sc.item.vertical,
                "label": sc.item.meta["label"], "date": sc.item.when.strftime("%Y-%m-%d"),
                "score": round(sc.score, 3), "excitement": round(sc.excitement, 3),
                "taste": round(sc.personalization, 3), "tier": sc.reasons[0].split(" · ")[0],
                "reasons": sc.reasons[1:]}

    out = [to_dict(r) for r in ranked]
    Path("web_static/data").mkdir(parents=True, exist_ok=True)
    Path("web_static/data/upcoming.json").write_text(json.dumps(out))
    print(f"refreshed upcoming.json: {len(out)} fixtures")


if __name__ == "__main__":
    main()

"""Refresh only web_static/data/upcoming.json from live fixtures.

Used by the scheduled cron so the demo's "Coming up" stays current without
rebuilding (and possibly regressing) the rest of the bundle. Needs only the
committed data/f1_events.csv + data/soccer_events.csv + the live schedules.

Usage:
    python scripts/refresh_upcoming.py
"""

from __future__ import annotations

import json
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


def main() -> None:
    fixtures, logos = collect_upcoming_fixtures(
        _load("data/f1_events.csv"), _load("data/soccer_events.csv"))

    user = load_user_profile(["configs/f1.yaml", "configs/nba.yaml", "configs/soccer.yaml"],
                             "data/__no_ratings__.csv")
    ranked = UpcomingRecommender(fixtures).recommend(user, top=25)

    def to_dict(r) -> dict:
        sc = r.scored
        d = {"rank": r.rank, "item_id": sc.item.item_id, "sport": sc.item.vertical,
             "label": sc.item.meta["label"], "date": sc.item.when.strftime("%Y-%m-%d"),
             "score": round(sc.score, 3), "excitement": round(sc.excitement, 3),
             "taste": round(sc.personalization, 3), "tier": sc.reasons[0].split(" · ")[0],
             "reasons": sc.reasons[1:]}
        d.update(logos.get(sc.item.item_id, {}))     # crest URLs for soccer tiles
        return d

    out = [to_dict(r) for r in ranked]
    Path("web_static/data").mkdir(parents=True, exist_ok=True)
    Path("web_static/data/upcoming.json").write_text(json.dumps(out))
    print(f"refreshed upcoming.json: {len(out)} fixtures")


if __name__ == "__main__":
    main()

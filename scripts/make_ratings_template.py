"""Write a ratings template of events you probably watched but haven't rated yet.

Bundles the newest, most-rateable events into one CSV to fill:
  * F1 races from the last season and this season's completed rounds,
  * football matches involving a club you follow (recent seasons),
  * (kept) the 2024 NBA playoff games.

Already-rated events (in data/my_ratings.csv) are excluded, so you only see
what's new. Fill the `rating` column (1-5 = how much YOU enjoyed watching),
delete rows you didn't see, then run `scripts/merge_ratings.py` to fold them
into data/my_ratings.csv (non-destructive — it upserts, never overwrites).

Usage:
    python scripts/make_ratings_template.py            # -> data/my_ratings_template.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.excitement import ExcitementIndex  # noqa: E402
from src.core.features import normalize_soccer  # noqa: E402
from src.core.profile import load_user_profile  # noqa: E402

OUT = "data/my_ratings_template.csv"
F1_SEASONS = (2025, 2026)          # last season + this season's completed rounds
SOCCER_SEASONS = (2025, 2026)      # your clubs' recent matches
SOCCER_TOP = 45                    # keep it a bounded set: the most memorable ones


def _load(name: str) -> pd.DataFrame | None:
    p = Path(f"data/{name}")
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def main() -> None:
    followed = load_user_profile(["configs/soccer.yaml"]).followed_entities
    rated = set()
    rp = Path("data/my_ratings.csv")
    if rp.exists():
        rated = set(pd.read_csv(rp)["item_id"].astype(str))

    parts = []

    f1 = _load("f1_events.csv")
    if f1 is not None:
        f = f1[f1["season"].isin(F1_SEASONS)].copy()
        f["sport"] = "f1"
        f["label"] = "R" + f["round"].astype(str) + " " + f["event_name"] + " " + f["season"].astype(str)
        parts.append(f[["item_id", "sport", "date", "label"]])

    soccer = _load("soccer_events.csv")
    if soccer is not None:
        s = soccer[soccer["season"].isin(SOCCER_SEASONS)].copy()
        mine = s[[bool({h, a} & followed) for h, a in zip(s["home"], s["away"])]].copy()
        # bound it to the most memorable ones (a 38-game season is too much to rate)
        ex = ExcitementIndex().score(normalize_soccer(mine))
        mine = mine.assign(_ex=ex.to_numpy()).nlargest(SOCCER_TOP, "_ex")
        mine["sport"] = "soccer"
        lg = mine["league"].map({"eng.1": "PL", "uefa.champions": "UCL"}).fillna("")
        mine["label"] = mine["home"] + " v " + mine["away"] + " (" + lg + ")"
        parts.append(mine[["item_id", "sport", "date", "label"]])

    nba = _load("nba_events.csv")
    if nba is not None:
        n = nba[(nba["season"] == 2024) & (nba["is_playoff"] == 1)].copy()
        n["sport"] = "nba"
        n["label"] = n["away"] + " @ " + n["home"] + " (playoff)"
        parts.append(n[["item_id", "sport", "date", "label"]])

    tmpl = pd.concat(parts, ignore_index=True)
    tmpl["date"] = pd.to_datetime(tmpl["date"], utc=True).dt.tz_localize(None)
    tmpl = tmpl[~tmpl["item_id"].astype(str).isin(rated)]                  # only unrated
    tmpl = tmpl.sort_values(["sport", "date"]).reset_index(drop=True)
    tmpl["date"] = tmpl["date"].dt.strftime("%Y-%m-%d")
    tmpl["rating"] = ""                                                    # <- you fill: 1-5

    tmpl[["item_id", "sport", "date", "label", "rating"]].to_csv(OUT, index=False)
    by = tmpl.groupby("sport").size().to_dict()
    print(f"Wrote {OUT}: {len(tmpl)} unrated events to consider {by}")
    print("Fill the `rating` column (1-5), delete rows you didn't watch, then:")
    print("    python scripts/merge_ratings.py")


if __name__ == "__main__":
    main()

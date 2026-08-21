"""Write a ratings template of things you probably watched but haven't rated.

One CSV to fill across verticals:
  * F1 races from the last season and this season's completed rounds,
  * football matches involving a club you follow (most memorable ones),
  * the 2024 NBA playoff games,
  * the best-known Movies and TV shows (once their TMDB catalogs are built).

Already-rated items are excluded (per-vertical). Fill the `rating` column
(1-5 = how much YOU enjoyed it), delete rows you didn't see, then run
`scripts/merge_ratings.py` — it routes each vertical to the right ratings file
(sports -> my_ratings.csv; movie/tv -> my_<vertical>_ratings.csv), non-destructively.

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
F1_SEASONS = (2025, 2026)
SOCCER_SEASONS = (2025, 2026)
SOCCER_TOP = 45
SCREEN_TOP = 40                     # top movies / TV shows to offer for rating


def _load(name: str) -> pd.DataFrame | None:
    p = Path(f"data/{name}")
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def _load_catalog(name: str) -> pd.DataFrame | None:
    p = Path(f"data/{name}")
    return pd.read_csv(p) if p.exists() else None


def _rated_ids(name: str) -> set[str]:
    p = Path(f"data/{name}")
    return set(pd.read_csv(p)["item_id"].astype(str)) if p.exists() else set()


def main() -> None:
    followed = load_user_profile(["configs/soccer.yaml"]).followed_entities
    parts = []

    f1 = _load("f1_events.csv")
    if f1 is not None:
        f = f1[f1["season"].isin(F1_SEASONS)].copy()
        f["vertical"] = "f1"
        f["label"] = "R" + f["round"].astype(str) + " " + f["event_name"] + " " + f["season"].astype(str)
        parts.append(f[["item_id", "vertical", "date", "label"]])

    soccer = _load("soccer_events.csv")
    if soccer is not None:
        s = soccer[soccer["season"].isin(SOCCER_SEASONS)].copy()
        mine = s[[bool({h, a} & followed) for h, a in zip(s["home"], s["away"])]].copy()
        ex = ExcitementIndex().score(normalize_soccer(mine))
        mine = mine.assign(_ex=ex.to_numpy()).nlargest(SOCCER_TOP, "_ex")
        mine["vertical"] = "soccer"
        lg = mine["league"].map({"eng.1": "PL", "uefa.champions": "UCL"}).fillna("")
        mine["label"] = mine["home"] + " v " + mine["away"] + " (" + lg + ")"
        parts.append(mine[["item_id", "vertical", "date", "label"]])

    nba = _load("nba_events.csv")
    if nba is not None:
        n = nba[(nba["season"] == 2024) & (nba["is_playoff"] == 1)].copy()
        n["vertical"] = "nba"
        n["label"] = n["away"] + " @ " + n["home"] + " (playoff)"
        parts.append(n[["item_id", "vertical", "date", "label"]])

    for kind, catalog in (("movie", "movies_catalog.csv"), ("tv", "tv_catalog.csv")):
        cat = _load_catalog(catalog)
        if cat is not None and len(cat):
            top = cat.nlargest(SCREEN_TOP, "votes").copy()
            top["vertical"] = kind
            top["date"] = pd.to_datetime(top["year"].fillna(0).astype(int).astype(str) + "-01-01",
                                         errors="coerce")
            top["label"] = top["title"] + " (" + top["year"].fillna(0).astype(int).astype(str) + ")"
            parts.append(top[["item_id", "vertical", "date", "label"]])

    tmpl = pd.concat(parts, ignore_index=True)
    tmpl["date"] = pd.to_datetime(tmpl["date"], utc=True).dt.tz_localize(None)

    # exclude already-rated, per vertical (sports share my_ratings.csv; media split)
    rated = {"sports": _rated_ids("my_ratings.csv"),
             "movie": _rated_ids("my_movie_ratings.csv"),
             "tv": _rated_ids("my_tv_ratings.csv")}
    def seen(row):
        bucket = row["vertical"] if row["vertical"] in ("movie", "tv") else "sports"
        return str(row["item_id"]) in rated[bucket]
    tmpl = tmpl[~tmpl.apply(seen, axis=1)]

    tmpl = tmpl.sort_values(["vertical", "date"]).reset_index(drop=True)
    tmpl["date"] = tmpl["date"].dt.strftime("%Y-%m-%d")
    tmpl["rating"] = ""                                                    # <- you fill: 1-5

    tmpl[["item_id", "vertical", "date", "label", "rating"]].to_csv(OUT, index=False)
    by = tmpl.groupby("vertical").size().to_dict()
    print(f"Wrote {OUT}: {len(tmpl)} unrated items to consider {by}")
    if "movie" not in by:
        print("(Movies/TV appear here once their catalogs are built — add a TMDB key,")
        print(" then: python scripts/build_dataset.py configs/movies.yaml)")
    print("Fill the `rating` column (1-5), delete what you didn't watch, then:")
    print("    python scripts/merge_ratings.py")


if __name__ == "__main__":
    main()

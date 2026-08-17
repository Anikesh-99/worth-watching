"""Build a static version of the dashboard for GitHub Pages.

GitHub Pages can't run Python, so we bake the real pipeline's output into JSON:
every event is scored by the actual SportRecommender / AnimeRecommender here,
and the static page only filters/sorts/slices in the browser. The ML scores are
genuine — just precomputed.

Privacy: the PUBLIC build uses NO personal ratings (default taste prior for
sports, cold-start genres for anime). Nothing personal is shipped.

Usage:
    python scripts/build_static.py   # -> web_static/data/*.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.features import unify  # noqa: E402
from src.core.profile import load_user_profile  # noqa: E402
from src.media.recommender import AnimeRecommender  # noqa: E402
from src.sports.personalize import DEFAULT_WEIGHTS  # noqa: E402
from src.sports.recommender import SportRecommender  # noqa: E402

OUT = Path("web_static/data")
_MIN, _MAX = datetime(1990, 1, 1), datetime(2100, 1, 1)


def _load(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def _scored_dict(sc, rank: int) -> dict:
    return {
        "rank": rank,
        "item_id": sc.item.item_id,
        "sport": sc.item.vertical,
        "label": sc.item.meta["label"],
        "date": sc.item.when.strftime("%Y-%m-%d"),
        "score": round(sc.score, 3),
        "excitement": round(sc.excitement, 3),
        "taste": round(sc.personalization, 3),
        "tier": sc.reasons[0].split(" · ")[0],
        "reasons": sc.reasons[1:],
    }


def _busiest_window(df: pd.DataFrame, days: int = 14) -> tuple[str, str]:
    dates = df["date"].sort_values().dt.normalize()
    span = pd.Timedelta(f"{days}D")
    best, best_n = dates.min(), -1
    for d in dates.unique():
        d = pd.Timestamp(d)
        n = ((dates >= d) & (dates < d + span)).sum()
        if n > best_n:
            best_n, best = n, d
    return best.strftime("%Y-%m-%d"), (best + pd.Timedelta(f"{days - 1}D")).strftime("%Y-%m-%d")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # PUBLIC profile: followed entities from config, NO personal ratings.
    user = load_user_profile(["configs/f1.yaml", "configs/nba.yaml"], "data/__no_ratings__.csv")

    # ---- sports: score every event with the default prior --------------
    df = unify(_load("data/f1_events.csv"), _load("data/nba_events.csv"))
    rec = SportRecommender(df, weights=DEFAULT_WEIGHTS)
    scored = rec.score(rec.generate_candidates(_MIN, _MAX), user)
    sports = [_scored_dict(s, 0) for s in scored]
    (OUT / "sports.json").write_text(json.dumps(sports))

    # ---- anime: cold-start recommendations -----------------------------
    anime = []
    anime_meta = None
    cat = _load_csv("data/anime_catalog.csv")
    if cat is not None:
        au = load_user_profile(["configs/anime.yaml"], "data/__no_ratings__.csv")
        arec = AnimeRecommender(cat)
        ranked = arec.recommend(au, top=60)
        for r in ranked:
            d = _scored_dict(r.scored, r.rank)
            it = r.scored.item
            d["date"] = (it.meta.get("type") or "") + (f" · {it.when.year}" if it.when else "")
            anime.append(d)
        anime_meta = {"catalog_size": len(cat), "ratings": 0,
                      "cold_start_genres": sorted(au.followed_entities)}
    (OUT / "anime.json").write_text(json.dumps(anime))

    # ---- meta ----------------------------------------------------------
    ds, de = _busiest_window(df)
    meta = {
        "followed": sorted(user.followed_entities),
        "weights": {"followed_boost": DEFAULT_WEIGHTS.followed_boost,
                    "stakes_boost": DEFAULT_WEIGHTS.stakes_boost},
        "calibrated_from_ratings": 0,
        "date_min": df["date"].min().strftime("%Y-%m-%d"),
        "date_max": df["date"].max().strftime("%Y-%m-%d"),
        "counts": {k: int(v) for k, v in df.groupby("sport").size().items()},
        "default_start": ds, "default_end": de,
        "anime": anime_meta,
    }
    (OUT / "meta.json").write_text(json.dumps(meta))

    print(f"Wrote static bundle -> {OUT}/")
    print(f"  sports events: {len(sports)} | anime recs: {len(anime)} | window {ds}..{de}")


def _load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


if __name__ == "__main__":
    main()

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
from src.core.excitement import ExcitementIndex  # noqa: E402
from src.core.features import normalize_f1, unify  # noqa: E402
from src.core.profile import load_user_profile  # noqa: E402
from src.media.book_recommender import BookRecommender  # noqa: E402
from src.media.content import augment_catalog  # noqa: E402
from src.media.music_recommender import MusicRecommender  # noqa: E402
from src.media.recommender import AnimeRecommender  # noqa: E402
from src.sports.fixtures import f1_fixtures, fetch_f1_schedule  # noqa: E402
from src.sports.personalize import DEFAULT_WEIGHTS  # noqa: E402
from src.sports.recommender import SportRecommender  # noqa: E402
from src.sports.upcoming import UpcomingRecommender  # noqa: E402

OUT = Path("web_static/data")
_MIN, _MAX = datetime(1990, 1, 1), datetime(2100, 1, 1)


def _load(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def _scored_dict(sc, rank: int, images: dict | None = None) -> dict:
    d = {
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
    if images and sc.item.item_id in images:
        d.update(images[sc.item.item_id])       # image / home_logo / away_logo
    return d


def _image_lookup() -> dict:
    """item_id -> image fields: media covers, and matchup crests for sports."""
    images: dict = {}
    for path, col in [("data/anime_catalog.csv", "image_url"),
                      ("data/books_catalog.csv", "image_url"),
                      ("data/music_catalog.csv", "image_url")]:
        df = _load_csv(path)
        if df is not None and col in df.columns:
            for i, v in zip(df["item_id"], df[col].fillna("")):
                if v:
                    images[str(i)] = {"image": v}
    for path in ("data/nba_events.csv", "data/soccer_events.csv"):
        df = _load_csv(path)
        if df is not None and "home_logo" in df.columns:
            for i, hl, al in zip(df["item_id"], df["home_logo"].fillna(""), df["away_logo"].fillna("")):
                if hl or al:
                    images[str(i)] = {"home_logo": hl, "away_logo": al}
    return images


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
    images = _image_lookup()

    # PUBLIC profile: followed entities from config, NO personal ratings.
    user = load_user_profile(["configs/f1.yaml", "configs/nba.yaml", "configs/soccer.yaml"],
                             "data/__no_ratings__.csv")

    # ---- sports: score every event with the default prior --------------
    df = unify(_load("data/f1_events.csv"), _load("data/nba_events.csv"), _load("data/soccer_events.csv"))
    rec = SportRecommender(df, weights=DEFAULT_WEIGHTS)
    scored = rec.score(rec.generate_candidates(_MIN, _MAX), user)
    sports = [_scored_dict(s, 0, images) for s in scored]
    (OUT / "sports.json").write_text(json.dumps(sports))

    # ---- upcoming: rank live fixtures by watchability -------------------
    upcoming = _upcoming(user)
    (OUT / "upcoming.json").write_text(json.dumps(upcoming))

    # ---- media verticals: cold-start recommendations -------------------
    # anime/books use cold-start (public catalogs, no personal data). music is
    # personalized — its catalog IS the user's taste, so it uses real ratings.
    media_specs = [
        ("anime", "data/anime_catalog.csv", "configs/anime.yaml", AnimeRecommender, None),
        ("book", "data/books_catalog.csv", "configs/books.yaml", BookRecommender, None),
        ("music", "data/music_catalog.csv", "configs/music.yaml", MusicRecommender, "data/my_music_ratings.csv"),
    ]
    media_meta = {}
    counts = {}
    for vertical, catalog_path, config, RecClass, ratings_path in media_specs:
        cat = _load_csv(catalog_path)
        recs = []
        if cat is not None:
            if ratings_path and Path(ratings_path).exists():   # personalized vertical
                rated = pd.read_csv(ratings_path)
                if set(RecClass.tag_cols) & set(rated.columns):
                    cat = augment_catalog(cat, rated)
                mu = load_user_profile([config], ratings_path)
            else:                                              # cold-start
                mu = load_user_profile([config], "data/__no_ratings__.csv")
            for r in RecClass(cat).recommend(mu, top=60):
                d = _scored_dict(r.scored, r.rank, images)
                it = r.scored.item
                sub = it.meta.get("type") or it.meta.get("author") or it.meta.get("artist") or ""
                d["date"] = (str(sub)[:22] + (f" · {it.when.year}" if it.when and it.when.year > 1400 else "")).strip(" ·")
                recs.append(d)
            media_meta[vertical] = {"catalog_size": len(cat), "ratings": len(mu.ratings),
                                    "cold_start": sorted(mu.followed_entities)}
        (OUT / f"{vertical}.json").write_text(json.dumps(recs))
        counts[vertical] = len(recs)

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
        "media": media_meta,
        "upcoming": len(upcoming),
    }
    (OUT / "meta.json").write_text(json.dumps(meta))

    print(f"Wrote static bundle -> {OUT}/  (upcoming: {len(upcoming)})")
    print(f"  sports events: {len(sports)} | media: {counts} | window {ds}..{de}")


def _load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def _upcoming(user) -> list[dict]:
    """Rank live upcoming fixtures (F1 now; others when their seasons are live)."""
    try:
        f1 = _load("data/f1_events.csv")
        races = fetch_f1_schedule()
        if f1 is None or not races:
            return []
        u = normalize_f1(f1)
        excitement = dict(zip(u["item_id"], ExcitementIndex().score(u)))
        fixtures = f1_fixtures(races, f1, excitement)
        ranked = UpcomingRecommender(fixtures).recommend(user, top=20)
        return [_scored_dict(r.scored, r.rank) for r in ranked]
    except Exception as exc:  # never let a live fetch break the build
        print(f"  upcoming skipped: {exc}")
        return []


if __name__ == "__main__":
    main()

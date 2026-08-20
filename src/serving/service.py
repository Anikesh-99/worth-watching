"""WatchlistService — loads data once and answers watch-list queries.

Wraps the same SportRecommender the CLI uses, so the dashboard and
scripts/watchlist.py return identical results. Taste weights are calibrated
from the user's ratings at startup (neutral if none), keeping the API honest to
the Phase 4 finding.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.core.features import unify
from src.core.profile import load_user_profile
from src.media.book_recommender import BookRecommender
from src.media.content import augment_catalog
from src.media.music_recommender import MusicRecommender
from src.media.recommender import AnimeRecommender
from src.sports.personalize import DEFAULT_WEIGHTS, SportTaste, calibrate_per_sport
from src.sports.recommender import SportRecommender


class WatchlistService:
    def __init__(self, data_dir: str = "data",
                 configs: tuple[str, ...] = ("configs/f1.yaml", "configs/nba.yaml")) -> None:
        d = Path(data_dir)

        def _load(name: str):
            p = d / name
            return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None

        self.df = unify(_load("f1_events.csv"), _load("nba_events.csv"), _load("soccer_events.csv"))
        if self.df.empty:
            raise RuntimeError("No event data. Run scripts/build_dataset.py first.")

        self._configs = [*configs, "configs/soccer.yaml"]
        self.recalibrate()   # loads ratings, calibrates taste weights, builds the ranker

        # Media verticals (optional): loaded only if their catalog was built.
        # Each entry: vertical -> (recommender, user_profile).
        self.media: dict[str, tuple] = {}
        media_specs = [
            ("anime", "anime_catalog.csv", "configs/anime.yaml", "my_anime_ratings.csv", AnimeRecommender),
            ("book", "books_catalog.csv", "configs/books.yaml", "my_books_ratings.csv", BookRecommender),
            ("music", "music_catalog.csv", "configs/music.yaml", "my_music_ratings.csv", MusicRecommender),
        ]
        for vertical, catalog_name, config, ratings_name, RecClass in media_specs:
            cp = d / catalog_name
            if not cp.exists():
                continue
            catalog = pd.read_csv(cp)
            rpath = d / ratings_name
            # Ground taste in rated items' own tags (AniList genres, Spotify
            # top-artist genres) so the dashboard matches the CLI recommenders.
            if rpath.exists():
                rated = pd.read_csv(rpath)
                if set(RecClass.tag_cols) & set(rated.columns):
                    catalog = augment_catalog(catalog, rated)
            self.media[vertical] = (RecClass(catalog), load_user_profile([config], str(rpath)))

    def recalibrate(self) -> None:
        """(Re)load ratings, recompute taste weights, rebuild the sports ranker.

        Called at startup and after a new rating is logged, so feedback takes
        effect immediately without restarting the app.
        """
        self.user = load_user_profile(self._configs)
        self.taste = SportTaste(default=DEFAULT_WEIGHTS)
        if self.user.ratings:
            rated = self.df[self.df["item_id"].isin(self.user.ratings)].copy()
            rated["rating"] = rated["item_id"].map(self.user.ratings)
            self.taste = calibrate_per_sport(rated, self.user)   # each sport learns its own
        self.rec = SportRecommender(self.df, weights=self.taste)

    # -- metadata for the UI ---------------------------------------------

    def meta(self) -> dict:
        default_start, default_end = self._busiest_window(days=14)
        media = {}
        for vertical, (rec, user) in self.media.items():
            media[vertical] = {
                "catalog_size": len(rec.df),
                "ratings": len(user.ratings),
                "cold_start": sorted(user.followed_entities),
            }
        return {
            "followed": sorted(self.user.followed_entities),
            "weights": {"followed_boost": self.taste.default.followed_boost,
                        "stakes_boost": self.taste.default.stakes_boost},
            "weights_by_sport": {
                sport: {"followed_boost": w.followed_boost, "stakes_boost": w.stakes_boost}
                for sport, w in self.taste.per_sport.items()},
            "calibrated_from_ratings": len(self.user.ratings),
            "date_min": self.df["date"].min().strftime("%Y-%m-%d"),
            "date_max": self.df["date"].max().strftime("%Y-%m-%d"),
            "counts": self.df.groupby("sport").size().to_dict(),
            "default_start": default_start,
            "default_end": default_end,
            "media": media,
        }

    # -- media recommendations (anime / book) ----------------------------

    def media_recs(self, vertical: str, top: int = 25) -> list[dict]:
        entry = self.media.get(vertical)
        if entry is None:
            return []
        rec, user = entry
        out = []
        for r in rec.recommend(user, top=top):
            sc, it = r.scored, r.scored.item
            sub = it.meta.get("type") or it.meta.get("author") or it.meta.get("artist") or ""
            out.append({
                "rank": r.rank,
                "item_id": it.item_id,
                "sport": vertical,
                "label": it.meta["label"],
                "date": (str(sub)[:22] + (f" · {it.when.year}" if it.when and it.when.year > 1400 else "")).strip(" ·"),
                "score": round(sc.score, 3),
                "excitement": round(sc.excitement, 3),
                "taste": round(sc.personalization, 3),
                "tier": sc.reasons[0].split(" · ")[0],
                "reasons": sc.reasons[1:],
            })
        return out

    def _busiest_window(self, days: int) -> tuple[str, str]:
        dates = self.df["date"].sort_values().dt.normalize()
        best_start, best_count = dates.min(), -1
        for d in dates.unique():
            d = pd.Timestamp(d)
            count = ((dates >= d) & (dates < d + pd.Timedelta(days=days))).sum()
            if count > best_count:
                best_count, best_start = count, d
        return best_start.strftime("%Y-%m-%d"), (best_start + pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")

    # -- the watch-list ---------------------------------------------------

    def watchlist(self, start: str, end: str, sport: str = "all", top: int = 25) -> list[dict]:
        s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
        ranked = self.rec.watchlist(s, e, self.user)
        out = []
        for r in ranked:
            sc = r.scored
            if sport != "all" and sc.item.vertical != sport:
                continue
            out.append({
                "rank": r.rank,
                "item_id": sc.item.item_id,
                "sport": sc.item.vertical,
                "label": sc.item.meta["label"],
                "date": sc.item.when.strftime("%Y-%m-%d"),
                "score": round(sc.score, 3),
                "excitement": round(sc.excitement, 3),
                "taste": round(sc.personalization, 3),
                "tier": sc.reasons[0].split(" · ")[0],
                "reasons": sc.reasons[1:],
            })
            if len(out) >= top:
                break
        # re-rank within the filtered/sport view
        for i, row in enumerate(out, 1):
            row["rank"] = i
        return out

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
from src.media.recommender import AnimeRecommender
from src.sports.personalize import DEFAULT_WEIGHTS, calibrate
from src.sports.recommender import SportRecommender


class WatchlistService:
    def __init__(self, data_dir: str = "data",
                 configs: tuple[str, ...] = ("configs/f1.yaml", "configs/nba.yaml")) -> None:
        d = Path(data_dir)
        f1 = pd.read_csv(d / "f1_events.csv", parse_dates=["date"]) if (d / "f1_events.csv").exists() else None
        nba = pd.read_csv(d / "nba_events.csv", parse_dates=["date"]) if (d / "nba_events.csv").exists() else None
        self.df = unify(f1, nba)
        if self.df.empty:
            raise RuntimeError("No event data. Run scripts/build_dataset.py first.")

        self.user = load_user_profile(list(configs))
        self.weights = DEFAULT_WEIGHTS
        if self.user.ratings:
            rated = self.df[self.df["item_id"].isin(self.user.ratings)].copy()
            rated["rating"] = rated["item_id"].map(self.user.ratings)
            self.weights = calibrate(rated, self.user)

        self.rec = SportRecommender(self.df, weights=self.weights)

        # Media vertical (optional): loaded only if a catalog has been built.
        self.anime_rec = None
        self.anime_user = None
        catalog_path = d / "anime_catalog.csv"
        if catalog_path.exists():
            catalog = pd.read_csv(catalog_path)
            self.anime_rec = AnimeRecommender(catalog)
            self.anime_user = load_user_profile(["configs/anime.yaml"], str(d / "my_anime_ratings.csv"))

    # -- metadata for the UI ---------------------------------------------

    def meta(self) -> dict:
        default_start, default_end = self._busiest_window(days=14)
        anime = None
        if self.anime_rec is not None:
            anime = {
                "catalog_size": len(self.anime_rec.df),
                "ratings": len(self.anime_user.ratings),
                "cold_start_genres": sorted(self.anime_user.followed_entities),
            }
        return {
            "followed": sorted(self.user.followed_entities),
            "weights": {"followed_boost": self.weights.followed_boost,
                        "stakes_boost": self.weights.stakes_boost},
            "calibrated_from_ratings": len(self.user.ratings),
            "date_min": self.df["date"].min().strftime("%Y-%m-%d"),
            "date_max": self.df["date"].max().strftime("%Y-%m-%d"),
            "counts": self.df.groupby("sport").size().to_dict(),
            "default_start": default_start,
            "default_end": default_end,
            "anime": anime,
        }

    # -- media watch-list -------------------------------------------------

    def anime(self, top: int = 25) -> list[dict]:
        if self.anime_rec is None:
            return []
        out = []
        for r in self.anime_rec.recommend(self.anime_user, top=top):
            sc = r.scored
            out.append({
                "rank": r.rank,
                "item_id": sc.item.item_id,
                "sport": "anime",
                "label": sc.item.meta["label"],
                "date": (sc.item.meta.get("type") or "") + (f" · {sc.item.when.year}" if sc.item.when else ""),
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

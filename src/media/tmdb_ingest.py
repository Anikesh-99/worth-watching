"""TMDB ingestion: The Movie Database -> one tidy catalog row per film / show.

The screen vertical's ingest, sibling to anime (Jikan) and books (Open Library).
One class serves both media types — `TMDBIngest("movie")` and
`TMDBIngest("tv")` — because TMDB's discover/genre endpoints are identical bar
the path and a couple of field names. Rows carry content features (genres) plus
a community-quality prior (vote_average); posters come from TMDB's keyless image
CDN, so the demo can hotlink them just like the crests.

Needs a free TMDB API key in the environment as TMDB_API_KEY (see README). Each
page is cached to disk so re-runs are instant.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv optional
    pass

CATALOG_COLUMNS = ["item_id", "tmdb_id", "title", "kind", "year", "rating",
                   "votes", "genres", "image_url"]

_BASE = "https://api.themoviedb.org/3"
_IMG = "https://image.tmdb.org/t/p/w500"          # keyless poster CDN


class TMDBIngest:
    def __init__(self, media_type: str = "movie", cache_dir: str = "data/tmdb_cache") -> None:
        if media_type not in ("movie", "tv"):
            raise ValueError("media_type must be 'movie' or 'tv'")
        self.kind = media_type
        self.key = os.getenv("TMDB_API_KEY")
        self._gmap: dict[int, str] | None = None
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._sess = requests.Session()
        self._sess.headers.update({"User-Agent": "worth-watching/1.0"})

    def _get(self, path: str, params: dict, cache_key: str | None = None) -> dict:
        if not self.key:
            raise RuntimeError("TMDB_API_KEY not set — add a free key to .env (see README).")
        if cache_key:
            f = self.cache / f"{cache_key}.json"
            if f.exists():
                return json.loads(f.read_text())
        data: dict = {}
        for attempt in range(5):
            try:
                r = self._sess.get(f"{_BASE}{path}", params={"api_key": self.key, **params}, timeout=20)
                if r.status_code == 429:                     # rate-limited
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as exc:
                logging.debug("tmdb retry %s (%s): %s", path, attempt, exc)
                time.sleep(1.0 * (attempt + 1))
        if cache_key and data:
            (self.cache / f"{cache_key}.json").write_text(json.dumps(data))
        time.sleep(0.25)
        return data

    def genre_map(self) -> dict[int, str]:
        if self._gmap is None:
            body = self._get(f"/genre/{self.kind}/list", {}, cache_key=f"{self.kind}_genres")
            self._gmap = {g["id"]: g["name"] for g in body.get("genres", [])}
        return self._gmap

    def search(self, title: str, year: int | None = None) -> dict | None:
        """First matching catalog row for a title (for importing external ratings)."""
        params: dict = {"query": title, "language": "en-US", "include_adult": "false"}
        if year:
            params["year" if self.kind == "movie" else "first_air_date_year"] = int(year)
        body = self._get(f"/search/{self.kind}", params)
        for r in body.get("results", []):
            row = self._row(r, self.genre_map())
            if row:
                return row
        return None

    def _row(self, r: dict, genres: dict[int, str]) -> dict | None:
        tid = r.get("id")
        title = r.get("title") or r.get("name")               # movie vs tv field
        if not tid or not title:
            return None
        date = r.get("release_date") or r.get("first_air_date") or ""
        poster = r.get("poster_path")
        return {
            "item_id": f"{self.kind}-{tid}",
            "tmdb_id": tid,
            "title": title,
            "kind": self.kind,
            "year": int(date[:4]) if date[:4].isdigit() else None,
            "rating": r.get("vote_average"),
            "votes": r.get("vote_count") or 0,
            "genres": "|".join(genres.get(g, "") for g in r.get("genre_ids", []) if genres.get(g)),
            "image_url": f"{_IMG}{poster}" if poster else "",
        }

    def build(self, pages: int = 14, min_votes: int = 800) -> pd.DataFrame:
        """Catalog of the best-known titles (by vote count), genre-tagged.

        Sorting by vote_count surfaces the renowned, widely-seen titles — the
        ones a user is most likely to have an opinion about — rather than obscure
        high-scorers. vote_average is kept as the quality prior.
        """
        genres = self.genre_map()
        rows: dict[str, dict] = {}
        for p in range(1, pages + 1):
            body = self._get(f"/discover/{self.kind}", {
                "sort_by": "vote_count.desc", "vote_count.gte": min_votes,
                "page": p, "language": "en-US",
            }, cache_key=f"{self.kind}_disc_{p:03d}")
            for r in body.get("results", []):
                row = self._row(r, genres)
                if row:
                    rows[row["item_id"]] = row
            logging.info("%s catalog: %d titles (through page %d)", self.kind, len(rows), p)
        return pd.DataFrame(list(rows.values()), columns=CATALOG_COLUMNS)

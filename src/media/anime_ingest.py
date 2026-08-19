"""Anime ingestion: Jikan (MyAnimeList) catalog -> one tidy row per title.

The media vertical's answer to the sports ingest. Where sports rows carry
excitement features, anime rows carry *content* features (genres, themes) plus
context (year, community score, popularity). The recommender scores these by
taste match, not excitement — same Recommender interface, different engine.

Jikan is keyless but rate-limited (~3 req/s); each catalog page is cached to
disk so re-runs are instant and polite.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

CATALOG_COLUMNS = [
    "item_id", "mal_id", "title", "type", "year", "score", "members",
    "genres", "themes", "image_url",
]

_TOP = "https://api.jikan.moe/v4/top/anime"


class AnimeIngest:
    def __init__(self, cache_dir: str = "data/anime_cache") -> None:
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._sess = requests.Session()
        self._sess.headers.update({"User-Agent": "worth-watching/1.0"})

    def _page(self, page: int) -> dict:
        f = self.cache / f"top_{page:03d}.json"
        if f.exists():
            return json.loads(f.read_text())
        data: dict = {"data": []}
        for attempt in range(5):
            try:
                r = self._sess.get(_TOP, params={"page": page, "limit": 25}, timeout=20)
                if r.status_code in (429, 500, 503, 504):   # rate-limit / MAL flakiness
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                body = r.json()
                if body.get("data"):
                    data = body
                    break
                time.sleep(1.0 * (attempt + 1))
            except Exception as exc:
                logging.debug("retry page %s (%s): %s", page, attempt, exc)
                time.sleep(1.0 * (attempt + 1))
        # Only cache non-empty pages so transient 504s refetch next run.
        if data.get("data"):
            f.write_text(json.dumps(data))
        time.sleep(0.5)  # ~2 req/s, under Jikan's limit
        return data

    @staticmethod
    def _row(a: dict) -> dict | None:
        if not a.get("mal_id") or not a.get("title"):
            return None
        return {
            "item_id": f"anime-{a['mal_id']}",
            "mal_id": a["mal_id"],
            "title": a["title"],
            "type": a.get("type") or "",
            "year": a.get("year") or (a.get("aired", {}).get("prop", {}).get("from", {}).get("year")),
            "score": a.get("score"),
            "members": a.get("members") or 0,
            "genres": "|".join(g["name"] for g in a.get("genres", [])),
            "themes": "|".join(g["name"] for g in a.get("themes", [])),
            "image_url": a.get("images", {}).get("jpg", {}).get("large_image_url", ""),
        }

    def build(self, pages: int = 24) -> pd.DataFrame:
        """Tidy catalog of the top `pages`*25 anime by community ranking."""
        rows: dict[str, dict] = {}
        for p in range(1, pages + 1):
            page = self._page(p)
            for a in page.get("data", []):
                row = self._row(a)
                if row:
                    rows[row["item_id"]] = row
            logging.info("anime catalog: %d titles (through page %d)", len(rows), p)
        return pd.DataFrame(list(rows.values()), columns=CATALOG_COLUMNS)

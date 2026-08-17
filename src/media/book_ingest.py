"""Book ingestion: Open Library subject endpoints -> one tidy row per book.

The books vertical's catalog. Keyless (Open Library asks only for a
User-Agent). Each subject page returns works with rich subject tags, author,
year, and edition_count (a renown/popularity proxy we use as the quality
prior). Subject tags are noisy (library-metadata like "Accessible book" leaks
in), so we clean and normalize them for content-based matching.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

CATALOG_COLUMNS = ["item_id", "title", "author", "year", "editions", "subjects"]

_SUBJECT_URL = "https://openlibrary.org/subjects/{slug}.json"

# Library/format metadata that isn't a genre — dropped from tags.
_NOISE = {
    "accessible book", "protected daisy", "in library", "large type books",
    "overdrive", "internet archive wishlist", "popular print disabled books",
    "ebook", "reading level", "lending library", "collectible books", "nyt:",
}


def _clean_subjects(raw: list[str], cap: int = 10) -> list[str]:
    out: list[str] = []
    seen = set()
    for s in raw or []:
        t = " ".join(str(s).split()).strip()
        low = t.lower()
        if not t or len(t) > 40 or "," in t or low in _NOISE:
            continue
        if any(low.startswith(n) for n in _NOISE):
            continue
        norm = t.title()
        if norm.lower() not in seen:
            seen.add(norm.lower())
            out.append(norm)
        if len(out) >= cap:
            break
    return out


class BookIngest:
    def __init__(self, cache_dir: str = "data/book_cache") -> None:
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._sess = requests.Session()
        self._sess.headers.update({"User-Agent": "worth-watching/1.0 (github.com/Anikesh-99)"})

    def _subject(self, slug: str, limit: int) -> dict:
        f = self.cache / f"{slug}.json"
        if f.exists():
            return json.loads(f.read_text())
        data: dict = {"works": []}
        for attempt in range(4):
            try:
                r = self._sess.get(_SUBJECT_URL.format(slug=slug), params={"limit": limit}, timeout=20)
                if r.status_code in (429, 503):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                body = r.json()
                if body.get("works"):
                    data = body
                    break
            except Exception as exc:
                logging.debug("retry %s (%s): %s", slug, attempt, exc)
                time.sleep(1.0 * (attempt + 1))
        if data.get("works"):
            f.write_text(json.dumps(data))
        time.sleep(0.4)
        return data

    @staticmethod
    def _row(w: dict) -> dict | None:
        key = w.get("key")
        if not key or not w.get("title"):
            return None
        authors = [a.get("name") for a in w.get("authors", []) if a.get("name")]
        subs = _clean_subjects(w.get("subject", []))
        if not subs:
            return None
        return {
            "item_id": f"book-{key}",
            "title": w["title"],
            "author": authors[0] if authors else "",
            "year": w.get("first_publish_year"),
            "editions": w.get("edition_count") or 1,
            "subjects": "|".join(subs),
        }

    def build(self, subjects: list[str], limit: int = 50) -> pd.DataFrame:
        rows: dict[str, dict] = {}
        for slug in subjects:
            data = self._subject(slug, limit)
            for w in data.get("works", []):
                row = self._row(w)
                if row:
                    rows[row["item_id"]] = row
            logging.info("book catalog: %d titles (through subject '%s')", len(rows), slug)
        return pd.DataFrame(list(rows.values()), columns=CATALOG_COLUMNS)

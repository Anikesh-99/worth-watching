"""MusicBrainz genre enrichment (keyless) — the genres Spotify won't give us.

Spotify's restricted app tier returns no artist genres, so we look them up by
name from MusicBrainz, which has rich genre/tag data for known artists. Rate
limit is ~1 req/sec; every lookup is cached to disk so a re-run is instant.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

_H = {"User-Agent": "worth-watching/1.0 (github.com/Anikesh-99)"}
_CACHE = Path("data/mb_cache")


def _slug(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())[:50] or "x"


def artist_genres(name: str, cap: int = 5) -> list[str]:
    """Top genre tags for an artist by name (cached; [] if unknown)."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    cf = _CACHE / f"{_slug(name)}.json"
    if cf.exists():
        return json.loads(cf.read_text())

    genres: list[str] = []
    try:
        r = requests.get("https://musicbrainz.org/ws/2/artist",
                         params={"query": f'artist:"{name}"', "fmt": "json", "limit": 1},
                         headers=_H, timeout=15)
        arts = r.json().get("artists", []) if r.status_code == 200 else []
        if arts:
            time.sleep(1.1)  # respect MusicBrainz rate limit
            mbid = arts[0]["id"]
            d = requests.get(f"https://musicbrainz.org/ws/2/artist/{mbid}",
                             params={"inc": "genres+tags", "fmt": "json"},
                             headers=_H, timeout=15).json()
            src = d.get("genres") or d.get("tags") or []
            genres = [g["name"].title() for g in sorted(src, key=lambda z: -z.get("count", 0))][:cap]
    except Exception:
        genres = []

    cf.write_text(json.dumps(genres))
    time.sleep(1.1)
    return genres

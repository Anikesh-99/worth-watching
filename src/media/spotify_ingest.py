"""Spotify ingestion: top artists (taste) + new releases (candidates).

Uses only endpoints Spotify still supports for new apps (user-top-read,
browse/new-releases, artists) — NOT the deprecated recommendations/related/
audio-features endpoints. Genres come from artist objects (Spotify's artist
genres are sparse, so some rows carry no tags — a documented limitation).

Auth: reads the token saved by scripts/spotify_auth.py and refreshes it with
SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET from the environment.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()  # load SPOTIFY_* from a local .env if present
except ImportError:
    pass

TOKEN_FILE = Path("data/.spotify_token.json")
CATALOG_COLUMNS = ["item_id", "name", "artist", "year", "popularity", "genres"]
RATING_COLUMNS = ["item_id", "rating", "name", "genres"]


class SpotifyClient:
    def __init__(self) -> None:
        if not TOKEN_FILE.exists():
            raise RuntimeError("No Spotify token. Run scripts/spotify_auth.py first.")
        self._tok = json.loads(TOKEN_FILE.read_text())
        self._sess = requests.Session()

    def _refresh(self) -> None:
        cid, secret = os.environ.get("SPOTIFY_CLIENT_ID"), os.environ.get("SPOTIFY_CLIENT_SECRET")
        basic = base64.b64encode(f"{cid}:{secret}".encode()).decode()
        r = self._sess.post("https://accounts.spotify.com/api/token",
                            data={"grant_type": "refresh_token", "refresh_token": self._tok["refresh_token"]},
                            headers={"Authorization": f"Basic {basic}"}, timeout=15)
        r.raise_for_status()
        self._tok["access_token"] = r.json()["access_token"]
        TOKEN_FILE.write_text(json.dumps(self._tok))

    def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(3):
            r = self._sess.get(f"https://api.spotify.com/v1{path}", params=params,
                               headers={"Authorization": f"Bearer {self._tok['access_token']}"}, timeout=20)
            if r.status_code == 401:
                self._refresh()
                continue
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 2)) + 1)
                continue
            r.raise_for_status()
            return r.json()
        return {}

    # -- taste: your top artists -----------------------------------------

    def top_artists(self, limit: int = 50, time_range: str = "medium_term") -> pd.DataFrame:
        data = self._get("/me/top/artists", {"limit": min(limit, 50), "time_range": time_range})
        rows = []
        items = data.get("items", [])
        n = len(items)
        for i, a in enumerate(items):
            rows.append({
                "item_id": f"music-{a['id']}",
                "rating": round(10 * (1 - i / max(n, 1)), 1),   # rank -> pseudo-rating (top = ~10)
                "name": a["name"],
                "genres": "|".join(a.get("genres", [])),
            })
        return pd.DataFrame(rows, columns=RATING_COLUMNS)

    # -- candidates: discover new artists in your genres --------------------
    #
    # Spotify's restricted tier gives no artist genres and blocks batch /artists,
    # /browse, related-artists and recommendations. What still works: genre
    # artist-search (limit<=10) and /artists/{id}/albums. So we search each of
    # your taste genres for artists, take their recent albums, and tag each album
    # with the genre it was discovered under (known by construction).

    def genre_artists(self, genre: str, limit: int = 10) -> list[dict]:
        try:
            data = self._get("/search", {"q": f'genre:"{genre}"', "type": "artist", "limit": min(limit, 10)})
        except Exception as exc:
            logging.debug("genre search %s failed: %s", genre, exc)
            return []
        return [{"id": a["id"], "name": a["name"], "popularity": a.get("popularity", 50)}
                for a in (data.get("artists") or {}).get("items", [])]

    def artist_recent_albums(self, artist_id: str, limit: int = 3) -> list[dict]:
        try:
            data = self._get(f"/artists/{artist_id}/albums",
                             {"include_groups": "album,single", "limit": min(limit, 10)})
        except Exception as exc:
            logging.debug("albums %s failed: %s", artist_id, exc)
            return []
        return data.get("items", [])

    def discover_candidates(self, seed_genres: list[str], per_genre: int = 10,
                            albums_per_artist: int = 2, exclude_ids: set[str] | None = None) -> pd.DataFrame:
        exclude_ids = exclude_ids or set()
        rows: dict[str, dict] = {}
        for genre in seed_genres:
            for art in self.genre_artists(genre, per_genre):
                if art["id"] in exclude_ids:
                    continue
                for alb in self.artist_recent_albums(art["id"], albums_per_artist):
                    rows[f"music-album-{alb['id']}"] = {
                        "item_id": f"music-album-{alb['id']}",
                        "name": alb["name"],
                        "artist": art["name"],
                        "year": (alb.get("release_date") or "0")[:4],
                        "popularity": art["popularity"],
                        "genres": genre.title(),  # discovered under this genre
                    }
        return pd.DataFrame(list(rows.values()), columns=CATALOG_COLUMNS)

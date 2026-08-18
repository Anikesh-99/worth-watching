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
from pathlib import Path

import pandas as pd
import requests

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

    # -- candidates: new releases ----------------------------------------

    def _artist_genres(self, artist_ids: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for i in range(0, len(artist_ids), 50):
            batch = artist_ids[i:i + 50]
            data = self._get("/artists", {"ids": ",".join(batch)})
            for a in data.get("artists", []) or []:
                if a:
                    out[a["id"]] = a.get("genres", [])
        return out

    def new_releases(self, limit: int = 50) -> pd.DataFrame:
        albums = []
        offset = 0
        while len(albums) < limit:
            data = self._get("/browse/new-releases", {"limit": 50, "offset": offset})
            items = (data.get("albums") or {}).get("items", [])
            if not items:
                break
            albums.extend(items)
            offset += 50
        albums = albums[:limit]

        artist_ids = list({a["artists"][0]["id"] for a in albums if a.get("artists")})
        genres = self._artist_genres(artist_ids)

        rows = []
        for a in albums:
            if not a.get("artists"):
                continue
            aid = a["artists"][0]["id"]
            rows.append({
                "item_id": f"music-album-{a['id']}",
                "name": a["name"],
                "artist": a["artists"][0]["name"],
                "year": (a.get("release_date") or "0")[:4],
                "popularity": a.get("popularity", 50),
                "genres": "|".join(genres.get(aid, [])),
            })
        return pd.DataFrame(rows, columns=CATALOG_COLUMNS)

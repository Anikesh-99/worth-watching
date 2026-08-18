"""Probe which Spotify endpoints your app can actually reach.

Spotify restricted many endpoints for new development-mode apps, and which ones
vary. This hits several candidate sources with your token and prints the status
of each, so we build the ingest on endpoints that work rather than guessing.

Run after scripts/spotify_auth.py:
    python scripts/spotify_probe.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TOKEN_FILE = Path("data/.spotify_token.json")


def _refresh(tok: dict) -> None:
    cid, secret = os.environ.get("SPOTIFY_CLIENT_ID"), os.environ.get("SPOTIFY_CLIENT_SECRET")
    basic = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    r = requests.post("https://accounts.spotify.com/api/token",
                      data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"]},
                      headers={"Authorization": f"Basic {basic}"}, timeout=15)
    r.raise_for_status()
    tok["access_token"] = r.json()["access_token"]
    TOKEN_FILE.write_text(json.dumps(tok))


def main() -> None:
    if not TOKEN_FILE.exists():
        sys.exit("No token. Run scripts/spotify_auth.py first.")
    tok = json.loads(TOKEN_FILE.read_text())
    _refresh(tok)  # ensure fresh
    hdr = {"Authorization": f"Bearer {tok['access_token']}"}

    # get a top-artist id to test artist-albums
    top = requests.get("https://api.spotify.com/v1/me/top/artists",
                       params={"limit": 1}, headers=hdr, timeout=15)
    artist_id = None
    if top.status_code == 200 and top.json().get("items"):
        a = top.json()["items"][0]
        artist_id = a["id"]
        print(f"top artist sample: {a['name']} (id {artist_id}, genres {a.get('genres')})\n")

    probes = [
        ("me/top/artists", "/me/top/artists", {"limit": 5}),
        ("me/top/tracks", "/me/top/tracks", {"limit": 5}),
        ("search plain text (album)", "/search", {"q": "Radiohead", "type": "album", "limit": 5}),
        ("search tag:new (album)", "/search", {"q": "tag:new", "type": "album", "limit": 5}),
        ("search genre filter (artist)", "/search", {"q": 'genre:"indie rock"', "type": "artist", "limit": 5}),
        ("search year filter (album)", "/search", {"q": "year:2025", "type": "album", "limit": 5}),
        ("browse/new-releases", "/browse/new-releases", {"limit": 5}),
    ]
    if artist_id:
        probes.append((f"artists/{{id}}/albums", f"/artists/{artist_id}/albums",
                       {"include_groups": "album,single", "limit": 5}))

    print(f"{'endpoint':32} {'status':>6}  note")
    for label, path, params in probes:
        try:
            r = requests.get(f"https://api.spotify.com/v1{path}", params=params, headers=hdr, timeout=15)
            note = ""
            if r.status_code == 200:
                d = r.json()
                if "items" in d:
                    note = f"{len(d['items'])} items"
                elif "albums" in d and isinstance(d["albums"], dict):
                    note = f"{len(d['albums'].get('items', []))} albums"
                elif "artists" in d and isinstance(d["artists"], dict):
                    note = f"{len(d['artists'].get('items', []))} artists"
            else:
                note = (r.text or "")[:80].replace("\n", " ")
            print(f"{label:32} {r.status_code:>6}  {note}")
        except Exception as exc:
            print(f"{label:32} {'ERR':>6}  {exc}")


if __name__ == "__main__":
    main()

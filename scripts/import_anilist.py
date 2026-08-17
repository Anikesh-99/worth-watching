"""Import your scored anime list from AniList (public API, no auth).

AniList exposes public lists over GraphQL, so no file export is needed — just
your username (list visibility must be public, the default). Each scored entry
brings its genres + tags, which become the taste signal (works even for titles
outside our catalog).

Usage:
    python scripts/import_anilist.py <your_anilist_username>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

_QUERY = """
query($name: String) {
  MediaListCollection(userName: $name, type: ANIME) {
    lists { entries {
      score(format: POINT_100)
      media { idMal title { romaji english }
        genres tags { name isMediaSpoiler } }
    } }
  }
}
"""


def main(username: str) -> None:
    r = requests.post(
        "https://graphql.anilist.co",
        json={"query": _QUERY, "variables": {"name": username}},
        headers={"User-Agent": "worth-watching/1.0", "Content-Type": "application/json"},
        timeout=20,
    )
    if r.status_code == 404 or (r.status_code == 200 and r.json().get("errors")):
        sys.exit(f"AniList user '{username}' not found or list is private.")
    r.raise_for_status()
    lists = r.json()["data"]["MediaListCollection"]["lists"]

    rows = {}
    for lst in lists:
        for e in lst["entries"]:
            score = e.get("score") or 0
            m = e["media"]
            if score <= 0 or not m.get("idMal"):
                continue  # skip unscored / no MAL mapping
            title = (m["title"].get("english") or m["title"].get("romaji") or "").strip()
            themes = [t["name"] for t in (m.get("tags") or []) if not t.get("isMediaSpoiler")][:6]
            rows[m["idMal"]] = {
                "item_id": f"anime-{m['idMal']}",
                "rating": round(score / 10, 1),          # POINT_100 -> 1..10
                "title": title,
                "genres": "|".join(m.get("genres") or []),
                "themes": "|".join(themes),
            }

    if not rows:
        sys.exit("No scored anime found. Add titles you've watched and score them on AniList first.")

    df = pd.DataFrame(rows.values())
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/my_anime_ratings.csv", index=False)
    print(f"Imported {len(df)} scored anime from AniList -> data/my_anime_ratings.csv")
    print(df["rating"].round().astype(int).value_counts().sort_index().to_string())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/import_anilist.py <anilist_username>")
    main(sys.argv[1])

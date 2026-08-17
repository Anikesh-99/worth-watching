"""Import your real ratings from a Goodreads CSV export.

On Goodreads: Account -> My Books -> Import/Export -> Export Library (CSV).
Point this at it. Goodreads' CSV has your rating but no genres, so each rated
book is enriched with subject tags from Open Library (keyless, cached), giving
the content-based recommender a taste signal.

Usage:
    python scripts/import_goodreads.py ~/Downloads/goodreads_library_export.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.media.book_ingest import _clean_subjects  # noqa: E402

CACHE = Path("data/book_cache/enrich")
_SESS = requests.Session()
_SESS.headers.update({"User-Agent": "worth-watching/1.0 (github.com/Anikesh-99)"})


def _get(url: str, params: dict | None = None) -> dict | None:
    try:
        r = _SESS.get(url, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def _enrich(title: str, author: str, isbn: str) -> tuple[str, list[str]]:
    """Return (item_id, subjects) for a book via Open Library, cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key_hash = "".join(c for c in f"{isbn}-{title}"[:60] if c.isalnum())
    cf = CACHE / f"{key_hash}.json"
    if cf.exists():
        d = json.loads(cf.read_text())
        return d["item_id"], d["subjects"]

    work_key, subjects = "", []
    # 1) ISBN -> edition -> work
    if isbn:
        ed = _get(f"https://openlibrary.org/isbn/{isbn}.json")
        if ed and ed.get("works"):
            work_key = ed["works"][0].get("key", "")
    # 2) fallback: search by title + author
    if not work_key:
        res = _get("https://openlibrary.org/search.json",
                   {"title": title, "author": author, "fields": "key,subject", "limit": 1})
        docs = (res or {}).get("docs") or []
        if docs:
            work_key = docs[0].get("key", "")
            subjects = _clean_subjects(docs[0].get("subject", []))
    # 3) fetch work subjects if we have a key but no subjects yet
    if work_key and not subjects:
        w = _get(f"https://openlibrary.org{work_key}.json")
        subjects = _clean_subjects((w or {}).get("subjects", []))

    item_id = f"book-{work_key}" if work_key else f"book-gr-{key_hash}"
    cf.write_text(json.dumps({"item_id": item_id, "subjects": subjects}))
    time.sleep(0.25)
    return item_id, subjects


def main(path: str) -> None:
    p = Path(path)
    if not p.exists():
        sys.exit(f"file not found: {path}")
    gr = pd.read_csv(p)
    gr = gr[pd.to_numeric(gr.get("My Rating"), errors="coerce").fillna(0) > 0].copy()
    if gr.empty:
        sys.exit("No rated books (My Rating > 0) found in the export.")

    print(f"Enriching {len(gr)} rated books with subjects from Open Library…")
    rows = []
    for i, (_, b) in enumerate(gr.iterrows(), 1):
        isbn = str(b.get("ISBN13") or b.get("ISBN") or "").strip().strip('="')
        item_id, subjects = _enrich(str(b["Title"]), str(b.get("Author", "")), isbn)
        rows.append({"item_id": item_id, "rating": int(b["My Rating"]),
                     "title": b["Title"], "subjects": "|".join(subjects)})
        if i % 20 == 0:
            print(f"  {i}/{len(gr)}…")

    out = pd.DataFrame(rows)
    Path("data").mkdir(exist_ok=True)
    out.to_csv("data/my_books_ratings.csv", index=False)
    n_tagged = (out["subjects"].str.len() > 0).sum()
    print(f"Imported {len(out)} books -> data/my_books_ratings.csv ({n_tagged} enriched with subjects)")
    print(out["rating"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/import_goodreads.py <goodreads_library_export.csv>")
    main(sys.argv[1])

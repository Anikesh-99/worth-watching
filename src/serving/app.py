"""FastAPI app serving the watch-list API and the dashboard.

Run:
    uvicorn src.serving.app:app --reload
    # then open http://127.0.0.1:8000
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from src.serving.service import WatchlistService

app = FastAPI(title="Worth Watching?", docs_url="/api/docs")

_INDEX = Path(__file__).resolve().parents[2] / "web" / "index.html"


@lru_cache(maxsize=1)
def _service() -> WatchlistService:
    return WatchlistService()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX.read_text()


@app.get("/api/meta")
def meta() -> dict:
    return _service().meta()


@app.get("/api/watchlist")
def watchlist(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    sport: str = Query("all", pattern="^(all|f1|nba)$"),
    top: int = Query(25, ge=1, le=100),
) -> dict:
    svc = _service()
    return {"items": svc.watchlist(start, end, sport, top),
            "weights": svc.meta()["weights"]}


@app.get("/api/media")
def media(
    vertical: str = Query(..., pattern="^(anime|book|music)$"),
    top: int = Query(25, ge=1, le=100),
) -> dict:
    svc = _service()
    return {"items": svc.media_recs(vertical, top), "meta": svc.meta()["media"].get(vertical)}

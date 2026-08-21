"""Build a tidy per-event dataset for any sport vertical from its config.

Usage:
    python scripts/build_dataset.py configs/f1.yaml
    python scripts/build_dataset.py configs/nba.yaml

Dispatches on the config's `vertical` field, so adding a sport is a new
config + ingest module, not a rewrite. Each run writes data/<vertical>_events.csv.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.media.anime_ingest import AnimeIngest  # noqa: E402
from src.media.book_ingest import BookIngest  # noqa: E402
from src.media.tmdb_ingest import TMDBIngest  # noqa: E402
from src.sports.ingest import F1Ingest  # noqa: E402
from src.sports.nba_ingest import NBAIngest  # noqa: E402
from src.sports.soccer_ingest import SoccerIngest  # noqa: E402


def _build_f1(cfg: dict):
    years = sorted({*cfg["seasons"]["train"], *cfg["seasons"]["test"]})
    df = F1Ingest(cache_dir=cfg.get("cache_dir", "data/f1cache")).build(years)
    chaos_cols = ["item_id", "event_name", "n_dnf", "winner_margin_s", "total_positions_moved"]
    return df, "total_positions_moved", chaos_cols


def _build_nba(cfg: dict):
    seasons = sorted({*cfg["seasons"]["train"], *cfg["seasons"]["test"]})
    df = NBAIngest(cache_dir=cfg.get("cache_dir", "data/nba_cache")).build(seasons)
    chaos_cols = ["item_id", "away", "home", "final_margin", "overtime_periods", "lead_changes"]
    return df, "lead_changes", chaos_cols


def _build_anime(cfg: dict):
    df = AnimeIngest(cache_dir=cfg.get("cache_dir", "data/anime_cache")).build(
        pages=int(cfg.get("catalog_pages", 24)))
    return df, "score", ["item_id", "title", "type", "year", "score", "genres"]


def _build_book(cfg: dict):
    df = BookIngest(cache_dir=cfg.get("cache_dir", "data/book_cache")).build(
        subjects=cfg["subjects"], limit=int(cfg.get("limit_per_subject", 50)))
    return df, "editions", ["item_id", "title", "author", "year", "editions", "subjects"]


def _build_screen(cfg: dict):
    df = TMDBIngest(media_type=cfg["vertical"], cache_dir=cfg.get("cache_dir", "data/tmdb_cache")).build(
        pages=int(cfg.get("catalog_pages", 14)), min_votes=int(cfg.get("min_votes", 800)))
    return df, "votes", ["item_id", "title", "kind", "year", "rating", "votes", "genres"]


def _build_soccer(cfg: dict):
    df = SoccerIngest(cache_dir=cfg.get("cache_dir", "data/soccer_cache")).build(
        leagues=cfg["leagues"], seasons=cfg["seasons"])
    return df, "total_goals", ["item_id", "league", "away", "home", "away_score",
                               "home_score", "total_goals", "red_cards", "came_from_behind"]


BUILDERS = {"f1": _build_f1, "nba": _build_nba, "anime": _build_anime,
            "book": _build_book, "soccer": _build_soccer,
            "movie": _build_screen, "tv": _build_screen}

# sports datasets are seasonal events; media is a catalog. Name the output and
# the summary accordingly.
_OUTPUT = {"f1": "f1_events.csv", "nba": "nba_events.csv", "soccer": "soccer_events.csv",
           "anime": "anime_catalog.csv", "book": "books_catalog.csv",
           "movie": "movies_catalog.csv", "tv": "tv_catalog.csv"}


def main(config_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = yaml.safe_load(Path(config_path).read_text())
    vertical = cfg["vertical"]

    print(f"Building '{vertical}' dataset from {config_path}")
    df, sort_key, sample_cols = BUILDERS[vertical](cfg)

    Path("data").mkdir(exist_ok=True)
    out = Path("data") / _OUTPUT[vertical]
    df.to_csv(out, index=False)

    print(f"\nWrote {len(df)} rows -> {out}")
    if "season" in df.columns:
        print("\nPer-season counts:")
        print(df.groupby("season").size().to_string())
    print("\nSample (top by '%s'):" % sort_key)
    print(df.sort_values(sort_key, ascending=False)[sample_cols].head(6).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/build_dataset.py <configs/*.yaml>")
    main(sys.argv[1])

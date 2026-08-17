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
from src.sports.ingest import F1Ingest  # noqa: E402
from src.sports.nba_ingest import NBAIngest  # noqa: E402


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


BUILDERS = {"f1": _build_f1, "nba": _build_nba}


def main(config_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = yaml.safe_load(Path(config_path).read_text())
    vertical = cfg["vertical"]

    print(f"Building '{vertical}' dataset from {config_path}")
    df, sort_key, sample_cols = BUILDERS[vertical](cfg)

    Path("data").mkdir(exist_ok=True)
    out = Path("data") / f"{vertical}_events.csv"
    df.to_csv(out, index=False)

    print(f"\nWrote {len(df)} events -> {out}")
    print("\nPer-season counts:")
    print(df.groupby("season").size().to_string())
    print("\nSample (highest-excitement by '%s'):" % sort_key)
    print(df.sort_values(sort_key, ascending=False)[sample_cols].head(6).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/build_dataset.py <configs/*.yaml>")
    main(sys.argv[1])

"""F1 ingestion: FastF1 race results -> one tidy row per race.

Each row is a candidate item for the recommender, carrying the objective
"excitement" features whose labels come free from the box score (margins,
overtakes-by-proxy, DNFs). Kept deliberately results-only for speed and
reliability; lap/telemetry-derived features (true on-track lead changes)
are a documented later enhancement, not needed to train a first model.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)

import fastf1  # noqa: E402  (import after quieting its logger)


# Columns of the tidy per-race dataframe every downstream stage relies on.
RACE_COLUMNS = [
    "item_id", "season", "round", "event_name", "country", "date",
    "n_classified", "n_dnf", "winner_margin_s", "total_positions_moved",
    "max_gain", "podium_from_outside_top5", "winner_grid",
]


class F1Ingest:
    """Builds the tidy per-race dataframe from FastF1.

    Parameters
    ----------
    cache_dir:
        Directory FastF1 uses to cache downloaded session data. Created if
        missing so first run works out of the box.
    """

    def __init__(self, cache_dir: str = "data/f1cache") -> None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)

    # -- feature extraction for a single race -----------------------------

    def _race_row(self, year: int, rnd: int) -> dict | None:
        """Return one tidy feature row for a race, or None if it can't load."""
        try:
            s = fastf1.get_session(year, rnd, "R")
            s.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception as exc:  # a future/cancelled/unavailable round
            logging.debug("skip %s R%s: %s", year, rnd, exc)
            return None

        r = s.results
        if r is None or r.empty:
            return None

        grid = pd.to_numeric(r["GridPosition"], errors="coerce")
        pos = pd.to_numeric(r["Position"], errors="coerce")
        status = r["Status"].astype(str)

        # A driver is "classified" when they took the flag on the lead lap
        # ("Finished") or a lap down ("Lapped" / "+N Lap"); everything else
        # ("Retired", "Accident", "Engine", "Collision", ...) is a DNF.
        classified = status.str.contains(
            r"Finished|Lapped|\+\d+ Lap", regex=True, na=False
        )
        n_dnf = int((~classified).sum())

        # Winner margin: P2's Time is already the gap to P1 in FastF1.
        p2 = r.loc[pos == 2, "Time"]
        winner_margin_s = (
            float(p2.iloc[0].total_seconds()) if len(p2) and pd.notna(p2.iloc[0]) else float("nan")
        )

        moved = (grid - pos).abs()
        gain = grid - pos  # positive = climbed the field
        podium = r.loc[pos <= 3]
        podium_grid = pd.to_numeric(podium["GridPosition"], errors="coerce")
        winner_grid = grid[pos == 1]

        ev = s.event
        return {
            "item_id": f"f1-{year}-{int(rnd):02d}",
            "season": year,
            "round": int(rnd),
            "event_name": str(ev.get("EventName", f"Round {rnd}")),
            "country": str(ev.get("Country", "")),
            "date": pd.to_datetime(ev.get("EventDate")).to_pydatetime(),
            "n_classified": int(classified.sum()),
            "n_dnf": n_dnf,
            "winner_margin_s": winner_margin_s,
            "total_positions_moved": int(moved.sum(skipna=True)),
            "max_gain": int(gain.max(skipna=True)) if gain.notna().any() else 0,
            "podium_from_outside_top5": int((podium_grid > 5).sum()),
            "winner_grid": int(winner_grid.iloc[0]) if len(winner_grid) else -1,
        }

    # -- season / multi-season assembly -----------------------------------

    def build_season(self, year: int, max_rounds: int = 24) -> pd.DataFrame:
        """Tidy dataframe of every completed race in one season."""
        rows = []
        for rnd in range(1, max_rounds + 1):
            row = self._race_row(year, rnd)
            if row is None:
                # No more races this season (future/unheld rounds) -> stop early.
                if rnd > 1:
                    break
                continue
            rows.append(row)
            logging.info("ingested %s", row["item_id"])
        return pd.DataFrame(rows, columns=RACE_COLUMNS)

    def build(self, years: list[int]) -> pd.DataFrame:
        """Tidy dataframe across several seasons, sorted chronologically."""
        frames = [self.build_season(y) for y in years]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RACE_COLUMNS)
        return df.sort_values(["season", "round"]).reset_index(drop=True)

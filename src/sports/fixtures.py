"""Build upcoming fixtures with pre-game watchability features.

Turns a schedule of not-yet-played events into the fixtures table the
UpcomingRecommender consumes: WATCH_FEATURES (stakes/pedigree/competitiveness/
form) + meta (item_id, sport, label, date, entities). Pedigree/competitiveness
come from HISTORY (all prior events), which for a future fixture is trivially
lookahead-free.

F1 schedule is fetched best-effort from Jolpica (Ergast successor). NBA/soccer
upcoming arrive when those seasons are live (soccer lands with its vertical);
this module also accepts a caller-supplied fixture list so it works year-round.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import requests

from src.sports.watchability import WATCH_FEATURES

FIXTURE_COLUMNS = ["item_id", "sport", "label", "date", "entities", *WATCH_FEATURES]


def _circuit_pedigree(hist_f1: pd.DataFrame, excitement: dict[str, float]) -> dict[str, float]:
    """Mean historical excitement per country (venue), as the F1 pedigree signal."""
    df = hist_f1.copy()
    df["ex"] = df["item_id"].map(excitement)
    return df.groupby("country")["ex"].mean().to_dict()


def f1_fixtures(races: list[dict], hist_f1: pd.DataFrame, excitement: dict[str, float]) -> pd.DataFrame:
    """`races`: dicts with season, round, event_name, country, date, season_len."""
    pedigree = _circuit_pedigree(hist_f1, excitement)
    rows = []
    for r in races:
        rows.append({
            "item_id": f"f1-{r['season']}-{int(r['round']):02d}-up",
            "sport": "f1",
            "label": r["event_name"],
            "date": pd.to_datetime(r["date"]),
            "entities": [],                                   # driver-level not modelled yet
            "stakes": min(1.0, r["round"] / max(1, r.get("season_len", 24))),
            "pedigree": float(pedigree.get(r["country"], 0.5)),
            "competitiveness": 0.5,                           # needs qualifying grid (future)
            "form": 0.5,
        })
    return pd.DataFrame(rows, columns=FIXTURE_COLUMNS)


def fetch_f1_schedule(after: datetime | None = None) -> list[dict]:
    """Upcoming races of the current F1 season from Jolpica (best-effort)."""
    after = after or datetime.now()
    try:
        r = requests.get("https://api.jolpi.ca/ergast/f1/current.json",
                         headers={"User-Agent": "worth-watching/1.0"}, timeout=15)
        r.raise_for_status()
        races = r.json()["MRData"]["RaceTable"]["Races"]
    except Exception as exc:
        logging.warning("F1 schedule fetch failed: %s", exc)
        return []
    n = len(races)
    out = []
    for rc in races:
        d = rc.get("date")
        if d and pd.to_datetime(d) >= pd.Timestamp(after.date()):
            out.append({"season": int(rc["season"]), "round": int(rc["round"]),
                        "event_name": rc["raceName"],
                        "country": rc["Circuit"]["Location"]["country"],
                        "date": d, "season_len": n})
    return out

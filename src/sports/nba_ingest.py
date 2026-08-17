"""NBA ingestion: ESPN (keyless) scoreboard -> one tidy row per game.

Mirrors the F1 ingest's contract — one feature-rich row per event whose
excitement labels come free from the box score. NBA excitement lives in the
*flow* of a game, so the features are derived from per-quarter linescores the
scoreboard returns reliably: final margin, overtime, lead changes across
period boundaries, and whether the winner had to come from behind.

Data source: site.api.espn.com scoreboard, iterated by date. Each day's raw
JSON is cached to disk so the build is reproducible and re-runs are instant.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

GAME_COLUMNS = [
    "item_id", "season", "date", "away", "home", "away_score", "home_score",
    "final_margin", "overtime_periods", "lead_changes", "winner_came_from_behind",
    "max_abs_lead", "is_playoff",
]

_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"


class NBAIngest:
    def __init__(self, cache_dir: str = "data/nba_cache") -> None:
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._sess = requests.Session()
        self._sess.headers.update({"User-Agent": "Mozilla/5.0"})

    # -- fetching (cached per day) ----------------------------------------

    def _scoreboard(self, day: date) -> dict:
        """Return the raw ESPN scoreboard JSON for one day, disk-cached."""
        key = day.strftime("%Y%m%d")
        f = self.cache / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())
        data: dict = {"events": []}
        for attempt in range(3):
            try:
                resp = self._sess.get(_SCOREBOARD, params={"dates": key, "limit": 100}, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:  # transient network / rate limit
                logging.debug("retry %s (%s): %s", key, attempt, exc)
                time.sleep(0.5 * (attempt + 1))
        f.write_text(json.dumps(data))
        time.sleep(0.05)  # be polite to ESPN
        return data

    # -- feature extraction for a single game -----------------------------

    @staticmethod
    def _game_row(event: dict, season: int) -> dict | None:
        comp = event["competitions"][0]
        if comp.get("status", {}).get("type", {}).get("state") != "post":
            return None  # not a completed game

        teams = {c["homeAway"]: c for c in comp["competitors"]}
        home, away = teams.get("home"), teams.get("away")
        if not home or not away:
            return None

        def line(c: dict) -> list[int]:
            out = []
            for q in c.get("linescores", []) or []:
                try:
                    out.append(int(float(q.get("value", q.get("displayValue", 0)))))
                except (TypeError, ValueError):
                    out.append(0)
            return out

        hl, al = line(home), line(away)
        n_periods = max(len(hl), len(al))
        if n_periods < 4:
            return None  # malformed / in-progress record

        hs, as_ = int(home["score"]), int(away["score"])

        # Cumulative score after each period -> lead sign sequence.
        h_cum = a_cum = 0
        lead = 0
        lead_changes = 0
        max_abs_lead = 0
        lead_after_q3 = 0
        for i in range(n_periods):
            h_cum += hl[i] if i < len(hl) else 0
            a_cum += al[i] if i < len(al) else 0
            diff = h_cum - a_cum
            max_abs_lead = max(max_abs_lead, abs(diff))
            sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
            if sign != 0 and sign != lead:
                if lead != 0:
                    lead_changes += 1
                lead = sign
            if i == 2:  # end of regulation Q3
                lead_after_q3 = sign

        final_sign = 1 if hs > as_ else -1
        came_from_behind = int(lead_after_q3 != 0 and lead_after_q3 != final_sign)

        stype = event.get("season", {}).get("type") or comp.get("type", {}).get("id")
        is_playoff = int(str(stype) == "3")

        gid = event.get("id", "")
        return {
            "item_id": f"nba-{gid}",
            "season": season,
            "date": pd.to_datetime(event.get("date")).to_pydatetime(),
            "away": away["team"]["abbreviation"],
            "home": home["team"]["abbreviation"],
            "away_score": as_,
            "home_score": hs,
            "final_margin": abs(hs - as_),
            "overtime_periods": max(0, n_periods - 4),
            "lead_changes": lead_changes,
            "winner_came_from_behind": came_from_behind,
            "max_abs_lead": max_abs_lead,
            "is_playoff": is_playoff,
        }

    # -- season / multi-season assembly -----------------------------------

    @staticmethod
    def _season_dates(season: int) -> list[date]:
        """Calendar dates for an NBA season labelled by its Finals year.

        Season `2024` == the 2023-24 season: Oct 1 2023 -> Jun 30 2024.
        """
        start = date(season - 1, 10, 1)
        end = date(season, 6, 30)
        days = (end - start).days
        return [start + timedelta(d) for d in range(days + 1)]

    def build_season(self, season: int) -> pd.DataFrame:
        rows: dict[str, dict] = {}
        for day in self._season_dates(season):
            for ev in self._scoreboard(day).get("events", []) or []:
                try:
                    row = self._game_row(ev, season)
                except Exception as exc:
                    logging.debug("skip game: %s", exc)
                    row = None
                if row:
                    rows[row["item_id"]] = row  # dedup by game id
            if day.day == 1:
                logging.info("nba %s .. %s (%d games so far)", season, day, len(rows))
        return pd.DataFrame(list(rows.values()), columns=GAME_COLUMNS)

    def build(self, seasons: list[int]) -> pd.DataFrame:
        frames = [self.build_season(s) for s in seasons]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GAME_COLUMNS)
        return df.sort_values(["season", "date"]).reset_index(drop=True)

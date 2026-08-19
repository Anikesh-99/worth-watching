"""Soccer ingestion: ESPN (keyless) scoreboard -> one tidy row per match.

Covers Premier League (`eng.1`) and Champions League (`uefa.champions`), same
shape as the NBA ingest. Excitement features come from the score plus the goal/
card `details` timeline the scoreboard already carries: margin, total goals,
red cards, lead changes, come-from-behind, late drama. Team crest logos are
captured for the UI's matchup tiles.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

MATCH_COLUMNS = [
    "item_id", "league", "season", "date", "away", "home", "away_score", "home_score",
    "away_logo", "home_logo", "final_margin", "total_goals", "red_cards",
    "lead_changes", "came_from_behind", "late_drama", "is_knockout",
]

_SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"


def _minute(clock: str | None) -> int:
    m = re.match(r"(\d+)", clock or "")
    return int(m.group(1)) if m else 0


class SoccerIngest:
    def __init__(self, cache_dir: str = "data/soccer_cache") -> None:
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._sess = requests.Session()
        self._sess.headers.update({"User-Agent": "Mozilla/5.0"})

    def _scoreboard(self, league: str, day: date) -> dict:
        key = f"{league.replace('.', '_')}_{day:%Y%m%d}"
        f = self.cache / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())
        data: dict = {"events": []}
        for attempt in range(3):
            try:
                r = self._sess.get(_SB.format(league=league),
                                   params={"dates": day.strftime("%Y%m%d")}, timeout=15)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as exc:
                logging.debug("retry %s %s: %s", league, key, exc)
                time.sleep(0.5 * (attempt + 1))
        f.write_text(json.dumps(data))
        time.sleep(0.05)
        return data

    @staticmethod
    def _match_row(event: dict, league: str, season: int, day: date) -> dict | None:
        comp = event["competitions"][0]
        if comp.get("status", {}).get("type", {}).get("state") != "post":
            return None
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        home, away = teams.get("home"), teams.get("away")
        if not home or not away:
            return None
        try:
            hs, as_ = int(home["score"]), int(away["score"])
        except (TypeError, ValueError):
            return None

        hid, aid = home["team"]["id"], away["team"]["id"]
        goals, red_cards, late = [], 0, 0
        for d in comp.get("details", []) or []:
            if d.get("redCard"):
                red_cards += 1
            if d.get("scoringPlay"):
                mnt = _minute(d.get("clock", {}).get("displayValue"))
                goals.append((mnt, d.get("team", {}).get("id"), d.get("scoreValue", 1)))
                if mnt >= 80:
                    late = 1

        # replay goals in minute order for lead changes + come-from-behind
        run = {hid: 0, aid: 0}
        lead = lead_changes = 0
        home_behind = away_behind = False
        for _, tid, val in sorted(goals, key=lambda g: g[0]):
            if tid in run:
                run[tid] += val
            diff = run[hid] - run[aid]
            if diff < 0:
                home_behind = True
            if diff > 0:
                away_behind = True
            sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
            if sign != 0 and sign != lead:
                if lead != 0:
                    lead_changes += 1
                lead = sign

        if hs > as_:
            cfb = int(home_behind)
        elif as_ > hs:
            cfb = int(away_behind)
        else:
            cfb = int(home_behind or away_behind)   # someone equalized

        # CL knockout stage runs Feb-Jun; PL is league play.
        is_knockout = int(league == "uefa.champions" and day.month in (2, 3, 4, 5, 6))

        return {
            "item_id": f"soccer-{event.get('id','')}",
            "league": league, "season": season, "date": pd.to_datetime(event.get("date")).to_pydatetime(),
            "away": away["team"]["abbreviation"], "home": home["team"]["abbreviation"],
            "away_score": as_, "home_score": hs,
            "away_logo": away["team"].get("logo", ""), "home_logo": home["team"].get("logo", ""),
            "final_margin": abs(hs - as_), "total_goals": hs + as_, "red_cards": red_cards,
            "lead_changes": lead_changes, "came_from_behind": cfb,
            "late_drama": late, "is_knockout": is_knockout,
        }

    @staticmethod
    def _season_dates(season: int) -> list[date]:
        start, end = date(season - 1, 8, 1), date(season, 6, 30)   # Aug -> Jun
        return [start + timedelta(d) for d in range((end - start).days + 1)]

    def build(self, leagues: list[str], seasons: list[int]) -> pd.DataFrame:
        rows: dict[str, dict] = {}
        for league in leagues:
            for season in seasons:
                for day in self._season_dates(season):
                    for ev in self._scoreboard(league, day).get("events", []) or []:
                        try:
                            row = self._match_row(ev, league, season, day)
                        except Exception as exc:
                            logging.debug("skip match: %s", exc)
                            row = None
                        if row:
                            rows[row["item_id"]] = row
                    if day.day == 1:
                        logging.info("%s %s .. %s (%d matches)", league, season, day, len(rows))
        return pd.DataFrame(list(rows.values()), columns=MATCH_COLUMNS)

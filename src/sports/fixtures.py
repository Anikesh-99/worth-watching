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


def _team_pedigree(hist_soccer: pd.DataFrame, excitement: dict[str, float]) -> dict[str, float]:
    """Mean historical excitement per club, as the soccer pedigree signal.

    Each match contributes its excitement to BOTH clubs, then we average per
    club. A future fixture's pedigree is the mean of its two clubs' values —
    trivially lookahead-free (it's all prior matches).
    """
    df = hist_soccer.copy()
    df["ex"] = df["item_id"].map(excitement)
    long = pd.concat([
        df[["home", "ex"]].rename(columns={"home": "team"}),
        df[["away", "ex"]].rename(columns={"away": "team"}),
    ], ignore_index=True)
    return long.groupby("team")["ex"].mean().to_dict()


def soccer_fixtures(matches: list[dict], hist_soccer: pd.DataFrame,
                    excitement: dict[str, float],
                    standings: dict[str, dict] | None = None) -> pd.DataFrame:
    """`matches`: dicts with home, away, date, league, is_knockout (from the fetch).

    `standings` maps league -> {team: {strength, games}} (see standings.py). When
    supplied, competitiveness is the clubs' current table closeness; otherwise it
    stays neutral 0.5.
    """
    from src.sports.standings import competitiveness

    ped = _team_pedigree(hist_soccer, excitement)
    standings = standings or {}
    rows = []
    for m in matches:
        home, away = m["home"], m["away"]
        # average the two clubs' historical pedigree; unseen clubs get a neutral
        # 0.5 (same default as F1's per-circuit pedigree), not skipped.
        pedigree = float((ped.get(home, 0.5) + ped.get(away, 0.5)) / 2)
        table = standings.get(m.get("league", ""), {})
        rows.append({
            "item_id": m["item_id"],
            "sport": "soccer",
            "label": f"{home} v {away}",                      # football: home first (matches normalize_soccer)
            "date": pd.to_datetime(m["date"], utc=True).tz_localize(None),
            "entities": [away, home],                         # followed-club personalization
            # same stakes formula as normalize_soccer -> upcoming & history agree
            "stakes": float(m.get("is_knockout", 0)) * 0.7 + 0.3,
            "pedigree": pedigree,
            "competitiveness": competitiveness(home, away, table, "soccer"),
            "form": 0.5,
        })
    return pd.DataFrame(rows, columns=FIXTURE_COLUMNS)


def fetch_soccer_schedule(after: datetime | None = None, days_ahead: int = 10,
                          leagues: tuple[str, ...] = ("eng.1", "uefa.champions")) -> list[dict]:
    """Upcoming PL + CL fixtures over the next `days_ahead` days from ESPN (best-effort).

    Keeps only not-yet-played matches (`state == "pre"`); returns club abbreviations,
    crest logos, kickoff datetime, and a knockout-stage flag.
    """
    from datetime import timedelta

    after = after or datetime.now()
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})
    sb = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
    seen: dict[str, dict] = {}
    for league in leagues:
        for offset in range(days_ahead + 1):
            day = (after + timedelta(days=offset)).date()
            try:
                r = sess.get(sb.format(league=league),
                             params={"dates": day.strftime("%Y%m%d")}, timeout=15)
                r.raise_for_status()
                events = r.json().get("events", []) or []
            except Exception as exc:
                logging.debug("soccer schedule %s %s: %s", league, day, exc)
                continue
            for ev in events:
                comp = (ev.get("competitions") or [{}])[0]
                if comp.get("status", {}).get("type", {}).get("state") != "pre":
                    continue
                teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
                home, away = teams.get("home"), teams.get("away")
                if not home or not away:
                    continue
                item_id = f"soccer-{ev.get('id', '')}-up"
                seen[item_id] = {
                    "item_id": item_id, "league": league, "date": ev.get("date"),
                    "home": home["team"].get("abbreviation", ""),
                    "away": away["team"].get("abbreviation", ""),
                    "home_logo": home["team"].get("logo", ""),
                    "away_logo": away["team"].get("logo", ""),
                    "is_knockout": int(league == "uefa.champions" and day.month in (2, 3, 4, 5, 6)),
                }
    return list(seen.values())


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


def collect_upcoming_fixtures(f1_events: pd.DataFrame | None = None,
                              soccer_events: pd.DataFrame | None = None
                              ) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Fetch live schedules and build ONE combined fixtures table across sports.

    The single source of truth for "Coming up": F1 races (Jolpica) + PL/CL
    matches this week (ESPN), scored on the same WatchabilityIndex. Each sport
    is best-effort — a failed fetch or missing history drops that sport, never
    the whole list. Returns (fixtures, logos) where logos maps item_id -> crest
    URLs for the soccer tiles. Imports are local to keep this module's top clean
    of core deps.
    """
    from src.core.excitement import ExcitementIndex
    from src.core.features import normalize_f1, normalize_soccer

    parts: list[pd.DataFrame] = []
    logos: dict[str, dict] = {}

    if f1_events is not None and len(f1_events):
        try:
            races = fetch_f1_schedule()
            if races:
                fu = normalize_f1(f1_events)
                ex = dict(zip(fu["item_id"], ExcitementIndex().score(fu)))
                parts.append(f1_fixtures(races, f1_events, ex))
        except Exception as exc:
            logging.warning("upcoming F1 skipped: %s", exc)

    if soccer_events is not None and len(soccer_events):
        try:
            matches = fetch_soccer_schedule()
            if matches:
                from src.sports.standings import fetch_standings
                su = normalize_soccer(soccer_events)
                ex = dict(zip(su["item_id"], ExcitementIndex().score(su)))
                # current table per league (for competitiveness); best-effort
                leagues = {m.get("league", "") for m in matches if m.get("league")}
                standings = {lg: fetch_standings("soccer", lg) for lg in leagues}
                parts.append(soccer_fixtures(matches, soccer_events, ex, standings))
                for m in matches:
                    if m.get("home_logo") or m.get("away_logo"):
                        logos[m["item_id"]] = {"home_logo": m.get("home_logo", ""),
                                               "away_logo": m.get("away_logo", "")}
        except Exception as exc:
            logging.warning("upcoming soccer skipped: %s", exc)

    fixtures = (pd.concat(parts, ignore_index=True) if parts
                else pd.DataFrame(columns=FIXTURE_COLUMNS))
    return fixtures, logos

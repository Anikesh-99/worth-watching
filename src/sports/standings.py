"""Current league standings -> a pre-game competitiveness signal.

For an UPCOMING fixture, "how evenly matched are these two?" is best answered
by how close the sides sit in strength right now. We reduce each team to a
single strength number — points-per-game (soccer) or win-% (NBA) — and turn the
gap between the two into competitiveness (small gap -> close contest -> high).

Standings are fetched best-effort from ESPN (keyless). Early in a season the
table is uninformative (few or zero games played), so we gate on a minimum
games-played and fall back to neutral 0.5 rather than pretending to know. The
daily refresh sharpens the signal as games accrue.
"""

from __future__ import annotations

import logging
import math

import requests

# ESPN standings endpoints per sport. Soccer strength = points-per-game (0..3),
# NBA strength = win fraction (0..1); the decay scale is chosen per range so a
# "typical top-vs-bottom" gap lands near ~0.15.
_ENDPOINT = {
    "soccer": "https://site.api.espn.com/apis/v2/sports/soccer/{league}/standings",
    "nba": "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings",
}
_SCALE = {"soccer": 0.8, "nba": 0.35}
_MIN_GAMES = {"soccer": 3, "nba": 5}


def fetch_standings(sport: str, league: str | None = None) -> dict[str, dict]:
    """Return {team_abbr: {"strength": float, "games": int}} (empty on failure).

    strength is points-per-game for soccer, win fraction for NBA — both "higher
    is stronger", on their own scale (compared only against same-sport teams).
    """
    url = _ENDPOINT.get(sport)
    if not url:
        return {}
    try:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "Mozilla/5.0"})
        r = sess.get(url.format(league=league or ""), timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logging.warning("standings fetch failed (%s/%s): %s", sport, league, exc)
        return {}

    table: dict[str, dict] = {}
    # ESPN nests entries under children[].standings.entries (NBA has two
    # conference children; soccer a single league child).
    groups = data.get("children") or [data]
    for grp in groups:
        for e in (grp.get("standings") or {}).get("entries", []) or []:
            abbr = e.get("team", {}).get("abbreviation")
            if not abbr:
                continue
            stats = {s.get("name"): s.get("value") for s in e.get("stats", [])}
            games = int(stats.get("gamesPlayed") or 0)
            if sport == "soccer":
                pts = float(stats.get("points") or 0.0)
                strength = pts / games if games else 0.0
            else:  # nba
                strength = float(stats.get("winPercent") or 0.0)
            table[abbr] = {"strength": strength, "games": games}
    return table


def competitiveness(home: str, away: str, table: dict[str, dict], sport: str) -> float:
    """Pre-game closeness of two teams from a standings table, in [0, 1].

    Neutral 0.5 when either side is missing or hasn't played enough games — we
    don't fabricate certainty from an empty early-season table.
    """
    h, a = table.get(home), table.get(away)
    if not h or not a:
        return 0.5
    if min(h["games"], a["games"]) < _MIN_GAMES.get(sport, 3):
        return 0.5
    gap = abs(h["strength"] - a["strength"])
    return round(math.exp(-gap / _SCALE.get(sport, 0.8)), 4)

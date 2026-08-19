"""Pre-game features for the predictive excitement model.

The whole point of "upcoming" recommendations is predicting how exciting an
event WILL be, from information available BEFORE it starts. So every feature
here is reconstructed *as of the event date* — standings, form and head-to-head
history use only prior events. This is enforced structurally: a single
chronological forward pass computes each event's features from running state,
then updates that state with the event's result. Nothing from an event (or any
later event) can enter its own feature row.

Target for training is the post-hoc excitement index (what actually happened),
so the model learns pre-game signals -> realized excitement.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

NBA_FEATURES = ["standings_gap", "combined_winpct", "home_form", "away_form",
                "h2h_excitement", "is_playoff"]
F1_FEATURES = ["stakes", "circuit_hist", "season_form"]

_NEUTRAL = 0.5  # default for a team/circuit/pairing with no prior history


def build_nba_pregame(nba: pd.DataFrame, excitement: dict[str, float]) -> pd.DataFrame:
    """One row per game with pre-game features + the realized excitement target.

    `excitement` maps item_id -> post-hoc excitement (the training target).
    Standings/form reset per season; head-to-head accumulates across seasons.
    """
    df = nba.sort_values("date").reset_index(drop=True)
    record: dict = defaultdict(lambda: [0, 0])          # (season, team) -> [wins, games]
    form: dict = defaultdict(lambda: deque(maxlen=5))    # (season, team) -> recent W/L
    h2h: dict = defaultdict(lambda: [0.0, 0])            # frozenset({a,b}) -> [sum_ex, count]

    rows = []
    for _, g in df.iterrows():
        s, home, away = g["season"], g["home"], g["away"]
        hk, ak = (s, home), (s, away)
        hw = record[hk][0] / record[hk][1] if record[hk][1] else _NEUTRAL
        aw = record[ak][0] / record[ak][1] if record[ak][1] else _NEUTRAL
        hf = float(np.mean(form[hk])) if form[hk] else _NEUTRAL
        af = float(np.mean(form[ak])) if form[ak] else _NEUTRAL
        pair = h2h[frozenset((home, away))]
        h2h_ex = pair[0] / pair[1] if pair[1] else _NEUTRAL

        rows.append({
            "item_id": g["item_id"], "sport": "nba", "season": s, "date": g["date"],
            "standings_gap": abs(hw - aw),      # close teams -> competitive
            "combined_winpct": hw + aw,          # two strong teams
            "home_form": hf, "away_form": af,    # both in good form
            "h2h_excitement": h2h_ex,            # this pairing tends to deliver
            "is_playoff": float(g["is_playoff"]),
            "excitement": excitement[g["item_id"]],
        })

        # --- update state AFTER recording features (no lookahead) ---
        home_win = g["home_score"] > g["away_score"]
        record[hk][1] += 1
        record[ak][1] += 1
        record[hk][0] += int(home_win)
        record[ak][0] += int(not home_win)
        form[hk].append(int(home_win))
        form[ak].append(int(not home_win))
        e = excitement[g["item_id"]]
        pair[0] += e
        pair[1] += 1

    return pd.DataFrame(rows)


def build_f1_pregame(f1: pd.DataFrame, excitement: dict[str, float]) -> pd.DataFrame:
    """One row per race with pre-game features + realized excitement target.

    From race-level data we can reconstruct: stakes (round in the calendar),
    circuit history (how exciting this venue has been), and season form (recent
    races). Driver-championship tightness needs per-driver standings and is a
    documented FastF1 enrichment, not available from the race-aggregate table.
    """
    df = f1.sort_values(["season", "round"]).reset_index(drop=True)
    season_len = df.groupby("season")["round"].max().to_dict()   # calendar known in advance
    circuit: dict = defaultdict(lambda: [0.0, 0])                 # country -> [sum_ex, count]
    season_recent: dict = defaultdict(lambda: deque(maxlen=3))    # season -> recent excitement

    rows = []
    for _, g in df.iterrows():
        s, country, rnd = g["season"], g["country"], g["round"]
        ch = circuit[country]
        circuit_hist = ch[0] / ch[1] if ch[1] else _NEUTRAL
        sr = season_recent[s]
        season_form = float(np.mean(sr)) if sr else _NEUTRAL

        rows.append({
            "item_id": g["item_id"], "sport": "f1", "season": s, "date": g["date"],
            "stakes": rnd / max(1, season_len.get(s, 1)),   # later rounds decide titles
            "circuit_hist": circuit_hist,                    # some venues reliably deliver
            "season_form": season_form,                      # is this season producing action
            "excitement": excitement[g["item_id"]],
        })

        e = excitement[g["item_id"]]
        circuit[country][0] += e
        circuit[country][1] += 1
        season_recent[s].append(e)

    return pd.DataFrame(rows)

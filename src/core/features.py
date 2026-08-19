"""Unified excitement features — the layer that makes ONE cross-sport model legitimate.

A 7-second F1 gap and a 7-point NBA margin only become the same thing after
each sport is mapped onto a shared vocabulary. We do that with fixed,
interpretable parametric transforms (NOT scalers fitted on the data), so:

  * every feature lands on a comparable ~[0, 1] scale where higher = more
    exciting, regardless of sport;
  * there is zero train/test leakage (no statistic is learned from the rows);
  * each transform is a one-line, defensible definition you can explain.

Each vertical implements `normalize_*(raw_df) -> unified_df`. `unify()` stacks
them into the single table the shared model trains on.
"""

from __future__ import annotations

import math

import pandas as pd

# The shared feature vocabulary. Higher = more exciting, all on ~[0, 1].
UNIFIED_FEATURES = ["competitiveness", "volatility", "comeback", "chaos", "upset", "stakes"]
# `entities` holds spoiler-safe participants (NBA teams) the personalization
# layer matches against followed_entities; empty where unavailable (F1 drivers).
META_COLUMNS = ["item_id", "sport", "label", "date", "season", "entities"]


def _decay(x: pd.Series | float, scale: float) -> pd.Series | float:
    """exp(-x/scale): a small gap/margin -> ~1 (tight), a large one -> ~0."""
    return (-(x.astype(float)) / scale).apply(math.exp) if isinstance(x, pd.Series) else math.exp(-x / scale)


def _sat(x: pd.Series, cap: float) -> pd.Series:
    """min(x/cap, 1): saturating linear ramp to 1 at `cap`."""
    return (x.astype(float) / cap).clip(0.0, 1.0)


def normalize_f1(df: pd.DataFrame) -> pd.DataFrame:
    """Map the F1 per-race table onto the unified feature vocabulary."""
    out = pd.DataFrame()
    out["competitiveness"] = _decay(df["winner_margin_s"].fillna(30.0), scale=12.0)  # 0s->1, ~12s->0.37
    out["volatility"] = _sat(df["total_positions_moved"], cap=150.0)                  # Hungary '21 ~154
    out["comeback"] = _sat(df["max_gain"].clip(lower=0), cap=15.0)                    # biggest climb up the order
    out["chaos"] = _sat(df["n_dnf"], cap=8.0)                                         # retirements
    out["upset"] = _sat((df["winner_grid"] - 1).clip(lower=0), cap=10.0)             # winner started off pole
    # Stakes: later rounds decide championships. round / season length.
    season_len = df.groupby("season")["round"].transform("max").clip(lower=1)
    out["stakes"] = (df["round"] / season_len).clip(0.0, 1.0)

    out["item_id"] = df["item_id"]
    out["sport"] = "f1"
    out["label"] = df["event_name"]
    out["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    out["season"] = df["season"]
    # Every driver races every round, so a followed *driver* carries no signal;
    # F1 personalization leans on stakes instead (see personalize.py).
    out["entities"] = [[] for _ in range(len(df))]
    return out[META_COLUMNS + UNIFIED_FEATURES]


def normalize_nba(df: pd.DataFrame) -> pd.DataFrame:
    """Map the NBA per-game table onto the unified feature vocabulary."""
    out = pd.DataFrame()
    out["competitiveness"] = _decay(df["final_margin"], scale=10.0)                   # 0pt->1, ~10pt->0.37
    out["volatility"] = _sat(df["lead_changes"], cap=3.0)                             # saturates (quarter-boundary proxy)
    out["comeback"] = df["winner_came_from_behind"].astype(float).clip(0.0, 1.0)      # trailed after Q3, won
    out["chaos"] = _sat(df["overtime_periods"], cap=1.0)                              # any overtime = max drama
    out["upset"] = 0.0                                                                 # needs pre-game odds/standings (future)
    out["stakes"] = df["is_playoff"].astype(float).clip(0.0, 1.0)

    out["item_id"] = df["item_id"]
    out["sport"] = "nba"
    out["label"] = df["away"].astype(str) + " @ " + df["home"].astype(str)
    out["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    out["season"] = df["season"]
    out["entities"] = [[a, h] for a, h in zip(df["away"].astype(str), df["home"].astype(str))]
    return out[META_COLUMNS + UNIFIED_FEATURES]


def normalize_soccer(df: pd.DataFrame) -> pd.DataFrame:
    """Map the soccer per-match table onto the unified feature vocabulary."""
    out = pd.DataFrame()
    out["competitiveness"] = _decay(df["final_margin"], scale=1.5)          # 0-goal draw -> 1
    out["volatility"] = _sat(df["lead_changes"], cap=3.0)                    # back-and-forth
    out["comeback"] = df["came_from_behind"].astype(float).clip(0.0, 1.0)
    # eventful: red cards, a goal-fest, and/or late drama
    goalfest = (pd.to_numeric(df["total_goals"], errors="coerce") >= 4).astype(float)
    out["chaos"] = (0.5 * _sat(df["red_cards"], 2.0) + 0.4 * goalfest
                    + 0.3 * df["late_drama"].astype(float)).clip(0.0, 1.0)
    out["upset"] = 0.0                                                       # needs table/odds (future)
    out["stakes"] = df["is_knockout"].astype(float).clip(0.0, 1.0) * 0.7 + 0.3

    out["item_id"] = df["item_id"]
    out["sport"] = "soccer"
    out["label"] = df["away"].astype(str) + " v " + df["home"].astype(str)
    out["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    out["season"] = df["season"]
    out["entities"] = [[a, h] for a, h in zip(df["away"].astype(str), df["home"].astype(str))]
    return out[META_COLUMNS + UNIFIED_FEATURES]


def unify(f1_df: pd.DataFrame | None = None, nba_df: pd.DataFrame | None = None,
          soccer_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stack normalized per-sport tables into one training table."""
    parts = []
    for df, norm in [(f1_df, normalize_f1), (nba_df, normalize_nba), (soccer_df, normalize_soccer)]:
        if df is not None and len(df):
            parts.append(norm(df))
    if not parts:
        return pd.DataFrame(columns=META_COLUMNS + UNIFIED_FEATURES)
    return pd.concat(parts, ignore_index=True).sort_values(["date"]).reset_index(drop=True)

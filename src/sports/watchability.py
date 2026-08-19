"""Watchability — a transparent pre-game index for UPCOMING events.

The predictive model showed realized excitement is ~unpredictable from pre-game
signals, so we DON'T pretend to forecast thrillers. Instead we rank upcoming
events by what is genuinely knowable beforehand and what actually drives "should
I watch this": stakes, matchup pedigree, and expected competitiveness. It's a
hand-weighted, interpretable index — honest about what it is.

Every component is normalized to [0, 1] (higher = more worth watching):
  * stakes          — playoff/knockout/late-season importance
  * pedigree        — how much this matchup/venue has delivered historically
  * competitiveness — how evenly matched (standings proximity)
  * form            — are both sides / the season currently producing quality
"""

from __future__ import annotations

import pandas as pd

WATCH_FEATURES = ["stakes", "pedigree", "competitiveness", "form"]
WEIGHTS = {"stakes": 0.35, "pedigree": 0.25, "competitiveness": 0.25, "form": 0.15}


class WatchabilityIndex:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(WEIGHTS)

    def score(self, df: pd.DataFrame) -> pd.Series:
        total = sum(self.weights.values())
        s = sum(df[f] * w for f, w in self.weights.items())
        return (s / total).clip(0.0, 1.0)


def reasons(row: pd.Series, tier: str) -> list[str]:
    """Spoiler-free, pre-game reasons — nothing about a result (there isn't one)."""
    out: list[str] = []
    sport = row.get("sport", "")
    if row.get("stakes", 0) >= 0.8:
        out.append("playoff game" if sport in ("nba",) else
                   "knockout leg" if sport in ("soccer",) and row.get("is_knockout") else
                   "late-season · title implications")
    if row.get("pedigree", 0) >= 0.66:
        out.append("this matchup tends to deliver")
    if row.get("competitiveness", 0) >= 0.66:
        out.append("evenly matched on form")
    return out

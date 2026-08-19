"""UpcomingRecommender — rank upcoming fixtures by watchability x personalization.

Same two-stage shape as the rest of the platform, but the objective is the
transparent pre-game WatchabilityIndex (not a fragile excitement forecast).
Reasons are inherently spoiler-free — the event hasn't happened.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.core.interfaces import Item, Ranked, Scored, UserProfile
from src.sports.personalize import DEFAULT_WEIGHTS, TasteWeights, personalize
from src.sports.watchability import WATCH_FEATURES, WatchabilityIndex
from src.sports.watchability import reasons as watch_reasons

_TIER = [(0.62, "must-watch"), (0.42, "worth it"), (0.0, "optional")]


def _tier(score: float) -> str:
    return next(label for thresh, label in _TIER if score >= thresh)


class UpcomingRecommender:
    vertical = "upcoming"

    def __init__(self, fixtures: pd.DataFrame, weights: TasteWeights = DEFAULT_WEIGHTS) -> None:
        self.fixtures = fixtures.reset_index(drop=True)
        self.weights = weights

    def recommend(self, user: UserProfile, top: int = 15) -> list[Ranked]:
        if self.fixtures.empty:
            return []
        watch = WatchabilityIndex().score(self.fixtures).to_numpy()
        scored = []
        for w, (_, row) in zip(watch, self.fixtures.iterrows()):
            item = Item(
                item_id=row["item_id"], vertical=row["sport"],
                when=row["date"].to_pydatetime() if hasattr(row["date"], "to_pydatetime") else row["date"],
                features={f: float(row[f]) for f in WATCH_FEATURES},
                meta={"label": row["label"], "entities": list(row.get("entities") or []),
                      "sport": row["sport"]},
            )
            mult, preasons = personalize(item, user, self.weights)
            score = float(w) * mult
            rs = [f"{_tier(score)} · {row['label']} ({row['sport'].upper()})"]
            # watchability already surfaces stakes; from personalization keep only
            # the "you follow …" reason so stakes isn't stated twice.
            rs += watch_reasons(row, _tier(score)) + [r for r in preasons if "you follow" in r]
            scored.append(Scored(item=item, excitement=float(w), personalization=mult, reasons=rs))
        # "Coming up" is chronological — soonest first — with watchability breaking
        # ties within the same day. (Ranking far-future high-stakes events to the
        # top is wrong for an "upcoming" view.)
        ordered = sorted(scored, key=lambda s: (s.item.when, -s.score))
        return [Ranked(rank=i + 1, scored=s) for i, s in enumerate(ordered)][:top]

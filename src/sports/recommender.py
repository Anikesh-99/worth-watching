"""SportRecommender — the two-stage pipeline behind the shared interface.

generate_candidates (recall events in a time window) -> score (excitement x
personalization, with spoiler-free reasons) -> rank. One instance can serve a
single sport or several at once, because it operates on the *unified* table;
the media vertical will implement this same protocol.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd

from src.core.excitement import ExcitementIndex
from src.core.features import UNIFIED_FEATURES
from src.core.interfaces import Item, Ranked, Scored, UserProfile
from src.sports.personalize import DEFAULT_WEIGHTS, TasteWeights, personalize

# An excitement scorer maps a features frame -> a [0, 1] score per row.
Scorer = Callable[[pd.DataFrame], "pd.Series[float]"]

_TIER = [(0.66, "must-watch"), (0.4, "solid"), (0.0, "optional")]


def _tier(score: float) -> str:
    return next(label for thresh, label in _TIER if score >= thresh)


class SportRecommender:
    """Implements the `Recommender` protocol over a unified events table."""

    def __init__(self, unified: pd.DataFrame, scorer: Scorer | None = None,
                 vertical: str = "sports", weights: TasteWeights = DEFAULT_WEIGHTS) -> None:
        self.unified = unified.reset_index(drop=True)
        self.scorer = scorer or ExcitementIndex().score
        self.vertical = vertical
        self.weights = weights

    # -- stage 1: candidate generation ------------------------------------

    def generate_candidates(self, start: datetime, end: datetime) -> list[Item]:
        win = self.unified[(self.unified["date"] >= start) & (self.unified["date"] <= end)]
        items = []
        for _, row in win.iterrows():
            items.append(Item(
                item_id=row["item_id"],
                vertical=row["sport"],
                when=row["date"].to_pydatetime() if hasattr(row["date"], "to_pydatetime") else row["date"],
                features={f: float(row[f]) for f in UNIFIED_FEATURES},
                meta={"label": row["label"], "entities": list(row["entities"]),
                      "sport": row["sport"]},
            ))
        return items

    # -- stage 2: scoring (excitement x personalization) ------------------

    def score(self, items: list[Item], user: UserProfile) -> list[Scored]:
        if not items:
            return []
        feats = pd.DataFrame([it.features for it in items])[UNIFIED_FEATURES]
        excite = pd.Series(self.scorer(feats)).to_numpy()
        scored = []
        for it, ex in zip(items, excite):
            mult, reasons = personalize(it, user, self.weights)
            reasons = [f"{_tier(float(ex))} · {it.meta['label']} ({it.vertical.upper()})", *reasons]
            scored.append(Scored(item=it, excitement=float(ex), personalization=float(mult), reasons=reasons))
        return scored

    # -- stage 3: ranking -------------------------------------------------

    def rank(self, scored: list[Scored]) -> list[Ranked]:
        ordered = sorted(scored, key=lambda s: s.score, reverse=True)
        return [Ranked(rank=i + 1, scored=s) for i, s in enumerate(ordered)]

    # convenience: full pipeline for a window
    def watchlist(self, start: datetime, end: datetime, user: UserProfile, top: int | None = None) -> list[Ranked]:
        ranked = self.rank(self.score(self.generate_candidates(start, end), user))
        return ranked[:top] if top else ranked

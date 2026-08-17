"""Excitement scoring — the objective, user-independent layer.

Two scorers over the unified features:

  * `ExcitementIndex` — a transparent weighted composite. No training, fully
    interpretable, and the baseline the learned model must justify itself
    against.

  * `ExcitementModel` — a single LightGBM LambdaRank model shared across every
    sport. Honest note: absent human excitement ratings at scale, it is
    *bootstrapped* on relevance grades derived from the index. That makes it a
    demonstration of the learning-to-rank pipeline and a cross-sport
    feature-importance tool today; its lasting role is to be retrained on the
    user's real ratings in Phase 4 (`fit` accepts any relevance label).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from src.core.features import UNIFIED_FEATURES

# Transparent weights (sum to 1). Closeness dominates; stakes nudges ties.
INDEX_WEIGHTS: dict[str, float] = {
    "competitiveness": 0.30,
    "comeback": 0.20,
    "volatility": 0.20,
    "chaos": 0.15,
    "upset": 0.10,
    "stakes": 0.05,
}


@dataclass
class ExcitementIndex:
    weights: dict[str, float] = field(default_factory=lambda: dict(INDEX_WEIGHTS))

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Weighted sum of unified features -> excitement in [0, 1]."""
        total = sum(self.weights.values())
        s = sum(df[f] * w for f, w in self.weights.items())
        return (s / total).clip(0.0, 1.0)

    def grades(self, df: pd.DataFrame, n_grades: int = 5) -> np.ndarray:
        """Bin the index into integer relevance grades for LambdaRank labels."""
        idx = self.score(df)
        # Rank-based binning is robust to the index's skew; ties share a grade.
        ranks = idx.rank(method="average", pct=True)
        return np.floor(ranks * n_grades).clip(0, n_grades - 1).astype(int).to_numpy()


def group_sizes(keys: pd.Series) -> list[int]:
    """Contiguous group sizes for LGBMRanker (rows must be pre-sorted by key)."""
    return keys.groupby(keys, sort=False).size().tolist()


@dataclass
class ExcitementModel:
    """Shared cross-sport LambdaRank model over the unified features."""

    params: dict = field(default_factory=lambda: dict(
        objective="lambdarank", n_estimators=300, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, random_state=0, n_jobs=-1, verbose=-1,
    ))
    model: LGBMRanker | None = None
    features: list[str] = field(default_factory=lambda: list(UNIFIED_FEATURES))

    def fit(self, df: pd.DataFrame, y: np.ndarray, group_key: str = "season") -> "ExcitementModel":
        """Train on relevance labels `y`, grouping rows into ranking queries.

        `df` must be sorted so that rows sharing a (sport, group_key) query are
        contiguous. Any `y` works — index-derived grades now, real user ratings
        later.
        """
        d = df.reset_index(drop=True)
        gk = d["sport"].astype(str) + "-" + d[group_key].astype(str)
        self.model = LGBMRanker(**self.params)
        self.model.fit(d[self.features], y, group=group_sizes(gk))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ExcitementModel is not fitted")
        return self.model.predict(df[self.features])

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("ExcitementModel is not fitted")
        return pd.Series(self.model.feature_importances_, index=self.features).sort_values(ascending=False)

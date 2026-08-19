"""PredictiveExcitementModel — pre-game features -> predicted excitement.

A LightGBM regressor that learns realized excitement from pre-game signals, so
it can score UPCOMING fixtures that have no box score yet. Trained per sport
(the feature sets differ) and evaluated with a temporal split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


@dataclass
class PredictiveExcitementModel:
    features: list[str]
    params: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=0, n_jobs=-1, verbose=-1,
    ))
    model: LGBMRegressor | None = None

    def fit(self, train: pd.DataFrame, target: str = "excitement") -> "PredictiveExcitementModel":
        self.model = LGBMRegressor(**self.params)
        self.model.fit(train[self.features], train[target])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fitted")
        return self.model.predict(df[self.features])

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model not fitted")
        return pd.Series(self.model.feature_importances_, index=self.features).sort_values(ascending=False)

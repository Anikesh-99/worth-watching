"""Evaluation harness — shared by every vertical.

Two rules keep this honest:

  1. **Temporal split.** Train on earlier seasons, test on the latest. A random
     split would leak the future into the past and inflate every metric.

  2. **What "ground truth" means here.** The transparent `ExcitementIndex`
     *defines* excitement, so it trivially ranks its own grades — it is not a
     baseline to beat. The learned model's job is to (a) generalize that
     ranking to a season it never saw and (b) beat naive baselines
     (chronological order, a single feature). Those are the comparisons below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def temporal_split(df: pd.DataFrame, test_seasons: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train (earlier seasons) and test (given seasons)."""
    test = df[df["season"].isin(test_seasons)].reset_index(drop=True)
    train = df[~df["season"].isin(test_seasons)].reset_index(drop=True)
    return train, test


def ndcg_grouped(df: pd.DataFrame, score_col: str, gain_col: str,
                 group_cols: list[str], k: int = 10) -> float:
    """Mean NDCG@k over ranking queries (each group is one query)."""
    scores = []
    for _, g in df.groupby(group_cols):
        if len(g) < 2:
            continue
        y_true = g[gain_col].to_numpy().reshape(1, -1).astype(float)
        y_score = g[score_col].to_numpy().reshape(1, -1).astype(float)
        scores.append(ndcg_score(y_true, y_score, k=k))
    return float(np.mean(scores)) if scores else float("nan")


def spearman(df: pd.DataFrame, a_col: str, b_col: str) -> float:
    """Rank correlation between two orderings (no scipy dependency)."""
    return float(df[a_col].corr(df[b_col], method="spearman"))


def mrr_grouped(df: pd.DataFrame, score_col: str, gain_col: str, group_cols: list[str]) -> float:
    """Mean reciprocal rank of the first top-grade item per query."""
    rr = []
    for _, g in df.groupby(group_cols):
        if len(g) < 2:
            continue
        top = g[gain_col].max()
        order = g.sort_values(score_col, ascending=False).reset_index(drop=True)
        hits = order.index[order[gain_col] == top]
        rr.append(1.0 / (hits[0] + 1) if len(hits) else 0.0)
    return float(np.mean(rr)) if rr else float("nan")

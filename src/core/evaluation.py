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


def map_grouped(df: pd.DataFrame, score_col: str, gain_col: str, group_cols: list[str],
                k: int = 10, rel_threshold: float | None = None) -> float:
    """Mean Average Precision@k. `rel_threshold` binarizes the graded gain into
    relevant / not (default: the query's own top grade counts as relevant)."""
    aps = []
    for _, g in df.groupby(group_cols):
        if len(g) < 2:
            continue
        thr = g[gain_col].max() if rel_threshold is None else rel_threshold
        order = g.sort_values(score_col, ascending=False).reset_index(drop=True)
        rel = (order[gain_col] >= thr).to_numpy().astype(float)[:k]
        n_rel = rel.sum()
        if n_rel == 0:
            continue
        hits = 0
        precisions = []
        for i, r in enumerate(rel):
            if r:
                hits += 1
                precisions.append(hits / (i + 1))
        aps.append(sum(precisions) / min(k, int(n_rel)))
    return float(np.mean(aps)) if aps else float("nan")


# ---- beyond-accuracy: a good watch-list isn't five games of one team -----------

def intra_list_diversity(items: list, sim) -> float:
    """1 - mean pairwise similarity of a produced list (higher = more diverse).

    `sim(a, b) -> [0, 1]` is a domain similarity (e.g. same sport + shared team).
    A single-item (or empty) list is maximally diverse by convention (1.0).
    """
    n = len(items)
    if n < 2:
        return 1.0
    total = sum(sim(items[i], items[j]) for i in range(n) for j in range(i + 1, n))
    return float(1.0 - total / (n * (n - 1) / 2))


def novelty(items: list, popularity: dict) -> float:
    """Mean self-information (-log2 popularity share): rewards less-obvious picks.

    `popularity[item_id]` is a count/frequency; rarer items score higher. Items
    absent from the map are treated as maximally novel for that pass.
    """
    if not items:
        return float("nan")
    total = sum(popularity.values()) or 1
    info = []
    for it in items:
        share = popularity.get(it, 0.5) / total
        info.append(-np.log2(share) if share > 0 else np.log2(total))
    return float(np.mean(info))

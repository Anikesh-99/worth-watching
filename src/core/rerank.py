"""Maximal Marginal Relevance — a diversity-aware stage-3 re-ranker.

Accuracy is not the objective: a watch-list that is five games of the same team
is "accurate" and useless. MMR greedily builds the list, trading a little
relevance for diversity via one knob:

    next = argmax_i   λ · relevance(i)  −  (1 − λ) · max_{j already picked} sim(i, j)

λ = 1.0 reproduces the pure relevance ranking; lower λ spreads the list across
teams/sports. It's the standard, interpretable way to inject diversity, and it's
domain-agnostic — `relevance` and `sim` are supplied by the caller.
"""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def mmr_rerank(items: Sequence[T], relevance: Callable[[T], float],
               sim: Callable[[T, T], float], lam: float = 0.7,
               k: int | None = None) -> list[T]:
    """Greedy MMR re-rank. `relevance(item)` and `sim(a, b)` land in [0, 1]."""
    remaining = list(items)
    k = len(remaining) if k is None else min(k, len(remaining))
    selected: list[T] = []
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=relevance)          # first pick = most relevant
        else:
            best = max(remaining, key=lambda it: lam * relevance(it)
                       - (1 - lam) * max(sim(it, s) for s in selected))
        selected.append(best)
        remaining.remove(best)
    return selected


def event_similarity(a: dict, b: dict) -> float:
    """Similarity of two sports events for diversity: same sport + shared team.

    Each takes {"sport": str, "entities": list[str]}. Same sport contributes 0.5;
    Jaccard overlap of the two entity sets contributes the other 0.5. Two games
    of the same team in the same league are near-1; a race vs an NBA game is 0.
    """
    same_sport = 0.5 if a.get("sport") == b.get("sport") else 0.0
    ea, eb = set(a.get("entities") or []), set(b.get("entities") or [])
    jac = len(ea & eb) / len(ea | eb) if (ea or eb) else 0.0
    return same_sport + 0.5 * jac

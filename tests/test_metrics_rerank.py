"""Tests for the new eval metrics (MAP, diversity, novelty) and the MMR re-rank."""

from __future__ import annotations

import pandas as pd

from src.core.evaluation import intra_list_diversity, map_grouped, novelty
from src.core.rerank import event_similarity, mmr_rerank


# ---------------------------------------------------------------------- MAP
def _query(scores, gains):
    return pd.DataFrame({"g": 0, "score": scores, "rating": gains})


def test_map_perfect_vs_reversed() -> None:
    gains = [5, 4, 1, 1]
    perfect = _query([4, 3, 2, 1], gains)     # score order == rating order
    reversed_ = _query([1, 2, 3, 4], gains)   # worst first
    ap_perfect = map_grouped(perfect, "score", "rating", ["g"], k=10, rel_threshold=4)
    ap_reversed = map_grouped(reversed_, "score", "rating", ["g"], k=10, rel_threshold=4)
    assert ap_perfect == 1.0                  # both relevant items ranked top
    assert ap_reversed < ap_perfect


# -------------------------------------------------------------- diversity/novelty
def test_intra_list_diversity_bounds() -> None:
    same = [{"sport": "nba", "entities": ["A", "B"]}, {"sport": "nba", "entities": ["A", "B"]}]
    diff = [{"sport": "nba", "entities": ["A", "B"]}, {"sport": "f1", "entities": ["Z"]}]
    assert intra_list_diversity(same, event_similarity) == 0.0    # identical -> no diversity
    assert intra_list_diversity(diff, event_similarity) == 1.0    # disjoint sport+teams
    assert intra_list_diversity([{"sport": "nba", "entities": []}], event_similarity) == 1.0  # singleton


def test_novelty_prefers_rare() -> None:
    pop = {"common": 100, "rare": 1}
    assert novelty(["rare"], pop) > novelty(["common"], pop)


# --------------------------------------------------------------------- MMR
def test_mmr_lambda1_is_pure_relevance() -> None:
    items = [{"item_id": "a"}, {"item_id": "b"}, {"item_id": "c"}]
    rel = {"a": 0.9, "b": 0.5, "c": 0.1}
    order = mmr_rerank(items, relevance=lambda it: rel[it["item_id"]],
                       sim=lambda x, y: 0.0, lam=1.0)
    assert [it["item_id"] for it in order] == ["a", "b", "c"]


def test_mmr_promotes_diverse_item() -> None:
    # Two near-duplicate high-relevance items + one slightly-lower diverse item.
    items = [
        {"item_id": "dup1", "sport": "nba", "entities": ["A", "B"]},
        {"item_id": "dup2", "sport": "nba", "entities": ["A", "B"]},
        {"item_id": "div", "sport": "f1", "entities": ["Z"]},
    ]
    rel = {"dup1": 1.0, "dup2": 0.95, "div": 0.9}
    order = mmr_rerank(items, relevance=lambda it: rel[it["item_id"]],
                       sim=event_similarity, lam=0.5)
    # after picking dup1, the diverse item should beat the near-duplicate dup2
    assert order[0]["item_id"] == "dup1"
    assert order[1]["item_id"] == "div"


def test_event_similarity() -> None:
    a = {"sport": "nba", "entities": ["A", "B"]}
    assert event_similarity(a, a) == 1.0                                   # identical
    assert event_similarity(a, {"sport": "f1", "entities": ["Z"]}) == 0.0  # nothing shared
    half = event_similarity(a, {"sport": "nba", "entities": ["C", "D"]})   # same sport only
    assert half == 0.5

"""Tests for the media embedding tower + nearest-neighbour retrieval."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.interfaces import UserProfile
from src.media.content import ContentRecommender
from src.media.embeddings import build_item_embeddings, project_taste
from src.media.retrieval import EmbeddingRetriever, recall_at_k


# ---------------------------------------------------------------- embeddings
def test_build_embeddings_unit_rows_and_dim() -> None:
    matrix = np.eye(6)                         # 6 items, 6 tags
    emb, svd = build_item_embeddings(matrix, dim=3)
    assert emb.shape == (6, 3)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0)   # unit rows
    assert svd is not None


def test_build_embeddings_degrades_when_too_small() -> None:
    emb, svd = build_item_embeddings(np.array([[1.0, 0.0], [0.0, 1.0]]), dim=8)
    assert svd is None                          # can't factorize -> keep tag space
    assert emb.shape == (2, 2)


def test_project_taste_passthrough_when_no_svd() -> None:
    v = np.array([3.0, 4.0])
    assert np.allclose(project_taste(v, None), [0.6, 0.8])   # just unit-normalized


# ---------------------------------------------------------------- retriever
def _emb():
    # four clearly separated unit vectors
    return np.array([[1.0, 0, 0], [0.9, 0.1, 0], [0, 1.0, 0], [0, 0, 1.0]])


def test_exact_retriever_returns_nearest_first() -> None:
    r = EmbeddingRetriever(_emb(), backend="exact")
    idx = r.query(np.array([1.0, 0, 0]), k=2)
    assert idx[0] == 0 and set(idx.tolist()) == {0, 1}     # the two x-aligned rows


def test_lsh_retriever_returns_k_and_recall_defined() -> None:
    emb = np.random.default_rng(0).standard_normal((50, 8))
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[7]
    exact = EmbeddingRetriever(emb, backend="exact").query(q, 5)
    approx = EmbeddingRetriever(emb, backend="lsh", n_planes=24, seed=1).query(q, 5)
    assert len(approx) == 5
    assert exact[0] == 7                                   # query is its own nearest
    assert 0.0 <= recall_at_k(exact, approx) <= 1.0


# ------------------------------------------------------- ContentRecommender wiring
def _catalog() -> pd.DataFrame:
    rows = [                                    # ids carry the vertical prefix, as real catalogs do
        ("content-a1", "Action One", "Action|Adventure", "", 8.0),
        ("content-a2", "Action Two", "Action|Adventure", "", 7.5),
        ("content-a3", "Romance One", "Romance|Drama", "", 8.2),
        ("content-a4", "Romance Two", "Romance|Drama", "", 7.0),
        ("content-a5", "SciFi One", "SciFi|Space", "", 9.0),
    ]
    return pd.DataFrame(rows, columns=["item_id", "title", "genres", "themes", "score"]).assign(year=2020)


def test_nearest_finds_same_genre() -> None:
    rec = ContentRecommender(_catalog())
    nbrs = rec.nearest("content-a1", k=1)       # Action One -> Action Two is closest
    assert nbrs == ["content-a2"]


def test_retrieve_prunes_candidate_set() -> None:
    rec = ContentRecommender(_catalog())
    user = UserProfile(followed_entities={"Romance", "Drama"})   # cold-start taste = romance
    keep = rec.retrieve_candidates(user, k=2)
    assert len(keep) == 2
    full = rec.recommend(user, top=5)
    pruned = rec.recommend(user, top=5, retrieve=2)
    assert len(pruned) <= 2 <= len(full)        # retrieval shrinks the candidate pool

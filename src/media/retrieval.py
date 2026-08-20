"""Nearest-neighbour retrieval over item embeddings — stage 1 of two-stage recs.

Retrieval shrinks the catalog to the K items nearest a taste vector, which the
exact scorer then ranks (stage 2). Two backends make the accuracy/speed tradeoff
concrete:

  * ``exact``  — full cosine over the embedding matrix (numpy). Correct, and the
    right choice at these catalog sizes.
  * ``lsh``    — random-hyperplane locality-sensitive hashing. Each item gets a
    sign code from `n_planes` random projections; a query ranks items by Hamming
    distance of their codes, exact-scoring only a small candidate pool. This is
    approximate (recall < 1) but sub-linear in spirit — the shape a real ANN
    index (hnswlib/faiss) takes at 10M items. Built from scratch to show *how*
    ANN trades recall for speed, not just to import it.
"""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class EmbeddingRetriever:
    def __init__(self, embeddings: np.ndarray, backend: str = "exact",
                 n_planes: int = 16, seed: int = 0) -> None:
        self.emb = embeddings                        # (n_items, dim), unit rows
        self.backend = backend
        if backend == "lsh" and len(embeddings):
            rng = np.random.default_rng(seed)
            self.planes = rng.standard_normal((n_planes, embeddings.shape[1]))
            self.codes = (embeddings @ self.planes.T) > 0   # (n_items, n_planes) bool

    def query(self, vec: np.ndarray, k: int) -> np.ndarray:
        """Row indices of the top-k nearest items to `vec` (best first)."""
        if not len(self.emb):
            return np.array([], dtype=int)
        vec = _unit(vec)
        k = min(k, len(self.emb))
        if self.backend != "lsh":
            sims = self.emb @ vec
            top = np.argpartition(-sims, k - 1)[:k]
            return top[np.argsort(-sims[top])]
        # LSH: shortlist by Hamming distance of sign codes, then exact-rank the pool.
        qcode = (self.planes @ vec) > 0
        ham = (self.codes != qcode).sum(axis=1)
        pool = np.argsort(ham)[:max(4 * k, k)]       # candidate pool (approximate step)
        sims = self.emb[pool] @ vec
        return pool[np.argsort(-sims)][:k]


def recall_at_k(exact_idx: np.ndarray, approx_idx: np.ndarray) -> float:
    """Fraction of the exact top-k that an approximate retrieval also returned."""
    if not len(exact_idx):
        return float("nan")
    return len(set(exact_idx.tolist()) & set(approx_idx.tolist())) / len(exact_idx)

"""Dense item embeddings for the media vertical (the retrieval "tower").

The content matrix is a sparse (items x tags) multi-hot. That's fine for exact
cosine, but it's high-dimensional and doesn't capture that "shounen" and
"action" co-occur. `TruncatedSVD` (LSA) factorizes it into a low-rank dense
space where correlated tags collapse into shared latent factors — so items are
comparable by a short dense vector, and a taste vector projects into the *same*
space. That dense space is what a nearest-neighbour retriever indexes.

At these catalog sizes SVD is instant; the value is the two-tower shape (item
embedding + taste projection into one space), which is the seam an ANN service
slots into at scale.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _unit_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)


def build_item_embeddings(matrix: np.ndarray, dim: int = 32, seed: int = 0
                          ) -> tuple[np.ndarray, TruncatedSVD | None]:
    """(items x tags) -> (unit-normalized items x dim, fitted svd).

    Degrades gracefully: a catalog too small to factorize (few items/tags) keeps
    the original tag space (svd=None), so retrieval still works — just unreduced.
    """
    if matrix.size == 0:
        return matrix, None
    n_items, n_tags = matrix.shape
    k = min(dim, n_tags - 1, n_items - 1)
    if k < 2:
        return _unit_rows(matrix), None
    svd = TruncatedSVD(n_components=k, random_state=seed)
    emb = svd.fit_transform(matrix)
    return _unit_rows(emb), svd


def project_taste(taste_tagspace: np.ndarray, svd: TruncatedSVD | None) -> np.ndarray:
    """Map a taste vector from tag space into the item embedding space (unit)."""
    if svd is None:
        return _unit(taste_tagspace)
    return _unit(svd.transform(taste_tagspace.reshape(1, -1))[0])

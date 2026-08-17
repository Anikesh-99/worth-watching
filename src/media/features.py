"""Content features for the media vertical: multi-hot genre + theme vectors.

Where sports features are excitement signals, media features are *content*: a
binary vector over the union of genres and themes. A user's taste is the
rating-weighted sum of the vectors of titles they've rated; a candidate's fit
is the cosine similarity to that taste vector. Simple, transparent,
content-based collaborative-free recommendation — appropriate when you have one
user's ratings, not millions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Default tag columns (anime). Books override with ("subjects",).
DEFAULT_TAG_COLS = ("genres", "themes")


def _tags(row: pd.Series, tag_cols: tuple[str, ...] = DEFAULT_TAG_COLS) -> list[str]:
    out: list[str] = []
    for col in tag_cols:
        val = row.get(col)
        if isinstance(val, str) and val:
            out.extend(t for t in val.split("|") if t)
    return out


def build_vocab(df: pd.DataFrame, tag_cols: tuple[str, ...] = DEFAULT_TAG_COLS) -> list[str]:
    """Sorted union of all tags in the catalog's tag columns."""
    vocab: set[str] = set()
    for _, row in df.iterrows():
        vocab.update(_tags(row, tag_cols))
    return sorted(vocab)


def content_matrix(df: pd.DataFrame, vocab: list[str],
                   tag_cols: tuple[str, ...] = DEFAULT_TAG_COLS) -> np.ndarray:
    """Binary (n_items x n_tags) matrix over the vocab."""
    index = {t: i for i, t in enumerate(vocab)}
    m = np.zeros((len(df), len(vocab)), dtype=float)
    for r, (_, row) in enumerate(df.iterrows()):
        for t in _tags(row, tag_cols):
            j = index.get(t)
            if j is not None:
                m[r, j] = 1.0
    return m


def quality_prior(df: pd.DataFrame, score_col: str = "score") -> np.ndarray:
    """A community-quality column min-max normalized to [0, 1] (user-independent)."""
    s = pd.to_numeric(df[score_col], errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or hi == lo:
        return np.full(len(df), 0.5)
    return ((s - lo) / (hi - lo)).fillna(0.4).to_numpy()

"""BookRecommender — the books media vertical.

Thin subclass of ContentRecommender: taste = Open Library subject tags, quality
= edition count (renown proxy). Same engine as anime, different columns.
"""

from __future__ import annotations

import pandas as pd

from src.media.content import ContentRecommender


class BookRecommender(ContentRecommender):
    vertical = "book"
    tag_cols = ("subjects",)
    title_col = "title"
    quality_col = "editions"
    year_col = "year"
    id_prefix = "book-"
    tiers = [(0.66, "must-read"), (0.4, "worth it"), (0.0, "maybe")]

    def _candidate_meta(self, row: pd.Series) -> dict:
        meta = super()._candidate_meta(row)
        meta["author"] = row.get("author", "")
        return meta

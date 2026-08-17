"""AnimeRecommender — the anime media vertical.

A thin subclass of the shared ContentRecommender: anime taste = genres + themes,
quality = community MAL score. Kept as its own class for a clear public name and
so the type reads meaningfully across the codebase.
"""

from __future__ import annotations

import pandas as pd

from src.media.content import ContentRecommender


class AnimeRecommender(ContentRecommender):
    vertical = "anime"
    tag_cols = ("genres", "themes")
    title_col = "title"
    quality_col = "score"
    year_col = "year"
    id_prefix = "anime-"

    def _candidate_meta(self, row: pd.Series) -> dict:
        meta = super()._candidate_meta(row)
        meta["type"] = row.get("type", "")
        return meta

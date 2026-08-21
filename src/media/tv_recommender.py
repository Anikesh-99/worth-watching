"""TVRecommender — the TV-shows media vertical.

Thin subclass of ContentRecommender over the same TMDB catalog shape as movies;
only the id prefix and vertical name differ.
"""

from __future__ import annotations

from src.media.content import ContentRecommender


class TVRecommender(ContentRecommender):
    vertical = "tv"
    tag_cols = ("genres",)
    title_col = "title"
    quality_col = "rating"
    year_col = "year"
    id_prefix = "tv-"
    tiers = [(0.66, "must-watch"), (0.4, "worth it"), (0.0, "maybe")]

"""MovieRecommender — the movies media vertical.

Thin subclass of ContentRecommender: taste = TMDB genre tags, quality = TMDB
community rating. Same engine as anime/books, different columns.
"""

from __future__ import annotations

from src.media.content import ContentRecommender


class MovieRecommender(ContentRecommender):
    vertical = "movie"
    tag_cols = ("genres",)
    title_col = "title"
    quality_col = "rating"
    year_col = "year"
    id_prefix = "movie-"
    tiers = [(0.66, "must-watch"), (0.4, "worth it"), (0.0, "maybe")]

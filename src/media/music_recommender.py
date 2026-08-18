"""MusicRecommender — the music (Spotify) vertical.

Thin subclass of ContentRecommender: taste = artist genres, quality = Spotify
popularity. Candidates are new-release albums; your top artists (as pseudo-rated
rows, merged via augment_catalog) supply the taste signal. Same engine as anime
and books — genres are just another tag vocabulary.
"""

from __future__ import annotations

import pandas as pd

from src.media.content import ContentRecommender


class MusicRecommender(ContentRecommender):
    vertical = "music"
    tag_cols = ("genres",)
    title_col = "name"
    quality_col = "popularity"
    year_col = "year"
    id_prefix = "music-"
    tiers = [(0.66, "on repeat"), (0.4, "worth a spin"), (0.0, "maybe")]

    def _candidate_meta(self, row: pd.Series) -> dict:
        meta = super()._candidate_meta(row)
        meta["artist"] = row.get("artist", "")
        return meta

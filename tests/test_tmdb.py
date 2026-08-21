"""Tests for the TMDB ingest (movie/tv row parsing) and the two recommenders."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.interfaces import UserProfile
from src.media.movie_recommender import MovieRecommender
from src.media.tmdb_ingest import TMDBIngest
from src.media.tv_recommender import TVRecommender


# ------------------------------------------------------------------ ingest
def test_movie_row_parses() -> None:
    ing = TMDBIngest("movie")                     # no key needed to parse a row
    row = ing._row(
        {"id": 603, "title": "The Matrix", "release_date": "1999-03-30",
         "vote_average": 8.2, "vote_count": 24000, "poster_path": "/abc.jpg",
         "genre_ids": [28, 878]},
        {28: "Action", 878: "Science Fiction"})
    assert row["item_id"] == "movie-603" and row["kind"] == "movie"
    assert row["title"] == "The Matrix" and row["year"] == 1999
    assert row["rating"] == 8.2
    assert row["genres"] == "Action|Science Fiction"
    assert row["image_url"] == "https://image.tmdb.org/t/p/w500/abc.jpg"


def test_tv_row_uses_name_and_air_date() -> None:
    ing = TMDBIngest("tv")
    row = ing._row(
        {"id": 1396, "name": "Breaking Bad", "first_air_date": "2008-01-20",
         "vote_average": 8.9, "vote_count": 12000, "poster_path": "/x.jpg", "genre_ids": [18]},
        {18: "Drama"})
    assert row["item_id"] == "tv-1396" and row["kind"] == "tv"
    assert row["title"] == "Breaking Bad" and row["year"] == 2008 and row["genres"] == "Drama"


def test_row_skips_incomplete_and_validates_type() -> None:
    ing = TMDBIngest("movie")
    assert ing._row({"id": 1, "release_date": "2000-01-01"}, {}) is None   # no title
    assert ing._row({"title": "x"}, {}) is None                            # no id
    with pytest.raises(ValueError):
        TMDBIngest("podcast")


def test_get_without_key_raises() -> None:
    ing = TMDBIngest("movie")
    ing.key = None
    with pytest.raises(RuntimeError):
        ing._get("/discover/movie", {})


# --------------------------------------------------------------- recommenders
def _catalog(prefix):
    rows = [
        (f"{prefix}-1", "Aegis", "Action|Science Fiction", 8.0, 2020),
        (f"{prefix}-2", "Letters", "Romance|Drama", 7.0, 2019),
        (f"{prefix}-3", "Nightfall", "Action|Thriller", 8.5, 2021),
        (f"{prefix}-4", "Orbit", "Science Fiction|Adventure", 7.8, 2022),
    ]
    return pd.DataFrame(rows, columns=["item_id", "title", "genres", "rating", "year"]).assign(image_url="")


def test_movie_recommender_taste_boosts_matching_genre() -> None:
    from datetime import datetime
    rec = MovieRecommender(_catalog("movie"))
    user = UserProfile(followed_entities={"Science Fiction"})
    items = rec.generate_candidates(datetime(1900, 1, 1), datetime(2100, 1, 1))
    scored = {s.item.item_id: s for s in rec.score(items, user)}
    assert len(scored) == 4 and all(s.item.vertical == "movie" for s in scored.values())
    # sci-fi titles get a taste multiplier > 1; a non-sci-fi title stays neutral
    assert scored["movie-1"].personalization > 1.0    # Action|Science Fiction
    assert scored["movie-4"].personalization > 1.0    # Science Fiction|Adventure
    assert scored["movie-2"].personalization <= 1.0   # Romance|Drama — no sci-fi


def test_tv_recommender_vertical_and_prefix() -> None:
    rec = TVRecommender(_catalog("tv"))
    recs = rec.recommend(UserProfile(followed_entities={"Drama"}), top=2)
    assert all(r.scored.item.vertical == "tv" for r in recs)
    assert all(r.scored.item.item_id.startswith("tv-") for r in recs)

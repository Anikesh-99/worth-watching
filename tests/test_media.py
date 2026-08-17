"""Tests for the anime (media) vertical — network-free with a synthetic catalog."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.core.interfaces import Ranked, Scored, UserProfile
from src.media.features import build_vocab, content_matrix, quality_prior
from src.media.recommender import AnimeRecommender


def _catalog() -> pd.DataFrame:
    return pd.DataFrame([
        dict(item_id="anime-1", mal_id=1, title="Action A", type="TV", year=2015,
             score=8.0, members=1000, genres="Action|Fantasy", themes=""),
        dict(item_id="anime-2", mal_id=2, title="Romance B", type="TV", year=2016,
             score=7.0, members=900, genres="Romance|Slice of Life", themes=""),
        dict(item_id="anime-3", mal_id=3, title="Action C", type="TV", year=2018,
             score=8.5, members=1200, genres="Action|Adventure", themes=""),
        dict(item_id="anime-4", mal_id=4, title="Romance D", type="TV", year=2019,
             score=6.0, members=500, genres="Romance|Drama", themes=""),
    ])


def test_vocab_and_matrix() -> None:
    cat = _catalog()
    vocab = build_vocab(cat)
    assert "Action" in vocab and "Romance" in vocab and vocab == sorted(vocab)
    m = content_matrix(cat, vocab)
    assert m.shape == (4, len(vocab))
    assert set(m.flatten()) <= {0.0, 1.0}


def test_quality_prior_normalized() -> None:
    q = quality_prior(_catalog())
    assert q.min() >= 0.0 and q.max() <= 1.0


def test_taste_ranks_matching_genre_higher() -> None:
    # Like Action (anime-1 high), dislike Romance (anime-2 low).
    user = UserProfile(ratings={"anime-1": 9, "anime-2": 3})
    rec = AnimeRecommender(_catalog())
    ranked = rec.recommend(user, top=10)  # excludes rated 1 & 2
    ids = [r.scored.item.item_id for r in ranked]
    assert ids.index("anime-3") < ids.index("anime-4")  # Action beats Romance
    assert all(isinstance(r, Ranked) for r in ranked)


def test_cold_start_uses_followed_genres() -> None:
    user = UserProfile(followed_entities={"Action"})
    rec = AnimeRecommender(_catalog())
    ranked = rec.recommend(user, top=10)
    top_genres = ranked[0].scored.item.meta["genres"]
    assert "Action" in top_genres


def test_reasons_are_content_based() -> None:
    user = UserProfile(ratings={"anime-1": 9, "anime-2": 3})
    rec = AnimeRecommender(_catalog())
    scored = rec.score(rec.generate_candidates(datetime(2000, 1, 1), datetime(2100, 1, 1)), user)
    for s in scored:
        assert isinstance(s, Scored)
        blob = " ".join(s.reasons).lower()
        assert "spoiler" not in blob  # never leaks plot
        if len(s.reasons) > 1:
            assert "matches your taste" in s.reasons[1]


def test_implements_recommender_interface() -> None:
    rec = AnimeRecommender(_catalog())
    assert callable(rec.generate_candidates) and callable(rec.score) and callable(rec.rank)
    assert rec.vertical == "anime"

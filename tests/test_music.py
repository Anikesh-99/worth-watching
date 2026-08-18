"""Tests for the music (Spotify) vertical — network-free with a synthetic catalog."""

from __future__ import annotations

import pandas as pd

from src.core.interfaces import Ranked, UserProfile
from src.media.content import augment_catalog
from src.media.music_recommender import MusicRecommender


def _catalog() -> pd.DataFrame:
    return pd.DataFrame([
        dict(item_id="music-album-1", name="Synth Nights", artist="A", year=2025,
             popularity=80, genres="electropop|synthpop"),
        dict(item_id="music-album-2", name="Trap House", artist="B", year=2025,
             popularity=75, genres="hip hop|trap"),
        dict(item_id="music-album-3", name="Neon Dreams", artist="C", year=2025,
             popularity=60, genres="synthpop|indie pop"),
        dict(item_id="music-album-4", name="Boom Bap", artist="D", year=2025,
             popularity=55, genres="hip hop|rap"),
    ])


def test_taste_ranks_matching_genre_higher() -> None:
    # Top artist rows carry genres; merge them so taste is grounded.
    rated = pd.DataFrame([dict(item_id="music-top1", rating=9, name="Fav", genres="synthpop")])
    cat = augment_catalog(_catalog(), rated)
    user = UserProfile(ratings={"music-top1": 9})
    ranked = MusicRecommender(cat).recommend(user, top=10)
    ids = [r.scored.item.item_id for r in ranked if r.scored.item.item_id.startswith("music-album")]
    # synthpop albums (1, 3) should out-rank hip hop albums (2, 4)
    assert ids.index("music-album-3") < ids.index("music-album-4")
    assert all(isinstance(r, Ranked) for r in ranked)


def test_cold_start_uses_followed_genres() -> None:
    user = UserProfile(followed_entities={"hip hop"})
    ranked = MusicRecommender(_catalog()).recommend(user, top=10)
    assert "hip hop" in ranked[0].scored.item.meta["genres"]


def test_tiers_and_interface() -> None:
    rec = MusicRecommender(_catalog())
    assert rec.vertical == "music"
    scored = rec.score(rec.generate_candidates(__import__("datetime").datetime(2000, 1, 1),
                                               __import__("datetime").datetime(2100, 1, 1)),
                       UserProfile(followed_entities={"synthpop"}))
    tiers = {s.reasons[0].split(" · ")[0] for s in scored}
    assert tiers & {"on repeat", "worth a spin", "maybe"}

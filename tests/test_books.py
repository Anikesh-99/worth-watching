"""Tests for the books vertical — network-free with a synthetic catalog."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.core.interfaces import Ranked, UserProfile
from src.media.book_ingest import _clean_subjects
from src.media.book_recommender import BookRecommender


def _catalog() -> pd.DataFrame:
    return pd.DataFrame([
        dict(item_id="book-1", title="Dragon Saga", author="A", year=2001,
             editions=100, subjects="Fantasy|Adventure"),
        dict(item_id="book-2", title="Love Story", author="B", year=2002,
             editions=90, subjects="Romance|Drama"),
        dict(item_id="book-3", title="Wizard Quest", author="C", year=2003,
             editions=80, subjects="Fantasy|Magic"),
        dict(item_id="book-4", title="A Royal Affair", author="D", year=2004,
             editions=50, subjects="Romance|History"),
    ])


def test_clean_subjects_drops_noise_and_normalizes() -> None:
    raw = ["fantasy", "Accessible book", "Protected DAISY",
           "a very long sentence style subject that should be dropped entirely here",
           "Fiction, general", "Magic"]
    cleaned = _clean_subjects(raw)
    assert "Fantasy" in cleaned and "Magic" in cleaned
    assert "Accessible Book" not in cleaned and "Protected Daisy" not in cleaned
    assert not any("," in c for c in cleaned)            # comma tags dropped
    assert all(len(c) <= 40 for c in cleaned)            # long sentences dropped


def test_taste_ranks_matching_subject_higher() -> None:
    user = UserProfile(ratings={"book-1": 5, "book-2": 1})  # likes Fantasy, dislikes Romance
    rec = BookRecommender(_catalog())
    ranked = rec.recommend(user, top=10)  # excludes read 1 & 2
    ids = [r.scored.item.item_id for r in ranked]
    assert ids.index("book-3") < ids.index("book-4")  # Fantasy beats Romance
    assert all(isinstance(r, Ranked) for r in ranked)


def test_cold_start_uses_followed_subjects() -> None:
    user = UserProfile(followed_entities={"Fantasy"})
    rec = BookRecommender(_catalog())
    ranked = rec.recommend(user, top=10)
    assert "Fantasy" in ranked[0].scored.item.meta["subjects"]


def test_reasons_are_content_based_and_book_flavored() -> None:
    user = UserProfile(ratings={"book-1": 5, "book-2": 1})
    rec = BookRecommender(_catalog())
    scored = rec.score(rec.generate_candidates(datetime(1990, 1, 1), datetime(2100, 1, 1)), user)
    tiers = {s.reasons[0].split(" · ")[0] for s in scored}
    assert tiers & {"must-read", "worth it", "maybe"}      # book-flavored tiers
    for s in scored:
        if len(s.reasons) > 1:
            assert "matches your taste" in s.reasons[1]


def test_implements_interface() -> None:
    rec = BookRecommender(_catalog())
    assert rec.vertical == "book"
    assert callable(rec.generate_candidates) and callable(rec.score) and callable(rec.rank)

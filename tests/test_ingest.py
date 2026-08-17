"""Label-correctness tests for the F1 ingest.

The excitement model is only as trustworthy as its labels, so we pin the
feature extraction against races with well-known outcomes: a chaotic,
high-retirement wet race versus a lights-to-flag procession. These are the
"known thriller vs blowout" checks called for in the design's verification
section. They hit the network on first run and then use FastF1's cache.
"""

from __future__ import annotations

import math

import pytest

from src.core.interfaces import Item, Scored, UserProfile
from src.sports.ingest import RACE_COLUMNS, F1Ingest


@pytest.fixture(scope="module")
def ingest() -> F1Ingest:
    return F1Ingest(cache_dir="data/f1cache")


def test_row_has_all_columns(ingest: F1Ingest) -> None:
    row = ingest._race_row(2023, 1)  # Bahrain
    assert row is not None
    assert set(row) == set(RACE_COLUMNS)


def test_bahrain_2023_retirement_count(ingest: F1Ingest) -> None:
    # Bahrain 2023 had exactly 3 retirements; lapped runners must NOT count.
    row = ingest._race_row(2023, 1)
    assert row["n_dnf"] == 3
    assert row["n_classified"] == 17
    assert 10 < row["winner_margin_s"] < 13  # Perez ~12s behind Verstappen


def test_thriller_beats_procession_on_chaos(ingest: F1Ingest) -> None:
    # 2023 Dutch GP (wet, 3 DNFs, big climbs) should show more field movement
    # and more retirements than the 2023 Belgian GP (a routine Verstappen win).
    dutch = ingest._race_row(2023, 13)
    belgian = ingest._race_row(2023, 12)
    assert dutch["total_positions_moved"] > belgian["total_positions_moved"]
    assert dutch["n_dnf"] >= belgian["n_dnf"]


def test_future_round_returns_none(ingest: F1Ingest) -> None:
    assert ingest._race_row(2023, 30) is None  # no round 30 exists


# -- pure-unit checks of the shared interface (no network) ----------------

def test_scored_score_is_product() -> None:
    item = Item(item_id="f1-2023-13", vertical="f1", when=__import__("datetime").datetime(2023, 8, 27))
    s = Scored(item=item, excitement=0.8, personalization=1.5, reasons=["close finish"])
    assert math.isclose(s.score, 1.2)


def test_user_profile_defaults() -> None:
    u = UserProfile()
    assert u.followed_entities == set()
    assert u.ratings == {}

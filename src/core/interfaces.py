"""Core abstractions shared by every recommender vertical.

The whole "platform, not two projects" claim rests on this file: a sports
match and an unwatched anime are the *same shape* of problem — score an
item's personalized worth-your-time-ness, then rank a candidate set. Any
vertical (F1, NBA, media) implements the `Recommender` protocol below and
gets the weekly watch-list + evaluation harness for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Item:
    """A single recommendable thing (one race, one game, one title).

    `features` holds the raw, vertical-specific inputs to the excitement
    model. `meta` holds display-only fields (names, dates) that must never
    leak the result into a spoiler-free ranking.
    """

    item_id: str
    vertical: str                       # "f1", "nba", "media", ...
    when: datetime                      # event start (or release) time
    features: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scored:
    item: Item
    excitement: float                   # objective, user-independent
    personalization: float              # taste multiplier for this user
    reasons: list[str] = field(default_factory=list)  # spoiler-free "why"

    @property
    def score(self) -> float:
        return self.excitement * self.personalization


@dataclass(frozen=True)
class Ranked:
    rank: int
    scored: Scored


@dataclass
class UserProfile:
    """Everything the personalization layer needs about one user."""

    followed_entities: set[str] = field(default_factory=set)  # drivers/teams
    ratings: dict[str, float] = field(default_factory=dict)    # item_id -> 1..5


@runtime_checkable
class Recommender(Protocol):
    """The one interface every vertical implements."""

    vertical: str

    def generate_candidates(self, start: datetime, end: datetime) -> list[Item]:
        """Stage 1 — cheap recall of items in a time window."""
        ...

    def score(self, items: list[Item], user: UserProfile) -> list[Scored]:
        """Stage 2 — excitement x personalization, with spoiler-free reasons."""
        ...

    def rank(self, scored: list[Scored]) -> list[Ranked]:
        """Order a scored candidate set into the final watch-list."""
        ...

"""Personalization layer — turns an objective excitement score into *your* score.

Deliberately thin and spoiler-free. It multiplies the objective excitement by
a taste factor built only from pre-game-safe facts (which followed team is
playing, how high the stakes are) and returns human-readable reasons that never
reveal what actually happened.

This is where the user's real ratings will plug in (Phase 4). For now the
followed-entities rules give real signal for NBA; F1 leans on stakes, since
every driver races every round (see features.py).
"""

from __future__ import annotations

from src.core.interfaces import Item, UserProfile

FOLLOWED_BOOST = 0.35   # per followed team involved
STAKES_BOOST = 0.25     # scaled by the item's stakes feature
STAKES_THRESHOLD = 0.8  # only "late-season / playoff" events earn a stakes reason


def personalize(item: Item, user: UserProfile) -> tuple[float, list[str]]:
    """Return (taste_multiplier, spoiler_free_reasons) for one item."""
    multiplier = 1.0
    reasons: list[str] = []

    involved = sorted(set(item.meta.get("entities", [])) & user.followed_entities)
    if involved:
        multiplier += FOLLOWED_BOOST * len(involved)
        reasons.append(f"features {', '.join(involved)} — you follow " +
                       ("them" if len(involved) > 1 else involved[0]))

    stakes = float(item.features.get("stakes", 0.0))
    if stakes >= STAKES_THRESHOLD:
        multiplier += STAKES_BOOST * stakes
        reasons.append("playoff game" if item.vertical == "nba"
                       else "late-season round — championship implications")

    return multiplier, reasons

"""Personalization layer — turns an objective excitement score into *your* score.

Spoiler-free throughout: it multiplies excitement by a taste factor built from
pre-game-safe facts (which followed team plays, how high the stakes are) and
returns reasons that never reveal a result.

Evidence-based, not assumed. Phase 4 evaluation against real ratings showed the
intuitive "boost your teams / boost playoffs" assumption was *refuted* for this
user (followed teams and high stakes correlated slightly negatively with their
ratings). So the boost magnitudes are calibrated from the user's own ratings
and clamped at zero — a refuted signal contributes nothing rather than
actively hurting. `calibrate()` learns them; the defaults below are the prior
used before any ratings exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.core.interfaces import Item, UserProfile

STAKES_THRESHOLD = 0.8  # only "late-season / playoff" events earn a stakes reason
MIN_SPORT_RATINGS = 8   # below this, a sport falls back to the prior (too few to fit)


@dataclass(frozen=True)
class TasteWeights:
    """How strongly followed teams and stakes move *this* user's score."""

    followed_boost: float = 0.35   # prior, before any ratings are seen
    stakes_boost: float = 0.25

    # cap so a single strong signal can't dominate excitement
    CAP: float = 0.5


DEFAULT_WEIGHTS = TasteWeights()


def personalize(item: Item, user: UserProfile,
                weights: TasteWeights = DEFAULT_WEIGHTS) -> tuple[float, list[str]]:
    """Return (taste_multiplier, spoiler_free_reasons) for one item.

    Reasons are shown even when the corresponding boost is zero — a followed
    team is useful context to display, independent of whether it moves score.
    """
    multiplier = 1.0
    reasons: list[str] = []

    involved = sorted(set(item.meta.get("entities", [])) & user.followed_entities)
    if involved:
        multiplier += weights.followed_boost * len(involved)
        reasons.append(f"features {', '.join(involved)} — you follow " +
                       ("them" if len(involved) > 1 else involved[0]))

    stakes = float(item.features.get("stakes", 0.0))
    if stakes >= STAKES_THRESHOLD:
        multiplier += weights.stakes_boost * stakes
        reasons.append("playoff game" if item.vertical == "nba"
                       else "late-season round — championship implications")

    return multiplier, reasons


def calibrate(rated: pd.DataFrame, user: UserProfile) -> TasteWeights:
    """Learn taste weights from the user's ratings; clamp refuted signals to 0.

    `rated` must have columns: entities (list), stakes, rating.
    - followed_boost scales with how much higher followed events are rated;
    - stakes_boost scales with the rank correlation between stakes and rating.
    Both are clamped to [0, CAP] so a negative (refuted) effect contributes 0.
    """
    df = rated.copy()
    df["followed"] = df["entities"].map(lambda e: bool(set(e) & user.followed_entities))

    if df["followed"].nunique() > 1:
        delta = df.loc[df.followed, "rating"].mean() - df.loc[~df.followed, "rating"].mean()
        followed = max(0.0, min(TasteWeights.CAP, 0.3 * delta))  # ~0.3 per rating point of premium
    else:
        followed = 0.0

    corr = df["stakes"].corr(df["rating"], method="spearman")
    stakes = 0.0 if pd.isna(corr) else max(0.0, min(TasteWeights.CAP, 0.5 * corr))

    # cast to plain float so weights stay JSON-serializable (corr/mean give np.float64)
    return TasteWeights(followed_boost=round(float(followed), 3),
                        stakes_boost=round(float(stakes), 3))


@dataclass(frozen=True)
class SportTaste:
    """Per-sport taste weights with a fallback for sports lacking enough ratings.

    A single learned knob across all sports is the wrong prior when a user
    watches, say, soccer for their clubs but NBA for the spectacle. This lets
    each sport carry its own weights; sports below MIN_SPORT_RATINGS fall back to
    `default` (the neutral prior) rather than borrowing another sport's verdict.
    """

    per_sport: dict[str, TasteWeights] = field(default_factory=dict)
    default: TasteWeights = DEFAULT_WEIGHTS

    def for_sport(self, sport: str) -> TasteWeights:
        return self.per_sport.get(sport, self.default)


def calibrate_per_sport(rated: pd.DataFrame, user: UserProfile) -> SportTaste:
    """Calibrate each sport on its OWN ratings; sparse sports keep the prior.

    `rated` must additionally carry a `sport` column. A sport with fewer than
    MIN_SPORT_RATINGS rated events is left to the default prior — we don't fit a
    followed/stakes effect from a handful of points (it would be noise), and we
    don't inherit another sport's (see the module docstring).
    """
    per: dict[str, TasteWeights] = {}
    for sport, grp in rated.groupby("sport"):
        if len(grp) >= MIN_SPORT_RATINGS:
            per[str(sport)] = calibrate(grp, user)
    return SportTaste(per_sport=per, default=DEFAULT_WEIGHTS)

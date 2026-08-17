"""AnimeRecommender — the media vertical, on the SAME Recommender interface.

This is the payoff of Phase 1's abstraction. It generates candidates, scores,
and ranks exactly like SportRecommender, but the engine is completely
different: content-based taste matching instead of excitement.

Mirroring the sports split so results read consistently:
  * excitement    -> community quality prior (MAL score), user-independent
  * personalization-> taste match (cosine of your rating-weighted genre profile)
  * score          -> excitement x personalization
Reasons are spoiler-free: they name the genres/themes you like, never plot.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.core.interfaces import Item, Ranked, Scored, UserProfile
from src.media.features import build_vocab, content_matrix, quality_prior

_TASTE_LO, _TASTE_HI = 0.6, 1.4   # taste multiplier range
_TIER = [(0.66, "must-watch"), (0.4, "worth it"), (0.0, "maybe")]


def _tier(score: float) -> str:
    return next(label for thresh, label in _TIER if score >= thresh)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class AnimeRecommender:
    vertical = "anime"

    def __init__(self, catalog: pd.DataFrame) -> None:
        self.df = catalog.reset_index(drop=True)
        self.vocab = build_vocab(self.df)
        self.matrix = content_matrix(self.df, self.vocab)          # titles x tags
        self.unit = np.vstack([_unit(row) for row in self.matrix]) # row-normalized
        self.quality = quality_prior(self.df)
        self._row_of = {iid: i for i, iid in enumerate(self.df["item_id"])}

    # -- taste profile from the user's ratings ----------------------------

    def _taste_vector(self, user: UserProfile) -> tuple[np.ndarray, set[str]]:
        """Rating-weighted genre/theme vector + the set of genres you like.

        Falls back to the followed genres (cold start) when no rating overlaps
        the catalog, so genre preference shapes the ranking, not just reasons.
        """
        rated = {f"anime-{k}" if not str(k).startswith("anime-") else str(k): v
                 for k, v in user.ratings.items()}
        idx, weights = [], []
        for iid, r in rated.items():
            row = self._row_of.get(iid)
            if row is not None:
                idx.append(row)
                weights.append(float(r))
        if idx:
            w = np.array(weights)
            w = w - w.mean() if len(w) > 1 else w  # center: liked>0, disliked<0
            taste = (self.matrix[idx] * w[:, None]).sum(axis=0)
            liked = {self.vocab[j] for j in np.where(taste > 0)[0]}
            return _unit(taste), liked

        # cold start: taste = the followed genres/themes as a preference vector
        vocab_idx = {t: j for j, t in enumerate(self.vocab)}
        taste = np.zeros(len(self.vocab))
        liked = set()
        for tag in user.followed_entities:
            j = vocab_idx.get(tag)
            if j is not None:
                taste[j] = 1.0
                liked.add(tag)
        return _unit(taste), liked

    # -- Recommender protocol ---------------------------------------------

    def generate_candidates(self, start: datetime, end: datetime) -> list[Item]:
        yr = self.df["year"]
        mask = (pd.to_numeric(yr, errors="coerce").fillna(0).between(start.year, end.year))
        items = []
        for _, row in self.df[mask].iterrows():
            when = datetime(int(row["year"]), 1, 1) if pd.notna(row["year"]) else start
            items.append(Item(
                item_id=row["item_id"], vertical="anime", when=when,
                features={"quality": float(self.quality[self._row_of[row["item_id"]]])},
                meta={"label": row["title"], "type": row["type"],
                      "genres": row["genres"] if isinstance(row["genres"], str) else "",
                      "themes": row["themes"] if isinstance(row["themes"], str) else ""},
            ))
        return items

    def score(self, items: list[Item], user: UserProfile) -> list[Scored]:
        taste, liked = self._taste_vector(user)
        # cold start: fall back to followed genres if no ratings overlap catalog
        if not liked and user.followed_entities:
            liked = set(user.followed_entities)
        scored = []
        for it in items:
            row = self._row_of[it.item_id]
            excitement = float(self.quality[row])
            cos = float(self.unit[row] @ taste) if taste.any() else 0.0  # [-1,1]
            taste_mult = _TASTE_LO + (_TASTE_HI - _TASTE_LO) * (cos + 1) / 2
            tags = set((it.meta.get("genres") or "").split("|")) | set((it.meta.get("themes") or "").split("|"))
            overlap = sorted(tags & liked)
            reasons = [f"{_tier(excitement * taste_mult)} · {it.meta['label']} (ANIME)"]
            if overlap:
                reasons.append("matches your taste for " + ", ".join(overlap[:3]))
            scored.append(Scored(item=it, excitement=excitement, personalization=taste_mult, reasons=reasons))
        return scored

    def rank(self, scored: list[Scored]) -> list[Ranked]:
        ordered = sorted(scored, key=lambda s: s.score, reverse=True)
        return [Ranked(rank=i + 1, scored=s) for i, s in enumerate(ordered)]

    def recommend(self, user: UserProfile, top: int = 15,
                  start: datetime | None = None, end: datetime | None = None) -> list[Ranked]:
        """Rank UNwatched catalog titles for the user."""
        s = start or datetime(1960, 1, 1)
        e = end or datetime(2100, 1, 1)
        rated = {f"anime-{k}" if not str(k).startswith("anime-") else str(k) for k in user.ratings}
        cands = [it for it in self.generate_candidates(s, e) if it.item_id not in rated]
        return self.rank(self.score(cands, user))[:top]

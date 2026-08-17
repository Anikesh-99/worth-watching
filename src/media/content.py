"""ContentRecommender — one content-based engine for every media vertical.

Anime and books are the same problem: represent each item by its tags
(genres/themes, or subjects), build a rating-weighted taste vector, and rank
unrated items by cosine similarity, tempered by a community-quality prior.
Both `AnimeRecommender` and `BookRecommender` are thin subclasses that only set
which columns hold the tags/label/quality and how to render an item.

Same `Recommender` protocol as the sports engine — different scoring, one
interface.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.core.interfaces import Item, Ranked, Scored, UserProfile
from src.media.features import build_vocab, content_matrix, quality_prior

_TASTE_LO, _TASTE_HI = 0.6, 1.4  # taste multiplier range


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class ContentRecommender:
    # --- override these per vertical ---
    vertical = "content"
    tag_cols: tuple[str, ...] = ("genres", "themes")
    title_col = "title"
    quality_col = "score"
    year_col = "year"
    id_prefix = "content-"
    tiers = [(0.66, "must-watch"), (0.4, "worth it"), (0.0, "maybe")]

    def __init__(self, catalog: pd.DataFrame) -> None:
        self.df = catalog.reset_index(drop=True)
        self.vocab = build_vocab(self.df, self.tag_cols)
        self.matrix = content_matrix(self.df, self.vocab, self.tag_cols)
        self.unit = np.vstack([_unit(row) for row in self.matrix]) if len(self.df) else self.matrix
        self.quality = quality_prior(self.df, self.quality_col)
        self._row_of = {iid: i for i, iid in enumerate(self.df["item_id"])}

    def _tier(self, score: float) -> str:
        return next(label for thresh, label in self.tiers if score >= thresh)

    def _norm_id(self, k) -> str:
        s = str(k)
        return s if s.startswith(self.id_prefix) else f"{self.id_prefix}{s}"

    def _candidate_meta(self, row: pd.Series) -> dict:
        """Display + tag fields for an Item. Subclasses add vertical-specific keys."""
        meta = {"label": row[self.title_col]}
        for c in self.tag_cols:
            meta[c] = row[c] if isinstance(row.get(c), str) else ""
        return meta

    # --- taste profile ---------------------------------------------------

    def _taste_vector(self, user: UserProfile) -> tuple[np.ndarray, set[str]]:
        rated = {self._norm_id(k): v for k, v in user.ratings.items()}
        idx, weights = [], []
        for iid, r in rated.items():
            row = self._row_of.get(iid)
            if row is not None:
                idx.append(row)
                weights.append(float(r))
        if idx:
            w = np.array(weights)
            w = w - w.mean() if len(w) > 1 else w
            taste = (self.matrix[idx] * w[:, None]).sum(axis=0)
            liked = {self.vocab[j] for j in np.where(taste > 0)[0]}
            return _unit(taste), liked

        # cold start: taste = followed tags as a preference vector
        vocab_idx = {t: j for j, t in enumerate(self.vocab)}
        taste = np.zeros(len(self.vocab))
        liked = set()
        for tag in user.followed_entities:
            j = vocab_idx.get(tag)
            if j is not None:
                taste[j] = 1.0
                liked.add(tag)
        return _unit(taste), liked

    # --- Recommender protocol -------------------------------------------

    def generate_candidates(self, start: datetime, end: datetime) -> list[Item]:
        yr = pd.to_numeric(self.df[self.year_col], errors="coerce").fillna(0)
        mask = yr.between(start.year, end.year)
        items = []
        for _, row in self.df[mask].iterrows():
            y = row[self.year_col]
            when = datetime(int(y), 1, 1) if pd.notna(y) and y else start
            items.append(Item(
                item_id=row["item_id"], vertical=self.vertical, when=when,
                features={"quality": float(self.quality[self._row_of[row["item_id"]]])},
                meta=self._candidate_meta(row),
            ))
        return items

    def score(self, items: list[Item], user: UserProfile) -> list[Scored]:
        taste, liked = self._taste_vector(user)
        scored = []
        for it in items:
            row = self._row_of[it.item_id]
            excitement = float(self.quality[row])
            cos = float(self.unit[row] @ taste) if taste.any() else 0.0
            taste_mult = _TASTE_LO + (_TASTE_HI - _TASTE_LO) * (cos + 1) / 2
            tags = set()
            for c in self.tag_cols:
                tags |= set((it.meta.get(c) or "").split("|"))
            overlap = sorted(t for t in (tags & liked) if t)
            reasons = [f"{self._tier(excitement * taste_mult)} · {it.meta['label']} ({self.vertical.upper()})"]
            if overlap:
                reasons.append("matches your taste for " + ", ".join(overlap[:3]))
            scored.append(Scored(item=it, excitement=excitement, personalization=taste_mult, reasons=reasons))
        return scored

    def rank(self, scored: list[Scored]) -> list[Ranked]:
        ordered = sorted(scored, key=lambda s: s.score, reverse=True)
        return [Ranked(rank=i + 1, scored=s) for i, s in enumerate(ordered)]

    def recommend(self, user: UserProfile, top: int = 15,
                  start: datetime | None = None, end: datetime | None = None) -> list[Ranked]:
        s = start or datetime(1400, 1, 1)
        e = end or datetime(2100, 1, 1)
        rated = {self._norm_id(k) for k in user.ratings}
        cands = [it for it in self.generate_candidates(s, e) if it.item_id not in rated]
        return self.rank(self.score(cands, user))[:top]

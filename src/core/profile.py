"""Load a UserProfile from the sport configs and the ratings CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.core.interfaces import UserProfile


def load_user_profile(config_paths: list[str], ratings_path: str = "data/my_ratings.csv") -> UserProfile:
    followed: set[str] = set()
    for cp in config_paths:
        p = Path(cp)
        if p.exists():
            cfg = yaml.safe_load(p.read_text()) or {}
            followed.update(str(e) for e in cfg.get("followed_entities", []))

    ratings: dict[str, float] = {}
    rp = Path(ratings_path)
    if rp.exists():
        r = pd.read_csv(rp)
        ratings = {str(i): float(v) for i, v in zip(r["item_id"], r["rating"])}

    return UserProfile(followed_entities=followed, ratings=ratings)

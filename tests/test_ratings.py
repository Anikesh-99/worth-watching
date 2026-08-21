"""Tests for the ratings tooling — star conversion and non-destructive merge
routing. Never touches the real data/my_*ratings.csv (uses tmp files)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]


def _load(script: str):
    spec = importlib.util.spec_from_file_location(script, _ROOT / "scripts" / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_letterboxd_star_conversion() -> None:
    s2 = _load("import_letterboxd")._stars_to_5
    assert s2(5.0) == 5 and s2(0.5) == 1 and s2(3.5) == 4 and s2(2.0) == 2
    assert s2(0) is None and s2("") is None                  # unrated -> skipped


def test_merge_upsert_adds_and_updates(tmp_path) -> None:
    upsert = _load("merge_ratings")._upsert
    f = tmp_path / "r.csv"
    pd.DataFrame({"item_id": ["a", "b"], "rating": [3, 4], "watched": [1, 1]}).to_csv(f, index=False)

    new = pd.DataFrame({"item_id": ["b", "c"], "rating": [5, 2]})   # b updates, c is new
    added, total = upsert(str(f), new, watched=True)
    assert added == 1 and total == 3                               # +1 new (c), 3 total
    out = pd.read_csv(f).set_index("item_id")
    assert out.loc["b", "rating"] == 5                            # updated, not duplicated
    assert out.loc["c", "rating"] == 2 and out.loc["c", "watched"] == 1
    assert (tmp_path / "r.csv.bak").exists()                      # backup written


def test_merge_media_upsert_no_watched_col(tmp_path) -> None:
    upsert = _load("merge_ratings")._upsert
    f = tmp_path / "movies.csv"                                    # no existing file
    new = pd.DataFrame({"item_id": ["movie-1"], "rating": [4]})
    added, total = upsert(str(f), new, watched=False)
    assert added == 1 and total == 1
    assert "watched" not in pd.read_csv(f).columns                # media ratings need no 'watched'

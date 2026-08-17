# Worth Watching?

A personalized recommendation **platform** that answers, spoiler-free:
*"What's actually worth my time tonight?"* — across live sport and anime.

Two very different recommendation engines sit behind **one `Recommender`
interface**: a sports **excitement** ranker (F1 + NBA) and a **content-based**
anime ranker. A sports match and an unwatched anime are the same problem shape —
score an item's personalized worth-your-time-ness, then rank a candidate set —
which is what makes this a platform rather than two scripts.

## Architecture

```
                      ┌─────────────────────────────┐
                      │  Recommender (Protocol)      │
                      │  generate_candidates → score │
                      │  → rank                      │
                      └───────────────┬─────────────┘
              ┌───────────────────────┼────────────────────────┐
       SportRecommender                              AnimeRecommender
   excitement × personalization                content quality × taste match
        │           │                                 │         │
   ExcitementIndex  calibrate(ratings)         genre/theme     rating-weighted
   + LightGBM       (evidence-based)           vectors         taste profile
        │                                            │
   FastF1 (F1) · ESPN (NBA)                    Jikan / MyAnimeList
        └──────────────┬─────────────────────────────┘
             shared: evaluation harness (temporal split · NDCG · MRR · Spearman)
                       UserProfile · Item · Scored · FastAPI dashboard
```

See [the design doc](../.claude/plans/wise-snuggling-corbato.md) for the full plan.

## Why this design is interesting

There is no free "I enjoyed this game" label. So the system separates two
models:

- an **objective excitement model** — labels come free from the box score
  (final margin, position changes, DNFs, stakes) and generalize across users;
- a thin **personalization layer** — your followed drivers/teams and a small
  set of your own ratings.

Ranking is framed as **learning-to-rank** and evaluated with a **temporal
split** (train on past seasons, test on the latest) — never a random split.

## Status

**Phase 1 — data foundation (done): two sport verticals, one tidy schema.**
- **F1** via [FastF1](https://docs.fastf1.dev/) (keyless, cached) → one row
  per race: DNFs, winner margin, positions moved, biggest climb.
- **NBA** via ESPN's keyless scoreboard → one row per game: final margin,
  overtime, lead changes, come-from-behind, all from per-quarter linescores.

NBA (~1,300 games/season) supplies the training volume F1's ~24 races/year
can't, and proves the ingest contract generalizes across sports.

**Phase 2 — shared excitement model (done).** One normalization layer maps
both sports onto a shared feature vocabulary (`competitiveness`, `volatility`,
`comeback`, `chaos`, `upset`, `stakes`), and a single LightGBM LambdaRank model
ranks across all 4,261 events. Evaluated with a temporal split (train ≤2023,
test 2024).

Honest result: the model is bootstrapped on a transparent weighted **index**
(no human ratings yet), so its ~1.0 agreement with that index is expected, not
a quality claim. The meaningful, non-circular finding is on NBA (large ranking
queries): naive chronological NDCG@10 **0.56** → single-feature **0.92**,
i.e. multi-signal ranking clearly helps. F1's per-month queries are too small
(1-3 races) for reliable NDCG. Cross-sport feature importance is dominated by
`competitiveness`, then `volatility`/`comeback`. Real quality evaluation
arrives in Phase 4 against the user's own ratings.

**Phase 3 — recommender + spoiler-free watch-list (done).** The two-stage
`Recommender` (candidate generation → excitement × personalization → rank) runs
across both sports in one ranked list. Personalization boosts followed teams
and high-stakes games; **reasons are spoiler-free** (they name the matchup,
the followed team, and playoff status — never the result). A placeholder
`data/my_ratings.csv` is seeded so it runs today; replace it with your own 1-5
ratings and personalization/eval use them immediately.

**Phase 4 — evaluation against real ratings (done).** Measured every ranker
against 98 of the author's own 1-5 ratings (24 F1 + 74 NBA). Honest results:

| ranker | Spearman vs your ratings |
|---|---|
| chronological (naive) | 0.18 |
| competitiveness only | 0.65 |
| **excitement index** | **0.66** |
| personalized (assumed boosts) | 0.64 |
| personalized (calibrated) | 0.66 |
| Ridge / GBM trained on ratings (OOF) | 0.62 / 0.60 |

Two findings, both kept honestly:
- **Core hypothesis validated:** objective box-score excitement predicts real
  enjoyment (Spearman 0.66); a small model trained on the ratings does *not*
  beat the transparent index at n=98.
- **Personalization hypothesis refuted for this user:** followed teams averaged
  3.0 vs 3.4, and stakes correlated -0.22 with ratings — this author rates on
  intrinsic excitement, not team loyalty. So the assumed "+35% for your team"
  boost *hurt* (0.66 -> 0.64). The fix: `personalize.calibrate()` learns the
  boosts from ratings and clamps refuted signals to zero, recovering 0.66.
  Personalization is now evidence-based, not assumed.

**Phase 5 — FastAPI backend + web dashboard (done).** A self-contained page
(no external assets) served by FastAPI shows the spoiler-free watch-list for any
date window, with a sport filter and a per-event `excite × taste = score`
breakdown. The calibration banner surfaces the Phase 4 finding to the viewer.
A **Sports / Anime mode toggle** unifies both verticals in one UI — the same
card renderer shows a sports watch-list or content-based anime recommendations.
API: `GET /api/meta`, `GET /api/watchlist?start=&end=&sport=&top=`,
`GET /api/anime?top=`.

**Phase 6 — media vertical (anime) on the same interface (done).** The payoff
of Phase 1's abstraction: `AnimeRecommender` implements the exact same
`Recommender` protocol as sports, but the engine is completely different —
**content-based taste matching** instead of excitement. Catalog via Jikan
(MyAnimeList, keyless). Mirrors the sports split so results read consistently:
`excitement` = community quality prior, `personalization` = cosine of your
rating-weighted genre/theme profile, reasons name the genres you like (never
plot). Ratings come from a template or a MAL XML export
(`scripts/import_mal.py`); `scripts/evaluate_anime.py` reuses the shared
evaluation harness. **Two different scoring engines, one interface — the
platform claim, proven.**

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/build_dataset.py configs/f1.yaml    # sports data
./.venv/bin/python scripts/build_dataset.py configs/nba.yaml
./.venv/bin/python scripts/build_dataset.py configs/anime.yaml # media catalog (Jikan)
./.venv/bin/python scripts/train_excitement.py                 # shared model + temporal eval
./.venv/bin/python scripts/evaluate.py                         # sports: eval vs your ratings
./.venv/bin/python scripts/recommend_anime.py                  # anime recommendations
./.venv/bin/python scripts/serve.py                            # dashboard -> http://127.0.0.1:8000
./.venv/bin/python -m pytest -q                                 # 31 tests, network-free + cached
```

All six phases complete. Next natural extensions: books (Goodreads CSV) and
music (Last.fm) verticals — each is a new ingest + a `Recommender` on the same
interface.

## Layout

```
src/core/interfaces.py     # Recommender protocol shared by every vertical
src/core/features.py       # per-sport -> unified excitement feature vocabulary
src/core/excitement.py     # transparent index + shared LightGBM LambdaRank model
src/core/evaluation.py     # temporal split + NDCG / MRR / Spearman
src/core/profile.py        # load followed entities + ratings -> UserProfile
src/sports/ingest.py       # F1: FastF1 -> tidy per-race dataframe
src/sports/nba_ingest.py   # NBA: ESPN scoreboard -> tidy per-game dataframe
src/sports/personalize.py  # spoiler-free taste multiplier + reasons
src/sports/recommender.py  # SportRecommender: the two-stage Recommender
configs/{f1,nba}.yaml      # sport-specific config (seasons, followed entities)
scripts/build_dataset.py   # dispatches on config `vertical`
scripts/train_excitement.py# trains + evaluates the shared model
scripts/make_ratings_template.py # human-readable ratings template to fill
scripts/evaluate.py        # eval every ranker against your real ratings
scripts/watchlist.py       # end-to-end spoiler-free watch-list (CLI)
src/media/anime_ingest.py  # anime: Jikan catalog -> tidy content table
src/media/features.py      # genre/theme content vectors + quality prior
src/media/recommender.py   # AnimeRecommender: content-based, same interface
src/serving/service.py     # loads data + calibrated weights, answers queries
src/serving/app.py         # FastAPI: /api/meta, /api/watchlist, dashboard
web/index.html             # self-contained dashboard (no external assets)
scripts/serve.py           # launch the dashboard
scripts/recommend_anime.py # media watch-list; import_mal.py imports your list
tests/                     # 31 tests across sports + media verticals
```

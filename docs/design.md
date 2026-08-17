# "Worth Watching?" — A Personalized Recommendation Platform (Sports flagship + Media vertical)

## Context

The user is building a portfolio project to demonstrate **recommendation-systems (recsys)**
competence for their resume. Two hard constraints drove the design: (1) **easy, personal data
access**, and (2) **the project must scratch a real personal itch / align with a hobby** — so it
reads as a system the user actually uses, not a recycled MovieLens tutorial.

Chosen direction: a **sports-match recommender as the flagship**, deliberately architected so a
**cross-media recommender (anime / books / film)** can plug in later as a second vertical behind
one shared interface. This gives the resume a "platform," not "two projects in a repo."

**Core problem it solves for the user:** limited free time + a flood of F1 races / soccer /
NBA games → "which of this weekend's events is actually worth my two hours?", answered
**spoiler-free**, and a "should I catch up on this one I missed?" mode.

**Key decisions already made with the user:**
- **Label strategy: Hybrid.** An objective *excitement* model trained on free box-score labels
  (closeness, position changes, upsets, stakes) + a thin *personalization* layer tuned on ~30–50
  events the user rates themselves.
- **Delivery surface:** a **weekly ranked watch-list + a browsable web dashboard**.
- **Sport scope:** **F1 first** (cleanest single data source, the sport the user follows most),
  but the pipeline is sport-agnostic so NBA/soccer are configuration, not rewrites.

## Goals / Non-Goals

**Goals**
- Two-stage recsys pipeline (candidate generation → learning-to-rank) behind a clean, reusable
  `Recommender` interface.
- Honest, temporal offline evaluation (NDCG@k, MRR, Spearman) — no random splits.
- A spoiler-free weekly watch-list and a web dashboard the user genuinely uses.
- Extensibility proven by design: the media vertical implements the same interface.

**Non-Goals (YAGNI)**
- No real-time streaming / live in-play updates. Weekly batch is enough.
- No user accounts / multi-user system — single user (you).
- Media vertical (anime/books/film) is **designed for, not built in, v1**. It is Phase 6.
- No mobile app; the dashboard is web-only.

## Architecture

**The one abstraction that makes it a platform** — every vertical implements:

```
class Recommender(Protocol):
    def generate_candidates(self, when) -> list[Item]      # stage 1: cheap recall
    def score(self, items, user_profile) -> list[Scored]   # stage 2: excitement x personalization
    def rank(self, scored) -> list[Ranked]                 # spoiler-free ordering + reasons
```

A **sports match** and an **unwatched anime** are the same shape of problem: score an item's
personalized worth-your-time-ness, then rank a candidate set. The sports vertical proves the hard
part (novel domain, manufactured labels); the media vertical proves generality.

**Two-stage pipeline**
1. **Candidate generation** — recall the set of relevant events for a window (e.g. this weekend's
   races / recent races you missed). Trivial for F1 (~24/yr); becomes load-bearing for media.
2. **Ranking** — `excitement_score(item)` × `personalization(item, you)` → **LambdaMART
   (LightGBM `lambdarank`)** optimizing NDCG directly, with human-readable, **non-spoiler**
   reasons ("championship implications for a driver you follow", "historically close on this
   circuit") — never the result.

**Two separable models (the interview centerpiece)**
- **Excitement model** — generalizes across all users. Labels are *free* from box scores:
  final margin / gap, number of position changes & overtakes, lead changes, DNFs, safety cars,
  qualifying-to-race spread, championship stakes at that round.
- **Personalization layer** — thin, yours. Followed drivers/constructors, rivalry, whether the
  round matters for a favorite's title fight; tuned/validated on your ~30–50 self-rated events.

## Data Sources (all free, no paid auth)

- **FastF1** (`pip install fastf1`) — primary. Lap times, position changes, pit stops, gaps,
  telemetry, results for 2018+, with local caching. This is the "easy data access" win and the
  source of excitement features. **Historical training data comes from here.**
- **Jolpica-F1** (`api.jolpi.ca/ergast/…`) — drop-in mirror of the deprecated Ergast REST API,
  for lightweight schedule/standings/live lookups.
- **Your ratings** — a small CSV (`data/my_ratings.csv`: `event_id, rating_1_5, watched`) you
  fill for ~30–50 memorable/forgettable races. This is the personalization + eval ground truth.
- *(Phase 6, media vertical, not v1):* Jikan API (MyAnimeList, no auth), Goodreads CSV export,
  Letterboxd export.

## Components / Modules

Proposed repo layout (Python, `src/` package):

- `src/core/interfaces.py` — `Recommender` protocol, `Item`, `Scored`, `Ranked`, `UserProfile`.
- `src/core/evaluation.py` — temporal split + NDCG@k, MRR, Spearman; reused by every vertical.
- `src/sports/ingest.py` — FastF1 / Jolpica fetch + cache → tidy per-event dataframe.
- `src/sports/features.py` — excitement feature engineering (margins, position changes, DNFs,
  stakes, circuit history).
- `src/sports/excitement_model.py` — LightGBM regressor/ranker on free box-score labels.
- `src/sports/personalize.py` — followed-entity + rivalry + title-stakes adjustments.
- `src/sports/recommender.py` — implements `Recommender` for F1 (wires the two stages).
- `src/serving/watchlist.py` — builds the weekly spoiler-free ranked list + reason strings.
- `src/serving/app.py` — **FastAPI** backend serving JSON (`/watchlist`, `/event/{id}`).
- `web/` — lightweight dashboard (static HTML/JS or a small React app) consuming the API.
  *(Streamlit is an acceptable faster alternative if the API+frontend proves heavy — decide at
  Phase 5; note the tradeoff in the README.)*
- `configs/f1.yaml` — sport-specific config (series, seasons, followed entities) proving the
  pipeline is config-driven, not F1-hardcoded.
- `notebooks/` — EDA + model-selection story for the write-up.
- `README.md` — architecture, **honest results** (including anything that didn't work), and the
  resume talking points.

## Evaluation

- **Temporal split:** train on seasons ≤ 2023, validate 2024, test 2025. State this loudly.
- **Excitement model:** NDCG@k and Spearman vs. objective excitement labels; feature-importance
  plot (which factors predict a thriller).
- **Personalization / end-to-end:** NDCG@k and MRR against your held-out self-ratings — does the
  personalized ranking put the races *you* loved near the top?
- **Baselines to beat:** chronological order, and popularity/"marquee race" ordering. Beating a
  sensible baseline is the honest bar; report it even if narrow.

## Build Phases (milestones, each independently demoable)

1. **Ingest + cache** F1 data via FastF1; tidy per-event dataframe; `configs/f1.yaml`.
2. **Excitement features + model**; feature-importance write-up; temporal eval vs. baseline.
3. **`Recommender` interface + two-stage wiring**; personalization layer; your ratings CSV.
4. **End-to-end evaluation harness** (NDCG/MRR/Spearman) reused across stages.
5. **Weekly watch-list + FastAPI + dashboard**; spoiler-free reason strings.
6. *(Stretch / proves generality)* **Media vertical** implementing the same interface (Jikan +
   Goodreads/Letterboxd), sharing `core/evaluation.py`.

## Verification (how to know it works end-to-end)

- `pytest` unit tests on feature engineering (label correctness on a known thriller vs. blowout)
  and on the `Recommender` interface contract.
- Run the eval harness: confirm the excitement model **beats the chronological/popularity
  baseline** on NDCG@k on the held-out 2025 season, and that personalized ranking beats
  un-personalized on your held-out ratings.
- Launch the FastAPI app; hit `/watchlist` and confirm it returns a ranked, **spoiler-free** list
  with reason strings; open the dashboard and eyeball a real upcoming/recent weekend.
- Sanity check by hand: a famously chaotic race (e.g. a wet, many-DNF, late-lead-change GP) should
  rank far above a lights-to-flag procession.

## Resume talking points (why this reads as senior)

- Two-stage candidate-gen → **learning-to-rank** pipeline behind a reusable interface.
- **Manufacturing a training signal where none exists** (free objective labels + thin personal
  layer) — a real ML-systems design decision, not a Kaggle download.
- **Temporal evaluation** and beating honest baselines; feature-importance interpretability.
- A **platform** framing proven by a second vertical on the same interface.

## Future Work

- Build the media vertical (Phase 6) into a real second product.
- Add NBA/soccer via new `configs/*.yaml` + ingest adapters (pipeline already sport-agnostic).
- Scheduled job that emails the weekly watch-list ("it runs itself" story).

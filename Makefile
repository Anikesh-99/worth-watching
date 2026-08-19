# Worth Watching? — one-command reproduction.
# Usage: make setup && make data && make train && make test && make serve
PY := ./.venv/bin/python

.PHONY: setup data data-sports data-media train eval serve test static update clean

setup:                ## create venv + install deps
	python3 -m venv .venv
	./.venv/bin/pip install -q --upgrade pip
	./.venv/bin/pip install -q -r requirements.txt

data: data-sports data-media  ## build every dataset

data-sports:
	$(PY) scripts/build_dataset.py configs/f1.yaml
	$(PY) scripts/build_dataset.py configs/nba.yaml
	$(PY) scripts/build_dataset.py configs/soccer.yaml

data-media:
	$(PY) scripts/build_dataset.py configs/anime.yaml
	$(PY) scripts/build_dataset.py configs/books.yaml

train:                ## train + temporally evaluate the shared excitement model
	$(PY) scripts/train_excitement.py

eval:                 ## evaluate rankers against your ratings (needs data/my_ratings.csv)
	$(PY) scripts/evaluate.py

anime:                ## anime recommendations (cold-start until you add ratings)
	$(PY) scripts/recommend_anime.py

serve:                ## launch the dashboard at http://127.0.0.1:8000
	$(PY) scripts/serve.py

test:                 ## run the test suite (network-free + cached)
	$(PY) -m pytest -q

static:               ## rebuild the public demo bundle (images + live upcoming)
	$(PY) scripts/build_static.py

update:               ## GROW: pull new results, re-sync music, rebuild the demo
	-$(PY) scripts/build_dataset.py configs/soccer.yaml
	-$(PY) scripts/build_dataset.py configs/nba.yaml
	-$(PY) scripts/build_dataset.py configs/f1.yaml
	-$(PY) scripts/build_music.py
	$(PY) scripts/build_static.py
	@echo ""
	@echo "Updated. New sports ratings recalibrate on next 'make serve'."
	@echo "Anime/books: re-run import_anilist.py / import_goodreads.py to re-sync."
	@echo "Commit + push to deploy the refreshed demo (or let the daily cron do it)."

clean:                ## remove derived data (keeps caches)
	rm -f data/*.csv models/*.joblib

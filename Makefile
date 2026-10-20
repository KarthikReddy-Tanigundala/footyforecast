# FootyForecast task runner (POSIX make). Windows users: see make.ps1.
PY ?= python
RSCRIPT ?= Rscript

.PHONY: help setup r-setup extract refresh build tune features ml ml-tune archetypes explain backtest predict app test lint report all

help:      ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

setup:     ## Create venv + install Python deps (editable)
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install -e .

r-setup:   ## Initialise renv + pinned R deps
	$(RSCRIPT) R/setup_renv.R

extract:   ## fitzRoy pulls -> data/raw (cached)
	$(RSCRIPT) R/extract_data.R

refresh:   ## In-season incremental extract
	$(RSCRIPT) R/extract_data.R --incremental

build:     ## Normalise raw extracts -> data/processed
	$(PY) -m footyforecast.io

tune:      ## Grid-search Elo on 2000-2016 + walk-forward re-tuning
	$(PY) -m footyforecast.elo --mode flat && $(PY) -m footyforecast.elo --mode venue && $(PY) -m footyforecast.elo --mode walkforward

features:  ## Pre-match feature table
	$(PY) -m footyforecast.features

ml:        ## Walk-forward ML predictions 2014-2026
	$(PY) -m footyforecast.model_match

ml-tune:   ## Re-tune ML on 2014-2016, then walk-forward
	$(PY) -m footyforecast.model_match --tune

archetypes: ## Player clustering + UMAP (needs raw player stats)
	$(PY) -m footyforecast.clustering

explain:   ## Global SHAP plots
	$(PY) -m footyforecast.explain

backtest:  ## Walk-forward 2017-2026 -> results/leaderboard.md
	$(PY) -m footyforecast.evaluation.backtest

predict:   ## Next-round predictions + predictions/latest.json
	$(PY) -m footyforecast.predict

app:       ## Run the Streamlit dashboard
	$(PY) -m streamlit run app/streamlit_app.py

test:      ## pytest
	$(PY) -m pytest

lint:      ## ruff check + format check
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

report:    ## Render the R home-ground-advantage report
	quarto render R/reports/home_ground_advantage.qmd --output-dir ../../results

all: build features backtest predict ## Build -> features -> backtest -> predict

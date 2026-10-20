"""Cached loaders for the dashboard. The app only reads committed artefacts, never the APIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
# Streamlit Cloud runs from a plain checkout (no `pip install -e .`), so expose src/ directly.
sys.path.insert(0, str(ROOT / "src"))

from footyforecast.evaluation import metrics  # noqa: E402

RESULTS = ROOT / "results"
PREDICTIONS = ROOT / "predictions"


@st.cache_data(show_spinner=False)
def backtest_summary() -> dict:
    return json.loads((RESULTS / "backtest_results.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def backtest_games() -> pd.DataFrame:
    return pd.read_parquet(RESULTS / "backtest_predictions.parquet")


@st.cache_data(show_spinner=False)
def latest_feed() -> dict:
    return json.loads((PREDICTIONS / "latest.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def calibration_tables(ours: str, n_bins: int = 10) -> dict[str, pd.DataFrame]:
    games = backtest_games().dropna(subset=["outcome", "p_ours"])
    out = {ours: metrics.calibration_table(games["p_ours"], games["outcome"], n_bins)}
    market = games.dropna(subset=["p_punters"])
    out["Punters"] = metrics.calibration_table(market["p_punters"], market["outcome"], n_bins)
    return out


@st.cache_data(show_spinner=False)
def player_seasons() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / "player_seasons.parquet")


@st.cache_data(show_spinner=False)
def archetypes() -> dict:
    return json.loads((RESULTS / "archetypes.json").read_text(encoding="utf-8"))

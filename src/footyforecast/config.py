"""Central configuration: every tunable lives here, never hardcoded in modules.

Season windows follow the errata in CLAUDE.md §17:
    * Elo warms up from 1990 so ratings are settled long before evaluation.
    * Hyperparameters are tuned on 2000–2016 ONLY.
    * Everything is evaluated walk-forward on 2017–2026 (the Squiggle era).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_REFERENCE = ROOT / "data" / "reference"
RESULTS = ROOT / "results"
PREDICTIONS = ROOT / "predictions"
MODELS = ROOT / "models"

# ---- Season windows ------------------------------------------------------------
ELO_HISTORY_START = 1990
ML_HISTORY_START = 2012
TUNE_SEASONS = tuple(range(2000, 2017))
EVAL_SEASONS = tuple(range(2017, 2027))
# Before each season, hyperparameters are re-tuned on the trailing N completed seasons,
# the same process the live model follows each pre-season. A rolling (not expanding)
# window lets the system track a changing game (6-6-6 in 2019, the stand rule in 2021).
TUNING_WINDOW_SEASONS = 10
LAST_COMPLETED_SEASON = 2026
NEXT_SEASON = 2027

# ---- Season quirks -------------------------------------------------------------
# 2020 was played with 16-minute quarters (80% of normal length), so raw margins
# understate dominance. Scale them up for the Elo update; report MAE unscaled.
SEASON_MARGIN_SCALE: dict[int, float] = {2020: 1.25}


# ---- Elo -------------------------------------------------------------------------
@dataclass(frozen=True)
class EloParams:
    """Elo hyperparameters, chosen by `python -m footyforecast.elo` on 2000–2016 only.

    Tuning notes (results/elo_tuning_*.csv): the log-loss surface is flat in carryover
    between 0.9 and 1.0 (0.5851 vs 0.5847 over 3,266 games, well inside noise), so 0.9 is
    kept. It is a mild between-season regression that reflects real list turnover at no
    measurable cost. Venue-aware HGA beats the best flat HGA (log loss 0.5851 vs 0.5914,
    +29 bits over the tuning seasons).
    """

    initial: float = 1500.0
    k: float = 100.0
    margin_cap: float = 80.0  # margins mapped linearly onto [0, 1] over [-cap, +cap]
    carryover: float = 0.9  # share of a team's deviation from the league mean kept between seasons
    # Home-ground advantage, in Elo points.
    hga_mode: str = "venue"  # "flat" or "venue"
    hga_flat: float = 60.0  # flat mode: every nominal home team gets this (best flat setting)
    hga_base: float = 10.0  # venue mode: nominal-home edge (crowd share, rooms, scheduling)
    hga_familiarity_w: float = 20.0  # × (log1p(home games at venue) − log1p(away games at venue))
    hga_travel_w: float = 8.0  # × (log1p(away km travelled) − log1p(home km travelled))
    familiarity_window_days: int = 730
    points_per_elo: float = 0.1619  # Elo-diff → points-margin slope, fitted on the tuning seasons


ELO = EloParams()
# Ablation: the best flat-HGA system, kept to show what venue awareness buys.
ELO_FLAT = EloParams(hga_mode="flat", k=100.0, carryover=0.9, hga_flat=60.0, points_per_elo=0.1679)

# Expansion clubs built from draft concessions entered far below average. Clubs not
# listed start at EloParams.initial (Adelaide 1991, Fremantle 1995 and Port Adelaide
# 1997 were competitive from day one).
EXPANSION_INITIAL: dict[str, float] = {"Gold Coast": 1300.0, "Greater Western Sydney": 1300.0}

# ---- Evaluation -----------------------------------------------------------------
DRAW_IS_CORRECT_TIP = True  # Squiggle/tipping-comp convention: everyone gets a draw
CALIBRATION_BINS = 10

# ---- Squiggle etiquette -----------------------------------------------------------
REPO_URL = "https://github.com/KarthikReddy-Tanigundala/footyforecast"


def squiggle_user_agent() -> str:
    """User-Agent for Squiggle requests: identifies the bot and a contact email (from env)."""
    contact = os.getenv("FOOTY_CONTACT_EMAIL", "").strip()
    suffix = f"; {contact}" if contact else ""
    return f"FootyForecast/0.1 (+{REPO_URL}{suffix})"

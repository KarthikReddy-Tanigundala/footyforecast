"""Scoring metrics, matching Squiggle's leaderboard definitions.

Conventions used throughout the project:
    * Every probability is P(home team wins), written ``p_home``.
    * ``outcome`` is 1 for a home win, 0 for an away win and 0.5 for a draw.
    * Margins are signed from the home team's perspective (home − away points).

Bits comes from Monash University's Probabilistic Footy Tipping competition:
    correct tip:  1 + log2(p)
    wrong tip:    1 + log2(1 − p)
    draw:         1 + 0.5 · log2(p · (1 − p))
where p is the probability given to the tipped team. Written with ``p_home`` it is
1 + log2(p_home) for a home win, 1 + log2(1 − p_home) for an away win, and the
draw formula unchanged. Higher is better; a coin-flip tipper scores 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footyforecast.config import CALIBRATION_BINS, DRAW_IS_CORRECT_TIP

# Probabilities of exactly 0 or 1 would score −inf bits; clip to a hair inside.
_EPS = 1e-3


def _clip(p_home: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(p_home, dtype=float), _EPS, 1 - _EPS)


def bits(p_home: np.ndarray | pd.Series, outcome: np.ndarray | pd.Series) -> np.ndarray:
    """Per-game bits for home-win probabilities against outcomes (1 / 0 / 0.5)."""
    p = _clip(p_home)
    y = np.asarray(outcome, dtype=float)
    return np.where(
        y == 0.5,
        1 + 0.5 * np.log2(p * (1 - p)),
        np.where(y == 1, 1 + np.log2(p), 1 + np.log2(1 - p)),
    )


def tips_correct(
    p_home: np.ndarray | pd.Series,
    outcome: np.ndarray | pd.Series,
    draw_is_correct: bool = DRAW_IS_CORRECT_TIP,
) -> np.ndarray:
    """1 where the tip was right. A probability of exactly 0.5 tips the home team."""
    tip_home = np.asarray(p_home, dtype=float) >= 0.5
    y = np.asarray(outcome, dtype=float)
    correct = np.where(y == 1, tip_home, ~tip_home)
    return np.where(y == 0.5, draw_is_correct, correct).astype(int)


def margin_abs_error(
    pred_margin: np.ndarray | pd.Series, actual_margin: np.ndarray | pd.Series
) -> np.ndarray:
    """Absolute error between predicted and actual home margins, in points."""
    return np.abs(np.asarray(pred_margin, dtype=float) - np.asarray(actual_margin, dtype=float))


def log_loss(p_home: np.ndarray | pd.Series, outcome: np.ndarray | pd.Series) -> float:
    """Mean binary cross-entropy (natural log). Draws count as y = 0.5."""
    p = _clip(p_home)
    y = np.asarray(outcome, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def brier(p_home: np.ndarray | pd.Series, outcome: np.ndarray | pd.Series) -> float:
    """Mean squared error of the probability. Draws count as y = 0.5."""
    p = np.asarray(p_home, dtype=float)
    y = np.asarray(outcome, dtype=float)
    return float(np.mean((p - y) ** 2))


def calibration_table(
    p_home: np.ndarray | pd.Series,
    outcome: np.ndarray | pd.Series,
    n_bins: int = CALIBRATION_BINS,
) -> pd.DataFrame:
    """Equal-width reliability bins: mean predicted vs observed home-win rate (draws excluded)."""
    p = np.asarray(p_home, dtype=float)
    y = np.asarray(outcome, dtype=float)
    keep = y != 0.5
    p, y = p[keep], y[keep]
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    frame = pd.DataFrame({"bin": idx, "p": p, "y": y})
    table = (
        frame.groupby("bin")
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reindex(range(n_bins))
        .assign(bin_lower=edges[:-1], bin_upper=edges[1:])
    )
    table["n"] = table["n"].fillna(0).astype(int)
    return table.reset_index(drop=True)


def expected_calibration_error(
    p_home: np.ndarray | pd.Series,
    outcome: np.ndarray | pd.Series,
    n_bins: int = CALIBRATION_BINS,
) -> float:
    """Count-weighted mean |predicted − observed| across reliability bins."""
    table = calibration_table(p_home, outcome, n_bins).dropna(subset=["mean_predicted"])
    if table["n"].sum() == 0:
        return float("nan")
    gap = (table["mean_predicted"] - table["observed_rate"]).abs()
    return float(np.average(gap, weights=table["n"]))


def summarise(
    p_home: np.ndarray | pd.Series,
    outcome: np.ndarray | pd.Series,
    pred_margin: np.ndarray | pd.Series | None = None,
    actual_margin: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Squiggle-style scorecard for a set of predictions."""
    p = np.asarray(p_home, dtype=float)
    y = np.asarray(outcome, dtype=float)
    out: dict[str, float] = {
        "games": int(len(p)),
        "tips": int(tips_correct(p, y).sum()),
        "accuracy": float(tips_correct(p, y).mean()) if len(p) else float("nan"),
        "bits": float(bits(p, y).sum()),
        "bits_per_game": float(bits(p, y).mean()) if len(p) else float("nan"),
        "log_loss": log_loss(p, y) if len(p) else float("nan"),
        "brier": brier(p, y) if len(p) else float("nan"),
        "ece": expected_calibration_error(p, y),
    }
    if pred_margin is not None and actual_margin is not None:
        err = margin_abs_error(pred_margin, actual_margin)
        out["mae"] = float(np.nanmean(err)) if np.isfinite(err).any() else float("nan")
    return out

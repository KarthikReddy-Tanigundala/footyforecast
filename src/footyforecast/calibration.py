"""Probability calibration (Platt / isotonic), walk-forward, plus reliability diagrams.

For evaluation season S the calibrator is fitted on the out-of-sample predictions of the
previous ``CAL_WINDOW`` seasons (S−3 … S−1). Each of those was made by a model trained
before its own round, so the calibrator never sees season S or any in-sample fit.

Platt scaling (a 1-feature logistic regression on logit p) is the default: with ~600
calibration games per window it is far more stable than isotonic regression's step
function, which is kept as an ablation.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from footyforecast import config  # noqa: E402
from footyforecast.evaluation import metrics  # noqa: E402

CAL_WINDOW = 3  # seasons
_EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_platt(p: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """σ(a·logit(p) + b). a > 1 sharpens an under-confident model, a < 1 softens it."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(
        _logit(p).reshape(-1, 1), np.asarray(y, dtype=int)
    )
    return lambda q: lr.predict_proba(_logit(q).reshape(-1, 1))[:, 1]


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p, y)
    return lambda q: iso.predict(np.asarray(q, dtype=float))


METHODS = {"platt": fit_platt, "isotonic": fit_isotonic}


def calibrate_walk_forward(
    preds: pd.DataFrame,
    p_col: str,
    method: str = "platt",
    seasons: tuple[int, ...] = config.EVAL_SEASONS,
    window: int = CAL_WINDOW,
) -> pd.Series:
    """Calibrated probabilities for ``seasons``; each season's map is fitted on earlier seasons only.

    ``preds`` needs game_id, season, outcome and ``p_col``. Draws are left out of the fit.
    Returns a Series indexed like ``preds`` (NaN outside ``seasons``).
    """
    out = pd.Series(np.nan, index=preds.index)
    for s, mapper in walk_forward_mappers(preds, p_col, method, seasons, window).items():
        rows = preds["season"] == s
        out[rows] = mapper(preds.loc[rows, p_col].to_numpy())
    return out


def walk_forward_mappers(
    preds: pd.DataFrame,
    p_col: str,
    method: str = "platt",
    seasons: tuple[int, ...] = config.EVAL_SEASONS,
    window: int = CAL_WINDOW,
) -> dict[int, Callable[[np.ndarray], np.ndarray]]:
    """The calibration map used for each season, fitted on the previous ``window`` seasons."""
    fit = METHODS[method]
    mappers = {}
    for s in seasons:
        cal = preds[preds["season"].between(s - window, s - 1) & preds["outcome"].isin([0.0, 1.0])]
        cal = cal.dropna(subset=[p_col])
        if len(cal) < 100:
            raise ValueError(f"only {len(cal)} calibration games before {s} for {p_col}")
        mappers[s] = fit(cal[p_col].to_numpy(), cal["outcome"].to_numpy())
    return mappers


def reliability_figure(
    panels: dict[str, dict[str, pd.Series]],
    outcome: pd.Series,
    path=config.RESULTS / "calibration_plot.png",
) -> pd.DataFrame:
    """Side-by-side reliability diagrams (before vs after) with Brier scores in the legend."""
    colors = {"before": "#898781", "after": "#2a78d6"}
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5), sharey=True)
    axes = np.atleast_1d(axes)
    summary = []
    for ax, (title, series) in zip(axes, panels.items(), strict=True):
        ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, zorder=0)
        for label, p in series.items():
            mask = p.notna() & outcome.notna()
            table = metrics.calibration_table(p[mask], outcome[mask]).dropna(
                subset=["mean_predicted"]
            )
            table = table[table["n"] >= 10]
            brier = metrics.brier(p[mask], outcome[mask])
            ece = metrics.expected_calibration_error(p[mask], outcome[mask])
            summary.append({"model": title, "version": label, "brier": brier, "ece": ece})
            ax.plot(
                table["mean_predicted"], table["observed_rate"], marker="o", ms=6, lw=2,
                color=colors.get(label, "#2a78d6"), mec="white", mew=1.5,
                label=f"{label}: Brier {brier:.4f}, ECE {ece:.3f}",
            )  # fmt: skip
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted home-win probability")
        ax.grid(color="#e1e0d9", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].set_ylabel("Observed home-win rate")
    fig.suptitle(
        f"Calibration, walk-forward {config.EVAL_SEASONS[0]}–{config.EVAL_SEASONS[-1]} (draws excluded)",
        x=0.01, ha="left", fontsize=13,
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return pd.DataFrame(summary)

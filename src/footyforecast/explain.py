"""Plain-English reasons for each tip, plus global SHAP plots.

Per-game reasons
----------------
The shipped model is a logistic regression, so its SHAP values are exact and cheap:
φ_j = β_j · z_j in log-odds (see ``model_match.logit_contributions``). The 29 features are
grouped into footy reasons (Elo rating, home ground and travel, recent form, selected 22, …).
Each reason is quantified as a counterfactual:

    Δ = P(favourite) − P(favourite if this reason were neutral)

i.e. the reason's log-odds contribution is removed and the result passed back through the
season's calibration map. Δ is in percentage points and reads naturally ("+18 pts from a
higher Elo rating"). It isn't additive across reasons, and the text never claims it is.

Global plots (``python -m footyforecast.explain``)
----------------------------------------------------
* SHAP beeswarm and Elo-gap dependence plot for the shipped logistic model (LinearExplainer).
* Mean |SHAP| for the logistic vs the XGBoost ablation (TreeExplainer; XGBoost's binary
  output is a single log-odds array, unlike sklearn classifiers' per-class arrays), as a
  check that two very different models lean on the same evidence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from footyforecast import config
from footyforecast.features import FEATURE_LABELS, FEATURES, FORM_STATS

logger = logging.getLogger(__name__)

REASON_GROUPS: dict[str, list[str]] = {
    "elo": ["rating_diff"],
    "venue": ["hga_elo", "fam_diff", "travel_diff", "interstate_home", "interstate_away"],
    "form": ["win5_diff", "margin5_diff", "margin3_diff"],
    "ladder": ["rank_diff", "pct_log_ratio"],
    "selection": [
        "sel_rating_diff",
        "sel_changes_home",
        "sel_changes_away",
        "sel_inexperienced_diff",
    ],
    "dominance": [f"{s}_form_diff" for s in FORM_STATS],
    "rest": ["rest_home", "rest_away", "short_break_home", "short_break_away"],
    "h2h": ["h2h_win_rate", "h2h_margin"],
    "finals": ["is_final"],
}
assert sorted(sum(REASON_GROUPS.values(), [])) == sorted(FEATURES), (
    "every feature needs a reason group"
)

STAT_WORDS = {
    "contested_possessions": "contested ball",
    "clearances": "clearances",
    "inside_fifties": "inside-50s",
    "metres_gained": "metres gained",
    "intercepts": "intercepts",
    "tackles": "tackles",
    "rating_points": "player ratings",
}
MIN_PTS = 1.0  # reasons moving the tip by less than 1 percentage point are not mentioned


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _poss(team: str) -> str:
    """Possessive: "Geelong's", "Brisbane Lions'"."""
    return f"{team}'" if team.endswith("s") else f"{team}'s"


def _ordinal(n: float) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


ASPECTS = {
    "elo": "Elo rating",
    "venue": "home-ground and travel",
    "form": "recent form",
    "ladder": "ladder position",
    "selection": "selected-22",
    "dominance": "recent stat dominance",
    "rest": "rest",
    "h2h": "head-to-head",
    "finals": "finals",
}


def _data_direction(group: str, g: pd.Series) -> int:
    """Which side the raw data favours on this reason: +1 home, −1 away, 0 neither."""
    if group == "dominance":
        stat = max(FORM_STATS, key=lambda s: abs(g[f"phi_{s}_form_diff"]))
        value = g[f"{stat}_form_diff"]
    else:
        value = {
            "elo": g["rating_diff"],
            "venue": g["hga_elo"],
            "form": g["margin5_diff"],
            "ladder": g["rank_diff"],
            "selection": g["sel_rating_diff"],
            "rest": (g["rest_home"] or 0) - (g["rest_away"] or 0),
            "h2h": g["h2h_margin"],
            "finals": 0.0,
        }[group]
    return 0 if pd.isna(value) or value == 0 else int(np.sign(value))


def _phrase(group: str, helps_home: bool, g: pd.Series) -> str:
    """Describe a reason factually.

    The text names the team the *data* favours. When the model's contribution points the other
    way (correlated features, e.g. recent margins once the Elo rating already counts them), it
    says the edge is discounted rather than inventing a reason.
    """
    data_dir = _data_direction(group, g)
    model_dir = 1 if helps_home else -1
    home, away = g["home_team"], g["away_team"]
    if data_dir != 0 and data_dir != model_dir:
        edge_team = home if data_dir > 0 else away
        why = (
            "already priced into the Elo rating"
            if group in {"form", "dominance", "ladder", "h2h"}
            else "outweighed by related factors"
        )
        return f"{_poss(edge_team)} {ASPECTS[group]} edge discounted ({why})"

    team, other = (home, away) if helps_home else (away, home)
    side, oside = ("home", "away") if helps_home else ("away", "home")
    if group == "elo":
        return f"{_poss(team)} higher Elo rating ({g[f'elo_{side}']:.0f} v {g[f'elo_{oside}']:.0f})"
    if group == "venue":
        if helps_home:
            km = g["travel_away_km"]
            trip = f", {other} travelling {km:,.0f} km" if km >= 300 else ""
            return f"home ground at {g['venue']}{trip}"
        return f"{team} more at home at {g['venue']}"
    if group == "form":
        return f"{_poss(team)} better recent form ({abs(g['margin5_diff']):.0f}-point margin edge over the last 5)"
    if group == "ladder":
        return f"{team} higher on the ladder ({_ordinal(g[f'{side}_rank'])} v {_ordinal(g[f'{oside}_rank'])})"
    if group == "selection":
        changes = g.get(f"sel_changes_{oside}")
        extra = (
            f" ({other} made {changes:.0f} changes)" if pd.notna(changes) and changes >= 3 else ""
        )
        return f"{_poss(team)} stronger selected 22{extra}"
    if group == "dominance":
        stat = max(FORM_STATS, key=lambda s: abs(g[f"phi_{s}_form_diff"]))
        return f"{team} winning the {STAT_WORDS[stat]} lately"
    if group == "rest":
        rest_other = g.get(f"rest_{oside}")
        if pd.notna(rest_other) and rest_other <= 6:
            return f"{other} backing up on a {rest_other:.0f}-day break"
        return f"{team} better rested"
    if group == "h2h":
        return f"{_poss(team)} recent head-to-head record against {other}"
    return f"finals football ({team})"


def explain_games(
    games: pd.DataFrame, mappers: dict[int, Callable[[np.ndarray], np.ndarray]], top: int = 3
) -> pd.DataFrame:
    """Reasons for each game.

    ``games`` needs: game_id, season, home_team, away_team, venue, elo_home, elo_away,
    travel_away_km, home_rank, away_rank, the feature columns, and phi_base + phi_<feature>.
    ``mappers`` maps season → calibration function (identity if a season is missing).
    """
    rows = []
    for _, g in games.iterrows():
        mapper = mappers.get(int(g["season"]), lambda p: p)
        logit = g["phi_base"] + sum(g[f"phi_{f}"] for f in FEATURES)
        p_home = float(mapper(np.array([_sigmoid(logit)]))[0])
        home_fav = p_home >= 0.5
        fav, dog = (
            (g["home_team"], g["away_team"]) if home_fav else (g["away_team"], g["home_team"])
        )
        p_fav = p_home if home_fav else 1 - p_home

        reasons = []
        for group, feats in REASON_GROUPS.items():
            phi = sum(g[f"phi_{f}"] for f in feats)
            p_without = float(mapper(np.array([_sigmoid(logit - phi)]))[0])
            delta = 100 * (p_fav - (p_without if home_fav else 1 - p_without))
            if abs(delta) < MIN_PTS:
                continue
            reasons.append({"group": group, "pts": delta, "text": _phrase(group, phi > 0, g)})
        reasons.sort(key=lambda r: -abs(r["pts"]))
        shown = reasons[:top]
        detail = "; ".join(f"{r['pts']:+.0f} pts {r['text']}" for r in shown)
        why = f"{fav} {p_fav:.0%} over {dog}" + (
            f": {detail}" if detail else ": evenly matched on every factor"
        )
        rows.append(
            {"game_id": g["game_id"], "p_home_explained": p_home, "why": why, "reasons": shown}
        )
    return pd.DataFrame(rows)


def context(matches: pd.DataFrame, elo: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Everything the phrases need, one row per game."""
    from footyforecast.benchmark import ladder_before_match

    lad = ladder_before_match(matches)[["game_id", "home_rank", "away_rank"]]
    return (
        matches[["game_id", "season", "home_team", "away_team", "venue"]]
        .merge(elo[["game_id", "elo_home", "elo_away", "travel_away_km"]], on="game_id")
        .merge(lad, on="game_id")
        .merge(features[["game_id", *FEATURES]], on="game_id")
    )


# ---- Global SHAP plots ----------------------------------------------------------------------
def global_plots() -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    from footyforecast.io import load_matches
    from footyforecast.model_match import load_params, make_models

    matches = load_matches()
    feats = pd.read_parquet(config.DATA_PROCESSED / "features.parquet")
    df = feats.merge(matches[["game_id", "outcome"]], on="game_id")
    df = df[(df["season"] >= config.ML_HISTORY_START) & df["outcome"].isin([0.0, 1.0])]
    X, y = df[FEATURES].astype(float), df["outcome"].astype(int)
    labels = [FEATURE_LABELS[f] for f in FEATURES]

    models = make_models(load_params())
    logit = models["logit"].fit(X, y)
    imp, sc = logit.named_steps["simpleimputer"], logit.named_steps["standardscaler"]
    lr = logit.named_steps["logisticregression"]
    Z = pd.DataFrame(sc.transform(imp.transform(X)), columns=labels)
    lin = shap.LinearExplainer(
        (lr.coef_[0], lr.intercept_[0]), shap.maskers.Independent(Z, max_samples=len(Z))
    )
    sv = lin(Z)

    xgb = models["xgb"].fit(X, y)
    tree_sv = shap.TreeExplainer(xgb)(X)  # binary XGBoost: one (n, features) log-odds array
    tree_vals = np.asarray(tree_sv.values)

    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("Shipped model: SHAP values (log-odds)", loc="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.RESULTS / "shap_beeswarm.png", dpi=150, facecolor="white")
    plt.close("all")

    plt.figure()
    shap.plots.scatter(sv[:, FEATURE_LABELS["rating_diff"]], show=False)
    plt.title("Elo rating gap vs its SHAP value (standardised gap)", loc="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.RESULTS / "shap_dependence_elo.png", dpi=150, facecolor="white")
    plt.close("all")

    imp_table = pd.DataFrame(
        {
            "feature": labels,
            "logistic": np.abs(sv.values).mean(axis=0),
            "xgboost": np.abs(tree_vals).mean(axis=0),
        }
    ).sort_values("logistic", ascending=False)
    top = imp_table.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ypos = np.arange(len(top))
    ax.barh(ypos + 0.2, top["logistic"], height=0.38, color="#2a78d6", label="Logistic (shipped)")
    ax.barh(ypos - 0.2, top["xgboost"], height=0.38, color="#eb6834", label="XGBoost (ablation)")
    ax.set_yticks(ypos, top["feature"])
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title("Two different models, the same evidence", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#e1e0d9", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.RESULTS / "shap_importance.png", dpi=150, facecolor="white")
    plt.close(fig)
    imp_table.to_csv(config.RESULTS / "shap_importance.csv", index=False)
    return imp_table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    table = global_plots()
    logger.info("Top drivers:\n%s", table.head(10).round(3).to_string(index=False))

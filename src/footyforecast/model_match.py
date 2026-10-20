"""Walk-forward ML models: XGBoost and logistic win classifiers, XGBoost and ridge margin models.

Every round from 2014 onwards is predicted by models refit on all games dated before
that round's cutoff (2012+, the era with player data). The 2014–2016 predictions are
out-of-sample warm-up: they pick the XGBoost settings (`tune`) and feed the first
calibration window, so 2017–2026 stays untouched by any choice made here.

Writes results/ml_oos_predictions.parquet (committed). Rescoring it is deterministic, so
CI can verify the leaderboard without refitting hundreds of models.

Run:  python -m footyforecast.model_match            (≈ 3 min)
      python -m footyforecast.model_match --tune     (grid on 2014–2016, ≈ 5 min)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from footyforecast import config
from footyforecast.evaluation import metrics
from footyforecast.features import FEATURES

logger = logging.getLogger(__name__)

FIRST_OOS_SEASON = 2014  # 2012–13 are training-only warm-up
WARMUP_SEASONS = (2014, 2015, 2016)
PARAMS_PATH = config.RESULTS / "ml_params.json"
OOS_PATH = config.RESULTS / "ml_oos_predictions.parquet"

DEFAULT_PARAMS = {
    "xgb": {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.03},
    "logit_C": 0.1,
    "ridge_alpha": 30.0,
    "classifier": "logit",
    "margin_model": "ridge",
}
_XGB_FIXED = {
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "n_jobs": 4,
    "random_state": 42,
}


def load_params() -> dict:
    return (
        json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
        if PARAMS_PATH.exists()
        else DEFAULT_PARAMS
    )


def make_models(params: dict) -> dict:
    xgb = {**params["xgb"], **_XGB_FIXED}
    return {
        "xgb": XGBClassifier(objective="binary:logistic", eval_metric="logloss", **xgb),
        "logit": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=params["logit_C"], max_iter=2000),
        ),
        "margin": XGBRegressor(objective="reg:squarederror", **xgb),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=params.get("ridge_alpha", DEFAULT_PARAMS["ridge_alpha"])),
        ),
    }


def _design(features: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    df = features.merge(matches[["game_id", "round_label", "outcome", "margin"]], on="game_id")
    return (
        df[df["season"] >= config.ML_HISTORY_START]
        .sort_values(["cutoff", "game_id"])
        .reset_index(drop=True)
    )


def walk_forward(
    features: pd.DataFrame,
    matches: pd.DataFrame,
    params: dict | None = None,
    seasons: tuple[int, ...] | None = None,
    game_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Refit before every round and predict it. Returns one OOS row per predicted game.

    By default every played game in ``seasons`` is predicted; pass ``game_ids`` to predict
    exactly those games instead (e.g. next round's unplayed fixture).
    """
    params = params or load_params()
    df = _design(features, matches)
    if game_ids is not None:
        target = df[df["game_id"].isin(game_ids)]
    else:
        seasons = seasons or tuple(range(FIRST_OOS_SEASON, config.LAST_COMPLETED_SEASON + 1))
        target = df[df["season"].isin(seasons) & df["outcome"].notna()]
    X_all = df[FEATURES].astype(float)

    rows = []
    for cutoff, batch in target.groupby("cutoff", sort=True):
        # Train on games *played* before this round starts. Selecting by round (cutoff) instead
        # would let a postponed game from an earlier round, played after this round began, leak in.
        train = df[(df["date"] < cutoff) & df["outcome"].notna()]
        assert train["date"].max() < cutoff, "training data reaches into the predicted round"
        decisive = train[train["outcome"] != 0.5]  # draws are scored, not learned from
        models = make_models(params)
        Xtr = X_all.loc[decisive.index]
        models["xgb"].fit(Xtr, decisive["outcome"].astype(int))
        models["logit"].fit(Xtr, decisive["outcome"].astype(int))
        models["margin"].fit(X_all.loc[train.index], train["margin"])
        models["ridge"].fit(X_all.loc[train.index], train["margin"])
        Xb = X_all.loc[batch.index]
        frame = pd.DataFrame(
            {
                "game_id": batch["game_id"].to_numpy(),
                "season": batch["season"].to_numpy(),
                "cutoff": cutoff,
                "last_train_date": train["date"].max(),
                "n_train": len(train),
                "p_xgb": models["xgb"].predict_proba(Xb)[:, 1],
                "p_logit": models["logit"].predict_proba(Xb)[:, 1],
                "margin_xgb": models["margin"].predict(Xb),
                "margin_ridge": models["ridge"].predict(Xb),
            }
        )
        rows.append(pd.concat([frame, logit_contributions(models["logit"], Xb)], axis=1))
    return pd.concat(rows, ignore_index=True)


def logit_contributions(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Exact SHAP values of the fitted logistic model, in log-odds, one column per feature.

    For a linear model on standardised inputs, SHAP's LinearExplainer with the training data as
    background gives φ_j = β_j · z_j (the training mean of z is 0). ``phi_base`` is the intercept,
    so σ(phi_base + Σ φ_j) reproduces the model's probability (tested).
    """
    imp = pipeline.named_steps["simpleimputer"]
    sc = pipeline.named_steps["standardscaler"]
    lr = pipeline.named_steps["logisticregression"]
    z = sc.transform(imp.transform(X))
    phi = pd.DataFrame(z * lr.coef_[0], columns=[f"phi_{f}" for f in FEATURES])
    phi.insert(0, "phi_base", lr.intercept_[0])
    return phi


def tune(features: pd.DataFrame, matches: pd.DataFrame) -> dict:
    """Grid-search both classifiers on the 2014–2016 warm-up predictions only, then pick one.

    The ML candidate that goes forward to evaluation (``classifier``) is whichever of the
    tuned XGBoost and the tuned logistic regression has the lower warm-up log loss, so the
    evaluation seasons play no part in choosing it.
    """
    outcome = matches.set_index("game_id")[["outcome", "margin"]]
    results = []

    def score(params: dict, label: str) -> None:
        oos = walk_forward(features, matches, params, seasons=WARMUP_SEASONS).join(
            outcome, on="game_id"
        )
        results.append(
            {
                "search": label,
                **params["xgb"],
                "logit_C": params["logit_C"],
                "xgb_log_loss": metrics.log_loss(oos["p_xgb"], oos["outcome"]),
                "logit_log_loss": metrics.log_loss(oos["p_logit"], oos["outcome"]),
                "ridge_alpha": params.get("ridge_alpha", DEFAULT_PARAMS["ridge_alpha"]),
                "margin_mae": float(np.mean(np.abs(oos["margin_xgb"] - oos["margin"]))),
                "ridge_mae": float(np.mean(np.abs(oos["margin_ridge"] - oos["margin"]))),
            }
        )
        logger.info("tune %s", results[-1])

    for n, d in itertools.product([50, 100, 150, 300], [1, 2, 3]):
        score(
            {"xgb": {"n_estimators": n, "max_depth": d, "learning_rate": 0.03}, "logit_C": 0.1},
            "xgb",
        )
    stumps = {"n_estimators": 50, "max_depth": 1, "learning_rate": 0.03}
    for c in (0.01, 0.03, 0.1, 0.3, 1.0):
        score({"xgb": stumps, "logit_C": c}, "logit")
    for a in (3.0, 10.0, 30.0, 100.0, 300.0):
        score({"xgb": stumps, "logit_C": 0.1, "ridge_alpha": a}, "ridge")

    res = pd.DataFrame(results)
    res.to_csv(config.RESULTS / "ml_tuning.csv", index=False)
    best_xgb = res[res["search"] == "xgb"].sort_values("xgb_log_loss").iloc[0]
    best_logit = res[res["search"] == "logit"].sort_values("logit_log_loss").iloc[0]
    best_ridge = res[res["search"] == "ridge"].sort_values("ridge_mae").iloc[0]
    chosen = {
        "xgb": {
            "n_estimators": int(best_xgb["n_estimators"]),
            "max_depth": int(best_xgb["max_depth"]),
            "learning_rate": 0.03,
        },
        "logit_C": float(best_logit["logit_C"]),
        "ridge_alpha": float(best_ridge["ridge_alpha"]),
        "classifier": "xgb" if best_xgb["xgb_log_loss"] < best_logit["logit_log_loss"] else "logit",
        "margin_model": "xgb" if best_xgb["margin_mae"] < best_ridge["ridge_mae"] else "ridge",
        "warmup_log_loss": {
            "xgb": float(best_xgb["xgb_log_loss"]),
            "logit": float(best_logit["logit_log_loss"]),
        },
        "warmup_margin_mae": {
            "xgb": float(best_xgb["margin_mae"]),
            "ridge": float(best_ridge["ridge_mae"]),
        },
        "tuned_on": list(WARMUP_SEASONS),
    }
    PARAMS_PATH.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    return chosen


def main() -> None:
    from footyforecast.io import load_matches

    parser = argparse.ArgumentParser(description="Walk-forward ML predictions.")
    parser.add_argument(
        "--tune", action="store_true", help="grid-search XGBoost/logistic on 2014–2016 first"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    matches = load_matches()
    features = pd.read_parquet(config.DATA_PROCESSED / "features.parquet")
    if args.tune:
        logger.info("Chosen params: %s", tune(features, matches))
    t0 = time.time()
    oos = walk_forward(features, matches)
    oos.to_parquet(OOS_PATH, index=False)
    logger.info(
        "%d OOS predictions (%d rounds) in %.0fs → %s",
        len(oos),
        oos["cutoff"].nunique(),
        time.time() - t0,
        OOS_PATH,
    )


if __name__ == "__main__":
    main()

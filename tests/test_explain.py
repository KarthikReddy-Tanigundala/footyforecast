"""Explanations: exact SHAP for the logistic model and factual plain-English reasons."""

import numpy as np
import pandas as pd
import pytest

from footyforecast.explain import _poss, explain_games
from footyforecast.features import FEATURES
from footyforecast.model_match import logit_contributions, make_models


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(600, len(FEATURES))), columns=FEATURES)
    X.iloc[::17, 3] = np.nan  # exercise the imputer
    y = (X["rating_diff"].fillna(0) + 0.5 * rng.normal(size=600) > 0).astype(int)
    params = {"xgb": {"n_estimators": 10, "max_depth": 1, "learning_rate": 0.1}, "logit_C": 1.0}
    return make_models(params)["logit"].fit(X, y), X


def test_contributions_reproduce_the_probability(fitted):
    model, X = fitted
    phi = logit_contributions(model, X)
    p = 1 / (1 + np.exp(-phi.sum(axis=1)))
    np.testing.assert_allclose(p, model.predict_proba(X)[:, 1], atol=1e-12)


def test_contributions_equal_shap_linear_explainer(fitted):
    shap = pytest.importorskip("shap")
    model, X = fitted
    imp, sc = model.named_steps["simpleimputer"], model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]
    Z = sc.transform(imp.transform(X))
    explainer = shap.LinearExplainer(
        (lr.coef_[0], lr.intercept_[0]), shap.maskers.Independent(Z, max_samples=len(Z))
    )
    ours = logit_contributions(model, X).drop(columns="phi_base").to_numpy()
    np.testing.assert_allclose(ours, explainer(Z).values, atol=1e-8)


def _game(**overrides) -> pd.DataFrame:
    row = {
        "game_id": "G1", "season": 2026, "home_team": "Geelong", "away_team": "Sydney",
        "venue": "Kardinia Park", "elo_home": 1600.0, "elo_away": 1500.0, "travel_away_km": 700.0,
        "home_rank": 2, "away_rank": 6, "phi_base": 0.0,
        **{f: 0.0 for f in FEATURES}, **{f"phi_{f}": 0.0 for f in FEATURES},
    }  # fmt: skip
    row.update(overrides)
    return pd.DataFrame([row])


def test_single_factor_reason_is_quantified_from_the_favourite_view():
    out = explain_games(_game(rating_diff=100.0, phi_rating_diff=0.8), {}).iloc[0]
    p = 1 / (1 + np.exp(-0.8))
    assert out["why"].startswith(f"Geelong {p:.0%} over Sydney")
    assert "Geelong's higher Elo rating (1600 v 1500)" in out["why"]
    assert out["reasons"][0]["pts"] == pytest.approx(100 * (p - 0.5))


def test_reason_against_the_data_says_discounted_instead_of_inventing():
    # Geelong has the better recent margins, but the model leans away from that (mean reversion).
    g = _game(rating_diff=150.0, phi_rating_diff=1.0, margin5_diff=20.0, phi_margin5_diff=-0.3)
    why = explain_games(g, {}).iloc[0]["why"]
    assert "Geelong's recent form edge discounted" in why
    assert "Sydney's better recent form" not in why


def test_possessives():
    assert _poss("Brisbane Lions") == "Brisbane Lions'"
    assert _poss("Geelong") == "Geelong's"

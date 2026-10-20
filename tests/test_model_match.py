"""ML walk-forward integrity and the live (unplayed game) prediction path."""

import numpy as np
import pandas as pd
import pytest

from footyforecast import config
from footyforecast.calibration import calibrate_walk_forward, fit_platt

HAS_FEATURES = (config.DATA_PROCESSED / "features.parquet").exists()
FAST = {"xgb": {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1}, "logit_C": 0.3}


def test_platt_sharpens_an_underconfident_model():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.95, 4000)
    y = rng.binomial(1, true_p)
    shrunk = 0.5 + 0.6 * (true_p - 0.5)  # under-confident: squashed towards 0.5
    mapper = fit_platt(shrunk, y)
    fixed = mapper(shrunk)
    assert np.mean((fixed - y) ** 2) < np.mean((shrunk - y) ** 2)
    assert fixed.max() > shrunk.max() and fixed.min() < shrunk.min()


def test_calibration_uses_only_earlier_seasons():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"season": np.repeat([2014, 2015, 2016, 2017], 200)})
    df["p"] = rng.uniform(0.1, 0.9, len(df))
    df["outcome"] = rng.binomial(1, df["p"]).astype(float)
    base = calibrate_walk_forward(df, "p", seasons=(2017,))
    scrambled = df.copy()
    scrambled.loc[scrambled["season"] == 2017, "outcome"] = 1 - scrambled["outcome"]
    after = calibrate_walk_forward(scrambled, "p", seasons=(2017,))
    pd.testing.assert_series_equal(base, after)  # 2017 outcomes never touch 2017's calibration


@pytest.mark.skipif(not HAS_FEATURES, reason="needs data/processed/features.parquet")
def test_walk_forward_trains_only_on_games_before_each_round():
    from footyforecast.io import load_matches
    from footyforecast.model_match import walk_forward

    matches = load_matches()
    feats = pd.read_parquet(config.DATA_PROCESSED / "features.parquet")
    ids = set(
        matches.loc[
            (matches["season"] == 2016) & matches["round_label"].isin(["R5", "R6"]), "game_id"
        ]
    )
    oos = walk_forward(feats, matches, FAST, game_ids=ids)
    assert set(oos["game_id"]) == ids
    assert (oos["last_train_date"] < oos["cutoff"]).all()
    assert oos[["p_xgb", "p_logit"]].stack().between(0, 1).all()


@pytest.mark.skipif(not HAS_FEATURES, reason="needs data/processed/features.parquet")
def test_unplayed_games_can_be_predicted():
    """Pretend the 2026 Grand Final hasn't been played: the live path must still tip it."""
    from footyforecast.elo import EloEngine, load_schedule
    from footyforecast.features import build_features
    from footyforecast.io import load_geography, load_matches, load_player_games, load_team_games
    from footyforecast.model_match import walk_forward

    matches = load_matches()
    gf = matches.index[(matches["season"] == 2026) & (matches["round_label"] == "GF")]
    matches.loc[gf, ["home_score", "away_score", "margin", "outcome"]] = np.nan
    pg = load_player_games()
    pg = pg[~pg["game_id"].isin(matches.loc[gf, "game_id"])]  # its line-ups are unknown too
    elo = EloEngine(matches, load_geography(), config.ELO, schedule=load_schedule()).run()
    feats = build_features(matches, elo, load_team_games(), pg)
    row = feats.set_index("game_id").loc[matches.loc[gf, "game_id"]]
    assert row["sel_rating_diff"].notna().all()  # fell back to each side's last line-up
    out = walk_forward(feats, matches, FAST, game_ids=set(matches.loc[gf, "game_id"]))
    assert len(out) == 1 and 0 < out["p_xgb"].iat[0] < 1

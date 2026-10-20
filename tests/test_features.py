"""Feature leakage guards: nothing dated on/after a round's cutoff may influence its features."""

import numpy as np
import pandas as pd
import pytest

from footyforecast import config
from footyforecast.features import FEATURES, _asof, build_features
from footyforecast.io import PLAYER_STATS, build_team_games

HAS_DATA = (config.DATA_PROCESSED / "player_games.parquet").exists()


def test_asof_never_matches_same_day_or_later():
    target = pd.DataFrame(
        {"team": ["A", "A"], "cutoff": pd.to_datetime(["2020-03-20", "2020-03-27"])}
    )
    hist = pd.DataFrame(
        {
            "team": ["A", "A", "A"],
            "date": pd.to_datetime(["2020-03-13", "2020-03-20", "2020-03-26"]),
            "x": [1, 2, 3],
        }
    )
    out = _asof(target, hist, "team", ["x"])
    assert out["x"].tolist() == [1, 3]  # the 20 March row is the cutoff day itself: excluded


@pytest.mark.skipif(not HAS_DATA, reason="needs data/processed (run make build)")
def test_scrambling_the_future_leaves_past_features_unchanged():
    from footyforecast.io import load_matches, load_player_games

    matches = load_matches()
    matches = matches[matches["season"].between(2014, 2016)].reset_index(drop=True)
    pg = load_player_games()
    pg = pg[pg["game_id"].isin(matches["game_id"])].reset_index(drop=True)
    # Elo inputs are held fixed (Elo leakage has its own tests); use a neutral stand-in.
    elo = pd.DataFrame(
        {"game_id": matches["game_id"], "elo_home": 1500.0, "elo_away": 1500.0, "hga_elo": 0.0,
         "fam_home": 0, "fam_away": 0, "travel_home_km": 0.0, "travel_away_km": 0.0}
    )  # fmt: skip
    base = build_features(matches, elo, build_team_games(pg), pg)

    split = pd.Timestamp("2016-06-01")
    rng = np.random.default_rng(7)
    m2 = matches.copy()
    late = m2["date"] >= split
    m2.loc[late, "home_score"] = rng.integers(20, 160, late.sum())
    m2.loc[late, "away_score"] = rng.integers(20, 160, late.sum())
    m2["margin"] = m2["home_score"] - m2["away_score"]
    m2["outcome"] = np.select([m2["margin"] > 0, m2["margin"] < 0], [1.0, 0.0], 0.5)
    pg2 = pg.copy()
    late_pg = pg2["date"] >= split
    for s in PLAYER_STATS:
        pg2.loc[late_pg, s] = rng.normal(50, 30, late_pg.sum())
    after = build_features(m2, elo, build_team_games(pg2), pg2)

    early = base["cutoff"] <= split
    assert early.sum() > 300
    pd.testing.assert_frame_equal(base.loc[early, FEATURES], after.loc[early, FEATURES])
    # ...and the scramble really did reach later games (the test would catch a leak).
    assert not base.loc[~early, FEATURES].equals(after.loc[~early, FEATURES])

"""Archetype pipeline pieces and the similarity tool."""

import numpy as np
import pandas as pd
import pytest

from footyforecast.clustering import name_clusters, player_seasons, within_season_z
from footyforecast.similarity import Z_COLS, most_similar
from footyforecast.style import FEATURES, STYLE_STATS


def test_within_season_z_is_centred_and_scaled_per_season():
    rng = np.random.default_rng(0)
    ps = pd.DataFrame(rng.normal(5, 2, size=(200, len(FEATURES))), columns=FEATURES)
    ps["season"] = np.repeat([2020, 2021], 100)
    ps.loc[ps["season"] == 2021, FEATURES] += 10  # an era shift must not leak into the z-scores
    z = within_season_z(ps)
    for _, grp in z.groupby(ps["season"]):
        np.testing.assert_allclose(grp.mean(), 0, atol=1e-9)
        np.testing.assert_allclose(grp.std(), 1, atol=1e-9)


def test_templates_name_obvious_profiles():
    profiles = pd.DataFrame(0.0, index=[0, 1, 2], columns=FEATURES)
    profiles.loc[0, ["hitouts", "ruck_contests", "hitouts_to_advantage"]] = 3.5
    profiles.loc[1, ["goals", "marks_inside_fifty", "shots_at_goal", "marks_on_lead"]] = 2.0
    profiles.loc[
        2, ["clearances", "centre_clearances", "stoppage_clearances", "contested_possessions"]
    ] = 1.6
    names = name_clusters(profiles)
    assert names == {0: "Ruck", 1: "Key Forward", 2: "Inside Midfielder"}


def test_player_seasons_keep_only_full_games_and_enough_of_them():
    rows = []
    for g in range(10):
        for key, tog in (("P1", 80), ("P2", 30 if g < 5 else 80)):  # P2: half the games are cameos
            rows.append(
                {"player_key": key, "season": 2024, "date": pd.Timestamp("2024-03-01") + pd.Timedelta(days=7 * g),
                 "player_name": key, "team": "Geelong", "position": "C", "time_on_ground": tog,
                 **{s: 1.0 for s in STYLE_STATS}, "effective_disposals": 1.0}
            )  # fmt: skip
    ps = player_seasons(pd.DataFrame(rows))
    assert ps["player_key"].tolist() == ["P1"]  # P2 has only 5 full games (< 8)
    assert ps["position_group"].iat[0] == "Midfielder"


def test_most_similar_excludes_the_player_and_finds_a_twin():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(5, len(Z_COLS)))
    z[3] = 2 * z[0]  # same shape, double the volume: cosine similarity 1
    ps = pd.DataFrame(z, columns=Z_COLS)
    ps["player_key"] = ["A", "B", "C", "D", "E"]
    ps["player_name"] = ps["player_key"]
    ps["team"], ps["season"], ps["archetype"], ps["games"] = "X", 2025, "Ruck", 20
    out = most_similar(ps, "A", 2025, n=3)
    assert "A" not in set(out["player_name"])
    assert out["player_name"].iat[0] == "D"
    assert out["similarity"].iat[0] == pytest.approx(1.0)

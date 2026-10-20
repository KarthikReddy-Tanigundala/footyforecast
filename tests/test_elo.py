"""Elo maths and leakage safety on small synthetic seasons."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from footyforecast import config
from footyforecast.elo import EloEngine, Geography, expected_home, margin_to_result

GEO = Geography(
    team_base={"A": (-37.8, 145.0), "B": (-37.8, 145.0), "C": (-31.9, 115.9), "D": (-34.9, 138.6)},
    venues={"MCG": (-37.82, 144.98), "Perth": (-31.95, 115.89), "AdOval": (-34.92, 138.6)},
)
FLAT = replace(config.ELO, hga_mode="flat", hga_flat=30.0, k=60.0, carryover=0.6)


def make_matches(seasons=(2001,), rounds=6, seed=0) -> pd.DataFrame:
    """Round-robin-ish fixture for 4 teams: 2 games a round, one week apart."""
    rng = np.random.default_rng(seed)
    pairings = [("A", "B", "MCG"), ("C", "D", "Perth"), ("D", "A", "AdOval"), ("B", "C", "MCG")]
    rows = []
    for season in seasons:
        start = pd.Timestamp(f"{season}-03-20")
        for r in range(1, rounds + 1):
            games = pairings[:2] if r % 2 else pairings[2:]
            for g, (h, a, v) in enumerate(games):
                margin = float(rng.integers(-60, 61))
                rows.append(
                    {
                        "game_id": f"{season}-{r:02d}-{g}",
                        "season": season,
                        "round_label": str(r),
                        "date": start + pd.Timedelta(days=7 * (r - 1) + g),
                        "home_team": h,
                        "away_team": a,
                        "venue": v,
                        "margin": margin,
                    }
                )
    df = pd.DataFrame(rows)
    df["outcome"] = np.sign(df["margin"]).map({1.0: 1.0, -1.0: 0.0, 0.0: 0.5})
    return df


def test_expectation_is_symmetric_and_centred():
    assert expected_home(0) == pytest.approx(0.5)
    assert expected_home(120) + expected_home(-120) == pytest.approx(1.0)
    assert expected_home(400) == pytest.approx(10 / 11)


def test_margin_mapping_is_linear_and_capped():
    assert margin_to_result(0, 80) == 0.5
    assert margin_to_result(40, 80) == 0.75
    assert margin_to_result(200, 80) == 1.0
    assert margin_to_result(-200, 80) == 0.0


def test_updates_are_zero_sum_within_a_season():
    engine = EloEngine(make_matches(), GEO, FLAT)
    engine.run()
    assert engine.ratings.sum() == pytest.approx(4 * 1500.0)


def test_first_game_uses_initial_ratings_and_home_advantage():
    preds = EloEngine(make_matches(), GEO, FLAT).run()
    first = preds.iloc[0]
    assert first["elo_home"] == first["elo_away"] == 1500.0
    assert first["p_home"] == pytest.approx(expected_home(30.0))


def test_season_carryover_regresses_to_mean():
    m = make_matches(seasons=(2001, 2002))
    engine = EloEngine(m[m["season"] == 2001], GEO, FLAT)
    engine.run()
    end_2001 = dict(zip(engine.team_names, engine.ratings, strict=True))
    preds = EloEngine(m, GEO, FLAT).run().merge(m[["game_id", "season", "home_team"]], on="game_id")
    first_2002 = preds[preds["season"] == 2002].iloc[0]
    mean = np.mean(list(end_2001.values()))
    expected = mean + 0.6 * (end_2001[first_2002["home_team"]] - mean)
    assert first_2002["elo_home"] == pytest.approx(expected)


def test_predictions_never_see_results_from_their_own_round_or_later():
    preds = EloEngine(make_matches(seasons=(2001, 2002)), GEO, FLAT).run()
    known = preds["last_result_date"].dropna()
    assert (known < preds.loc[known.index, "cutoff_date"]).all()


def test_changing_future_results_leaves_past_predictions_unchanged():
    m = make_matches(seasons=(2001,), rounds=8)
    base = EloEngine(m, GEO, FLAT).run().set_index("game_id")
    perturbed = m.copy()
    late = perturbed["round_label"].astype(int) >= 5
    perturbed.loc[late, "margin"] = -perturbed.loc[late, "margin"] + 37
    after = EloEngine(perturbed, GEO, FLAT).run().set_index("game_id")
    early_ids = m.loc[
        m["round_label"].astype(int) <= 5, "game_id"
    ]  # round 5 tips precede round 5 results
    pd.testing.assert_frame_equal(base.loc[early_ids], after.loc[early_ids])


def test_postponed_game_result_is_not_used_before_it_is_played():
    m = make_matches(rounds=4)
    # Round 1's first game is postponed until after round 3 has started.
    m.loc[0, "date"] = m.loc[m["round_label"] == "3", "date"].max() + pd.Timedelta(days=1)
    base = EloEngine(m, GEO, FLAT).run().set_index("game_id")
    flipped = m.copy()
    flipped.loc[0, "margin"] = -flipped.loc[0, "margin"] + 50
    after = EloEngine(flipped, GEO, FLAT).run().set_index("game_id")
    r2_r3 = m.loc[m["round_label"].isin(["2", "3"]), "game_id"]
    pd.testing.assert_frame_equal(base.loc[r2_r3], after.loc[r2_r3])


def test_unplayed_games_are_predicted_but_not_applied():
    m = make_matches(rounds=4)
    m.loc[m["round_label"] == "4", ["margin", "outcome"]] = np.nan
    engine = EloEngine(m, GEO, FLAT)
    preds = engine.run()
    assert preds["p_home"].notna().all()
    assert engine.ratings.sum() == pytest.approx(4 * 1500.0)


def test_venue_hga_favours_familiar_home_side_after_travel():
    venue = replace(FLAT, hga_mode="venue", hga_base=0.0, hga_familiarity_w=20.0, hga_travel_w=5.0)
    preds = EloEngine(make_matches(rounds=6), GEO, venue).run()
    m = make_matches(rounds=6)
    games = preds.merge(m, on="game_id")
    # C (Perth-based) hosting D (Adelaide-based): D flies ~2,100 km, so C gets a travel edge.
    interstate = games.query("venue == 'Perth'")["hga_elo"]
    assert (interstate > 0).all()
    # A and B both live in Melbourne, so their MCG derby carries no travel edge and only a
    # small familiarity one: always well below the interstate trip's advantage.
    derby = games.query("venue == 'MCG' and home_team == 'A'")["hga_elo"]
    assert (derby.abs() < interstate.min()).all()

"""Name normalisation and cross-source game linking."""

import pandas as pd
import pytest

from footyforecast import io


def test_aliases_fold_historic_and_source_specific_names():
    names = pd.Series(["Footscray", "GWS", "Brisbane Bears", "Kangaroos", "Geelong"])
    assert io.canonical_team(names).tolist() == [
        "Western Bulldogs",
        "Greater Western Sydney",
        "Brisbane Lions",
        "North Melbourne",
        "Geelong",
    ]


def test_unknown_team_fails_loudly():
    with pytest.raises(ValueError, match="Unknown team"):
        io.canonical_team(pd.Series(["Tasmania Devils FC"]))


def test_sponsor_venue_names_map_to_canonical():
    out = io.canonical_venue(pd.Series(["Marvel Stadium", "GMHBA Stadium", "M.C.G."]))
    assert out.tolist() == ["Docklands", "Kardinia Park", "M.C.G."]


def test_link_games_matches_reversed_home_and_away():
    matches = pd.DataFrame(
        {
            "game_id": ["G1", "G2"],
            "date": pd.to_datetime(["2015-10-03", "2015-09-26"]),
            "home_team": ["West Coast", "Hawthorn"],
            "away_team": ["Hawthorn", "Fremantle"],
        }
    )
    # Footywire lists Hawthorn at home for the 2015 Grand Final; AFL Tables lists West Coast.
    odds = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-10-03", "2015-09-26"]),
            "h": ["Hawthorn", "Hawthorn"],
            "a": ["West Coast", "Fremantle"],
        }
    )
    linked = io.link_games(odds, matches, "h", "a").set_index("game_id")
    assert linked.loc["G1", "flipped"]
    assert not linked.loc["G2", "flipped"]


def test_bookmaker_devig_sums_to_one_and_flips_reversed_games():
    matches = pd.DataFrame(
        {
            "game_id": ["G1"],
            "season": [2015],
            "date": pd.to_datetime(["2015-10-03"]),
            "home_team": ["West Coast"],
            "away_team": ["Hawthorn"],
        }
    )
    odds = pd.DataFrame(
        {
            "Date": ["2015-10-03"],
            "Home.Team": ["Hawthorn"],
            "Away.Team": ["West Coast"],
            "Home.Win.Odds": [1.63],
            "Away.Win.Odds": [2.30],
            "Home.Line.Odds": [-9.5],
        }
    )
    out = io.build_bookmaker(odds, matches).iloc[0]
    p_hawthorn = (1 / 1.63) / (1 / 1.63 + 1 / 2.30)
    assert out["p_home"] == pytest.approx(1 - p_hawthorn)  # home here = West Coast
    assert out["pred_margin"] == pytest.approx(-9.5)  # Hawthorn tipped by 9.5 → West Coast −9.5

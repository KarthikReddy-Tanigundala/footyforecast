"""Backtest integrity: the leakage guard, ladder baseline and reproducibility of the leaderboard."""

import json

import numpy as np
import pandas as pd
import pytest

from footyforecast import benchmark, config
from footyforecast.elo import EloEngine, load_schedule
from footyforecast.evaluation import metrics
from footyforecast.evaluation.backtest import OURS, assert_walk_forward


def test_walk_forward_guard_catches_lookahead():
    ok = pd.DataFrame(
        {
            "cutoff_date": pd.to_datetime(["2020-03-19"]),
            "last_result_date": pd.to_datetime(["2020-03-15"]),
        }
    )
    assert_walk_forward(ok)
    leaky = ok.assign(last_result_date=pd.to_datetime(["2020-03-19"]))
    with pytest.raises(AssertionError, match="used results"):
        assert_walk_forward(leaky)


def test_ladder_uses_only_completed_rounds_and_previous_final_ladder_in_round_one():
    def game(gid, season, rnd, day, h, a, margin):
        return {
            "game_id": gid, "season": season, "round_label": rnd, "is_final": False,
            "date": pd.Timestamp(f"{season}-03-{day:02d}"), "home_team": h, "away_team": a,
            "home_score": 80 + max(margin, 0), "away_score": 80 + max(-margin, 0), "margin": margin,
        }  # fmt: skip

    m = pd.DataFrame(
        [
            game("a", 2001, "R1", 1, "A", "B", 30),  # A finishes 2001 on top
            game("b", 2001, "R2", 8, "A", "C", 10),
            game("c", 2002, "R1", 1, "B", "A", 5),  # 2002 round 1 uses the 2001 final ladder
            game("d", 2002, "R2", 8, "C", "B", -20),  # round 2 uses 2002 round-1 results only
        ]
    )
    ladder = benchmark.ladder_before_match(m).set_index("game_id")
    assert ladder.loc["c", "away_rank"] == 1  # A, last year's minor premier
    assert ladder.loc["d", "away_rank"] == 1  # B won round 1 of 2002


@pytest.mark.skipif(
    not (config.DATA_PROCESSED / "matches.parquet").exists()
    or not (config.RESULTS / "backtest_results.json").exists(),
    reason="needs data/processed and results/ (run make build + backtest)",
)
def test_committed_leaderboard_is_reproducible():
    from footyforecast.io import load_geography, load_matches

    matches = load_matches()
    preds = EloEngine(matches, load_geography(), config.ELO, schedule=load_schedule()).run()
    assert_walk_forward(preds)
    df = matches.merge(preds, on="game_id")
    df = df[df["season"].isin(config.EVAL_SEASONS) & df["outcome"].notna()]
    committed = json.loads((config.RESULTS / "backtest_results.json").read_text(encoding="utf-8"))
    ours = next(r for r in committed["pooled"] if r["model"] == OURS)
    assert len(df) == ours["games"]
    assert metrics.bits(df["p_home"], df["outcome"]).sum() == pytest.approx(ours["bits"], abs=1e-6)
    assert np.mean(np.abs(df["pred_margin"] - df["margin"])) == pytest.approx(ours["mae"], abs=1e-6)

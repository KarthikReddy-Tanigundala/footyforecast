"""Baselines and external benchmarks that every FootyForecast model has to beat.

All functions return one row per game with ``game_id``, ``p_home`` and ``pred_margin``,
the same shape as the Elo output, so every model is scored by the same code.

* Home team: always tip the home side, with a constant confidence equal to the home
  win rate in the tuning seasons (a tipper needs a probability to earn bits).
* Higher ladder: tip the side higher on the ladder as it stood before the round. Its
  confidence is that rule's hit rate in the tuning seasons. In round 1 the previous
  season's final ladder is used.
* Bookmaker: Footywire head-to-head odds with the overround removed (2010–2020).
* Squiggle: every public model's published probability and margin, including
  "Punters" (the betting-market consensus Squiggle collects) and "Aggregate" (the
  average of all models, a wisdom-of-crowds ceiling).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footyforecast import config


def _played(matches: pd.DataFrame, seasons: tuple[int, ...]) -> pd.DataFrame:
    return matches[matches["season"].isin(seasons) & matches["outcome"].notna()]


def home_baseline(
    matches: pd.DataFrame, tune_seasons: tuple[int, ...] = config.TUNE_SEASONS
) -> pd.DataFrame:
    tune = _played(matches, tune_seasons)
    rate = float(tune["outcome"].mean())  # draws count as half a win
    return pd.DataFrame(
        {"game_id": matches["game_id"], "p_home": rate, "pred_margin": float(tune["margin"].mean())}
    )


def _standings(games: pd.DataFrame, teams: list[str], tiebreak: dict[str, int]) -> dict[str, int]:
    """Ladder rank (1 = top) from completed games: 4 points a win, 2 a draw, then percentage."""
    rows = {t: [0, 0.0, 0.0] for t in teams}  # points, points for, points against
    for g in games.itertuples():
        h, a = rows[g.home_team], rows[g.away_team]
        h[1] += g.home_score
        h[2] += g.away_score
        a[1] += g.away_score
        a[2] += g.home_score
        if g.margin > 0:
            h[0] += 4
        elif g.margin < 0:
            a[0] += 4
        else:
            h[0] += 2
            a[0] += 2
    order = sorted(
        teams,
        key=lambda t: (
            -rows[t][0],
            -(rows[t][1] / rows[t][2] if rows[t][2] else 0.0),
            tiebreak.get(t, 99),
        ),
    )
    return {t: i + 1 for i, t in enumerate(order)}


def ladder_before_match(matches: pd.DataFrame) -> pd.DataFrame:
    """Each side's ladder position as it stood before its round started (no leakage)."""
    out = []
    prev_final: dict[str, int] = {}
    for _, sm in matches.groupby("season", sort=True):
        teams = sorted(set(sm["home_team"]) | set(sm["away_team"]))
        played = sm.dropna(subset=["margin"])
        first_day = sm.groupby("round_label")["date"].transform("min")
        for cutoff, games in sm.groupby(first_day, sort=True):
            before = played[played["date"] < cutoff]
            ranks = (
                _standings(before, teams, prev_final)
                if len(before)
                else {t: prev_final.get(t, 99) for t in teams}
            )
            out.append(
                pd.DataFrame(
                    {
                        "game_id": games["game_id"],
                        "home_rank": games["home_team"].map(ranks),
                        "away_rank": games["away_team"].map(ranks),
                    }
                )
            )
        home_and_away = played[~played["is_final"]]
        if len(home_and_away):
            prev_final = _standings(home_and_away, teams, prev_final)
    return pd.concat(out, ignore_index=True)


def ladder_baseline(
    matches: pd.DataFrame, ladder: pd.DataFrame, tune_seasons: tuple[int, ...] = config.TUNE_SEASONS
) -> pd.DataFrame:
    df = matches[["game_id", "season", "outcome"]].merge(ladder, on="game_id")
    home_higher = df["home_rank"] <= df["away_rank"]
    tune = df[df["season"].isin(tune_seasons) & df["outcome"].notna()]
    tune_higher = tune["home_rank"] <= tune["away_rank"]
    hit_rate = float(np.where(tune_higher, tune["outcome"], 1 - tune["outcome"]).mean())
    return pd.DataFrame(
        {
            "game_id": df["game_id"],
            "p_home": np.where(home_higher, hit_rate, 1 - hit_rate),
            "pred_margin": np.nan,
        }
    )


def bookmaker_baseline(bookmaker: pd.DataFrame) -> pd.DataFrame:
    return bookmaker[["game_id", "p_home", "pred_margin"]].copy()


def squiggle_models(tips: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One prediction frame per Squiggle source."""
    return {
        str(name): grp[["game_id", "p_home", "pred_margin"]]
        .drop_duplicates("game_id")
        .reset_index(drop=True)
        for name, grp in tips.groupby("source")
    }

"""Predict the next round and publish the machine-readable tips feed.

Writes:
    predictions/latest.json                     what the dashboard and Squiggle read
    predictions/{season}/round_{label}.csv      an immutable archive of each round's tips
    data/processed/elo_ratings.parquet          current ratings

The probabilities come from whichever model the backtest shipped (results/backtest_results.json
"headline"): raw Elo, Elo + Platt, or the XGBoost model + Platt. Calibration maps are fitted on
the last three completed seasons' out-of-sample predictions, exactly as in the backtest.

Out of season (no unplayed games in the fixture) the feed reports ``status: off-season``
together with the current ratings and next season's pre-season ratings (after the
between-season regression), so the dashboard still has something true to show.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from footyforecast import __version__, config
from footyforecast.calibration import CAL_WINDOW, fit_platt
from footyforecast.elo import EloEngine, load_schedule
from footyforecast.evaluation.backtest import ELO_CAL, ELO_RAW, ML_NAMES
from footyforecast.io import load_geography, load_matches

logger = logging.getLogger(__name__)


def headline_model() -> str:
    path = config.RESULTS / "backtest_results.json"
    if not path.exists():
        return ELO_RAW
    return json.loads(path.read_text(encoding="utf-8")).get("headline", ELO_RAW)


def shipped_predictions(
    matches: pd.DataFrame, elo: pd.DataFrame, game_ids: set[str], headline: str
) -> pd.DataFrame:
    """P(home win) and predicted margin for ``game_ids`` from the shipped model."""
    elo_sub = elo[elo["game_id"].isin(game_ids)][["game_id", "p_home", "pred_margin"]]
    if headline == ELO_RAW or not game_ids:
        return elo_sub
    decisive = matches[matches["outcome"].isin([0.0, 1.0])]
    last = int(decisive["season"].max())
    window = decisive[decisive["season"].between(last - CAL_WINDOW + 1, last)][
        ["game_id", "outcome"]
    ]
    if headline == ELO_CAL:
        hist = window.merge(elo, on="game_id")
        mapper = fit_platt(hist["p_home"].to_numpy(), hist["outcome"].to_numpy())
        return elo_sub.assign(p_home=mapper(elo_sub["p_home"].to_numpy()))
    if headline in ML_NAMES.values():
        from footyforecast.features import build_features
        from footyforecast.io import load_player_games, load_team_games
        from footyforecast.model_match import OOS_PATH, load_params, walk_forward

        params = load_params()
        col = f"p_{params.get('classifier', 'logit')}"
        margin_col = f"margin_{params.get('margin_model', 'ridge')}"
        feats = build_features(matches, elo, load_team_games(), load_player_games())
        new = walk_forward(feats, matches, game_ids=game_ids)
        hist = window.merge(pd.read_parquet(OOS_PATH), on="game_id")
        mapper = fit_platt(hist[col].to_numpy(), hist["outcome"].to_numpy())
        return pd.DataFrame(
            {
                "game_id": new["game_id"],
                "p_home": mapper(new[col].to_numpy()),
                "pred_margin": new[margin_col],
            }
        )
    raise ValueError(f"Unknown headline model {headline!r}")


def next_round(matches: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Unplayed games in the earliest upcoming round, with their predictions."""
    upcoming = matches[matches["outcome"].isna()].merge(preds, on="game_id")
    if upcoming.empty:
        return upcoming
    first = upcoming.sort_values("kickoff").iloc[0]
    rnd = upcoming[
        (upcoming["season"] == first["season"]) & (upcoming["round_label"] == first["round_label"])
    ]
    return rnd.sort_values("kickoff").reset_index(drop=True)


def preseason_ratings(ratings: pd.DataFrame, carryover: float) -> pd.DataFrame:
    """Apply the between-season regression the engine will apply at the next season start."""
    active = ratings[ratings["active"]].copy()
    mean = active["rating"].mean()
    active["preseason_rating"] = mean + carryover * (active["rating"] - mean)
    return active.sort_values("preseason_rating", ascending=False).reset_index(drop=True)


def feed_rows(games: pd.DataFrame) -> list[dict]:
    rows = []
    for g in games.itertuples():
        home_fav = g.p_home >= 0.5
        rows.append(
            {
                "squiggle_game_id": None
                if pd.isna(g.squiggle_game_id)
                else int(g.squiggle_game_id),
                "season": int(g.season),
                "round": str(g.round_label),
                "kickoff_local": pd.Timestamp(g.kickoff).isoformat(),
                "venue": g.venue,
                "hteam": g.home_team,
                "ateam": g.away_team,
                "tip": g.home_team if home_fav else g.away_team,
                "hconfidence": round(100 * float(g.p_home), 1),
                "hmargin": round(float(g.pred_margin), 1),
                "elo_home": round(float(g.elo_home), 1),
                "elo_away": round(float(g.elo_away), 1),
                "hga_elo": round(float(g.hga_elo), 1),
            }
        )
    return rows


def run() -> dict:
    matches, geo = load_matches(), load_geography()
    schedule = load_schedule()
    engine = EloEngine(matches, geo, config.ELO, schedule=schedule)
    preds = engine.run()
    ratings = engine.final_ratings()
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(config.DATA_PROCESSED / "elo_ratings.parquet", index=False)

    games = next_round(matches, preds)
    headline = headline_model()
    if len(games):
        shipped = shipped_predictions(matches, preds, set(games["game_id"]), headline)
        games = games.drop(columns=["p_home", "pred_margin"]).merge(shipped, on="game_id")
    next_params = schedule.get(config.NEXT_SEASON, config.ELO)
    pre = preseason_ratings(ratings, next_params.carryover)
    last_played = matches.dropna(subset=["outcome"]).iloc[-1]

    feed: dict = {
        "model": headline,
        "version": __version__,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "in-season" if len(games) else "off-season",
        "last_result": {
            "season": int(last_played["season"]),
            "round": str(last_played["round_label"]),
            "date": last_played["date"].date().isoformat(),
            "match": f"{last_played['home_team']} {int(last_played['home_score'])} v "
            f"{last_played['away_team']} {int(last_played['away_score'])}",
        },
        "games": feed_rows(games),
        "ratings": [
            {
                "team": r.team,
                "rating": round(r.rating, 1),
                "preseason_rating": round(r.preseason_rating, 1),
            }
            for r in pre.itertuples()
        ],
        "notes": "hconfidence = P(home win) in %, hmargin = predicted home margin in points.",
    }
    if len(games):
        season, label = int(games["season"].iat[0]), str(games["round_label"].iat[0])
        feed["season"], feed["round"] = season, label
        archive = (
            config.PREDICTIONS
            / str(season)
            / f"round_{re.sub(r'[^A-Za-z0-9]+', '_', label).lower()}.csv"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():  # tips are immutable once published
            pd.DataFrame(feed["games"]).to_csv(archive, index=False)
        logger.info("Predicted %d games for %s %s", len(games), season, label)
    else:
        logger.info(
            "Off-season: no unplayed games in the fixture yet (next season %d).", config.NEXT_SEASON
        )

    config.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    (config.PREDICTIONS / "latest.json").write_text(
        json.dumps(feed, indent=2, default=_json_default), encoding="utf-8"
    )
    return feed


def _json_default(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

"""Pre-match feature engineering for the ML models. Every feature is known before the round.

The one rule: a game's features may use only information dated strictly before its
round's first game (its ``cutoff``). It is enforced mechanically, not by convention:

* Each history table (team results, team stats, player ratings, head-to-head) stores
  rolling values *including* each row's own game.
* Features are attached with ``pd.merge_asof(..., allow_exact_matches=False)`` on the
  cutoff. That picks the latest history row dated strictly before the round starts, so
  a game's own result, or any later one, cannot leak in.

The one exception is *who plays*. Team selections are announced before the round, so the
selected 22 (proxied in the backtest by the players who took the field) is pre-match
information, while those players' ratings come only from earlier games.
``tests/test_features.py`` perturbs every result and stat from a date onwards and
checks that no earlier feature moves.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from footyforecast import benchmark, config
from footyforecast.io import team_bases, venues

logger = logging.getLogger(__name__)

# Team-level stats whose recent dominance (for − against) is tracked as form.
FORM_STATS = [
    "contested_possessions",
    "clearances",
    "inside_fifties",
    "metres_gained",
    "intercepts",
    "tackles",
    "rating_points",
]
FORM_WINDOW = 5  # games
PLAYER_WINDOW = 10  # a player's last N games define his current rating
SELECTION_SIZE = (
    22  # best-22 sum keeps eras comparable (the substitute rule added a 23rd player in 2021)
)
REST_CAP_DAYS = 21  # season openers and byes otherwise dominate the rest feature

FEATURES = [
    # Elo (walk-forward) and venue
    "rating_diff", "hga_elo", "fam_diff", "travel_diff", "interstate_home", "interstate_away",
    # results form
    "win5_diff", "margin5_diff", "margin3_diff",
    # schedule
    "rest_home", "rest_away", "short_break_home", "short_break_away",
    # ladder at the time
    "rank_diff", "pct_log_ratio",
    # head-to-head
    "h2h_win_rate", "h2h_margin",
    # stat dominance form (home − away of each side's rolling for − against)
    *[f"{s}_form_diff" for s in FORM_STATS],
    # team selection
    "sel_rating_diff", "sel_changes_home", "sel_changes_away", "sel_inexperienced_diff",
    "is_final",
]  # fmt: skip


# ---- Helpers -------------------------------------------------------------------------------
def round_cutoffs(matches: pd.DataFrame) -> pd.Series:
    """The first game date of each game's (season, round), i.e. when tips are locked in."""
    key = matches["season"].astype(str) + "|" + matches["round_label"].astype(str)
    return matches["date"].groupby(key).transform("min")


def _asof(
    target: pd.DataFrame,
    history: pd.DataFrame,
    by: str | None,
    cols: list[str],
    left_on: str = "cutoff",
) -> pd.DataFrame:
    """Attach the latest ``history`` row (per ``by``) dated strictly before ``target[left_on]``."""
    keys = [by] if by else []
    t = target.reset_index(drop=False).sort_values(left_on)
    h = (
        history[[*keys, "date", *cols]]
        .rename(columns={"date": "_hist_date"})
        .sort_values("_hist_date")
    )
    merged = pd.merge_asof(
        t,
        h,
        left_on=left_on,
        right_on="_hist_date",
        by=by,
        direction="backward",
        allow_exact_matches=False,
    )
    return merged.set_index("index").sort_index()


def _team_long(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) from the team's perspective, played games only."""
    played = matches.dropna(subset=["margin"])
    home = pd.DataFrame(
        {
            "game_id": played["game_id"],
            "date": played["date"],
            "team": played["home_team"],
            "opponent": played["away_team"],
            "margin": played["margin"],
        }
    )
    away = home.assign(
        team=played["away_team"], opponent=played["home_team"], margin=-played["margin"]
    )
    long = pd.concat([home, away], ignore_index=True)
    long["win"] = np.select([long["margin"] > 0, long["margin"] < 0], [1.0, 0.0], 0.5)
    return long.sort_values(["team", "date", "game_id"]).reset_index(drop=True)


# ---- Feature blocks ---------------------------------------------------------------------------
def results_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Rolling win rate / margin per team, including each row's own game (attach via _asof)."""
    long = _team_long(matches)
    g = long.groupby("team", group_keys=False)
    long["win5"] = g["win"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    long["margin5"] = g["margin"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    long["margin3"] = g["margin"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return long


def head_to_head(matches: pd.DataFrame) -> pd.DataFrame:
    """Rolling last-5 meetings per pair, stored from the alphabetically-first team's view."""
    long = _team_long(matches)
    long = long[long["team"] < long["opponent"]].copy()
    long["pair"] = long["team"] + "|" + long["opponent"]
    g = long.groupby("pair", group_keys=False)
    long["h2h_win_rate_first"] = g["win"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    long["h2h_margin_first"] = g["margin"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    return long


def team_stat_form(team_games: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean of each stat's for − against per team, including the row's own game."""
    tg = team_games.sort_values(["team", "date", "game_id"]).copy()
    g = tg.groupby("team", group_keys=False)
    for s in FORM_STATS:
        tg[f"{s}_dom"] = g[f"{s}_diff"].transform(
            lambda x: x.rolling(FORM_WINDOW, min_periods=1).mean()
        )
    return tg


def selection_strength(
    player_games: pd.DataFrame, matches: pd.DataFrame, cutoffs: pd.Series
) -> pd.DataFrame:
    """Per (game, team): best-22 sum of selected players' recent ratings, changes, inexperience.

    Played games use the players who took the field. Unplayed games use each team's most
    recent line-up until official selections are wired in (Phase 4).
    """
    pg = player_games.sort_values(["player_id", "date", "game_id"]).copy()
    g = pg.groupby("player_id", group_keys=False)
    pg["rating_recent"] = g["rating_points"].transform(
        lambda x: x.rolling(PLAYER_WINDOW, min_periods=1).mean()
    )
    pg["games_so_far"] = g.cumcount() + 1

    selected = _selections(pg, matches)
    selected = selected.merge(cutoffs.rename("cutoff"), left_on="game_id", right_index=True)
    prior = _asof(selected, pg, "player_id", ["rating_recent", "games_so_far"])
    prior["rating_recent"] = prior["rating_recent"].fillna(_debut_rating_asof(pg, prior))
    prior["games_so_far"] = prior["games_so_far"].fillna(0)

    def best22(x: pd.Series) -> float:
        return float(x.nlargest(SELECTION_SIZE).sum())

    agg = prior.groupby(["game_id", "team"]).agg(
        sel_rating=("rating_recent", best22),
        sel_inexperienced=("games_so_far", lambda x: int((x < 10).sum())),
    )
    agg["sel_changes"] = _selection_changes(selected, matches)
    return agg.reset_index()


def _debut_rating_asof(pg: pd.DataFrame, prior: pd.DataFrame) -> pd.Series:
    """Rating to assume for a player with no history: the mean debut-game rating so far.

    Computed as of each round's cutoff (an expanding mean), so later debuts can't leak
    backwards. Everyone "debuts" in the data in 2012, so real debuts are counted from 2013.
    Before then, fall back to the league-average rating to date.
    """
    by_date = pg.sort_values("date")
    debuts = by_date[(by_date["games_so_far"] == 1) & (by_date["date"] >= "2013-01-01")].copy()
    debuts["debut_mean"] = debuts["rating_points"].expanding().mean()
    league = by_date.assign(league_mean=by_date["rating_points"].expanding().mean())
    fill = _asof(prior[["cutoff"]], debuts, None, ["debut_mean"])["debut_mean"]
    fallback = _asof(prior[["cutoff"]], league, None, ["league_mean"])["league_mean"]
    return fill.fillna(fallback).fillna(0.0)


def _selections(pg: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    played = pg[["game_id", "team", "player_id"]]
    upcoming = matches[matches["margin"].isna()]
    if upcoming.empty:
        return played
    last_game = pg.sort_values("date").groupby("team")["game_id"].last()
    rows = []
    for g in upcoming.itertuples():
        for team in (g.home_team, g.away_team):
            if team in last_game.index:
                ids = pg.loc[pg["game_id"] == last_game[team], "player_id"]
                rows.append(
                    pd.DataFrame({"game_id": g.game_id, "team": team, "player_id": ids.to_numpy()})
                )
    return pd.concat([played, *rows], ignore_index=True)


def _selection_changes(selected: pd.DataFrame, matches: pd.DataFrame) -> pd.Series:
    """Players in this game's 22 who were not in the team's previous game."""
    dates = matches.set_index("game_id")["date"]
    squads = selected.groupby(["team", "game_id"])["player_id"].apply(frozenset).reset_index()
    squads["date"] = squads["game_id"].map(dates)
    squads = squads.sort_values(["team", "date"])
    prev = squads.groupby("team")["player_id"].shift(1)
    changes = [
        len(cur - p) if isinstance(p, frozenset) else np.nan
        for cur, p in zip(squads["player_id"], prev, strict=True)
    ]
    return pd.Series(changes, index=pd.MultiIndex.from_frame(squads[["game_id", "team"]]))


# ---- Assembly ------------------------------------------------------------------------------------
def build_features(
    matches: pd.DataFrame,
    elo_preds: pd.DataFrame,
    team_games: pd.DataFrame | None = None,
    player_games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per game: ``game_id`` + every column in ``FEATURES`` (NaN where unknown)."""
    m = matches.copy()
    m["cutoff"] = round_cutoffs(m)
    out = m[
        ["game_id", "season", "date", "cutoff", "home_team", "away_team", "venue", "is_final"]
    ].copy()
    out["is_final"] = out["is_final"].astype(float)

    # Elo + venue (already walk-forward: ratings as at the cutoff)
    e = elo_preds.set_index("game_id")
    out["rating_diff"] = out["game_id"].map(e["elo_home"] - e["elo_away"])
    out["hga_elo"] = out["game_id"].map(e["hga_elo"])
    out["fam_diff"] = out["game_id"].map(np.log1p(e["fam_home"]) - np.log1p(e["fam_away"]))
    out["travel_diff"] = out["game_id"].map(
        np.log1p(e["travel_away_km"]) - np.log1p(e["travel_home_km"])
    )
    state = team_bases().set_index("team")["state"]
    venue_state = venues().set_index("venue")["state"]
    vs = out["venue"].map(venue_state)
    out["interstate_home"] = (out["home_team"].map(state) != vs).astype(float)
    out["interstate_away"] = (out["away_team"].map(state) != vs).astype(float)

    # Results form + rest
    form = results_form(m)
    for side in ("home", "away"):
        tgt = out[["cutoff", "date"]].assign(team=out[f"{side}_team"])
        a = _asof(tgt, form, "team", ["win5", "margin5", "margin3"])
        for c in ("win5", "margin5", "margin3"):
            out[f"{c}_{side}"] = a[c]
        rest = (out["date"] - a["_hist_date"]).dt.days.clip(upper=REST_CAP_DAYS)
        out[f"rest_{side}"] = rest
        out[f"short_break_{side}"] = (rest <= 6).astype(float).where(rest.notna())
    for c in ("win5", "margin5", "margin3"):
        out[f"{c}_diff"] = out[f"{c}_home"] - out[f"{c}_away"]

    # Ladder as it stood
    lad = benchmark.ladder_before_match(m).set_index("game_id")
    out["rank_diff"] = out["game_id"].map(lad["away_rank"] - lad["home_rank"])  # + favours home
    out["pct_log_ratio"] = out["game_id"].map(np.log(lad["home_pct"] / lad["away_pct"]))

    # Head-to-head, re-oriented to the home side
    h2h = head_to_head(m)
    first = np.where(out["home_team"] < out["away_team"], out["home_team"], out["away_team"])
    second = np.where(out["home_team"] < out["away_team"], out["away_team"], out["home_team"])
    tgt = out[["cutoff"]].assign(pair=pd.Series(first, index=out.index) + "|" + second)
    a = _asof(tgt, h2h, "pair", ["h2h_win_rate_first", "h2h_margin_first"])
    home_is_first = out["home_team"] < out["away_team"]
    out["h2h_win_rate"] = np.where(
        home_is_first, a["h2h_win_rate_first"], 1 - a["h2h_win_rate_first"]
    )
    out["h2h_margin"] = np.where(home_is_first, a["h2h_margin_first"], -a["h2h_margin_first"])

    # Stat dominance form
    if team_games is not None and len(team_games):
        sf = team_stat_form(team_games)
        dom = [f"{s}_dom" for s in FORM_STATS]
        sides = {}
        for side in ("home", "away"):
            tgt = out[["cutoff"]].assign(team=out[f"{side}_team"])
            sides[side] = _asof(tgt, sf, "team", dom)
        for s in FORM_STATS:
            out[f"{s}_form_diff"] = sides["home"][f"{s}_dom"] - sides["away"][f"{s}_dom"]
    else:
        for s in FORM_STATS:
            out[f"{s}_form_diff"] = np.nan

    # Selection
    if player_games is not None and len(player_games):
        sel = selection_strength(player_games, m, m.set_index("game_id")["cutoff"]).set_index(
            ["game_id", "team"]
        )
        for side in ("home", "away"):
            idx = pd.MultiIndex.from_arrays([out["game_id"], out[f"{side}_team"]])
            for c in ("sel_rating", "sel_changes", "sel_inexperienced"):
                out[f"{c}_{side}"] = sel[c].reindex(idx).to_numpy()
        out["sel_rating_diff"] = out["sel_rating_home"] - out["sel_rating_away"]
        out["sel_inexperienced_diff"] = (
            out["sel_inexperienced_home"] - out["sel_inexperienced_away"]
        )
    else:
        for c in (
            "sel_rating_diff",
            "sel_changes_home",
            "sel_changes_away",
            "sel_inexperienced_diff",
        ):
            out[c] = np.nan

    return out[["game_id", "season", "date", "cutoff", *FEATURES]].reset_index(drop=True)


def main() -> None:
    from footyforecast.elo import EloEngine, load_schedule
    from footyforecast.io import load_geography, load_matches, load_player_games, load_team_games

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    matches = load_matches()
    elo = EloEngine(matches, load_geography(), config.ELO, schedule=load_schedule()).run()
    feats = build_features(matches, elo, load_team_games(), load_player_games())
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(config.DATA_PROCESSED / "features.parquet", index=False)
    recent = feats[feats["season"] >= config.ML_HISTORY_START]
    logger.info("features: %d games × %d features", len(feats), len(FEATURES))
    logger.info("missing share (2012+):\n%s", recent[FEATURES].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()

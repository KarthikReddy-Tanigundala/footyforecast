"""Elo rating system for the AFL, built from first principles.

How it works
------------
* Every club starts at 1500 (expansion clubs lower, see ``config.EXPANSION_INITIAL``).
* Expected home result:  E = 1 / (1 + 10^(−(R_home − R_away + HGA) / 400)).
* Margin-aware update: the actual result S is the home margin mapped linearly onto
  [0, 1] over [−cap, +cap], so a 60-point thrashing moves ratings more than a
  2-point win.  R_home' = R_home + K·(S − E),  R_away' = R_away − K·(S − E)  (zero-sum).
* Between seasons each rating is pulled back towards the league mean
  (R' = mean + carryover · (R − mean)) to account for list changes.
* Home-ground advantage (HGA) is either flat or venue-aware:
      HGA = base + w_fam · (log1p(home games at venue) − log1p(away games at venue))
                 + w_travel · (log1p(away km travelled) − log1p(home km travelled))
  Ground familiarity is counted over a trailing window, using only games already
  played. This lets Geelong at Kardinia Park earn a big edge while two MCG tenants
  get almost none, and treats the 2020 hubs as neutral without special-casing them.
* The rating gap (plus HGA) maps to a points margin through a slope fitted on the
  tuning seasons.

Leakage safety
--------------
Games are predicted round by round. Before predicting a round, the engine applies
every result whose date is strictly before that round's first game, and nothing
else. A round's tips therefore never see any of its own results, a rescheduled
game's result, or anything later, and Opening Round/Round 0 numbering cannot
cause an off-by-one because ordering is by date, never by round number.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
from bisect import bisect_left
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from footyforecast import config
from footyforecast.config import EloParams
from footyforecast.evaluation import metrics

logger = logging.getLogger(__name__)


# ---- Pure functions ---------------------------------------------------------------
def expected_home(diff: float | np.ndarray) -> float | np.ndarray:
    """Logistic expectation for the home side given (R_home − R_away + HGA)."""
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(diff, dtype=float) / 400.0))


def margin_to_result(margin: float, cap: float) -> float:
    """Map a signed home margin onto [0, 1] linearly over [−cap, +cap] (draw → 0.5)."""
    return (min(max(margin, -cap), cap) + cap) / (2.0 * cap)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class Geography:
    """Team home bases and venue coordinates used for travel distances."""

    team_base: dict[str, tuple[float, float]]
    venues: dict[str, tuple[float, float]]

    def travel_km(self, team: str, venue: str) -> float:
        base, ground = self.team_base.get(team), self.venues.get(venue)
        if base is None or ground is None:
            return 0.0
        return haversine_km(*base, *ground)


# ---- Engine -----------------------------------------------------------------------
class EloEngine:
    """Runs the Elo system over a match table and returns one leakage-safe prediction per game.

    ``matches`` needs: game_id, season, round_label, date, home_team, away_team, venue, margin
    (NaN for games not yet played). Unplayed games are predicted but never applied.
    """

    def __init__(
        self,
        matches: pd.DataFrame,
        geo: Geography,
        params: EloParams = config.ELO,
        schedule: dict[int, EloParams] | None = None,
    ):
        """``schedule`` swaps in season-specific parameters (walk-forward re-tuning) at each
        season start; seasons not in it keep whatever parameters were active before."""
        self.params = params
        self.schedule = schedule or {}
        self.geo = geo
        m = matches.sort_values(["date", "game_id"]).reset_index(drop=True)
        self.matches = m

        teams = pd.unique(pd.concat([m["home_team"], m["away_team"]]))
        self.team_names = list(teams)
        t_idx = {t: i for i, t in enumerate(teams)}
        v_idx = {v: i for i, v in enumerate(pd.unique(m["venue"]))}

        self.home = m["home_team"].map(t_idx).to_numpy()
        self.away = m["away_team"].map(t_idx).to_numpy()
        self.venue = m["venue"].map(v_idx).to_numpy()
        self.day = (m["date"].to_numpy().astype("datetime64[D]").astype(np.int64)).astype(int)
        self.season = m["season"].to_numpy().astype(int)
        scale = m["season"].map(config.SEASON_MARGIN_SCALE).fillna(1.0)
        self.margin_for_update = (m["margin"] * scale).to_numpy(dtype=float)
        self.played = m["margin"].notna().to_numpy()

        self.travel_home = np.array(
            [geo.travel_km(t, v) for t, v in zip(m["home_team"], m["venue"], strict=True)]
        )
        self.travel_away = np.array(
            [geo.travel_km(t, v) for t, v in zip(m["away_team"], m["venue"], strict=True)]
        )

        # Round batches, ordered by their first game.
        batch_key = m["season"].astype(str) + "|" + m["round_label"].astype(str)
        frame = pd.DataFrame({"key": batch_key.to_numpy(), "day": self.day, "i": np.arange(len(m))})
        frame["first_day"] = frame.groupby("key")["day"].transform("min")
        self.batches: list[tuple[int, int, np.ndarray]] = [
            (int(first), int(self.season[grp["i"].iat[0]]), grp["i"].to_numpy())
            for (first, _), grp in frame.groupby(["first_day", "key"], sort=True)
        ]

        # Played games in the order their results become known.
        played_idx = np.flatnonzero(self.played)
        self.result_order = played_idx[np.argsort(self.day[played_idx], kind="stable")]

        self.ratings = np.full(len(teams), np.nan)
        self.familiarity: dict[tuple[int, int], list[int]] = {}
        self.active_by_season: dict[int, set[int]] = {}
        self.current_season: int | None = None
        self.last_result_day: int | None = None

    # -- state transitions --------------------------------------------------------
    def _ensure_team(self, t: int, season: int) -> None:
        if np.isnan(self.ratings[t]):
            name = self.team_names[t]
            fresh = season > config.ELO_HISTORY_START
            self.ratings[t] = (
                config.EXPANSION_INITIAL.get(name, self.params.initial)
                if fresh
                else self.params.initial
            )

    def _new_season(self, season: int) -> None:
        self.params = self.schedule.get(season, self.params)
        prev = self.active_by_season.get(self.current_season or -1, set())
        if prev:
            idx = np.fromiter(prev, dtype=int)
            mean = float(np.mean(self.ratings[idx]))
            self.ratings[idx] = mean + self.params.carryover * (self.ratings[idx] - mean)
        self.current_season = season

    def _hga(self, i: int, cutoff_day: int) -> tuple[float, int, int]:
        p = self.params
        if p.hga_mode == "flat":
            return p.hga_flat, 0, 0
        lo = cutoff_day - p.familiarity_window_days
        fam_h = self._count(self.home[i], self.venue[i], lo)
        fam_a = self._count(self.away[i], self.venue[i], lo)
        hga = (
            p.hga_base
            + p.hga_familiarity_w * (math.log1p(fam_h) - math.log1p(fam_a))
            + p.hga_travel_w * (math.log1p(self.travel_away[i]) - math.log1p(self.travel_home[i]))
        )
        return hga, fam_h, fam_a

    def _count(self, team: int, venue: int, lo_day: int) -> int:
        days = self.familiarity.get((team, venue))
        return 0 if not days else len(days) - bisect_left(days, lo_day)

    def _apply(self, i: int) -> None:
        """Apply game i's result. Uses the HGA as it stood when the game was tipped."""
        h, a = self.home[i], self.away[i]
        e = expected_home(self.ratings[h] - self.ratings[a] + self._pred_hga[i])
        s = margin_to_result(self.margin_for_update[i], self.params.margin_cap)
        delta = self.params.k * (s - e)
        self.ratings[h] += delta
        self.ratings[a] -= delta
        for t in (h, a):
            self.familiarity.setdefault((t, self.venue[i]), []).append(self.day[i])
            self.active_by_season.setdefault(int(self.season[i]), set()).add(int(t))
        self.last_result_day = int(self.day[i])

    # -- main loop -----------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        n = len(self.matches)
        r_home, r_away = np.full(n, np.nan), np.full(n, np.nan)
        hga = np.full(n, np.nan)
        fam_h, fam_a = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        cutoff, last_known = np.zeros(n, dtype=int), np.full(n, -1, dtype=int)
        points_per_elo = np.full(n, np.nan)
        self._pred_hga = hga

        ptr = 0
        for start_day, season, idx in self.batches:
            while ptr < len(self.result_order) and self.day[self.result_order[ptr]] < start_day:
                self._apply(self.result_order[ptr])
                ptr += 1
            if season != self.current_season:
                self._new_season(season)
            for i in idx:
                self._ensure_team(self.home[i], season)
                self._ensure_team(self.away[i], season)
                hga[i], fam_h[i], fam_a[i] = self._hga(i, start_day)
                r_home[i], r_away[i] = self.ratings[self.home[i]], self.ratings[self.away[i]]
                points_per_elo[i] = self.params.points_per_elo
                cutoff[i] = start_day
                last_known[i] = -1 if self.last_result_day is None else self.last_result_day
        while ptr < len(self.result_order):  # trailing results update the final ratings
            self._apply(self.result_order[ptr])
            ptr += 1

        diff = r_home - r_away + hga
        epoch = np.datetime64("1970-01-01", "D")
        return pd.DataFrame(
            {
                "game_id": self.matches["game_id"].to_numpy(),
                "elo_home": r_home,
                "elo_away": r_away,
                "hga_elo": hga,
                "elo_diff": diff,
                "p_home": expected_home(diff),
                "pred_margin": points_per_elo * diff,
                "fam_home": fam_h,
                "fam_away": fam_a,
                "travel_home_km": self.travel_home,
                "travel_away_km": self.travel_away,
                "cutoff_date": epoch + cutoff.astype("timedelta64[D]"),
                "last_result_date": np.where(
                    last_known >= 0,
                    epoch + last_known.astype("timedelta64[D]"),
                    np.datetime64("NaT", "D"),
                ),
            }
        )

    def final_ratings(self) -> pd.DataFrame:
        """Ratings after every played result has been applied (call after ``run``)."""
        last = self.active_by_season.get(max(self.active_by_season, default=-1), set())
        rows = [
            {"team": self.team_names[t], "rating": float(self.ratings[t]), "active": t in last}
            for t in range(len(self.team_names))
            if not np.isnan(self.ratings[t])
        ]
        return pd.DataFrame(rows).sort_values("rating", ascending=False).reset_index(drop=True)


# ---- Fitting & tuning -----------------------------------------------------------------
def fit_points_per_elo(
    preds: pd.DataFrame, matches: pd.DataFrame, seasons: tuple[int, ...]
) -> float:
    """Least-squares slope (through the origin) of actual margin on Elo diff, tuning seasons only."""
    df = preds.merge(matches[["game_id", "season", "margin"]], on="game_id")
    df = df[df["season"].isin(seasons) & df["margin"].notna()]
    x, y = df["elo_diff"].to_numpy(), df["margin"].to_numpy()
    return float((x * y).sum() / (x * x).sum())


def score(preds: pd.DataFrame, matches: pd.DataFrame, seasons: tuple[int, ...]) -> dict[str, float]:
    df = preds.merge(matches[["game_id", "season", "margin", "outcome"]], on="game_id")
    df = df[df["season"].isin(seasons) & df["outcome"].notna()]
    return metrics.summarise(df["p_home"], df["outcome"], df["pred_margin"], df["margin"])


def tune(
    matches: pd.DataFrame,
    geo: Geography,
    grid: dict[str, list],
    base: EloParams = config.ELO,
    seasons: tuple[int, ...] = config.TUNE_SEASONS,
) -> pd.DataFrame:
    """Grid-search Elo hyperparameters by log loss on the tuning seasons.

    Matches after the last tuning season are dropped up front, so the evaluation
    seasons cannot influence the chosen parameters even by accident.
    """
    history = matches[matches["season"] <= max(seasons)]
    rows = []
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = replace(base, **dict(zip(keys, values, strict=True)))
        preds = EloEngine(history, geo, params).run()
        slope = fit_points_per_elo(preds, history, seasons)
        preds["pred_margin"] = slope * preds["elo_diff"]
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "points_per_elo": slope,
                **score(preds, history, seasons),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


TUNING_GRIDS: dict[str, dict[str, list]] = {
    "flat": {
        "hga_mode": ["flat"],
        "k": [55, 70, 85, 100, 115, 130],
        "carryover": [0.5, 0.6, 0.7, 0.8, 0.9],
        "hga_flat": [30, 40, 50, 60, 70, 80],
    },
    "venue": {
        "hga_mode": ["venue"],
        "k": [70, 85, 100, 115, 130],
        "carryover": [0.6, 0.7, 0.8, 0.9],
        "hga_base": [0, 10, 20],
        "hga_familiarity_w": [10, 20, 30],
        "hga_travel_w": [4, 8, 12],
    },
}
# Carryover is capped at 0.9 by design: some regression to the mean between seasons reflects
# list turnover, and the log-loss surface is flat above 0.9 (see EloParams notes).
WALK_FORWARD_GRID: dict[str, list] = {**TUNING_GRIDS["venue"], "carryover": [0.7, 0.8, 0.9]}
SCHEDULE_PATH = config.RESULTS / "elo_params_by_season.csv"
_PARAM_FIELDS = [f for f in EloParams.__dataclass_fields__ if f != "hga_mode"]


def _tune_season(job: tuple[int, pd.DataFrame, Geography, dict[str, list], int]) -> dict:
    season, matches, geo, grid, window = job
    seasons = tuple(range(season - window, season))
    best = tune(matches, geo, grid, seasons=seasons).iloc[0].to_dict()
    return {"season": season, "window": f"{seasons[0]}-{seasons[-1]}", **best}


def walk_forward_schedule(
    matches: pd.DataFrame,
    geo: Geography,
    seasons: tuple[int, ...],
    grid: dict[str, list] = WALK_FORWARD_GRID,
    window: int = config.TUNING_WINDOW_SEASONS,
) -> pd.DataFrame:
    """Re-tune before every season on the trailing ``window`` completed seasons.

    This is exactly what the live model does each pre-season, so the backtest scores the
    tuning *process* rather than one parameter set picked with hindsight. Seasons run in
    parallel processes; each only ever sees matches from before it starts.
    """
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(s, matches[matches["season"] < s], geo, grid, window) for s in seasons]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(_tune_season, jobs))
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def load_schedule(path=SCHEDULE_PATH) -> dict[int, EloParams]:
    """Season → parameters chosen by ``walk_forward_schedule`` (committed in results/)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: run `python -m footyforecast.elo --mode walkforward`"
        )
    df = pd.read_csv(path)
    return {
        int(r["season"]): replace(
            config.ELO,
            hga_mode=str(r.get("hga_mode", "venue")),
            **{
                f: type(getattr(config.ELO, f))(r[f])
                for f in _PARAM_FIELDS
                if f in r and pd.notna(r[f])
            },
        )
        for r in df.to_dict("records")
    }


def main() -> None:
    from footyforecast.io import load_geography, load_matches

    parser = argparse.ArgumentParser(description="Tune the Elo system.")
    parser.add_argument(
        "--mode",
        choices=[*TUNING_GRIDS, "walkforward"],
        default="venue",
        help="flat/venue: one grid search on 2000–2016; walkforward: re-tune before every season",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    matches, geo = load_matches(), load_geography()
    if args.mode == "walkforward":
        seasons = (*config.EVAL_SEASONS, config.NEXT_SEASON)
        schedule = walk_forward_schedule(matches[matches["outcome"].notna()], geo, seasons)
        SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
        schedule.to_csv(SCHEDULE_PATH, index=False)
        cols = [
            "season",
            "window",
            "k",
            "carryover",
            "hga_base",
            "hga_familiarity_w",
            "hga_travel_w",
            "points_per_elo",
            "log_loss",
        ]
        logger.info("Walk-forward parameters:\n%s", schedule[cols].round(4).to_string(index=False))
        return

    results = tune(matches, geo, TUNING_GRIDS[args.mode])
    out = config.RESULTS / f"elo_tuning_{args.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    best = results.iloc[0].to_dict()
    logger.info("Best %s params (log loss %.4f): %s", args.mode, best["log_loss"], best)
    logger.info("Current config: %s", asdict(config.ELO))


if __name__ == "__main__":
    main()

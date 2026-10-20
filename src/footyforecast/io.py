"""Load the raw fitzRoy extracts and normalise them into canonical processed tables.

The sources disagree on almost every name:
    * teams: AFL Tables says "Footscray" and "GWS", Squiggle says "Western Bulldogs" and
      "Greater Western Sydney"; the Bears/Lions and Kangaroos eras need folding together.
    * venues: AFL Tables uses historic names ("Docklands", "Kardinia Park"), Squiggle's
      fixture uses sponsor names ("Marvel Stadium", "GMHBA Stadium").
    * rounds: AFL Tables labels the 2024+ Opening Round "R1"; Squiggle calls it round 0.
      Rounds are therefore never used to join sources. Games are matched on
      (date, home team, away team), and ordering is always by date.

Every processed table uses Squiggle's team names, the names our tips feed is published in.

Run ``python -m footyforecast.io`` to rebuild ``data/processed/*.parquet``.
"""

from __future__ import annotations

import json
import logging
from functools import cache

import numpy as np
import pandas as pd

from footyforecast import config
from footyforecast.elo import Geography

logger = logging.getLogger(__name__)

FINALS_NAMES = {
    "WF": "Wildcard Final",  # introduced in 2026
    "EF": "Elimination Final",
    "QF": "Qualifying Final",
    "SF": "Semi Final",
    "PF": "Preliminary Final",
    "GF": "Grand Final",
}


# ---- Reference data -----------------------------------------------------------------
@cache
def _team_aliases() -> dict[str, str]:
    df = pd.read_csv(config.DATA_REFERENCE / "team_aliases.csv")
    return dict(zip(df["alias"], df["team"], strict=True))


@cache
def _venue_aliases() -> dict[str, str]:
    df = pd.read_csv(config.DATA_REFERENCE / "venue_aliases.csv")
    return dict(zip(df["alias"], df["venue"], strict=True))


@cache
def team_bases() -> pd.DataFrame:
    return pd.read_csv(config.DATA_REFERENCE / "team_bases.csv")


@cache
def venues() -> pd.DataFrame:
    return pd.read_csv(config.DATA_REFERENCE / "venues.csv")


def canonical_team(names: pd.Series) -> pd.Series:
    """Map any source's team name to the canonical (Squiggle) name; fail loudly on unknowns."""
    out = names.map(lambda n: _team_aliases().get(n, n))
    unknown = set(out.dropna()) - set(team_bases()["team"])
    if unknown:
        raise ValueError(
            f"Unknown team name(s) {sorted(unknown)}: add them to data/reference/team_aliases.csv"
        )
    return out


def canonical_venue(names: pd.Series) -> pd.Series:
    """Map sponsor/alternate venue names to the canonical (AFL Tables) name."""
    out = names.map(lambda n: _venue_aliases().get(n, n))
    unknown = set(out.dropna()) - set(venues()["venue"])
    if unknown:
        logger.warning("Venue(s) without coordinates (travel treated as 0): %s", sorted(unknown))
    return out


def load_geography() -> Geography:
    tb, vn = team_bases(), venues()
    return Geography(
        team_base={r.team: (r.lat, r.lon) for r in tb.itertuples()},
        venues={r.venue: (r.lat, r.lon) for r in vn.itertuples()},
    )


# ---- Raw → processed ---------------------------------------------------------------------
def _read_raw(name: str) -> pd.DataFrame:
    path = config.DATA_RAW / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: run `Rscript R/extract_data.R` (or .\\make.ps1 extract)"
        )
    return pd.read_parquet(path)


def _outcome(margin: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([margin > 0, margin < 0], [1.0, 0.0], 0.5), index=margin.index
    ).where(margin.notna())


def build_matches(
    results: pd.DataFrame, squiggle_games: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Canonical one-row-per-game table: AFL Tables results plus not-yet-played Squiggle fixture games."""
    played = pd.DataFrame(
        {
            "game_id": "AT" + results["Game"].astype(int).astype(str),
            "season": results["Season"].astype(int),
            "round_label": results["Round"].astype(str),
            "is_final": results["Round.Type"].eq("Finals"),
            "date": pd.to_datetime(results["Date"]),
            "home_team": canonical_team(results["Home.Team"]),
            "away_team": canonical_team(results["Away.Team"]),
            "venue": canonical_venue(results["Venue"]),
            "home_score": results["Home.Points"].astype(float),
            "away_score": results["Away.Points"].astype(float),
        }
    )
    played["round_name"] = played["round_label"].map(
        lambda r: FINALS_NAMES.get(r, f"Round {r[1:]}" if r.startswith("R") else r)
    )
    played["kickoff"] = played["date"]
    played["squiggle_game_id"] = pd.NA

    upcoming = pd.DataFrame()
    if squiggle_games is not None and len(squiggle_games):
        sq = _squiggle_games_frame(squiggle_games)
        played = _attach_squiggle_ids(played, sq)
        matched = set(played["squiggle_game_id"].dropna())
        todo = sq[(sq["complete"] < 100) & ~sq["squiggle_game_id"].isin(matched)]
        upcoming = pd.DataFrame(
            {
                "game_id": "SQ" + todo["squiggle_game_id"].astype(str),
                "season": todo["year"],
                "round_label": todo["roundname"],
                "round_name": todo["roundname"],
                "is_final": todo["is_final"] > 0,
                "date": todo["kickoff"].dt.normalize(),
                "kickoff": todo["kickoff"],
                "home_team": todo["hteam"],
                "away_team": todo["ateam"],
                "venue": todo["venue"],
                "home_score": np.nan,
                "away_score": np.nan,
                "squiggle_game_id": todo["squiggle_game_id"],
            }
        )

    matches = pd.concat([played, upcoming], ignore_index=True)
    matches["margin"] = matches["home_score"] - matches["away_score"]
    matches["outcome"] = _outcome(matches["margin"])
    matches["squiggle_game_id"] = matches["squiggle_game_id"].astype("Int64")
    cols = [
        "game_id", "squiggle_game_id", "season", "round_label", "round_name", "is_final", "date",
        "kickoff", "home_team", "away_team", "venue", "home_score", "away_score", "margin", "outcome",
    ]  # fmt: skip
    return matches[cols].sort_values(["date", "game_id"]).reset_index(drop=True)


def _squiggle_games_frame(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "squiggle_game_id": raw["id"].astype(int),
            "year": raw["year"].astype(int),
            "round": raw["round"].astype(int),
            "roundname": raw["roundname"].astype(str),
            "is_final": raw["is_final"].astype(int),
            "complete": pd.to_numeric(raw["complete"]),
            "kickoff": pd.to_datetime(raw["date"]),  # local time at the venue
            "hteam": canonical_team(raw["hteam"]),
            "ateam": canonical_team(raw["ateam"]),
            "venue": canonical_venue(raw["venue"]),
        }
    )


def link_games(df: pd.DataFrame, matches: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Attach our game_id to rows keyed by (date, home, away).

    Sources sometimes disagree on who the "home" team was (finals, neutral venues; e.g.
    Footywire lists Hawthorn at home in the 2015 Grand Final, AFL Tables lists West Coast),
    so rows are matched on the unordered pair and ``flipped`` marks reversed rows.
    """
    keys = matches[["game_id", "date", "home_team", "away_team"]].rename(
        columns={"home_team": "_h", "away_team": "_a"}
    )
    df = df.assign(_row=np.arange(len(df)))
    direct = df.merge(keys, left_on=["date", home, away], right_on=["date", "_h", "_a"])
    swapped = df.merge(keys, left_on=["date", home, away], right_on=["date", "_a", "_h"])
    out = pd.concat([direct.assign(flipped=False), swapped.assign(flipped=True)], ignore_index=True)
    unmatched = len(df) - out["_row"].nunique()
    if unmatched > 0:
        logger.warning("%d rows could not be linked to a game", unmatched)
    return out.drop(columns=["_h", "_a", "_row"])


def _attach_squiggle_ids(played: pd.DataFrame, sq: pd.DataFrame) -> pd.DataFrame:
    """Add Squiggle's game id, kickoff time and official round name (e.g. "Opening Round")."""
    keyed = sq.assign(date=sq["kickoff"].dt.normalize())[
        ["date", "hteam", "ateam", "squiggle_game_id", "kickoff", "roundname"]
    ]
    linked = link_games(keyed, played, "hteam", "ateam")[
        ["game_id", "squiggle_game_id", "kickoff", "roundname"]
    ]
    merged = played.drop(columns=["squiggle_game_id", "kickoff"]).merge(
        linked, on="game_id", how="left"
    )
    merged["kickoff"] = merged["kickoff"].fillna(merged["date"])
    merged["round_name"] = merged["roundname"].fillna(merged["round_name"])
    return merged.drop(columns=["roundname"])


def build_squiggle_tips(tips: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Squiggle model predictions as P(home win) + home margin, keyed to our game_id.

    Joins on Squiggle's game id. If Squiggle lists the teams the other way round from
    AFL Tables (it happens for some neutral-venue games), the probability and margin are
    flipped so everything stays from our home team's perspective.
    """
    t = pd.DataFrame(
        {
            "source": tips["source"].astype(str),
            "source_id": tips["sourceid"].astype(int),
            "squiggle_game_id": tips["gameid"].astype(int),
            "hteam": canonical_team(tips["hteam"]),
            "p_home": pd.to_numeric(tips["hconfidence"], errors="coerce") / 100.0,
            "pred_margin": pd.to_numeric(tips["hmargin"], errors="coerce"),
            "squiggle_bits": pd.to_numeric(tips["bits"], errors="coerce"),
            "squiggle_err": pd.to_numeric(tips["err"], errors="coerce"),
        }
    )
    keys = matches.dropna(subset=["squiggle_game_id"])[
        ["game_id", "squiggle_game_id", "home_team", "season"]
    ]
    keys = keys.assign(squiggle_game_id=keys["squiggle_game_id"].astype(int))
    out = t.merge(keys, on="squiggle_game_id", how="inner")
    flipped = out["hteam"] != out["home_team"]
    out.loc[flipped, "p_home"] = 1 - out.loc[flipped, "p_home"]
    out.loc[flipped, "pred_margin"] = -out.loc[flipped, "pred_margin"]
    dropped = len(t) - len(out)
    if dropped:
        logger.warning("%d Squiggle tips could not be matched to a game", dropped)
    return out.drop(columns=["hteam", "home_team"])


def build_bookmaker(odds: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Footywire head-to-head odds → de-vigged P(home win).

    p_home = (1/odds_home) / (1/odds_home + 1/odds_away) removes the bookmaker's margin
    (the overround) proportionally. Footywire's odds history stops in 2020; from 2017
    Squiggle's "Punters" source gives the market view through to today.
    """
    o = pd.DataFrame(
        {
            "date": pd.to_datetime(odds["Date"]),
            "home_team": canonical_team(odds["Home.Team"]),
            "away_team": canonical_team(odds["Away.Team"]),
            "odds_home": pd.to_numeric(odds["Home.Win.Odds"], errors="coerce"),
            "odds_away": pd.to_numeric(odds["Away.Win.Odds"], errors="coerce"),
            "line_home": pd.to_numeric(odds["Home.Line.Odds"], errors="coerce"),
        }
    ).dropna(subset=["odds_home", "odds_away"])
    inv_h, inv_a = 1 / o["odds_home"], 1 / o["odds_away"]
    o["overround"] = inv_h + inv_a - 1
    o["p_home"] = inv_h / (inv_h + inv_a)
    o["pred_margin"] = -o["line_home"]  # a home line of −20.5 means home tipped by 20.5
    out = link_games(o, matches, "home_team", "away_team")
    out.loc[out["flipped"], "p_home"] = 1 - out.loc[out["flipped"], "p_home"]
    out.loc[out["flipped"], "pred_margin"] = -out.loc[out["flipped"], "pred_margin"]
    out = out.merge(matches[["game_id", "season"]], on="game_id")
    return out[
        ["game_id", "season", "odds_home", "odds_away", "overround", "p_home", "pred_margin"]
    ]


# ---- Player stats (Fryzigg 2012–2025 + AFL source 2025+) -------------------------------------------
PLAYER_STATS = [
    "rating_points",
    "contested_possessions",
    "clearances",
    "inside_fifties",
    "metres_gained",
    "intercepts",
    "tackles",
]


def _std_fryzigg(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": raw["season"].astype(int),
            "date": pd.to_datetime(raw["match_date"]),
            "home_team": canonical_team(raw["match_home_team"]),
            "away_team": canonical_team(raw["match_away_team"]),
            "team": canonical_team(raw["player_team"]),
            "player_key": "FZ" + raw["player_id"].astype(int).astype(str),
            "player_name": raw["player_first_name"].str.strip()
            + " "
            + raw["player_last_name"].str.strip(),
            "position": raw["player_position"],
            "time_on_ground": pd.to_numeric(raw["time_on_ground_percentage"], errors="coerce"),
            **{s: pd.to_numeric(raw[s], errors="coerce") for s in PLAYER_STATS},
            "source": "fryzigg",
        }
    )


def _std_afl(raw: pd.DataFrame) -> pd.DataFrame:
    local = pd.to_datetime(raw["utcStartTime"], utc=True).dt.tz_convert("Australia/Melbourne")
    return pd.DataFrame(
        {
            "season": raw["season"].astype(int),
            "date": local.dt.tz_localize(None).dt.normalize(),
            "home_team": canonical_team(raw["home.team.name"]),
            "away_team": canonical_team(raw["away.team.name"]),
            "team": canonical_team(raw["team.name"]),
            "player_key": "AFL" + raw["player.player.player.playerId"].astype(str),
            "player_name": raw["player.player.player.givenName"].str.strip()
            + " "
            + raw["player.player.player.surname"].str.strip(),
            "position": raw.get("player.player.position", pd.Series(pd.NA, index=raw.index)),
            "time_on_ground": pd.to_numeric(raw["timeOnGroundPercentage"], errors="coerce"),
            "rating_points": pd.to_numeric(raw["ratingPoints"], errors="coerce"),
            "contested_possessions": pd.to_numeric(raw["contestedPossessions"], errors="coerce"),
            "clearances": pd.to_numeric(raw["clearances.totalClearances"], errors="coerce"),
            "inside_fifties": pd.to_numeric(raw["inside50s"], errors="coerce"),
            "metres_gained": pd.to_numeric(raw["metresGained"], errors="coerce"),
            "intercepts": pd.to_numeric(raw["intercepts"], errors="coerce"),
            "tackles": pd.to_numeric(raw["tackles"], errors="coerce"),
            "source": "afl",
        }
    )


def build_player_games(
    fryzigg: pd.DataFrame, afl: pd.DataFrame | None, matches: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """One row per player per game, keyed to our game_id, with one player id across sources.

    Fryzigg is the primary source (2012–2025) and the AFL source fills later seasons. Both
    cover 2025, which is used to (a) check the two sources agree and (b) map AFL player
    ids onto Fryzigg ids, so a 2026 player's rating history continues from 2025.
    """
    fz = _std_fryzigg(fryzigg)
    check: dict[str, float] = {}
    if afl is None or afl.empty:
        combined = fz
    else:
        af = _std_afl(afl)
        overlap = sorted(set(fz["season"]) & set(af["season"]))
        keys = ["date", "team", "player_name"]
        both = fz[fz["season"].isin(overlap)].merge(
            af[af["season"].isin(overlap)], on=keys, suffixes=("_fz", "_afl")
        )
        n_afl_overlap = int(af["season"].isin(overlap).sum())
        check = {
            "overlap_seasons": overlap,
            "afl_rows_matched": len(both) / max(n_afl_overlap, 1),
            "rating_points_agree": float(
                (both["rating_points_fz"] - both["rating_points_afl"]).abs().lt(0.05).mean()
            ),
            "metres_gained_agree": float(
                (both["metres_gained_fz"] - both["metres_gained_afl"]).abs().lt(0.5).mean()
            ),
        }
        id_map = both.drop_duplicates("player_key_afl").set_index("player_key_afl")["player_key_fz"]
        later = af[~af["season"].isin(fz["season"].unique())].copy()
        later["player_key"] = later["player_key"].map(id_map).fillna(later["player_key"])
        check["later_players_mapped"] = (
            float(later["player_key"].str.startswith("FZ").mean()) if len(later) else 1.0
        )
        combined = pd.concat([fz, later], ignore_index=True)

    linked = link_games(combined, matches, "home_team", "away_team")
    unlinked = len(combined) - len(linked)
    if unlinked:
        logger.warning("%d player rows not linked to a game", unlinked)
    cols = [
        "game_id",
        "season",
        "date",
        "team",
        "player_key",
        "player_name",
        "position",
        "time_on_ground",
        *PLAYER_STATS,
        "source",
    ]
    out = linked[cols].rename(columns={"player_key": "player_id"})
    return out.sort_values(["date", "game_id", "team", "player_id"]).reset_index(drop=True), check


def build_team_games(player_games: pd.DataFrame) -> pd.DataFrame:
    """Team totals per game plus each stat's for − against (the dominance the form features roll)."""
    stats = [s for s in PLAYER_STATS]
    tg = player_games.groupby(["game_id", "season", "date", "team"], as_index=False)[stats].sum(
        min_count=1
    )
    opp = tg[["game_id", "team", *stats]].rename(
        columns={"team": "opponent", **{s: f"{s}_against" for s in stats}}
    )
    tg = tg.merge(opp, on="game_id")
    tg = tg[tg["team"] != tg["opponent"]].copy()
    for s in stats:
        tg[f"{s}_diff"] = tg[s] - tg[f"{s}_against"]
    return tg.sort_values(["date", "game_id", "team"]).reset_index(drop=True)


# ---- Processed loaders ---------------------------------------------------------------------
def _processed(name: str) -> pd.DataFrame:
    path = config.DATA_PROCESSED / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: run `python -m footyforecast.io` (or .\\make.ps1 build)"
        )
    return pd.read_parquet(path)


def load_matches() -> pd.DataFrame:
    return _processed("matches")


def load_squiggle_tips() -> pd.DataFrame:
    return _processed("squiggle_tips")


def load_bookmaker() -> pd.DataFrame:
    return _processed("bookmaker_footywire")


def load_player_games() -> pd.DataFrame:
    return _processed("player_games")


def load_team_games() -> pd.DataFrame:
    return _processed("team_games")


def build_all() -> dict[str, pd.DataFrame]:
    """Rebuild every processed table from data/raw and write it to data/processed.

    Player-level raw extracts are large and gitignored. Without them (e.g. in CI) the
    committed player_games/team_games tables are kept as they are.
    """
    sq_games = _read_raw("squiggle_games")
    matches = build_matches(_read_raw("results"), sq_games)
    tables = {
        "matches": matches,
        "squiggle_tips": build_squiggle_tips(_read_raw("squiggle_tips"), matches),
        "bookmaker_footywire": build_bookmaker(_read_raw("odds"), matches),
    }
    if (config.DATA_RAW / "player_stats_fryzigg.parquet").exists():
        afl = (
            _read_raw("player_stats_afl")
            if (config.DATA_RAW / "player_stats_afl.parquet").exists()
            else None
        )
        player_games, check = build_player_games(_read_raw("player_stats_fryzigg"), afl, matches)
        tables["player_games"] = player_games
        tables["team_games"] = build_team_games(player_games)
        if check:
            logger.info("Fryzigg vs AFL source on the overlap: %s", check)
            (config.DATA_PROCESSED / "player_source_check.json").write_text(
                json.dumps(check, indent=2), encoding="utf-8"
            )
    else:
        logger.info("No raw player stats: keeping the committed player_games/team_games tables")
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(config.DATA_PROCESSED / f"{name}.parquet", index=False)
        logger.info("%-20s %6d rows", name, len(df))

    recent = matches[matches["season"] >= 2017]
    linked = recent["squiggle_game_id"].notna().mean()
    logger.info("Squiggle game ids linked for %.1f%% of 2017+ games", 100 * linked)
    logger.info("Upcoming (unplayed) games in fixture: %d", int(matches["margin"].isna().sum()))
    return tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_all()

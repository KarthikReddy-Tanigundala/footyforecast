"""'Which players are most like X?' Cosine similarity on within-season style z-scores.

Cosine similarity compares the *shape* of a player's statistical profile rather than its
volume. Two inside midfielders match even if one averages 22 disposals and the other 30,
which suits a "plays like" question. Because the z-scores are within-season, a 2014 player
can be compared fairly with a 2026 one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footyforecast.style import FEATURES

# Radar axes: one or two signature stats per role, shown as within-season percentiles.
RADAR_STATS = {
    "disposals": "Disposals",
    "contested_possessions": "Contested poss.",
    "clearances": "Clearances",
    "inside_fifties": "Inside 50s",
    "metres_gained": "Metres gained",
    "marks": "Marks",
    "intercepts": "Intercepts",
    "tackles": "Tackles",
    "goals": "Goals",
    "hitouts": "Hitouts",
}

Z_COLS = [f"z_{c}" for c in FEATURES]


def _unit(m: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norm == 0, 1, norm)


def find_player(ps: pd.DataFrame, name: str, season: int | None = None) -> pd.Series:
    """The player-season for ``name`` (case-insensitive); latest season if none is given."""
    rows = ps[ps["player_name"].str.casefold() == name.casefold()]
    if season is not None:
        rows = rows[rows["season"] == season]
    if rows.empty:
        raise KeyError(
            f"No qualifying player-season for {name!r}" + (f" in {season}" if season else "")
        )
    return rows.sort_values("season").iloc[-1]


def most_similar(
    ps: pd.DataFrame,
    player_key: str,
    season: int,
    n: int = 10,
    seasons: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Top-``n`` player-seasons most like (player_key, season), excluding the player himself."""
    target = ps[(ps["player_key"] == player_key) & (ps["season"] == season)]
    if target.empty:
        raise KeyError(f"{player_key} has no qualifying {season} season")
    pool = ps[ps["player_key"] != player_key]
    if seasons is not None:
        pool = pool[pool["season"].between(*seasons)]
    sims = _unit(pool[Z_COLS].to_numpy()) @ _unit(target[Z_COLS].to_numpy())[0]
    out = pool.assign(similarity=sims).sort_values("similarity", ascending=False)
    # One row per player: their single most similar season.
    out = out.drop_duplicates("player_key").head(n)
    cols = ["player_name", "team", "season", "archetype", "games", "similarity"]
    return out[cols].reset_index(drop=True)


def radar(ps: pd.DataFrame, rows: list[pd.Series]) -> pd.DataFrame:
    """Within-season percentile (0–100) of each radar stat for the given player-seasons."""
    out = []
    for r in rows:
        season = ps[ps["season"] == r["season"]]
        pct = {
            label: float((season[stat] <= r[stat]).mean() * 100)
            for stat, label in RADAR_STATS.items()
        }
        out.append({"player": f"{r['player_name']} ({r['season']})", **pct})
    return pd.DataFrame(out).set_index("player")

"""Player archetypes: cluster player-seasons by playing style, validate against positions.

Pipeline
--------
1. Player-game stats from Fryzigg (2012–2025) + the AFL source (2026), one player id.
2. Per player-season averages over "full" games (≥ 50% time on ground, so sub and injury
   cameos don't dilute a role), keeping players with at least 8 such games.
3. **Within-season z-scores**: each stat is measured against that season's peers, so a
   2013 tackler and a 2025 tackler are compared on role rather than on how the game's
   tackle count drifted over time.
4. PCA (90% of variance) → K-means for the archetypes; a Gaussian mixture on the same
   space gives soft memberships (hybrid players). UMAP is fitted only for the 2-D picture.
5. K is chosen by silhouette within 6–9, the range earlier AFL work found interpretable.
   Positions are **not** used to choose K, because they are the validation.
6. Validation: adjusted Rand index and NMI between clusters and named positions (grouped),
   plus the full crosstab. Named positions are the team-sheet slot, a noisy label (a
   "half-forward" may play a wing), so partial agreement is expected.
7. Names come from domain templates (e.g. "Ruck": hitouts, ruck contests). Each cluster's
   centroid profile is matched to one template by the Hungarian algorithm, so naming is
   reproducible rather than eyeballed.

Run:  python -m footyforecast.clustering    (≈ 1 min; needs the raw player extracts)
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture

from footyforecast import config
from footyforecast.style import FEATURES, POSITION_GROUPS, STYLE_STATS

logger = logging.getLogger(__name__)

SEED = 42
MIN_GAMES = 8
MIN_TOG = 50
K_RANGE = range(4, 13)
K_INTERPRETABLE = range(6, 10)

# Archetype templates: the stats that should stand out (+) or be low (−) for each role.
ARCHETYPES: dict[str, dict[str, float]] = {
    "Ruck": {"hitouts": 1, "ruck_contests": 1, "hitouts_to_advantage": 1, "contested_marks": 0.3},
    "Inside Midfielder": {
        "clearances": 1, "centre_clearances": 1, "stoppage_clearances": 1,
        "contested_possessions": 1, "handballs": 0.7, "ground_ball_gets": 0.6, "hitouts": -0.5,
    },
    "Outside Midfielder / Wing": {
        "uncontested_possessions": 1, "metres_gained": 0.8, "kicks": 0.6, "inside_fifties": 0.8,
        "marks": 0.4, "clearances": -0.3, "rebounds": -0.2,
    },
    "Rebounding Defender": {
        "rebounds": 1, "score_launches": 0.8, "kicks": 0.6, "uncontested_possessions": 0.6,
        "intercept_marks": 0.4, "goals": -0.6, "inside_fifties": -0.3,
    },
    "Key / Intercept Defender": {
        "spoils": 1, "one_percenters": 1, "intercept_marks": 0.6, "contested_marks": 0.5,
        "goals": -0.8, "clearances": -0.6, "inside_fifties": -0.6,
    },
    "Key Forward": {
        "goals": 1, "marks_inside_fifty": 1, "contested_marks": 0.8, "shots_at_goal": 0.8,
        "marks_on_lead": 0.8, "tackles": -0.3, "clearances": -0.4,
    },
    "Small / Pressure Forward": {
        "tackles_inside_fifty": 1, "f50_ground_ball_gets": 1, "goal_assists": 0.6, "goals": 0.5,
        "pressure_acts": 0.5, "contested_marks": -0.5, "rebounds": -0.6,
    },
    "Midfield-Forward": {
        "score_involvements": 1, "goal_assists": 0.7, "inside_fifties": 0.7, "clearances": 0.4,
        "goals": 0.5, "rebounds": -0.6, "spoils": -0.6,
    },
    "General Defender": {
        "def_half_pressure_acts": 1, "intercepts": 0.7, "one_percenters": 0.5, "spoils": 0.5,
        "rebounds": 0.5, "goals": -0.7, "inside_fifties": -0.4,
    },
    "Fringe / Role Player": {"rating_points": -1, "disposals": -0.8, "metres_gained": -0.5},
}  # fmt: skip


# ---- Data ----------------------------------------------------------------------------------
def load_player_game_stats() -> pd.DataFrame:
    from footyforecast.io import _read_raw, combine_player_sources

    fz = _read_raw("player_stats_fryzigg")
    afl_path = config.DATA_RAW / "player_stats_afl.parquet"
    afl = pd.read_parquet(afl_path) if afl_path.exists() else None
    games, _ = combine_player_sources(fz, afl, [*STYLE_STATS, "effective_disposals"])
    return games


def player_seasons(
    games: pd.DataFrame, min_games: int = MIN_GAMES, min_tog: int = MIN_TOG
) -> pd.DataFrame:
    """Per-game averages per player-season over full games, with modal named position."""
    full = games[games["time_on_ground"] >= min_tog].copy()
    full["position_group"] = full["position"].map(POSITION_GROUPS)
    g = full.sort_values("date").groupby(["player_key", "season"])
    ps = g[STYLE_STATS].mean()
    sums = g[["effective_disposals", "disposals"]].sum()
    ps["disposal_efficiency"] = sums["effective_disposals"] / sums["disposals"].replace(0, np.nan)
    ps["games"] = g.size()
    ps["player_name"] = g["player_name"].last()
    ps["team"] = g["team"].agg(lambda s: s.mode().iat[0])
    ps["position_group"] = g["position_group"].agg(
        lambda s: s.mode().iat[0] if s.notna().any() else None
    )
    ps["time_on_ground"] = g["time_on_ground"].mean()
    ps = ps[ps["games"] >= min_games].reset_index()
    ps[FEATURES] = ps[FEATURES].fillna(0.0)
    return ps


def within_season_z(ps: pd.DataFrame, cols: list[str] = FEATURES) -> pd.DataFrame:
    """z-score each stat against the same season's qualifying players."""
    g = ps.groupby("season")[cols]
    return ((ps[cols] - g.transform("mean")) / g.transform("std").replace(0, 1)).fillna(0.0)


# ---- Clustering ----------------------------------------------------------------------------
def choose_k(X: np.ndarray) -> tuple[int, pd.DataFrame]:
    """Inertia (elbow), silhouette and GMM BIC for each K; pick best silhouette within 6–9."""
    rows = []
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(X), size=min(4000, len(X)), replace=False)
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED).fit(X)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X[sample], km.labels_[sample]),
                "gmm_bic": gmm.bic(X),
            }
        )
    table = pd.DataFrame(rows)
    pick = table[table["k"].isin(K_INTERPRETABLE)].sort_values("silhouette", ascending=False)
    return int(pick.iloc[0]["k"]), table


def name_clusters(profiles: pd.DataFrame) -> dict[int, str]:
    """Match each cluster's mean z-profile to one archetype template (Hungarian algorithm)."""
    names = list(ARCHETYPES)
    templates = pd.DataFrame(ARCHETYPES).T.reindex(columns=profiles.columns).fillna(0.0)
    t = templates.to_numpy()
    p = profiles.to_numpy()
    tn = t / np.linalg.norm(t, axis=1, keepdims=True)
    pn = p / np.linalg.norm(p, axis=1, keepdims=True)
    score = pn @ tn.T  # cosine similarity, clusters × templates
    rows, cols = linear_sum_assignment(-score)
    return {int(profiles.index[r]): names[c] for r, c in zip(rows, cols, strict=True)}


def fit_archetypes(ps: pd.DataFrame, k: int | None = None) -> tuple[pd.DataFrame, dict]:
    z = within_season_z(ps)
    pca = PCA(n_components=0.90, random_state=SEED).fit(z)
    X = pca.transform(z)
    selection = None
    if k is None:
        k, selection = choose_k(X)
    km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X)
    gmm = GaussianMixture(
        n_components=k, covariance_type="full", random_state=SEED, means_init=km.cluster_centers_
    ).fit(X)
    probs = gmm.predict_proba(X)

    out = ps.copy()
    out["cluster"] = km.labels_
    profiles = z.groupby(km.labels_).mean()
    names = name_clusters(profiles)
    out["archetype"] = out["cluster"].map(names)
    # EM can move GMM components away from the K-means centres they started from, so map
    # components to clusters explicitly (max overlap, Hungarian) before reading memberships.
    overlap = pd.crosstab(km.labels_, gmm.predict(X)).reindex(
        index=range(k), columns=range(k), fill_value=0
    )
    _, comp_for_cluster = linear_sum_assignment(-overlap.to_numpy())
    probs = probs[:, comp_for_cluster]  # column c now = cluster c
    rows = np.arange(len(out))
    out["membership"] = probs[rows, km.labels_]
    others = probs.copy()
    others[rows, km.labels_] = -1
    second = others.argmax(axis=1)
    out["secondary_archetype"] = [names[int(j)] for j in second]
    out["secondary_membership"] = probs[rows, second]
    out[[f"z_{c}" for c in FEATURES]] = z.to_numpy()

    labelled = out.dropna(subset=["position_group"])
    validation = {
        "k": k,
        "pca_components": int(pca.n_components_),
        "player_seasons": len(out),
        "ari_vs_position": float(
            adjusted_rand_score(labelled["position_group"], labelled["archetype"])
        ),
        "nmi_vs_position": float(
            normalized_mutual_info_score(labelled["position_group"], labelled["archetype"])
        ),
        "crosstab": pd.crosstab(labelled["archetype"], labelled["position_group"]).to_dict(),
        "purity": _purity(labelled),
        "profiles": profiles.rename(index=names).round(2).to_dict(orient="index"),
        "sizes": out["archetype"].value_counts().to_dict(),
        "k_selection": None if selection is None else selection.to_dict(orient="records"),
    }
    return out, validation


def _purity(labelled: pd.DataFrame) -> list[dict]:
    """Per archetype: its most common named position and that position's share of it."""
    ct = pd.crosstab(labelled["archetype"], labelled["position_group"])
    return [
        {
            "archetype": a,
            "player_seasons": int(row.sum()),
            "top_named_position": str(row.idxmax()),
            "share": float(row.max() / row.sum()),
            "recall": float(row.max() / ct[row.idxmax()].sum()),
        }
        for a, row in ct.iterrows()
    ]


def add_umap(out: pd.DataFrame) -> pd.DataFrame:
    """2-D UMAP of the z-scored style space, for the scatter plot only (never for clustering)."""
    import umap  # heavy import (numba), so only loaded here

    z = out[[f"z_{c}" for c in FEATURES]].to_numpy()
    emb = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=SEED).fit_transform(z)
    return out.assign(umap_x=emb[:, 0], umap_y=emb[:, 1])


def plots(out: pd.DataFrame, validation: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = pd.DataFrame(validation["k_selection"]) if validation["k_selection"] else None
    if sel is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(sel["k"], sel["inertia"], marker="o", color="#2a78d6")
        axes[0].set_title("Elbow: K-means inertia", loc="left")
        axes[1].plot(sel["k"], sel["silhouette"], marker="o", color="#2a78d6")
        axes[1].axvline(validation["k"], color="#898781", lw=1, ls="--")
        axes[1].set_title(f"Silhouette (chosen K = {validation['k']})", loc="left")
        for ax in axes:
            ax.set_xlabel("K")
            ax.grid(color="#e1e0d9", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(config.RESULTS / "cluster_selection.png", dpi=150, facecolor="white")
        plt.close(fig)

    # Okabe-Ito: colour-blind safe for up to 8 categories, plus grey for a 9th/10th.
    palette = [
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#D55E00",
        "#F0E442",
        "#000000",
        "#999999",
        "#7F3C8D",
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, grp) in enumerate(sorted(out.groupby("archetype"), key=lambda kv: -len(kv[1]))):
        ax.scatter(
            grp["umap_x"],
            grp["umap_y"],
            s=5,
            alpha=0.6,
            color=palette[i % len(palette)],
            label=f"{name} ({len(grp):,})",
        )
    ax.set_title(
        f"AFL player archetypes, {out['season'].min()}–{out['season'].max()} "
        f"({len(out):,} player-seasons, UMAP of style stats)",
        loc="left",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.legend(markerscale=4, frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(config.RESULTS / "cluster_viz.png", dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ps = player_seasons(load_player_game_stats())
    out, validation = fit_archetypes(ps)
    out = add_umap(out)
    keep = [
        "player_key", "player_name", "team", "season", "games", "time_on_ground", "position_group",
        "cluster", "archetype", "membership", "secondary_archetype", "secondary_membership",
        "umap_x", "umap_y", *FEATURES, *[f"z_{c}" for c in FEATURES],
    ]  # fmt: skip
    out[keep].to_parquet(config.DATA_PROCESSED / "player_seasons.parquet", index=False)
    (config.RESULTS / "archetypes.json").write_text(
        json.dumps(validation, indent=2, default=float), encoding="utf-8"
    )
    plots(out, validation)
    logger.info(
        "K=%d, %d player-seasons; ARI %.3f, NMI %.3f vs named positions; sizes %s",
        validation["k"], validation["player_seasons"], validation["ari_vs_position"],
        validation["nmi_vs_position"], validation["sizes"],
    )  # fmt: skip


if __name__ == "__main__":
    main()

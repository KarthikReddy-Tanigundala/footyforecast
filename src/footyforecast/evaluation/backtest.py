"""Walk-forward backtest over 2017–2026 and the "FootyForecast vs the field" leaderboard.

Each FootyForecast prediction is produced before its round starts, using only results
dated before that round (see ``elo.EloEngine``). The guard below re-checks this on the
output (fold integrity), so a future refactor cannot silently introduce lookahead.

Outputs:
    results/backtest_results.json       per-model, per-season and pooled metrics
    results/backtest_predictions.parquet game-level predictions used by the dashboard
    results/leaderboard.md               the human-readable leaderboard
    data/processed/elo_ratings.parquet   current ratings after the last result
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from footyforecast import benchmark, config
from footyforecast.elo import EloEngine, load_schedule
from footyforecast.evaluation import metrics
from footyforecast.io import load_bookmaker, load_geography, load_matches, load_squiggle_tips

logger = logging.getLogger(__name__)

OURS = "FootyForecast Elo"
MODEL_KIND = {
    OURS: "FootyForecast",
    "Elo (frozen 2000–16 params)": "FootyForecast ablation",
    "Elo (flat HGA)": "FootyForecast ablation",
    "Home team": "Baseline",
    "Higher ladder": "Baseline",
    "Bookmaker (Footywire)": "Market",
    "Punters": "Market",
    "Aggregate": "Crowd",
}


def assert_walk_forward(preds: pd.DataFrame) -> None:
    """Fold integrity: every prediction's latest known result predates its round's cutoff."""
    known = preds.dropna(subset=["last_result_date"])
    leaks = known[known["last_result_date"] >= known["cutoff_date"]]
    if len(leaks):
        raise AssertionError(
            f"{len(leaks)} predictions used results from on/after their round start"
        )


def score_models(
    preds: dict[str, pd.DataFrame], games: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-season and pooled scorecards for every model over the games it tipped."""
    by_season, pooled = [], []
    for name, p in preds.items():
        df = games.merge(p[["game_id", "p_home", "pred_margin"]], on="game_id").dropna(
            subset=["p_home"]
        )
        if df.empty:
            continue
        for season, sdf in df.groupby("season"):
            row = metrics.summarise(
                sdf["p_home"], sdf["outcome"], sdf["pred_margin"], sdf["margin"]
            )
            by_season.append({"model": name, "season": int(season), **row})
        row = metrics.summarise(df["p_home"], df["outcome"], df["pred_margin"], df["margin"])
        pooled.append({"model": name, "seasons": int(df["season"].nunique()), **row})
    return pd.DataFrame(by_season), pd.DataFrame(pooled)


def head_to_head(
    preds: dict[str, pd.DataFrame], games: pd.DataFrame, ours: str = OURS
) -> pd.DataFrame:
    """Compare every model with FootyForecast on exactly the games both tipped."""
    mine = preds[ours][["game_id", "p_home", "pred_margin"]].rename(
        columns={"p_home": "p_ours", "pred_margin": "m_ours"}
    )
    rows = []
    for name, p in preds.items():
        if name == ours:
            continue
        df = games.merge(p, on="game_id").merge(mine, on="game_id").dropna(subset=["p_home"])
        if df.empty:
            continue
        theirs = metrics.bits(df["p_home"], df["outcome"])
        ours_bits = metrics.bits(df["p_ours"], df["outcome"])
        rows.append(
            {
                "model": name,
                "games": len(df),
                "seasons": f"{df['season'].min()}–{df['season'].max()}",
                "their_bits": float(theirs.sum()),
                "our_bits": float(ours_bits.sum()),
                "bits_delta": float(ours_bits.sum() - theirs.sum()),
                "their_mae": float(
                    np.nanmean(metrics.margin_abs_error(df["pred_margin"], df["margin"]))
                )
                if df["pred_margin"].notna().any()
                else np.nan,
                "our_mae": float(np.mean(metrics.margin_abs_error(df["m_ours"], df["margin"]))),
            }
        )
    return pd.DataFrame(rows).sort_values("bits_delta")


def season_ranks(by_season: pd.DataFrame, games: pd.DataFrame, ours: str = OURS) -> pd.DataFrame:
    """Where FootyForecast would have finished each season on Squiggle's leaderboard.

    The field is every Squiggle source (including Punters and Aggregate) that tipped every
    game that season; our own ablations and the naive baselines are not counted.
    """
    n_games = games.groupby("season").size()
    excluded = {
        m for m, kind in MODEL_KIND.items() if kind in {"FootyForecast ablation", "Baseline"}
    }
    excluded.add("Bookmaker (Footywire)")
    rows = []
    for season, sdf in by_season.groupby("season"):
        full = sdf[(sdf["games"] == n_games[season]) & ~sdf["model"].isin(excluded)]
        rank_bits = full["bits"].rank(ascending=False, method="min")
        rank_tips = full["tips"].rank(ascending=False, method="min")
        rank_mae = full["mae"].rank(ascending=True, method="min")
        me = full["model"] == ours
        rows.append(
            {
                "season": int(season),
                "models": int(len(full)),
                "rank_bits": int(rank_bits[me].iloc[0]),
                "rank_tips": int(rank_tips[me].iloc[0]),
                "rank_mae": int(rank_mae[me].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def validate_metrics(tips: pd.DataFrame, games: pd.DataFrame) -> dict[str, float]:
    """Check our bits/MAE implementation reproduces the values Squiggle publishes."""
    df = tips.merge(games[["game_id", "outcome", "margin"]], on="game_id").dropna(
        subset=["squiggle_bits"]
    )
    ours = metrics.bits(df["p_home"], df["outcome"])
    bits_gap = np.abs(ours - df["squiggle_bits"])
    err = df.dropna(subset=["pred_margin", "squiggle_err"])
    mae_gap = np.abs(
        metrics.margin_abs_error(err["pred_margin"], err["margin"]) - err["squiggle_err"]
    )
    return {
        "tips_checked": int(len(df)),
        "bits_within_0_01": float((bits_gap < 0.01).mean()),
        "bits_max_abs_gap": float(bits_gap.max()),
        "err_within_0_5": float((mae_gap < 0.5).mean()),
    }


def _fmt(x: float, nd: int = 1) -> str:
    return "–" if pd.isna(x) else f"{x:,.{nd}f}"


def write_leaderboard(
    pooled: pd.DataFrame,
    by_season: pd.DataFrame,
    h2h: pd.DataFrame,
    ranks: pd.DataFrame,
    validation: dict[str, float],
    n_eval_games: int,
) -> str:
    n_seasons = len(config.EVAL_SEASONS)
    full = pooled[pooled["games"] == n_eval_games].sort_values("bits", ascending=False)
    lines = [
        "# FootyForecast vs The Field",
        "",
        f"Walk-forward backtest, **{config.EVAL_SEASONS[0]}–{config.EVAL_SEASONS[-1]}** "
        f"({n_eval_games:,} games, {n_seasons} seasons). Every FootyForecast tip was made before its "
        "round using only earlier results, and before each season the Elo hyperparameters were "
        f"re-tuned on the previous {config.TUNING_WINDOW_SEASONS} seasons only, the same process "
        "the live model runs each pre-season.",
        "",
        "Metrics follow Squiggle: **tips** = correct winners (draws count as correct), **bits** = "
        "Monash probabilistic score (higher is better; a coin flip scores 0), **MAE** = mean absolute "
        "margin error in points.",
        "",
        "## Models that tipped every game",
        "",
        "| # | Model | Type | Tips | Accuracy | Bits | Bits / season | MAE | Brier |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(full.itertuples(), 1):
        name = f"**{r.model}**" if r.model == OURS else r.model
        lines.append(
            f"| {i} | {name} | {MODEL_KIND.get(r.model, 'Squiggle model')} | {r.tips:,} | "
            f"{r.accuracy:.1%} | {_fmt(r.bits)} | {_fmt(r.bits / n_seasons)} | {_fmt(r.mae, 2)} | "
            f"{r.brier:.4f} |"
        )

    lines += [
        "",
        "## Season by season: FootyForecast vs the market and the crowd",
        "",
        "| Season | FF tips | FF bits | FF MAE | Punters bits | Punters MAE | Aggregate bits | Aggregate MAE | FF rank (bits) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    piv = by_season.set_index(["model", "season"])
    for r in ranks.itertuples():
        s = r.season

        def g(model: str, col: str, s: int = s) -> float:
            return piv[col].get((model, s), np.nan)

        lines.append(
            f"| {s} | {int(g(OURS, 'tips'))} | {_fmt(g(OURS, 'bits'))} | {_fmt(g(OURS, 'mae'), 2)} | "
            f"{_fmt(g('Punters', 'bits'))} | {_fmt(g('Punters', 'mae'), 2)} | "
            f"{_fmt(g('Aggregate', 'bits'))} | {_fmt(g('Aggregate', 'mae'), 2)} | "
            f"{r.rank_bits} / {r.models} |"
        )

    lines += [
        "",
        "## Same-games comparison with every Squiggle model",
        "",
        "Many public models only tipped some seasons, so each one is compared with FootyForecast on "
        "exactly the games both tipped. Positive Δ bits means FootyForecast scored more.",
        "",
        "| Model | Seasons | Games | Their bits | FF bits | Δ bits | Their MAE | FF MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in h2h.itertuples():
        lines.append(
            f"| {r.model} | {r.seasons} | {r.games:,} | {_fmt(r.their_bits)} | {_fmt(r.our_bits)} | "
            f"{r.bits_delta:+.1f} | {_fmt(r.their_mae, 2)} | {_fmt(r.our_mae, 2)} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- **Honest framing:** the betting market and the Squiggle Aggregate are strong. Tips are a "
        "noisy metric, so judge models on bits and MAE across all ten seasons, not one lucky year.",
        "- Bookmaker (Footywire) odds are de-vigged proportionally and only exist for 2010–2020, so they "
        "appear in the same-games table; Squiggle's *Punters* source carries the market from 2017 to today.",
        f"- **Metric check:** our bits implementation matches Squiggle's published per-tip bits within "
        f"0.01 for {validation['bits_within_0_01']:.1%} of {validation['tips_checked']:,} tips, and "
        f"margin error within 0.5 points for {validation['err_within_0_5']:.1%}.",
        "- 2020 was played with 16-minute quarters, so margins (and MAE) run smaller that season.",
        f"- Generated {datetime.now(UTC):%Y-%m-%d} by `python -m footyforecast.evaluation.backtest`.",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    matches, geo = load_matches(), load_geography()
    tips = load_squiggle_tips()

    schedule = load_schedule()
    engine = EloEngine(matches, geo, config.ELO, schedule=schedule)
    elo = engine.run()
    ablations = {
        "Elo (frozen 2000–16 params)": EloEngine(matches, geo, config.ELO).run(),
        "Elo (flat HGA)": EloEngine(matches, geo, config.ELO_FLAT).run(),
    }
    for p in (elo, *ablations.values()):
        assert_walk_forward(p)

    ladder = benchmark.ladder_before_match(matches)
    preds: dict[str, pd.DataFrame] = {
        OURS: elo,
        **ablations,
        "Home team": benchmark.home_baseline(matches),
        "Higher ladder": benchmark.ladder_baseline(matches, ladder),
        "Bookmaker (Footywire)": benchmark.bookmaker_baseline(load_bookmaker()),
        **benchmark.squiggle_models(tips),
    }

    games = matches[matches["season"].isin(config.EVAL_SEASONS) & matches["outcome"].notna()][
        ["game_id", "season", "outcome", "margin"]
    ]
    by_season, pooled = score_models(preds, games)
    h2h = head_to_head(preds, games)
    ranks = season_ranks(by_season, games)
    validation = validate_metrics(tips, games)

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    md = write_leaderboard(pooled, by_season, h2h, ranks, validation, len(games))
    (config.RESULTS / "leaderboard.md").write_text(md, encoding="utf-8", newline="\n")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "eval_seasons": list(config.EVAL_SEASONS),
        "elo_params": asdict(config.ELO),
        "games": int(len(games)),
        "pooled": pooled.to_dict("records"),
        "by_season": by_season.to_dict("records"),
        "season_ranks": ranks.to_dict("records"),
        "head_to_head": h2h.to_dict("records"),
        "metric_validation": validation,
    }
    (config.RESULTS / "backtest_results.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )

    market = preds["Punters"].rename(columns={"p_home": "p_punters", "pred_margin": "m_punters"})
    crowd = preds["Aggregate"].rename(
        columns={"p_home": "p_aggregate", "pred_margin": "m_aggregate"}
    )
    game_level = (
        matches.merge(elo.drop(columns=["last_result_date"]), on="game_id")
        .merge(market, on="game_id", how="left")
        .merge(crowd, on="game_id", how="left")
    )
    game_level = game_level[game_level["season"] >= config.EVAL_SEASONS[0]]
    game_level.to_parquet(config.RESULTS / "backtest_predictions.parquet", index=False)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    engine.final_ratings().to_parquet(config.DATA_PROCESSED / "elo_ratings.parquet", index=False)

    ours = pooled.set_index("model").loc[OURS]
    logger.info(
        "%s %d–%d: %d/%d tips (%.1f%%), %.1f bits, MAE %.2f",
        OURS, config.EVAL_SEASONS[0], config.EVAL_SEASONS[-1], ours["tips"], ours["games"],
        100 * ours["accuracy"], ours["bits"], ours["mae"],
    )  # fmt: skip
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

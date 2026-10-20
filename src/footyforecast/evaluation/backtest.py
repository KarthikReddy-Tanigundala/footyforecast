"""Walk-forward backtest over 2017–2026 and the "FootyForecast vs the field" leaderboard.

Each FootyForecast prediction is produced before its round starts, using only results
dated before that round (see ``elo.EloEngine``). The guard below re-checks this on the
output (fold integrity), so a future refactor cannot silently introduce lookahead.

FootyForecast candidates (the shipped one is picked by a rule fixed in advance, below):
    * FootyForecast Elo                   walk-forward Elo
    * FootyForecast Elo + Platt           Elo with walk-forward Platt calibration
    * FootyForecast ML (… + Platt)        pre-match features → classifier → Platt, where the
                                          classifier (XGBoost or logistic) was chosen on the
                                          2014–2016 warm-up seasons, before evaluation
The shipped model is the candidate with the most bits over 2017–2026. Choosing among three
on the evaluation window is a small optimistic bias, disclosed in the leaderboard; the
2027 live season is the truly out-of-sample test.

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
from footyforecast.calibration import (
    calibrate_walk_forward,
    reliability_figure,
    walk_forward_mappers,
)
from footyforecast.elo import EloEngine, load_schedule
from footyforecast.evaluation import metrics
from footyforecast.io import load_bookmaker, load_geography, load_matches, load_squiggle_tips
from footyforecast.model_match import load_params

logger = logging.getLogger(__name__)

ELO_RAW = "FootyForecast Elo"
ELO_CAL = "FootyForecast Elo + Platt"
ML_NAMES = {
    "xgb": "FootyForecast ML (XGBoost + Platt)",
    "logit": "FootyForecast ML (logistic + Platt)",
}
ABLATION_NAMES = {"xgb": "XGBoost + Platt", "logit": "Logistic + Platt"}
CANDIDATES = (ELO_RAW, ELO_CAL, *ML_NAMES.values())  # only the warm-up-chosen ML one competes
OURS = ELO_RAW  # default headline before ML predictions exist; run() applies the selection rule
MODEL_KIND = {
    ELO_RAW: "FootyForecast",
    ELO_CAL: "FootyForecast",
    **dict.fromkeys(ML_NAMES.values(), "FootyForecast"),
    **dict.fromkeys(ABLATION_NAMES.values(), "FootyForecast ablation"),
    "XGBoost (uncalibrated)": "FootyForecast ablation",
    "Logistic (uncalibrated)": "FootyForecast ablation",
    "XGBoost + isotonic": "FootyForecast ablation",
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
        m
        for m, kind in MODEL_KIND.items()
        if kind in {"FootyForecast", "FootyForecast ablation", "Baseline"} and m != ours
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


def paired_bootstrap(
    preds: dict[str, pd.DataFrame],
    games: pd.DataFrame,
    a: str,
    b: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Is model ``a`` really ahead of ``b`` on bits? Round-clustered paired bootstrap.

    Per-game bits differences are summed per round, and rounds are resampled with
    replacement, because games in one round share conditions (weather, upsets, rule
    tweaks) and are not independent. Reports the total difference over the window with
    a 95% interval and the share of resamples in which ``a`` comes out ahead.
    """
    df = games.merge(preds[a][["game_id", "p_home"]], on="game_id").merge(
        preds[b][["game_id", "p_home"]], on="game_id", suffixes=("_a", "_b")
    )
    df["diff"] = metrics.bits(df["p_home_a"], df["outcome"]) - metrics.bits(
        df["p_home_b"], df["outcome"]
    )
    per_round = df.groupby(["season", "round_label"])["diff"].sum().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(per_round, size=(n_boot, len(per_round)), replace=True).sum(axis=1)
    return {
        "a": a,
        "b": b,
        "games": int(len(df)),
        "bits_diff": float(per_round.sum()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_a_ahead": float((draws > 0).mean()),
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
    ours: str = OURS,
    calibration: pd.DataFrame | None = None,
    significance: list[dict] | None = None,
) -> str:
    n_seasons = len(config.EVAL_SEASONS)
    full = pooled[pooled["games"] == n_eval_games].sort_values("bits", ascending=False)
    lines = [
        "# FootyForecast vs The Field",
        "",
        f"Walk-forward backtest, **{config.EVAL_SEASONS[0]}–{config.EVAL_SEASONS[-1]}** "
        f"({n_eval_games:,} games, {n_seasons} seasons). Every FootyForecast tip was made before its "
        "round using only earlier results. Before each season the Elo hyperparameters were re-tuned "
        f"on the previous {config.TUNING_WINDOW_SEASONS} seasons; the ML models were refit before "
        "every round; calibration maps were fitted on the previous three seasons' out-of-sample "
        "predictions only.",
        "",
        "Metrics follow Squiggle: **tips** = correct winners (draws count as correct), **bits** = "
        "Monash probabilistic score (higher is better; a coin flip scores 0), **MAE** = mean absolute "
        "margin error in points.",
        "",
        f"**Shipped model: {ours}** (bold below).",
        "",
        "## Models that tipped every game",
        "",
        "| # | Model | Type | Tips | Accuracy | Bits | Bits / season | MAE | Brier |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(full.itertuples(), 1):
        name = f"**{r.model}**" if r.model == ours else r.model
        lines.append(
            f"| {i} | {name} | {MODEL_KIND.get(r.model, 'Squiggle model')} | {r.tips:,} | "
            f"{r.accuracy:.1%} | {_fmt(r.bits)} | {_fmt(r.bits / n_seasons)} | {_fmt(r.mae, 2)} | "
            f"{r.brier:.4f} |"
        )

    cands = pooled[pooled["model"].isin(CANDIDATES)].sort_values("bits", ascending=False)
    lines += [
        "",
        "## Which FootyForecast model ships",
        "",
        "Rule fixed before the evaluation was run: ship the candidate with the most bits over "
        f"{config.EVAL_SEASONS[0]}–{config.EVAL_SEASONS[-1]}. Picking the best of three on the same "
        "window flatters the winner slightly; the 2027 live season is the clean out-of-sample test.",
        "",
        "| Candidate | Bits | Accuracy | MAE | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in cands.itertuples():
        name = f"**{r.model}**" if r.model == ours else r.model
        lines.append(
            f"| {name} | {_fmt(r.bits)} | {r.accuracy:.1%} | {_fmt(r.mae, 2)} | "
            f"{r.brier:.4f} | {r.ece:.3f} |"
        )

    if significance:
        lines += [
            "",
            "### Is the difference real?",
            "",
            "Paired bootstrap over rounds (2,000 resamples; rounds are resampled because games in a "
            "round are not independent). Δ is the shipped model's total bits minus the other model's "
            "over the same games.",
            "",
            "| Shipped vs | Δ bits | 95% interval | Resamples where shipped is ahead |",
            "|---|---:|---:|---:|",
        ]
        for r in significance:
            lines.append(
                f"| {r['b']} | {r['bits_diff']:+.1f} | [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}] | "
                f"{r['p_a_ahead']:.0%} |"
            )

    if calibration is not None and len(calibration):
        lines += [
            "",
            "## Calibration",
            "",
            "Brier score and expected calibration error (ECE: count-weighted |predicted − observed| "
            "over ten probability bins), pooled over the evaluation window, draws excluded.",
            "",
            "| Model | Version | Brier | ECE |",
            "|---|---|---:|---:|",
        ]
        for r in calibration.itertuples():
            lines.append(f"| {r.model} | {r.version} | {r.brier:.4f} | {r.ece:.3f} |")
        lines += ["", "![Reliability diagrams](calibration_plot.png)"]

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
            f"| {s} | {int(g(ours, 'tips'))} | {_fmt(g(ours, 'bits'))} | {_fmt(g(ours, 'mae'), 2)} | "
            f"{_fmt(g('Punters', 'bits'))} | {_fmt(g('Punters', 'mae'), 2)} | "
            f"{_fmt(g('Aggregate', 'bits'))} | {_fmt(g('Aggregate', 'mae'), 2)} | "
            f"{r.rank_bits} / {r.models} |"
        )

    lines += [
        "",
        "## Same-games comparison with every Squiggle model",
        "",
        f"Many public models only tipped some seasons, so each one is compared with {ours} on "
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
        "- Bookmaker (Footywire) odds are de-vigged proportionally and only exist for 2010–2020, so "
        "they appear in the same-games table; Squiggle's *Punters* source carries the market from "
        "2017 to today.",
        f"- **Metric check:** our bits implementation matches Squiggle's published per-tip bits within "
        f"0.01 for {validation['bits_within_0_01']:.1%} of {validation['tips_checked']:,} tips, and "
        f"margin error within 0.5 points for {validation['err_within_0_5']:.1%}.",
        "- 2020 was played with 16-minute quarters, so margins (and MAE) run smaller that season.",
        "- Player statistics: Fryzigg for 2012–2025 and the AFL source for 2026, cross-checked on 2025.",
        f"- Generated {datetime.now(UTC):%Y-%m-%d} by `python -m footyforecast.evaluation.backtest`.",
        "",
    ]
    return "\n".join(lines)


def build_predictions(
    matches: pd.DataFrame, tips: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, EloEngine]:
    """Every model's predictions (one frame each) plus the frame used for calibration.

    The ML models are read from results/ml_oos_predictions.parquet (written by
    ``python -m footyforecast.model_match``); without it only the Elo candidates compete.
    """
    geo = load_geography()
    engine = EloEngine(matches, geo, config.ELO, schedule=load_schedule())
    elo = engine.run()
    ablations = {
        "Elo (frozen 2000–16 params)": EloEngine(matches, geo, config.ELO).run(),
        "Elo (flat HGA)": EloEngine(matches, geo, config.ELO_FLAT).run(),
    }
    for p in (elo, *ablations.values()):
        assert_walk_forward(p)

    elo_cols = elo[["game_id", "p_home", "pred_margin"]].rename(
        columns={"p_home": "p_elo", "pred_margin": "m_elo"}
    )
    cal = matches[["game_id", "season", "outcome", "margin"]].merge(elo_cols, on="game_id")
    cal = cal[cal["season"] >= config.EVAL_SEASONS[0] - 3].reset_index(drop=True)
    cal["p_elo_platt"] = calibrate_walk_forward(cal, "p_elo", "platt")

    def frame(p: str, m: str) -> pd.DataFrame:
        out = cal[["game_id", p, m]].rename(columns={p: "p_home", m: "pred_margin"})
        return out.dropna(subset=["p_home"])

    preds: dict[str, pd.DataFrame] = {ELO_RAW: elo, ELO_CAL: frame("p_elo_platt", "m_elo")}
    oos_path = config.RESULTS / "ml_oos_predictions.parquet"
    if oos_path.exists():
        oos = pd.read_parquet(oos_path)
        leaky = oos[oos["last_train_date"] >= oos["cutoff"]]
        if len(leaky):
            raise AssertionError(
                f"{len(leaky)} ML predictions trained on games from their own round"
            )
        cal = cal.merge(
            oos[["game_id", "p_xgb", "p_logit", "margin_xgb", "margin_ridge"]],
            on="game_id",
            how="left",
        )
        cal["p_xgb_platt"] = calibrate_walk_forward(cal, "p_xgb", "platt")
        cal["p_xgb_iso"] = calibrate_walk_forward(cal, "p_xgb", "isotonic")
        cal["p_logit_platt"] = calibrate_walk_forward(cal, "p_logit", "platt")
        params = load_params()
        chosen = params.get("classifier", "logit")
        other = "xgb" if chosen == "logit" else "logit"
        margin_col = f"margin_{params.get('margin_model', 'ridge')}"
        preds[ML_NAMES[chosen]] = frame(f"p_{chosen}_platt", margin_col)
        preds[ABLATION_NAMES[other]] = frame(f"p_{other}_platt", margin_col)
        preds["XGBoost (uncalibrated)"] = frame("p_xgb", "margin_xgb")
        preds["Logistic (uncalibrated)"] = frame("p_logit", margin_col)
        preds["XGBoost + isotonic"] = frame("p_xgb_iso", "margin_xgb")
    else:
        logger.warning(
            "No ML predictions (%s): run `python -m footyforecast.model_match`", oos_path
        )

    ladder = benchmark.ladder_before_match(matches)
    preds |= {
        **ablations,
        "Home team": benchmark.home_baseline(matches),
        "Higher ladder": benchmark.ladder_baseline(matches, ladder),
        "Bookmaker (Footywire)": benchmark.bookmaker_baseline(load_bookmaker()),
        **benchmark.squiggle_models(tips),
    }
    return preds, cal, engine


def tip_reasons(
    matches: pd.DataFrame, preds: dict[str, pd.DataFrame], cal: pd.DataFrame, ours: str
) -> pd.DataFrame:
    """Plain-English reasons for every evaluation-season tip of the shipped logistic model."""
    from footyforecast.explain import context, explain_games

    oos_path = config.RESULTS / "ml_oos_predictions.parquet"
    if (
        ours not in ML_NAMES.values()
        or load_params().get("classifier") != "logit"
        or not oos_path.exists()
    ):
        return pd.DataFrame(columns=["game_id", "why"])
    phi = pd.read_parquet(oos_path).filter(regex="^(game_id|phi_)")
    feats = pd.read_parquet(config.DATA_PROCESSED / "features.parquet")
    ctx = context(matches, preds[ELO_RAW], feats).merge(phi, on="game_id")
    ctx = ctx[ctx["season"].isin(config.EVAL_SEASONS)]
    explained = explain_games(ctx, walk_forward_mappers(cal, "p_logit"))
    # The reasons must describe exactly the probabilities that were scored.
    shipped = explained.merge(preds[ours][["game_id", "p_home"]], on="game_id")
    gap = (shipped["p_home_explained"] - shipped["p_home"]).abs().max()
    if gap > 1e-9:
        raise AssertionError(
            f"explanations disagree with the shipped probabilities (max gap {gap:.2e})"
        )
    return explained[["game_id", "why"]]


def pick_headline(pooled: pd.DataFrame) -> str:
    """The pre-registered rule: ship the FootyForecast candidate with the most bits."""
    cands = pooled[pooled["model"].isin(CANDIDATES)]
    return str(cands.sort_values("bits", ascending=False).iloc[0]["model"])


def run() -> dict:
    matches = load_matches()
    tips = load_squiggle_tips()
    preds, cal, engine = build_predictions(matches, tips)

    games = matches[matches["season"].isin(config.EVAL_SEASONS) & matches["outcome"].notna()][
        ["game_id", "season", "round_label", "outcome", "margin"]
    ]
    by_season, pooled = score_models(preds, games)
    ours = pick_headline(pooled)
    rivals = [m for m in (*CANDIDATES, "Punters", "Aggregate") if m in preds and m != ours]
    significance = [paired_bootstrap(preds, games, ours, other) for other in rivals]
    h2h = head_to_head(preds, games, ours)
    ranks = season_ranks(by_season, games, ours)
    validation = validate_metrics(tips, games)

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    evalc = cal[cal["season"].isin(config.EVAL_SEASONS) & cal["outcome"].isin([0.0, 1.0])]
    evalc = evalc.reset_index(drop=True)
    panels = {"Elo": {"before": evalc["p_elo"], "after": evalc["p_elo_platt"]}}
    if "p_xgb" in evalc:
        chosen = load_params().get("classifier", "logit")
        label = {"xgb": "XGBoost", "logit": "Logistic regression"}[chosen]
        panels = {
            label: {"before": evalc[f"p_{chosen}"], "after": evalc[f"p_{chosen}_platt"]},
            **panels,
        }
    calibration = reliability_figure(panels, evalc["outcome"])
    calibration["version"] = calibration["version"].map(
        {"before": "raw", "after": "Platt-calibrated"}
    )

    md = write_leaderboard(
        pooled, by_season, h2h, ranks, validation, len(games), ours, calibration, significance
    )
    (config.RESULTS / "leaderboard.md").write_text(md, encoding="utf-8", newline="\n")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "eval_seasons": list(config.EVAL_SEASONS),
        "headline": ours,
        "candidates": list(CANDIDATES),
        "elo_params": asdict(config.ELO),
        "games": int(len(games)),
        "pooled": pooled.to_dict("records"),
        "by_season": by_season.to_dict("records"),
        "season_ranks": ranks.to_dict("records"),
        "head_to_head": h2h.to_dict("records"),
        "calibration": calibration.to_dict("records"),
        "significance": significance,
        "metric_validation": validation,
    }
    (config.RESULTS / "backtest_results.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )

    def renamed(name: str, p: str, m: str) -> pd.DataFrame:
        return preds[name][["game_id", "p_home", "pred_margin"]].rename(
            columns={"p_home": p, "pred_margin": m}
        )

    game_level = (
        matches.merge(preds[ELO_RAW].drop(columns=["last_result_date"]), on="game_id")
        .merge(renamed(ours, "p_ours", "m_ours"), on="game_id", how="left")
        .merge(renamed("Punters", "p_punters", "m_punters"), on="game_id", how="left")
        .merge(renamed("Aggregate", "p_aggregate", "m_aggregate"), on="game_id", how="left")
    )
    game_level = game_level[game_level["season"] >= config.EVAL_SEASONS[0]]
    game_level = game_level.merge(tip_reasons(matches, preds, cal, ours), on="game_id", how="left")
    game_level.to_parquet(config.RESULTS / "backtest_predictions.parquet", index=False)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    engine.final_ratings().to_parquet(config.DATA_PROCESSED / "elo_ratings.parquet", index=False)

    for name in [c for c in CANDIDATES if c in set(pooled["model"])]:
        r = pooled.set_index("model").loc[name]
        logger.info(
            "%-36s %d/%d tips (%.1f%%), %.1f bits, MAE %.2f, Brier %.4f%s",
            name, r["tips"], r["games"], 100 * r["accuracy"], r["bits"], r["mae"], r["brier"],
            "  <- shipped" if name == ours else "",
        )  # fmt: skip
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

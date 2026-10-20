"""FootyForecast dashboard: This Round · Model vs The Field · Player Archetypes · Player Similarity.

Run locally:  streamlit run app/streamlit_app.py
Everything shown is read from committed artefacts (results/, predictions/), so the app
never calls fitzRoy or Squiggle and stays well inside Streamlit Cloud's 1 GB limit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components import charts, data
from footyforecast.evaluation import metrics  # noqa: E402  (src/ is put on the path by components.data)

st.set_page_config(page_title="FootyForecast", page_icon="🏉", layout="wide")

STYLE = """
<style>
  .block-container { padding-top: 2rem; max-width: 1200px; }
  .ff-sub { color: #52514e; margin-top: -0.6rem; }
  .ff-card { border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 0.9rem 1rem; margin-bottom: 0.8rem; background: #fcfcfb; }
  .ff-card h4 { margin: 0 0 0.2rem 0; font-size: 1.02rem; }
  .ff-meta { color: #898781; font-size: 0.85rem; }
  .ff-banner { border-left: 3px solid #2a78d6; background: #f4f7fc; padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 1rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

summary = data.backtest_summary()
games = data.backtest_games()
feed = data.latest_feed()
pooled = pd.DataFrame(summary["pooled"]).set_index("model")
by_season = pd.DataFrame(summary["by_season"])
OURS = summary.get("headline", "FootyForecast Elo")
seasons = summary["eval_seasons"]

st.title("🏉 FootyForecast")
st.markdown(
    '<p class="ff-sub">AFL win probabilities and margins from a from-scratch Elo system and a '
    "pre-match feature model (form, selection strength, travel), scored walk-forward against the "
    "betting market and Squiggle's public models on Squiggle's own metrics.</p>",
    unsafe_allow_html=True,
)

tab_round, tab_field, tab_arch, tab_sim = st.tabs(
    ["This Round", "Model vs The Field", "Player Archetypes", "Player Similarity"]
)


# ---- Tab 1: This Round ---------------------------------------------------------------------
def tick(correct: bool) -> str:
    return "✓ correct" if correct else "✗ wrong"


def tip_text(p_home: float, home: str, away: str) -> str:
    return f"{home} {p_home:.0%}" if p_home >= 0.5 else f"{away} {1 - p_home:.0%}"


with tab_round:
    if feed["status"] == "in-season" and feed["games"]:
        st.subheader(f"{feed['season']} · {feed['round']}")
        st.caption(
            f"Tips generated {feed['generated_at'][:16].replace('T', ' ')} UTC, before the round starts."
        )
        for g in feed["games"]:
            with st.container():
                st.markdown(
                    f'<div class="ff-card"><h4>{g["hteam"]} v {g["ateam"]}</h4>'
                    f'<div class="ff-meta">{g["venue"]} · {g["kickoff_local"][:16].replace("T", " ")} local</div></div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    charts.win_probability_bar(g["hconfidence"] / 100, g["hteam"], g["ateam"]),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                fav = g["tip"]
                if g.get("why"):
                    st.markdown(f"**Why:** {g['why']}")
                st.caption(
                    f"Tip: **{fav}** by {abs(g['hmargin']):.0f} points · home-ground edge "
                    f"{g['hga_elo']:+.0f} Elo · ratings {g['elo_home']:.0f} v {g['elo_away']:.0f}"
                )
    else:
        last = feed["last_result"]
        st.markdown(
            f'<div class="ff-banner"><b>Off-season.</b> The {last["season"]} season finished on '
            f"{pd.Timestamp(last['date']):%d %B %Y} ({last['match']}). The 2027 fixture has not been "
            "published yet; tips for each round will appear here before the first bounce, and in "
            "<code>predictions/latest.json</code> for Squiggle.</div>",
            unsafe_allow_html=True,
        )
        ratings = pd.DataFrame(feed["ratings"])
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(
                charts.ratings_bars(ratings, "preseason_rating", "2027 pre-season Elo ratings"),
                width="stretch",
            )
        with right:
            st.markdown("**How to read this**")
            st.markdown(
                "Each club's rating after the 2026 Grand Final, pulled part-way back towards the "
                "league average (1500) for list changes over the summer. A 100-point gap is worth "
                f"roughly {100 * summary['elo_params']['points_per_elo']:.0f} points on a neutral ground."
            )
            st.dataframe(
                ratings.rename(
                    columns={
                        "team": "Club",
                        "rating": "End of 2026",
                        "preseason_rating": "2027 start",
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "End of 2026": st.column_config.NumberColumn(format="%.0f"),
                    "2027 start": st.column_config.NumberColumn(format="%.0f"),
                },
            )

    st.divider()
    st.subheader("Replay any round, 2017–2026")
    st.caption(
        "Every tip below was made before that round, using only results available at the time."
    )
    c1, c2 = st.columns([1, 2])
    season = c1.selectbox("Season", list(reversed(seasons)), index=0)
    sg = games[games["season"] == season]
    round_order = sg.groupby("round_name")["date"].min().sort_values().index.tolist()
    rnd = c2.selectbox("Round", round_order, index=len(round_order) - 1)
    rg = sg[sg["round_name"] == rnd].sort_values("kickoff")

    ff_correct = np.where(rg["outcome"] == 0.5, True, (rg["p_ours"] >= 0.5) == (rg["outcome"] == 1))
    mk_correct = np.where(
        rg["outcome"] == 0.5, True, (rg["p_punters"] >= 0.5) == (rg["outcome"] == 1)
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("FootyForecast tips", f"{int(ff_correct.sum())} / {len(rg)}")
    m2.metric("FootyForecast bits", f"{metrics.bits(rg['p_ours'], rg['outcome']).sum():.2f}")
    m3.metric("Market tips", f"{int(mk_correct.sum())} / {len(rg)}")
    m4.metric("Market bits", f"{metrics.bits(rg['p_punters'], rg['outcome']).sum():.2f}")

    table = pd.DataFrame(
        {
            "Date": rg["kickoff"].dt.strftime("%a %d %b"),
            "Match": rg["home_team"] + " v " + rg["away_team"],
            "Venue": rg["venue"],
            "FootyForecast": [
                tip_text(p, h, a)
                for p, h, a in zip(rg["p_ours"], rg["home_team"], rg["away_team"], strict=True)
            ],
            "Predicted margin": [
                f"{(h if m >= 0 else a)} by {abs(m):.0f}"
                for m, h, a in zip(rg["m_ours"], rg["home_team"], rg["away_team"], strict=True)
            ],
            "Market": [
                tip_text(p, h, a) if pd.notna(p) else "–"
                for p, h, a in zip(rg["p_punters"], rg["home_team"], rg["away_team"], strict=True)
            ],
            "Result": [
                f"{h} {int(hs)} – {int(as_)} {a}"
                for h, hs, as_, a in zip(
                    rg["home_team"],
                    rg["home_score"],
                    rg["away_score"],
                    rg["away_team"],
                    strict=True,
                )
            ],
            "FF tip": [tick(c) for c in ff_correct],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    if "why" in rg and rg["why"].notna().any():
        st.markdown("**Why the shipped model tipped each game**")
        st.markdown("\n".join(f"- {w}" for w in rg["why"].dropna()))
    st.caption(
        "Each reason is how many percentage points the tip would move if that factor were "
        "neutral, from exact SHAP values of the logistic model. Where the model leans against "
        "a team's edge (correlated factors), the reason says the edge is discounted."
    )


# ---- Tab 2: Model vs The Field -----------------------------------------------------------------
with tab_field:
    ours = pooled.loc[OURS]
    market = pooled.loc["Punters"]
    n_seasons = len(seasons)
    st.markdown(
        f"Walk-forward backtest over **{seasons[0]}–{seasons[-1]}** ({summary['games']:,} games). "
        f"Shipped model: **{OURS}**. Every tip was made before its round with models refit on "
        "earlier games only; model choices were made on 2014–2016 warm-up seasons, never on the "
        "seasons scored here."
    )
    t1, t2, t3, t4 = st.columns(4)
    t1.metric(
        "Tip accuracy",
        f"{ours['accuracy']:.1%}",
        f"{100 * (ours['accuracy'] - market['accuracy']):+.1f} pts vs market",
    )
    t2.metric(
        "Bits per season",
        f"{ours['bits'] / n_seasons:.1f}",
        f"{(ours['bits'] - market['bits']) / n_seasons:+.1f} vs market",
    )
    t3.metric(
        "Margin MAE (points)",
        f"{ours['mae']:.2f}",
        f"{ours['mae'] - market['mae']:+.2f} vs market",
        delta_color="inverse",
    )
    v = summary["metric_validation"]
    t4.metric(
        f"Scoring matches Squiggle ({v['tips_checked']:,} tips)",
        f"{v['bits_within_0_01']:.0%}",
        help="Share of Squiggle's published per-tip bits that our metrics code reproduces to within 0.01.",
    )

    st.plotly_chart(charts.bits_by_season(by_season, OURS), width="stretch")

    st.subheader("Leaderboard: models that tipped every game")
    full = (
        pooled[pooled["games"] == summary["games"]]
        .sort_values("bits", ascending=False)
        .reset_index()
    )
    full.insert(0, "#", range(1, len(full) + 1))
    st.dataframe(
        full[["#", "model", "tips", "accuracy", "bits", "mae", "brier"]].rename(
            columns={
                "model": "Model",
                "tips": "Tips",
                "accuracy": "Accuracy",
                "bits": "Bits",
                "mae": "MAE",
                "brier": "Brier",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="percent"),
            "Bits": st.column_config.NumberColumn(format="%.1f"),
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "Brier": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        "Honest read: the shipped model beats the Phase 1 Elo by about 20 bits and climbs to "
        "mid-table on Squiggle's recent leaderboards, but most of that gain comes from calibration, "
        "and the betting market and the Squiggle Aggregate remain clearly ahead (see below)."
    )

    with st.expander("Which model ships, and is the difference real?", expanded=True):
        cands = pooled.loc[[c for c in summary["candidates"] if c in pooled.index]]
        st.dataframe(
            cands.reset_index()[["model", "bits", "accuracy", "mae", "brier", "ece"]]
            .sort_values("bits", ascending=False)
            .rename(
                columns={
                    "model": "Candidate",
                    "bits": "Bits",
                    "accuracy": "Accuracy",
                    "mae": "MAE",
                    "brier": "Brier",
                    "ece": "ECE",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="percent"),
                "Bits": st.column_config.NumberColumn(format="%.1f"),
                "MAE": st.column_config.NumberColumn(format="%.2f"),
                "Brier": st.column_config.NumberColumn(format="%.4f"),
                "ECE": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        sig = pd.DataFrame(summary.get("significance", []))
        if len(sig):
            st.markdown(
                "Paired bootstrap over rounds (2,000 resamples). An interval that crosses zero "
                "means the gap could be luck."
            )
            st.dataframe(
                sig.assign(
                    Interval=sig.apply(
                        lambda r: f"[{r['ci_low']:+.1f}, {r['ci_high']:+.1f}]", axis=1
                    )
                )[["b", "bits_diff", "Interval", "p_a_ahead"]].rename(
                    columns={
                        "b": "Shipped vs",
                        "bits_diff": "Δ bits",
                        "p_a_ahead": "Shipped ahead in",
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Δ bits": st.column_config.NumberColumn(format="%+.1f"),
                    "Shipped ahead in": st.column_config.NumberColumn(format="percent"),
                },
            )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.reliability(data.calibration_tables(OURS), OURS), width="stretch")
        st.caption(
            "Points on the diagonal mean the stated probabilities can be taken at face value. The "
            "raw Elo was under-confident at the extremes; walk-forward Platt scaling fixed that "
            "(ECE 0.040 → 0.020). Bins with fewer than 10 games are hidden."
        )
    with right:
        ranks = pd.DataFrame(summary["season_ranks"])
        st.markdown("**Where FootyForecast would have finished on Squiggle's leaderboard (bits)**")
        st.dataframe(
            ranks.assign(
                Finish=ranks["rank_bits"].astype(str) + " of " + ranks["models"].astype(str)
            )[["season", "Finish", "rank_tips", "rank_mae"]].rename(
                columns={"season": "Season", "rank_tips": "Rank on tips", "rank_mae": "Rank on MAE"}
            ),
            hide_index=True,
            width="stretch",
        )

    with st.expander("Same-games comparison with every Squiggle model"):
        h2h = pd.DataFrame(summary["head_to_head"])
        st.dataframe(
            h2h.rename(
                columns={
                    "model": "Model",
                    "seasons": "Seasons",
                    "games": "Games",
                    "their_bits": "Their bits",
                    "our_bits": "FF bits",
                    "bits_delta": "Δ bits",
                    "their_mae": "Their MAE",
                    "our_mae": "FF MAE",
                }
            ),  # fmt: skip
            hide_index=True,
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(format="%.1f")
                for c in ("Their bits", "FF bits", "Δ bits")
            }
            | {c: st.column_config.NumberColumn(format="%.2f") for c in ("Their MAE", "FF MAE")},
        )

# ---- Tab 3: Player Archetypes ---------------------------------------------------------------
from footyforecast.style import FEATURES as STYLE_FEATURES  # noqa: E402

STAT_LABELS = {f: f.replace("_", " ").replace("fifty", "50").capitalize() for f in STYLE_FEATURES}
ps = data.player_seasons()
arch = data.archetypes()

with tab_arch:
    st.markdown(
        f"**{arch['k']} playing-style archetypes** from {arch['player_seasons']:,} player-seasons "
        f"({ps['season'].min()}–{ps['season'].max()}, at least 8 games with 50%+ time on ground), "
        "built from 38 per-game stats measured against each season's peers → PCA → K-means. K was "
        "chosen by silhouette, never by the position labels used to validate it."
    )
    v1, v2, v3 = st.columns(3)
    v1.metric("Adjusted Rand index vs named positions", f"{arch['ari_vs_position']:.2f}")
    v2.metric("NMI vs named positions", f"{arch['nmi_vs_position']:.2f}")
    v3.metric(
        "Hybrid player-seasons",
        f"{(ps['membership'] < 0.7).mean():.0%}",
        help="Mixture-model membership of their archetype below 70%",
    )

    order = list(pd.Series(arch["sizes"]).sort_values(ascending=False).index)
    seasons_sel = st.select_slider(
        "Seasons",
        options=sorted(ps["season"].unique()),
        value=(ps["season"].min(), ps["season"].max()),
    )
    view = ps[ps["season"].between(*seasons_sel)]
    st.plotly_chart(charts.archetype_multiples(view, order), width="stretch")

    pick = st.selectbox("Archetype profile", order)
    left, right = st.columns([3, 2])
    with left:
        profile = pd.Series(arch["profiles"][pick])
        st.plotly_chart(charts.profile_bars(profile, STAT_LABELS), width="stretch")
    with right:
        purity = pd.DataFrame(arch["purity"]).set_index("archetype").loc[pick]
        st.markdown(f"**{pick}**: {arch['sizes'][pick]:,} player-seasons")
        st.markdown(
            f"Most often named at **{purity['top_named_position']}** ({purity['share']:.0%} of the "
            f"archetype); it captures {purity['recall']:.0%} of all named "
            f"{purity['top_named_position']}s."
        )
        latest = ps[(ps["archetype"] == pick) & (ps["season"] == ps["season"].max())]
        st.markdown(f"Highest-rated clear examples in {ps['season'].max()} (fit ≥ 90%):")
        st.dataframe(
            latest[latest["membership"] >= 0.9]
            .sort_values("rating_points", ascending=False)
            .head(8)[["player_name", "team", "games", "membership"]]
            .rename(
                columns={
                    "player_name": "Player",
                    "team": "Club",
                    "games": "Games",
                    "membership": "Fit",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={"Fit": st.column_config.NumberColumn(format="percent")},
        )
    with st.expander("Validation: archetypes × named positions"):
        st.dataframe(pd.DataFrame(arch["crosstab"]).fillna(0).astype(int), width="stretch")
        st.caption(
            "Named positions are team-sheet slots, a noisy label (a 'half-forward' may spend the "
            "day on a wing), so partial agreement is expected. Defenders split by style "
            "(spoil/intercept vs rebound) rather than by key vs general position."
        )


# ---- Tab 4: Player Similarity ------------------------------------------------------------------
from footyforecast.similarity import most_similar, radar  # noqa: E402

with tab_sim:
    st.markdown(
        "Find the players whose statistical *shape* is closest: cosine similarity on 38 style "
        "stats, each measured against its own season, so eras compare fairly."
    )
    c1, c2, c3 = st.columns([1, 2, 1])
    season_s = c1.selectbox("Season", sorted(ps["season"].unique(), reverse=True), key="sim_season")
    pool = ps[ps["season"] == season_s].sort_values("player_name").reset_index(drop=True)
    labels_s = (pool["player_name"] + " (" + pool["team"] + ")").tolist()
    starts = [i for i, lab in enumerate(labels_s) if lab.startswith("Marcus Bontempelli")]
    who = c2.selectbox("Player", labels_s, index=starts[0] if starts else 0)
    any_era = c3.toggle("Search all seasons", value=True)
    row = pool.iloc[labels_s.index(who)]
    mix = sorted(
        [
            (row["archetype"], row["membership"]),
            (row["secondary_archetype"], row["secondary_membership"]),
        ],
        key=lambda t: -t[1],
    )
    st.markdown(
        f"**{row['player_name']}**, {row['team']} {season_s}: archetype **{row['archetype']}** · "
        "style mix " + " · ".join(f"{name} {share:.0%}" for name, share in mix if share >= 0.05)
    )
    sims = most_similar(
        ps,
        row["player_key"],
        int(season_s),
        n=10,
        seasons=None if any_era else (int(season_s), int(season_s)),
    )
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(
            sims.rename(
                columns={
                    "player_name": "Player",
                    "team": "Club",
                    "season": "Season",
                    "archetype": "Archetype",
                    "games": "Games",
                    "similarity": "Similarity",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={"Similarity": st.column_config.NumberColumn(format="%.2f")},
        )
    with right:
        best = sims.iloc[0]
        match = ps[
            (ps["player_name"] == best["player_name"]) & (ps["season"] == best["season"])
        ].iloc[0]
        st.plotly_chart(charts.radar(radar(ps, [row, match])), width="stretch")
        st.caption(
            "Percentile within each player's own season; the radar compares the closest match."
        )


st.divider()
st.caption(
    "Data: AFL Tables and Footywire via the fitzRoy R package; model tips and the betting-market "
    "consensus from the Squiggle API (cached, with an identifying User-Agent). Champion Data's "
    "licensed statistics are not used. Built by Karthik Reddy Tanigundala."
)

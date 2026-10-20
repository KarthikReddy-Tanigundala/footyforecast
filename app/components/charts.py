"""Plotly chart helpers and design tokens for the FootyForecast dashboard.

Colour roles (validated with the dataviz palette checker, all-pairs for 3 series):
    FootyForecast = blue, Punters (betting market) = orange, Aggregate (crowd) = aqua.
A model keeps its colour on every chart, whatever its rank. Aqua sits below 3:1
contrast on the light surface, so every chart that uses it also has direct labels or a
table beside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---- Design tokens -------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#e34948"  # diverging negative pole (below-average ratings)
GOOD, CRITICAL = "#0ca30c", "#d03b3b"  # status: always paired with an icon + label


def model_colors(ours: str) -> dict[str, str]:
    """Fixed colour per role: our shipped model blue, the market orange, the crowd aqua."""
    return {ours: BLUE, "Punters": ORANGE, "Aggregate": AQUA}


FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left")
        if title
        else None,
        height=height,
        margin=dict(l=8, r=8, t=48 if title else 16, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(color=INK_2)
        ),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), ticks="")
    fig.update_yaxes(
        gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED)
    )
    return fig


def bits_by_season(by_season: pd.DataFrame, ours: str) -> go.Figure:
    """Line: bits each season for FootyForecast vs the market vs the crowd (one y-axis)."""
    fig = go.Figure()
    for model, color in model_colors(ours).items():
        d = by_season[by_season["model"] == model].sort_values("season")
        if d.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=d["season"],
                y=d["bits"],
                name=model,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color, line=dict(color=SURFACE, width=2)),
                hovertemplate="<b>%{y:.1f} bits</b> · " + model + "<extra></extra>",
            )
        )
        last = d.iloc[-1]
        fig.add_annotation(
            x=last["season"], y=last["bits"], text=f"{last['bits']:.1f}", showarrow=False,
            xanchor="left", xshift=8, font=dict(color=INK_2, size=11),
        )  # fmt: skip
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(
        title=dict(text="Bits (higher is better)", font=dict(color=MUTED)), rangemode="tozero"
    )
    return _base(fig, "Bits per season: FootyForecast vs the market and the crowd")


def reliability(cal: dict[str, pd.DataFrame], ours: str) -> go.Figure:
    """Reliability diagram: predicted vs observed home-win rate, with the perfect-calibration diagonal."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
            line=dict(color=AXIS, width=1), hoverinfo="skip",
        )
    )  # fmt: skip
    for model, table in cal.items():
        t = table.dropna(subset=["mean_predicted"])
        t = t[t["n"] >= 10]
        color = model_colors(ours).get(model, MUTED)
        fig.add_trace(
            go.Scatter(
                x=t["mean_predicted"],
                y=t["observed_rate"],
                name=model,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=9, color=color, line=dict(color=SURFACE, width=2)),
                customdata=t["n"],
                hovertemplate="<b>%{y:.0%} won</b> when tipped %{x:.0%} · n=%{customdata}<extra>"
                + model
                + "</extra>",
            )
        )
    fig.update_xaxes(
        title=dict(text="Predicted home-win probability", font=dict(color=MUTED)),
        range=[0, 1],
        tickformat=".0%",
    )
    fig.update_yaxes(
        title=dict(text="Observed home-win rate", font=dict(color=MUTED)),
        range=[0, 1],
        tickformat=".0%",
    )
    fig = _base(fig, "Calibration (reliability diagram)", height=440)
    # Legend below the plot: the square plot area leaves no room beside a long title.
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0), margin=dict(b=8)
    )
    return fig


def ratings_bars(ratings: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    """Horizontal diverging bars: each club's rating relative to the league average (1500)."""
    d = ratings.sort_values(value_col)
    delta = d[value_col] - 1500
    colors = np.where(delta >= 0, BLUE, RED)
    fig = go.Figure(
        go.Bar(
            x=delta,
            y=d["team"],
            orientation="h",
            marker=dict(color=colors, line=dict(color=SURFACE, width=2)),
            width=0.6,
            customdata=d[value_col],
            hovertemplate="<b>%{customdata:.0f}</b> (%{x:+.0f} vs average)<extra>%{y}</extra>",
            text=[f"{v:.0f}" for v in d[value_col]],
            textposition="outside",
            textfont=dict(color=INK_2, size=11),
            cliponaxis=False,
        )
    )
    fig.update_xaxes(
        title=dict(text="Elo points above / below league average (1500)", font=dict(color=MUTED)),
        zeroline=True,
        zerolinecolor=AXIS,
    )
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2))
    return _base(fig, title, height=560)


def win_probability_bar(p_home: float, home: str, away: str) -> go.Figure:
    """Single split bar for one game: home share vs away share of the win probability."""
    fig = go.Figure()
    for share, team, color in ((p_home, home, BLUE), (1 - p_home, away, ORANGE)):
        fig.add_trace(
            go.Bar(
                x=[share], y=[""], orientation="h", marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                name=team, text=f"{team} {share:.0%}", textposition="inside", insidetextanchor="middle",
                textfont=dict(color="white", size=12), hovertemplate=f"<b>{share:.0%}</b> {team}<extra></extra>",
            )
        )  # fmt: skip
    fig.update_layout(barmode="stack", showlegend=False, height=46, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False)
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE)
    return fig

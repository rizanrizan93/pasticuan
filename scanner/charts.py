from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go


def make_signal_chart(frame: pd.DataFrame, signal: Mapping[str, Any], bars: int = 180) -> go.Figure:
    data = frame.iloc[-bars:].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="OHLC",
            increasing_line_color="#20c997",
            decreasing_line_color="#ff5c6c",
        )
    )
    for column, color, width in (
        ("EMA20", "#f6c85f", 1.3),
        ("EMA50", "#6f9ceb", 1.3),
        ("EMA200", "#ad75f4", 1.5),
    ):
        if column in data:
            fig.add_trace(
                go.Scatter(x=data.index, y=data[column], name=column, line=dict(color=color, width=width))
            )
    zone_low, zone_high = signal.get("entry_low"), signal.get("entry_high")
    if pd.notna(zone_low) and pd.notna(zone_high):
        fig.add_hrect(
            y0=float(zone_low),
            y1=float(zone_high),
            fillcolor="rgba(38, 166, 154, 0.17)",
            line_width=0,
            annotation_text="Entry zone",
            annotation_position="top left",
        )
    levels = (
        ("entry", "Entry", "#22d3ee", "dash"),
        ("stop_loss", "SL", "#ff5c6c", "solid"),
        ("tp1", "TP1", "#f6c85f", "dot"),
        ("tp2", "TP2", "#20c997", "dot"),
    )
    for key, label, color, dash in levels:
        value = signal.get(key)
        if pd.notna(value):
            fig.add_hline(
                y=float(value),
                line_color=color,
                line_dash=dash,
                line_width=1.25,
                annotation_text=f"{label} {float(value):,.0f}",
                annotation_position="right",
            )
    fig.update_layout(
        title=f"{signal.get('ticker', '')} · {signal.get('setup', '')}",
        template="plotly_dark",
        height=620,
        margin=dict(l=20, r=80, t=55, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ScanConfig


SETUPS = (
    "PULLBACK_CONTINUATION",
    "BREAKOUT_RETEST",
    "REVERSAL_ACCUMULATION",
    "UNICORN_SNIPER_ICT",
)


@dataclass
class BacktestTrade:
    ticker: str
    setup: str
    signal_date: object
    entry_date: object
    exit_date: object
    entry: float
    stop: float
    target: float
    exit_price: float
    result: str
    r_multiple: float
    holding_bars: int


def _breakout_retest_mask(df: pd.DataFrame) -> pd.Series:
    breakout = (
        (df["Close"] > df["HIGH55_PREV"] + 0.05 * df["ATR14"])
        & (df["VOL_RATIO"] >= 1.25)
        & (df["BODY_ATR"] >= 0.40)
        & (df["Close"] > df["Open"])
    )
    signal = pd.Series(False, index=df.index)
    positions = np.flatnonzero(breakout.fillna(False).to_numpy())
    for pos in positions:
        level = float(df["HIGH55_PREV"].iloc[pos])
        end = min(len(df), pos + 11)
        for j in range(pos + 1, end):
            atr_value = float(df["ATR14"].iloc[j])
            if not math.isfinite(atr_value) or atr_value <= 0:
                continue
            touched = df["Low"].iloc[j] <= level + 0.45 * atr_value
            held = df["Close"].iloc[j] >= level - 0.10 * atr_value
            confirmed = bool(df["BULL_REJECTION"].iloc[j]) or df["Close"].iloc[j] > df["Open"].iloc[j]
            if touched and held and confirmed:
                signal.iloc[j] = True
                break
            if df["Close"].iloc[j] < level - 1.15 * atr_value:
                break
    return signal


def historical_signal_mask(df: pd.DataFrame, setup: str) -> pd.Series:
    if setup == "PULLBACK_CONTINUATION":
        trend = (df["EMA20"] > df["EMA50"]) & (df["EMA50"] > df["EMA200"]) & (df["Close"] > df["EMA50"])
        momentum = (df["ROC60"] > 0.04) & (df["DIST_52W_HIGH"] > -0.18)
        touch = (df["Low"] <= df["EMA20"] + 0.35 * df["ATR14"]) & (df["Close"] >= df["EMA50"])
        confirm = df["BULL_REJECTION"] | ((df["Close"] > df["High"].shift(1)) & (df["Close"] > df["Open"]))
        vol_ok = df["Volume"].rolling(3).mean() <= 1.05 * df["VOL_MA20"]
        return (trend & momentum & touch & confirm & vol_ok).fillna(False)
    if setup == "BREAKOUT_RETEST":
        return _breakout_retest_mask(df)
    if setup == "REVERSAL_ACCUMULATION":
        prior_high = df["High"].shift(30).rolling(120).max()
        decline = df["Close"] / prior_high - 1 <= -0.12
        width = (df["High"].rolling(30).max() - df["Low"].rolling(30).min()) / df["Close"]
        based = width <= 0.32
        accumulation = (df["CMF20"] > 0.02) & (df["OBV_SLOPE10"] > 0)
        sweep = (df["Low"] < df["LOW20_PREV"]) & (df["Close"] > df["LOW20_PREV"])
        sweep_recent = sweep.astype(float).rolling(15).max().gt(0)
        choch = (df["Close"] > df["LAST_PIVOT_HIGH"] + 0.05 * df["ATR14"]) & (df["VOL_RATIO"] >= 1.05)
        return (decline & based & accumulation & sweep_recent & choch).fillna(False)
    if setup == "UNICORN_SNIPER_ICT":
        sweep = (
            (df["Low"] < df["LOW20_PREV"])
            & (df["Close"] > df["LOW20_PREV"])
            & (df["CLOSE_LOCATION"] >= 0.55)
        )
        sweep_recent = sweep.astype(float).rolling(25).max().gt(0)
        bos = (df["Close"] > df["LAST_PIVOT_HIGH"] + 0.05 * df["ATR14"]) & (df["BODY_ATR"] >= 0.55)
        fvg_recent = df["BULL_FVG"].astype(float).rolling(5).max().gt(0)
        return (sweep_recent & bos & fvg_recent).fillna(False)
    raise ValueError(f"Unknown setup: {setup}")


def simulate_setup(
    df: pd.DataFrame, ticker: str, setup: str, config: ScanConfig
) -> list[BacktestTrade]:
    mask = historical_signal_mask(df, setup)
    tradeable = (
        (df["Close"] >= config.min_price)
        & (df["ADTV20"] >= config.min_adtv_idr)
        & (df["ZERO_VOL20"] <= config.max_zero_volume_ratio)
        & (df["ATR_PCT"] >= config.min_atr_pct)
        & (df["ATR_PCT"] <= config.max_atr_pct)
    )
    if "BENCH_CLOSE" in df and df["BENCH_CLOSE"].notna().any():
        regime_ok = ~(
            (df["BENCH_CLOSE"] < df["BENCH_EMA200"])
            & (df["BENCH_ROC20"] < 0)
        )
        tradeable &= regime_ok.fillna(False)
    mask &= tradeable.fillna(False)
    candidates = np.flatnonzero(mask.to_numpy())
    trades: list[BacktestTrade] = []
    next_allowed = 0
    horizon = config.backtest_horizon_bars
    for pos in candidates:
        if pos < max(205, next_allowed) or pos + 1 >= len(df):
            continue
        atr_value = float(df["ATR14"].iloc[pos])
        if not math.isfinite(atr_value) or atr_value <= 0:
            continue
        entry_pos = pos + 1
        entry = float(df["Open"].iloc[entry_pos])
        if not math.isfinite(entry) or entry <= 0:
            continue
        risk = max(1.35 * atr_value, entry * 0.025)
        stop = entry - risk
        target = entry + config.backtest_target_rr * risk
        if stop <= 0:
            continue
        last_pos = min(len(df) - 1, entry_pos + horizon - 1)
        exit_price = float(df["Close"].iloc[last_pos])
        exit_pos = last_pos
        result = "TIME_EXIT"
        for j in range(entry_pos, last_pos + 1):
            day_open = float(df["Open"].iloc[j])
            day_low = float(df["Low"].iloc[j])
            day_high = float(df["High"].iloc[j])
            hit_stop = day_low <= stop
            hit_target = day_high >= target
            if day_open <= stop:
                exit_price, exit_pos, result = day_open, j, "LOSS_GAP"
                break
            # Conservative convention: if both barriers appear inside one daily
            # candle, assume the stop was touched first.
            if hit_stop:
                exit_price, exit_pos, result = stop, j, "LOSS"
                break
            if hit_target:
                exit_price, exit_pos, result = target, j, "WIN"
                break
        net_return = (
            exit_price / entry
            - 1.0
            - config.fee_roundtrip_pct
            - config.slippage_roundtrip_pct
        )
        r_multiple = net_return / (risk / entry)
        trades.append(
            BacktestTrade(
                ticker=ticker,
                setup=setup,
                signal_date=df.index[pos],
                entry_date=df.index[entry_pos],
                exit_date=df.index[exit_pos],
                entry=entry,
                stop=stop,
                target=target,
                exit_price=exit_price,
                result=result,
                r_multiple=round(float(r_multiple), 4),
                holding_bars=int(exit_pos - entry_pos + 1),
            )
        )
        next_allowed = max(pos + config.backtest_min_gap_bars, exit_pos + 1)
    return trades


def _max_losing_streak(values: Iterable[float]) -> int:
    maximum = current = 0
    for value in values:
        if value <= 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def aggregate_backtest(trades: pd.DataFrame, config: ScanConfig) -> pd.DataFrame:
    columns = [
        "setup",
        "historical_events",
        "historical_hit_rate",
        "bayes_probability",
        "expectancy_r",
        "profit_factor",
        "max_losing_streak",
        "sample_reliability",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for setup, group in trades.groupby("setup", sort=False):
        r = group["r_multiple"].astype(float)
        count = len(group)
        wins = int(group["result"].eq("WIN").sum()) if "result" in group else int((r > 0).sum())
        hit_rate = wins / count if count else np.nan
        bayes = (wins + config.beta_prior_wins) / (
            count + config.beta_prior_wins + config.beta_prior_losses
        )
        gross_win = r[r > 0].sum()
        gross_loss = -r[r <= 0].sum()
        pf = gross_win / gross_loss if gross_loss > 0 else np.nan
        reliability = "HIGH" if count >= 50 else "MEDIUM" if count >= 20 else "LOW"
        rows.append(
            {
                "setup": setup,
                "historical_events": count,
                "historical_hit_rate": round(100 * hit_rate, 1),
                "bayes_probability": round(100 * bayes, 1),
                "expectancy_r": round(float(r.mean()), 3),
                "profit_factor": round(float(pf), 2) if np.isfinite(pf) else np.nan,
                "max_losing_streak": _max_losing_streak(r.tolist()),
                "sample_reliability": reliability,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def run_walkforward_validation(
    prepared: dict[str, pd.DataFrame], config: ScanConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or ScanConfig()
    all_trades: list[dict[str, object]] = []
    for ticker, frame in prepared.items():
        if len(frame) < 225:
            continue
        for setup in SETUPS:
            for trade in simulate_setup(frame, ticker, setup, cfg):
                all_trades.append(trade.__dict__)
    trades = pd.DataFrame(all_trades)
    stats = aggregate_backtest(trades, cfg)
    return stats, trades


def attach_backtest_stats(signals: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    if signals.empty or stats.empty:
        result = signals.copy()
        for column in (
            "historical_events",
            "historical_hit_rate",
            "bayes_probability",
            "expectancy_r",
            "profit_factor",
            "max_losing_streak",
            "sample_reliability",
        ):
            if column not in result:
                result[column] = np.nan
        return result
    return signals.merge(stats, on="setup", how="left")

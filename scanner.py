"""IDX Super Scanner Hardened: single-file production core."""

from __future__ import annotations

__version__ = "2.1.0-hardened-flat"


# ---- models ----

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MarketContext:
    regime: str = "UNKNOWN"
    benchmark_close: float | None = None
    benchmark_roc20: float | None = None
    breadth_ema50: float | None = None
    breadth_ema200: float | None = None
    reason: str = "Benchmark tidak tersedia"


@dataclass
class SetupPlan:
    ticker: str
    setup: str
    detected: bool
    setup_score: float
    signal_date: Any = None
    zone_created_date: Any = None
    entry_low: float | None = None
    entry_high: float | None = None
    entry: float | None = None
    entry_type: str = "CONDITIONAL"
    trigger: float | None = None
    stop_loss: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    tp1_basis: str = "R_MULTIPLE"
    tp2_basis: str = "R_MULTIPLE"
    rr1: float | None = None
    rr2: float | None = None
    distance_atr: float | None = None
    zone_age_bars: int | None = None
    valid_until: Any = None
    invalidated: bool = False
    action: str = "NO_SETUP"
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence"] = " • ".join(self.evidence)
        result["blockers"] = " • ".join(self.blockers)
        return result


@dataclass
class DownloadReport:
    requested: list[str]
    downloaded: list[str]
    failed: dict[str, str]
    benchmark_ok: bool = False
    provider: str = "Yahoo Finance via yfinance"
    adjusted_prices: bool = True
    downloaded_at: Any = None
    warnings: dict[str, str] = field(default_factory=dict)
    source_tiers: dict[str, str] = field(default_factory=dict)


# ---- config ----

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanConfig:
    """Fail-closed defaults for daily-bar IDX scanning and Stockbit sizing."""

    min_bars: int = 220
    min_price: float = 50.0
    min_adtv_idr: float = 2_000_000_000.0
    min_atr_pct: float = 0.008
    max_atr_pct: float = 0.12
    max_zero_volume_ratio: float = 0.10
    min_score: float = 70.0
    execution_score: float = 78.0
    min_rr1: float = 1.80
    min_rr2: float = 2.50
    max_stop_pct: float = 0.08
    ready_distance_atr: float = 0.35
    max_entry_gap_atr: float = 0.20
    watch_distance_atr: float = 2.0
    max_zone_age_bars: int = 30
    max_data_lag_days: int = 5
    max_absolute_data_age_days: int = 10
    fee_roundtrip_pct: float = 0.0045
    slippage_roundtrip_pct: float = 0.0020
    backtest_horizon_bars: int = 20
    backtest_entry_window_bars: int = 5
    backtest_min_gap_bars: int = 10
    walkforward_min_train_fraction: float = 0.60
    walkforward_folds: int = 4
    beta_prior_wins: float = 8.0
    beta_prior_losses: float = 8.0
    fundamental_top_n: int = 50
    min_fundamental_coverage: float = 60.0
    min_fundamental_score: float = 55.0
    real_money_mode: bool = True
    require_fundamentals: bool = True
    require_market_status: bool = True
    require_news_review: bool = True
    max_context_age_days: int = 5

    # Legacy v2 defaults; the effective v4.2.5 ScanConfig is defined below.
    account_size_idr: float = 10_000_000.0
    risk_per_trade_pct: float = 0.01
    max_portfolio_risk_pct: float = 0.02
    max_positions: int = 2
    max_position_pct: float = 0.40
    buy_fee_pct: float = 0.0015
    sell_fee_pct: float = 0.0025
    order_slippage_pct: float = 0.0020

    def replace(self, **changes: object) -> "ScanConfig":
        values = self.__dict__.copy()
        values.update(changes)
        return ScanConfig(**values)


# ---- price_rules ----

import math


def idx_tick_size(price: float) -> int:
    """Return the IDX regular-market price fraction effective since May 2016."""
    if not math.isfinite(price) or price <= 0:
        return 1
    if price < 200:
        return 1
    if price < 500:
        return 2
    if price < 2_000:
        return 5
    if price < 5_000:
        return 10
    return 25


def round_idx_price(price: float | None, direction: str = "nearest") -> float | None:
    if price is None or not math.isfinite(float(price)) or float(price) <= 0:
        return None
    value = float(price)
    tick = idx_tick_size(value)
    scaled = value / tick
    if direction == "up":
        rounded = math.ceil(scaled - 1e-12) * tick
    elif direction == "down":
        rounded = math.floor(scaled + 1e-12) * tick
    else:
        rounded = round(scaled) * tick
    # A rounded value may cross into a new fraction band. One more pass makes
    # boundary prices (200, 500, 2,000, 5,000) exchange-valid.
    tick2 = idx_tick_size(float(rounded))
    scaled2 = rounded / tick2
    if direction == "up":
        rounded = math.ceil(scaled2 - 1e-12) * tick2
    elif direction == "down":
        rounded = math.floor(scaled2 + 1e-12) * tick2
    else:
        rounded = round(scaled2) * tick2
    return float(max(1, rounded))


def idx_ara_pct(reference_price: float) -> float:
    if reference_price <= 200:
        return 0.35
    if reference_price <= 5_000:
        return 0.25
    return 0.20


def idx_arb_pct(_reference_price: float) -> float:
    return 0.15


def is_valid_idx_price(price: float | None) -> bool:
    """Return True when a regular-market order price is positive and on tick."""
    if price is None:
        return False
    try:
        value = float(price)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(value) or value < 50:
        return False
    tick = idx_tick_size(value)
    return abs(value / tick - round(value / tick)) <= 1e-9


def idx_daily_price_band(reference_price: float) -> tuple[float | None, float | None]:
    """Conservative regular-board ARB/ARA band rounded to valid IDX ticks."""
    if not math.isfinite(reference_price) or reference_price <= 0:
        return None, None
    lower = round_idx_price(reference_price * (1 - idx_arb_pct(reference_price)), "up")
    upper = round_idx_price(reference_price * (1 + idx_ara_pct(reference_price)), "down")
    return lower, upper


def within_idx_daily_price_band(price: float | None, reference_price: float) -> bool:
    if not is_valid_idx_price(price):
        return False
    lower, upper = idx_daily_price_band(reference_price)
    return bool(lower is not None and upper is not None and lower <= float(price) <= upper)


def near_upper_auto_rejection(previous_close: float, close: float, high: float) -> bool:
    if previous_close <= 0:
        return False
    daily_return = close / previous_close - 1.0
    locked_at_high = abs(high - close) <= idx_tick_size(close) * 0.51
    return bool(locked_at_high and daily_return >= 0.90 * idx_ara_pct(previous_close))


# ---- indicators ----

import numpy as np
import pandas as pd


OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift(1)
    return pd.concat(
        [(df["High"] - df["Low"]), (df["High"] - prev).abs(), (df["Low"] - prev).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.where(loss.ne(0), 100.0).fillna(50.0)


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_ = atr(df, length).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean().fillna(0.0)


def cmf(df: pd.DataFrame, length: int = 20) -> pd.Series:
    spread = (df["High"] - df["Low"]).replace(0, np.nan)
    multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / spread
    money_flow = multiplier.fillna(0.0) * df["Volume"]
    return money_flow.rolling(length).sum() / df["Volume"].rolling(length).sum().replace(0, np.nan)


def mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    raw = typical * df["Volume"]
    direction = typical.diff()
    positive = raw.where(direction > 0, 0.0).rolling(length).sum()
    negative = raw.where(direction < 0, 0.0).rolling(length).sum()
    ratio = positive / negative.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50.0)


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["Close"].diff()).fillna(0.0)
    return (sign * df["Volume"]).cumsum()


def confirmed_pivot(series: pd.Series, left: int = 3, right: int = 3, mode: str = "high") -> pd.Series:
    window = left + right + 1
    roll = series.rolling(window, center=True, min_periods=window)
    raw = series.eq(roll.max()) if mode == "high" else series.eq(roll.min())
    # Place the pivot value on its confirmation bar. No future observation is
    # available to a historical signal before this shifted timestamp.
    return series.where(raw).shift(right)


def prepare_indicators(df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    for col in OHLCV:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    out["Volume"] = out["Volume"].fillna(0.0).clip(lower=0)

    for length in (10, 20, 50, 100, 200):
        out[f"EMA{length}"] = ema(out["Close"], length)
    out["ATR14"] = atr(out, 14)
    out["ATR_PCT"] = out["ATR14"] / out["Close"].replace(0, np.nan)
    out["RSI14"] = rsi(out["Close"], 14)
    out["ADX14"] = adx(out, 14)
    macd = ema(out["Close"], 12) - ema(out["Close"], 26)
    signal = ema(macd, 9)
    out["MACD"] = macd
    out["MACD_HIST"] = macd - signal
    out["CMF20"] = cmf(out, 20)
    out["MFI14"] = mfi(out, 14)
    out["OBV"] = obv(out)
    out["OBV_SLOPE10"] = out["OBV"].diff(10) / out["Volume"].rolling(20).mean().replace(0, np.nan)

    out["VOL_MA20"] = out["Volume"].rolling(20).mean()
    out["VOL_RATIO"] = out["Volume"] / out["VOL_MA20"].replace(0, np.nan)
    out["VALUE"] = out["Close"] * out["Volume"]
    out["ADTV20"] = out["VALUE"].rolling(20).mean()
    out["ZERO_VOL20"] = out["Volume"].eq(0).rolling(20).mean()
    typical = (out["High"] + out["Low"] + out["Close"]) / 3
    out["VWAP20"] = (typical * out["Volume"]).rolling(20).sum() / out["Volume"].rolling(20).sum().replace(0, np.nan)

    for length in (20, 60, 120):
        out[f"ROC{length}"] = out["Close"].pct_change(length)
    out["HIGH20_PREV"] = out["High"].shift(1).rolling(20).max()
    out["HIGH55_PREV"] = out["High"].shift(1).rolling(55).max()
    out["HIGH252"] = out["High"].rolling(252, min_periods=120).max()
    out["LOW20_PREV"] = out["Low"].shift(1).rolling(20).min()
    out["LOW55_PREV"] = out["Low"].shift(1).rolling(55).min()
    out["DIST_52W_HIGH"] = out["Close"] / out["HIGH252"].replace(0, np.nan) - 1

    out["PIVOT_HIGH"] = confirmed_pivot(out["High"], 3, 3, "high")
    out["PIVOT_LOW"] = confirmed_pivot(out["Low"], 3, 3, "low")
    out["LAST_PIVOT_HIGH"] = out["PIVOT_HIGH"].ffill()
    out["LAST_PIVOT_LOW"] = out["PIVOT_LOW"].ffill()

    body = (out["Close"] - out["Open"]).abs()
    candle_range = (out["High"] - out["Low"]).replace(0, np.nan)
    out["BODY_ATR"] = body / out["ATR14"].replace(0, np.nan)
    out["CLOSE_LOCATION"] = (out["Close"] - out["Low"]) / candle_range
    out["BULL_CANDLE"] = out["Close"] > out["Open"]
    out["BEAR_CANDLE"] = out["Close"] < out["Open"]
    out["BULL_REJECTION"] = (out["CLOSE_LOCATION"] > 0.65) & (out["Close"] > out["Open"])
    out["RANGE_CONTRACTION20"] = out["ATR14"] / out["ATR14"].rolling(60).median().replace(0, np.nan)

    # Bullish fair-value gap with displacement. The condition is known at t.
    out["BULL_FVG"] = (
        (out["Low"] > out["High"].shift(2))
        & (out["Close"].shift(1) > out["Open"].shift(1))
        & (out["BODY_ATR"].shift(1) >= 0.65)
        & (out["VOL_RATIO"].shift(1) >= 1.15)
    )
    out["FVG_LOW"] = out["High"].shift(2).where(out["BULL_FVG"])
    out["FVG_HIGH"] = out["Low"].where(out["BULL_FVG"])

    if benchmark is not None and not benchmark.empty and "Close" in benchmark:
        bench_close = benchmark["Close"].reindex(out.index).ffill()
        out["BENCH_CLOSE"] = bench_close
        out["BENCH_EMA50"] = ema(bench_close, 50)
        out["BENCH_EMA200"] = ema(bench_close, 200)
        out["BENCH_ROC20"] = bench_close.pct_change(20)
        out["REL_STRENGTH60"] = out["ROC60"] - bench_close.pct_change(60)
    else:
        out["BENCH_CLOSE"] = np.nan
        out["BENCH_EMA50"] = np.nan
        out["BENCH_EMA200"] = np.nan
        out["BENCH_ROC20"] = np.nan
        out["REL_STRENGTH60"] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


# ---- setups ----

import math
from typing import Callable

import numpy as np
import pandas as pd



def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _last_true_index(mask: pd.Series, lookback: int) -> object | None:
    recent = mask.fillna(False).iloc[-lookback:]
    hits = recent[recent]
    return hits.index[-1] if len(hits) else None


def _bars_since(df: pd.DataFrame, timestamp: object | None) -> int | None:
    if timestamp is None:
        return None
    locations = np.flatnonzero(df.index == timestamp)
    if len(locations) == 0:
        return None
    return int(len(df) - 1 - locations[-1])


def _distance_to_zone(close: float, zone_low: float, zone_high: float, atr: float) -> float:
    if atr <= 0:
        return float("inf")
    if zone_low <= close <= zone_high:
        return 0.0
    if close > zone_high:
        return (close - zone_high) / atr
    return (zone_low - close) / atr


def _plan_prices(
    plan: SetupPlan,
    df: pd.DataFrame,
    atr_value: float,
    raw_entry: float,
    raw_stop: float,
    tp1_rr: float = 1.8,
    tp2_rr: float = 3.0,
) -> SetupPlan:
    if plan.entry_low is None or plan.entry_high is None or plan.entry_low >= plan.entry_high:
        plan.invalidated = True
        plan.reason = "Zona entry tidak memiliki rentang harga yang valid"
        return plan
    plan.entry_low = round_idx_price(plan.entry_low, "down")
    plan.entry_high = round_idx_price(plan.entry_high, "up")
    plan.entry = round_idx_price(raw_entry, "up")
    plan.trigger = round_idx_price(plan.trigger if plan.trigger is not None else raw_entry, "up")
    plan.stop_loss = round_idx_price(raw_stop, "down")
    if plan.entry is None or plan.stop_loss is None or plan.stop_loss >= plan.entry:
        plan.invalidated = True
        plan.reason = "Struktur tidak menghasilkan risiko positif yang valid"
        return plan
    risk = plan.entry - plan.stop_loss
    # Use only confirmed, already-observable resistance. A structural level is
    # selected only when it still preserves the minimum R multiple; otherwise
    # the target falls back to an explicit R objective instead of inventing a
    # future resistance.
    resistance: list[float] = []
    for column in ("PIVOT_HIGH", "HIGH20_PREV", "HIGH55_PREV", "HIGH252"):
        if column not in df:
            continue
        for value in pd.to_numeric(df[column].iloc[-252:], errors="coerce").dropna().tolist():
            number = _finite(value)
            if number > plan.entry:
                resistance.append(number)
    resistance = sorted(set(resistance))

    def target_for(minimum_rr: float, after: float = 0.0) -> tuple[float | None, str]:
        floor = plan.entry + minimum_rr * risk
        candidates = [level for level in resistance if level >= floor and level > after]
        if candidates:
            raw = max(floor, candidates[0] - idx_tick_size(candidates[0]))
            rounded = round_idx_price(raw, "down")
            if rounded is None or rounded < floor:
                rounded = round_idx_price(floor, "up")
            return rounded, "CONFIRMED_RESISTANCE"
        return round_idx_price(floor, "up"), f"{minimum_rr:.1f}R_FALLBACK"

    plan.tp1, plan.tp1_basis = target_for(tp1_rr)
    plan.tp2, plan.tp2_basis = target_for(tp2_rr, float(plan.tp1 or 0))
    if plan.tp1 is not None:
        plan.rr1 = round((plan.tp1 - plan.entry) / risk, 2)
    if plan.tp2 is not None:
        plan.rr2 = round((plan.tp2 - plan.entry) / risk, 2)
    return plan


def detect_pullback_continuation(df: pd.DataFrame, ticker: str) -> SetupPlan:
    name = "PULLBACK_CONTINUATION"
    plan = SetupPlan(ticker=ticker, setup=name, detected=False, setup_score=0.0)
    if len(df) < 205:
        plan.reason = "Data tren jangka panjang belum cukup"
        return plan
    row = df.iloc[-1]
    prev = df.iloc[-2]
    close = _finite(row["Close"])
    atr_v = _finite(row["ATR14"])
    if close <= 0 or atr_v <= 0:
        plan.reason = "ATR/harga tidak valid"
        return plan

    ema20, ema50, ema200 = (_finite(row[x]) for x in ("EMA20", "EMA50", "EMA200"))
    trend = ema20 > ema50 > ema200 and close > ema50 and ema20 > _finite(df["EMA20"].iloc[-11])
    momentum = _finite(row["ROC60"], -1) > 0.04 and _finite(row["DIST_52W_HIGH"], -1) > -0.18
    support = ema20 if close >= ema20 - 0.45 * atr_v else max(ema50, _finite(row["VWAP20"]))
    recent = df.iloc[-5:]
    touched = bool((recent["Low"] <= recent["EMA20"] + 0.35 * recent["ATR14"]).any())
    held = close >= ema50 - 0.25 * atr_v
    pullback = trend and touched and held
    confirmation = bool(row["BULL_REJECTION"]) or (
        close > _finite(prev["High"]) and close > _finite(row["Open"])
    )
    vol_contract = _finite(recent["Volume"].iloc[:-1].mean()) < 0.92 * _finite(df["VOL_MA20"].iloc[-1], 1)
    relative = _finite(row["REL_STRENGTH60"], 0) > 0

    score = 0.0
    score += 25 if trend else 0
    score += 15 if momentum else 0
    score += 18 if pullback else (8 if trend and held else 0)
    score += 10 if vol_contract else 4
    score += 12 if confirmation else 3
    score += 10 if _finite(row["CMF20"]) > -0.03 else 0
    score += 10 if relative else 4
    plan.setup_score = min(100.0, score)
    plan.detected = bool(trend and momentum and pullback)
    if not plan.detected:
        plan.reason = "Belum memenuhi kombinasi uptrend, momentum, dan pullback ke value area"
        return plan

    touch_mask = (df["Low"] <= df["EMA20"] + 0.35 * df["ATR14"]) & (df["Close"] >= df["EMA50"])
    created = _last_true_index(touch_mask, 10) or df.index[-1]
    zone_low = support - 0.35 * atr_v
    zone_high = support + 0.40 * atr_v
    recent_low = _finite(df["Low"].iloc[-7:].min())
    pivot_low = _finite(row["LAST_PIVOT_LOW"], recent_low)
    structural_low = min(recent_low, pivot_low) if pivot_low > close - 4 * atr_v else recent_low
    raw_stop = structural_low - 0.20 * atr_v
    if confirmation and _distance_to_zone(close, zone_low, zone_high, atr_v) <= 0.5:
        raw_entry = max(close, _finite(row["High"]) + idx_tick_size(close))
        plan.entry_type = "BUY_STOP_CONFIRMATION"
        plan.action = "READY_TRIGGER"
    else:
        raw_entry = (zone_low + zone_high) / 2
        plan.entry_type = "LIMIT_ON_PULLBACK_THEN_CONFIRM"
        plan.action = "WAIT_PULLBACK_CONFIRMATION"
    plan.signal_date = df.index[-1]
    plan.zone_created_date = created
    plan.zone_age_bars = _bars_since(df, created)
    plan.valid_until = pd.Timestamp(df.index[-1]) + pd.offsets.BDay(10)
    plan.entry_low, plan.entry_high = zone_low, zone_high
    plan.trigger = _finite(row["High"]) + idx_tick_size(close)
    plan.distance_atr = round(_distance_to_zone(close, zone_low, zone_high, atr_v), 2)
    plan.evidence = [
        "EMA20 > EMA50 > EMA200",
        "Momentum 3 bulan positif",
        "Pullback menyentuh value area",
    ]
    if vol_contract:
        plan.evidence.append("Volume mengecil saat pullback")
    if confirmation:
        plan.evidence.append("Ada reclaim/rejection bullish")
    plan.reason = "Kelanjutan tren setelah pullback terkontrol"
    return _plan_prices(plan, df, atr_v, raw_entry, raw_stop)


def detect_breakout_retest(df: pd.DataFrame, ticker: str) -> SetupPlan:
    name = "BREAKOUT_RETEST"
    plan = SetupPlan(ticker=ticker, setup=name, detected=False, setup_score=0.0)
    if len(df) < 205:
        plan.reason = "Data belum cukup"
        return plan
    row = df.iloc[-1]
    close, atr_v = _finite(row["Close"]), _finite(row["ATR14"])
    if close <= 0 or atr_v <= 0:
        plan.reason = "ATR/harga tidak valid"
        return plan
    breakout_mask = (
        (df["Close"] > df["HIGH55_PREV"] + 0.05 * df["ATR14"])
        & (df["VOL_RATIO"] >= 1.25)
        & (df["BODY_ATR"] >= 0.40)
        & (df["Close"] > df["Open"])
    )
    breakout_date = _last_true_index(breakout_mask, 18)
    if breakout_date is None:
        plan.reason = "Belum ada breakout 55-hari dengan volume dan displacement"
        return plan
    pos = int(np.flatnonzero(df.index == breakout_date)[-1])
    breakout_row = df.iloc[pos]
    resistance = _finite(breakout_row["HIGH55_PREV"])
    breakout_atr = _finite(breakout_row["ATR14"], atr_v)
    post = df.iloc[pos + 1 :] if pos + 1 < len(df) else df.iloc[0:0]
    retest_mask = (
        (post["Low"] <= resistance + 0.45 * post["ATR14"])
        & (post["Low"] >= resistance - 1.0 * post["ATR14"])
        & (post["Close"] >= resistance - 0.10 * post["ATR14"])
    )
    retest_date = _last_true_index(retest_mask, min(12, len(post))) if not post.empty else None
    invalidated = bool((post["Close"] < resistance - 1.15 * post["ATR14"]).any()) if not post.empty else False
    confirmation = False
    retest_low = resistance - 0.6 * atr_v
    if retest_date is not None:
        rpos = int(np.flatnonzero(df.index == retest_date)[-1])
        retest_low = _finite(df["Low"].iloc[max(pos + 1, rpos - 2) : rpos + 1].min(), retest_low)
        latest_retest = df.iloc[rpos]
        confirmation = bool(latest_retest["BULL_REJECTION"]) or _finite(latest_retest["Close"]) > resistance
    trend = _finite(row["EMA20"]) > _finite(row["EMA50"]) > _finite(row["EMA200"])
    relative = _finite(row["REL_STRENGTH60"], 0) > 0
    breakout_quality = min(1.0, _finite(breakout_row["VOL_RATIO"]) / 2.0)
    score = 20 * float(trend) + 25 * breakout_quality + 15 * min(1.0, _finite(breakout_row["BODY_ATR"]))
    score += 22 if retest_date is not None else 6
    score += 10 if confirmation else 2
    score += 8 if relative else 3
    plan.setup_score = round(min(100.0, score), 1)
    plan.detected = not invalidated
    plan.invalidated = invalidated
    if invalidated:
        plan.reason = "Breakout sudah gagal: penutupan menembus bawah level invalidasi"
        return plan

    zone_low = resistance - 0.35 * atr_v
    zone_high = resistance + 0.35 * atr_v
    in_retest_area = _distance_to_zone(close, zone_low, zone_high, atr_v) <= 0.45
    if retest_date is not None and confirmation and in_retest_area:
        raw_entry = max(close, _finite(row["High"]) + idx_tick_size(close))
        plan.entry_type = "BUY_STOP_AFTER_RETEST"
        plan.action = "READY_TRIGGER"
    else:
        raw_entry = resistance + 0.10 * atr_v
        plan.entry_type = "LIMIT_RETEST_WITH_RECLAIM"
        plan.action = "WAIT_RETEST"
    raw_stop = min(retest_low, resistance - 0.70 * atr_v) - 0.15 * atr_v
    plan.signal_date = breakout_date
    plan.zone_created_date = breakout_date
    plan.zone_age_bars = _bars_since(df, breakout_date)
    plan.valid_until = pd.Timestamp(breakout_date) + pd.offsets.BDay(25)
    plan.entry_low, plan.entry_high = zone_low, zone_high
    plan.trigger = max(resistance, _finite(row["High"])) + idx_tick_size(close)
    plan.distance_atr = round(_distance_to_zone(close, zone_low, zone_high, atr_v), 2)
    plan.evidence = [
        "Breakout high 55-hari",
        f"Volume breakout {_finite(breakout_row['VOL_RATIO']):.2f}x",
        "Displacement bullish",
    ]
    if retest_date is not None:
        plan.evidence.append("Retest level breakout terdeteksi")
    if confirmation:
        plan.evidence.append("Retest ditutup dengan reclaim")
    plan.reason = "Breakout tervalidasi; eksekusi hanya setelah retest/reclaim"
    return _plan_prices(plan, df, atr_v, raw_entry, raw_stop)


def detect_reversal_accumulation(df: pd.DataFrame, ticker: str) -> SetupPlan:
    name = "REVERSAL_ACCUMULATION"
    plan = SetupPlan(ticker=ticker, setup=name, detected=False, setup_score=0.0)
    if len(df) < 205:
        plan.reason = "Data belum cukup"
        return plan
    row = df.iloc[-1]
    close, atr_v = _finite(row["Close"]), _finite(row["ATR14"])
    if close <= 0 or atr_v <= 0:
        plan.reason = "ATR/harga tidak valid"
        return plan
    prior = df.iloc[-150:-30]
    base = df.iloc[-30:]
    prior_high = _finite(prior["High"].max(), close)
    base_low, base_high = _finite(base["Low"].min(), close), _finite(base["High"].max(), close)
    decline = base_low / prior_high - 1 if prior_high > 0 else 0
    base_width = (base_high - base_low) / close
    based = decline <= -0.12 and base_width <= 0.32
    contraction = _finite(row["RANGE_CONTRACTION20"], 2) <= 0.95
    accumulation = (
        _finite(base["CMF20"].iloc[-10:].mean()) > 0.02
        and _finite(row["OBV_SLOPE10"]) > 0
    )
    sweep_mask = (
        (df["Low"] < df["LOW20_PREV"])
        & (df["Close"] > df["LOW20_PREV"])
        & (df["CLOSE_LOCATION"] > 0.58)
    )
    sweep_date = _last_true_index(sweep_mask, 25)
    if sweep_date is not None:
        spos = int(np.flatnonzero(df.index == sweep_date)[-1])
        sweep_low = _finite(df.iloc[spos]["Low"])
    else:
        spos, sweep_low = len(df) - 30, base_low
    choch_mask = (
        (df["Close"] > df["LAST_PIVOT_HIGH"] + 0.05 * df["ATR14"])
        & (df["Close"] > df["EMA20"])
        & (df["VOL_RATIO"] >= 1.05)
    )
    post_choch = choch_mask.iloc[spos:] if sweep_date is not None else choch_mask.iloc[-15:]
    choch_date = _last_true_index(post_choch, len(post_choch)) if len(post_choch) else None
    choch = choch_date is not None
    invalidated = close < sweep_low - 0.20 * atr_v

    score = 0.0
    score += 18 if based else 5
    score += 12 if contraction else 3
    score += 20 if accumulation else (8 if _finite(row["CMF20"]) > 0 else 0)
    score += 20 if sweep_date is not None else 0
    score += 22 if choch else 4
    score += 8 if close > _finite(row["EMA50"]) else 2
    plan.setup_score = min(100.0, score)
    plan.detected = bool(based and accumulation and sweep_date is not None and not invalidated)
    plan.invalidated = invalidated
    if not plan.detected:
        plan.reason = "Belum ada rangkaian decline–base–akumulasi–liquidity sweep yang lengkap"
        return plan

    structure_level = _finite(row["LAST_PIVOT_HIGH"], base_high)
    if choch_date is not None:
        cpos = int(np.flatnonzero(df.index == choch_date)[-1])
        structure_level = _finite(df.iloc[cpos]["LAST_PIVOT_HIGH"], structure_level)
    # Anchor the CHOCH retest to the broken structure. EMA20 is confluence,
    # not allowed to invert the zone when it already sits above resistance.
    zone_low = structure_level - 0.45 * atr_v
    zone_high = structure_level + 0.35 * atr_v
    in_zone = _distance_to_zone(close, zone_low, zone_high, atr_v) <= 0.45
    confirmation = bool(row["BULL_REJECTION"]) or close > _finite(df["High"].iloc[-2])
    if choch and in_zone and confirmation:
        raw_entry = max(close, _finite(row["High"]) + idx_tick_size(close))
        plan.entry_type = "BUY_STOP_AFTER_CHOCH"
        plan.action = "READY_TRIGGER"
    elif choch:
        raw_entry = (zone_low + zone_high) / 2
        plan.entry_type = "LIMIT_ON_CHOCH_RETEST"
        plan.action = "WAIT_RETEST"
    else:
        raw_entry = structure_level + idx_tick_size(structure_level)
        plan.entry_type = "BUY_STOP_AFTER_CHOCH"
        plan.action = "WAIT_CHOCH"
    raw_stop = sweep_low - 0.20 * atr_v
    plan.signal_date = sweep_date
    plan.zone_created_date = choch_date or sweep_date
    plan.zone_age_bars = _bars_since(df, plan.zone_created_date)
    plan.valid_until = pd.Timestamp(sweep_date) + pd.offsets.BDay(30)
    plan.entry_low, plan.entry_high = zone_low, zone_high
    plan.trigger = structure_level + idx_tick_size(structure_level)
    plan.distance_atr = round(_distance_to_zone(close, zone_low, zone_high, atr_v), 2)
    plan.evidence = ["Penurunan diikuti base", "Proxy CMF/OBV menguat", "Sell-side liquidity sweep"]
    if contraction:
        plan.evidence.append("Volatilitas berkontraksi")
    if choch:
        plan.evidence.append("CHOCH/BOS bullish terkonfirmasi")
    plan.reason = "Reversal hanya dapat dieksekusi setelah perubahan struktur bullish"
    return _plan_prices(plan, df, atr_v, raw_entry, raw_stop, 1.8, 3.0)


def detect_unicorn_sniper(df: pd.DataFrame, ticker: str) -> SetupPlan:
    name = "UNICORN_SNIPER_ICT"
    plan = SetupPlan(ticker=ticker, setup=name, detected=False, setup_score=0.0)
    if len(df) < 120:
        plan.reason = "Data struktur belum cukup"
        return plan
    row = df.iloc[-1]
    close, atr_v = _finite(row["Close"]), _finite(row["ATR14"])
    if close <= 0 or atr_v <= 0:
        plan.reason = "ATR/harga tidak valid"
        return plan
    sweep_mask = (
        (df["Low"] < df["LOW20_PREV"])
        & (df["Close"] > df["LOW20_PREV"])
        & (df["CLOSE_LOCATION"] >= 0.55)
    )
    sweep_date = _last_true_index(sweep_mask, 35)
    if sweep_date is None:
        plan.reason = "Belum ada sell-side liquidity sweep bullish"
        return plan
    spos = int(np.flatnonzero(df.index == sweep_date)[-1])
    sweep_low = _finite(df.iloc[spos]["Low"])
    bos_mask = (
        (df["Close"] > df["LAST_PIVOT_HIGH"] + 0.05 * df["ATR14"])
        & (df["BODY_ATR"] >= 0.55)
        & (df["Close"] > df["Open"])
    )
    bos_post = bos_mask.iloc[spos + 1 :]
    bos_date = _last_true_index(bos_post, min(20, len(bos_post))) if len(bos_post) else None
    if bos_date is None:
        plan.setup_score = 28.0
        plan.reason = "Liquidity sweep ada, tetapi displacement/BOS belum terkonfirmasi"
        return plan
    bpos = int(np.flatnonzero(df.index == bos_date)[-1])
    fvg_window = df.iloc[max(spos + 1, bpos - 2) : min(len(df), bpos + 6)]
    fvg_hits = fvg_window[fvg_window["BULL_FVG"].fillna(False)]
    if fvg_hits.empty:
        plan.setup_score = 50.0
        plan.reason = "Sweep dan BOS ada, tetapi FVG displacement tidak valid"
        return plan
    fvg_date = fvg_hits.index[-1]
    fpos = int(np.flatnonzero(df.index == fvg_date)[-1])
    fvg_low = _finite(df.loc[fvg_date, "FVG_LOW"])
    fvg_high = _finite(df.loc[fvg_date, "FVG_HIGH"])
    if fvg_high <= fvg_low:
        plan.reason = "FVG tidak valid"
        return plan

    # Last down candle before displacement is used as an objective bullish OB proxy.
    search_ob = df.iloc[spos : max(spos + 1, fpos)]
    bear = search_ob[search_ob["BEAR_CANDLE"].fillna(False)]
    ob_overlap = False
    if not bear.empty:
        ob_row = bear.iloc[-1]
        ob_low, ob_high = _finite(ob_row["Low"]), max(_finite(ob_row["Open"]), _finite(ob_row["Close"]))
        overlap_low, overlap_high = max(fvg_low, ob_low), min(fvg_high, ob_high)
        if overlap_high > overlap_low:
            zone_low, zone_high = overlap_low, overlap_high
            ob_overlap = True
        else:
            zone_low, zone_high = fvg_low, fvg_high
    else:
        zone_low, zone_high = fvg_low, fvg_high

    after_fvg = df.iloc[fpos + 1 :]
    invalidated = (
        bool(
            (
                (after_fvg["Close"] < fvg_low - 0.15 * after_fvg["ATR14"])
                | (after_fvg["Close"] < sweep_low)
            ).any()
        )
        if not after_fvg.empty
        else False
    )
    dealing_high = _finite(df["High"].iloc[spos : bpos + 1].max(), close)
    equilibrium = (sweep_low + dealing_high) / 2
    discount = (zone_low + zone_high) / 2 <= equilibrium
    volume_ok = _finite(df.loc[bos_date, "VOL_RATIO"], 0) >= 1.05
    confirmation = bool(row["BULL_REJECTION"]) or close > _finite(df["High"].iloc[-2])
    distance = _distance_to_zone(close, zone_low, zone_high, atr_v)

    score = 20 + 25 + 15 + 15
    score += 10 if ob_overlap else 3
    score += 10 if discount else 2
    score += 5 if volume_ok else 0
    plan.setup_score = min(100.0, float(score))
    plan.detected = not invalidated
    plan.invalidated = invalidated
    if invalidated:
        plan.reason = "FVG/low sweep sudah ditutup tembus; zona tidak lagi valid"
        return plan

    in_zone = distance <= 0.35
    if in_zone and confirmation:
        raw_entry = max(close, _finite(row["High"]) + idx_tick_size(close))
        plan.entry_type = "BUY_STOP_FVG_RECLAIM"
        plan.action = "READY_TRIGGER"
    else:
        raw_entry = (zone_low + zone_high) / 2
        plan.entry_type = "LIMIT_FVG_THEN_RECLAIM"
        plan.action = "WAIT_FVG_RETRACE"
    raw_stop = min(sweep_low, zone_low - 0.45 * atr_v) - 0.10 * atr_v
    plan.signal_date = sweep_date
    plan.zone_created_date = fvg_date
    plan.zone_age_bars = _bars_since(df, fvg_date)
    plan.valid_until = pd.Timestamp(fvg_date) + pd.offsets.BDay(30)
    plan.entry_low, plan.entry_high = zone_low, zone_high
    plan.trigger = max(zone_high, _finite(row["High"])) + idx_tick_size(close)
    plan.distance_atr = round(distance, 2)
    plan.evidence = ["Sell-side liquidity sweep", "Bullish BOS dengan displacement", "Bullish FVG valid"]
    if ob_overlap:
        plan.evidence.append("FVG overlap dengan order-block proxy")
    if discount:
        plan.evidence.append("Zona berada di discount dealing range")
    plan.reason = "SMC/ICT dipakai sebagai timing confluence, bukan bukti standalone"
    return _plan_prices(plan, df, atr_v, raw_entry, raw_stop, 1.8, 3.0)


SETUP_DETECTORS: tuple[Callable[[pd.DataFrame, str], SetupPlan], ...] = (
    detect_pullback_continuation,
    detect_breakout_retest,
    detect_reversal_accumulation,
    detect_unicorn_sniper,
)


def detect_all_setups(df: pd.DataFrame, ticker: str) -> list[SetupPlan]:
    return [detector(df, ticker) for detector in SETUP_DETECTORS]


# ---- data ----

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable

import pandas as pd



TICKER_COLUMNS = ("ticker", "tickers", "symbol", "symbols", "kode", "code", "emiten", "stock")


def _daily_ohlcv_cache_path(ticker: str) -> Path:
    safe = re.sub(r"[^A-Z0-9._-]", "_", str(ticker).upper())
    root = _cache_root() / "ohlcv_daily"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{safe}.csv"


def _load_daily_ohlcv_cache(ticker: str) -> pd.DataFrame:
    try:
        path = _daily_ohlcv_cache_path(ticker)
        if not path.is_file():
            return pd.DataFrame()
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        return _clean_ohlcv(frame, strict=True)
    except Exception:
        return pd.DataFrame()


def _write_daily_ohlcv_cache(ticker: str, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        return
    tmp: Path | None = None
    try:
        path = _daily_ohlcv_cache_path(ticker)
        tmp = path.with_suffix(".tmp")
        frame.to_csv(tmp, index=True, index_label="Date")
        tmp.replace(path)
    except Exception:
        try:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        except Exception:
            pass


def normalize_idx_ticker(value: object) -> str | None:
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL", "TICKER"}:
        return None
    text = re.sub(r"\s+", "", text)
    text = text.replace(".IDX", "").replace("IDX:", "")
    if text.endswith(".JK"):
        base = text[:-3]
    else:
        base = text
    if not re.fullmatch(r"[A-Z0-9]{3,8}", base):
        return None
    return f"{base}.JK"


def parse_ticker_csv(source: bytes | BinaryIO | pd.DataFrame, max_tickers: int = 1_200) -> list[str]:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        payload = BytesIO(source) if isinstance(source, bytes) else source
        try:
            frame = pd.read_csv(payload, sep=None, engine="python")
        except UnicodeDecodeError:
            if hasattr(payload, "seek"):
                payload.seek(0)
            frame = pd.read_csv(payload, encoding="latin-1", sep=None, engine="python")
    if frame.empty or len(frame.columns) == 0:
        return []
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    selected = next((lookup[name] for name in TICKER_COLUMNS if name in lookup), frame.columns[0])
    result: list[str] = []
    seen: set[str] = set()
    for value in frame[selected].tolist():
        ticker = normalize_idx_ticker(value)
        if ticker and ticker not in seen:
            result.append(ticker)
            seen.add(ticker)
        if len(result) >= max_tickers:
            break
    return result


def _clean_ohlcv(frame: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.columns = [str(c).title() for c in out.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in out.columns for c in required):
        return pd.DataFrame()
    out = out[required]
    out.index = pd.to_datetime(out.index, errors="coerce")
    if getattr(out.index, "tz", None) is not None:
        # Yahoo normally labels IDX bars in Asia/Jakarta, but this is not
        # guaranteed for every endpoint/version. Convert before dropping the
        # timezone so intraday session dates and opening windows stay correct.
        out.index = out.index.tz_convert("Asia/Jakarta").tz_localize(None)
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["Volume"] = out["Volume"].fillna(0.0).clip(lower=0)
    valid = (
        out[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
        & out["High"].ge(out[["Open", "Low", "Close"]].max(axis=1))
        & out["Low"].le(out[["Open", "High", "Close"]].min(axis=1))
    )
    if strict:
        out = out[valid]
    return out


def ohlcv_quality_issues(frame: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if frame is None or frame.empty:
        return ["OHLCV kosong"]
    if frame.index.has_duplicates:
        issues.append("Tanggal duplikat")
    if not frame.index.is_monotonic_increasing:
        issues.append("Tanggal tidak terurut")
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(column in frame for column in required):
        return ["Kolom OHLCV tidak lengkap"]
    if frame[required].isna().any().any():
        issues.append("OHLCV mengandung nilai kosong")
    if (frame["Volume"] < 0).any():
        issues.append("Volume negatif")
    valid = (
        frame[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
        & frame["High"].ge(frame[["Open", "Low", "Close"]].max(axis=1))
        & frame["Low"].le(frame[["Open", "High", "Close"]].min(axis=1))
    )
    if not bool(valid.all()):
        issues.append("Bar OHLC tidak konsisten")
    jumps = frame["Close"].pct_change().abs()
    if bool(jumps.gt(0.80).any()):
        issues.append("Lompatan adjusted price >80%; corporate action/data wajib diverifikasi")
    if len(frame) >= 20 and float(frame["Volume"].tail(20).eq(0).mean()) > 0.10:
        issues.append("Lebih dari 10% bar terakhir bervolume nol")
    return issues


def _extract_batch(raw: pd.DataFrame, ticker: str, total: int) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        return _clean_ohlcv(raw) if total == 1 else pd.DataFrame()
    level0 = set(map(str, raw.columns.get_level_values(0)))
    level1 = set(map(str, raw.columns.get_level_values(1)))
    try:
        if ticker in level0:
            return _clean_ohlcv(raw[ticker])
        if ticker in level1:
            return _clean_ohlcv(raw.xs(ticker, axis=1, level=1))
    except (KeyError, ValueError):
        return pd.DataFrame()
    return pd.DataFrame()


def download_ohlcv(
    tickers: Iterable[str], period: str = "3y", batch_size: int = 30
) -> tuple[dict[str, pd.DataFrame], DownloadReport]:
    import yfinance as yf

    requested = list(dict.fromkeys(tickers))
    histories: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}
    warnings: dict[str, str] = {}
    source_tiers: dict[str, str] = {}
    skip_individual_retry: set[str] = set()
    for start in range(0, len(requested), batch_size):
        batch = requested[start : start + batch_size]
        try:
            raw = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                repair=False,
                actions=False,
                # Bounded concurrency is slower than Yahoo's unrestricted
                # default, but materially reduces 300-ticker rate-limit storms.
                threads=4,
                progress=False,
                timeout=25,
            )
            for ticker in batch:
                extracted = _extract_batch(raw, ticker, len(batch))
                quality = ohlcv_quality_issues(extracted)
                frame = _clean_ohlcv(extracted, strict=True)
                if not frame.empty:
                    histories[ticker] = frame
                    source_tiers[ticker] = "LIVE_YAHOO"
                    _write_daily_ohlcv_cache(ticker, frame)
                    if quality:
                        warnings[ticker] = " • ".join(quality)
                else:
                    failed[ticker] = "Data batch kosong"
            if raw is None or raw.empty:
                skip_individual_retry.update(batch)
        except Exception as exc:  # network providers can fail per batch
            for ticker in batch:
                failed[ticker] = f"Batch gagal: {type(exc).__name__}"
            if "RATE" in f"{type(exc).__name__} {exc}".upper() or "TOO MANY" in str(exc).upper():
                skip_individual_retry.update(batch)

    missing = [t for t in requested if t not in histories]

    # A provider outage must not erase previously verified bars. Freshness and
    # current-bar gates still run later, so an old cache can support research
    # but can never silently become an executable order.
    for ticker in list(missing):
        cached = _load_daily_ohlcv_cache(ticker)
        if cached.empty:
            continue
        histories[ticker] = cached
        source_tiers[ticker] = "CACHE_FALLBACK"
        cache_date = pd.Timestamp(cached.index[-1]).date().isoformat()
        warnings[ticker] = f"OHLCV memakai cache fallback; bar terakhir {cache_date}"
        failed.pop(ticker, None)

    missing = [t for t in requested if t not in histories and t not in skip_individual_retry]

    def retry_one(ticker: str) -> tuple[str, pd.DataFrame, str | None]:
        try:
            frame = yf.Ticker(ticker).history(
                period=period, interval="1d", auto_adjust=True, repair=False, actions=False, timeout=20
            )
            extracted = _clean_ohlcv(frame)
            clean = _clean_ohlcv(extracted, strict=True)
            quality = " • ".join(ohlcv_quality_issues(extracted))
            audit = (quality or None) if not clean.empty else "Data individual kosong"
            return ticker, clean, audit
        except Exception as exc:
            return ticker, pd.DataFrame(), f"{type(exc).__name__}: {str(exc)[:100]}"

    if missing:
        with ThreadPoolExecutor(max_workers=min(2, len(missing))) as pool:
            futures = [pool.submit(retry_one, ticker) for ticker in missing]
            for future in as_completed(futures):
                ticker, frame, error = future.result()
                if not frame.empty:
                    histories[ticker] = frame
                    source_tiers[ticker] = "LIVE_YAHOO_RETRY"
                    _write_daily_ohlcv_cache(ticker, frame)
                    failed.pop(ticker, None)
                    if error:
                        warnings[ticker] = error
                else:
                    failed[ticker] = error or "Tidak ada data"

    for ticker in requested:
        source_tiers.setdefault(ticker, "UNAVAILABLE" if ticker not in histories else "LIVE_YAHOO")

    report = DownloadReport(
        requested,
        sorted(histories),
        failed,
        downloaded_at=pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
        warnings=warnings,
        source_tiers=source_tiers,
    )
    return histories, report


def download_benchmark(period: str = "3y") -> pd.DataFrame:
    import yfinance as yf

    try:
        frame = yf.Ticker("^JKSE").history(
            period=period, interval="1d", auto_adjust=True, repair=False, actions=False, timeout=20
        )
        clean = _clean_ohlcv(frame, strict=True)
        if not clean.empty:
            _write_daily_ohlcv_cache("^JKSE", clean)
        return clean
    except Exception:
        return _load_daily_ohlcv_cache("^JKSE")


# ---- fundamentals ----

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

import numpy as np
import pandas as pd



def _num(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _linear_score(value: float, bad: float, good: float, higher_is_better: bool = True) -> float | None:
    if not np.isfinite(value):
        return None
    if good == bad:
        return 50.0
    ratio = (value - bad) / (good - bad)
    if not higher_is_better:
        ratio = 1 - ratio
    return float(np.clip(100 * ratio, 0, 100))


def score_fundamentals(info: dict[str, Any]) -> dict[str, Any]:
    sector_text = str(info.get("sector") or "")
    industry_text = str(info.get("industry") or "")
    is_financial = "financial" in sector_text.lower() or "bank" in industry_text.lower()
    debt_equity_raw = _num(info.get("debtToEquity"))
    debt_equity = debt_equity_raw / 100 if np.isfinite(debt_equity_raw) else np.nan
    total_cash = _num(info.get("totalCash"))
    total_debt = _num(info.get("totalDebt"))
    market_cap = _num(info.get("marketCap"))
    fcf = _num(info.get("freeCashflow"))
    ocf = _num(info.get("operatingCashflow"))
    cash_to_debt = total_cash / total_debt if np.isfinite(total_cash) and total_debt > 0 else (
        5.0 if np.isfinite(total_cash) and total_debt == 0 else np.nan
    )
    fcf_yield = fcf / market_cap if np.isfinite(fcf) and market_cap > 0 else np.nan
    metrics = {
        "revenue_growth": _num(info.get("revenueGrowth")),
        "earnings_growth": _num(info.get("earningsGrowth")),
        "gross_margin": _num(info.get("grossMargins")),
        "operating_margin": _num(info.get("operatingMargins")),
        "net_margin": _num(info.get("profitMargins")),
        "roe": _num(info.get("returnOnEquity")),
        "roa": _num(info.get("returnOnAssets")),
        "debt_equity": debt_equity,
        "current_ratio": _num(info.get("currentRatio")),
        "cash_to_debt": cash_to_debt,
        "operating_cash_flow": ocf,
        "free_cash_flow": fcf,
        "fcf_yield": fcf_yield,
        "trailing_pe": _num(info.get("trailingPE")),
        "forward_pe": _num(info.get("forwardPE")),
        "price_to_book": _num(info.get("priceToBook")),
        "peg_ratio": _num(info.get("pegRatio")),
        "market_cap": market_cap,
        "sector": sector_text or industry_text,
        "company_name": info.get("shortName") or info.get("longName") or "",
        "fundamental_model": "FINANCIAL" if is_financial else "GENERAL",
    }
    weighted: list[tuple[float, float]] = []
    applicable_weight = 0.0

    def add(score: float | None, weight: float) -> None:
        nonlocal applicable_weight
        applicable_weight += weight
        if score is not None and np.isfinite(score):
            weighted.append((float(score), weight))

    add(_linear_score(metrics["revenue_growth"], -0.05, 0.20), 14)
    add(_linear_score(metrics["earnings_growth"], -0.10, 0.25), 14)
    add(_linear_score(metrics["roe"], 0.05, 0.22), 10)
    add(_linear_score(metrics["roa"], 0.01, 0.10), 7)
    add(_linear_score(metrics["gross_margin"], 0.10, 0.45), 6)
    add(_linear_score(metrics["operating_margin"], 0.02, 0.20), 7)
    add(_linear_score(metrics["net_margin"], 0.01, 0.15), 6)
    if not is_financial:
        add(_linear_score(metrics["debt_equity"], 2.0, 0.3, higher_is_better=True), 8)
        add(_linear_score(metrics["current_ratio"], 0.8, 2.0), 5)
        add(_linear_score(metrics["cash_to_debt"], 0.1, 1.2), 5)
        add(100.0 if np.isfinite(ocf) and ocf > 0 else 0.0 if np.isfinite(ocf) else None, 6)
        add(100.0 if np.isfinite(fcf) and fcf > 0 else 0.0 if np.isfinite(fcf) else None, 6)
        add(_linear_score(metrics["fcf_yield"], 0.0, 0.08), 3)
    peg = metrics["peg_ratio"]
    peg_score = None
    if np.isfinite(peg):
        peg_score = 100.0 if 0 < peg <= 1.5 else 65.0 if peg <= 2.5 else 20.0 if peg > 0 else 0.0
    add(peg_score, 3)
    score = sum(value * weight for value, weight in weighted) / sum(weight for _, weight in weighted) if weighted else np.nan
    coverage = sum(weight for _, weight in weighted) / applicable_weight if applicable_weight else 0.0
    red_flags: list[str] = []
    if np.isfinite(metrics["revenue_growth"]) and metrics["revenue_growth"] < 0:
        red_flags.append("Revenue menyusut")
    if np.isfinite(metrics["earnings_growth"]) and metrics["earnings_growth"] < 0:
        red_flags.append("Laba menyusut")
    if not is_financial:
        if np.isfinite(ocf) and ocf <= 0:
            red_flags.append("OCF negatif")
        if np.isfinite(fcf) and fcf <= 0:
            red_flags.append("FCF negatif")
        if np.isfinite(debt_equity) and debt_equity > 2:
            red_flags.append("DER tinggi")
    if np.isfinite(metrics["net_margin"]) and metrics["net_margin"] <= 0:
        red_flags.append("Margin bersih negatif")
    metrics.update(
        {
            "fundamental_score": round(float(score), 1) if np.isfinite(score) else np.nan,
            "fundamental_coverage": round(100 * coverage, 1),
            "fundamental_reliability": "HIGH" if coverage >= 0.70 else "MEDIUM" if coverage >= 0.45 else "LOW",
            "fundamental_red_flags": " • ".join(red_flags),
        }
    )
    return metrics


def fetch_one_fundamental(ticker: str) -> dict[str, Any]:
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).get_info()
        row = score_fundamentals(info or {})
        row.update(
            {
                "ticker": ticker,
                "fundamental_error": "",
                "fundamental_provider": "Yahoo Finance via yfinance",
                "fundamental_fetched_at": pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
            }
        )
        return row
    except Exception as exc:
        return {
            "ticker": ticker,
            "fundamental_score": np.nan,
            "fundamental_coverage": 0.0,
            "fundamental_reliability": "NONE",
            "fundamental_red_flags": "",
            "fundamental_error": f"{type(exc).__name__}: {str(exc)[:100]}",
            "fundamental_provider": "Yahoo Finance via yfinance",
            "fundamental_fetched_at": pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
        }


def fetch_fundamentals(tickers: Iterable[str], max_workers: int = 2) -> pd.DataFrame:
    names = list(dict.fromkeys(tickers))
    rows: list[dict[str, Any]] = []
    if not names:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=min(max_workers, len(names))) as pool:
        futures = {pool.submit(fetch_one_fundamental, ticker): ticker for ticker in names}
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def attach_fundamentals(signals: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    if fundamentals.empty:
        result = signals.copy()
        result["fundamental_score"] = np.nan
        result["fundamental_coverage"] = 0.0
        result["fundamental_reliability"] = "NONE"
        result["fundamental_red_flags"] = ""
        result["fundamental_error"] = "Fundamental tidak diambil/tersedia"
        result["composite_score"] = result["quality_score"]
        return result
    result = signals.merge(fundamentals, on="ticker", how="left")
    usable = result["fundamental_coverage"].fillna(0) >= 60
    result["composite_score"] = result["quality_score"]
    result.loc[usable, "composite_score"] = (
        0.78 * result.loc[usable, "quality_score"] + 0.22 * result.loc[usable, "fundamental_score"]
    ).round(1)
    return result


def _fundamental_append_blocker(frame: pd.DataFrame, index: object, message: str) -> None:
    prior = str(frame.at[index, "blockers"] or "").strip()
    if message not in prior:
        frame.at[index, "blockers"] = f"{prior} • {message}" if prior else message
        count = pd.to_numeric(frame.at[index, "blocker_count"], errors="coerce")
        frame.at[index, "blocker_count"] = int(count) + 1 if pd.notna(count) else 1


def apply_fundamental_gate(
    signals: pd.DataFrame, config: ScanConfig | None = None
) -> pd.DataFrame:
    """Fail closed: missing, weak, or materially adverse fundamentals cannot execute."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    for column, default in (
        ("fundamental_score", np.nan),
        ("fundamental_coverage", 0.0),
        ("fundamental_reliability", "NONE"),
        ("fundamental_red_flags", ""),
        ("fundamental_error", "Fundamental tidak tersedia"),
    ):
        if column not in out:
            out[column] = default
    severe_pattern = (
        "Revenue menyusut|Laba menyusut|OCF negatif|FCF negatif|DER tinggi|Margin bersih negatif"
    )
    for idx, row in out.iterrows():
        coverage = pd.to_numeric(row.get("fundamental_coverage"), errors="coerce")
        score = pd.to_numeric(row.get("fundamental_score"), errors="coerce")
        red_flags = str(row.get("fundamental_red_flags") or "")
        missing = pd.isna(coverage) or coverage < cfg.min_fundamental_coverage or pd.isna(score)
        weak = not pd.isna(score) and score < cfg.min_fundamental_score
        severe = bool(pd.Series([red_flags]).str.contains(severe_pattern, regex=True).iloc[0])
        if cfg.real_money_mode and cfg.require_fundamentals and missing:
            if out.at[idx, "status"] == "EXECUTION_READY":
                out.at[idx, "status"] = "WATCHLIST_ENTRY"
            _fundamental_append_blocker(out, idx, "Fundamental coverage tidak memenuhi hard gate")
        if weak or severe:
            if out.at[idx, "status"] == "EXECUTION_READY":
                out.at[idx, "status"] = "WATCHLIST_ENTRY"
            _fundamental_append_blocker(out, idx, "Fundamental quality hard gate gagal")
    out["status_rank"] = out["status"].map(
        {"EXECUTION_READY": 0, "WATCHLIST_ENTRY": 1, "REJECT": 2}
    )
    return out


# ---- context_inputs ----

from io import BytesIO
from typing import BinaryIO

import numpy as np
import pandas as pd



def _read_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    """Read user CSV defensively, including UTF-8 BOM and common delimiters."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    payload = BytesIO(source) if isinstance(source, bytes) else source
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            if hasattr(payload, "seek"):
                payload.seek(0)
            return pd.read_csv(payload, sep=None, engine="python", encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"CSV tidak dapat dibaca: {last_error}")


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ya", "aktif", "active"}


def _column(frame: pd.DataFrame, name: str, default: object) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _context_append_blocker(frame: pd.DataFrame, index: object, message: str) -> None:
    prior = str(frame.at[index, "blockers"] or "").strip()
    if message not in prior:
        frame.at[index, "blockers"] = f"{prior} • {message}" if prior else message
        count = pd.to_numeric(frame.at[index, "blocker_count"], errors="coerce")
        frame.at[index, "blocker_count"] = int(count) + 1 if pd.notna(count) else 1


def _downgrade(frame: pd.DataFrame, index: object, message: str, reject: bool = False) -> None:
    if reject:
        frame.at[index, "status"] = "REJECT"
    elif frame.at[index, "status"] == "EXECUTION_READY":
        frame.at[index, "status"] = "WATCHLIST_ENTRY"
    _context_append_blocker(frame, index, message)


def parse_market_status_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    """Parse a user-supplied official IDX status snapshot.

    Required columns: ticker and as_of. Optional columns: suspended,
    special_monitoring, fca, special_notation, corporate_action, sharia,
    source_url. Missing flags are treated as False, never as verified coverage.
    """
    frame = _read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "ticker" not in frame or "as_of" not in frame:
        raise ValueError("Market-status CSV wajib memiliki kolom ticker dan as_of")
    out = pd.DataFrame()
    out["ticker"] = frame["ticker"].map(normalize_idx_ticker)
    out["market_status_asof"] = pd.to_datetime(frame["as_of"], errors="coerce")
    for column in ("suspended", "special_monitoring", "fca", "corporate_action", "sharia"):
        out[column] = frame[column].map(_truthy) if column in frame else False
    out["special_notation"] = _column(frame, "special_notation", "").fillna("").astype(str).str.strip()
    out["market_status_source"] = _column(frame, "source_url", "").fillna("").astype(str).str.strip()
    out["market_status_verified"] = out["market_status_asof"].notna() & out["market_status_source"].str.startswith(
        ("https://www.idx.co.id", "https://idx.co.id")
    )
    return out.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last")


def apply_market_status_gate(
    signals: pd.DataFrame,
    market_status: pd.DataFrame,
    config: ScanConfig | None = None,
    asof: object | None = None,
) -> pd.DataFrame:
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if market_status is None or market_status.empty:
        out["market_status_coverage"] = "MISSING"
        if cfg.real_money_mode and cfg.require_market_status:
            for idx in out.index:
                _downgrade(out, idx, "Status resmi IDX belum dilampirkan")
    else:
        out = out.merge(market_status, on="ticker", how="left")
        reference = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
        status_time = pd.to_datetime(out["market_status_asof"], errors="coerce")
        out["market_status_age_days"] = (reference.normalize() - status_time.dt.normalize()).dt.days
        verified_mask = out["market_status_verified"].map(_truthy).astype(bool)
        out["market_status_coverage"] = np.where(verified_mask, "VERIFIED", "MISSING")
        for idx, row in out.iterrows():
            verified = _truthy(row.get("market_status_verified", False))
            age = pd.to_numeric(row.get("market_status_age_days"), errors="coerce")
            if cfg.real_money_mode and cfg.require_market_status and (
                not verified or pd.isna(age) or age < 0 or age > cfg.max_context_age_days
            ):
                _downgrade(out, idx, "Status IDX tidak terverifikasi atau kedaluwarsa")
            if _truthy(row.get("suspended", False)):
                _downgrade(out, idx, "Saham berstatus suspensi", reject=True)
            if _truthy(row.get("fca", False)) or _truthy(row.get("special_monitoring", False)):
                _downgrade(out, idx, "Papan Pemantauan Khusus/FCA tidak lolos real-money gate", reject=True)
            notation_value = row.get("special_notation")
            notation = "" if pd.isna(notation_value) else str(notation_value).strip()
            if notation:
                _downgrade(out, idx, f"Notasi khusus IDX: {notation}")
            if _truthy(row.get("corporate_action", False)):
                _downgrade(out, idx, "Corporate action aktif: level adjusted wajib diverifikasi")
    out["status_rank"] = out["status"].map({"EXECUTION_READY": 0, "WATCHLIST_ENTRY": 1, "REJECT": 2})
    return out


def parse_news_review_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    """Parse human/assistant-reviewed public news and official disclosures."""
    frame = _read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "ticker" not in frame or "reviewed_at" not in frame:
        raise ValueError("News-review CSV wajib memiliki kolom ticker dan reviewed_at")
    frame["ticker"] = frame["ticker"].map(normalize_idx_ticker)
    frame["news_reviewed_at"] = pd.to_datetime(frame["reviewed_at"], errors="coerce")
    frame["news_review_status"] = _column(frame, "review_status", "COMPLETE").fillna("COMPLETE").astype(str).str.upper()
    frame["news_title"] = _column(frame, "title", "").fillna("").astype(str)
    frame["news_sentiment"] = _column(frame, "sentiment", "NEUTRAL").fillna("NEUTRAL").astype(str).str.upper()
    frame["news_materiality"] = _column(frame, "materiality", "LOW").fillna("LOW").astype(str).str.upper()
    frame["news_event_date"] = pd.to_datetime(_column(frame, "event_date", pd.NaT), errors="coerce")
    frame["news_source_url"] = _column(frame, "source_url", "").fillna("").astype(str)
    frame["news_verified"] = _column(frame, "verified", False).map(_truthy)

    rows: list[dict[str, object]] = []
    for ticker, group in frame.dropna(subset=["ticker"]).groupby("ticker", sort=False):
        group = group.sort_values("news_reviewed_at")
        material = group[group["news_title"].str.len().gt(0)]
        positive = material[(material["news_sentiment"] == "POSITIVE") & material["news_verified"]]
        negative = material[(material["news_sentiment"] == "NEGATIVE") & material["news_verified"]]
        severe_negative = negative[negative["news_materiality"].isin(["HIGH", "CRITICAL"])]
        latest = group.iloc[-1]
        titles = material.tail(3)["news_title"].tolist()
        rows.append(
            {
                "ticker": ticker,
                "news_reviewed_at": latest["news_reviewed_at"],
                "news_review_status": latest["news_review_status"],
                "verified_catalyst_count": int(len(positive)),
                "verified_negative_count": int(len(negative)),
                "severe_negative_news": bool(len(severe_negative)),
                "catalyst_summary": " | ".join(titles),
                "news_sources": " | ".join(material.tail(3)["news_source_url"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def apply_news_gate(
    signals: pd.DataFrame,
    news_review: pd.DataFrame,
    config: ScanConfig | None = None,
    asof: object | None = None,
) -> pd.DataFrame:
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if news_review is None or news_review.empty:
        out["news_review_status"] = "MISSING"
        if cfg.real_money_mode and cfg.require_news_review:
            for idx in out.index:
                _downgrade(out, idx, "News/catalyst review belum tersedia")
    else:
        out = out.merge(news_review, on="ticker", how="left")
        reference = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
        review_time = pd.to_datetime(out["news_reviewed_at"], errors="coerce")
        out["news_review_age_days"] = (reference.normalize() - review_time.dt.normalize()).dt.days
        for idx, row in out.iterrows():
            status = str(row.get("news_review_status") or "MISSING").upper()
            age = pd.to_numeric(row.get("news_review_age_days"), errors="coerce")
            if cfg.real_money_mode and cfg.require_news_review and (
                status != "COMPLETE" or pd.isna(age) or age < 0 or age > cfg.max_context_age_days
            ):
                _downgrade(out, idx, "News/catalyst review tidak lengkap atau kedaluwarsa")
            if _truthy(row.get("severe_negative_news", False)):
                _downgrade(out, idx, "Berita negatif material terverifikasi", reject=True)
    out["status_rank"] = out["status"].map({"EXECUTION_READY": 0, "WATCHLIST_ENTRY": 1, "REJECT": 2})
    return out


def parse_broker_summary_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    """Aggregate Stockbit/exported broker summary without claiming beneficial ownership."""
    frame = _read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {"ticker", "date", "broker_code"}
    if not required.issubset(frame.columns):
        raise ValueError("Broker-summary CSV wajib memiliki ticker, date, dan broker_code")
    buy_col = "buy_value" if "buy_value" in frame else "buy_volume" if "buy_volume" in frame else None
    sell_col = "sell_value" if "sell_value" in frame else "sell_volume" if "sell_volume" in frame else None
    if buy_col is None or sell_col is None:
        raise ValueError("Broker-summary CSV memerlukan buy_value/sell_value atau buy_volume/sell_volume")
    frame["ticker"] = frame["ticker"].map(normalize_idx_ticker)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["buy"] = pd.to_numeric(frame[buy_col], errors="coerce").fillna(0).clip(lower=0)
    frame["sell"] = pd.to_numeric(frame[sell_col], errors="coerce").fillna(0).clip(lower=0)
    frame["net"] = frame["buy"] - frame["sell"]
    frame["gross"] = frame["buy"] + frame["sell"]
    rows: list[dict[str, object]] = []
    for ticker, group in frame.dropna(subset=["ticker", "date"]).groupby("ticker", sort=False):
        dates = sorted(group["date"].dt.normalize().unique())[-10:]
        recent = group[group["date"].dt.normalize().isin(dates)]
        net = float(recent["net"].sum())
        gross = float(recent["gross"].sum())
        ratio = net / gross if gross > 0 else np.nan
        broker_net = recent.groupby("broker_code")["net"].sum().sort_values(ascending=False)
        label = "ACCUMULATION_PROXY" if ratio >= 0.08 else "DISTRIBUTION_PROXY" if ratio <= -0.08 else "NEUTRAL"
        rows.append(
            {
                "ticker": ticker,
                "broksum_asof": recent["date"].max(),
                "broksum_days": len(dates),
                "broksum_net": net,
                "broksum_net_ratio": ratio,
                "broksum_signal": label,
                "top_net_buy_brokers": ", ".join(map(str, broker_net.head(3).index.tolist())),
                "top_net_sell_brokers": ", ".join(map(str, broker_net.tail(3).index.tolist())),
                "broksum_note": "Proxy kode broker; bukan identitas beneficial owner",
            }
        )
    return pd.DataFrame(rows)


def attach_broker_summary(signals: pd.DataFrame, broksum: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if broksum is None or broksum.empty:
        out["broksum_signal"] = "UNAVAILABLE"
        out["broksum_note"] = "Data broker summary tidak dilampirkan"
        return out
    out = out.merge(broksum, on="ticker", how="left")
    out["broksum_signal"] = out["broksum_signal"].fillna("UNAVAILABLE")
    for idx in out.index[out["broksum_signal"].eq("DISTRIBUTION_PROXY")]:
        _downgrade(out, idx, "Broker-summary menunjukkan distribution proxy")
    out["status_rank"] = out["status"].map({"EXECUTION_READY": 0, "WATCHLIST_ENTRY": 1, "REJECT": 2})
    return out


# ---- risk ----

import math

import numpy as np
import pandas as pd



def _risk_append_blocker(frame: pd.DataFrame, index: object, message: str) -> None:
    prior = str(frame.at[index, "blockers"] or "").strip()
    if message not in prior:
        frame.at[index, "blockers"] = f"{prior} • {message}" if prior else message
        count = pd.to_numeric(frame.at[index, "blocker_count"], errors="coerce")
        frame.at[index, "blocker_count"] = int(count) + 1 if pd.notna(count) else 1


def size_stockbit_order(
    entry: float,
    stop_loss: float,
    config: ScanConfig | None = None,
) -> dict[str, float | int | str]:
    """Size a regular-market order so the fee/slippage-adjusted loss stays capped."""
    cfg = config or ScanConfig()
    values = (entry, stop_loss, cfg.account_size_idr, cfg.risk_per_trade_pct)
    if not all(math.isfinite(float(value)) for value in values):
        return {"sizing_status": "INVALID_LEVELS", "suggested_lots": 0}
    if entry <= 0 or stop_loss <= 0 or stop_loss >= entry:
        return {"sizing_status": "INVALID_LEVELS", "suggested_lots": 0}
    if not is_valid_idx_price(entry) or not is_valid_idx_price(stop_loss):
        return {"sizing_status": "INVALID_TICK", "suggested_lots": 0}

    half_slippage = cfg.order_slippage_pct / 2
    effective_buy = entry * (1 + cfg.buy_fee_pct + half_slippage)
    effective_stop_proceeds = stop_loss * (1 - cfg.sell_fee_pct - half_slippage)
    risk_per_share = effective_buy - effective_stop_proceeds
    capital_per_lot = 100 * effective_buy
    risk_per_lot = 100 * risk_per_share
    risk_budget = cfg.account_size_idr * cfg.risk_per_trade_pct
    position_cap = cfg.account_size_idr * cfg.max_position_pct
    if risk_per_lot <= 0 or capital_per_lot <= 0:
        return {"sizing_status": "INVALID_RISK", "suggested_lots": 0}

    lots_by_risk = math.floor(risk_budget / risk_per_lot)
    lots_by_capital = math.floor(position_cap / capital_per_lot)
    lots = max(0, min(lots_by_risk, lots_by_capital))
    capital_required = lots * capital_per_lot
    max_loss = lots * risk_per_lot
    status = "OK" if lots >= 1 else "ACCOUNT_TOO_SMALL_FOR_ONE_LOT"
    return {
        "sizing_status": status,
        "risk_budget_idr": round(risk_budget, 0),
        "risk_per_share_net": round(risk_per_share, 4),
        "risk_per_lot_idr": round(risk_per_lot, 0),
        "lots_by_risk": int(lots_by_risk),
        "lots_by_capital": int(lots_by_capital),
        "suggested_lots": int(lots),
        "shares": int(lots * 100),
        "capital_required_idr": round(capital_required, 0),
        "position_pct": round(100 * capital_required / cfg.account_size_idr, 2),
        "max_loss_idr": round(max_loss, 0),
        "max_loss_pct_account": round(100 * max_loss / cfg.account_size_idr, 3),
        "portfolio_max_positions": int(cfg.max_positions),
        "portfolio_risk_cap_idr": round(cfg.account_size_idr * cfg.max_portfolio_risk_pct, 0),
    }


def attach_position_sizing(signals: pd.DataFrame, config: ScanConfig | None = None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    rows: list[dict[str, float | int | str]] = []
    for row in out.itertuples(index=False):
        try:
            entry, stop = float(row.entry), float(row.stop_loss)
        except (TypeError, ValueError):
            entry, stop = float("nan"), float("nan")
        rows.append(size_stockbit_order(entry, stop, cfg))
    sizing = pd.DataFrame(rows, index=out.index)
    for column in sizing:
        out[column] = sizing[column]
    for idx in out.index[out["suggested_lots"].fillna(0).lt(1)]:
        if out.at[idx, "status"] == "EXECUTION_READY":
            out.at[idx, "status"] = "WATCHLIST_ENTRY"
        _risk_append_blocker(out, idx, "Ukuran posisi aman kurang dari 1 lot")
    out["status_rank"] = out["status"].map(
        {"EXECUTION_READY": 0, "WATCHLIST_ENTRY": 1, "REJECT": 2}
    )
    return out.replace([np.inf, -np.inf], np.nan)


# ---- charts ----

from typing import Any, Mapping

import pandas as pd


def make_signal_chart(frame: pd.DataFrame, signal: Mapping[str, Any], bars: int = 180) -> go.Figure:
    import plotly.graph_objects as go

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


# ---- engine ----

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd



STATUS_ORDER = {"EXECUTION_READY": 0, "WATCHLIST_ENTRY": 1, "REJECT": 2}


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


class ScanEngine:
    def __init__(self, config: ScanConfig | None = None):
        self.config = config or ScanConfig()

    def _market_context(
        self, prepared: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None
    ) -> tuple[MarketContext, pd.DataFrame | None]:
        bench_ind: pd.DataFrame | None = None
        above50: list[bool] = []
        above200: list[bool] = []
        for frame in prepared.values():
            if not frame.empty:
                last = frame.iloc[-1]
                if pd.notna(last.get("EMA50")):
                    above50.append(bool(last["Close"] > last["EMA50"]))
                if pd.notna(last.get("EMA200")):
                    above200.append(bool(last["Close"] > last["EMA200"]))
        breadth50 = 100 * float(np.mean(above50)) if above50 else None
        breadth200 = 100 * float(np.mean(above200)) if above200 else None
        context = MarketContext(breadth_ema50=breadth50, breadth_ema200=breadth200)
        if benchmark is None or benchmark.empty or len(benchmark) < 205:
            context.reason = "Data IHSG tidak tersedia/cukup; sinyal tidak boleh langsung dieksekusi"
            return context, bench_ind
        today = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).normalize()
        benchmark_age = max(0, (today - pd.Timestamp(benchmark.index[-1]).normalize()).days)
        if benchmark_age > self.config.max_absolute_data_age_days:
            context.reason = f"Data IHSG berumur {benchmark_age} hari; regime tidak dapat dipercaya"
            return context, bench_ind
        bench_ind = prepare_indicators(benchmark)
        last = bench_ind.iloc[-1]
        close = _number(last["Close"])
        ema50, ema200 = _number(last["EMA50"]), _number(last["EMA200"])
        roc20 = _number(last["ROC20"])
        context.benchmark_close = close
        context.benchmark_roc20 = roc20
        risk_on = close > ema50 > ema200 and roc20 > 0 and (breadth50 is None or breadth50 >= 52)
        risk_off = (close < ema200 and roc20 < 0) or (breadth50 is not None and breadth50 < 35)
        if risk_on:
            context.regime = "RISK_ON"
            context.reason = "IHSG di atas EMA50/200, momentum positif, breadth mendukung"
        elif risk_off:
            context.regime = "RISK_OFF"
            context.reason = "IHSG/breadth menunjukkan risiko pasar tinggi"
        else:
            context.regime = "NEUTRAL"
            context.reason = "Sinyal IHSG dan breadth belum seragam"
        return context, bench_ind

    def _tradeability(
        self, frame: pd.DataFrame, asof: pd.Timestamp
    ) -> tuple[list[str], dict[str, float | int | str]]:
        cfg = self.config
        blockers: list[str] = []
        last = frame.iloc[-1]
        previous = frame.iloc[-2] if len(frame) >= 2 else last
        close = _number(last["Close"], 0)
        atr_pct = _number(last.get("ATR_PCT"), 0)
        adtv = _number(last.get("ADTV20"), 0)
        zero_vol = _number(last.get("ZERO_VOL20"), 1)
        value_today = _number(last.get("VALUE"), 0)
        lag = max(0, (pd.Timestamp(asof).normalize() - pd.Timestamp(frame.index[-1]).normalize()).days)
        now_jakarta = pd.Timestamp.now(tz="Asia/Jakarta")
        today = now_jakarta.tz_localize(None).normalize()
        absolute_age = max(0, (today - pd.Timestamp(frame.index[-1]).normalize()).days)
        current_bar_incomplete = (
            pd.Timestamp(frame.index[-1]).date() == now_jakarta.date()
            and (now_jakarta.hour, now_jakarta.minute) < (16, 15)
        )
        if len(frame) < cfg.min_bars:
            blockers.append(f"Riwayat hanya {len(frame)} bar (<{cfg.min_bars})")
        if close < cfg.min_price:
            blockers.append(f"Harga Rp{close:,.0f} di bawah minimum")
        if adtv < cfg.min_adtv_idr:
            blockers.append(f"ADTV20 Rp{adtv/1e9:.2f} miliar di bawah gate")
        if zero_vol > cfg.max_zero_volume_ratio:
            blockers.append(f"Hari volume nol {zero_vol:.0%} terlalu tinggi")
        if atr_pct < cfg.min_atr_pct:
            blockers.append(f"ATR {atr_pct:.1%} terlalu rendah/stagnan")
        if atr_pct > cfg.max_atr_pct:
            blockers.append(f"ATR {atr_pct:.1%} terlalu ekstrem")
        if lag > cfg.max_data_lag_days:
            blockers.append(f"Data tertinggal {lag} hari dari universe")
        if absolute_age > cfg.max_absolute_data_age_days:
            blockers.append(f"Data absolut sudah berumur {absolute_age} hari")
        if current_bar_incomplete:
            blockers.append("Daily bar hari ini belum dianggap final")
        if adtv > 0 and value_today < 0.15 * adtv:
            blockers.append("Nilai transaksi bar terakhir sangat rendah")
        if len(frame) >= 2 and near_upper_auto_rejection(
            _number(previous["Close"]), close, _number(last["High"])
        ):
            blockers.append("Harga dekat/terkunci ARA; risiko mengejar harga")
        metrics: dict[str, float | int | str] = {
            "last_price": close,
            "last_date": pd.Timestamp(frame.index[-1]).date().isoformat(),
            "data_lag_days": lag,
            "absolute_data_age_days": absolute_age,
            "current_bar_incomplete": int(current_bar_incomplete),
            "adtv20_idr": adtv,
            "atr_pct": atr_pct,
            "zero_volume_ratio20": zero_vol,
            "volume_ratio": _number(last.get("VOL_RATIO")),
            "rsi14": _number(last.get("RSI14")),
            "adx14": _number(last.get("ADX14")),
            "cmf20": _number(last.get("CMF20")),
            "roc60": _number(last.get("ROC60")),
            "distance_52w_high": _number(last.get("DIST_52W_HIGH")),
            "relative_strength60": _number(last.get("REL_STRENGTH60")),
        }
        return blockers, metrics

    def _finalize(
        self,
        plan: SetupPlan,
        frame: pd.DataFrame,
        context: MarketContext,
        trade_blockers: list[str],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = self.config
        result = plan.to_dict()
        close = float(metrics["last_price"])
        atr_value = _number(frame.iloc[-1].get("ATR14"), 0)
        score_adjustment = {"RISK_ON": 3.0, "NEUTRAL": 0.0, "RISK_OFF": -8.0, "UNKNOWN": -5.0}
        quality_score = round(max(0.0, min(100.0, plan.setup_score + score_adjustment[context.regime])), 1)
        blockers = list(trade_blockers)
        status = "REJECT"

        if plan.detected and not plan.invalidated:
            if plan.zone_age_bars is not None and plan.zone_age_bars > cfg.max_zone_age_bars:
                blockers.append(f"Zona berumur {plan.zone_age_bars} bar; sudah kedaluwarsa")
            if plan.valid_until is not None and pd.Timestamp(frame.index[-1]) > pd.Timestamp(plan.valid_until):
                blockers.append("Masa berlaku setup sudah habis")
            if plan.distance_atr is not None and plan.distance_atr > cfg.watch_distance_atr:
                blockers.append(f"Harga {plan.distance_atr:.2f} ATR dari zona; terlalu jauh")
                plan.action = "TOO_EXTENDED_WAIT_NEW_BASE"
                result["action"] = plan.action
            if plan.entry_low is not None and close < plan.entry_low - 0.75 * atr_value:
                blockers.append("Harga sudah menutup jauh di bawah zona entry")
            if plan.entry and plan.stop_loss:
                stop_pct = (plan.entry - plan.stop_loss) / plan.entry
                result["stop_pct"] = stop_pct
                if stop_pct > cfg.max_stop_pct:
                    blockers.append(f"Jarak SL {stop_pct:.1%} melebihi batas")
            else:
                result["stop_pct"] = np.nan
                blockers.append("Level entry/SL tidak valid")
            levels = (plan.entry, plan.stop_loss, plan.tp1, plan.tp2)
            if not all(is_valid_idx_price(level) for level in levels):
                blockers.append("Satu atau lebih level order tidak sesuai fraksi harga IDX")
            if all(level is not None for level in levels):
                if not (float(plan.stop_loss) < float(plan.entry) < float(plan.tp1) < float(plan.tp2)):
                    blockers.append("Urutan SL < entry < TP1 < TP2 tidak valid")
            if plan.entry is not None and not within_idx_daily_price_band(plan.entry, close):
                blockers.append("Entry berada di luar rentang auto-rejection sesi berikutnya")
            if (plan.rr1 or 0) < cfg.min_rr1 or (plan.rr2 or 0) < cfg.min_rr2:
                blockers.append("Risk/reward di bawah minimum")
            if quality_score < cfg.min_score:
                blockers.append(f"Quality score {quality_score:.0f} di bawah {cfg.min_score:.0f}")
            if context.regime == "RISK_OFF":
                blockers.append("Regime IHSG RISK_OFF")
            elif context.regime == "UNKNOWN":
                blockers.append("Regime IHSG tidak dapat diverifikasi")

            ready_action = plan.action in {"READY_TRIGGER", "READY_LIMIT"}
            close_enough = plan.distance_atr is not None and plan.distance_atr <= cfg.ready_distance_atr
            if not ready_action:
                blockers.append("Retest/reclaim/entry trigger belum lengkap")
            elif quality_score < cfg.execution_score:
                blockers.append(
                    f"Quality score {quality_score:.0f} belum mencapai execution threshold {cfg.execution_score:.0f}"
                )
            if plan.distance_atr is None:
                blockers.append("Jarak ke zona tidak dapat dihitung")
            if (
                ready_action
                and close_enough
                and not blockers
                and quality_score >= cfg.execution_score
                and context.regime in {"RISK_ON", "NEUTRAL"}
            ):
                status = "EXECUTION_READY"
            else:
                status = "WATCHLIST_ENTRY"
        result["quality_score"] = quality_score
        result["status"] = status
        result["blockers"] = " • ".join(blockers)
        result["blocker_count"] = len(blockers)
        result["market_regime"] = context.regime
        result["market_reason"] = context.reason
        result["breadth_ema50"] = context.breadth_ema50
        result["breadth_ema200"] = context.breadth_ema200
        result.update(metrics)
        if quality_score >= 88 and not blockers:
            grade = "A"
        elif quality_score >= 78 and len(blockers) <= 1:
            grade = "B+"
        elif quality_score >= 70:
            grade = "B"
        else:
            grade = "C"
        result["grade"] = grade
        result["status_rank"] = STATUS_ORDER[status]
        return result

    def scan(
        self, histories: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        if not histories:
            return {
                "signals": pd.DataFrame(),
                "universe": pd.DataFrame(),
                "prepared": {},
                "market_context": MarketContext(),
            }
        prepared: dict[str, pd.DataFrame] = {}
        for ticker, frame in histories.items():
            if frame is not None and not frame.empty:
                prepared[ticker] = prepare_indicators(frame, benchmark)
        if not prepared:
            return {
                "signals": pd.DataFrame(),
                "universe": pd.DataFrame(),
                "prepared": {},
                "market_context": MarketContext(),
            }
        context, _ = self._market_context(prepared, benchmark)
        asof_candidates = [pd.Timestamp(frame.index[-1]) for frame in prepared.values()]
        if benchmark is not None and not benchmark.empty:
            asof_candidates.append(pd.Timestamp(benchmark.index[-1]))
        asof = max(asof_candidates)
        signal_rows: list[dict[str, Any]] = []
        universe_rows: list[dict[str, Any]] = []

        for ticker, frame in prepared.items():
            trade_blockers, metrics = self._tradeability(frame, asof)
            plans = detect_all_setups(frame, ticker)
            finalized = [self._finalize(plan, frame, context, trade_blockers, metrics) for plan in plans]
            detected = [row for row in finalized if row["detected"]]
            signal_rows.extend(detected)
            candidates = detected or finalized
            best = sorted(candidates, key=lambda x: (x["status_rank"], -x["quality_score"]))[0]
            universe_rows.append(
                {
                    "ticker": ticker,
                    "best_setup": best["setup"] if best["detected"] else "NO_SETUP",
                    "status": best["status"],
                    "quality_score": best["quality_score"],
                    "grade": best["grade"],
                    "reason": best["reason"],
                    "blockers": best["blockers"],
                    **metrics,
                }
            )

        signals = pd.DataFrame(signal_rows)
        universe = pd.DataFrame(universe_rows)
        if not signals.empty:
            signals = signals.sort_values(
                ["status_rank", "quality_score", "rr2", "adtv20_idr"],
                ascending=[True, False, False, False],
                na_position="last",
            ).reset_index(drop=True)
        if not universe.empty:
            universe["status_rank"] = universe["status"].map(STATUS_ORDER)
            universe = universe.sort_values(
                ["status_rank", "quality_score", "adtv20_idr"], ascending=[True, False, False]
            ).drop(columns="status_rank").reset_index(drop=True)
        return {
            "signals": signals,
            "universe": universe,
            "prepared": prepared,
            "market_context": context,
            "asof": asof,
            "config": asdict(self.config),
        }


# ---- backtest ----

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd



SETUPS = (
    "PULLBACK_CONTINUATION",
    "BREAKOUT_RETEST",
    "REVERSAL_ACCUMULATION",
    "UNICORN_SNIPER_ICT",
)
DETECTOR_BY_SETUP = {detector.__name__: detector for detector in SETUP_DETECTORS}
DETECTORS = {
    "PULLBACK_CONTINUATION": SETUP_DETECTORS[0],
    "BREAKOUT_RETEST": SETUP_DETECTORS[1],
    "REVERSAL_ACCUMULATION": SETUP_DETECTORS[2],
    "UNICORN_SNIPER_ICT": SETUP_DETECTORS[3],
}


@dataclass
class BacktestEvent:
    ticker: str
    setup: str
    signal_date: object
    market_regime: str
    quality_score: float
    order_type: str
    planned_entry: float
    stop: float
    tp1: float
    tp2: float
    filled: bool
    fill_date: object | None = None
    fill_wait_bars: int | None = None
    entry: float | None = None
    exit_date: object | None = None
    exit_price: float | None = None
    result: str = "NO_FILL"
    r_multiple: float | None = None
    holding_bars: int | None = None
    tp1_hit: bool = False
    tp2_hit: bool = False
    time_to_tp1_bars: int | None = None
    no_fill_reason: str = ""
    is_oos: bool = False
    oos_fold: int = 0


def _broad_candidate_mask(df: pd.DataFrame, setup: str) -> pd.Series:
    """Cheap superset; the actual signal is always rebuilt by the live detector."""
    bullish_confirmation = df["BULL_REJECTION"] | (
        (df["Close"] > df["High"].shift(1)) & (df["Close"] > df["Open"])
    )
    if setup == "PULLBACK_CONTINUATION":
        trend = (df["EMA20"] > df["EMA50"]) & (df["EMA50"] > df["EMA200"]) & (df["Close"] > df["EMA50"])
        momentum = (df["ROC60"] > 0.04) & (df["DIST_52W_HIGH"] > -0.18)
        touch = (df["Low"] <= df["EMA20"] + 0.35 * df["ATR14"]).rolling(5).max().gt(0)
        return (trend & momentum & touch & bullish_confirmation).fillna(False)
    if setup == "BREAKOUT_RETEST":
        breakout = (
            (df["Close"] > df["HIGH55_PREV"] + 0.05 * df["ATR14"])
            & (df["VOL_RATIO"] >= 1.25)
            & (df["BODY_ATR"] >= 0.40)
            & (df["Close"] > df["Open"])
        )
        return (breakout.rolling(18).max().gt(0) & bullish_confirmation).fillna(False)
    if setup == "REVERSAL_ACCUMULATION":
        sweep = (
            (df["Low"] < df["LOW20_PREV"])
            & (df["Close"] > df["LOW20_PREV"])
            & (df["CLOSE_LOCATION"] > 0.58)
        )
        return (
            sweep.rolling(25).max().gt(0)
            & (df["CMF20"].rolling(10).mean() > -0.02)
            & bullish_confirmation
        ).fillna(False)
    if setup == "UNICORN_SNIPER_ICT":
        sweep = (
            (df["Low"] < df["LOW20_PREV"])
            & (df["Close"] > df["LOW20_PREV"])
            & (df["CLOSE_LOCATION"] >= 0.55)
        )
        bos = (
            (df["Close"] > df["LAST_PIVOT_HIGH"] + 0.05 * df["ATR14"])
            & (df["BODY_ATR"] >= 0.55)
            & (df["Close"] > df["Open"])
        )
        return (
            sweep.rolling(35).max().gt(0)
            & bos.rolling(20).max().gt(0)
            & df["BULL_FVG"].rolling(25).max().gt(0)
            & bullish_confirmation
        ).fillna(False)
    raise ValueError(f"Unknown setup: {setup}")


def historical_signal_mask(df: pd.DataFrame, setup: str) -> pd.Series:
    """Compatibility helper: candidate dates, not a substitute live signal."""
    mask = _broad_candidate_mask(df, setup)
    if len(mask) > 0:
        mask.iloc[: min(205, len(mask))] = False
    return mask


def _historical_context(df: pd.DataFrame) -> MarketContext:
    row = df.iloc[-1]
    values = [row.get(name) for name in ("BENCH_CLOSE", "BENCH_EMA50", "BENCH_EMA200", "BENCH_ROC20")]
    try:
        close, ema50, ema200, roc20 = (float(value) for value in values)
    except (TypeError, ValueError):
        return MarketContext(regime="UNKNOWN", reason="Benchmark historis tidak tersedia")
    if not all(np.isfinite(value) for value in (close, ema50, ema200, roc20)):
        return MarketContext(regime="UNKNOWN", reason="Benchmark historis tidak tersedia")
    if close > ema50 > ema200 and roc20 > 0:
        regime, reason = "RISK_ON", "IHSG historis trend/momentum positif"
    elif close < ema200 and roc20 < 0:
        regime, reason = "RISK_OFF", "IHSG historis di bawah EMA200 dan momentum negatif"
    else:
        regime, reason = "NEUTRAL", "Regime historis campuran"
    return MarketContext(regime=regime, benchmark_close=close, benchmark_roc20=roc20, reason=reason)


def _historical_gate_inputs(df: pd.DataFrame, config: ScanConfig) -> tuple[list[str], dict[str, float | str]]:
    row = df.iloc[-1]
    prev = df.iloc[-2]

    def number(name: str, default: float = float("nan")) -> float:
        try:
            value = float(row.get(name))
            return value if np.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    close = number("Close", 0)
    adtv = number("ADTV20", 0)
    atr_pct = number("ATR_PCT", 0)
    zero = number("ZERO_VOL20", 1)
    blockers: list[str] = []
    if len(df) < config.min_bars:
        blockers.append("Riwayat tidak cukup")
    if close < config.min_price:
        blockers.append("Harga di bawah minimum")
    if adtv < config.min_adtv_idr:
        blockers.append("ADTV di bawah gate")
    if zero > config.max_zero_volume_ratio:
        blockers.append("Hari volume nol terlalu tinggi")
    if not config.min_atr_pct <= atr_pct <= config.max_atr_pct:
        blockers.append("ATR di luar gate")
    if adtv > 0 and number("VALUE", 0) < 0.15 * adtv:
        blockers.append("Nilai transaksi bar sinyal terlalu rendah")
    if near_upper_auto_rejection(float(prev["Close"]), close, float(row["High"])):
        blockers.append("ARA chase")
    metrics: dict[str, float | str] = {
        "last_price": close,
        "last_date": pd.Timestamp(df.index[-1]).date().isoformat(),
        "adtv20_idr": adtv,
        "atr_pct": atr_pct,
        "zero_volume_ratio20": zero,
        "volume_ratio": number("VOL_RATIO"),
        "rsi14": number("RSI14"),
        "adx14": number("ADX14"),
        "cmf20": number("CMF20"),
        "roc60": number("ROC60"),
        "distance_52w_high": number("DIST_52W_HIGH"),
        "relative_strength60": number("REL_STRENGTH60"),
        "data_lag_days": 0.0,
        "absolute_data_age_days": 0.0,
        "current_bar_incomplete": 0.0,
    }
    return blockers, metrics


def _simulate_order(
    df: pd.DataFrame,
    signal_pos: int,
    ticker: str,
    setup: str,
    plan: object,
    quality_score: float,
    regime: str,
    config: ScanConfig,
) -> BacktestEvent:
    planned_entry = float(plan.entry)
    stop = float(plan.stop_loss)
    tp1 = float(plan.tp1)
    tp2 = float(plan.tp2)
    atr_signal = float(df["ATR14"].iloc[signal_pos])
    event = BacktestEvent(
        ticker=ticker,
        setup=setup,
        signal_date=df.index[signal_pos],
        market_regime=regime,
        quality_score=quality_score,
        order_type=str(plan.entry_type),
        planned_entry=planned_entry,
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        filled=False,
    )
    start = signal_pos + 1
    end = min(len(df) - 1, signal_pos + config.backtest_entry_window_bars)
    if start > end:
        event.no_fill_reason = "Tidak ada bar setelah sinyal"
        return event

    fill_pos: int | None = None
    fill_price: float | None = None
    is_stop_order = "BUY_STOP" in str(plan.entry_type)
    for pos in range(start, end + 1):
        day = df.iloc[pos]
        day_open, day_high, day_low = (float(day[name]) for name in ("Open", "High", "Low"))
        if is_stop_order:
            if day_open > planned_entry + config.max_entry_gap_atr * atr_signal:
                event.no_fill_reason = "Gap di atas toleransi; order dibatalkan"
                return event
            if day_open >= planned_entry:
                fill_pos, fill_price = pos, day_open
                break
            if day_high >= planned_entry:
                fill_pos, fill_price = pos, planned_entry
                break
        else:
            if day_open < stop:
                event.no_fill_reason = "Gap di bawah invalidasi sebelum limit fill"
                return event
            if day_open <= planned_entry:
                fill_pos, fill_price = pos, day_open
                break
            if day_low <= planned_entry <= day_high:
                fill_pos, fill_price = pos, planned_entry
                break
    if fill_pos is None or fill_price is None:
        event.no_fill_reason = "Entry tidak tersentuh dalam jendela order"
        return event
    if fill_price <= stop:
        event.no_fill_reason = "Fill tidak valid terhadap stop"
        return event
    rr1_at_fill = (tp1 - fill_price) / (fill_price - stop)
    if rr1_at_fill < config.min_rr1:
        event.no_fill_reason = "Gap/fill menurunkan RR1 di bawah minimum"
        return event

    event.filled = True
    event.fill_date = df.index[fill_pos]
    event.fill_wait_bars = fill_pos - signal_pos
    event.entry = fill_price
    last_pos = min(len(df) - 1, fill_pos + config.backtest_horizon_bars - 1)
    exit_pos = last_pos
    exit_price = float(df["Close"].iloc[last_pos])
    result = "TIME_EXIT"
    tp1_pos: int | None = None
    for pos in range(fill_pos, last_pos + 1):
        day_open = float(df["Open"].iloc[pos])
        day_low = float(df["Low"].iloc[pos])
        day_high = float(df["High"].iloc[pos])
        if day_open <= stop:
            exit_price, exit_pos, result = day_open, pos, "LOSS_GAP"
            break
        # Daily-bar ambiguity is resolved against the strategy: stop first.
        if day_low <= stop:
            exit_price, exit_pos, result = stop, pos, "LOSS"
            break
        if day_high >= tp1:
            exit_price, exit_pos, result = tp1, pos, "WIN_TP1"
            tp1_pos = pos
            event.tp1_hit = True
            event.tp2_hit = day_high >= tp2
            break
    risk_pct = (fill_price - stop) / fill_price
    net_return = exit_price / fill_price - 1 - config.fee_roundtrip_pct - config.slippage_roundtrip_pct
    event.exit_date = df.index[exit_pos]
    event.exit_price = exit_price
    event.result = result
    event.r_multiple = round(float(net_return / risk_pct), 4) if risk_pct > 0 else np.nan
    event.holding_bars = int(exit_pos - fill_pos + 1)
    event.time_to_tp1_bars = int(tp1_pos - fill_pos + 1) if tp1_pos is not None else None
    return event


def simulate_setup(
    df: pd.DataFrame, ticker: str, setup: str, config: ScanConfig
) -> list[BacktestEvent]:
    detector = DETECTORS[setup]
    candidates = np.flatnonzero(historical_signal_mask(df, setup).to_numpy())
    events: list[BacktestEvent] = []
    next_allowed = 0
    engine = ScanEngine(config)
    for pos in candidates:
        if pos < max(205, next_allowed) or pos + 1 >= len(df):
            continue
        snapshot = df.iloc[: pos + 1]
        plan = detector(snapshot, ticker)
        blockers, metrics = _historical_gate_inputs(snapshot, config)
        context = _historical_context(snapshot)
        finalized = engine._finalize(plan, snapshot, context, blockers, metrics)
        if finalized["status"] != "EXECUTION_READY":
            continue
        event = _simulate_order(
            df,
            pos,
            ticker,
            setup,
            plan,
            float(finalized["quality_score"]),
            context.regime,
            config,
        )
        events.append(event)
        if event.filled and event.exit_date is not None:
            exit_pos = int(np.flatnonzero(df.index == event.exit_date)[-1])
            next_allowed = max(pos + config.backtest_min_gap_bars, exit_pos + 1)
        else:
            next_allowed = pos + config.backtest_min_gap_bars
    return events


def _max_losing_streak(values: Iterable[float]) -> int:
    maximum = current = 0
    for value in values:
        if value <= 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def aggregate_backtest(events: pd.DataFrame, config: ScanConfig) -> pd.DataFrame:
    columns = [
        "setup",
        "signal_events_oos",
        "historical_events",
        "filled_events",
        "entry_fill_rate_5d",
        "entry_fill_ci_low",
        "entry_fill_ci_high",
        "historical_hit_rate",
        "bayes_probability",
        "tp1_ci_low",
        "tp1_ci_high",
        "expectancy_r",
        "profit_factor",
        "max_losing_streak",
        "median_fill_bars",
        "median_time_to_tp1_bars",
        "tp1_time_p25",
        "tp1_time_p75",
        "sample_reliability",
        "validation_scope",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    sample = events.copy()
    if "is_oos" in sample and sample["is_oos"].fillna(False).any():
        sample = sample[sample["is_oos"].fillna(False)]
    rows: list[dict[str, object]] = []
    for setup, group in sample.groupby("setup", sort=False):
        filled = group[group.get("filled", True).fillna(False)] if "filled" in group else group
        total_signals = len(group)
        count = len(filled)
        r = pd.to_numeric(filled.get("r_multiple"), errors="coerce").dropna()
        if "tp1_hit" in filled:
            win_mask = filled["tp1_hit"].fillna(False).astype(bool)
            wins = int(win_mask.sum())
        elif "result" in filled:
            win_mask = filled["result"].eq("WIN_TP1")
            wins = int(win_mask.sum())
        else:
            wins = int((r > 0).sum())
            win_mask = pd.Series(False, index=filled.index)
        fill_low, fill_high = _wilson_interval(count, total_signals)
        hit_low, hit_high = _wilson_interval(wins, count)
        bayes = (wins + config.beta_prior_wins) / (
            count + config.beta_prior_wins + config.beta_prior_losses
        ) if count >= 0 else np.nan
        gross_win = r[r > 0].sum()
        gross_loss = -r[r <= 0].sum()
        profit_factor = gross_win / gross_loss if gross_loss > 0 else np.nan
        fill_wait = (
            pd.to_numeric(filled["fill_wait_bars"], errors="coerce").dropna()
            if "fill_wait_bars" in filled
            else pd.Series(dtype=float)
        )
        tp_time = (
            pd.to_numeric(filled.loc[win_mask, "time_to_tp1_bars"], errors="coerce").dropna()
            if "time_to_tp1_bars" in filled
            else pd.Series(dtype=float)
        )
        reliability = "HIGH" if count >= 50 else "MEDIUM" if count >= 30 else "LOW"
        rows.append(
            {
                "setup": setup,
                "signal_events_oos": total_signals,
                "historical_events": count,
                "filled_events": count,
                "entry_fill_rate_5d": round(100 * count / total_signals, 1) if total_signals else np.nan,
                "entry_fill_ci_low": round(100 * fill_low, 1),
                "entry_fill_ci_high": round(100 * fill_high, 1),
                "historical_hit_rate": round(100 * wins / count, 1) if count else np.nan,
                "bayes_probability": round(100 * bayes, 1) if np.isfinite(bayes) else np.nan,
                "tp1_ci_low": round(100 * hit_low, 1),
                "tp1_ci_high": round(100 * hit_high, 1),
                "expectancy_r": round(float(r.mean()), 3) if len(r) else np.nan,
                "profit_factor": round(float(profit_factor), 2) if np.isfinite(profit_factor) else np.nan,
                "max_losing_streak": _max_losing_streak(r.tolist()),
                "median_fill_bars": round(float(fill_wait.median()), 1) if len(fill_wait) else np.nan,
                "median_time_to_tp1_bars": round(float(tp_time.median()), 1) if len(tp_time) else np.nan,
                "tp1_time_p25": round(float(tp_time.quantile(0.25)), 1) if len(tp_time) else np.nan,
                "tp1_time_p75": round(float(tp_time.quantile(0.75)), 1) if len(tp_time) else np.nan,
                "sample_reliability": reliability,
                "validation_scope": "EXPANDING_WINDOW_OOS" if "is_oos" in events else "ALL_EVENTS",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _assign_oos_folds(events: pd.DataFrame, config: ScanConfig) -> pd.DataFrame:
    out = events.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out = out.sort_values(["signal_date", "ticker", "setup"]).reset_index(drop=True)
    unique_dates = np.array(sorted(out["signal_date"].dropna().unique()))
    out["is_oos"] = False
    out["oos_fold"] = 0
    if len(unique_dates) < 5:
        return out
    train_count = max(1, int(math.floor(len(unique_dates) * config.walkforward_min_train_fraction)))
    test_dates = unique_dates[train_count:]
    if not len(test_dates):
        return out
    folds = np.array_split(test_dates, min(config.walkforward_folds, len(test_dates)))
    for number, dates in enumerate(folds, start=1):
        mask = out["signal_date"].isin(dates)
        out.loc[mask, "is_oos"] = True
        out.loc[mask, "oos_fold"] = number
    return out


def run_walkforward_validation(
    prepared: dict[str, pd.DataFrame], config: ScanConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or ScanConfig()
    all_events: list[dict[str, object]] = []
    for ticker, frame in prepared.items():
        if len(frame) < 225:
            continue
        for setup in SETUPS:
            all_events.extend(asdict(event) for event in simulate_setup(frame, ticker, setup, cfg))
    events = pd.DataFrame(all_events)
    if not events.empty:
        events = _assign_oos_folds(events, cfg)
    stats = aggregate_backtest(events, cfg)
    return stats, events


def attach_backtest_stats(signals: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "signal_events_oos",
        "historical_events",
        "filled_events",
        "entry_fill_rate_5d",
        "entry_fill_ci_low",
        "entry_fill_ci_high",
        "historical_hit_rate",
        "bayes_probability",
        "tp1_ci_low",
        "tp1_ci_high",
        "expectancy_r",
        "profit_factor",
        "max_losing_streak",
        "median_fill_bars",
        "median_time_to_tp1_bars",
        "tp1_time_p25",
        "tp1_time_p75",
        "sample_reliability",
        "validation_scope",
    ]
    if signals.empty or stats.empty:
        result = signals.copy()
        for column in columns:
            if column not in result:
                result[column] = np.nan
        return result
    return signals.merge(stats, on="setup", how="left")


__all__ = [
    "ScanConfig", "ScanEngine", "MarketContext", "SetupPlan",
    "download_ohlcv", "download_benchmark", "parse_ticker_csv",
    "run_walkforward_validation", "attach_backtest_stats",
    "fetch_fundamentals", "attach_fundamentals", "apply_fundamental_gate",
    "parse_market_status_csv", "parse_news_review_csv", "parse_broker_summary_csv",
    "apply_market_status_gate", "apply_news_gate", "attach_broker_summary",
    "attach_position_sizing", "make_signal_chart", "__version__",
]

# =============================================================================
# v3.0 autonomous real-money hardening overrides
# =============================================================================

__version__ = "3.0.0-autonomous-fail-closed"

from urllib.parse import urlparse
import re


@dataclass(frozen=True)
class ScanConfig:
    """Strict autonomous defaults.

    EXECUTION_READY is intentionally rare. Any unavailable, stale, ambiguous,
    or statistically weak input fails closed to WATCHLIST_ENTRY/REJECT.
    """

    min_bars: int = 220
    min_price: float = 50.0
    min_adtv_idr: float = 2_000_000_000.0
    min_atr_pct: float = 0.008
    max_atr_pct: float = 0.10
    max_zero_volume_ratio: float = 0.05
    min_score: float = 72.0
    execution_score: float = 82.0
    min_rr1: float = 1.80
    min_rr2: float = 2.70
    max_stop_pct: float = 0.07
    ready_distance_atr: float = 0.30
    max_entry_gap_atr: float = 0.15
    watch_distance_atr: float = 1.75
    max_zone_age_bars: int = 20
    max_data_lag_days: int = 3
    max_absolute_data_age_days: int = 5
    fee_roundtrip_pct: float = 0.0040
    slippage_roundtrip_pct: float = 0.0025
    backtest_horizon_bars: int = 20
    backtest_entry_window_bars: int = 5
    backtest_min_gap_bars: int = 10
    walkforward_min_train_fraction: float = 0.60
    walkforward_folds: int = 4
    min_oos_unique_dates: int = 10
    beta_prior_wins: float = 8.0
    beta_prior_losses: float = 8.0
    fundamental_top_n: int = 50
    min_fundamental_coverage: float = 70.0
    min_fundamental_score: float = 60.0
    real_money_mode: bool = True
    require_fundamentals: bool = True
    require_market_status: bool = True
    require_news_review: bool = True
    require_validation: bool = True
    max_context_age_days: int = 2
    min_news_lookback_days: int = 7
    min_regime_universe_size: int = 200
    min_regime_coverage_pct: float = 80.0
    max_statement_age_days: int = 180

    # OOS hard gate. These are portfolio-protection thresholds, not promises.
    min_oos_signal_events: int = 30
    min_oos_filled_events: int = 20
    min_oos_fill_rate_pct: float = 25.0
    min_oos_bayes_probability_pct: float = 52.0
    min_oos_tp1_ci_low_pct: float = 35.0
    min_oos_expectancy_r: float = 0.10
    min_oos_profit_factor: float = 1.15
    max_oos_losing_streak: int = 8

    # Legacy v3 defaults; the effective v4.2.5 ScanConfig is defined below.
    account_size_idr: float = 10_000_000.0
    risk_per_trade_pct: float = 0.0075
    max_portfolio_risk_pct: float = 0.015
    max_positions: int = 2
    max_position_pct: float = 0.40
    buy_fee_pct: float = 0.0015
    sell_fee_pct: float = 0.0025
    order_slippage_pct: float = 0.0025
    max_order_pct_adtv: float = 0.005

    def replace(self, **changes: object) -> "ScanConfig":
        values = self.__dict__.copy()
        values.update(changes)
        return ScanConfig(**values)


# Preserve v2.1 detectors, then make their execution conditions stricter.
_detect_pullback_v21 = detect_pullback_continuation
_detect_breakout_v21 = detect_breakout_retest
_detect_reversal_v21 = detect_reversal_accumulation
_detect_unicorn_v21 = detect_unicorn_sniper


def detect_pullback_continuation(df: pd.DataFrame, ticker: str) -> SetupPlan:
    plan = _detect_pullback_v21(df, ticker)
    if not plan.detected or df.empty:
        return plan
    row = df.iloc[-1]
    recent = df.iloc[-5:]
    vol_contract = _finite(recent["Volume"].iloc[:-1].mean()) < 0.92 * _finite(df["VOL_MA20"].iloc[-1], 1)
    relative = _finite(row.get("REL_STRENGTH60"), 0) > 0
    flow_ok = _finite(row.get("CMF20"), -1) >= 0 and _finite(row.get("OBV_SLOPE10"), -1) > 0
    trend_strength = _finite(row.get("ADX14"), 0) >= 18
    if not (vol_contract and relative and flow_ok and trend_strength):
        plan.setup_score = min(plan.setup_score, 79.0)
        if plan.action.startswith("READY"):
            plan.action = "WAIT_STRICT_FLOW_CONFIRMATION"
        missing = []
        if not vol_contract:
            missing.append("volume pullback belum kontraksi")
        if not relative:
            missing.append("relative strength vs IHSG negatif")
        if not flow_ok:
            missing.append("CMF/OBV belum mendukung")
        if not trend_strength:
            missing.append("ADX < 18")
        plan.blockers.extend(missing)
        plan.reason += "; strict continuation gate belum lengkap"
    return plan


def detect_breakout_retest(df: pd.DataFrame, ticker: str) -> SetupPlan:
    plan = _detect_breakout_v21(df, ticker)
    if not plan.detected or df.empty:
        return plan
    row = df.iloc[-1]
    close = _finite(row["Close"])
    atr_v = _finite(row["ATR14"])
    trend = _finite(row["EMA20"]) > _finite(row["EMA50"]) > _finite(row["EMA200"])
    current_confirmation = bool(row["BULL_REJECTION"]) or (
        close > _finite(df["High"].iloc[-2]) and close > _finite(row["Open"])
    )
    in_zone = (
        plan.entry_low is not None
        and plan.entry_high is not None
        and atr_v > 0
        and _distance_to_zone(close, float(plan.entry_low), float(plan.entry_high), atr_v) <= 0.35
    )
    if not trend:
        plan.detected = False
        plan.setup_score = min(plan.setup_score, 65.0)
        plan.action = "NO_SETUP"
        plan.reason = "Breakout ditolak: EMA20 > EMA50 > EMA200 tidak terpenuhi"
        return plan
    if plan.action.startswith("READY") and not (current_confirmation and in_zone):
        plan.action = "WAIT_CURRENT_RETEST_CONFIRMATION"
        plan.setup_score = min(plan.setup_score, 79.0)
        plan.blockers.append("Confirmation retest harus terjadi pada bar terakhir")
    return plan


def detect_reversal_accumulation(df: pd.DataFrame, ticker: str) -> SetupPlan:
    plan = _detect_reversal_v21(df, ticker)
    if not plan.detected or df.empty:
        return plan
    row = df.iloc[-1]
    close = _finite(row["Close"])
    base = df.iloc[-30:]
    base_low = _finite(base["Low"].min(), close)
    base_high = _finite(base["High"].max(), close)
    base_width = (base_high - base_low) / close if close > 0 else 1.0
    higher_low = _finite(df["Low"].iloc[-5:].min()) > base_low
    flow_ok = _finite(row.get("CMF20"), -1) > 0.03 and _finite(row.get("OBV_SLOPE10"), -1) > 0
    structure_ok = close > _finite(row.get("EMA20")) and close > _finite(df["High"].iloc[-2])
    if base_width > 0.24:
        plan.detected = False
        plan.action = "NO_SETUP"
        plan.setup_score = min(plan.setup_score, 65.0)
        plan.reason = f"Base terlalu lebar ({base_width:.1%}); reversal belum terkontrol"
        return plan
    if plan.action.startswith("READY") and not (higher_low and flow_ok and structure_ok):
        plan.action = "WAIT_HIGHER_LOW_AND_FLOW"
        plan.setup_score = min(plan.setup_score, 79.0)
        plan.blockers.append("Higher-low, CMF/OBV, dan reclaim current bar wajib lengkap")
    return plan


def detect_unicorn_sniper(df: pd.DataFrame, ticker: str) -> SetupPlan:
    plan = _detect_unicorn_v21(df, ticker)
    if not plan.detected or df.empty:
        return plan
    evidence = set(plan.evidence)
    strict_ob = "FVG overlap dengan order-block proxy" in evidence
    strict_discount = "Zona berada di discount dealing range" in evidence
    volume_ok = False
    if plan.zone_created_date is not None and plan.zone_created_date in df.index:
        pos = int(np.flatnonzero(df.index == plan.zone_created_date)[-1])
        start = max(0, pos - 3)
        volume_ok = bool((pd.to_numeric(df["VOL_RATIO"].iloc[start : pos + 1], errors="coerce") >= 1.20).any())
    if not (strict_ob and strict_discount and volume_ok):
        plan.setup_score = min(plan.setup_score, 74.0)
        if plan.action.startswith("READY"):
            plan.action = "WAIT_STRICT_UNICORN_CONFLUENCE"
        plan.blockers.append("Strict Unicorn memerlukan OB×FVG overlap, discount, dan volume displacement ≥1.20x")
        plan.reason = "Sweep–BOS–FVG valid, tetapi belum strict Unicorn execution grade"
    return plan


SETUP_DETECTORS = (
    detect_pullback_continuation,
    detect_breakout_retest,
    detect_reversal_accumulation,
    detect_unicorn_sniper,
)


def _is_exact_official_idx_url(value: object) -> bool:
    try:
        parsed = urlparse(str(value).strip())
    except Exception:
        return False
    return (
        parsed.scheme == "https"
        and (parsed.hostname or "").lower() in {"idx.co.id", "www.idx.co.id"}
        and bool(parsed.path and parsed.path != "/")
    )


def parse_market_status_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    """Strict parser retained for audit/import compatibility.

    A row is verified only when all status flags are explicitly present, the
    coverage flag is true, and the source host is exactly idx.co.id.
    """
    frame = _read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {
        "ticker", "as_of", "suspended", "special_monitoring", "fca",
        "special_notation", "corporate_action", "source_url",
        "coverage_complete", "verification_method",
    }
    if not {"ticker", "as_of"}.issubset(frame.columns):
        raise ValueError("Market-status CSV wajib memiliki kolom ticker dan as_of")
    out = pd.DataFrame(index=frame.index)
    out["ticker"] = frame["ticker"].map(normalize_idx_ticker)
    out["market_status_asof"] = pd.to_datetime(frame["as_of"], errors="coerce")
    for column in ("suspended", "special_monitoring", "fca", "corporate_action", "sharia"):
        out[column] = frame[column].map(_truthy) if column in frame else False
    out["special_notation"] = _column(frame, "special_notation", "").fillna("").astype(str).str.strip()
    out["market_status_source"] = _column(frame, "source_url", "").fillna("").astype(str).str.strip()
    out["market_status_method"] = _column(frame, "verification_method", "").fillna("").astype(str).str.strip()
    explicit_columns = required.issubset(frame.columns)
    coverage = _column(frame, "coverage_complete", False).map(_truthy)
    official = out["market_status_source"].map(_is_exact_official_idx_url)
    out["market_status_verified"] = (
        explicit_columns
        & coverage
        & official
        & out["market_status_asof"].notna()
        & out["market_status_method"].str.len().gt(0)
    )
    out["market_status_components"] = _column(frame, "coverage_components", "MANUAL_IMPORT").fillna("").astype(str)
    return out.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last")


IDX_AUTOMATION_URLS = {
    "stock_list": "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham/",
    "watchlist": "https://www.idx.co.id/id/perusahaan-tercatat/daftar-efek-pemantauan-khusus",
    "suspension": "https://www.idx.co.id/id/berita/suspensi",
    "long_suspension": "https://www.idx.co.id/id/perusahaan-tercatat/suspensi-lebih-dari-6-bulan/",
    "corporate_actions": "https://www.idx.co.id/id/perusahaan-tercatat/aksi-korporasi",
}


def _fetch_official_idx_pages(timeout: int = 20) -> tuple[dict[str, str], dict[str, str]]:
    import requests

    pages: dict[str, str] = {}
    errors: dict[str, str] = {}
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; IDXSuperScanner/3.2; research-client)",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.7",
    }
    for key, url in IDX_AUTOMATION_URLS.items():
        try:
            response = session.get(url, timeout=timeout, headers=headers, allow_redirects=True)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            if not _is_exact_official_idx_url(response.url):
                raise RuntimeError("redirect keluar domain resmi IDX")
            text = response.text or ""
            if len(text) < 1_000:
                raise RuntimeError("respons terlalu pendek")
            pages[key] = text
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return pages, errors


def _html_text(value: str) -> str:
    clean = re.sub(r"<script\b[^>]*>.*?</script>", " ", value, flags=re.I | re.S)
    clean = re.sub(r"<style\b[^>]*>.*?</style>", " ", clean, flags=re.I | re.S)
    clean = re.sub(r"<[^>]+>", " ", clean)
    return re.sub(r"\s+", " ", clean).upper()


def _requested_mentions(text: str, tickers: Iterable[str]) -> set[str]:
    found: set[str] = set()
    for ticker in tickers:
        code = ticker.replace(".JK", "").upper()
        if re.search(rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])", text):
            found.add(ticker)
    return found


def fetch_automatic_market_status(tickers: Iterable[str], timeout: int = 20) -> pd.DataFrame:
    """Fetch official IDX public pages once and build a conservative blocklist.

    Absence is accepted only when every required page was successfully fetched
    and its semantic marker is present. Any provider failure leaves rows
    unverified, therefore incapable of EXECUTION_READY.
    """
    names = list(dict.fromkeys(tickers))
    if not names:
        return pd.DataFrame()
    now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    pages, errors = _fetch_official_idx_pages(timeout=timeout)
    text = {key: _html_text(value) for key, value in pages.items()}
    semantic_ok = {
        "stock_list": "DAFTAR SAHAM" in text.get("stock_list", "") or "STOCK LIST" in text.get("stock_list", ""),
        "watchlist": "PEMANTAUAN KHUSUS" in text.get("watchlist", "") or "SPECIAL MONITORING" in text.get("watchlist", ""),
        "suspension": "SUSPENSI" in text.get("suspension", "") or "PENGHENTIAN SEMENTARA" in text.get("suspension", ""),
        "long_suspension": "6 BULAN" in text.get("long_suspension", "") or "6 MONTH" in text.get("long_suspension", ""),
        "corporate_actions": "AKSI KORPORASI" in text.get("corporate_actions", "") or "CORPORATE ACTION" in text.get("corporate_actions", ""),
    }
    coverage_complete = len(errors) == 0 and all(semantic_ok.values())
    watchlist = _requested_mentions(text.get("watchlist", ""), names)
    suspended = _requested_mentions(
        text.get("suspension", "") + " " + text.get("long_suspension", ""), names
    )
    corporate = _requested_mentions(text.get("corporate_actions", ""), names)
    listed = _requested_mentions(text.get("stock_list", ""), names)
    source_join = " | ".join(IDX_AUTOMATION_URLS.values())
    components = ",".join(key for key, ok in semantic_ok.items() if ok)
    error_text = " | ".join(f"{key}:{value}" for key, value in errors.items())
    rows = []
    for ticker in names:
        row_verified = coverage_complete and ticker in listed
        rows.append(
            {
                "ticker": ticker,
                "market_status_asof": now,
                "suspended": ticker in suspended,
                "special_monitoring": ticker in watchlist,
                "fca": ticker in watchlist,
                "special_notation": "X/WATCHLIST" if ticker in watchlist else "",
                # Any ticker mentioned on the current corporate-action page is
                # blocked because adjusted levels may be invalid.
                "corporate_action": ticker in corporate,
                "sharia": False,
                "market_status_source": source_join,
                "market_status_method": "OFFICIAL_IDX_AUTOMATED_SCREEN",
                "market_status_components": components,
                "market_status_error": error_text,
                "market_status_verified": bool(row_verified),
            }
        )
    return pd.DataFrame(rows)


def apply_market_status_gate(
    signals: pd.DataFrame,
    market_status: pd.DataFrame,
    config: ScanConfig | None = None,
    asof: object | None = None,
) -> pd.DataFrame:
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    reference = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    if market_status is None or market_status.empty:
        out["market_status_coverage"] = "MISSING"
        for idx in out.index:
            if cfg.real_money_mode and cfg.require_market_status:
                _downgrade(out, idx, "Status otomatis IDX tidak tersedia")
        return out
    out = out.merge(market_status, on="ticker", how="left")
    status_time = pd.to_datetime(out.get("market_status_asof"), errors="coerce")
    out["market_status_age_days"] = (reference.normalize() - status_time.dt.normalize()).dt.days
    verified_mask = out.get("market_status_verified", False)
    if not isinstance(verified_mask, pd.Series):
        verified_mask = pd.Series(False, index=out.index)
    verified_mask = verified_mask.map(_truthy).astype(bool)
    out["market_status_coverage"] = np.where(verified_mask, "AUTO_VERIFIED", "MISSING")
    for idx, row in out.iterrows():
        verified = _truthy(row.get("market_status_verified", False))
        age = pd.to_numeric(row.get("market_status_age_days"), errors="coerce")
        if cfg.real_money_mode and cfg.require_market_status and (
            not verified or pd.isna(age) or age < 0 or age > cfg.max_context_age_days
        ):
            _downgrade(out, idx, "Status IDX otomatis tidak lengkap/terverifikasi/terkini")
        if _truthy(row.get("suspended", False)):
            _downgrade(out, idx, "Saham terdeteksi pada halaman suspensi IDX", reject=True)
        if _truthy(row.get("fca", False)) or _truthy(row.get("special_monitoring", False)):
            _downgrade(out, idx, "Papan Pemantauan Khusus/FCA terdeteksi", reject=True)
        notation = str(row.get("special_notation") or "").strip()
        if notation:
            _downgrade(out, idx, f"Notasi/status khusus IDX: {notation}", reject=True)
        if _truthy(row.get("corporate_action", False)):
            _downgrade(out, idx, "Aksi korporasi terdeteksi; level teknikal dibatalkan", reject=True)
    out["status_rank"] = out["status"].map(STATUS_ORDER)
    return out


def parse_news_review_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    """Strict review parser. Empty/default rows can never mean COMPLETE."""
    frame = _read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "ticker" not in frame or "reviewed_at" not in frame:
        raise ValueError("News-review CSV wajib memiliki kolom ticker dan reviewed_at")
    frame["ticker"] = frame["ticker"].map(normalize_idx_ticker)
    frame["news_reviewed_at"] = pd.to_datetime(frame["reviewed_at"], errors="coerce")
    frame["news_review_status"] = _column(frame, "review_status", "MISSING").fillna("MISSING").astype(str).str.upper()
    frame["news_title"] = _column(frame, "title", "").fillna("").astype(str)
    frame["news_sentiment"] = _column(frame, "sentiment", "NEUTRAL").fillna("NEUTRAL").astype(str).str.upper()
    frame["news_materiality"] = _column(frame, "materiality", "LOW").fillna("LOW").astype(str).str.upper()
    frame["news_source_url"] = _column(frame, "source_url", "").fillna("").astype(str)
    frame["news_verified"] = _column(frame, "verified", False).map(_truthy)
    frame["provider_query_ok"] = _column(frame, "provider_query_ok", False).map(_truthy)
    frame["items_reviewed"] = pd.to_numeric(_column(frame, "items_reviewed", np.nan), errors="coerce")
    frame["coverage_start"] = pd.to_datetime(_column(frame, "coverage_start", pd.NaT), errors="coerce")
    frame["coverage_end"] = pd.to_datetime(_column(frame, "coverage_end", pd.NaT), errors="coerce")
    frame["news_provider"] = _column(frame, "provider", "").fillna("").astype(str)
    rows: list[dict[str, object]] = []
    for ticker, group in frame.dropna(subset=["ticker"]).groupby("ticker", sort=False):
        group = group.sort_values("news_reviewed_at")
        material = group[group["news_title"].str.len().gt(0)]
        severe = material[
            (material["news_sentiment"] == "NEGATIVE")
            & material["news_materiality"].isin(["HIGH", "CRITICAL"])
            & material["news_verified"]
        ]
        latest = group.iloc[-1]
        structural_complete = bool(
            latest["provider_query_ok"]
            and pd.notna(latest["items_reviewed"])
            and latest["items_reviewed"] >= 0
            and pd.notna(latest["coverage_start"])
            and pd.notna(latest["coverage_end"])
            and bool(str(latest["news_provider"]).strip())
        )
        status = str(latest["news_review_status"]).upper()
        if status == "COMPLETE" and not structural_complete:
            status = "INCOMPLETE"
        rows.append(
            {
                "ticker": ticker,
                "news_reviewed_at": latest["news_reviewed_at"],
                "news_review_status": status,
                "provider_query_ok": bool(latest["provider_query_ok"]),
                "items_reviewed": int(latest["items_reviewed"]) if pd.notna(latest["items_reviewed"]) else np.nan,
                "coverage_start": latest["coverage_start"],
                "coverage_end": latest["coverage_end"],
                "news_provider": latest["news_provider"],
                "verified_catalyst_count": int(((material["news_sentiment"] == "POSITIVE") & material["news_verified"]).sum()),
                "verified_negative_count": int(((material["news_sentiment"] == "NEGATIVE") & material["news_verified"]).sum()),
                "severe_negative_news": bool(len(severe)),
                "ambiguous_material_news": False,
                "catalyst_summary": " | ".join(material.tail(3)["news_title"].tolist()),
                "news_sources": " | ".join(material.tail(3)["news_source_url"].tolist()),
            }
        )
    return pd.DataFrame(rows)


_NEGATIVE_NEWS_TERMS = (
    "SUSPENSI", "SUSPENSION", "GAGAL BAYAR", "DEFAULT", "PAILIT", "BANKRUPTCY",
    "PKPU", "FRAUD", "KORUPSI", "CORRUPTION", "DELISTING", "PENIPUAN",
    "PENYIDIKAN", "INVESTIGATION", "GUGATAN", "LAWSUIT", "EKUITAS NEGATIF",
    "DISCLAIMER OPINION", "ADVERSE OPINION", "REVERSE STOCK",
)
_EVENT_RISK_TERMS = (
    "RIGHTS ISSUE", "HAK MEMESAN EFEK TERLEBIH DAHULU", "PRIVATE PLACEMENT",
    "MERGER", "AKUISISI", "ACQUISITION", "STOCK SPLIT", "DIVESTASI",
    "TENDER OFFER", "MATERIAL TRANSACTION", "TRANSAKSI MATERIAL",
)


def _news_item_fields(item: dict[str, Any]) -> tuple[str, str, str, pd.Timestamp | None]:
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = str(content.get("title") or item.get("title") or "").strip()
    summary = str(content.get("summary") or content.get("description") or item.get("summary") or "").strip()
    canonical = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else {}
    click = content.get("clickThroughUrl") if isinstance(content.get("clickThroughUrl"), dict) else {}
    url = str(canonical.get("url") or click.get("url") or content.get("link") or item.get("link") or "").strip()
    raw_date = content.get("pubDate") or item.get("providerPublishTime") or item.get("pubDate")
    published = None
    try:
        if isinstance(raw_date, (int, float)):
            published = pd.to_datetime(raw_date, unit="s", utc=True).tz_convert("Asia/Jakarta").tz_localize(None)
        elif raw_date:
            parsed = pd.to_datetime(raw_date, utc=True, errors="coerce")
            if pd.notna(parsed):
                published = parsed.tz_convert("Asia/Jakarta").tz_localize(None)
    except Exception:
        published = None
    return title, summary, url, published


def _fetch_idx_disclosure_page(timeout: int = 20) -> tuple[str, bool, str]:
    import requests

    url = "https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi"
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; IDXSuperScanner/3.2)"},
            allow_redirects=True,
        )
        if response.status_code != 200 or not _is_exact_official_idx_url(response.url):
            raise RuntimeError(f"HTTP/redirect {response.status_code}")
        text = _html_text(response.text or "")
        ok = len(text) > 1_000 and ("KETERBUKAAN INFORMASI" in text or "DISCLOSURE" in text)
        return text, ok, "" if ok else "semantic marker tidak ditemukan"
    except Exception as exc:
        return "", False, f"{type(exc).__name__}: {str(exc)[:120]}"


def fetch_automatic_news_review(
    tickers: Iterable[str], lookback_days: int = 7, max_workers: int = 4
) -> pd.DataFrame:
    """Automatically review Yahoo news plus the current official IDX disclosure page.

    The classifier is intentionally conservative: negative or event-risk terms
    block execution rather than trying to interpret them optimistically.
    """
    import yfinance as yf

    names = list(dict.fromkeys(tickers))
    if not names:
        return pd.DataFrame()
    now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    start = now - pd.Timedelta(days=lookback_days)
    disclosure_text, disclosure_ok, disclosure_error = _fetch_idx_disclosure_page()
    disclosure_mentions = _requested_mentions(disclosure_text, names) if disclosure_ok else set()

    def one(ticker: str) -> dict[str, Any]:
        query_ok = False
        error = ""
        items: list[dict[str, Any]] = []
        try:
            obj = yf.Ticker(ticker)
            try:
                raw = obj.get_news(count=30)
            except Exception:
                raw = obj.news
            items = list(raw or [])
            query_ok = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:100]}"
        reviewed = []
        severe = False
        ambiguous = ticker in disclosure_mentions
        sources = []
        titles = []
        positive_count = 0
        negative_count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            title, summary, url, published = _news_item_fields(item)
            if published is not None and published < start:
                continue
            text = f"{title} {summary}".upper()
            is_negative = any(term in text for term in _NEGATIVE_NEWS_TERMS)
            is_event = any(term in text for term in _EVENT_RISK_TERMS)
            severe = severe or is_negative
            ambiguous = ambiguous or is_event
            if is_negative:
                negative_count += 1
            elif any(term in text for term in ("PROFIT", "LABA", "GROWTH", "PERTUMBUHAN", "CONTRACT", "KONTRAK")):
                positive_count += 1
            if title:
                titles.append(title)
            if url:
                sources.append(url)
            reviewed.append(item)
        complete = query_ok and disclosure_ok
        return {
            "ticker": ticker,
            "news_reviewed_at": now,
            "news_review_status": "COMPLETE" if complete else "INCOMPLETE",
            "provider_query_ok": bool(complete),
            "items_reviewed": len(reviewed),
            "coverage_start": start,
            "coverage_end": now,
            "news_provider": "Yahoo Finance + official IDX disclosure page",
            "verified_catalyst_count": positive_count,
            "verified_negative_count": negative_count,
            "severe_negative_news": severe,
            "ambiguous_material_news": ambiguous,
            "catalyst_summary": " | ".join(titles[:3]),
            "news_sources": " | ".join(sources[:3]),
            "news_error": " | ".join(value for value in (error, disclosure_error) if value),
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(names))) as pool:
        futures = [pool.submit(one, ticker) for ticker in names]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def apply_news_gate(
    signals: pd.DataFrame,
    news_review: pd.DataFrame,
    config: ScanConfig | None = None,
    asof: object | None = None,
) -> pd.DataFrame:
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    reference = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    if news_review is None or news_review.empty:
        out["news_review_status"] = "MISSING"
        for idx in out.index:
            if cfg.real_money_mode and cfg.require_news_review:
                _downgrade(out, idx, "Review berita otomatis tidak tersedia")
        return out
    out = out.merge(news_review, on="ticker", how="left")
    review_source = out["news_reviewed_at"] if "news_reviewed_at" in out else pd.Series(pd.NaT, index=out.index)
    start_source = out["coverage_start"] if "coverage_start" in out else pd.Series(pd.NaT, index=out.index)
    end_source = out["coverage_end"] if "coverage_end" in out else pd.Series(pd.NaT, index=out.index)
    review_time = pd.to_datetime(review_source, errors="coerce")
    coverage_start = pd.to_datetime(start_source, errors="coerce")
    coverage_end = pd.to_datetime(end_source, errors="coerce")
    out["news_review_age_days"] = (reference.normalize() - review_time.dt.normalize()).dt.days
    out["news_lookback_days"] = (coverage_end.dt.normalize() - coverage_start.dt.normalize()).dt.days
    for idx, row in out.iterrows():
        status = str(row.get("news_review_status") or "MISSING").upper()
        age = pd.to_numeric(row.get("news_review_age_days"), errors="coerce")
        lookback = pd.to_numeric(row.get("news_lookback_days"), errors="coerce")
        complete = (
            status == "COMPLETE"
            and _truthy(row.get("provider_query_ok", False))
            and pd.notna(age)
            and 0 <= age <= cfg.max_context_age_days
            and pd.notna(lookback)
            and lookback >= cfg.min_news_lookback_days
            and bool(str(row.get("news_provider") or "").strip())
        )
        if cfg.real_money_mode and cfg.require_news_review and not complete:
            _downgrade(out, idx, "Coverage berita otomatis tidak lengkap/terkini")
        if _truthy(row.get("severe_negative_news", False)):
            _downgrade(out, idx, "Berita negatif material terdeteksi otomatis", reject=True)
        if _truthy(row.get("ambiguous_material_news", False)):
            _downgrade(out, idx, "Aksi korporasi/berita material ambigu: order diblokir", reject=True)
    out["status_rank"] = out["status"].map(STATUS_ORDER)
    return out


def _assign_oos_folds(events: pd.DataFrame, config: ScanConfig) -> pd.DataFrame:
    out = events.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out = out.sort_values(["signal_date", "ticker", "setup"]).reset_index(drop=True)
    unique_dates = np.array(sorted(out["signal_date"].dropna().unique()))
    out["is_oos"] = False
    out["oos_fold"] = 0
    out["oos_eligible"] = len(unique_dates) >= config.min_oos_unique_dates
    if not bool(out["oos_eligible"].iloc[0]) if len(out) else True:
        return out
    train_count = max(1, int(math.floor(len(unique_dates) * config.walkforward_min_train_fraction)))
    test_dates = unique_dates[train_count:]
    if not len(test_dates):
        out["oos_eligible"] = False
        return out
    folds = np.array_split(test_dates, min(config.walkforward_folds, len(test_dates)))
    for number, dates in enumerate(folds, start=1):
        mask = out["signal_date"].isin(dates)
        out.loc[mask, "is_oos"] = True
        out.loc[mask, "oos_fold"] = number
    return out


def aggregate_backtest(events: pd.DataFrame, config: ScanConfig) -> pd.DataFrame:
    columns = [
        "setup", "signal_events_oos", "all_signal_events", "historical_events", "filled_events",
        "entry_fill_rate_5d", "entry_fill_ci_low", "entry_fill_ci_high",
        "historical_hit_rate", "bayes_probability", "tp1_ci_low", "tp1_ci_high",
        "expectancy_r", "profit_factor", "max_losing_streak", "median_fill_bars",
        "median_time_to_tp1_bars", "tp1_time_p25", "tp1_time_p75",
        "sample_reliability", "validation_scope",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    all_counts = events.groupby("setup").size().to_dict()
    eligible = "oos_eligible" in events and events["oos_eligible"].fillna(False).any()
    has_oos = "is_oos" in events and events["is_oos"].fillna(False).any()
    if not (eligible and has_oos):
        return pd.DataFrame(
            [
                {
                    "setup": setup,
                    "signal_events_oos": 0,
                    "all_signal_events": int(count),
                    "historical_events": 0,
                    "filled_events": 0,
                    "sample_reliability": "NONE",
                    "validation_scope": "INSUFFICIENT_OOS",
                }
                for setup, count in all_counts.items()
            ],
            columns=columns,
        )
    sample = events[events["is_oos"].fillna(False)].copy()
    rows: list[dict[str, object]] = []
    for setup, group in sample.groupby("setup", sort=False):
        filled = group[group["filled"].fillna(False)] if "filled" in group else group
        total_signals = len(group)
        count = len(filled)
        r = pd.to_numeric(filled.get("r_multiple"), errors="coerce").dropna()
        if "tp1_hit" in filled:
            win_mask = filled["tp1_hit"].fillna(False).astype(bool)
        elif "result" in filled:
            win_mask = filled["result"].eq("WIN_TP1")
        else:
            win_mask = r > 0
        wins = int(win_mask.sum())
        fill_low, fill_high = _wilson_interval(count, total_signals)
        hit_low, hit_high = _wilson_interval(wins, count)
        bayes = (wins + config.beta_prior_wins) / (count + config.beta_prior_wins + config.beta_prior_losses)
        gross_win = r[r > 0].sum()
        gross_loss = -r[r <= 0].sum()
        pf = gross_win / gross_loss if gross_loss > 0 else np.nan
        fill_wait = pd.to_numeric(filled.get("fill_wait_bars"), errors="coerce").dropna() if "fill_wait_bars" in filled else pd.Series(dtype=float)
        tp_time = pd.to_numeric(filled.loc[win_mask, "time_to_tp1_bars"], errors="coerce").dropna() if "time_to_tp1_bars" in filled else pd.Series(dtype=float)
        reliability = "HIGH" if count >= 50 else "MEDIUM" if count >= 30 else "LOW"
        rows.append(
            {
                "setup": setup,
                "signal_events_oos": total_signals,
                "all_signal_events": int(all_counts.get(setup, total_signals)),
                "historical_events": count,
                "filled_events": count,
                "entry_fill_rate_5d": round(100 * count / total_signals, 1) if total_signals else np.nan,
                "entry_fill_ci_low": round(100 * fill_low, 1),
                "entry_fill_ci_high": round(100 * fill_high, 1),
                "historical_hit_rate": round(100 * wins / count, 1) if count else np.nan,
                "bayes_probability": round(100 * bayes, 1),
                "tp1_ci_low": round(100 * hit_low, 1),
                "tp1_ci_high": round(100 * hit_high, 1),
                "expectancy_r": round(float(r.mean()), 3) if len(r) else np.nan,
                "profit_factor": round(float(pf), 2) if np.isfinite(pf) else np.nan,
                "max_losing_streak": _max_losing_streak(r.tolist()),
                "median_fill_bars": round(float(fill_wait.median()), 1) if len(fill_wait) else np.nan,
                "median_time_to_tp1_bars": round(float(tp_time.median()), 1) if len(tp_time) else np.nan,
                "tp1_time_p25": round(float(tp_time.quantile(0.25)), 1) if len(tp_time) else np.nan,
                "tp1_time_p75": round(float(tp_time.quantile(0.75)), 1) if len(tp_time) else np.nan,
                "sample_reliability": reliability,
                "validation_scope": "CHRONOLOGICAL_OOS_HOLDOUT",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def attach_backtest_stats(signals: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "signal_events_oos", "all_signal_events", "historical_events", "filled_events",
        "entry_fill_rate_5d", "entry_fill_ci_low", "entry_fill_ci_high",
        "historical_hit_rate", "bayes_probability", "tp1_ci_low", "tp1_ci_high",
        "expectancy_r", "profit_factor", "max_losing_streak", "median_fill_bars",
        "median_time_to_tp1_bars", "tp1_time_p25", "tp1_time_p75",
        "sample_reliability", "validation_scope",
    ]
    if signals.empty:
        return signals.copy()
    if stats is None or stats.empty:
        result = signals.copy()
        for column in columns:
            result[column] = np.nan
        result["validation_scope"] = "MISSING"
        return result
    return signals.merge(stats, on="setup", how="left")


def apply_validation_gate(signals: pd.DataFrame, config: ScanConfig | None = None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    for idx, row in out.iterrows():
        checks = {
            "scope": str(row.get("validation_scope") or "") == "CHRONOLOGICAL_OOS_HOLDOUT",
            "signals": _finite(row.get("signal_events_oos"), 0) >= cfg.min_oos_signal_events,
            "filled": _finite(row.get("filled_events"), 0) >= cfg.min_oos_filled_events,
            "fill_rate": _finite(row.get("entry_fill_rate_5d"), 0) >= cfg.min_oos_fill_rate_pct,
            "probability": _finite(row.get("bayes_probability"), 0) >= cfg.min_oos_bayes_probability_pct,
            "ci": _finite(row.get("tp1_ci_low"), 0) >= cfg.min_oos_tp1_ci_low_pct,
            "expectancy": _finite(row.get("expectancy_r"), -99) >= cfg.min_oos_expectancy_r,
            "profit_factor": _finite(row.get("profit_factor"), 0) >= cfg.min_oos_profit_factor,
            "losing_streak": _finite(row.get("max_losing_streak"), 999) <= cfg.max_oos_losing_streak,
        }
        passed = sum(checks.values())
        out.at[idx, "validation_gate_score"] = round(100 * passed / len(checks), 1)
        out.at[idx, "validation_gate_pass"] = bool(all(checks.values()))
        if cfg.real_money_mode and cfg.require_validation and not all(checks.values()):
            failed = ", ".join(name for name, ok in checks.items() if not ok)
            _downgrade(out, idx, f"OOS hard gate gagal: {failed}")
    out["status_rank"] = out["status"].map(STATUS_ORDER)
    return out


def attach_position_sizing(signals: pd.DataFrame, config: ScanConfig | None = None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    sized_rows = []
    for _, row in out.iterrows():
        sized_rows.append(size_stockbit_order(_finite(row.get("entry"), np.nan), _finite(row.get("stop_loss"), np.nan), cfg))
    sizing = pd.DataFrame(sized_rows, index=out.index)
    for column in sizing.columns:
        out[column] = sizing[column]
    for idx, row in out.iterrows():
        if str(row.get("status")) == "EXECUTION_READY":
            if int(_finite(row.get("suggested_lots"), 0)) < 1:
                _downgrade(out, idx, "Sizing Stockbit kurang dari 1 lot")
                continue
            capital = _finite(row.get("capital_required_idr"), 0)
            adtv = _finite(row.get("adtv20_idr"), 0)
            if adtv <= 0 or capital / adtv > cfg.max_order_pct_adtv:
                _downgrade(out, idx, "Ukuran order terlalu besar terhadap ADTV20")
    out["status_rank"] = out["status"].map(STATUS_ORDER)
    return out


def enforce_portfolio_execution_budget(
    signals: pd.DataFrame,
    config: ScanConfig | None = None,
    current_positions: int = 0,
    current_open_risk_idr: float = 0.0,
    current_invested_idr: float = 0.0,
) -> pd.DataFrame:
    """Select only executable orders that fit aggregate account constraints."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if "technical_execution_candidate" not in out:
        out["technical_execution_candidate"] = out["status"].eq("EXECUTION_READY")
    out["portfolio_selected"] = False
    out["execution_rank"] = np.nan
    # Rank available slots by technical quality plus the evidence already
    # resolved before portfolio allocation. This prevents a lower-confidence
    # candidate from consuming the only slot merely because its setup score is
    # marginally higher.
    out["pre_budget_confidence"] = (
        0.48 * pd.to_numeric(out.get("quality_score", 0), errors="coerce").fillna(0)
        + 0.12 * pd.to_numeric(out.get("market_status_confidence", 45), errors="coerce").fillna(45)
        + 0.08 * pd.to_numeric(out.get("news_confidence", 52), errors="coerce").fillna(52)
        + 0.10 * pd.to_numeric(out.get("fundamental_confidence", 50), errors="coerce").fillna(50)
        + 0.08 * pd.to_numeric(out.get("validation_confidence", 45), errors="coerce").fillna(45)
        + 0.08 * pd.to_numeric(out.get("quote_confidence", 68), errors="coerce").fillna(68)
        + 0.06 * pd.to_numeric(out.get("universe_confidence", 48), errors="coerce").fillna(48)
    ).round(1)
    max_risk = cfg.account_size_idr * cfg.max_portfolio_risk_pct
    remaining_risk = max(0.0, max_risk - max(0.0, current_open_risk_idr))
    remaining_cash = max(0.0, cfg.account_size_idr - max(0.0, current_invested_idr))
    slots = max(0, int(cfg.max_positions) - max(0, int(current_positions)))
    ready_index = out.index[out["status"].eq("EXECUTION_READY")].tolist()
    if not ready_index:
        return out
    sort_cols = [column for column in ("composite_score", "quality_score", "bayes_probability", "rr2", "adtv20_idr") if column in out]
    ranked = out.loc[ready_index].sort_values(sort_cols, ascending=False, na_position="last") if sort_cols else out.loc[ready_index]
    seen_tickers: set[str] = set()
    selected = 0
    for idx, row in ranked.iterrows():
        ticker = str(row.get("ticker") or "")
        risk = _finite(row.get("max_loss_idr"), float("inf"))
        capital = _finite(row.get("capital_required_idr"), float("inf"))
        reason = ""
        if ticker in seen_tickers:
            reason = "Hanya satu setup terbaik per ticker boleh dieksekusi"
        elif selected >= slots:
            reason = "Slot posisi portofolio sudah penuh"
        elif risk > remaining_risk:
            reason = "Risiko agregat portofolio melampaui batas"
        elif capital > remaining_cash:
            reason = "Kas tersedia tidak cukup"
        if reason:
            _downgrade(out, idx, reason)
            continue
        selected += 1
        seen_tickers.add(ticker)
        remaining_risk -= max(0.0, risk)
        remaining_cash -= max(0.0, capital)
        out.at[idx, "portfolio_selected"] = True
        out.at[idx, "execution_rank"] = selected
    out["portfolio_remaining_risk_idr"] = remaining_risk
    out["portfolio_remaining_cash_idr"] = remaining_cash
    out["status_rank"] = out["status"].map(STATUS_ORDER)
    return out


def finalize_execution_integrity(signals: pd.DataFrame, config: ScanConfig | None = None) -> pd.DataFrame:
    """Create the final machine-readable Stockbit order instruction."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    for idx, row in out.iterrows():
        gates = {
            "technical": _finite(row.get("quality_score"), 0) >= cfg.execution_score and int(_finite(row.get("blocker_count"), 0)) == 0,
            "validation": _truthy(row.get("validation_gate_pass", False)),
            "fundamental": _finite(row.get("fundamental_coverage"), 0) >= cfg.min_fundamental_coverage and _finite(row.get("fundamental_score"), 0) >= cfg.min_fundamental_score and not bool(str(row.get("fundamental_red_flags") or "").strip()),
            "idx_status": str(row.get("market_status_coverage") or "") == "AUTO_VERIFIED",
            "news": str(row.get("news_review_status") or "").upper() == "COMPLETE" and _truthy(row.get("provider_query_ok", False)),
            "sizing": str(row.get("sizing_status") or "") == "OK" and int(_finite(row.get("suggested_lots"), 0)) >= 1,
            "portfolio": _truthy(row.get("portfolio_selected", False)),
        }
        score = round(100 * sum(gates.values()) / len(gates), 1)
        out.at[idx, "execution_integrity_score"] = score
        ready = str(row.get("status")) == "EXECUTION_READY" and all(gates.values()) and score == 100.0
        if not ready and str(row.get("status")) == "EXECUTION_READY":
            failed = ", ".join(name for name, ok in gates.items() if not ok)
            _downgrade(out, idx, f"Integrity gate tidak 100%: {failed}")
        out.at[idx, "order_instruction"] = "BUY_LIMIT" if ready else "DO_NOT_BUY"
        out.at[idx, "stockbit_order_price"] = row.get("entry") if ready else np.nan
        out.at[idx, "stockbit_order_lots"] = int(_finite(row.get("suggested_lots"), 0)) if ready else 0
        out.at[idx, "automation_decision"] = "DIRECT_EXECUTION_ELIGIBLE" if ready else "BLOCKED"
    out["status_rank"] = out["status"].map(STATUS_ORDER)
    return out


__all__ = [
    "ScanConfig", "ScanEngine", "MarketContext", "SetupPlan",
    "download_ohlcv", "download_benchmark", "parse_ticker_csv",
    "run_walkforward_validation", "attach_backtest_stats", "apply_validation_gate",
    "fetch_fundamentals", "attach_fundamentals", "apply_fundamental_gate",
    "fetch_automatic_market_status", "fetch_automatic_news_review",
    "parse_market_status_csv", "parse_news_review_csv", "parse_broker_summary_csv",
    "apply_market_status_gate", "apply_news_gate", "attach_broker_summary",
    "attach_position_sizing", "enforce_portfolio_execution_budget",
    "finalize_execution_integrity", "make_signal_chart", "__version__",
]

# v3.0.1 corrective overrides: research-only ALL_EVENTS statistics and NaN-safe notation.

_apply_market_status_gate_v300 = apply_market_status_gate


def apply_market_status_gate(
    signals: pd.DataFrame,
    market_status: pd.DataFrame,
    config: ScanConfig | None = None,
    asof: object | None = None,
) -> pd.DataFrame:
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    reference = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    if market_status is None or market_status.empty:
        out["market_status_coverage"] = "MISSING"
        for idx in out.index:
            if cfg.real_money_mode and cfg.require_market_status:
                _downgrade(out, idx, "Status otomatis IDX tidak tersedia")
        return out
    out = out.merge(market_status, on="ticker", how="left")
    status_time = pd.to_datetime(out.get("market_status_asof"), errors="coerce")
    out["market_status_age_days"] = (reference.normalize() - status_time.dt.normalize()).dt.days
    verified_mask = out.get("market_status_verified", False)
    if not isinstance(verified_mask, pd.Series):
        verified_mask = pd.Series(False, index=out.index)
    verified_mask = verified_mask.map(_truthy).astype(bool)
    out["market_status_coverage"] = np.where(verified_mask, "AUTO_VERIFIED", "MISSING")
    for idx, row in out.iterrows():
        verified = _truthy(row.get("market_status_verified", False))
        age = pd.to_numeric(row.get("market_status_age_days"), errors="coerce")
        if cfg.real_money_mode and cfg.require_market_status and (
            not verified or pd.isna(age) or age < 0 or age > cfg.max_context_age_days
        ):
            _downgrade(out, idx, "Status IDX otomatis tidak lengkap/terverifikasi/terkini")
        if _truthy(row.get("suspended", False)):
            _downgrade(out, idx, "Saham terdeteksi pada halaman suspensi IDX", reject=True)
        if _truthy(row.get("fca", False)) or _truthy(row.get("special_monitoring", False)):
            _downgrade(out, idx, "Papan Pemantauan Khusus/FCA terdeteksi", reject=True)
        notation_value = row.get("special_notation")
        notation = "" if pd.isna(notation_value) else str(notation_value).strip()
        if notation:
            _downgrade(out, idx, f"Notasi/status khusus IDX: {notation}", reject=True)
        if _truthy(row.get("corporate_action", False)):
            _downgrade(out, idx, "Aksi korporasi terdeteksi; level teknikal dibatalkan", reject=True)
    out["status_rank"] = out["status"].map(STATUS_ORDER)
    return out


def aggregate_backtest(events: pd.DataFrame, config: ScanConfig) -> pd.DataFrame:
    columns = [
        "setup", "signal_events_oos", "all_signal_events", "historical_events", "filled_events",
        "entry_fill_rate_5d", "entry_fill_ci_low", "entry_fill_ci_high",
        "historical_hit_rate", "bayes_probability", "tp1_ci_low", "tp1_ci_high",
        "expectancy_r", "profit_factor", "max_losing_streak", "median_fill_bars",
        "median_time_to_tp1_bars", "tp1_time_p25", "tp1_time_p75",
        "sample_reliability", "validation_scope",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    all_counts = events.groupby("setup").size().to_dict()
    has_oos_columns = "is_oos" in events.columns and "oos_eligible" in events.columns
    if has_oos_columns:
        eligible = events["oos_eligible"].fillna(False).any()
        has_oos = events["is_oos"].fillna(False).any()
        if not (eligible and has_oos):
            return pd.DataFrame(
                [
                    {
                        "setup": setup,
                        "signal_events_oos": 0,
                        "all_signal_events": int(count),
                        "historical_events": 0,
                        "filled_events": 0,
                        "sample_reliability": "NONE",
                        "validation_scope": "INSUFFICIENT_OOS",
                    }
                    for setup, count in all_counts.items()
                ],
                columns=columns,
            )
        sample = events[events["is_oos"].fillna(False)].copy()
        scope = "CHRONOLOGICAL_OOS_HOLDOUT"
    else:
        sample = events.copy()
        scope = "ALL_EVENTS_RESEARCH_ONLY"
    rows: list[dict[str, object]] = []
    for setup, group in sample.groupby("setup", sort=False):
        filled = group[group["filled"].fillna(False)] if "filled" in group else group
        total_signals = len(group)
        count = len(filled)
        r = pd.to_numeric(filled.get("r_multiple"), errors="coerce").dropna()
        if "tp1_hit" in filled:
            win_mask = filled["tp1_hit"].fillna(False).astype(bool)
        elif "result" in filled:
            win_mask = filled["result"].eq("WIN_TP1")
        else:
            win_mask = r > 0
        wins = int(win_mask.sum())
        fill_low, fill_high = _wilson_interval(count, total_signals)
        hit_low, hit_high = _wilson_interval(wins, count)
        bayes = (wins + config.beta_prior_wins) / (count + config.beta_prior_wins + config.beta_prior_losses)
        gross_win = r[r > 0].sum()
        gross_loss = -r[r <= 0].sum()
        pf = gross_win / gross_loss if gross_loss > 0 else np.nan
        fill_wait = pd.to_numeric(filled.get("fill_wait_bars"), errors="coerce").dropna() if "fill_wait_bars" in filled else pd.Series(dtype=float)
        tp_time = pd.to_numeric(filled.loc[win_mask, "time_to_tp1_bars"], errors="coerce").dropna() if "time_to_tp1_bars" in filled else pd.Series(dtype=float)
        reliability = "HIGH" if count >= 50 else "MEDIUM" if count >= 30 else "LOW"
        rows.append(
            {
                "setup": setup,
                "signal_events_oos": total_signals if scope == "CHRONOLOGICAL_OOS_HOLDOUT" else 0,
                "all_signal_events": int(all_counts.get(setup, total_signals)),
                "historical_events": count,
                "filled_events": count,
                "entry_fill_rate_5d": round(100 * count / total_signals, 1) if total_signals else np.nan,
                "entry_fill_ci_low": round(100 * fill_low, 1),
                "entry_fill_ci_high": round(100 * fill_high, 1),
                "historical_hit_rate": round(100 * wins / count, 1) if count else np.nan,
                "bayes_probability": round(100 * bayes, 1),
                "tp1_ci_low": round(100 * hit_low, 1),
                "tp1_ci_high": round(100 * hit_high, 1),
                "expectancy_r": round(float(r.mean()), 3) if len(r) else np.nan,
                "profit_factor": round(float(pf), 2) if np.isfinite(pf) else np.nan,
                "max_losing_streak": _max_losing_streak(r.tolist()),
                "median_fill_bars": round(float(fill_wait.median()), 1) if len(fill_wait) else np.nan,
                "median_time_to_tp1_bars": round(float(tp_time.median()), 1) if len(tp_time) else np.nan,
                "tp1_time_p25": round(float(tp_time.quantile(0.25)), 1) if len(tp_time) else np.nan,
                "tp1_time_p75": round(float(tp_time.quantile(0.75)), 1) if len(tp_time) else np.nan,
                "sample_reliability": reliability,
                "validation_scope": scope,
            }
        )
    return pd.DataFrame(rows, columns=columns)


__all__ = list(dict.fromkeys(__all__ + ["apply_validation_gate", "fetch_automatic_market_status", "fetch_automatic_news_review", "enforce_portfolio_execution_budget", "finalize_execution_integrity"]))

# Ensure historical validation uses the exact same hardened detector objects as live scanning.
DETECTORS = {
    "PULLBACK_CONTINUATION": detect_pullback_continuation,
    "BREAKOUT_RETEST": detect_breakout_retest,
    "REVERSAL_ACCUMULATION": detect_reversal_accumulation,
    "UNICORN_SNIPER_ICT": detect_unicorn_sniper,
}
DETECTOR_BY_SETUP = {name: detector for name, detector in DETECTORS.items()}

# =============================================================================
# v3.1 execution-snapshot, financial-recency, and OHLCV smart-money gates
# =============================================================================

__version__ = "3.2.0-autonomous-fail-closed"


def _silent_accumulation_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    if frame is None or len(frame) < 25:
        return 0.0, np.nan
    row = frame.iloc[-1]
    recent = frame.iloc[-20:].copy()
    value = pd.to_numeric(recent.get("VALUE"), errors="coerce").fillna(0.0)
    direction = pd.to_numeric(recent["Close"], errors="coerce").diff()
    up_value = float(value[direction > 0].sum())
    down_value = float(value[direction < 0].sum())
    up_down = up_value / down_value if down_value > 0 else (3.0 if up_value > 0 else np.nan)
    cmf_value = _finite(row.get("CMF20"), -1)
    obv_slope = _finite(row.get("OBV_SLOPE10"), -1)
    close = _finite(row.get("Close"), 0)
    vwap = _finite(row.get("VWAP20"), float("inf"))
    relative = _finite(row.get("REL_STRENGTH60"), -1)
    vol_ratio = _finite(row.get("VOL_RATIO"), 0)
    score = 0.0
    score += 25 if cmf_value >= 0.05 else 15 if cmf_value > 0 else 0
    score += 20 if obv_slope > 0 else 0
    score += 15 if close >= vwap else 0
    score += 20 if np.isfinite(up_down) and up_down >= 1.15 else 10 if np.isfinite(up_down) and up_down >= 1.0 else 0
    score += 15 if relative > 0 else 0
    score += 5 if 0.70 <= vol_ratio <= 2.50 else 0
    return min(100.0, score), up_down


_tradeability_v300 = ScanEngine._tradeability


def _tradeability_v310(self: ScanEngine, frame: pd.DataFrame, asof: pd.Timestamp):
    blockers, metrics = _tradeability_v300(self, frame, asof)
    smart_score, up_down = _silent_accumulation_metrics(frame)
    metrics["silent_accumulation_score"] = smart_score
    metrics["up_down_value_ratio20"] = up_down
    if self.config.real_money_mode and smart_score < 60:
        blockers.append(f"Silent-accumulation proxy {smart_score:.0f}/100 di bawah 60")
    return blockers, metrics


ScanEngine._tradeability = _tradeability_v310


_historical_gate_inputs_v300 = _historical_gate_inputs


def _historical_gate_inputs(df: pd.DataFrame, config: ScanConfig):
    blockers, metrics = _historical_gate_inputs_v300(df, config)
    smart_score, up_down = _silent_accumulation_metrics(df)
    metrics["silent_accumulation_score"] = smart_score
    metrics["up_down_value_ratio20"] = up_down
    if config.real_money_mode and smart_score < 60:
        blockers.append("Silent-accumulation proxy di bawah gate")
    return blockers, metrics


_fetch_one_fundamental_v21 = fetch_one_fundamental


def fetch_one_fundamental(ticker: str) -> dict[str, Any]:
    import yfinance as yf

    now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    try:
        obj = yf.Ticker(ticker)
        info = obj.get_info() or {}
        row = score_fundamentals(info)

        # get_info already returns statement-period metadata for most symbols.
        # Reusing it avoids a second Yahoo quoteSummary request per ticker.
        latest_statement = pd.NaT
        raw_statement = info.get("mostRecentQuarter") or info.get("lastFiscalYearEnd")
        if raw_statement is not None:
            try:
                if isinstance(raw_statement, (int, float, np.integer, np.floating)):
                    latest_statement = (
                        pd.to_datetime(raw_statement, unit="s", utc=True)
                        .tz_convert("Asia/Jakarta")
                        .tz_localize(None)
                    )
                else:
                    latest_statement = _as_jakarta_naive_timestamp(raw_statement)
            except Exception:
                latest_statement = pd.NaT
        statement_error = ""
        statement_age = (
            int((now.normalize() - latest_statement.normalize()).days)
            if pd.notna(latest_statement)
            else np.nan
        )
        row.update(
            {
                "ticker": ticker,
                "latest_statement_date": latest_statement,
                "statement_age_days": statement_age,
                "fundamental_error": statement_error,
                "fundamental_provider": "Yahoo Finance via yfinance",
                "fundamental_fetched_at": pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
            }
        )
        return row
    except Exception as exc:
        return {
            "ticker": ticker,
            "fundamental_score": np.nan,
            "fundamental_coverage": 0.0,
            "fundamental_reliability": "NONE",
            "fundamental_red_flags": "",
            "latest_statement_date": pd.NaT,
            "statement_age_days": np.nan,
            "fundamental_error": f"{type(exc).__name__}: {str(exc)[:100]}",
            "fundamental_provider": "Yahoo Finance via yfinance",
            "fundamental_fetched_at": pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
        }


_apply_fundamental_gate_v21 = apply_fundamental_gate


def apply_fundamental_gate(signals: pd.DataFrame, config: ScanConfig | None = None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    out = _apply_fundamental_gate_v21(signals, cfg)
    if out.empty:
        return out
    if "statement_age_days" not in out:
        out["statement_age_days"] = np.nan
    max_statement_age = cfg.max_statement_age_days
    for idx, row in out.iterrows():
        age = pd.to_numeric(row.get("statement_age_days"), errors="coerce")
        if cfg.real_money_mode and cfg.require_fundamentals and (
            pd.isna(age) or age < 0 or age > max_statement_age
        ):
            if out.at[idx, "status"] == "EXECUTION_READY":
                out.at[idx, "status"] = "WATCHLIST_ENTRY"
            _fundamental_append_blocker(out, idx, "Tanggal laporan keuangan tidak tersedia atau terlalu lama")
    out["status_rank"] = out["status"].map(STATUS_ORDER)
    return out


def _fast_value(fast: Any, key: str, default: Any = np.nan) -> Any:
    try:
        if hasattr(fast, "get"):
            value = fast.get(key, default)
        else:
            value = getattr(fast, key, default)
        return value
    except Exception:
        return default


def fetch_execution_snapshots(tickers: Iterable[str], max_workers: int = 4) -> pd.DataFrame:
    """Fetch a fresh quote-state check for final candidates.

    This does not place orders and is not a replacement for an exchange/broker
    order book. It only prevents execution when the public quote is stale,
    inconsistent, non-equity, or has an excessive displayed spread.
    """
    import yfinance as yf

    names = list(dict.fromkeys(tickers))
    if not names:
        return pd.DataFrame()
    now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)

    def one(ticker: str) -> dict[str, Any]:
        try:
            obj = yf.Ticker(ticker)
            fast = obj.fast_info
            info = obj.get_info() or {}
            last_price = _num(_fast_value(fast, "last_price", info.get("regularMarketPrice")))
            previous_close = _num(_fast_value(fast, "previous_close", info.get("regularMarketPreviousClose")))
            bid = _num(info.get("bid"))
            ask = _num(info.get("ask"))
            volume = _num(_fast_value(fast, "last_volume", info.get("regularMarketVolume")))
            raw_time = info.get("regularMarketTime")
            quote_time = pd.NaT
            if raw_time:
                quote_time = pd.to_datetime(raw_time, unit="s", utc=True).tz_convert("Asia/Jakarta").tz_localize(None)
            spread = (ask - bid) / ((ask + bid) / 2) if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask >= bid else np.nan
            exchange = str(info.get("exchange") or _fast_value(fast, "exchange", "")).upper()
            quote_type = str(info.get("quoteType") or "").upper()
            market_state = str(info.get("marketState") or "UNKNOWN").upper()
            verified = bool(
                np.isfinite(last_price)
                and last_price > 0
                and np.isfinite(volume)
                and volume > 0
                and pd.notna(quote_time)
                and quote_type == "EQUITY"
                and exchange in {"JKT", "IDX", "JAKARTA"}
            )
            return {
                "ticker": ticker,
                "quote_checked_at": now,
                "quote_time": quote_time,
                "quote_last_price": last_price,
                "quote_previous_close": previous_close,
                "quote_bid": bid,
                "quote_ask": ask,
                "quote_spread_pct": spread,
                "quote_volume": volume,
                "quote_market_state": market_state,
                "quote_exchange": exchange,
                "quote_type": quote_type,
                "quote_verified": verified,
                "quote_error": "",
            }
        except Exception as exc:
            return {
                "ticker": ticker,
                "quote_checked_at": now,
                "quote_verified": False,
                "quote_error": f"{type(exc).__name__}: {str(exc)[:100]}",
            }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(names))) as pool:
        futures = [pool.submit(one, ticker) for ticker in names]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def apply_execution_snapshot_gate(
    signals: pd.DataFrame,
    snapshots: pd.DataFrame,
    config: ScanConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if snapshots is None or snapshots.empty:
        out["quote_verified"] = False
        for idx in out.index:
            if cfg.real_money_mode:
                _downgrade(out, idx, "Execution quote snapshot tidak tersedia")
        return out
    out = out.merge(snapshots, on="ticker", how="left")
    now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    quote_source = out["quote_time"] if "quote_time" in out else pd.Series(pd.NaT, index=out.index)
    quote_time = pd.to_datetime(quote_source, errors="coerce")
    out["quote_age_days"] = (now.normalize() - quote_time.dt.normalize()).dt.days
    for idx, row in out.iterrows():
        verified = _truthy(row.get("quote_verified", False))
        age = pd.to_numeric(row.get("quote_age_days"), errors="coerce")
        signal_price = _finite(row.get("last_price"), 0)
        quote_price = _finite(row.get("quote_last_price"), 0)
        atr_pct = max(0.0, _finite(row.get("atr_pct"), 0))
        tolerance = max(0.0125, 0.50 * atr_pct)
        divergence = abs(quote_price / signal_price - 1) if signal_price > 0 and quote_price > 0 else np.inf
        market_state = str(row.get("quote_market_state") or "UNKNOWN").upper()
        spread = pd.to_numeric(row.get("quote_spread_pct"), errors="coerce")
        if cfg.real_money_mode and (not verified or pd.isna(age) or age < 0 or age > 3):
            _downgrade(out, idx, "Quote publik tidak terverifikasi atau stale")
        if divergence > tolerance:
            _downgrade(out, idx, f"Harga quote berbeda {divergence:.1%} dari daily snapshot")
        if market_state == "REGULAR":
            _downgrade(out, idx, "Daily candle belum final saat sesi REGULAR; tunggu penutupan pasar")
            if pd.isna(spread) or spread < 0 or spread > 0.015:
                _downgrade(out, idx, "Spread live tidak tersedia atau >1,5%")
        elif market_state not in {"CLOSED", "PRE", "PREPRE", "POST", "UNKNOWN"}:
            _downgrade(out, idx, f"Market state tidak dikenali: {market_state}")
    out["status_rank"] = out["status"].map(STATUS_ORDER)
    return out


_finalize_execution_integrity_v300 = finalize_execution_integrity


def apply_universe_integrity_gate(
    signals: pd.DataFrame,
    requested_tickers: Iterable[str],
    prepared_tickers: Iterable[str],
    config: ScanConfig | None = None,
) -> pd.DataFrame:
    """Require a broad, high-coverage universe before trusting market breadth.

    Breadth calculated from a hand-picked shortlist is selection-biased. In
    real-money mode, execution is therefore blocked unless the uploaded
    universe is broad enough and most requested tickers produced valid data.
    """
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    requested = list(dict.fromkeys(str(t) for t in requested_tickers if str(t)))
    prepared = set(str(t) for t in prepared_tickers if str(t))
    requested_count = len(requested)
    prepared_count = sum(1 for ticker in requested if ticker in prepared)
    coverage_pct = 100.0 * prepared_count / requested_count if requested_count else 0.0
    passed = bool(
        requested_count >= cfg.min_regime_universe_size
        and coverage_pct >= cfg.min_regime_coverage_pct
    )
    out["universe_requested_count"] = requested_count
    out["universe_prepared_count"] = prepared_count
    out["universe_coverage_pct"] = round(coverage_pct, 1)
    out["universe_gate_pass"] = passed
    if cfg.real_money_mode and not passed:
        message = (
            f"Universe regime tidak memadai: {requested_count} ticker, "
            f"coverage {coverage_pct:.1f}%; minimum {cfg.min_regime_universe_size} "
            f"ticker dan {cfg.min_regime_coverage_pct:.0f}%"
        )
        for idx in out.index:
            _downgrade(out, idx, message)
    out["status_rank"] = out["status"].map(STATUS_ORDER)
    return out


def finalize_execution_integrity(signals: pd.DataFrame, config: ScanConfig | None = None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    out = _finalize_execution_integrity_v300(signals, cfg)
    if out.empty:
        return out
    for idx, row in out.iterrows():
        smart_ok = _finite(row.get("silent_accumulation_score"), 0) >= 60
        quote_ok = _truthy(row.get("quote_verified", False))
        universe_ok = _truthy(row.get("universe_gate_pass", False))
        prior_ready = str(row.get("status")) == "EXECUTION_READY"
        if prior_ready and not (smart_ok and quote_ok and universe_ok):
            failed = []
            if not smart_ok:
                failed.append("silent accumulation")
            if not quote_ok:
                failed.append("quote snapshot")
            if not universe_ok:
                failed.append("broad-universe breadth")
            _downgrade(out, idx, "Final autonomous gate gagal: " + ", ".join(failed))
        ready = (
            str(out.at[idx, "status"]) == "EXECUTION_READY"
            and smart_ok
            and quote_ok
            and universe_ok
        )
        # Ten mandatory layers: original seven + accumulation + quote + universe.
        base = _finite(row.get("execution_integrity_score"), 0)
        additional = int(smart_ok) + int(quote_ok) + int(universe_ok)
        score = round((base / 100 * 7 + additional) / 10 * 100, 1)
        out.at[idx, "execution_integrity_score"] = 100.0 if ready else min(score, 99.0)
        out.at[idx, "order_instruction"] = "BUY_LIMIT" if ready else "DO_NOT_BUY"
        out.at[idx, "stockbit_order_price"] = out.at[idx, "entry"] if ready else np.nan
        out.at[idx, "stockbit_order_lots"] = int(_finite(out.at[idx, "suggested_lots"], 0)) if ready else 0
        out.at[idx, "automation_decision"] = "DIRECT_EXECUTION_ELIGIBLE" if ready else "BLOCKED"
    out["status_rank"] = out["status"].map(STATUS_ORDER)
    return out


__all__ = list(dict.fromkeys(__all__ + ["fetch_execution_snapshots", "apply_execution_snapshot_gate", "apply_universe_integrity_gate"]))

# =============================================================================
# v4.1 resilient-evidence and independent portfolio overrides
# =============================================================================

__version__ = "4.1.0-independent-portfolio"

from pathlib import Path
import json
import os
import tempfile

# Expanded status ladder. PENDING_DATA keeps a technically valid setup visible
# while optional evidence is incomplete; BLOCKED_CONTEXT is reserved for an
# explicit adverse event rather than a transient provider failure.
STATUS_ORDER = {
    "EXECUTION_READY": 0,
    "READY_FOR_PRICE_VERIFY": 1,
    "PENDING_DATA": 2,
    "WATCHLIST_ENTRY": 3,
    "BLOCKED_CONTEXT": 4,
    "REJECT": 5,
}


@dataclass(frozen=True)
class ScanConfig:
    """Resilient real-money defaults.

    Critical trading defects still fail closed. Optional context is scored and
    cached so a single data-provider outage no longer erases an otherwise valid
    setup. Direct execution requires no critical blocker and a high weighted
    confidence score.
    """

    min_bars: int = 220
    min_price: float = 50.0
    min_adtv_idr: float = 2_000_000_000.0
    min_atr_pct: float = 0.008
    max_atr_pct: float = 0.10
    max_zero_volume_ratio: float = 0.05
    min_score: float = 72.0
    execution_score: float = 82.0
    min_rr1: float = 1.80
    min_rr2: float = 2.70
    max_stop_pct: float = 0.07
    ready_distance_atr: float = 0.30
    max_entry_gap_atr: float = 0.15
    watch_distance_atr: float = 1.75
    max_zone_age_bars: int = 20
    max_data_lag_days: int = 3
    max_absolute_data_age_days: int = 5
    fee_roundtrip_pct: float = 0.0040
    slippage_roundtrip_pct: float = 0.0025
    backtest_horizon_bars: int = 20
    backtest_entry_window_bars: int = 5
    backtest_min_gap_bars: int = 10
    walkforward_min_train_fraction: float = 0.60
    walkforward_folds: int = 4
    min_oos_unique_dates: int = 10
    beta_prior_wins: float = 8.0
    beta_prior_losses: float = 8.0
    fundamental_top_n: int = 80
    min_fundamental_coverage: float = 60.0
    min_fundamental_score: float = 55.0
    real_money_mode: bool = True
    require_fundamentals: bool = False
    require_market_status: bool = False
    require_news_review: bool = False
    require_validation: bool = False
    max_context_age_days: int = 3
    min_news_lookback_days: int = 7
    min_regime_universe_size: int = 200
    min_regime_coverage_pct: float = 80.0
    max_statement_age_days: int = 270

    # OOS reference thresholds. Weak/missing validation lowers confidence;
    # demonstrated negative expectancy remains a critical blocker.
    min_oos_signal_events: int = 30
    min_oos_filled_events: int = 20
    min_oos_fill_rate_pct: float = 25.0
    min_oos_bayes_probability_pct: float = 52.0
    min_oos_tp1_ci_low_pct: float = 35.0
    min_oos_expectancy_r: float = 0.10
    min_oos_profit_factor: float = 1.15
    max_oos_losing_streak: int = 8

    # Weighted evidence decision model.
    min_execution_confidence: float = 84.0
    min_pending_confidence: float = 68.0
    # Direct execution is never allowed below this evidence-coverage floor.
    # Completeness measures whether each evidence layer is populated; confidence
    # separately measures whether the populated evidence is strong/positive.
    min_data_completeness: float = 80.0
    # A direct core order needs at least a minimally populated fundamental
    # snapshot. Missing data remains visible as PENDING_DATA, never as a silent
    # pass. The stronger 60%/55 score thresholds still control the quality tier.
    min_direct_fundamental_coverage: float = 45.0
    market_status_cache_days: int = 7
    news_cache_days: int = 3
    fundamental_cache_days: int = 75
    provider_retry_count: int = 2
    max_intraday_stale_minutes: int = 20
    # A second price family is required by the real-money dashboard. It can be
    # supplied by an official IDX/Stockbit file or optional Twelve Data XIDX
    # EOD history. Missing verification creates READY_FOR_PRICE_VERIFY instead
    # of silently treating Yahoo plus its own cache as two sources.
    require_independent_price_verification: bool = True
    max_independent_price_age_days: int = 3
    max_secondary_price_divergence_pct: float = 0.0075
    min_secondary_overlap_bars: int = 5
    min_secondary_return_correlation: float = 0.97

    # Stockbit/portfolio defaults calibrated for this Rp5 million account.
    account_size_idr: float = 5_000_000.0
    cash_on_hand_idr: float = 5_000_000.0
    risk_per_trade_pct: float = 0.0050
    max_portfolio_risk_pct: float = 0.015
    max_positions: int = 3
    max_position_pct: float = 0.35
    buy_fee_pct: float = 0.0015
    sell_fee_pct: float = 0.0025
    order_slippage_pct: float = 0.0025
    max_order_pct_adtv: float = 0.005
    max_avg_down_loss_pct: float = 0.10
    max_avg_down_position_pct: float = 0.28

    def replace(self, **changes: object) -> "ScanConfig":
        values = self.__dict__.copy()
        values.update(changes)
        return ScanConfig(**values)


def _safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _append_pipe(frame: pd.DataFrame, index: object, column: str, message: str) -> None:
    if not message:
        return
    if column not in frame:
        frame[column] = ""
    prior = _safe_text(frame.at[index, column])
    pieces = [piece.strip() for piece in prior.split(" • ") if piece.strip()]
    if message not in pieces:
        pieces.append(message)
    frame.at[index, column] = " • ".join(pieces)


def _set_context_block(frame: pd.DataFrame, index: object, message: str, reject: bool = False) -> None:
    _append_pipe(frame, index, "critical_blockers", message)
    _append_pipe(frame, index, "blockers", message)
    current = _safe_text(frame.at[index, "status"] if "status" in frame else "")
    if reject:
        frame.at[index, "status"] = "REJECT"
    elif current not in {"REJECT", "BLOCKED_CONTEXT"}:
        frame.at[index, "status"] = "BLOCKED_CONTEXT"


def _cache_root() -> Path:
    override = os.environ.get("IDX_SCANNER_CACHE_DIR", "").strip()
    root = Path(override) if override else Path(__file__).resolve().parent / ".scanner_cache"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        root = Path(tempfile.gettempdir()) / "idx_super_scanner_cache"
        root.mkdir(parents=True, exist_ok=True)
    return root


def _load_cache(name: str) -> pd.DataFrame:
    path = _cache_root() / f"{name}.json"
    if not path.is_file():
        return pd.DataFrame()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def _write_cache(name: str, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        return
    path = _cache_root() / f"{name}.json"
    tmp = path.with_suffix(".tmp")
    payload = {
        "written_at": pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
        "rows": frame.replace({np.nan: None, pd.NaT: None}).to_dict("records"),
    }
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _as_jakarta_naive_timestamp(value: Any) -> pd.Timestamp:
    """Normalize cache/provider timestamps before age arithmetic.

    Cache JSON can contain ISO-8601 offsets while older/manual cache rows may
    be timezone-naive. Pandas rejects subtraction between the two forms, so all
    timestamps are converted to Jakarta wall time without timezone metadata.
    """
    stamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(stamp):
        return pd.NaT
    stamp = pd.Timestamp(stamp)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Jakarta").tz_localize(None)
    return stamp


def _cache_row_is_fresh(
    row: Mapping[str, Any] | None,
    timestamp_column: str,
    max_age_days: int,
    usable,
    now: pd.Timestamp | None = None,
) -> bool:
    if row is None or not usable(row):
        return False
    reference = now if now is not None else pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    stamp = _as_jakarta_naive_timestamp(row.get(timestamp_column))
    if pd.isna(stamp):
        return False
    age = (reference.normalize() - stamp.normalize()).days
    return 0 <= age <= max_age_days


def _merge_resilient_rows(
    current: pd.DataFrame,
    cached: pd.DataFrame,
    tickers: Iterable[str],
    timestamp_column: str,
    current_usable,
    cached_max_age_days: int,
    cache_label: str,
) -> pd.DataFrame:
    names = list(dict.fromkeys(tickers))
    now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    current_map = {} if current is None or current.empty else {
        str(row["ticker"]): row.to_dict() for _, row in current.dropna(subset=["ticker"]).iterrows()
    }
    cached_map = {} if cached is None or cached.empty else {
        str(row["ticker"]): row.to_dict() for _, row in cached.dropna(subset=["ticker"]).iterrows()
    }
    rows: list[dict[str, Any]] = []
    for ticker in names:
        cur = current_map.get(ticker)
        if cur is not None and current_usable(cur):
            cur = dict(cur)
            cur["evidence_source_tier"] = "LIVE"
            rows.append(cur)
            continue
        old = cached_map.get(ticker)
        old_ok = _cache_row_is_fresh(
            old,
            timestamp_column,
            cached_max_age_days,
            current_usable,
            now=now,
        )
        if old_ok:
            old = dict(old)
            old["evidence_source_tier"] = cache_label
            rows.append(old)
            continue
        fallback = dict(cur or {"ticker": ticker})
        fallback.setdefault("ticker", ticker)
        fallback["evidence_source_tier"] = "UNRESOLVED"
        rows.append(fallback)
    return pd.DataFrame(rows)


def fetch_resilient_market_status(tickers: Iterable[str], config: ScanConfig | None = None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    names = list(dict.fromkeys(tickers))
    current = pd.DataFrame()
    for _ in range(max(1, cfg.provider_retry_count)):
        current = fetch_automatic_market_status(names, timeout=8)
        if not current.empty and current.get("market_status_verified", pd.Series(dtype=bool)).map(_truthy).any():
            break
    cached = _load_cache("market_status")
    usable = lambda row: _truthy(row.get("market_status_verified", False))
    resolved = _merge_resilient_rows(
        current, cached, names, "market_status_asof", usable,
        cfg.market_status_cache_days, "CACHE_FALLBACK",
    )
    if not current.empty:
        live = current[current.get("market_status_verified", False).map(_truthy)].copy()
        if not live.empty:
            prior = cached[~cached.get("ticker", pd.Series(dtype=str)).isin(live["ticker"])] if not cached.empty and "ticker" in cached else pd.DataFrame()
            _write_cache("market_status", pd.concat([live, prior], ignore_index=True))
    return resolved


def fetch_resilient_news_review(
    tickers: Iterable[str], lookback_days: int = 7, config: ScanConfig | None = None
) -> pd.DataFrame:
    cfg = config or ScanConfig()
    names = list(dict.fromkeys(tickers))
    current = pd.DataFrame()
    for _ in range(max(1, cfg.provider_retry_count)):
        current = fetch_automatic_news_review(names, lookback_days=lookback_days)
        if not current.empty and current.get("provider_query_ok", pd.Series(dtype=bool)).map(_truthy).any():
            break
    cached = _load_cache("news_review")

    def usable(row: Mapping[str, Any]) -> bool:
        return (
            _safe_text(row.get("news_review_status")).upper() == "COMPLETE"
            and _truthy(row.get("provider_query_ok", False))
        )

    resolved = _merge_resilient_rows(
        current, cached, names, "news_reviewed_at", usable,
        cfg.news_cache_days, "CACHE_FALLBACK",
    )
    if not current.empty:
        mask = current.get("provider_query_ok", False)
        if not isinstance(mask, pd.Series):
            mask = pd.Series(False, index=current.index)
        live = current[mask.map(_truthy)].copy()
        if not live.empty:
            prior = cached[~cached.get("ticker", pd.Series(dtype=str)).isin(live["ticker"])] if not cached.empty and "ticker" in cached else pd.DataFrame()
            _write_cache("news_review", pd.concat([live, prior], ignore_index=True))
    return resolved


def fetch_resilient_fundamentals(tickers: Iterable[str], config: ScanConfig | None = None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    names = list(dict.fromkeys(tickers))
    cached = _load_cache("fundamentals")

    def usable(row: Mapping[str, Any]) -> bool:
        return _finite(row.get("fundamental_coverage"), 0) >= 45 and not _safe_text(row.get("fundamental_error"))

    # Cache-first is essential for Streamlit Community Cloud. Querying every
    # ticker again before consulting a valid 75-day cache causes avoidable Yahoo
    # crumb/rate-limit failures, especially for the full multibagger universe.
    cached_map = {} if cached is None or cached.empty else {
        str(row["ticker"]): row.to_dict()
        for _, row in cached.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last").iterrows()
    }
    now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    refresh_names = [
        ticker
        for ticker in names
        if not _cache_row_is_fresh(
            cached_map.get(ticker),
            "fundamental_fetched_at",
            cfg.fundamental_cache_days,
            usable,
            now=now,
        )
    ]
    current = fetch_fundamentals(refresh_names) if refresh_names else pd.DataFrame()

    resolved = _merge_resilient_rows(
        current, cached, names, "fundamental_fetched_at", usable,
        cfg.fundamental_cache_days, "CACHE_FALLBACK",
    )
    if not current.empty:
        live = current[current.apply(lambda row: usable(row.to_dict()), axis=1)].copy()
        if not live.empty:
            prior = cached[~cached.get("ticker", pd.Series(dtype=str)).isin(live["ticker"])] if not cached.empty and "ticker" in cached else pd.DataFrame()
            _write_cache("fundamentals", pd.concat([live, prior], ignore_index=True))
    return resolved


def apply_validation_gate(signals: pd.DataFrame, config: ScanConfig | None = None) -> pd.DataFrame:
    """Score historical evidence without discarding a live setup for missing data.

    A sufficiently large, clearly negative OOS sample remains a critical block.
    Otherwise the result contributes a confidence score and warning.
    """
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    out["validation_critical_blocker"] = False
    for idx, row in out.iterrows():
        checks = {
            "scope": _safe_text(row.get("validation_scope")) == "CHRONOLOGICAL_OOS_HOLDOUT",
            "signals": _finite(row.get("signal_events_oos"), 0) >= cfg.min_oos_signal_events,
            "filled": _finite(row.get("filled_events"), 0) >= cfg.min_oos_filled_events,
            "fill_rate": _finite(row.get("entry_fill_rate_5d"), 0) >= cfg.min_oos_fill_rate_pct,
            "probability": _finite(row.get("bayes_probability"), 0) >= cfg.min_oos_bayes_probability_pct,
            "ci": _finite(row.get("tp1_ci_low"), 0) >= cfg.min_oos_tp1_ci_low_pct,
            "expectancy": _finite(row.get("expectancy_r"), -99) >= cfg.min_oos_expectancy_r,
            "profit_factor": _finite(row.get("profit_factor"), 0) >= cfg.min_oos_profit_factor,
            "losing_streak": _finite(row.get("max_losing_streak"), 999) <= cfg.max_oos_losing_streak,
        }
        passed = sum(checks.values())
        raw_score = 100 * passed / len(checks)
        filled = _finite(row.get("filled_events"), 0)
        expectancy = _finite(row.get("expectancy_r"), np.nan)
        pf = _finite(row.get("profit_factor"), np.nan)
        bayes = _finite(row.get("bayes_probability"), np.nan)
        demonstrated_negative = bool(
            filled >= cfg.min_oos_filled_events
            and ((np.isfinite(expectancy) and expectancy < -0.05)
                 or (np.isfinite(pf) and pf < 0.85)
                 or (np.isfinite(bayes) and bayes < 43.0))
        )
        if all(checks.values()):
            confidence, tier = 100.0, "ROBUST"
        elif checks["scope"] and filled >= 10 and np.isfinite(expectancy) and expectancy > 0:
            confidence, tier = max(65.0, raw_score), "USABLE"
        elif demonstrated_negative:
            confidence, tier = min(25.0, raw_score), "NEGATIVE_EDGE"
        else:
            confidence, tier = max(45.0, min(64.0, raw_score)), "LIMITED"
        out.at[idx, "validation_gate_score"] = round(raw_score, 1)
        out.at[idx, "validation_confidence"] = round(confidence, 1)
        out.at[idx, "validation_tier"] = tier
        out.at[idx, "validation_gate_pass"] = bool(all(checks.values()))
        if demonstrated_negative:
            out.at[idx, "validation_critical_blocker"] = True
            _set_context_block(out, idx, "OOS menunjukkan edge negatif yang material")
        elif not all(checks.values()):
            failed = ", ".join(name for name, ok in checks.items() if not ok)
            _append_pipe(out, idx, "evidence_warnings", f"Validasi historis terbatas: {failed}")
    out["status_rank"] = out["status"].map(STATUS_ORDER).fillna(99)
    return out


def apply_fundamental_gate(signals: pd.DataFrame, config: ScanConfig | None = None) -> pd.DataFrame:
    """Use fundamentals as weighted quality evidence.

    Missing fundamentals no longer erase a short-term technical setup. A truly
    distressed combination (negative margin/cash flow plus leverage) blocks.
    """
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    defaults = {
        "fundamental_score": np.nan,
        "fundamental_coverage": 0.0,
        "fundamental_reliability": "NONE",
        "fundamental_red_flags": "",
        "fundamental_error": "Fundamental tidak tersedia",
        "statement_age_days": np.nan,
    }
    for column, default in defaults.items():
        if column not in out:
            out[column] = default
    out["fundamental_critical_blocker"] = False
    for idx, row in out.iterrows():
        score = _finite(row.get("fundamental_score"), np.nan)
        coverage = _finite(row.get("fundamental_coverage"), 0)
        age = _finite(row.get("statement_age_days"), np.nan)
        flags = _safe_text(row.get("fundamental_red_flags"))
        flag_set = {part.strip() for part in flags.split("•") if part.strip()}
        distressed = bool(
            ({"Margin bersih negatif", "OCF negatif"}.issubset(flag_set))
            or ({"DER tinggi", "OCF negatif"}.issubset(flag_set))
            or (coverage >= 60 and np.isfinite(score) and score < 25)
        )
        if distressed:
            confidence, tier = 0.0, "DISTRESSED"
            out.at[idx, "fundamental_critical_blocker"] = True
            _set_context_block(out, idx, "Fundamental distress: arus kas/profitabilitas/leverage tidak aman")
        elif coverage >= cfg.min_fundamental_coverage and np.isfinite(score) and score >= cfg.min_fundamental_score and (not np.isfinite(age) or age <= cfg.max_statement_age_days):
            confidence, tier = min(100.0, max(70.0, score)), "STRONG"
        elif coverage >= 45 and np.isfinite(score):
            confidence, tier = min(78.0, max(52.0, score)), "PARTIAL"
            _append_pipe(out, idx, "evidence_warnings", "Fundamental parsial atau belum mencapai quality threshold")
        else:
            confidence, tier = 50.0, "MISSING_NEUTRAL"
            _append_pipe(out, idx, "evidence_warnings", "Fundamental belum lengkap; bobot confidence dikurangi")
        out.at[idx, "fundamental_confidence"] = round(confidence, 1)
        out.at[idx, "fundamental_tier"] = tier
    out["status_rank"] = out["status"].map(STATUS_ORDER).fillna(99)
    return out


def apply_market_status_gate(
    signals: pd.DataFrame,
    market_status: pd.DataFrame,
    config: ScanConfig | None = None,
    asof: object | None = None,
) -> pd.DataFrame:
    """Block explicit IDX restrictions; score missing provider coverage."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    reference = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    if market_status is None or market_status.empty:
        market_status = pd.DataFrame({"ticker": out["ticker"].drop_duplicates()})
    out = out.merge(market_status, on="ticker", how="left")
    stamp_source = out["market_status_asof"] if "market_status_asof" in out else pd.Series(pd.NaT, index=out.index)
    stamp = pd.to_datetime(stamp_source, errors="coerce")
    out["market_status_age_days"] = (reference.normalize() - stamp.dt.normalize()).dt.days
    out["market_status_critical_blocker"] = False
    for idx, row in out.iterrows():
        verified = _truthy(row.get("market_status_verified", False))
        age = _finite(row.get("market_status_age_days"), np.nan)
        fresh = verified and np.isfinite(age) and 0 <= age <= cfg.market_status_cache_days
        source_tier = _safe_text(row.get("evidence_source_tier")) or ("LIVE" if verified else "UNRESOLVED")
        out.at[idx, "market_status_coverage"] = "AUTO_VERIFIED" if fresh else "FALLBACK_REQUIRED"
        out.at[idx, "market_status_confidence"] = 100.0 if fresh and source_tier == "LIVE" else 82.0 if fresh else 45.0
        if not fresh:
            _append_pipe(out, idx, "evidence_warnings", "Status IDX resmi belum lengkap; quote/OHLCV fallback akan digunakan")
        negative = []
        if _truthy(row.get("suspended", False)):
            negative.append("suspensi")
        if _truthy(row.get("fca", False)) or _truthy(row.get("special_monitoring", False)):
            negative.append("FCA/pemantauan khusus")
        notation = _safe_text(row.get("special_notation"))
        if notation:
            negative.append(f"notasi {notation}")
        if _truthy(row.get("corporate_action", False)):
            negative.append("aksi korporasi material")
        if negative:
            out.at[idx, "market_status_critical_blocker"] = True
            out.at[idx, "market_status_confidence"] = 0.0
            _set_context_block(out, idx, "Status IDX negatif: " + ", ".join(negative), reject=True)
    out["status_rank"] = out["status"].map(STATUS_ORDER).fillna(99)
    return out


def apply_news_gate(
    signals: pd.DataFrame,
    news_review: pd.DataFrame,
    config: ScanConfig | None = None,
    asof: object | None = None,
) -> pd.DataFrame:
    """Treat provider failure as reduced confidence, not as proof of bad news."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    reference = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    if news_review is None or news_review.empty:
        news_review = pd.DataFrame({"ticker": out["ticker"].drop_duplicates()})
    out = out.merge(news_review, on="ticker", how="left")
    review_source = out["news_reviewed_at"] if "news_reviewed_at" in out else pd.Series(pd.NaT, index=out.index)
    start_source = out["coverage_start"] if "coverage_start" in out else pd.Series(pd.NaT, index=out.index)
    end_source = out["coverage_end"] if "coverage_end" in out else pd.Series(pd.NaT, index=out.index)
    review_time = pd.to_datetime(review_source, errors="coerce")
    coverage_start = pd.to_datetime(start_source, errors="coerce")
    coverage_end = pd.to_datetime(end_source, errors="coerce")
    out["news_review_age_days"] = (reference.normalize() - review_time.dt.normalize()).dt.days
    out["news_lookback_days"] = (coverage_end.dt.normalize() - coverage_start.dt.normalize()).dt.days
    out["news_critical_blocker"] = False
    for idx, row in out.iterrows():
        status = _safe_text(row.get("news_review_status")).upper() or "MISSING"
        age = _finite(row.get("news_review_age_days"), np.nan)
        lookback = _finite(row.get("news_lookback_days"), np.nan)
        complete = bool(
            status == "COMPLETE"
            and _truthy(row.get("provider_query_ok", False))
            and np.isfinite(age) and 0 <= age <= cfg.news_cache_days
            and np.isfinite(lookback) and lookback >= cfg.min_news_lookback_days
        )
        positive = _finite(row.get("verified_catalyst_count"), 0)
        negative = _finite(row.get("verified_negative_count"), 0)
        disclosure_field_present = "idx_disclosure_query_ok" in out.columns
        disclosure_ok = _truthy(row.get("idx_disclosure_query_ok", False)) if disclosure_field_present else True
        confidence = 100.0 if complete and disclosure_ok else 88.0 if complete else 52.0
        if complete and positive > 0 and negative == 0 and disclosure_ok:
            confidence = 100.0
        elif complete and negative > 0:
            confidence = 75.0
        if complete and not disclosure_ok:
            _append_pipe(out, idx, "evidence_warnings", "Berita luas tersedia; cross-check keterbukaan IDX belum lengkap")
        out.at[idx, "news_confidence"] = confidence
        if not complete:
            _append_pipe(out, idx, "evidence_warnings", "Coverage berita parsial; tidak dianggap sebagai berita negatif")
        if _truthy(row.get("severe_negative_news", False)):
            out.at[idx, "news_critical_blocker"] = True
            out.at[idx, "news_confidence"] = 0.0
            _set_context_block(out, idx, "Berita negatif material terverifikasi", reject=True)
        elif _truthy(row.get("ambiguous_material_news", False)):
            # Corporate actions require adjusted levels, so this remains a block.
            out.at[idx, "news_critical_blocker"] = True
            out.at[idx, "news_confidence"] = 15.0
            _set_context_block(out, idx, "Aksi korporasi material belum direkonsiliasi")
    out["status_rank"] = out["status"].map(STATUS_ORDER).fillna(99)
    return out


def apply_execution_snapshot_gate(
    signals: pd.DataFrame,
    snapshots: pd.DataFrame,
    config: ScanConfig | None = None,
) -> pd.DataFrame:
    """Use quote snapshots as a confirmation layer with OHLCV fallback."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if snapshots is None or snapshots.empty:
        snapshots = pd.DataFrame({"ticker": out["ticker"].drop_duplicates()})
    out = out.merge(snapshots, on="ticker", how="left")
    now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    quote_source = out["quote_time"] if "quote_time" in out else pd.Series(pd.NaT, index=out.index)
    quote_time = pd.to_datetime(quote_source, errors="coerce")
    out["quote_age_days"] = (now.normalize() - quote_time.dt.normalize()).dt.days
    out["quote_critical_blocker"] = False
    for idx, row in out.iterrows():
        verified = _truthy(row.get("quote_verified", False))
        age = _finite(row.get("quote_age_days"), np.nan)
        signal_price = _finite(row.get("last_price"), 0)
        quote_price = _finite(row.get("quote_last_price"), 0)
        atr_pct = max(0.0, _finite(row.get("atr_pct"), 0))
        tolerance = max(0.015, 0.60 * atr_pct)
        divergence = abs(quote_price / signal_price - 1) if signal_price > 0 and quote_price > 0 else np.nan
        market_state = _safe_text(row.get("quote_market_state")).upper() or "UNKNOWN"
        spread = _finite(row.get("quote_spread_pct"), np.nan)
        if verified and np.isfinite(age) and 0 <= age <= 3:
            confidence = 100.0
        else:
            # A final, fresh OHLCV bar is an acceptable after-close fallback.
            data_age = _finite(row.get("absolute_data_age_days"), 999)
            current_incomplete = _truthy(row.get("current_bar_incomplete", False))
            confidence = 68.0 if data_age <= cfg.max_absolute_data_age_days and not current_incomplete else 25.0
            _append_pipe(out, idx, "evidence_warnings", "Quote snapshot tidak lengkap; menggunakan final OHLCV sebagai fallback")
        out.at[idx, "quote_confidence"] = confidence
        if np.isfinite(divergence) and divergence > tolerance:
            out.at[idx, "quote_critical_blocker"] = True
            out.at[idx, "quote_confidence"] = 0.0
            _set_context_block(out, idx, f"Konflik harga quote vs OHLCV {divergence:.1%}")
        if market_state == "REGULAR":
            out.at[idx, "quote_critical_blocker"] = True
            _set_context_block(out, idx, "Daily candle belum final saat sesi reguler")
            if not np.isfinite(spread) or spread < 0 or spread > 0.015:
                _append_pipe(out, idx, "critical_blockers", "Spread live tidak tersedia atau >1,5%")
        elif market_state not in {"CLOSED", "PRE", "PREPRE", "POST", "UNKNOWN", ""}:
            _append_pipe(out, idx, "evidence_warnings", f"Market state tidak dikenali: {market_state}")
    out["status_rank"] = out["status"].map(STATUS_ORDER).fillna(99)
    return out


def apply_universe_integrity_gate(
    signals: pd.DataFrame,
    requested_tickers: Iterable[str],
    prepared_tickers: Iterable[str],
    config: ScanConfig | None = None,
) -> pd.DataFrame:
    """Score breadth quality; do not erase a stock-specific setup."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    requested = list(dict.fromkeys(str(t) for t in requested_tickers if str(t)))
    prepared = set(str(t) for t in prepared_tickers if str(t))
    requested_count = len(requested)
    prepared_count = sum(1 for ticker in requested if ticker in prepared)
    coverage_pct = 100.0 * prepared_count / requested_count if requested_count else 0.0
    passed = bool(requested_count >= cfg.min_regime_universe_size and coverage_pct >= cfg.min_regime_coverage_pct)
    out["universe_requested_count"] = requested_count
    out["universe_prepared_count"] = prepared_count
    out["universe_coverage_pct"] = round(coverage_pct, 1)
    out["universe_gate_pass"] = passed
    out["universe_confidence"] = 100.0 if passed else 62.0 if prepared_count >= 50 else 48.0
    if not passed:
        _msg = (
            f"Breadth universe terbatas: {requested_count} ticker, coverage {coverage_pct:.1f}%; "
            "benchmark IHSG tetap digunakan"
        )
        for idx in out.index:
            _append_pipe(out, idx, "evidence_warnings", _msg)
    out["status_rank"] = out["status"].map(STATUS_ORDER).fillna(99)
    return out


def enforce_portfolio_execution_budget(
    signals: pd.DataFrame,
    config: ScanConfig | None = None,
    current_positions: int = 0,
    current_open_risk_idr: float = 0.0,
    current_invested_idr: float = 0.0,
    cash_on_hand_idr: float | None = None,
) -> pd.DataFrame:
    """Rank technically ready orders against actual cash and aggregate risk."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if "technical_execution_candidate" not in out:
        out["technical_execution_candidate"] = out["status"].eq("EXECUTION_READY")
    out["portfolio_selected"] = False
    out["execution_rank"] = np.nan
    max_risk = cfg.account_size_idr * cfg.max_portfolio_risk_pct
    remaining_risk = max(0.0, max_risk - max(0.0, current_open_risk_idr))
    if cash_on_hand_idr is None:
        cash_value = max(0.0, cfg.account_size_idr - max(0.0, current_invested_idr))
    else:
        cash_value = max(0.0, float(cash_on_hand_idr))
    remaining_cash = cash_value
    slots = max(0, int(cfg.max_positions) - max(0, int(current_positions)))
    candidate_index = out.index[out["status"].eq("EXECUTION_READY")].tolist()
    if not candidate_index:
        out["portfolio_remaining_risk_idr"] = remaining_risk
        out["portfolio_remaining_cash_idr"] = remaining_cash
        return out
    sort_cols = [c for c in ("pre_budget_confidence", "composite_score", "quality_score", "bayes_probability", "rr2", "adtv20_idr") if c in out]
    ranked = out.loc[candidate_index].sort_values(sort_cols, ascending=False, na_position="last") if sort_cols else out.loc[candidate_index]
    seen: set[str] = set()
    selected = 0
    for idx, row in ranked.iterrows():
        ticker = _safe_text(row.get("ticker"))
        risk = _finite(row.get("max_loss_idr"), float("inf"))
        capital = _finite(row.get("capital_required_idr"), float("inf"))
        reason = ""
        if ticker in seen:
            reason = "Hanya satu setup terbaik per ticker"
        elif selected >= slots:
            reason = "Slot posisi portofolio sudah penuh"
        elif risk > remaining_risk:
            reason = "Risiko agregat portofolio melampaui batas"
        elif capital > remaining_cash:
            reason = "Cash on hand tidak cukup"
        if reason:
            _append_pipe(out, idx, "portfolio_blockers", reason)
            if _safe_text(out.at[idx, "status"]) == "EXECUTION_READY":
                out.at[idx, "status"] = "PENDING_DATA"
            continue
        selected += 1
        seen.add(ticker)
        remaining_risk -= max(0.0, risk)
        remaining_cash -= max(0.0, capital)
        out.at[idx, "portfolio_selected"] = True
        out.at[idx, "execution_rank"] = selected
    out["portfolio_remaining_risk_idr"] = remaining_risk
    out["portfolio_remaining_cash_idr"] = remaining_cash
    out["status_rank"] = out["status"].map(STATUS_ORDER).fillna(99)
    return out


def _risk_layer_confidence(row: Mapping[str, Any], cfg: ScanConfig) -> float:
    entry = _finite(row.get("entry"), np.nan)
    stop = _finite(row.get("stop_loss"), np.nan)
    tp1 = _finite(row.get("tp1"), np.nan)
    tp2 = _finite(row.get("tp2"), np.nan)
    levels_ok = all(is_valid_idx_price(v) for v in (entry, stop, tp1, tp2)) and stop < entry < tp1 < tp2
    rr_ok = _finite(row.get("rr1"), 0) >= cfg.min_rr1 and _finite(row.get("rr2"), 0) >= cfg.min_rr2
    stop_ok = _finite(row.get("stop_pct"), 99) <= cfg.max_stop_pct
    sizing_ok = _safe_text(row.get("sizing_status")) == "OK" and int(_finite(row.get("suggested_lots"), 0)) >= 1
    return 100.0 * sum((levels_ok, rr_ok, stop_ok, sizing_ok)) / 4


def finalize_execution_integrity(signals: pd.DataFrame, config: ScanConfig | None = None) -> pd.DataFrame:
    """Produce a resilient, auditable Stockbit limit-order decision.

    Missing optional evidence reduces confidence and produces PENDING_DATA only
    when the weighted score falls below threshold. Explicit adverse evidence or
    invalid trade/risk mechanics remain hard blocks.
    """
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    for column, default in (
        ("critical_blockers", ""), ("evidence_warnings", ""),
        ("market_status_confidence", 45.0), ("news_confidence", 52.0),
        ("fundamental_confidence", 50.0), ("validation_confidence", 45.0),
        ("quote_confidence", 68.0), ("universe_confidence", 48.0),
        ("portfolio_selected", False), ("independent_price_verified", False),
        ("independent_price_state", "MISSING_INDEPENDENT"),
        ("independent_source_family", ""),
    ):
        if column not in out:
            out[column] = default
    for idx, row in out.iterrows():
        technical_ready = bool(
            _truthy(row.get("technical_setup_ready", False))
            or _truthy(row.get("technical_execution_candidate", False))
            or _safe_text(row.get("status")) == "EXECUTION_READY"
        )
        technical_conf = min(100.0, max(0.0, _finite(row.get("quality_score"), 0)))
        risk_conf = _risk_layer_confidence(row, cfg)
        market_conf = _finite(row.get("market_status_confidence"), 45)
        news_conf = _finite(row.get("news_confidence"), 52)
        fundamental_conf = _finite(row.get("fundamental_confidence"), 50)
        fundamental_coverage = _finite(row.get("fundamental_coverage"), 0)
        validation_conf = _finite(row.get("validation_confidence"), 45)
        quote_conf = _finite(row.get("quote_confidence"), 68)
        universe_conf = _finite(row.get("universe_confidence"), 48)

        # If official status is unavailable, a verified fresh quote plus recent
        # final OHLCV provides a provisional tradability fallback, never proof
        # against a published suspension.
        if market_conf < 60 and quote_conf >= 90 and _finite(row.get("absolute_data_age_days"), 999) <= cfg.max_absolute_data_age_days:
            market_conf = 70.0
            out.at[idx, "market_status_confidence"] = market_conf
            _append_pipe(out, idx, "evidence_warnings", "Status IDX memakai provisional quote/OHLCV fallback")

        weights = {
            "technical": (technical_conf, 0.35),
            "risk": (risk_conf, 0.20),
            "market_status": (market_conf, 0.10),
            "news": (news_conf, 0.08),
            "fundamental": (fundamental_conf, 0.10),
            "validation": (validation_conf, 0.07),
            "quote": (quote_conf, 0.05),
            "universe": (universe_conf, 0.05),
        }
        confidence = round(sum(value * weight for value, weight in weights.values()), 1)
        complete_layers = sum(value >= 70 for value, _ in weights.values())
        completeness = round(100 * complete_layers / len(weights), 1)
        critical = _safe_text(row.get("critical_blockers"))
        if _truthy(row.get("validation_critical_blocker", False)):
            critical = critical or "OOS edge negatif"
        if _truthy(row.get("fundamental_critical_blocker", False)):
            critical = critical or "Fundamental distress"
        if _truthy(row.get("market_status_critical_blocker", False)):
            critical = critical or "Status IDX negatif"
        if _truthy(row.get("news_critical_blocker", False)):
            critical = critical or "Berita material negatif"
        if _truthy(row.get("quote_critical_blocker", False)):
            critical = critical or "Quote/candle conflict"
        portfolio_ok = _truthy(row.get("portfolio_selected", False))
        if technical_ready and not portfolio_ok:
            _append_pipe(out, idx, "portfolio_blockers", "Belum dipilih oleh budget portofolio")

        direct = bool(
            technical_ready
            and not critical
            and risk_conf == 100.0
            and portfolio_ok
            and confidence >= cfg.min_execution_confidence
        )
        if direct:
            final_status = "EXECUTION_READY"
        elif critical:
            final_status = "REJECT" if _safe_text(row.get("status")) == "REJECT" else "BLOCKED_CONTEXT"
        elif technical_ready and confidence >= cfg.min_pending_confidence:
            final_status = "PENDING_DATA"
        else:
            current = _safe_text(row.get("status"))
            final_status = current if current in {"WATCHLIST_ENTRY", "REJECT"} else "WATCHLIST_ENTRY"

        out.at[idx, "status"] = final_status
        out.at[idx, "execution_integrity_score"] = confidence
        out.at[idx, "execution_confidence_score"] = confidence
        out.at[idx, "data_completeness_score"] = completeness
        out.at[idx, "critical_gate_pass"] = not bool(critical)
        out.at[idx, "evidence_state"] = "RESOLVED" if direct else "ADVERSE" if critical else "PARTIAL"
        out.at[idx, "order_instruction"] = "BUY_LIMIT" if direct else "DO_NOT_BUY"
        out.at[idx, "stockbit_order_price"] = row.get("entry") if direct else np.nan
        out.at[idx, "stockbit_order_lots"] = int(_finite(row.get("suggested_lots"), 0)) if direct else 0
        out.at[idx, "automation_decision"] = "DIRECT_EXECUTION_ELIGIBLE" if direct else "BLOCKED" if critical else "RETRY_OR_WATCH"
    out["status_rank"] = out["status"].map(STATUS_ORDER).fillna(99)
    return out


# ---- Stockbit portfolio snapshot and position decision engine ----


def parse_portfolio_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    frame = _read_csv(source)
    frame.columns = [
        str(column).replace("\ufeff", "").strip().lower().replace(" ", "_")
        for column in frame.columns
    ]
    aliases = {
        "symbol": "ticker", "kode": "ticker", "emiten": "ticker",
        "lot": "lots", "jumlah_lot": "lots", "qty_lot": "lots",
        "avg": "avg_price", "average": "avg_price", "average_price": "avg_price",
        "harga_rata_rata": "avg_price", "stop": "stop_loss", "sl": "stop_loss",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
    required = {"ticker", "lots", "avg_price"}
    if not required.issubset(frame.columns):
        found = ", ".join(map(str, frame.columns)) or "tidak ada"
        raise ValueError(
            "Portfolio CSV wajib memiliki kolom ticker, lots, dan avg_price. "
            f"Kolom terbaca: {found}"
        )
    out = pd.DataFrame()
    out["ticker"] = frame["ticker"].map(normalize_idx_ticker)
    out["lots"] = pd.to_numeric(frame["lots"], errors="coerce").fillna(0).astype(int)
    out["avg_price"] = pd.to_numeric(frame["avg_price"], errors="coerce")
    out["manual_stop_loss"] = pd.to_numeric(frame.get("stop_loss", np.nan), errors="coerce")
    out["manual_tp"] = pd.to_numeric(frame.get("take_profit", frame.get("tp", np.nan)), errors="coerce")
    out["notes"] = frame.get("notes", "").fillna("").astype(str) if "notes" in frame else ""
    out = out.dropna(subset=["ticker", "avg_price"])
    out = out[(out["lots"] > 0) & (out["avg_price"] > 0)]
    if out.empty:
        return out
    # Aggregate duplicated ticker rows at weighted average cost.
    out["shares"] = out["lots"] * 100
    out["cost_value"] = out["shares"] * out["avg_price"]
    grouped = out.groupby("ticker", as_index=False).agg(
        lots=("lots", "sum"), shares=("shares", "sum"), cost_value=("cost_value", "sum"),
        manual_stop_loss=("manual_stop_loss", "last"), manual_tp=("manual_tp", "last"), notes=("notes", "last"),
    )
    grouped["avg_price"] = grouped["cost_value"] / grouped["shares"]
    return grouped.drop(columns="cost_value")


def _portfolio_fundamental_lookup(fundamentals: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if fundamentals is None or fundamentals.empty or "ticker" not in fundamentals:
        return {}
    return {str(row["ticker"]): row.to_dict() for _, row in fundamentals.iterrows()}


def _portfolio_signal_lookup(signals: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if signals is None or signals.empty or "ticker" not in signals:
        return {}
    ranked = signals.copy()
    rank_col = "status_rank" if "status_rank" in ranked else None
    if rank_col:
        ranked = ranked.sort_values([rank_col, "quality_score"], ascending=[True, False], na_position="last")
    return {ticker: group.iloc[0].to_dict() for ticker, group in ranked.groupby("ticker", sort=False)}


def analyze_portfolio_positions(
    portfolio: pd.DataFrame,
    histories: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame | None = None,
    signals: pd.DataFrame | None = None,
    account_equity_idr: float | None = None,
    cash_on_hand_idr: float = 0.0,
    config: ScanConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Analyze Stockbit holdings independently from the universe scanner.

    A missing entry signal is neutral for an existing holding. CUT_LOSS is
    reserved for an actual stop breach, a confirmed multi-factor structure
    breakdown, or severe fundamental distress accompanied by price weakness.
    """
    cfg = config or ScanConfig()
    if portfolio is None or portfolio.empty:
        return pd.DataFrame(), {
            "positions": 0, "market_value_idr": 0.0, "cost_value_idr": 0.0,
            "unrealized_pnl_idr": 0.0, "unrealized_pnl_pct": 0.0,
            "open_risk_idr": 0.0, "open_risk_pct_equity": 0.0,
            "cash_on_hand_idr": float(cash_on_hand_idr),
            "estimated_equity_idr": float(account_equity_idr or cash_on_hand_idr),
            "equity_source": "INPUT" if account_equity_idr else "ESTIMATED",
        }

    fund_map = _portfolio_fundamental_lookup(fundamentals if fundamentals is not None else pd.DataFrame())
    signal_map = _portfolio_signal_lookup(signals if signals is not None else pd.DataFrame())
    provisional_rows: list[dict[str, Any]] = []
    total_market = 0.0
    total_cost = 0.0

    for _, position in portfolio.iterrows():
        ticker = str(position["ticker"])
        raw = histories.get(ticker)
        if raw is None or raw.empty:
            provisional_rows.append({
                **position.to_dict(), "last_price": np.nan, "position_action": "DATA_REQUIRED",
                "action_reason": "OHLCV posisi tidak tersedia", "current_value_idr": np.nan,
                "cost_value_idr": float(position["shares"] * position["avg_price"]),
            })
            continue

        frame = prepare_indicators(raw)
        if frame.empty:
            provisional_rows.append({
                **position.to_dict(), "last_price": np.nan, "position_action": "DATA_REQUIRED",
                "action_reason": "Indikator posisi tidak dapat dihitung", "current_value_idr": np.nan,
                "cost_value_idr": float(position["shares"] * position["avg_price"]),
            })
            continue

        row = frame.iloc[-1]
        prev = frame.iloc[-2] if len(frame) >= 2 else row
        close = _finite(row.get("Close"), np.nan)
        atr_v = max(_finite(row.get("ATR14"), 0.0), idx_tick_size(close) * 3)
        shares = int(position["shares"])
        avg = float(position["avg_price"])
        market_value = shares * close
        cost_value = shares * avg
        total_market += market_value
        total_cost += cost_value

        recent20 = frame.iloc[-20:]
        recent120 = frame.iloc[-120:]
        ema20_v = _finite(row.get("EMA20"), close)
        ema50_v = _finite(row.get("EMA50"), close)
        ema200_v = _finite(row.get("EMA200"), close)
        tick = idx_tick_size(close)

        pivot_lows = pd.to_numeric(recent120.get("PIVOT_LOW"), errors="coerce").dropna()
        pivot_lows = pivot_lows[pivot_lows <= close * 1.02]
        latest_pivot_low = _finite(pivot_lows.iloc[-1], np.nan) if len(pivot_lows) else np.nan
        support_candidates = [
            value for value in (ema20_v, ema50_v, latest_pivot_low, _finite(recent20["Low"].min(), np.nan))
            if np.isfinite(value) and value <= close * 1.02
        ]
        structural_support = max(support_candidates) if support_candidates else close - atr_v
        structural_stop = round_idx_price(
            max(cfg.min_price, structural_support - max(0.50 * atr_v, 2 * tick)), "down"
        )

        manual_stop = _finite(position.get("manual_stop_loss"), np.nan)
        existing_stop = manual_stop if np.isfinite(manual_stop) and manual_stop > 0 else np.nan
        valid_below_stops = [s for s in (existing_stop, structural_stop) if np.isfinite(s) and s < close]
        suggested_stop = max(valid_below_stops) if valid_below_stops else (
            existing_stop if np.isfinite(existing_stop) else structural_stop
        )

        stop_breached = bool(
            (np.isfinite(existing_stop) and close <= existing_stop)
            or close <= structural_stop
        )
        last_two = pd.to_numeric(frame["Close"].iloc[-2:], errors="coerce")
        two_close_break = bool(
            len(last_two) == 2
            and (last_two < min(structural_support, ema50_v)).all()
            and ema20_v < ema50_v
            and _finite(row.get("CMF20"), 0.0) < 0
            and _finite(row.get("OBV_SLOPE10"), 0.0) < 0
        )
        confirmed_breakdown = stop_breached or two_close_break

        risk_unit = max(close - suggested_stop, 1.20 * atr_v, 3 * tick) if suggested_stop < close else max(1.20 * atr_v, 3 * tick)
        pivot_highs = pd.to_numeric(recent120.get("PIVOT_HIGH"), errors="coerce").dropna()
        pivot_highs = sorted(float(v) for v in pivot_highs if float(v) > close)
        tp1_floor = close + max(1.50 * risk_unit, 1.50 * atr_v)
        tp1_cap = close + max(3.00 * risk_unit, 4.00 * atr_v)
        nearest_resistance = next((v for v in pivot_highs if v >= tp1_floor * 0.98 and v <= tp1_cap), np.nan)
        manual_tp = _finite(position.get("manual_tp"), np.nan)
        if np.isfinite(manual_tp) and manual_tp > close:
            suggested_tp1 = round_idx_price(manual_tp, "up")
            tp1_basis = "MANUAL_TP"
        elif np.isfinite(nearest_resistance):
            suggested_tp1 = round_idx_price(nearest_resistance, "up")
            tp1_basis = "NEAREST_PIVOT_RESISTANCE"
        else:
            suggested_tp1 = round_idx_price(tp1_floor, "up")
            tp1_basis = "R_MULTIPLE_ATR"

        min_tp_separation = max(0.75 * atr_v, 0.025 * close, 3 * idx_tick_size(suggested_tp1))
        tp2_floor = max(close + 2.70 * risk_unit, suggested_tp1 + min_tp_separation)
        tp2_cap = close + max(5.00 * risk_unit, 7.00 * atr_v)
        next_resistance = next((v for v in pivot_highs if v >= tp2_floor * 0.98 and v <= tp2_cap), np.nan)
        suggested_tp2 = round_idx_price(next_resistance if np.isfinite(next_resistance) else tp2_floor, "up")
        if suggested_tp2 <= suggested_tp1:
            suggested_tp2 = round_idx_price(suggested_tp1 + min_tp_separation, "up")
        tp2_basis = "NEXT_PIVOT_RESISTANCE" if np.isfinite(next_resistance) else "RUNNER_R_MULTIPLE"

        pnl = market_value - cost_value
        pnl_pct = close / avg - 1
        trend_up = close > ema50_v and ema20_v >= ema50_v and ema50_v >= 0.98 * ema200_v
        long_term_intact = close > ema200_v
        flow_positive = _finite(row.get("CMF20"), -1) >= 0 and _finite(row.get("OBV_SLOPE10"), -1) > 0
        trend_weak = close < ema50_v or ema20_v < ema50_v or _finite(row.get("CMF20"), 0) < 0
        momentum_hot = _finite(row.get("RSI14"), 50) >= 74 or close > ema20_v + 2.0 * atr_v
        near_support = abs(close - structural_support) <= 1.25 * atr_v
        bullish_confirmation = bool(
            _truthy(row.get("BULL_REJECTION", False))
            or (close > _finite(prev.get("High"), close) and close > _finite(row.get("Open"), close))
        )

        fund = fund_map.get(ticker, {})
        fund_flags = _safe_text(fund.get("fundamental_red_flags"))
        fund_score = _finite(fund.get("fundamental_score"), np.nan)
        fund_coverage = _finite(fund.get("fundamental_coverage"), 0.0)
        distressed = bool(
            ("Margin bersih negatif" in fund_flags and "OCF negatif" in fund_flags)
            or ("DER tinggi" in fund_flags and "OCF negatif" in fund_flags)
            or (fund_coverage >= 60 and np.isfinite(fund_score) and fund_score < 25)
        )
        severe_distress_break = distressed and close < ema200_v and not flow_positive

        sig = signal_map.get(ticker, {})
        scanner_status = _safe_text(sig.get("status"))
        scanner_setup_ready = scanner_status in {"EXECUTION_READY", "READY_FOR_PRICE_VERIFY", "PENDING_DATA"}
        internal_add_setup = bool(
            long_term_intact and near_support and flow_positive and bullish_confirmation
            and not confirmed_breakdown and not distressed
        )
        setup_ready = scanner_setup_ready or internal_add_setup
        add_confirmation = bullish_confirmation or scanner_setup_ready

        provisional_rows.append({
            **position.to_dict(),
            "last_price": close,
            "current_value_idr": market_value,
            "cost_value_idr": cost_value,
            "unrealized_pnl_idr": pnl,
            "unrealized_pnl_pct": pnl_pct,
            "ema20": ema20_v, "ema50": ema50_v, "ema200": ema200_v,
            "rsi14": _finite(row.get("RSI14"), np.nan),
            "cmf20": _finite(row.get("CMF20"), np.nan),
            "structural_support": structural_support,
            "existing_stop_loss": existing_stop,
            "structural_stop_loss": structural_stop,
            "suggested_stop_loss": suggested_stop,
            "suggested_tp1": suggested_tp1,
            "suggested_tp2": suggested_tp2,
            "tp1_basis": tp1_basis,
            "tp2_basis": tp2_basis,
            "trend_up": trend_up,
            "trend_weak": trend_weak,
            "long_term_structure_intact": long_term_intact,
            "flow_positive": flow_positive,
            "near_support": near_support,
            "bullish_confirmation": bullish_confirmation,
            "stop_breached": stop_breached,
            "confirmed_structure_breakdown": confirmed_breakdown,
            "momentum_hot": momentum_hot,
            "fundamental_distress": distressed,
            "scanner_setup": sig.get("setup", ""),
            "scanner_status": scanner_status,
            "portfolio_add_setup": "INTERNAL_CONFIRMED" if internal_add_setup else "",
            "setup_ready": setup_ready,
            "add_confirmation": add_confirmation,
            "severe_distress_break": severe_distress_break,
        })

    inferred_equity = total_market + max(0.0, cash_on_hand_idr)
    if account_equity_idr is not None and float(account_equity_idr) > 0:
        estimated_equity = max(float(account_equity_idr), total_market)
        equity_source = "ACCOUNT_EQUITY_INPUT"
    else:
        estimated_equity = max(inferred_equity, total_market, 1.0)
        equity_source = "POSITIONS_PLUS_CASH"
    estimated_equity = max(estimated_equity, 1.0)

    rows: list[dict[str, Any]] = []
    open_risk_total = 0.0
    for item in provisional_rows:
        if not np.isfinite(_finite(item.get("last_price"), np.nan)):
            rows.append(item)
            continue

        value = _finite(item.get("current_value_idr"), 0)
        weight = value / estimated_equity
        close = _finite(item.get("last_price"), 0)
        stop = _finite(item.get("suggested_stop_loss"), 0)
        shares = int(item.get("shares", 0))
        open_risk = max(0.0, close - stop) * shares + close * shares * cfg.sell_fee_pct
        open_risk_total += open_risk
        pnl_pct = _finite(item.get("unrealized_pnl_pct"), 0)

        action = "HOLD"
        reason = "Struktur utama dan money flow belum memberikan sinyal keluar"
        if _truthy(item.get("stop_breached", False)):
            action = "CUT_LOSS"
            reason = "Harga telah menyentuh/menembus stop aktif atau structural stop"
        elif _truthy(item.get("confirmed_structure_breakdown", False)):
            action = "CUT_LOSS"
            reason = "Dua penutupan mengonfirmasi breakdown support dengan trend dan flow bearish"
        elif _truthy(item.get("severe_distress_break", False)):
            action = "CUT_LOSS"
            reason = "Fundamental distress berat disertai struktur harga jangka panjang yang rusak"
        elif weight > cfg.max_position_pct:
            action = "REDUCE"
            reason = f"Bobot posisi {weight:.1%} melebihi batas {cfg.max_position_pct:.0%}"
        elif pnl_pct >= 0.15 and _truthy(item.get("momentum_hot", False)):
            action = "TAKE_PROFIT_PARTIAL"
            reason = "Profit signifikan dan harga overextended; realisasikan sebagian"
        elif pnl_pct >= 0.08 and not _truthy(item.get("flow_positive", False)):
            action = "REDUCE"
            reason = "Posisi masih profit tetapi money flow melemah"
        elif pnl_pct < 0:
            avg_allowed = bool(
                pnl_pct >= -cfg.max_avg_down_loss_pct
                and weight <= cfg.max_avg_down_position_pct
                and _truthy(item.get("long_term_structure_intact", False))
                and _truthy(item.get("near_support", False))
                and _truthy(item.get("flow_positive", False))
                and _truthy(item.get("add_confirmation", False))
                and not _truthy(item.get("fundamental_distress", False))
                and _truthy(item.get("setup_ready", False))
                and cash_on_hand_idr > 0
            )
            if avg_allowed:
                action = "AVG_DOWN_ALLOWED"
                reason = "Loss terbatas, support bertahan, flow dan candle konfirmasi positif, serta bobot masih aman"
            elif not _truthy(item.get("long_term_structure_intact", False)) or _truthy(item.get("fundamental_distress", False)):
                action = "DO_NOT_AVG_DOWN"
                reason = "Struktur jangka panjang belum sehat atau fundamental distress"
            elif _truthy(item.get("trend_weak", False)):
                action = "HOLD_TIGHT_STOP"
                reason = "Belum breakdown, tetapi trend/flow melemah; jangan tambah dan gunakan structural stop"
            else:
                action = "HOLD_NO_AVG"
                reason = "Belum ada konfirmasi lengkap untuk menambah posisi"
        elif _truthy(item.get("trend_weak", False)):
            action = "HOLD_TIGHT_STOP"
            reason = "Belum breakdown, tetapi trend menengah melemah"

        avg_lots = 0
        new_average = np.nan
        if action == "AVG_DOWN_ALLOWED":
            risk_budget = max(0.0, estimated_equity * cfg.risk_per_trade_pct - open_risk)
            per_share_risk = max(idx_tick_size(close), close - stop) + close * (cfg.buy_fee_pct + cfg.sell_fee_pct)
            lots_by_risk = int(risk_budget // (per_share_risk * 100)) if per_share_risk > 0 else 0
            lots_by_cash = int(cash_on_hand_idr // (close * 100 * (1 + cfg.buy_fee_pct)))
            max_extra_value = max(0.0, cfg.max_position_pct * estimated_equity - value)
            lots_by_weight = int(max_extra_value // (close * 100 * (1 + cfg.buy_fee_pct)))
            avg_lots = max(0, min(lots_by_risk, lots_by_cash, lots_by_weight))
            if avg_lots < 1:
                action = "HOLD_NO_AVG"
                reason = "Setup mendukung, tetapi cash/risk budget tidak cukup untuk 1 lot"
            else:
                new_shares = shares + avg_lots * 100
                new_average = (item["cost_value_idr"] + avg_lots * 100 * close * (1 + cfg.buy_fee_pct)) / new_shares

        item.update({
            "position_weight": weight,
            "position_weight_pct": weight * 100.0,
            "open_risk_idr": open_risk,
            "open_risk_pct_equity": open_risk / estimated_equity,
            "open_risk_pct_equity_pct": open_risk / estimated_equity * 100.0,
            "position_action": action,
            "action_reason": reason,
            "avg_down_lots": avg_lots,
            "avg_down_price": close if avg_lots >= 1 else np.nan,
            "new_average_after_avg": new_average,
            "equity_basis_idr": estimated_equity,
            "equity_source": equity_source,
        })
        rows.append(item)

    result = pd.DataFrame(rows)
    summary = {
        "positions": int(len(portfolio)),
        "market_value_idr": float(total_market),
        "cost_value_idr": float(total_cost),
        "unrealized_pnl_idr": float(total_market - total_cost),
        "unrealized_pnl_pct": float(total_market / total_cost - 1) if total_cost > 0 else 0.0,
        "open_risk_idr": float(open_risk_total),
        "open_risk_pct_equity": float(open_risk_total / estimated_equity),
        "cash_on_hand_idr": float(max(0.0, cash_on_hand_idr)),
        "estimated_equity_idr": float(estimated_equity),
        "inferred_equity_idr": float(inferred_equity),
        "equity_source": equity_source,
    }
    return result, summary


__all__ = list(dict.fromkeys(__all__ + [
    "fetch_resilient_market_status", "fetch_resilient_news_review", "fetch_resilient_fundamentals",
    "parse_portfolio_csv", "analyze_portfolio_positions",
]))

# v4.1 provider latency guard: fetch independent IDX pages concurrently.
def _fetch_official_idx_pages(timeout: int = 8) -> tuple[dict[str, str], dict[str, str]]:
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; IDXSuperScanner/4.1; research-client)",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.7",
    }

    def one(item: tuple[str, str]) -> tuple[str, str | None, str | None]:
        key, url = item
        try:
            response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            if not _is_exact_official_idx_url(response.url):
                raise RuntimeError("redirect keluar domain resmi IDX")
            text = response.text or ""
            if len(text) < 1_000:
                raise RuntimeError("respons terlalu pendek")
            return key, text, None
        except Exception as exc:
            return key, None, f"{type(exc).__name__}: {str(exc)[:120]}"

    pages: dict[str, str] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(IDX_AUTOMATION_URLS)) as pool:
        for key, text, error in pool.map(one, IDX_AUTOMATION_URLS.items()):
            if text is not None:
                pages[key] = text
            if error is not None:
                errors[key] = error
    return pages, errors


def _fetch_idx_disclosure_page(timeout: int = 8) -> tuple[str, bool, str]:
    import requests

    url = "https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi"
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; IDXSuperScanner/4.1)"},
            allow_redirects=True,
        )
        if response.status_code != 200 or not _is_exact_official_idx_url(response.url):
            raise RuntimeError(f"HTTP/redirect {response.status_code}")
        text = _html_text(response.text or "")
        ok = len(text) > 1_000 and ("KETERBUKAAN INFORMASI" in text or "DISCLOSURE" in text)
        return text, ok, "" if ok else "semantic marker tidak ditemukan"
    except Exception as exc:
        return "", False, f"{type(exc).__name__}: {str(exc)[:120]}"

# =============================================================================
# v4.2 specialty scanners: per-setup dashboard, Sniper Entry, BSJP/BPJS,
# Multibagger Radar, and ARA Hunter.
# =============================================================================

__version__ = "4.2.1-cache-timezone-hotfix"


def download_intraday_ohlcv(
    tickers: Iterable[str],
    period: str = "5d",
    interval: str = "5m",
    batch_size: int = 30,
) -> tuple[dict[str, pd.DataFrame], DownloadReport]:
    """Download bounded intraday OHLCV for specialty screens.

    BPJS uses 5-minute bars so the opening-range confirmation can become
    measurable shortly after 09:15 WIB. A per-ticker 15-minute fallback is
    retained when Yahoo does not return 5-minute data. Provider failure returns
    an audit report and never fabricates an intraday signal from daily data.
    """
    import yfinance as yf

    requested = list(dict.fromkeys(tickers))
    histories: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}
    warnings: dict[str, str] = {}
    fallback_interval = "15m" if interval == "5m" else None

    def mark_interval(frame: pd.DataFrame, value: str) -> pd.DataFrame:
        if frame is not None and not frame.empty:
            frame.attrs["source_interval"] = value
            try:
                frame.attrs["interval_minutes"] = float(value[:-1]) if value.endswith("m") else np.nan
            except Exception:
                frame.attrs["interval_minutes"] = np.nan
        return frame

    for start_idx in range(0, len(requested), max(1, int(batch_size))):
        batch = requested[start_idx : start_idx + max(1, int(batch_size))]
        try:
            raw = yf.download(
                batch,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                repair=False,
                actions=False,
                threads=True,
                progress=False,
                timeout=20,
                prepost=False,
            )
            for ticker in batch:
                frame = _extract_batch(raw, ticker, len(batch))
                frame = mark_interval(_clean_ohlcv(frame, strict=True), interval)
                if not frame.empty:
                    histories[ticker] = frame
                else:
                    failed[ticker] = f"Intraday batch {interval} kosong"
        except Exception as exc:
            for ticker in batch:
                failed[ticker] = f"Intraday batch {interval} gagal: {type(exc).__name__}"

    # Individual retry prevents one malformed batch response from discarding
    # the complete shortlist. When 5m is unavailable, 15m remains a valid but
    # slower fallback and is explicitly recorded in the audit report.
    missing = [ticker for ticker in requested if ticker not in histories]

    def retry_one(ticker: str) -> tuple[str, pd.DataFrame, str | None, str | None]:
        primary_error: str | None = None
        try:
            frame = yf.Ticker(ticker).history(
                period=period,
                interval=interval,
                auto_adjust=True,
                repair=False,
                actions=False,
                timeout=15,
                prepost=False,
            )
            clean = mark_interval(_clean_ohlcv(frame, strict=True), interval)
            if not clean.empty:
                return ticker, clean, None, None
            primary_error = f"Intraday individual {interval} kosong"
        except Exception as exc:
            primary_error = f"{type(exc).__name__}: {str(exc)[:100]}"

        if fallback_interval:
            try:
                frame = yf.Ticker(ticker).history(
                    period=period,
                    interval=fallback_interval,
                    auto_adjust=True,
                    repair=False,
                    actions=False,
                    timeout=15,
                    prepost=False,
                )
                clean = mark_interval(_clean_ohlcv(frame, strict=True), fallback_interval)
                if not clean.empty:
                    warning = f"{interval} tidak tersedia; memakai fallback {fallback_interval}"
                    return ticker, clean, None, warning
            except Exception as exc:
                fallback_error = f"{type(exc).__name__}: {str(exc)[:100]}"
                primary_error = f"{primary_error}; fallback {fallback_interval}: {fallback_error}"

        return ticker, pd.DataFrame(), primary_error or "Intraday tidak tersedia", None

    if missing:
        with ThreadPoolExecutor(max_workers=min(3, len(missing))) as pool:
            futures = [pool.submit(retry_one, ticker) for ticker in missing]
            for future in as_completed(futures):
                ticker, frame, error, warning = future.result()
                if not frame.empty:
                    histories[ticker] = frame
                    failed.pop(ticker, None)
                    if warning:
                        warnings[ticker] = warning
                else:
                    failed[ticker] = error or "Intraday tidak tersedia"

    report = DownloadReport(
        requested=requested,
        downloaded=sorted(histories),
        failed=failed,
        provider=f"Yahoo Finance via yfinance (intraday {interval})",
        adjusted_prices=True,
        downloaded_at=pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
        warnings=warnings,
    )
    return histories, report

def specialty_intraday_shortlist(
    prepared: Mapping[str, pd.DataFrame],
    core_signals: pd.DataFrame | None = None,
    max_candidates: int = 70,
) -> list[str]:
    """Rank a bounded liquid/momentum shortlist before intraday download."""
    rows: list[dict[str, Any]] = []
    core_tickers: set[str] = set()
    if core_signals is not None and not core_signals.empty and "ticker" in core_signals:
        core_tickers = set(core_signals["ticker"].astype(str))
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 25:
            continue
        row = frame.iloc[-1]
        prev = frame.iloc[-2]
        close = _finite(row.get("Close"), 0.0)
        prev_close = _finite(prev.get("Close"), 0.0)
        adtv = _finite(row.get("ADTV20"), 0.0)
        vol_ratio = _finite(row.get("VOL_RATIO"), 0.0)
        close_location = _finite(row.get("CLOSE_LOCATION"), 0.0)
        cmf_v = _finite(row.get("CMF20"), -1.0)
        rs_v = _finite(row.get("REL_STRENGTH60"), -1.0)
        daily_return = close / prev_close - 1 if prev_close > 0 else 0.0
        breakout = close > _finite(row.get("HIGH20_PREV"), float("inf"))
        score = 0.0
        score += min(35.0, 35.0 * adtv / 5_000_000_000.0) if adtv > 0 else 0.0
        score += min(20.0, 10.0 * max(0.0, vol_ratio - 0.5))
        score += 10.0 if close_location >= 0.70 else 5.0 if close_location >= 0.55 else 0.0
        score += 10.0 if cmf_v > 0.05 else 5.0 if cmf_v > 0 else 0.0
        score += 8.0 if rs_v > 0 else 0.0
        score += 7.0 if breakout else 0.0
        score += 5.0 if 0.005 <= daily_return <= 0.08 else 0.0
        score += 5.0 if ticker in core_tickers else 0.0
        rows.append({"ticker": ticker, "shortlist_score": score, "adtv": adtv})
    if not rows:
        return []
    ranked = pd.DataFrame(rows).sort_values(
        ["shortlist_score", "adtv"], ascending=[False, False]
    )
    return ranked.head(max(1, int(max_candidates)))["ticker"].tolist()


def _jakarta_timestamp(value: Any | None = None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="Asia/Jakarta")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Jakarta")
    return ts.tz_convert("Asia/Jakarta")


def _intraday_session(frame: pd.DataFrame, asof: Any | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    attrs = dict(getattr(frame, "attrs", {}) or {})
    out = _clean_ohlcv(frame, strict=True)
    if out.empty:
        return out
    out.attrs.update(attrs)
    if asof is not None:
        interval_minutes = _finite(attrs.get("interval_minutes"), np.nan)
        if not np.isfinite(interval_minutes) or interval_minutes <= 0:
            try:
                diffs = pd.Series(pd.DatetimeIndex(out.index)[1:] - pd.DatetimeIndex(out.index)[:-1])
                diffs = diffs.dt.total_seconds().div(60.0)
                diffs = diffs[(diffs > 0) & (diffs <= 60)]
                interval_minutes = float(diffs.median()) if not diffs.empty else 5.0
            except Exception:
                interval_minutes = 5.0
        # Yahoo timestamps intraday bars at bar-open. Exclude the currently
        # forming bar so ORB/VWAP logic never consumes future/incomplete data.
        bar_delta = pd.to_timedelta(float(interval_minutes), unit="min")
        cutoff = _jakarta_timestamp(asof).tz_localize(None) - bar_delta
        out = out[pd.DatetimeIndex(out.index) <= cutoff].copy()
        out.attrs.update(attrs)
        if out.empty:
            return out
    last_date = pd.Timestamp(out.index[-1]).date()
    session = out[pd.Index(out.index).map(lambda x: pd.Timestamp(x).date() == last_date)].copy()
    session.attrs.update(attrs)
    return session


def _intraday_interval_minutes(frame: pd.DataFrame) -> float:
    attr_value = _finite(getattr(frame, "attrs", {}).get("interval_minutes"), np.nan)
    if np.isfinite(attr_value) and attr_value > 0:
        return float(attr_value)
    if frame is None or len(frame.index) < 2:
        return np.nan
    try:
        index = pd.DatetimeIndex(frame.index)
        diffs = pd.Series(index[1:] - index[:-1]).dt.total_seconds().div(60.0)
        diffs = diffs[(diffs > 0) & (diffs <= 60)]
        return float(diffs.median()) if not diffs.empty else np.nan
    except Exception:
        return np.nan


def _intraday_metrics(
    frame: pd.DataFrame,
    now: Any | None = None,
    max_stale_minutes: int = 20,
) -> dict[str, Any]:
    reference = _jakarta_timestamp(now)
    session = _intraday_session(frame, asof=reference)
    if session.empty:
        return {
            "intraday_bars": 0.0,
            "intraday_interval_minutes": np.nan,
            "opening_range_bars": 0.0,
            "post_orb_bars": 0.0,
            "intraday_data_state": "NO_DATA",
            "session_vwap": np.nan,
            "session_close_location": np.nan,
            "late_volume_acceleration": np.nan,
            "opening_volume_ratio": np.nan,
            "orb_high": np.nan,
            "orb_low": np.nan,
            "intraday_last": np.nan,
            "intraday_return": np.nan,
            "intraday_session_date": None,
            "intraday_last_bar_time": pd.NaT,
            "intraday_age_minutes": np.nan,
            "intraday_fresh": False,
        }

    interval_minutes = _intraday_interval_minutes(session)
    if not np.isfinite(interval_minutes) or interval_minutes <= 0:
        interval_minutes = 5.0

    # BPJS defines the opening range as the first 15 minutes. This gives three
    # bars on 5m data and one bar on the 15m fallback.
    opening_range_bars = max(1, int(np.ceil(15.0 / interval_minutes)))
    orb_n = min(opening_range_bars, len(session))
    post_orb_bars = max(0, len(session) - opening_range_bars)

    typical = (session["High"] + session["Low"] + session["Close"]) / 3.0
    total_volume = float(session["Volume"].sum())
    session_vwap = (
        float((typical * session["Volume"]).sum() / total_volume)
        if total_volume > 0 else np.nan
    )
    day_high = float(session["High"].max())
    day_low = float(session["Low"].min())
    last_close = float(session["Close"].iloc[-1])
    location = (last_close - day_low) / (day_high - day_low) if day_high > day_low else 0.5

    late_acceleration = np.nan
    if len(session) >= 4:
        late_n = min(max(2, int(np.ceil(20.0 / interval_minutes))), max(2, len(session) // 3))
        late_n = min(late_n, len(session) - 1)
        late_mean = float(session["Volume"].tail(late_n).mean())
        prior = session["Volume"].iloc[:-late_n]
        prior_mean = float(prior.mean()) if len(prior) else np.nan
        late_acceleration = late_mean / prior_mean if np.isfinite(prior_mean) and prior_mean > 0 else np.nan

    opening_ratio = np.nan
    opening_slice = session["Volume"].head(orb_n)
    remainder = session["Volume"].iloc[opening_range_bars:]
    if len(opening_slice) and len(remainder):
        remainder_mean = float(remainder.mean())
        opening_mean = float(opening_slice.mean())
        opening_ratio = opening_mean / remainder_mean if remainder_mean > 0 else np.nan

    orb_high = float(session["High"].head(orb_n).max())
    orb_low = float(session["Low"].head(orb_n).min())
    first_open = float(session["Open"].iloc[0])
    intraday_return = last_close / first_open - 1 if first_open > 0 else np.nan
    last_bar_time = pd.Timestamp(session.index[-1])
    completed_at = last_bar_time + pd.to_timedelta(float(interval_minutes), unit="min")
    reference_naive = reference.tz_localize(None)
    age_minutes = max(0.0, (reference_naive - completed_at).total_seconds() / 60.0)
    same_session_date = last_bar_time.date() == reference_naive.date()
    stale_limit = max(float(max_stale_minutes), 2.0 * interval_minutes + 5.0)
    fresh = bool(same_session_date and age_minutes <= stale_limit)
    if not fresh:
        data_state = "STALE_SESSION"
    elif post_orb_bars >= 1:
        data_state = "LIVE_READY"
    else:
        data_state = "OPENING_RANGE_FORMING"

    return {
        "intraday_bars": float(len(session)),
        "intraday_interval_minutes": float(interval_minutes),
        "opening_range_bars": float(opening_range_bars),
        "post_orb_bars": float(post_orb_bars),
        "intraday_data_state": data_state,
        "session_vwap": session_vwap,
        "session_close_location": location,
        "late_volume_acceleration": late_acceleration,
        "opening_volume_ratio": opening_ratio,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "intraday_last": last_close,
        "intraday_return": intraday_return,
        "intraday_session_date": last_bar_time.date().isoformat(),
        "intraday_last_bar_time": last_bar_time,
        "intraday_age_minutes": round(age_minutes, 1),
        "intraday_fresh": fresh,
    }

def _critical_context_map(signals: pd.DataFrame | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if signals is None or signals.empty or "ticker" not in signals:
        return result
    for ticker, group in signals.groupby("ticker"):
        messages: list[str] = []
        for _, row in group.iterrows():
            critical = _safe_text(row.get("critical_blockers"))
            if critical:
                messages.append(critical)
            source_tier = _safe_text(row.get("ohlcv_source_tier")).upper()
            if source_tier in {"CACHE_FALLBACK", "UNAVAILABLE"}:
                messages.append("OHLCV daily bukan hasil live")
            if (
                _truthy(row.get("independent_price_required", False))
                and not _truthy(row.get("independent_price_verified", False))
            ):
                messages.append("Harga independen belum terverifikasi")
            if _truthy(row.get("market_status_critical_blocker", False)):
                messages.append("Status perdagangan IDX negatif")
            if _truthy(row.get("news_critical_blocker", False)):
                messages.append("Berita material negatif")
            if _truthy(row.get("quote_critical_blocker", False)):
                messages.append("Konflik quote/OHLCV")
        result[str(ticker)] = " • ".join(dict.fromkeys(messages))
    return result


def _fundamental_records(fundamentals: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if fundamentals is None or fundamentals.empty or "ticker" not in fundamentals:
        return {}
    return {
        str(row["ticker"]): row.to_dict()
        for _, row in fundamentals.drop_duplicates("ticker", keep="last").iterrows()
    }


def _specialty_sizing(
    entry: float,
    stop: float,
    cfg: ScanConfig,
    position_cap: float,
    risk_cap: float,
) -> dict[str, Any]:
    """Conservative indicative sizing for manual high-risk specialty trades."""
    specialty_cfg = cfg.replace(
        max_position_pct=min(cfg.max_position_pct, position_cap),
        risk_per_trade_pct=min(cfg.risk_per_trade_pct, risk_cap),
    )
    sized = size_stockbit_order(entry, stop, specialty_cfg)
    return {
        "suggested_lots": int(_finite(sized.get("suggested_lots"), 0)),
        "capital_required_idr": _finite(sized.get("capital_required_idr"), 0),
        "max_loss_idr": _finite(sized.get("max_loss_idr"), 0),
        "specialty_position_cap_pct": 100.0 * position_cap,
        "specialty_risk_cap_pct": 100.0 * risk_cap,
    }


def scan_sniper_entries(
    core_signals: pd.DataFrame | None,
    prepared: Mapping[str, pd.DataFrame],
    config: ScanConfig | None = None,
) -> pd.DataFrame:
    """Build a dedicated strict ICT Sniper table from Unicorn candidates."""
    cfg = config or ScanConfig()
    columns = [
        "ticker", "sniper_status", "sniper_score", "core_status", "setup",
        "last_price", "entry_low", "entry_high", "sniper_entry", "sniper_stop",
        "sniper_tp1", "sniper_tp2", "rr1", "rr2", "distance_atr",
        "volume_ratio", "silent_accumulation_score", "valid_until",
        "blockers", "reason",
    ]
    if core_signals is None or core_signals.empty:
        return pd.DataFrame(columns=columns)
    candidates = core_signals[core_signals.get("setup", pd.Series(index=core_signals.index, dtype=object)).eq("UNICORN_SNIPER_ICT")].copy()
    rows: list[dict[str, Any]] = []
    for _, signal in candidates.iterrows():
        ticker = str(signal.get("ticker"))
        frame = prepared.get(ticker)
        if frame is None or frame.empty:
            continue
        row = frame.iloc[-1]
        quality = _finite(signal.get("quality_score"), 0.0)
        distance = _finite(signal.get("distance_atr"), 99.0)
        volume_ratio = _finite(signal.get("volume_ratio"), _finite(row.get("VOL_RATIO"), 0.0))
        accumulation = _finite(signal.get("silent_accumulation_score"), 0.0)
        rr2 = _finite(signal.get("rr2"), 0.0)
        stop_pct = _finite(signal.get("stop_pct"), 1.0)
        core_status = _safe_text(signal.get("status"))
        daily_source_tier = _safe_text(signal.get("ohlcv_source_tier")).upper()
        daily_source_ok = daily_source_tier not in {"CACHE_FALLBACK", "UNAVAILABLE"}
        score = 0.0
        score += min(35.0, 0.35 * quality)
        score += 15.0 if distance <= 0.15 else 10.0 if distance <= 0.35 else 4.0 if distance <= 0.75 else 0.0
        score += 15.0 if volume_ratio >= 1.50 else 10.0 if volume_ratio >= 1.20 else 3.0 if volume_ratio >= 0.90 else 0.0
        score += 15.0 if accumulation >= 80 else 10.0 if accumulation >= 70 else 3.0 if accumulation >= 60 else 0.0
        score += 10.0 if rr2 >= 3.5 else 7.0 if rr2 >= 3.0 else 3.0 if rr2 >= cfg.min_rr2 else 0.0
        score += 10.0 if 0 < stop_pct <= 0.05 else 5.0 if stop_pct <= cfg.max_stop_pct else 0.0
        strict_ready = bool(
            core_status == "EXECUTION_READY"
            and score >= 82
            and distance <= 0.25
            and volume_ratio >= 1.20
            and accumulation >= 70
            and rr2 >= 3.0
            and stop_pct <= 0.055
            and daily_source_ok
        )
        if strict_ready:
            status = "SNIPER_READY"
        elif score >= 70 and distance <= 1.0:
            status = "WAIT_SNIPER_RETRACE"
        else:
            status = "SNIPER_REJECT"
        rows.append({
            "ticker": ticker,
            "sniper_status": status,
            "sniper_score": round(min(100.0, score), 1),
            "core_status": core_status,
            "setup": "STRICT_ICT_SNIPER",
            "last_price": _finite(signal.get("last_price"), _finite(row.get("Close"), np.nan)),
            "entry_low": signal.get("entry_low"),
            "entry_high": signal.get("entry_high"),
            "sniper_entry": signal.get("entry"),
            "sniper_stop": signal.get("stop_loss"),
            "sniper_tp1": signal.get("tp1"),
            "sniper_tp2": signal.get("tp2"),
            "rr1": signal.get("rr1"),
            "rr2": signal.get("rr2"),
            "distance_atr": distance,
            "volume_ratio": volume_ratio,
            "silent_accumulation_score": accumulation,
            "valid_until": signal.get("valid_until"),
            "blockers": " • ".join(filter(None, [
                _safe_text(signal.get("blockers")),
                "OHLCV daily bukan hasil live" if not daily_source_ok else "",
            ])),
            "reason": signal.get("reason", ""),
        })
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        rank = {"SNIPER_READY": 0, "WAIT_SNIPER_RETRACE": 1, "SNIPER_REJECT": 2}
        result["_rank"] = result["sniper_status"].map(rank).fillna(9)
        result = result.sort_values(["_rank", "sniper_score", "rr2"], ascending=[True, False, False]).drop(columns="_rank").reset_index(drop=True)
    return result


def scan_bsjp_candidates(
    prepared: Mapping[str, pd.DataFrame],
    intraday: Mapping[str, pd.DataFrame] | None = None,
    core_signals: pd.DataFrame | None = None,
    config: ScanConfig | None = None,
    now: Any | None = None,
    market_context: MarketContext | None = None,
) -> pd.DataFrame:
    """Strict Beli Sore Jual Pagi screen.

    Intraday evidence is mandatory for a READY label. Daily-only candidates are
    retained as watchlist rows rather than being promoted into an overnight order.
    """
    cfg = config or ScanConfig()
    intraday = intraday or {}
    critical_map = _critical_context_map(core_signals)
    now_jkt = _jakarta_timestamp(now)
    market_regime = market_context.regime if market_context is not None else "NOT_EVALUATED"
    regime_blocked = market_context is not None and market_regime in {"RISK_OFF", "UNKNOWN"}
    minute = now_jkt.hour * 60 + now_jkt.minute
    in_window = now_jkt.weekday() < 5 and (14 * 60 + 30) <= minute <= (15 * 60 + 49)
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 60:
            continue
        row = frame.iloc[-1]
        prev = frame.iloc[-2]
        close = _finite(row.get("Close"), 0.0)
        prev_close = _finite(prev.get("Close"), 0.0)
        if close <= 0 or prev_close <= 0:
            continue
        daily_return = close / prev_close - 1.0
        adtv = _finite(row.get("ADTV20"), 0.0)
        vol_ratio = _finite(row.get("VOL_RATIO"), 0.0)
        cmf_v = _finite(row.get("CMF20"), -1.0)
        obv_up = _finite(row.get("OBV_SLOPE10"), -1.0) > 0
        close_location = _finite(row.get("CLOSE_LOCATION"), 0.0)
        rs_v = _finite(row.get("REL_STRENGTH60"), -1.0)
        rsi_v = _finite(row.get("RSI14"), 50.0)
        atr_v = _finite(row.get("ATR14"), 0.0)
        ara_price = idx_daily_price_band(prev_close)[1]
        room_to_ara = ara_price / close - 1.0 if ara_price and close > 0 else np.nan
        already_near_ara = near_upper_auto_rejection(prev_close, close, _finite(row.get("High"), close))
        im = _intraday_metrics(
            intraday.get(ticker, pd.DataFrame()), now_jkt, cfg.max_intraday_stale_minutes
        )
        has_any_intraday = im["intraday_bars"] > 0
        has_intraday = im["intraday_bars"] >= 8 and _truthy(im["intraday_fresh"])
        above_session_vwap = has_intraday and im["intraday_last"] >= im["session_vwap"]
        late_volume = has_intraday and _finite(im["late_volume_acceleration"], 0.0) >= 1.25
        session_location_ok = has_intraday and _finite(im["session_close_location"], 0.0) >= 0.72
        daily_gate = bool(
            adtv >= max(cfg.min_adtv_idr, 3_000_000_000.0)
            and vol_ratio >= 1.10
            and cmf_v >= 0.03
            and obv_up
            and close_location >= 0.65
            and rs_v > 0
            and 48 <= rsi_v <= 76
            and -0.015 <= daily_return <= 0.08
            and not already_near_ara
            and (not np.isfinite(room_to_ara) or room_to_ara >= 0.04)
        )
        score = 0.0
        score += 15.0 if adtv >= 10_000_000_000 else 10.0 if adtv >= 5_000_000_000 else 5.0 if adtv >= 3_000_000_000 else 0.0
        score += 15.0 if vol_ratio >= 1.80 else 10.0 if vol_ratio >= 1.30 else 5.0 if vol_ratio >= 1.10 else 0.0
        score += 15.0 if cmf_v >= 0.10 and obv_up else 10.0 if cmf_v >= 0.03 and obv_up else 0.0
        score += 10.0 if close_location >= 0.80 else 6.0 if close_location >= 0.65 else 0.0
        score += 8.0 if rs_v > 0.05 else 5.0 if rs_v > 0 else 0.0
        score += 7.0 if 0.005 <= daily_return <= 0.05 else 3.0 if -0.015 <= daily_return <= 0.08 else 0.0
        score += 10.0 if above_session_vwap else 0.0
        score += 10.0 if late_volume else 0.0
        score += 10.0 if session_location_ok else 0.0
        context_messages = [critical_map.get(ticker, "")]
        if regime_blocked:
            context_messages.append(f"Regime IHSG {market_regime} tidak mengizinkan overnight entry")
        context_blocker = " • ".join(message for message in context_messages if message)
        if context_blocker:
            score = min(score, 55.0)
        score = min(100.0, score)
        # Keep only plausible watchlist rows to avoid flooding the specialty tab.
        if score < 55 and not daily_gate:
            continue
        entry = round_idx_price(im["intraday_last"] if has_intraday else close, "down")
        session = _intraday_session(intraday.get(ticker, pd.DataFrame()), asof=now_jkt)
        if has_intraday and not session.empty:
            recent_low = float(session["Low"].tail(min(4, len(session))).min())
            raw_stop = max(recent_low - idx_tick_size(recent_low), entry - 1.2 * max(atr_v / 8.0, idx_tick_size(entry)))
        else:
            raw_stop = entry - min(0.03 * entry, max(1.0 * atr_v, 3 * idx_tick_size(entry)))
        stop = round_idx_price(raw_stop, "down")
        if stop >= entry:
            stop = round_idx_price(entry - max(3 * idx_tick_size(entry), 0.02 * entry), "down")
        risk = max(entry - stop, idx_tick_size(entry))
        tp1 = round_idx_price(entry + max(1.5 * risk, 0.018 * entry), "up")
        tp2 = round_idx_price(entry + max(2.2 * risk, 0.032 * entry), "up")
        sizing = _specialty_sizing(entry, stop, cfg, position_cap=0.10, risk_cap=0.0035)
        ready = bool(
            in_window and daily_gate and has_intraday and above_session_vwap
            and late_volume and session_location_ok and score >= 82
            and not context_blocker and 0 < (entry - stop) / entry <= 0.035
        )
        if ready:
            status = "BSJP_READY"
            action = "BUY_SORE_SELL_NEXT_MORNING"
        elif not has_any_intraday:
            status = "BSJP_DATA_UNAVAILABLE"
            action = "RETRY_INTRADAY_5M"
        elif not _truthy(im["intraday_fresh"]):
            status = "BSJP_STALE_INTRADAY"
            action = "REFRESH_INTRADAY_DATA"
        elif not has_intraday:
            status = "BSJP_WAIT_SESSION_BARS"
            action = "RUN_AGAIN_AFTER_MORE_5M_BARS"
        elif not in_window:
            status = "BSJP_WATCHLIST"
            action = "RUN_AGAIN_14_30_15_49_WIB"
        else:
            status = "BSJP_WATCHLIST"
            action = "WAIT_LATE_SESSION_CONFIRMATION"
        rows.append({
            "ticker": ticker,
            "bsjp_status": status,
            "bsjp_score": round(score, 1),
            "action": action,
            "last_price": close,
            "entry": entry,
            "stop_loss": stop,
            "morning_tp1": tp1,
            "morning_tp2": tp2,
            "risk_pct": (entry - stop) / entry if entry > 0 else np.nan,
            "daily_return_pct": daily_return,
            "volume_ratio": vol_ratio,
            "adtv20_idr": adtv,
            "cmf20": cmf_v,
            "relative_strength60": rs_v,
            "session_vwap": im["session_vwap"],
            "session_close_location": im["session_close_location"],
            "late_volume_acceleration": im["late_volume_acceleration"],
            "room_to_ara_pct": room_to_ara,
            "intraday_bars": int(im["intraday_bars"]),
            "intraday_interval_minutes": im["intraday_interval_minutes"],
            "intraday_data_state": im["intraday_data_state"],
            "intraday_session_date": im["intraday_session_date"],
            "intraday_age_minutes": im["intraday_age_minutes"],
            "intraday_fresh": im["intraday_fresh"],
            "market_regime": market_regime,
            "execution_window": "14:30–15:49 WIB",
            "exit_window": "09:00–10:00 WIB next session",
            "order_instruction": "MANUAL_REVIEW_ONLY",
            "risk_class": "HIGH_OVERNIGHT_GAP_RISK",
            "blockers": context_blocker,
            **sizing,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {
            "BSJP_READY": 0,
            "BSJP_WATCHLIST": 1,
            "BSJP_WAIT_SESSION_BARS": 2,
            "BSJP_STALE_INTRADAY": 3,
            "BSJP_DATA_UNAVAILABLE": 4,
        }
        result["_rank"] = result["bsjp_status"].map(rank).fillna(9)
        result = result.sort_values(["_rank", "bsjp_score", "adtv20_idr"], ascending=[True, False, False]).drop(columns="_rank").reset_index(drop=True)
    return result


def scan_bpjs_candidates(
    prepared: Mapping[str, pd.DataFrame],
    intraday: Mapping[str, pd.DataFrame] | None = None,
    core_signals: pd.DataFrame | None = None,
    config: ScanConfig | None = None,
    now: Any | None = None,
    market_context: MarketContext | None = None,
) -> pd.DataFrame:
    """Strict Beli Pagi Jual Sore opening-range screen."""
    cfg = config or ScanConfig()
    intraday = intraday or {}
    critical_map = _critical_context_map(core_signals)
    now_jkt = _jakarta_timestamp(now)
    market_regime = market_context.regime if market_context is not None else "NOT_EVALUATED"
    regime_blocked = market_context is not None and market_regime in {"RISK_OFF", "UNKNOWN"}
    minute = now_jkt.hour * 60 + now_jkt.minute
    # The first post-ORB 5m candle closes at 09:20 WIB; READY before that
    # would use an incomplete confirmation bar.
    in_window = now_jkt.weekday() < 5 and (9 * 60 + 20) <= minute <= (10 * 60 + 45)
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 60:
            continue
        row = frame.iloc[-1]
        prev = frame.iloc[-2]
        close = _finite(row.get("Close"), 0.0)
        prev_close = _finite(prev.get("Close"), 0.0)
        adtv = _finite(row.get("ADTV20"), 0.0)
        rs_v = _finite(row.get("REL_STRENGTH60"), -1.0)
        cmf_v = _finite(row.get("CMF20"), -1.0)
        trend = _finite(row.get("EMA20"), 0.0) >= _finite(row.get("EMA50"), float("inf")) and close >= _finite(row.get("EMA50"), float("inf"))
        im = _intraday_metrics(
            intraday.get(ticker, pd.DataFrame()), now_jkt, cfg.max_intraday_stale_minutes
        )
        has_any_intraday = im["intraday_bars"] > 0
        has_intraday = (
            im["intraday_data_state"] == "LIVE_READY"
            and im["post_orb_bars"] >= 1
        )
        session = _intraday_session(intraday.get(ticker, pd.DataFrame()), asof=now_jkt)
        session_open = float(session["Open"].iloc[0]) if has_intraday and not session.empty else np.nan
        intraday_last = _finite(im["intraday_last"], close)
        gap = session_open / prev_close - 1 if np.isfinite(session_open) and prev_close > 0 else np.nan
        above_vwap = has_intraday and intraday_last >= _finite(im["session_vwap"], float("inf"))
        orb_break = has_intraday and intraday_last > _finite(im["orb_high"], float("inf"))
        opening_volume = has_intraday and _finite(im["opening_volume_ratio"], 0.0) >= 1.30
        location_ok = has_intraday and _finite(im["session_close_location"], 0.0) >= 0.70
        gap_ok = has_intraday and np.isfinite(gap) and -0.015 <= gap <= 0.035
        context_messages = [critical_map.get(ticker, "")]
        if regime_blocked:
            context_messages.append(f"Regime IHSG {market_regime} tidak mengizinkan momentum entry")
        context_blocker = " • ".join(message for message in context_messages if message)
        score = 0.0
        score += 15.0 if adtv >= 10_000_000_000 else 10.0 if adtv >= 5_000_000_000 else 5.0 if adtv >= 3_000_000_000 else 0.0
        score += 15.0 if trend else 0.0
        score += 10.0 if rs_v > 0 else 0.0
        score += 10.0 if cmf_v >= 0.05 else 5.0 if cmf_v > 0 else 0.0
        score += 15.0 if above_vwap else 0.0
        score += 15.0 if orb_break else 0.0
        score += 10.0 if opening_volume else 0.0
        score += 5.0 if location_ok else 0.0
        score += 5.0 if gap_ok else 0.0
        if context_blocker:
            score = min(score, 55.0)
        score = min(100.0, score)
        if score < 55 and not (trend and adtv >= 3_000_000_000):
            continue
        entry = round_idx_price(max(intraday_last, _finite(im["orb_high"], intraday_last) + idx_tick_size(intraday_last)), "up")
        if has_intraday:
            recent_low = float(session["Low"].tail(min(3, len(session))).min())
            raw_stop = max(_finite(im["orb_low"], recent_low), recent_low) - idx_tick_size(recent_low)
        else:
            raw_stop = entry * 0.97
        stop = round_idx_price(raw_stop, "down")
        if stop >= entry:
            stop = round_idx_price(entry - max(3 * idx_tick_size(entry), 0.02 * entry), "down")
        risk = max(entry - stop, idx_tick_size(entry))
        tp1 = round_idx_price(entry + max(1.6 * risk, 0.018 * entry), "up")
        tp2 = round_idx_price(entry + max(2.3 * risk, 0.032 * entry), "up")
        sizing = _specialty_sizing(entry, stop, cfg, position_cap=0.15, risk_cap=0.0040)
        ara_price = idx_daily_price_band(prev_close)[1]
        room_to_ara = ara_price / entry - 1.0 if ara_price and entry > 0 else np.nan
        ready = bool(
            in_window and has_intraday and trend and adtv >= max(cfg.min_adtv_idr, 3_000_000_000.0)
            and above_vwap and orb_break and opening_volume and location_ok and gap_ok
            and score >= 84 and not context_blocker
            and 0 < (entry - stop) / entry <= 0.035
            and (not np.isfinite(room_to_ara) or room_to_ara >= 0.04)
        )
        if ready:
            status = "BPJS_READY"
            action = "BUY_OPENING_BREAKOUT_SELL_BEFORE_CLOSE"
        elif not has_any_intraday:
            status = "BPJS_DATA_UNAVAILABLE"
            action = "RETRY_INTRADAY_5M"
        elif not _truthy(im["intraday_fresh"]):
            status = "BPJS_STALE_INTRADAY"
            action = "REFRESH_INTRADAY_DATA"
        elif not has_intraday:
            status = "BPJS_WAIT_OPENING_BARS"
            action = "RUN_AGAIN_AFTER_OPENING_RANGE"
        elif not in_window:
            status = "BPJS_WATCHLIST"
            action = "RUN_AGAIN_09_15_10_45_WIB"
        else:
            status = "BPJS_WATCHLIST"
            action = "WAIT_ORB_VWAP_CONFIRMATION"
        rows.append({
            "ticker": ticker,
            "bpjs_status": status,
            "bpjs_score": round(score, 1),
            "action": action,
            "last_price": close,
            "intraday_last": intraday_last,
            "opening_gap_pct": gap,
            "entry": entry,
            "stop_loss": stop,
            "day_tp1": tp1,
            "day_tp2": tp2,
            "risk_pct": (entry - stop) / entry if entry > 0 else np.nan,
            "session_vwap": im["session_vwap"],
            "orb_high": im["orb_high"],
            "orb_low": im["orb_low"],
            "opening_volume_ratio": im["opening_volume_ratio"],
            "session_close_location": im["session_close_location"],
            "adtv20_idr": adtv,
            "relative_strength60": rs_v,
            "room_to_ara_pct": room_to_ara,
            "intraday_bars": int(im["intraday_bars"]),
            "intraday_interval_minutes": im["intraday_interval_minutes"],
            "opening_range_bars": int(im["opening_range_bars"]),
            "post_orb_bars": int(im["post_orb_bars"]),
            "intraday_data_state": im["intraday_data_state"],
            "intraday_session_date": im["intraday_session_date"],
            "intraday_age_minutes": im["intraday_age_minutes"],
            "intraday_fresh": im["intraday_fresh"],
            "market_regime": market_regime,
            "execution_window": "09:20–10:45 WIB",
            "mandatory_exit": "Before regular-market close",
            "order_instruction": "MANUAL_REVIEW_ONLY",
            "risk_class": "HIGH_INTRADAY_EXECUTION_RISK",
            "blockers": context_blocker,
            **sizing,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {
            "BPJS_READY": 0,
            "BPJS_WATCHLIST": 1,
            "BPJS_WAIT_OPENING_BARS": 2,
            "BPJS_STALE_INTRADAY": 3,
            "BPJS_DATA_UNAVAILABLE": 4,
        }
        result["_rank"] = result["bpjs_status"].map(rank).fillna(9)
        result = result.sort_values(["_rank", "bpjs_score", "adtv20_idr"], ascending=[True, False, False]).drop(columns="_rank").reset_index(drop=True)
    return result


def scan_multibagger_candidates(
    prepared: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame | None,
    core_signals: pd.DataFrame | None = None,
    config: ScanConfig | None = None,
) -> pd.DataFrame:
    """Rank long-horizon growth/quality candidates; not a return guarantee."""
    cfg = config or ScanConfig()
    f_map = _fundamental_records(fundamentals)
    signal_map: dict[str, dict[str, Any]] = {}
    if core_signals is not None and not core_signals.empty and "ticker" in core_signals:
        ranked = core_signals.copy()
        ranked["_q"] = pd.to_numeric(ranked.get("composite_score", ranked.get("quality_score")), errors="coerce").fillna(0)
        for ticker, group in ranked.sort_values("_q", ascending=False).groupby("ticker", sort=False):
            signal_map[str(ticker)] = group.iloc[0].to_dict()
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 220:
            continue
        fund = f_map.get(ticker, {})
        coverage = _finite(fund.get("fundamental_coverage"), 0.0)
        if coverage <= 0:
            continue
        row = frame.iloc[-1]
        close = _finite(row.get("Close"), 0.0)
        adtv = _finite(row.get("ADTV20"), 0.0)
        revenue_growth = _finite(fund.get("revenue_growth"), np.nan)
        earnings_growth = _finite(fund.get("earnings_growth"), np.nan)
        roe = _finite(fund.get("roe"), np.nan)
        roa = _finite(fund.get("roa"), np.nan)
        net_margin = _finite(fund.get("net_margin"), np.nan)
        operating_margin = _finite(fund.get("operating_margin"), np.nan)
        debt_equity = _finite(fund.get("debt_equity"), np.nan)
        current_ratio = _finite(fund.get("current_ratio"), np.nan)
        cash_to_debt = _finite(fund.get("cash_to_debt"), np.nan)
        ocf = _finite(fund.get("operating_cash_flow"), np.nan)
        fcf = _finite(fund.get("free_cash_flow"), np.nan)
        peg = _finite(fund.get("peg_ratio"), np.nan)
        fcf_yield = _finite(fund.get("fcf_yield"), np.nan)
        market_cap = _finite(fund.get("market_cap"), np.nan)
        fundamental_model = _safe_text(fund.get("fundamental_model")) or "GENERAL"
        is_financial = fundamental_model == "FINANCIAL"
        red_flags = _safe_text(fund.get("fundamental_red_flags"))
        roc60 = _finite(row.get("ROC60"), -1.0)
        roc120 = _finite(row.get("ROC120"), -1.0)
        rs60 = _finite(row.get("REL_STRENGTH60"), -1.0)
        dist_high = _finite(row.get("DIST_52W_HIGH"), -1.0)
        cmf_v = _finite(row.get("CMF20"), -1.0)
        obv_up = _finite(row.get("OBV_SLOPE10"), -1.0) > 0
        accumulation, up_down = _silent_accumulation_metrics(frame)
        growth_score = 0.0
        growth_score += 15.0 if revenue_growth >= 0.20 else 10.0 if revenue_growth >= 0.10 else 4.0 if revenue_growth >= 0 else 0.0
        growth_score += 15.0 if earnings_growth >= 0.25 else 10.0 if earnings_growth >= 0.12 else 4.0 if earnings_growth >= 0 else 0.0
        profitability_score = 0.0
        profitability_score += 8.0 if roe >= 0.20 else 5.0 if roe >= 0.12 else 2.0 if roe >= 0.08 else 0.0
        profitability_score += 4.0 if roa >= 0.08 else 2.0 if roa >= 0.04 else 0.0
        profitability_score += 4.0 if operating_margin >= 0.15 else 2.0 if operating_margin >= 0.08 else 0.0
        profitability_score += 4.0 if net_margin >= 0.12 else 2.0 if net_margin >= 0.06 else 0.0
        balance_score = 0.0
        if not is_financial:
            balance_score += 4.0 if np.isfinite(ocf) and ocf > 0 else 0.0
            balance_score += 3.0 if np.isfinite(fcf) and fcf > 0 else 0.0
            balance_score += 3.0 if np.isfinite(debt_equity) and debt_equity <= 0.8 else 1.0 if np.isfinite(debt_equity) and debt_equity <= 1.5 else 0.0
            balance_score += 3.0 if np.isfinite(current_ratio) and current_ratio >= 1.5 else 1.0 if np.isfinite(current_ratio) and current_ratio >= 1.0 else 0.0
            balance_score += 2.0 if np.isfinite(cash_to_debt) and cash_to_debt >= 0.5 else 1.0 if np.isfinite(cash_to_debt) and cash_to_debt >= 0.2 else 0.0
        # Banks/financials require CAR, NPL, LDR and funding-quality data that
        # Yahoo's generic snapshot does not provide consistently. Do not award
        # fake solvency points from missing DER/current-ratio fields.
        solvency_fields = (debt_equity, current_ratio, cash_to_debt)
        solvency_coverage = 100.0 * sum(np.isfinite(value) for value in solvency_fields) / len(solvency_fields)
        valuation_score = 0.0
        valuation_score += 6.0 if np.isfinite(peg) and 0 < peg <= 1.5 else 3.0 if np.isfinite(peg) and peg <= 2.5 else 0.0
        valuation_score += 4.0 if np.isfinite(fcf_yield) and fcf_yield >= 0.04 else 2.0 if np.isfinite(fcf_yield) and fcf_yield > 0 else 0.0
        momentum_score = 0.0
        momentum_score += 6.0 if roc60 >= 0.15 else 4.0 if roc60 >= 0.07 else 0.0
        momentum_score += 5.0 if roc120 >= 0.25 else 3.0 if roc120 >= 0.12 else 0.0
        momentum_score += 4.0 if rs60 > 0 else 0.0
        momentum_score += 3.0 if dist_high >= -0.15 else 1.0 if dist_high >= -0.30 else 0.0
        momentum_score += 2.0 if close > _finite(row.get("EMA200"), float("inf")) else 0.0
        accumulation_score = 0.0
        accumulation_score += 5.0 if accumulation >= 80 else 3.0 if accumulation >= 65 else 0.0
        accumulation_score += 3.0 if cmf_v > 0 and obv_up else 0.0
        accumulation_score += 2.0 if adtv >= 3_000_000_000 else 1.0 if adtv >= 1_000_000_000 else 0.0
        total = growth_score + profitability_score + balance_score + valuation_score + momentum_score + accumulation_score
        severe_flags = any(flag in red_flags for flag in ("Margin bersih negatif", "OCF negatif", "DER tinggi"))
        if severe_flags:
            total = min(total, 69.0)
        if np.isfinite(market_cap) and market_cap < 300_000_000_000:
            total -= 4.0  # extreme micro-cap governance/liquidity penalty
        total = max(0.0, min(100.0, total))
        if total < 60:
            continue
        if (
            total >= 82 and coverage >= 70 and adtv >= 1_500_000_000
            and not severe_flags and not is_financial and solvency_coverage >= 66.0
        ):
            status = "MULTIBAGGER_A_CANDIDATE"
        elif total >= 72 and coverage >= 60 and not severe_flags:
            status = "MULTIBAGGER_B_CANDIDATE"
        else:
            status = "MULTIBAGGER_WATCHLIST"
        sig = signal_map.get(ticker, {})
        technical_entry_state = _safe_text(sig.get("status")) or "NO_ACTIVE_ENTRY_SETUP"
        rows.append({
            "ticker": ticker,
            "multibagger_status": status,
            "multibagger_score": round(total, 1),
            "growth_score": round(growth_score, 1),
            "profitability_score": round(profitability_score, 1),
            "balance_sheet_score": round(balance_score, 1),
            "valuation_score": round(valuation_score, 1),
            "momentum_score": round(momentum_score, 1),
            "accumulation_score": round(accumulation_score, 1),
            "fundamental_coverage": coverage,
            "fundamental_score": fund.get("fundamental_score"),
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "roe": roe,
            "roa": roa,
            "net_margin": net_margin,
            "debt_equity": debt_equity,
            "current_ratio": current_ratio,
            "cash_to_debt": cash_to_debt,
            "operating_cash_flow": ocf,
            "free_cash_flow": fcf,
            "solvency_coverage": round(solvency_coverage, 1),
            "fundamental_model": fundamental_model,
            "peg_ratio": peg,
            "fcf_yield": fcf_yield,
            "market_cap": market_cap,
            "roc60": roc60,
            "roc120": roc120,
            "relative_strength60": rs60,
            "distance_52w_high": dist_high,
            "silent_accumulation_score": accumulation,
            "up_down_value_ratio20": up_down,
            "adtv20_idr": adtv,
            "active_setup": sig.get("setup", ""),
            "technical_entry_state": technical_entry_state,
            "entry": sig.get("entry", np.nan),
            "stop_loss": sig.get("stop_loss", np.nan),
            "tp1": sig.get("tp1", np.nan),
            "tp2": sig.get("tp2", np.nan),
            "horizon": "12–36 months; quarterly review",
            "red_flags": red_flags,
            "note": (
                "Financial-sector solvency needs CAR/NPL/LDR verification; generic model caps A grade"
                if is_financial else "Candidate ranking, not a forecast or guaranteed multiple"
            ),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {"MULTIBAGGER_A_CANDIDATE": 0, "MULTIBAGGER_B_CANDIDATE": 1, "MULTIBAGGER_WATCHLIST": 2}
        result["_rank"] = result["multibagger_status"].map(rank).fillna(9)
        result = result.sort_values(["_rank", "multibagger_score", "adtv20_idr"], ascending=[True, False, False]).drop(columns="_rank").reset_index(drop=True)
    return result


def scan_ara_hunter_candidates(
    prepared: Mapping[str, pd.DataFrame],
    intraday: Mapping[str, pd.DataFrame] | None = None,
    core_signals: pd.DataFrame | None = None,
    config: ScanConfig | None = None,
    now: Any | None = None,
    market_context: MarketContext | None = None,
) -> pd.DataFrame:
    """Find high-momentum names with remaining ARA room; never auto-execute."""
    cfg = config or ScanConfig()
    intraday = intraday or {}
    critical_map = _critical_context_map(core_signals)
    now_jkt = _jakarta_timestamp(now)
    market_regime = market_context.regime if market_context is not None else "NOT_EVALUATED"
    regime_blocked = market_context is not None and market_regime in {"RISK_OFF", "UNKNOWN"}
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 60:
            continue
        row = frame.iloc[-1]
        prev = frame.iloc[-2]
        close = _finite(row.get("Close"), 0.0)
        prev_close = _finite(prev.get("Close"), 0.0)
        if close <= 0 or prev_close <= 0:
            continue
        ara_price = idx_daily_price_band(prev_close)[1]
        if not ara_price:
            continue
        daily_return = close / prev_close - 1.0
        room = ara_price / close - 1.0
        vol_ratio = _finite(row.get("VOL_RATIO"), 0.0)
        value_today = _finite(row.get("VALUE"), 0.0)
        adtv = _finite(row.get("ADTV20"), 0.0)
        close_location = _finite(row.get("CLOSE_LOCATION"), 0.0)
        body_atr = _finite(row.get("BODY_ATR"), 0.0)
        cmf_v = _finite(row.get("CMF20"), -1.0)
        obv_up = _finite(row.get("OBV_SLOPE10"), -1.0) > 0
        rsi_v = _finite(row.get("RSI14"), 50.0)
        mfi_v = _finite(row.get("MFI14"), 50.0)
        breakout20 = close > _finite(row.get("HIGH20_PREV"), float("inf"))
        opening_gap = _finite(row.get("Open"), close) / prev_close - 1.0
        accumulation, up_down = _silent_accumulation_metrics(frame)
        im = _intraday_metrics(
            intraday.get(ticker, pd.DataFrame()), now_jkt, cfg.max_intraday_stale_minutes
        )
        has_intraday = im["intraday_bars"] >= 6 and _truthy(im["intraday_fresh"])
        intraday_confirm = bool(
            has_intraday
            and _finite(im["intraday_last"], 0.0) >= _finite(im["session_vwap"], float("inf"))
            and _finite(im["late_volume_acceleration"], 0.0) >= 1.25
            and _finite(im["session_close_location"], 0.0) >= 0.75
        )
        near_or_locked = near_upper_auto_rejection(prev_close, close, _finite(row.get("High"), close))
        context_messages = [critical_map.get(ticker, "")]
        if regime_blocked:
            context_messages.append(f"Regime IHSG {market_regime} tidak mendukung momentum spekulatif")
        context_blocker = " • ".join(message for message in context_messages if message)
        score = 0.0
        score += 18.0 if vol_ratio >= 3.0 else 14.0 if vol_ratio >= 2.0 else 8.0 if vol_ratio >= 1.5 else 0.0
        score += 12.0 if value_today >= 20_000_000_000 else 8.0 if value_today >= 8_000_000_000 else 4.0 if value_today >= 3_000_000_000 else 0.0
        score += 12.0 if close_location >= 0.90 else 8.0 if close_location >= 0.80 else 0.0
        score += 10.0 if body_atr >= 1.0 else 6.0 if body_atr >= 0.70 else 0.0
        score += 12.0 if cmf_v >= 0.10 and obv_up else 7.0 if cmf_v > 0 and obv_up else 0.0
        score += 8.0 if breakout20 else 0.0
        score += 8.0 if 0.03 <= daily_return <= 0.12 else 4.0 if 0.01 <= daily_return <= 0.15 else 0.0
        score += 6.0 if 55 <= rsi_v <= 76 and 55 <= mfi_v <= 82 else 0.0
        score += 6.0 if accumulation >= 75 else 3.0 if accumulation >= 60 else 0.0
        score += 8.0 if intraday_confirm else 0.0
        if opening_gap > 0.08:
            score -= 10.0
        if near_or_locked or room < 0.035:
            score -= 20.0
        if context_blocker:
            score = min(score, 50.0)
        score = max(0.0, min(100.0, score))
        strict_gate = bool(
            adtv >= max(cfg.min_adtv_idr, 3_000_000_000.0)
            and value_today >= 5_000_000_000.0
            and vol_ratio >= 1.8
            and close_location >= 0.80
            and body_atr >= 0.70
            and cmf_v > 0
            and obv_up
            and breakout20
            and 0.02 <= daily_return <= 0.15
            and room >= 0.04
            and not near_or_locked
            and opening_gap <= 0.08
            and not context_blocker
        )
        if score < 60 and not strict_gate:
            continue
        if strict_gate and score >= 85 and intraday_confirm:
            status = "ARA_HUNTER_STRICT"
        elif strict_gate and score >= 78:
            status = "ARA_HUNTER_WATCHLIST"
        else:
            status = "ARA_HUNTER_REJECT"
        atr_v = _finite(row.get("ATR14"), close * 0.03)
        entry = round_idx_price(close, "up")
        stop = round_idx_price(entry - min(0.04 * entry, max(1.1 * atr_v, 3 * idx_tick_size(entry))), "down")
        if stop >= entry:
            stop = round_idx_price(entry - 3 * idx_tick_size(entry), "down")
        rows.append({
            "ticker": ticker,
            "ara_hunter_status": status,
            "ara_hunter_score": round(score, 1),
            "last_price": close,
            "previous_close": prev_close,
            "ara_price": ara_price,
            "room_to_ara_pct": room,
            "daily_return_pct": daily_return,
            "opening_gap_pct": opening_gap,
            "entry_reference": entry,
            "hard_stop": stop,
            "risk_pct": (entry - stop) / entry if entry > 0 else np.nan,
            "volume_ratio": vol_ratio,
            "value_today_idr": value_today,
            "adtv20_idr": adtv,
            "close_location": close_location,
            "body_atr": body_atr,
            "cmf20": cmf_v,
            "rsi14": rsi_v,
            "mfi14": mfi_v,
            "breakout20": breakout20,
            "silent_accumulation_score": accumulation,
            "up_down_value_ratio20": up_down,
            "intraday_confirmed": intraday_confirm,
            "intraday_data_state": im["intraday_data_state"],
            "intraday_session_date": im["intraday_session_date"],
            "intraday_age_minutes": im["intraday_age_minutes"],
            "market_regime": market_regime,
            "late_volume_acceleration": im["late_volume_acceleration"],
            "session_close_location": im["session_close_location"],
            "order_instruction": "SPECULATIVE_REVIEW_ONLY",
            "max_position_pct_equity": 0.05,
            "risk_class": "VERY_HIGH",
            "blockers": context_blocker or ("Sudah dekat/terkunci ARA" if near_or_locked else ""),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {"ARA_HUNTER_STRICT": 0, "ARA_HUNTER_WATCHLIST": 1, "ARA_HUNTER_REJECT": 2}
        result["_rank"] = result["ara_hunter_status"].map(rank).fillna(9)
        result = result.sort_values(["_rank", "ara_hunter_score", "value_today_idr"], ascending=[True, False, False]).drop(columns="_rank").reset_index(drop=True)
    return result


def build_specialty_screens(
    prepared: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame | None = None,
    core_signals: pd.DataFrame | None = None,
    market_context: MarketContext | None = None,
    intraday: Mapping[str, pd.DataFrame] | None = None,
    config: ScanConfig | None = None,
    now: Any | None = None,
) -> dict[str, pd.DataFrame]:
    """Create all specialty tables from the same prepared daily universe."""
    return {
        "sniper": scan_sniper_entries(core_signals, prepared, config=config),
        "bsjp": scan_bsjp_candidates(
            prepared, intraday=intraday, core_signals=core_signals, config=config,
            now=now, market_context=market_context,
        ),
        "bpjs": scan_bpjs_candidates(
            prepared, intraday=intraday, core_signals=core_signals, config=config,
            now=now, market_context=market_context,
        ),
        "multibagger": scan_multibagger_candidates(prepared, fundamentals, core_signals=core_signals, config=config),
        "ara_hunter": scan_ara_hunter_candidates(
            prepared, intraday=intraday, core_signals=core_signals, config=config,
            now=now, market_context=market_context,
        ),
    }


__all__ = list(dict.fromkeys(__all__ + [
    "download_intraday_ohlcv",
    "specialty_intraday_shortlist",
    "scan_sniper_entries",
    "scan_bsjp_candidates",
    "scan_bpjs_candidates",
    "scan_multibagger_candidates",
    "scan_ara_hunter_candidates",
    "build_specialty_screens",
]))


# ---- v4.2.2 evidence completeness and provider coverage upgrade ----

_DATA_LAYER_WEIGHTS: dict[str, float] = {
    "technical": 0.25,
    "risk": 0.15,
    "fundamental": 0.15,
    "validation": 0.15,
    "market_status": 0.10,
    "news": 0.10,
    "quote": 0.05,
    "universe": 0.05,
}


def _is_present_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return bool(pd.notna(value)) and (not isinstance(value, float) or np.isfinite(value))
    except Exception:
        return True


def _field_coverage(row: Mapping[str, Any], fields: Iterable[str]) -> float:
    names = tuple(fields)
    if not names:
        return 0.0
    present = sum(_is_present_value(row.get(name)) for name in names)
    return 100.0 * present / len(names)


def _technical_data_coverage(row: Mapping[str, Any]) -> float:
    return _field_coverage(
        row,
        (
            "ticker", "setup", "quality_score", "last_price", "entry_low",
            "entry_high", "entry", "stop_loss", "tp1", "tp2",
            "volume_ratio", "adtv20_idr", "distance_atr", "market_regime",
        ),
    )


def _risk_data_coverage(row: Mapping[str, Any]) -> float:
    return _field_coverage(
        row,
        (
            "entry", "stop_loss", "tp1", "tp2", "rr1", "rr2",
            "stop_pct", "suggested_lots", "capital_required_idr", "max_loss_idr",
        ),
    )


def _validation_data_coverage(row: Mapping[str, Any]) -> float:
    # Coverage is deliberately independent from whether thresholds pass. A small
    # but fully calculated OOS sample is complete evidence with low confidence,
    # not missing evidence.
    return _field_coverage(
        row,
        (
            "validation_scope", "signal_events_oos", "filled_events",
            "entry_fill_rate_5d", "bayes_probability", "tp1_ci_low",
            "expectancy_r", "profit_factor", "max_losing_streak",
            "median_fill_bars", "median_time_to_tp1_bars",
        ),
    )


def _market_status_data_coverage(row: Mapping[str, Any], cfg: ScanConfig) -> float:
    status = _safe_text(row.get("market_status_coverage")).upper()
    if status == "AUTO_VERIFIED":
        return 100.0
    components = {
        part.strip() for part in _safe_text(row.get("market_status_components")).split(",")
        if part.strip()
    }
    component_score = min(100.0, 20.0 * len(components))
    data_age = _finite(row.get("absolute_data_age_days"), 999)
    final_ohlcv = data_age <= cfg.max_absolute_data_age_days and not _truthy(row.get("current_bar_incomplete", False))
    quote_verified = _truthy(row.get("quote_verified", False))
    if quote_verified:
        # A verified quote confirms tradability at the quote timestamp, but it
        # cannot prove absence from suspension/FCA/corporate-action lists.
        return max(55.0, component_score)
    if final_ohlcv:
        # Recent OHLCV is only weak negative evidence against a stale/suspended
        # state. It must not be counted as mostly-complete official IDX status.
        return max(35.0, component_score)
    return max(20.0 if _is_present_value(row.get("market_status_asof")) else 0.0, component_score)


def _news_data_coverage(row: Mapping[str, Any], cfg: ScanConfig) -> float:
    status = _safe_text(row.get("news_review_status")).upper()
    provider_ok = _truthy(row.get("provider_query_ok", False))
    age = _finite(row.get("news_review_age_days"), np.nan)
    lookback = _finite(row.get("news_lookback_days"), np.nan)
    structural = (
        _is_present_value(row.get("news_reviewed_at"))
        and _is_present_value(row.get("coverage_start"))
        and _is_present_value(row.get("coverage_end"))
    )
    fresh = np.isfinite(age) and 0 <= age <= cfg.news_cache_days
    adequate_window = np.isfinite(lookback) and lookback >= cfg.min_news_lookback_days
    if status == "COMPLETE" and provider_ok and fresh and adequate_window:
        if "idx_disclosure_query_ok" in row and not _truthy(row.get("idx_disclosure_query_ok", False)):
            return 90.0
        return 100.0
    if provider_ok and structural and fresh:
        return 80.0 if adequate_window else 70.0
    if status in {"INCOMPLETE", "PARTIAL"} and structural:
        return 55.0
    if structural:
        return 40.0
    # Missing news does not imply negative news. It remains an explicit gap and
    # materially lowers confidence, but it should not erase otherwise complete
    # OHLCV, risk, fundamental, and OOS layers.
    return 20.0


def _quote_data_coverage(row: Mapping[str, Any], cfg: ScanConfig) -> float:
    verified = _truthy(row.get("quote_verified", False))
    age = _finite(row.get("quote_age_days"), np.nan)
    if verified and np.isfinite(age) and 0 <= age <= 3:
        return 100.0
    data_age = _finite(row.get("absolute_data_age_days"), 999)
    final_ohlcv = data_age <= cfg.max_absolute_data_age_days and not _truthy(row.get("current_bar_incomplete", False))
    # A final daily bar is a useful fallback, but it is the same Yahoo price
    # family as the scan input and is not an independent execution quote.
    return 60.0 if final_ohlcv else 25.0


def _universe_data_coverage(row: Mapping[str, Any]) -> float:
    requested = _finite(row.get("universe_requested_count"), 0)
    prepared = _finite(row.get("universe_prepared_count"), 0)
    explicit = _finite(row.get("universe_coverage_pct"), np.nan)
    if np.isfinite(explicit):
        return min(100.0, max(0.0, explicit))
    if requested > 0:
        return min(100.0, max(0.0, 100.0 * prepared / requested))
    return min(100.0, max(0.0, _finite(row.get("universe_confidence"), 0)))


def _layer_data_coverage(row: Mapping[str, Any], cfg: ScanConfig) -> dict[str, float]:
    technical = _technical_data_coverage(row)
    core_trade_fields = (
        "ticker", "quality_score", "last_price", "entry", "stop_loss",
        "tp1", "tp2", "rr1", "rr2", "stop_pct", "distance_atr",
        "adtv20_idr", "volume_ratio", "market_regime",
    )
    if all(_is_present_value(row.get(name)) for name in core_trade_fields):
        technical = max(technical, 100.0)
    risk = _risk_data_coverage(row)
    if _risk_layer_confidence(row, cfg) >= 100.0:
        risk = 100.0
    fundamental = min(100.0, max(0.0, _finite(row.get("fundamental_coverage"), 0.0)))
    validation = _validation_data_coverage(row)
    market_status = _market_status_data_coverage(row, cfg)
    news = _news_data_coverage(row, cfg)
    quote = _quote_data_coverage(row, cfg)

    # Backward-compatible inference for compact/synthetic rows: a high layer
    # confidence can only have been produced after that layer was evaluated.
    # Low confidence never upgrades coverage because it may be a neutral default.
    confidence_fallbacks = {
        "fundamental": _finite(row.get("fundamental_confidence"), 0.0),
        "validation": _finite(row.get("validation_confidence"), 0.0),
        "market_status": _finite(row.get("market_status_confidence"), 0.0),
        "news": _finite(row.get("news_confidence"), 0.0),
        "quote": _finite(row.get("quote_confidence"), 0.0),
    }
    if fundamental <= 0 and confidence_fallbacks["fundamental"] >= 70:
        fundamental = confidence_fallbacks["fundamental"]
    if confidence_fallbacks["validation"] >= 70:
        validation = max(validation, confidence_fallbacks["validation"])
    if confidence_fallbacks["market_status"] >= 70:
        market_status = max(market_status, confidence_fallbacks["market_status"])
    if confidence_fallbacks["news"] >= 70:
        news = max(news, confidence_fallbacks["news"])
    if confidence_fallbacks["quote"] >= 70:
        quote = max(quote, confidence_fallbacks["quote"])

    layers = {
        "technical": technical,
        "risk": risk,
        "fundamental": fundamental,
        "validation": validation,
        "market_status": market_status,
        "news": news,
        "quote": quote,
        "universe": _universe_data_coverage(row),
    }
    return {name: round(min(100.0, max(0.0, value)), 1) for name, value in layers.items()}


def _google_news_rss_items(ticker: str, lookback_days: int, timeout: int = 8) -> tuple[list[dict[str, Any]], bool, str]:
    """Retrieve a lightweight Indonesian Google News RSS query.

    This is used before Yahoo news to avoid repeated crumb/authorization errors.
    A successful empty RSS response is still valid coverage for the requested
    window; it is not interpreted as positive news.
    """
    import email.utils
    import requests
    import xml.etree.ElementTree as ET
    from urllib.parse import quote_plus

    code = normalize_idx_ticker(ticker).replace(".JK", "")
    query = quote_plus(f'"{code}" saham when:{max(1, int(lookback_days))}d')
    url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; IDXSuperScanner/4.2.2)"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items: list[dict[str, Any]] = []
        for node in root.findall(".//item")[:30]:
            title = (node.findtext("title") or "").strip()
            link = (node.findtext("link") or "").strip()
            description = (node.findtext("description") or "").strip()
            raw_date = (node.findtext("pubDate") or "").strip()
            published = None
            if raw_date:
                try:
                    parsed = email.utils.parsedate_to_datetime(raw_date)
                    published = pd.Timestamp(parsed).tz_convert("Asia/Jakarta").tz_localize(None)
                except Exception:
                    published = pd.to_datetime(raw_date, utc=True, errors="coerce")
                    if pd.notna(published):
                        published = published.tz_convert("Asia/Jakarta").tz_localize(None)
            items.append(
                {
                    "title": title,
                    "summary": description,
                    "link": link,
                    "pubDate": published.isoformat() if published is not None and pd.notna(published) else None,
                }
            )
        return items, True, ""
    except Exception as exc:
        return [], False, f"{type(exc).__name__}: {str(exc)[:120]}"


def fetch_automatic_news_review(
    tickers: Iterable[str], lookback_days: int = 7, max_workers: int = 3
) -> pd.DataFrame:
    """Review current news with Google News RSS, Yahoo fallback, and IDX disclosure.

    Google RSS is the primary broad-news source because it does not depend on a
    Yahoo crumb. Yahoo is queried only if RSS fails. Official IDX disclosure is
    retained as the event-risk cross-check.
    """
    import yfinance as yf

    names = list(dict.fromkeys(tickers))
    if not names:
        return pd.DataFrame()
    now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    start = now - pd.Timedelta(days=lookback_days)
    disclosure_text, disclosure_ok, disclosure_error = _fetch_idx_disclosure_page(timeout=8)
    disclosure_mentions = _requested_mentions(disclosure_text, names) if disclosure_ok else set()

    def one(ticker: str) -> dict[str, Any]:
        items, rss_ok, rss_error = _google_news_rss_items(ticker, lookback_days, timeout=8)
        yahoo_ok = False
        yahoo_error = ""
        if not rss_ok:
            try:
                obj = yf.Ticker(ticker)
                try:
                    raw = obj.get_news(count=30)
                except Exception:
                    raw = obj.news
                items = list(raw or [])
                yahoo_ok = True
            except Exception as exc:
                yahoo_error = f"{type(exc).__name__}: {str(exc)[:100]}"
        query_ok = rss_ok or yahoo_ok
        reviewed: list[dict[str, Any]] = []
        severe = False
        ambiguous = ticker in disclosure_mentions
        sources: list[str] = []
        titles: list[str] = []
        positive_count = 0
        negative_count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            title, summary, url, published = _news_item_fields(item)
            if published is not None and published < start:
                continue
            body = f"{title} {summary}".upper()
            is_negative = any(term in body for term in _NEGATIVE_NEWS_TERMS)
            is_event = any(term in body for term in _EVENT_RISK_TERMS)
            severe = severe or is_negative
            ambiguous = ambiguous or is_event
            if is_negative:
                negative_count += 1
            elif any(term in body for term in ("PROFIT", "LABA", "GROWTH", "PERTUMBUHAN", "CONTRACT", "KONTRAK")):
                positive_count += 1
            if title:
                titles.append(title)
            if url:
                sources.append(url)
            reviewed.append(item)
        # News breadth can be complete when the broad provider succeeds. IDX
        # disclosure failure remains a warning and lowers confidence separately,
        # but does not nullify a successful dated news query.
        complete = query_ok
        provider = "Google News RSS" if rss_ok else "Yahoo Finance fallback" if yahoo_ok else "UNAVAILABLE"
        if disclosure_ok:
            provider += " + official IDX disclosure"
        errors = " | ".join(value for value in (rss_error, yahoo_error, disclosure_error) if value)
        return {
            "ticker": ticker,
            "news_reviewed_at": now,
            "news_review_status": "COMPLETE" if complete else "INCOMPLETE",
            "provider_query_ok": bool(complete),
            "items_reviewed": len(reviewed),
            "coverage_start": start,
            "coverage_end": now,
            "news_provider": provider,
            "verified_catalyst_count": positive_count,
            "verified_negative_count": negative_count,
            "severe_negative_news": severe,
            "ambiguous_material_news": ambiguous,
            "catalyst_summary": " | ".join(titles[:3]),
            "news_sources": " | ".join(sources[:3]),
            "news_error": errors,
            "idx_disclosure_query_ok": bool(disclosure_ok),
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(names))) as pool:
        futures = [pool.submit(one, ticker) for ticker in names]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def finalize_execution_integrity(signals: pd.DataFrame, config: ScanConfig | None = None) -> pd.DataFrame:
    """Finalize orders with separate confidence and evidence completeness.

    `execution_confidence_score` measures how strong/favorable the evidence is.
    `data_completeness_score` measures how much of the required evidence was
    actually populated. A weak result can therefore be 95% complete but low
    confidence; that is materially different from missing data.
    """
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    for column, default in (
        ("critical_blockers", ""), ("evidence_warnings", ""),
        ("market_status_confidence", 45.0), ("news_confidence", 52.0),
        ("fundamental_confidence", 50.0), ("validation_confidence", 45.0),
        ("quote_confidence", 68.0), ("universe_confidence", 48.0),
        ("portfolio_selected", False), ("independent_price_verified", False),
        ("independent_price_state", "MISSING_INDEPENDENT"),
        ("independent_source_family", ""),
    ):
        if column not in out:
            out[column] = default
    for idx, row in out.iterrows():
        technical_ready = bool(
            _truthy(row.get("technical_setup_ready", False))
            or _truthy(row.get("technical_execution_candidate", False))
            or _safe_text(row.get("status")) == "EXECUTION_READY"
        )
        technical_conf = min(100.0, max(0.0, _finite(row.get("quality_score"), 0)))
        risk_conf = _risk_layer_confidence(row, cfg)
        market_conf = _finite(row.get("market_status_confidence"), 45)
        news_conf = _finite(row.get("news_confidence"), 52)
        fundamental_conf = _finite(row.get("fundamental_confidence"), 50)
        fundamental_coverage = _finite(row.get("fundamental_coverage"), 0)
        validation_conf = _finite(row.get("validation_confidence"), 45)
        quote_conf = _finite(row.get("quote_confidence"), 68)
        universe_conf = _finite(row.get("universe_confidence"), 48)

        if market_conf < 60 and quote_conf >= 90 and _finite(row.get("absolute_data_age_days"), 999) <= cfg.max_absolute_data_age_days:
            market_conf = 70.0
            out.at[idx, "market_status_confidence"] = market_conf
            _append_pipe(out, idx, "evidence_warnings", "Status IDX memakai provisional quote/OHLCV fallback")

        confidence_weights = {
            "technical": (technical_conf, 0.35),
            "risk": (risk_conf, 0.20),
            "market_status": (market_conf, 0.10),
            "news": (news_conf, 0.08),
            "fundamental": (fundamental_conf, 0.10),
            "validation": (validation_conf, 0.07),
            "quote": (quote_conf, 0.05),
            "universe": (universe_conf, 0.05),
        }
        confidence = round(sum(value * weight for value, weight in confidence_weights.values()), 1)

        coverages = _layer_data_coverage(row, cfg)
        completeness = round(sum(coverages[name] * weight for name, weight in _DATA_LAYER_WEIGHTS.items()), 1)
        for name, value in coverages.items():
            out.at[idx, f"{name}_data_coverage"] = value
        out.at[idx, "data_completeness_score"] = completeness
        out.at[idx, "data_completeness_tier"] = (
            "HIGH" if completeness >= 90 else
            "SUFFICIENT" if completeness >= cfg.min_data_completeness else
            "PARTIAL" if completeness >= 60 else "LOW"
        )
        missing_layers = [name for name, value in coverages.items() if value < 60]
        direct_fundamental_ok = bool(
            (not cfg.real_money_mode)
            or fundamental_coverage >= cfg.min_direct_fundamental_coverage
        )
        daily_source_tier = _safe_text(row.get("ohlcv_source_tier")).upper()
        direct_daily_source_ok = bool(
            (not cfg.real_money_mode)
            or daily_source_tier not in {"CACHE_FALLBACK", "UNAVAILABLE"}
        )
        independent_verified = _truthy(row.get("independent_price_verified", False))
        independent_price_ok = bool(
            (not cfg.real_money_mode)
            or (not cfg.require_independent_price_verification)
            or independent_verified
        )
        if not direct_fundamental_ok and "fundamental" not in missing_layers:
            missing_layers.append("fundamental")
        if not direct_daily_source_ok and "live_ohlcv" not in missing_layers:
            missing_layers.append("live_ohlcv")
        if not independent_price_ok and "independent_price" not in missing_layers:
            missing_layers.append("independent_price")
        out.at[idx, "data_missing_layers"] = " | ".join(missing_layers)
        if completeness < cfg.min_data_completeness:
            _append_pipe(
                out, idx, "evidence_warnings",
                f"Data completeness {completeness:.1f}% di bawah minimum {cfg.min_data_completeness:.0f}%",
            )
        if not direct_fundamental_ok:
            _append_pipe(
                out, idx, "evidence_warnings",
                f"Fundamental coverage {fundamental_coverage:.1f}% di bawah minimum direct-order "
                f"{cfg.min_direct_fundamental_coverage:.0f}%",
            )
        if not direct_daily_source_ok:
            _append_pipe(
                out, idx, "evidence_warnings",
                "OHLCV daily bukan hasil live; cache hanya boleh dipakai untuk riset/watchlist",
            )
        if not independent_price_ok:
            _append_pipe(
                out, idx, "evidence_warnings",
                "Direct order menunggu verifikasi harga dari keluarga provider independen",
            )

        critical = _safe_text(row.get("critical_blockers"))
        if _truthy(row.get("validation_critical_blocker", False)):
            critical = critical or "OOS edge negatif"
        if _truthy(row.get("fundamental_critical_blocker", False)):
            critical = critical or "Fundamental distress"
        if _truthy(row.get("market_status_critical_blocker", False)):
            critical = critical or "Status IDX negatif"
        if _truthy(row.get("news_critical_blocker", False)):
            critical = critical or "Berita material negatif"
        if _truthy(row.get("quote_critical_blocker", False)):
            critical = critical or "Quote/candle conflict"
        portfolio_ok = _truthy(row.get("portfolio_selected", False))
        if technical_ready and not portfolio_ok:
            _append_pipe(out, idx, "portfolio_blockers", "Belum dipilih oleh budget portofolio")

        base_direct_gates = bool(
            technical_ready
            and not critical
            and risk_conf == 100.0
            and portfolio_ok
            and direct_fundamental_ok
            and direct_daily_source_ok
        )
        projected_market_conf = 70.0 if market_conf < 60 else market_conf
        projected_quote_conf = max(100.0, quote_conf)
        projected_completeness = round(
            min(100.0, completeness + _DATA_LAYER_WEIGHTS["quote"] * (100.0 - coverages["quote"])),
            1,
        )
        projected_confidence = round(
            confidence
            + 0.10 * (projected_market_conf - market_conf)
            + 0.05 * (projected_quote_conf - quote_conf),
            1,
        )
        ready_except_independent = bool(
            base_direct_gates
            and projected_completeness >= cfg.min_data_completeness
            and projected_confidence >= cfg.min_execution_confidence
        )
        direct = bool(
            base_direct_gates
            and independent_price_ok
            and completeness >= cfg.min_data_completeness
            and confidence >= cfg.min_execution_confidence
        )
        if direct:
            final_status = "EXECUTION_READY"
        elif critical:
            final_status = "REJECT" if _safe_text(row.get("status")) == "REJECT" else "BLOCKED_CONTEXT"
        elif ready_except_independent and not independent_price_ok:
            final_status = "READY_FOR_PRICE_VERIFY"
        elif technical_ready and (confidence >= cfg.min_pending_confidence or completeness < cfg.min_data_completeness):
            final_status = "PENDING_DATA"
        else:
            current = _safe_text(row.get("status"))
            final_status = current if current in {"WATCHLIST_ENTRY", "REJECT"} else "WATCHLIST_ENTRY"

        out.at[idx, "status"] = final_status
        out.at[idx, "execution_integrity_score"] = confidence
        out.at[idx, "execution_confidence_score"] = confidence
        out.at[idx, "projected_completeness_with_independent_price"] = projected_completeness
        out.at[idx, "projected_confidence_with_independent_price"] = projected_confidence
        source_family = _safe_text(row.get("independent_source_family"))
        out.at[idx, "automated_provider_families"] = 4 if source_family == "TWELVE_DATA" else 3
        out.at[idx, "independent_price_provider_families"] = 2 if independent_verified else 1
        out.at[idx, "price_cross_validation_state"] = (
            "VERIFIED_INDEPENDENT" if independent_verified
            else _safe_text(row.get("independent_price_state")) or "MISSING_INDEPENDENT"
        )
        out.at[idx, "independent_price_required"] = bool(
            cfg.real_money_mode and cfg.require_independent_price_verification
        )
        out.at[idx, "critical_gate_pass"] = not bool(critical)
        out.at[idx, "evidence_state"] = (
            "RESOLVED" if direct else "ADVERSE" if critical else
            "SUFFICIENT_DATA" if completeness >= cfg.min_data_completeness else "PARTIAL"
        )
        out.at[idx, "order_instruction"] = "BUY_LIMIT" if direct else "DO_NOT_BUY"
        out.at[idx, "stockbit_order_price"] = row.get("entry") if direct else np.nan
        out.at[idx, "stockbit_order_lots"] = int(_finite(row.get("suggested_lots"), 0)) if direct else 0
        gate_checks = {
            "technical": technical_ready,
            "risk": risk_conf == 100.0,
            "context": not bool(critical),
            "portfolio": portfolio_ok,
            "fundamental": direct_fundamental_ok,
            "live_daily": direct_daily_source_ok,
            "independent_price": independent_price_ok,
            "completeness": (
                completeness >= cfg.min_data_completeness
                or (not independent_price_ok and projected_completeness >= cfg.min_data_completeness)
            ),
            "confidence": (
                confidence >= cfg.min_execution_confidence
                or (not independent_price_ok and projected_confidence >= cfg.min_execution_confidence)
            ),
        }
        failure_labels = {
            "technical": "TECHNICAL_TRIGGER_OR_DISTANCE",
            "risk": "RISK_LEVELS_OR_SIZING",
            "context": "CRITICAL_CONTEXT",
            "portfolio": "PORTFOLIO_BUDGET",
            "fundamental": "FUNDAMENTAL_COVERAGE",
            "live_daily": "DAILY_SOURCE_NOT_LIVE",
            "independent_price": "INDEPENDENT_PRICE_REQUIRED",
            "completeness": "DATA_COMPLETENESS",
            "confidence": "EXECUTION_CONFIDENCE",
        }
        failures = [failure_labels[name] for name, passed in gate_checks.items() if not passed]
        out.at[idx, "execution_gate_failures"] = " | ".join(failures)
        out.at[idx, "primary_execution_blocker"] = failures[0] if failures else "NONE"
        out.at[idx, "execution_readiness_pct"] = round(100.0 * sum(gate_checks.values()) / len(gate_checks), 1)
        out.at[idx, "automation_decision"] = (
            "DIRECT_EXECUTION_ELIGIBLE" if direct
            else "VERIFY_INDEPENDENT_PRICE" if final_status == "READY_FOR_PRICE_VERIFY"
            else "BLOCKED" if critical else "RETRY_OR_WATCH"
        )
    out["status_rank"] = out["status"].map(STATUS_ORDER).fillna(99)
    return out


__version__ = "4.2.4-real-money-data-hardening"
__all__ = list(dict.fromkeys(__all__ + [
    "fetch_automatic_news_review", "finalize_execution_integrity", "__version__",
]))


# =============================================================================
# v4.2.5 independent-price evidence and execution diagnostics
# =============================================================================

def _normalized_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _external_number(value: object) -> float:
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return number if np.isfinite(number) else np.nan
    text = str(value).strip().replace("Rp", "").replace("IDR", "").replace(" ", "")
    if not text:
        return np.nan
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        tail = text.rsplit(",", 1)[-1]
        text = text.replace(",", "." if len(tail) <= 2 else "")
    try:
        number = float(text)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _independent_source_family(source: object) -> str:
    text = _safe_text(source).upper()
    if "YAHOO" in text or "YFINANCE" in text:
        return "YAHOO"
    if "TWELVE" in text:
        return "TWELVE_DATA"
    if "IDX" in text or "BURSA EFEK" in text:
        return "IDX_OFFICIAL"
    if "STOCKBIT" in text:
        return "STOCKBIT"
    return "MANUAL_EXTERNAL"


def parse_independent_price_file(
    source: bytes | BinaryIO | pd.DataFrame,
    filename: str | None = None,
    default_source: str = "MANUAL_EXTERNAL_UPLOAD",
) -> pd.DataFrame:
    """Normalize an official IDX EOD export or a manual Stockbit quote file.

    Required semantic fields are ticker, date/as-of, and close/last price. The
    parser accepts common Indonesian and English column labels. A source label
    is retained so manual data can never be misrepresented as an automated feed.
    """
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        payload = BytesIO(source) if isinstance(source, bytes) else source
        if hasattr(payload, "seek"):
            payload.seek(0)
        suffix = Path(filename or getattr(source, "name", "")).suffix.lower()
        if suffix in {".xlsx", ".xlsm"}:
            frame = pd.read_excel(payload)
        else:
            try:
                frame = pd.read_csv(payload, sep=None, engine="python")
            except UnicodeDecodeError:
                if hasattr(payload, "seek"):
                    payload.seek(0)
                frame = pd.read_csv(payload, sep=None, engine="python", encoding="latin-1")
    if frame.empty:
        return pd.DataFrame(columns=[
            "ticker", "Date", "Open", "High", "Low", "Close", "Volume",
            "independent_source", "independent_source_family",
        ])

    lookup = {_normalized_column_name(column): column for column in frame.columns}

    def pick(aliases: Iterable[str], required: bool = False) -> object | None:
        selected = next((lookup[name] for name in aliases if name in lookup), None)
        if required and selected is None:
            raise ValueError("Kolom verifikasi harga tidak lengkap: ticker, date/asof, dan close/last_price wajib ada")
        return selected

    ticker_col = pick(("ticker", "tickers", "symbol", "kode", "kodesaham", "stockcode", "code"), True)
    date_col = pick(("date", "asof", "timestamp", "datetime", "tanggal", "tanggalperdagangan", "tradingdate", "time"), True)
    close_col = pick(("close", "closingprice", "lastprice", "last", "penutupan", "harga", "price", "lasttrade"), True)
    open_col = pick(("open", "openprice", "pembukaan"))
    high_col = pick(("high", "highprice", "tertinggi"))
    low_col = pick(("low", "lowprice", "terendah"))
    volume_col = pick(("volume", "vol", "totalvolume", "jumlahsaham"))
    source_col = pick(("source", "provider", "sumber", "datasource"))

    out = pd.DataFrame()
    out["ticker"] = frame[ticker_col].map(normalize_idx_ticker)
    def parse_external_date(value: object) -> pd.Timestamp | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        text = str(value).strip()
        iso_first = bool(re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", text))
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=not iso_first)
        return parsed if pd.notna(parsed) else None

    out["Date"] = frame[date_col].map(parse_external_date)
    for output, column in (("Open", open_col), ("High", high_col), ("Low", low_col), ("Close", close_col), ("Volume", volume_col)):
        out[output] = frame[column].map(_external_number) if column is not None else np.nan
    out["independent_source"] = (
        frame[source_col].map(_safe_text).replace("", default_source)
        if source_col is not None else default_source
    )
    out["independent_source_family"] = out["independent_source"].map(_independent_source_family)
    out = out.dropna(subset=["ticker", "Date", "Close"])
    out = out[out["Close"].gt(0)].copy()
    if out.empty:
        return out
    for column in ("Open", "High", "Low"):
        out[column] = out[column].where(out[column].gt(0))
    out["Volume"] = out["Volume"].where(out["Volume"].ge(0))
    out = out.sort_values(["ticker", "independent_source_family", "Date"])
    return out.drop_duplicates(["ticker", "independent_source_family", "Date"], keep="last").reset_index(drop=True)


def fetch_twelve_data_eod(
    tickers: Iterable[str],
    api_key: str,
    outputsize: int = 30,
    max_tickers: int = 12,
    timeout: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch bounded XIDX EOD validation history from Twelve Data.

    Twelve Data lists XIDX coverage as EOD and currently requires an eligible
    paid plan. This adapter is optional and only queries a candidate shortlist.
    API keys are passed as request parameters but are never written to output.
    """
    names = list(dict.fromkeys(tickers))[: max(0, int(max_tickers))]
    columns = [
        "ticker", "Date", "Open", "High", "Low", "Close", "Volume",
        "independent_source", "independent_source_family",
    ]
    if not names or not _safe_text(api_key):
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=["ticker", "status", "bars", "error"])
    import requests

    histories: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for ticker in names:
        symbol = ticker[:-3] if ticker.upper().endswith(".JK") else ticker
        try:
            response = requests.get(
                "https://api.twelvedata.com/time_series",
                params={
                    "symbol": symbol,
                    "exchange": "XIDX",
                    "interval": "1day",
                    "outputsize": max(5, min(100, int(outputsize))),
                    "timezone": "Asia/Jakarta",
                    "format": "JSON",
                    "apikey": api_key,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            values = payload.get("values", []) if isinstance(payload, dict) else []
            if not values:
                message = _safe_text(payload.get("message")) if isinstance(payload, dict) else "Respons kosong"
                reports.append({"ticker": ticker, "status": "FAILED", "bars": 0, "error": message or "Respons kosong"})
                continue
            frame = pd.DataFrame(values).rename(columns={
                "datetime": "Date", "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            frame["ticker"] = ticker
            frame["independent_source"] = "TWELVE_DATA_XIDX_EOD"
            frame["independent_source_family"] = "TWELVE_DATA"
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            for column in ("Open", "High", "Low", "Close", "Volume"):
                frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
            frame = frame[columns].dropna(subset=["Date", "Close"]).sort_values("Date")
            histories.append(frame)
            reports.append({"ticker": ticker, "status": "OK", "bars": len(frame), "error": ""})
        except Exception as exc:
            safe_error = str(exc).replace(str(api_key), "***")
            reports.append({"ticker": ticker, "status": "FAILED", "bars": 0, "error": f"{type(exc).__name__}: {safe_error[:140]}"})
    data = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame(columns=columns)
    return data, pd.DataFrame(reports)


def build_independent_price_validation(
    primary_histories: Mapping[str, pd.DataFrame],
    independent_data: pd.DataFrame | None,
    config: ScanConfig | None = None,
    now: Any | None = None,
) -> pd.DataFrame:
    """Compare latest price and, when available, recent return paths."""
    cfg = config or ScanConfig()
    result_columns = [
        "ticker", "independent_price_verified", "independent_price_state",
        "independent_source", "independent_source_family", "independent_asof",
        "independent_last_price", "independent_price_age_days",
        "independent_date_gap_days", "independent_price_divergence_pct",
        "independent_overlap_bars", "independent_close_mape_pct",
        "independent_return_correlation", "independent_price_confidence",
    ]
    if independent_data is None or independent_data.empty:
        return pd.DataFrame(columns=result_columns)
    required = {"ticker", "Date", "Close", "independent_source", "independent_source_family"}
    if not required.issubset(independent_data.columns):
        raise ValueError("Schema data harga independen tidak valid")
    now_jkt = _jakarta_timestamp(now).tz_localize(None)
    rows: list[dict[str, Any]] = []
    for ticker, ticker_data in independent_data.groupby("ticker", sort=False):
        primary = _clean_ohlcv(primary_histories.get(str(ticker), pd.DataFrame()), strict=True)
        if primary.empty:
            continue
        primary_close = primary[["Close"]].copy()
        primary_close.index = pd.to_datetime(primary_close.index).normalize()
        primary_close = primary_close[~primary_close.index.duplicated(keep="last")]
        primary_last_date = pd.Timestamp(primary_close.index[-1]).normalize()
        primary_last = float(primary_close["Close"].iloc[-1])
        candidates: list[dict[str, Any]] = []
        for family, group in ticker_data.groupby("independent_source_family", sort=False):
            group = group.copy().dropna(subset=["Date", "Close"]).sort_values("Date")
            if group.empty:
                continue
            group["_day"] = pd.to_datetime(group["Date"]).dt.normalize()
            group = group.drop_duplicates("_day", keep="last")
            latest = group.iloc[-1]
            external_date = pd.Timestamp(latest["Date"])
            if external_date.tzinfo is not None:
                external_date = external_date.tz_convert("Asia/Jakarta").tz_localize(None)
            external_last = float(latest["Close"])
            age_days = max(0, int((now_jkt.normalize() - external_date.normalize()).days))
            date_gap = abs(int((primary_last_date - external_date.normalize()).days))
            divergence = abs(external_last / primary_last - 1.0) if primary_last > 0 else np.nan
            tolerance = max(
                cfg.max_secondary_price_divergence_pct,
                2.0 * idx_tick_size(primary_last) / primary_last,
            )
            external_close = group.set_index("_day")[["Close"]].rename(columns={"Close": "external"})
            overlap = primary_close.rename(columns={"Close": "primary"}).join(external_close, how="inner").dropna()
            overlap_bars = len(overlap)
            mape = float((overlap["external"] / overlap["primary"] - 1.0).abs().median()) if overlap_bars else np.nan
            returns = overlap.pct_change().dropna()
            correlation = (
                float(returns["primary"].corr(returns["external"]))
                if len(returns) >= 3 and returns["primary"].std() > 0 and returns["external"].std() > 0
                else np.nan
            )
            needs_history = str(family) == "TWELVE_DATA"
            same_provider = str(family) == "YAHOO"
            history_ok = bool(
                not needs_history
                or (
                    overlap_bars >= cfg.min_secondary_overlap_bars
                    and ((np.isfinite(correlation) and correlation >= cfg.min_secondary_return_correlation)
                         or (np.isfinite(mape) and mape <= tolerance))
                )
            )
            fresh = age_days <= cfg.max_independent_price_age_days and date_gap <= cfg.max_independent_price_age_days
            aligned = np.isfinite(divergence) and divergence <= tolerance
            verified = bool(fresh and aligned and history_ok and not same_provider)
            if same_provider:
                state, confidence = "SAME_PROVIDER_FAMILY", 0.0
            elif verified:
                state, confidence = "VERIFIED_INDEPENDENT", 100.0
            elif not fresh:
                state, confidence = "STALE_INDEPENDENT", 20.0
            elif np.isfinite(divergence) and divergence > max(0.02, 2.0 * tolerance):
                state, confidence = "PRICE_CONFLICT", 0.0
            elif not history_ok:
                state, confidence = "HISTORY_MISMATCH", 10.0
            else:
                state, confidence = "PRICE_NOT_ALIGNED", 35.0
            priority = {"IDX_OFFICIAL": 4, "STOCKBIT": 3, "TWELVE_DATA": 2, "MANUAL_EXTERNAL": 1}.get(str(family), 0)
            candidates.append({
                "ticker": str(ticker),
                "independent_price_verified": verified,
                "independent_price_state": state,
                "independent_source": _safe_text(latest.get("independent_source")),
                "independent_source_family": str(family),
                "independent_asof": external_date,
                "independent_last_price": external_last,
                "independent_price_age_days": age_days,
                "independent_date_gap_days": date_gap,
                "independent_price_divergence_pct": divergence,
                "independent_overlap_bars": overlap_bars,
                "independent_close_mape_pct": mape,
                "independent_return_correlation": correlation,
                "independent_price_confidence": confidence,
                "_priority": priority,
            })
        if candidates:
            chosen = sorted(candidates, key=lambda item: (item["independent_price_verified"], item["_priority"], item["independent_price_confidence"]), reverse=True)[0]
            chosen.pop("_priority", None)
            rows.append(chosen)
    return pd.DataFrame(rows, columns=result_columns)


def apply_independent_price_gate(
    signals: pd.DataFrame,
    validation: pd.DataFrame | None,
    config: ScanConfig | None = None,
) -> pd.DataFrame:
    """Attach independent-price evidence without converting absence into bad data."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if validation is None or validation.empty:
        validation = pd.DataFrame({"ticker": out["ticker"].drop_duplicates()})
        validation["independent_price_verified"] = False
        validation["independent_price_state"] = "MISSING_INDEPENDENT"
        validation["independent_source"] = ""
        validation["independent_source_family"] = ""
        validation["independent_price_confidence"] = 0.0
    authoritative_columns = [column for column in validation.columns if column != "ticker"]
    if authoritative_columns:
        out = out.drop(columns=[column for column in authoritative_columns if column in out.columns])
    out = out.merge(validation, on="ticker", how="left")
    defaults = {
        "independent_price_verified": False,
        "independent_price_state": "MISSING_INDEPENDENT",
        "independent_source": "",
        "independent_source_family": "",
        "independent_price_confidence": 0.0,
    }
    for column, default in defaults.items():
        if column not in out:
            out[column] = default
        else:
            out[column] = out[column].fillna(default)
    if "quote_critical_blocker" not in out:
        out["quote_critical_blocker"] = False
    for idx, row in out.iterrows():
        verified = _truthy(row.get("independent_price_verified", False))
        state = _safe_text(row.get("independent_price_state")) or "MISSING_INDEPENDENT"
        if verified:
            out.at[idx, "independent_price_verified"] = True
            out.at[idx, "quote_confidence"] = max(100.0, _finite(row.get("quote_confidence"), 0.0))
        elif state in {"PRICE_CONFLICT", "HISTORY_MISMATCH"}:
            out.at[idx, "quote_critical_blocker"] = True
            _set_context_block(out, idx, f"Validasi harga independen gagal: {state}")
        else:
            _append_pipe(out, idx, "evidence_warnings", f"Harga independen belum terverifikasi: {state}")
    out["status_rank"] = out["status"].map(STATUS_ORDER).fillna(99)
    return out


__version__ = "4.2.5-multi-source-execution-funnel"
__all__ = list(dict.fromkeys(__all__ + [
    "parse_independent_price_file", "fetch_twelve_data_eod",
    "build_independent_price_validation", "apply_independent_price_gate",
    "__version__",
]))

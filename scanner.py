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

    # Stockbit/portfolio defaults for a Rp10 million individual account.
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
from typing import BinaryIO, Iterable

import pandas as pd



TICKER_COLUMNS = ("ticker", "tickers", "symbol", "symbols", "kode", "code", "emiten", "stock")


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
        out.index = out.index.tz_localize(None)
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
    tickers: Iterable[str], period: str = "3y", batch_size: int = 50
) -> tuple[dict[str, pd.DataFrame], DownloadReport]:
    import yfinance as yf

    requested = list(dict.fromkeys(tickers))
    histories: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}
    warnings: dict[str, str] = {}
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
                threads=True,
                progress=False,
                timeout=25,
            )
            for ticker in batch:
                extracted = _extract_batch(raw, ticker, len(batch))
                quality = ohlcv_quality_issues(extracted)
                frame = _clean_ohlcv(extracted, strict=True)
                if not frame.empty:
                    histories[ticker] = frame
                    if quality:
                        warnings[ticker] = " • ".join(quality)
                else:
                    failed[ticker] = "Data batch kosong"
        except Exception as exc:  # network providers can fail per batch
            for ticker in batch:
                failed[ticker] = f"Batch gagal: {type(exc).__name__}"

    missing = [t for t in requested if t not in histories]

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
        with ThreadPoolExecutor(max_workers=min(6, len(missing))) as pool:
            futures = [pool.submit(retry_one, ticker) for ticker in missing]
            for future in as_completed(futures):
                ticker, frame, error = future.result()
                if not frame.empty:
                    histories[ticker] = frame
                    failed.pop(ticker, None)
                    if error:
                        warnings[ticker] = error
                else:
                    failed[ticker] = error or "Tidak ada data"

    report = DownloadReport(
        requested,
        sorted(histories),
        failed,
        downloaded_at=pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
        warnings=warnings,
    )
    return histories, report


def download_benchmark(period: str = "3y") -> pd.DataFrame:
    import yfinance as yf

    try:
        frame = yf.Ticker("^JKSE").history(
            period=period, interval="1d", auto_adjust=True, repair=False, actions=False, timeout=20
        )
        return _clean_ohlcv(frame, strict=True)
    except Exception:
        return pd.DataFrame()


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


def fetch_fundamentals(tickers: Iterable[str], max_workers: int = 4) -> pd.DataFrame:
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
    if isinstance(source, pd.DataFrame):
        return source.copy()
    payload = BytesIO(source) if isinstance(source, bytes) else source
    try:
        return pd.read_csv(payload, sep=None, engine="python")
    except UnicodeDecodeError:
        if hasattr(payload, "seek"):
            payload.seek(0)
        return pd.read_csv(payload, sep=None, engine="python", encoding="latin-1")


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

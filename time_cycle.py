"""Objective Astronacci-style time-cycle intelligence for daily IDX analysis.

This module implements only publicly described, reproducible elements:
price/time/pattern/momentum, swing-to-swing timing, Fibonacci time projection,
calendar/lunar time markers, autocorrelation, spectral cycles, and price-time
confirmation.  It is not the proprietary Astronacci/Eye of Future formula and
never overrides structural invalidation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math

from eoff_reconstruction import (
    EOFFConfig, EOFF_VERSION, PUBLIC_ASTRO_PRIOR_WEIGHTS,
    analyze_eoff_reconstruction, setup_eoff_alignment,
)

import numpy as np
import pandas as pd

TIME_CYCLE_VERSION = "6.6.4-public-prior-eoff-weighting"
FIBONACCI_BARS = (5, 8, 13, 21, 34, 55, 89, 144, 233)
SYNODIC_MONTH_DAYS = 29.530588853
_NEW_MOON_EPOCH_UTC = pd.Timestamp("2000-01-06 18:14:00", tz="UTC")


@dataclass(frozen=True)
class TimeCycleConfig:
    min_bars: int = 180
    pivot_left: int = 3
    pivot_right: int = 3
    min_period: int = 8
    max_period: int = 144
    validation_tolerance_pct: float = 0.15
    window_tolerance_pct: float = 0.10
    min_validation_events: int = 12
    validation_min_confidence: float = 65.0
    quick_min_rr1: float = 1.50
    require_final_eod: bool = True
    lunar_enabled: bool = True
    eoff_enabled: bool = True
    eoff_ephemeris_enabled: bool = True
    eoff_min_fib_cluster: int = 4
    eoff_aspect_orb_deg: float = 3.0
    eoff_require_astro_fib_confluence: bool = True


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _clean_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    rename = {str(c).lower(): c for c in out.columns}
    mapping: dict[Any, str] = {}
    for canonical in ("Open", "High", "Low", "Close", "Volume"):
        original = rename.get(canonical.lower())
        if original is not None:
            mapping[original] = canonical
    out = out.rename(columns=mapping)
    required = {"High", "Low", "Close"}
    if not required.issubset(out.columns):
        return pd.DataFrame()
    for column in required | ({"Open", "Volume"} & set(out.columns)):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()]
    if out.index.tz is not None:
        out.index = out.index.tz_convert("Asia/Jakarta").tz_localize(None)
    return out


def _confirmed_pivot_positions(values: np.ndarray, left: int, right: int, mode: str) -> list[int]:
    positions: list[int] = []
    if len(values) < left + right + 3:
        return positions
    for i in range(left, len(values) - right):
        window = values[i - left:i + right + 1]
        value = values[i]
        if not np.isfinite(value) or not np.isfinite(window).all():
            continue
        if mode == "high":
            condition = value >= float(np.max(window)) and value > float(np.max(values[i-left:i])) and value >= float(np.max(values[i+1:i+right+1]))
        else:
            condition = value <= float(np.min(window)) and value < float(np.min(values[i-left:i])) and value <= float(np.min(values[i+1:i+right+1]))
        if condition:
            positions.append(i)
    return positions


def _filtered_intervals(positions: list[int], minimum: int, maximum: int) -> np.ndarray:
    if len(positions) < 2:
        return np.array([], dtype=float)
    values = np.diff(np.asarray(positions, dtype=float))
    return values[(values >= minimum) & (values <= maximum)]


def _chronological_same_type_intervals(
    high_positions: list[int],
    low_positions: list[int],
    minimum: int,
    maximum: int,
) -> np.ndarray:
    """Return high-to-high and low-to-low intervals in event-time order.

    Concatenating all high intervals before all low intervals leaks the pivot
    type into the validation order.  Each interval is therefore timestamped by
    its ending pivot and the two same-type streams are merged chronologically.
    """
    events: list[tuple[int, float]] = []
    for positions in (high_positions, low_positions):
        for previous, current in zip(positions, positions[1:]):
            interval = float(current - previous)
            if minimum <= interval <= maximum:
                events.append((int(current), interval))
    events.sort(key=lambda item: (item[0], item[1]))
    return np.asarray([interval for _, interval in events], dtype=float)


def _robust_cycle(intervals: np.ndarray) -> float:
    if intervals.size == 0:
        return np.nan
    recent = intervals[-10:]
    median = float(np.median(recent))
    mad = float(np.median(np.abs(recent - median)))
    if mad > 0:
        recent = recent[np.abs(recent - median) <= 2.5 * mad]
    if recent.size == 0:
        return median
    weights = np.linspace(0.65, 1.0, recent.size)
    return float(np.average(recent, weights=weights))


def _chronological_cycle_validation(intervals: np.ndarray, tolerance_pct: float) -> tuple[float, int, float]:
    if intervals.size < 5:
        return (np.nan, 0, np.nan)
    hits: list[float] = []
    errors: list[float] = []
    for i in range(4, intervals.size):
        history = intervals[max(0, i - 8):i]
        prediction = float(np.median(history))
        tolerance = max(2.0, prediction * tolerance_pct)
        error = abs(float(intervals[i]) - prediction)
        hits.append(float(error <= tolerance))
        errors.append(error / max(prediction, 1.0))
    return (100.0 * float(np.mean(hits)), len(hits), 100.0 * float(np.median(errors)))


def _autocorrelation_period(close: pd.Series, minimum: int, maximum: int) -> tuple[float, float]:
    values = np.log(pd.to_numeric(close, errors="coerce").replace(0, np.nan)).dropna().to_numpy(dtype=float)
    if values.size < max(100, 3 * minimum):
        return (np.nan, np.nan)
    x = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    detrended = values - (slope * x + intercept)
    best_period, best_corr = np.nan, -1.0
    max_lag = min(maximum, values.size // 3)
    for lag in range(minimum, max_lag + 1):
        left, right = detrended[:-lag], detrended[lag:]
        if left.size < 40 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            continue
        corr = float(np.corrcoef(left, right)[0, 1])
        if np.isfinite(corr) and corr > best_corr:
            best_period, best_corr = float(lag), corr
    if best_corr < 0.12:
        return (np.nan, best_corr)
    return (best_period, best_corr)


def _spectral_period(close: pd.Series, minimum: int, maximum: int) -> tuple[float, float]:
    values = np.log(pd.to_numeric(close, errors="coerce").replace(0, np.nan)).dropna().to_numpy(dtype=float)
    if values.size < max(128, 3 * minimum):
        return (np.nan, np.nan)
    values = values[-min(512, values.size):]
    x = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    detrended = values - (slope * x + intercept)
    detrended = detrended * np.hanning(values.size)
    spectrum = np.abs(np.fft.rfft(detrended)) ** 2
    freqs = np.fft.rfftfreq(values.size, d=1.0)
    candidates: list[tuple[float, float]] = []
    total_power = float(np.sum(spectrum[1:]))
    for i in range(1, len(freqs)):
        if freqs[i] <= 0:
            continue
        period = 1.0 / freqs[i]
        if minimum <= period <= maximum:
            candidates.append((float(spectrum[i]), float(period)))
    if not candidates or total_power <= 0:
        return (np.nan, np.nan)
    power, period = max(candidates, key=lambda item: item[0])
    return (period, 100.0 * power / total_power)


def _weighted_consensus(values: list[tuple[float, float]]) -> tuple[float, float]:
    usable = [(v, w) for v, w in values if np.isfinite(v) and v > 0 and w > 0]
    if not usable:
        return (np.nan, 0.0)
    array = np.asarray([v for v, _ in usable], dtype=float)
    weights = np.asarray([w for _, w in usable], dtype=float)
    consensus = float(np.average(array, weights=weights))
    dispersion = float(np.sqrt(np.average((array - consensus) ** 2, weights=weights)) / max(consensus, 1.0))
    agreement = max(0.0, min(100.0, 100.0 * (1.0 - dispersion / 0.35)))
    return (consensus, agreement)


def _proximity(age: float, target: float, tolerance_pct: float = 0.12) -> float:
    if not np.isfinite(age) or not np.isfinite(target) or target <= 0:
        return 0.0
    tolerance = max(2.0, target * tolerance_pct)
    return float(max(0.0, 100.0 * (1.0 - abs(age - target) / (2.0 * tolerance))))


def _nearest_fibonacci(age: float) -> tuple[float, float]:
    if not np.isfinite(age):
        return (np.nan, 0.0)
    target = min(FIBONACCI_BARS, key=lambda value: abs(float(age) - value))
    tolerance = max(1.0, 0.08 * target)
    score = max(0.0, 100.0 * (1.0 - abs(float(age) - target) / (2.0 * tolerance)))
    return (float(target), float(score))


def _lunar_phase(timestamp: pd.Timestamp) -> dict[str, float | str]:
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Jakarta")
    utc = stamp.tz_convert("UTC")
    days = (utc - _NEW_MOON_EPOCH_UTC).total_seconds() / 86400.0
    phase = (days % SYNODIC_MONTH_DAYS) / SYNODIC_MONTH_DAYS
    phase_days = phase * SYNODIC_MONTH_DAYS
    markers = {
        "NEW_MOON": 0.0,
        "FIRST_QUARTER": SYNODIC_MONTH_DAYS / 4.0,
        "FULL_MOON": SYNODIC_MONTH_DAYS / 2.0,
        "LAST_QUARTER": 3.0 * SYNODIC_MONTH_DAYS / 4.0,
    }
    distances = {
        name: min(abs(phase_days - marker), SYNODIC_MONTH_DAYS - abs(phase_days - marker))
        for name, marker in markers.items()
    }
    nearest = min(distances, key=distances.get)
    turn_distance = min(distances["NEW_MOON"], distances["FULL_MOON"])
    return {
        "lunar_phase": nearest,
        "lunar_phase_fraction": phase,
        "lunar_days_to_major_marker": float(turn_distance),
        "lunar_marker_score": float(max(0.0, 100.0 * (1.0 - turn_distance / 3.0))),
    }


def _lunar_pivot_validation(index: pd.DatetimeIndex, pivots: list[int]) -> tuple[float, float, int]:
    if len(pivots) < 8:
        return (np.nan, np.nan, 0)
    distances = []
    for position in pivots:
        distances.append(float(_lunar_phase(index[position])["lunar_days_to_major_marker"]))
    hit_rate = 100.0 * float(np.mean(np.asarray(distances) <= 2.0))
    baseline = 100.0 * min(1.0, 4.0 / (SYNODIC_MONTH_DAYS / 2.0))
    lift = hit_rate / baseline if baseline > 0 else np.nan
    return (hit_rate, lift, len(distances))


def _business_window(last_date: pd.Timestamp, bars_to_target: float, tolerance_bars: int) -> tuple[str, str]:
    base = pd.Timestamp(last_date).normalize()
    target = base + pd.offsets.BDay(max(0, int(round(bars_to_target))))
    start = target - pd.offsets.BDay(tolerance_bars)
    end = target + pd.offsets.BDay(tolerance_bars)
    return (start.date().isoformat(), end.date().isoformat())


def _idx_tick_size(price: float) -> int:
    """Local IDX regular-market price fraction helper.

    Kept local to avoid importing scanner.py and creating a circular import.
    """
    value = _finite(price, 0.0)
    if value < 200.0:
        return 1
    if value < 500.0:
        return 2
    if value < 2000.0:
        return 5
    if value < 5000.0:
        return 10
    return 25


def _round_idx_price(price: Any, direction: str = "nearest") -> float:
    value = _finite(price, np.nan)
    if not np.isfinite(value) or value <= 0:
        return np.nan
    tick = _idx_tick_size(value)
    scaled = value / tick
    if direction == "up":
        rounded = math.ceil(scaled - 1e-12) * tick
    elif direction == "down":
        rounded = math.floor(scaled + 1e-12) * tick
    else:
        rounded = round(scaled) * tick
    tick2 = _idx_tick_size(float(rounded))
    scaled2 = rounded / tick2
    if direction == "up":
        rounded = math.ceil(scaled2 - 1e-12) * tick2
    elif direction == "down":
        rounded = math.floor(scaled2 + 1e-12) * tick2
    else:
        rounded = round(scaled2) * tick2
    return float(max(tick2, rounded))


def _parse_business_date(value: Any) -> pd.Timestamp | None:
    try:
        parsed = pd.Timestamp(value).normalize()
        return None if pd.isna(parsed) else parsed
    except Exception:
        return None


def _window_midpoint(start: str, end: str) -> pd.Timestamp | None:
    start_ts = _parse_business_date(start)
    end_ts = _parse_business_date(end)
    if start_ts is None or end_ts is None:
        return None
    days = pd.bdate_range(start_ts, end_ts)
    if len(days) == 0:
        return start_ts
    return pd.Timestamp(days[(len(days) - 1) // 2]).normalize()


def _build_quick_buy_decision(
    df: pd.DataFrame,
    *,
    direction: str,
    phase: str,
    time_state: str,
    time_score: float,
    confidence: float,
    bullish_timing: float,
    continuation_timing: float,
    price_time: float,
    trend_score: float,
    window_start: str,
    window_end: str,
    bars_due: float,
    eoff: Mapping[str, Any],
    min_rr1: float = 1.50,
    require_final_eod: bool = True,
) -> dict[str, Any]:
    """Collapse the full time-cycle stack into an actionable daily decision.

    The projected date is a *conditional execution date*, not a guarantee.  A
    buy remains valid only when price trades in the structural zone and then
    confirms through the trigger without violating the stop.
    """
    unavailable = {
        "quick_buy_state": "NO_VALID_BUY_DATE",
        "quick_buy_action": "WAIT",
        "best_buy_date": "",
        "best_buy_date_basis": "",
        "best_buy_window_start": window_start,
        "best_buy_window_end": window_end,
        "best_buy_score": 0.0,
        "best_buy_confidence": 0.0,
        "best_buy_entry_low": np.nan,
        "best_buy_entry_high": np.nan,
        "best_buy_trigger": np.nan,
        "best_buy_stop_loss": np.nan,
        "best_buy_tp1": np.nan,
        "best_buy_tp2": np.nan,
        "best_buy_rr1": np.nan,
        "best_buy_rr2": np.nan,
        "best_buy_target_basis": "",
        "best_buy_order_plan": "NO_ORDER",
        "best_buy_reason": "Belum ada tanggal beli yang memenuhi bukti time-cycle dan struktur harga.",
        "best_buy_no_trade_condition": "Jangan membeli tanpa konfirmasi struktur harga.",
        "best_buy_summary": "WAIT — belum ada tanggal beli tervalidasi.",
    }
    if df is None or df.empty or time_state == "INSUFFICIENT_HISTORY":
        return unavailable

    last_date = pd.Timestamp(df.index[-1]).normalize()
    close = _finite(df["Close"].iloc[-1], np.nan)
    high = _finite(df["High"].iloc[-1], np.nan)
    low = _finite(df["Low"].iloc[-1], np.nan)
    if not all(np.isfinite(value) and value > 0 for value in (close, high, low)):
        return unavailable

    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = _finite(true_range.rolling(14, min_periods=5).mean().iloc[-1], max(high - low, _idx_tick_size(close) * 3.0))
    atr = max(atr, _idx_tick_size(close) * 3.0)
    ema20 = _finite(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1], close)
    ema50 = _finite(df["Close"].ewm(span=50, adjust=False).mean().iloc[-1], close)

    prior = df.iloc[:-1]
    prior_high_3 = _finite(prior["High"].tail(3).max(), high)
    prior_high_5 = _finite(prior["High"].tail(5).max(), prior_high_3)
    recent_low_10 = _finite(df["Low"].tail(10).min(), low)
    recent_low_20 = _finite(df["Low"].tail(20).min(), recent_low_10)
    recent_low_55 = _finite(df["Low"].tail(55).min(), recent_low_20)
    recent_high_20 = _finite(df["High"].tail(20).max(), high)
    recent_high_55 = _finite(df["High"].tail(55).max(), recent_high_20)
    recent_high_120 = _finite(df["High"].tail(120).max(), recent_high_55)

    candidate_date = _window_midpoint(window_start, window_end)
    date_basis = "TIME_CYCLE_WINDOW_CENTER"
    eoff_date = _parse_business_date(eoff.get("eoff_reversal_date"))
    window_start_ts = _parse_business_date(window_start)
    window_end_ts = _parse_business_date(window_end)
    eoff_bullish = bool(
        bool(eoff.get("eoff_signal_active"))
        and str(eoff.get("eoff_direction_bias") or "NEUTRAL").upper() == "BULLISH"
    )
    if (
        eoff_date is not None and window_start_ts is not None and window_end_ts is not None
        and window_start_ts <= eoff_date <= window_end_ts and eoff_bullish
        and bool(eoff.get("eoff_signal_active"))
    ):
        candidate_date = eoff_date
        date_basis = "EOFF_REVERSAL_DATE_WITHIN_CYCLE_WINDOW"

    phase_score = {
        "LATE_CORRECTION": 96.0,
        "CORRECTION": 78.0,
        "TRANSITION": 68.0,
        "EXPANSION": 72.0 if continuation_timing >= 70.0 else 55.0,
        "LATE_EXPANSION": 25.0,
    }.get(str(phase).upper(), 50.0)
    direction_score = max(bullish_timing, continuation_timing)
    eoff_score = _finite(eoff.get("eoff_reconstruction_score"), 50.0)
    if not eoff_bullish:
        eoff_score = min(50.0, eoff_score)
    buy_score = (
        0.27 * direction_score
        + 0.22 * price_time
        + 0.18 * confidence
        + 0.14 * phase_score
        + 0.10 * eoff_score
        + 0.09 * trend_score
    )
    buy_score = float(max(0.0, min(100.0, buy_score)))

    historical_hit = _finite(eoff.get("eoff_public_forward_hit_rate"), np.nan)
    if not np.isfinite(historical_hit):
        historical_hit = _finite(eoff.get("eoff_public_reversal_hit_rate"), np.nan)
    evidence_conf = confidence if not np.isfinite(historical_hit) else 0.72 * confidence + 0.28 * historical_hit
    buy_confidence = float(max(0.0, min(100.0, evidence_conf)))

    # Structural price zone: nearest support below/near current price.  The
    # confirmation trigger avoids blindly buying a date without price action.
    support_candidates = [value for value in (recent_low_10, recent_low_20, ema20, ema50) if np.isfinite(value) and value <= close * 1.015]
    support = max(support_candidates) if support_candidates else min(close, recent_low_20)
    if str(phase).upper() in {"EXPANSION", "TRANSITION"} and continuation_timing >= bullish_timing:
        support = max([value for value in (ema20, recent_low_10, recent_low_20) if np.isfinite(value) and value <= close * 1.02] or [support])
    entry_low = _round_idx_price(max(recent_low_55, support - 0.30 * atr), "down")
    entry_high = _round_idx_price(min(close * 1.02, support + 0.25 * atr), "up")
    if np.isfinite(entry_low) and np.isfinite(entry_high) and entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low

    # The trigger must be known *before* the latest candle.  Using the current
    # candle high plus one tick made close >= trigger mathematically impossible.
    trigger_base = prior_high_3 if str(phase).upper() in {"LATE_CORRECTION", "CORRECTION"} else prior_high_5
    trigger = _round_idx_price(trigger_base + _idx_tick_size(trigger_base), "up")
    stop_reference = min(recent_low_20, entry_low - 0.55 * atr if np.isfinite(entry_low) else recent_low_20)
    stop = _round_idx_price(stop_reference - _idx_tick_size(stop_reference), "down")
    if np.isfinite(trigger) and np.isfinite(stop) and trigger - stop < 0.80 * atr:
        stop = _round_idx_price(trigger - max(0.80 * atr, 3.0 * _idx_tick_size(trigger)), "down")

    target_candidates: list[float] = []
    for value in (recent_high_20, recent_high_55, recent_high_120):
        if np.isfinite(value) and np.isfinite(trigger) and value > trigger + 2.0 * _idx_tick_size(trigger):
            target_candidates.append(_round_idx_price(value, "down"))
    try:
        import json as _json
        roadmap = _json.loads(str(eoff.get("eoff_roadmap_json") or "[]"))
    except Exception:
        roadmap = []
    for row in roadmap if isinstance(roadmap, list) else []:
        for key in ("price_zone_low", "price_zone_high"):
            value = _finite(row.get(key) if isinstance(row, Mapping) else np.nan, np.nan)
            if np.isfinite(value) and np.isfinite(trigger) and value > trigger + 2.0 * _idx_tick_size(trigger):
                target_candidates.append(_round_idx_price(value, "down"))
    risk = trigger - stop if np.isfinite(trigger) and np.isfinite(stop) else np.nan
    target_candidates = sorted({float(v) for v in target_candidates if np.isfinite(v)})
    structural_targets = [value for value in target_candidates if np.isfinite(risk) and value >= trigger + min_rr1 * risk]
    target_basis = "PRICE_STRUCTURE"
    if structural_targets:
        tp1 = structural_targets[0]
        tp2 = structural_targets[1] if len(structural_targets) > 1 else _round_idx_price(trigger + 2.0 * risk, "up")
        if len(structural_targets) == 1:
            target_basis = "PRICE_STRUCTURE_PLUS_RISK_EXTENSION"
    elif np.isfinite(risk) and risk > 0:
        # A transparent risk extension prevents a half-built execution plan.
        # It is not treated as structural resistance and is labelled as such.
        tp1 = _round_idx_price(trigger + min_rr1 * risk, "up")
        tp2 = _round_idx_price(trigger + 2.0 * risk, "up")
        target_basis = "RISK_EXTENSION_NO_NEAR_RESISTANCE"
    else:
        tp1, tp2 = np.nan, np.nan
        target_basis = "UNAVAILABLE"
    rr1 = (tp1 - trigger) / risk if np.isfinite(tp1) and np.isfinite(risk) and risk > 0 else np.nan
    rr2 = (tp2 - trigger) / risk if np.isfinite(tp2) and np.isfinite(risk) and risk > 0 else np.nan

    direction_name = str(direction).upper()
    if direction_name == "BEARISH_REVERSAL_RISK":
        result = dict(unavailable)
        result.update({
            "quick_buy_state": "NO_BUY_BEARISH_WINDOW",
            "quick_buy_action": "AVOID_NEW_BUY",
            "best_buy_score": round(buy_score, 1),
            "best_buy_confidence": round(buy_confidence, 1),
            "best_buy_window_start": window_start,
            "best_buy_window_end": window_end,
            "best_buy_reason": "Time-cycle lebih kuat menunjukkan risiko puncak/koreksi daripada jendela beli.",
            "best_buy_summary": "AVOID — jendela reversal saat ini condong bearish.",
        })
        return result

    if candidate_date is None or buy_confidence < 50.0 or buy_score < 58.0:
        result = dict(unavailable)
        result.update({
            "quick_buy_state": "INSUFFICIENT_BUY_EVIDENCE",
            "quick_buy_action": "WAIT_FOR_EVIDENCE",
            "best_buy_score": round(buy_score, 1),
            "best_buy_confidence": round(buy_confidence, 1),
            "best_buy_window_start": window_start,
            "best_buy_window_end": window_end,
            "best_buy_reason": "Skor atau confidence belum cukup untuk menyimpulkan tanggal beli.",
            "best_buy_summary": f"WAIT — buy score {buy_score:.1f}, confidence {buy_confidence:.1f}%.",
        })
        return result

    plan_valid = bool(
        all(np.isfinite(value) and value > 0 for value in (entry_low, entry_high, trigger, stop, tp1, rr1))
        and entry_low <= entry_high
        and trigger > stop
        and tp1 > trigger
        and rr1 >= max(0.0, float(min_rr1)) - 1e-9
    )
    if not plan_valid:
        result = dict(unavailable)
        result.update({
            "quick_buy_state": "INVALID_TRADE_PLAN",
            "quick_buy_action": "WAIT_FOR_EVIDENCE",
            "best_buy_date": candidate_date.date().isoformat(),
            "best_buy_date_basis": date_basis,
            "best_buy_score": round(buy_score, 1),
            "best_buy_confidence": round(buy_confidence, 1),
            "best_buy_entry_low": entry_low,
            "best_buy_entry_high": entry_high,
            "best_buy_trigger": trigger,
            "best_buy_stop_loss": stop,
            "best_buy_tp1": tp1,
            "best_buy_tp2": tp2,
            "best_buy_rr1": round(rr1, 2) if np.isfinite(rr1) else np.nan,
            "best_buy_rr2": round(rr2, 2) if np.isfinite(rr2) else np.nan,
            "best_buy_target_basis": target_basis,
            "best_buy_reason": "Rencana harga belum lengkap atau RR minimum belum terpenuhi; tidak ada order yang boleh disiapkan.",
            "best_buy_summary": "WAIT — trade plan tidak valid.",
        })
        return result

    if str(time_state).upper() != "VALIDATED":
        result = dict(unavailable)
        result.update({
            "quick_buy_state": "LIMITED_EVIDENCE",
            "quick_buy_action": "WAIT_FOR_EVIDENCE",
            "best_buy_date": candidate_date.date().isoformat(),
            "best_buy_date_basis": date_basis,
            "best_buy_score": round(buy_score, 1),
            "best_buy_confidence": round(buy_confidence, 1),
            "best_buy_entry_low": entry_low,
            "best_buy_entry_high": entry_high,
            "best_buy_trigger": trigger,
            "best_buy_stop_loss": stop,
            "best_buy_tp1": tp1,
            "best_buy_tp2": tp2,
            "best_buy_rr1": round(rr1, 2),
            "best_buy_rr2": round(rr2, 2) if np.isfinite(rr2) else np.nan,
            "best_buy_target_basis": target_basis,
            "best_buy_reason": "Time-Cycle masih LIMITED_EVIDENCE; level ditampilkan untuk observasi, bukan order.",
            "best_buy_summary": "WAIT — Time-Cycle belum tervalidasi.",
        })
        return result

    bar_state = str(df.attrs.get("bar_state") or "").upper()
    now_jakarta = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).normalize()
    final_eod = bool(bar_state == "FINAL_EOD" or last_date < now_jakarta)
    in_window = bool(window_start_ts is not None and window_end_ts is not None and window_start_ts <= last_date <= window_end_ts)
    before_window = bool(window_start_ts is not None and last_date < window_start_ts)
    after_window = bool(window_end_ts is not None and last_date > window_end_ts)
    price_in_zone = bool(np.isfinite(entry_low) and np.isfinite(entry_high) and entry_low <= close <= entry_high)
    price_confirmed = bool(np.isfinite(trigger) and high >= trigger and close >= trigger)

    if after_window:
        action = "RECALCULATE_WINDOW"
        state = "BUY_WINDOW_EXPIRED"
    elif before_window:
        action = "WAIT_FOR_DATE"
        state = "PROJECTED_BUY_DATE"
    elif in_window and price_confirmed and (final_eod or not require_final_eod):
        action = "BUY_ON_CONFIRMED_TRIGGER"
        state = "BUY_WINDOW_CONFIRMED"
    elif in_window and price_confirmed:
        action = "WAIT_FINAL_EOD_CONFIRMATION"
        state = "TRIGGER_TOUCHED_PARTIAL_CANDLE"
    elif in_window and price_in_zone:
        action = "PREPARE_BUY_WAIT_TRIGGER"
        state = "BUY_ZONE_ACTIVE"
    elif in_window:
        action = "WAIT_FOR_PRICE_ZONE_OR_TRIGGER"
        state = "BUY_WINDOW_OPEN"
    else:
        action = "WAIT_FOR_DATE"
        state = "PROJECTED_BUY_DATE"

    order_plan = "BUY_LIMIT_IN_ZONE_THEN_BUY_STOP_ON_TRIGGER"
    if str(phase).upper() in {"EXPANSION", "TRANSITION"} and continuation_timing >= bullish_timing:
        order_plan = "BUY_ON_RETEST_OR_BUY_STOP_CONFIRMATION"
    date_text = candidate_date.date().isoformat()
    zone_text = f"Rp {entry_low:,.0f}–{entry_high:,.0f}" if np.isfinite(entry_low) and np.isfinite(entry_high) else "N/A"
    trigger_text = f"Rp {trigger:,.0f}" if np.isfinite(trigger) else "N/A"
    summary = f"{action} | tanggal utama {date_text} | zona {zone_text} | trigger {trigger_text}"
    reason = (
        f"Kesimpulan gabungan phase {phase}, direction {direction}, time score {time_score:.1f}, "
        f"price-time {price_time:.1f}, confidence {buy_confidence:.1f}%, dan EOFF "
        f"{str(eoff.get('eoff_strength_label') or 'LOW')}."
    )
    no_trade = (
        f"Batalkan bila daily close di bawah Rp {stop:,.0f}, arah berubah bearish, atau trigger tidak terjadi sampai "
        f"akhir window {window_end}."
        if np.isfinite(stop) else
        f"Batalkan bila arah berubah bearish atau trigger tidak terjadi sampai akhir window {window_end}."
    )
    return {
        "quick_buy_state": state,
        "quick_buy_action": action,
        "best_buy_date": date_text,
        "best_buy_date_basis": date_basis,
        "best_buy_window_start": window_start,
        "best_buy_window_end": window_end,
        "best_buy_score": round(buy_score, 1),
        "best_buy_confidence": round(buy_confidence, 1),
        "best_buy_entry_low": entry_low,
        "best_buy_entry_high": entry_high,
        "best_buy_trigger": trigger,
        "best_buy_stop_loss": stop,
        "best_buy_tp1": tp1,
        "best_buy_tp2": tp2,
        "best_buy_rr1": round(rr1, 2) if np.isfinite(rr1) else np.nan,
        "best_buy_rr2": round(rr2, 2) if np.isfinite(rr2) else np.nan,
        "best_buy_target_basis": target_basis,
        "best_buy_order_plan": order_plan,
        "best_buy_reason": reason,
        "best_buy_no_trade_condition": no_trade,
        "best_buy_summary": summary,
    }


def analyze_time_cycle(frame: pd.DataFrame | None, config: TimeCycleConfig | None = None) -> dict[str, Any]:
    cfg = config or TimeCycleConfig()
    df = _clean_frame(frame)
    unavailable = {
        "time_cycle_version": TIME_CYCLE_VERSION,
        "time_cycle_state": "INSUFFICIENT_HISTORY",
        "time_cycle_score": np.nan,
        "time_cycle_confidence": 0.0,
        "time_cycle_direction_bias": "NEUTRAL",
        "time_cycle_phase": "UNKNOWN",
        "dominant_cycle_bars": np.nan,
        "pivot_cycle_bars": np.nan,
        "autocorr_cycle_bars": np.nan,
        "spectral_cycle_bars": np.nan,
        "cycle_agreement_score": 0.0,
        "cycle_historical_hit_rate": np.nan,
        "cycle_validation_samples": 0,
        "cycle_median_error_pct": np.nan,
        "bullish_timing_score": 50.0,
        "continuation_timing_score": 50.0,
        "bearish_timing_score": 50.0,
        "price_time_confluence_score": 50.0,
        "fibonacci_time_score": 0.0,
        "fibonacci_low_target": np.nan,
        "fibonacci_high_target": np.nan,
        "cycle_age_from_low": np.nan,
        "cycle_age_from_high": np.nan,
        "bars_to_reversal_window": np.nan,
        "next_reversal_window_start": "",
        "next_reversal_window_end": "",
        "lunar_phase": "UNKNOWN",
        "lunar_days_to_major_marker": np.nan,
        "lunar_marker_score": 0.0,
        "lunar_historical_hit_rate": np.nan,
        "lunar_historical_lift": np.nan,
        "lunar_validation_samples": 0,
        "eoff_version": EOFF_VERSION,
        "eoff_state": "DISABLED" if not cfg.eoff_enabled else "INSUFFICIENT_HISTORY",
        "eoff_reconstruction_score": np.nan,
        "eoff_strength_label": "LOW",
        "eoff_signal_active": False,
        "eoff_direction_bias": "NEUTRAL",
        "eoff_time_power_score": 0.0,
        "eoff_price_power_score": 0.0,
        "eoff_pattern_score": 0.0,
        "eoff_momentum_score": 0.0,
        "eoff_astro_score": 0.0,
        "eoff_core_astro_score": 0.0,
        "eoff_adaptive_astro_score": 50.0,
        "eoff_adaptive_total_weight_pct": 50.0,
        "eoff_secondary_prior_share_pct": 50.0,
        "eoff_adaptive_active_factors": "",
        "eoff_adaptive_validation_state": "PUBLIC_PRIOR_NO_OOS_MODULATION",
        "eoff_astro_weight_policy": "PUBLIC_RECONSTRUCTION_PRIOR_WITH_OOS_MODULATION",
        "eoff_astro_prior_weights_json": "{\"INGRESS\": 0.1, \"MOON_DECLINATION\": 0.15, \"MOON_PHASE\": 0.25, \"PLANETARY_ASPECT\": 0.25, \"RETROGRADE\": 0.1, \"SUN_ANNUAL\": 0.15}",
        "eoff_phase_base_weight_pct": 25.0,
        "eoff_aspect_base_weight_pct": 25.0,
        "eoff_declination_base_weight_pct": 15.0,
        "eoff_ingress_base_weight_pct": 10.0,
        "eoff_retrograde_base_weight_pct": 10.0,
        "eoff_sun_base_weight_pct": 15.0,
        "eoff_phase_weight_pct": 25.0,
        "eoff_aspect_weight_pct": 25.0,
        "eoff_validation_path": "NONE",
        "eoff_fib_cluster_count": 0,
        "eoff_fib_unique_anchor_count": 0,
        "eoff_historical_hit_rate": np.nan,
        "eoff_historical_baseline_rate": np.nan,
        "eoff_historical_lift": np.nan,
        "eoff_confluence_historical_hit_rate": np.nan,
        "eoff_confluence_historical_events": 0,
        "eoff_confluence_historical_lift": np.nan,
        "eoff_public_validation_state": "INSUFFICIENT_EVENTS",
        "eoff_public_validation_method": "PUBLIC_FIXED_FAMILY_CHRONOLOGICAL_FORWARD_TEST",
        "eoff_public_directional_events": 0,
        "eoff_public_reversal_hit_rate": np.nan,
        "eoff_public_baseline_rate": np.nan,
        "eoff_public_lift": np.nan,
        "eoff_public_forward_hit_rate": np.nan,
        "eoff_public_median_directional_return_pct": np.nan,
        "eoff_sun_validation_state": "SHADOW_INSUFFICIENT_OOS",
        "eoff_sun_oos_events": 0,
        "eoff_sun_oos_lift": np.nan,
        "eoff_sun_oos_forward_hit_rate": np.nan,
        "eoff_sun_oos_median_return_pct": np.nan,
        "eoff_sun_weight_pct": 15.0,
        "eoff_sun_current_active": False,
        "eoff_sun_current_score": 50.0,
        "eoff_retrograde_validation_state": "SHADOW_INSUFFICIENT_OOS",
        "eoff_retrograde_oos_events": 0,
        "eoff_retrograde_oos_lift": np.nan,
        "eoff_retrograde_oos_forward_hit_rate": np.nan,
        "eoff_retrograde_oos_median_return_pct": np.nan,
        "eoff_retrograde_weight_pct": 10.0,
        "eoff_retrograde_current_active": False,
        "eoff_retrograde_current_score": 50.0,
        "eoff_ingress_validation_state": "SHADOW_INSUFFICIENT_OOS",
        "eoff_ingress_oos_events": 0,
        "eoff_ingress_oos_lift": np.nan,
        "eoff_ingress_oos_forward_hit_rate": np.nan,
        "eoff_ingress_oos_median_return_pct": np.nan,
        "eoff_ingress_weight_pct": 10.0,
        "eoff_ingress_current_active": False,
        "eoff_ingress_current_score": 50.0,
        "eoff_declination_validation_state": "SHADOW_INSUFFICIENT_OOS",
        "eoff_declination_oos_events": 0,
        "eoff_declination_oos_lift": np.nan,
        "eoff_declination_oos_forward_hit_rate": np.nan,
        "eoff_declination_oos_median_return_pct": np.nan,
        "eoff_declination_weight_pct": 15.0,
        "eoff_declination_current_active": False,
        "eoff_declination_current_score": 50.0,
        "eoff_historical_events": 0,
        "eoff_bars_to_cluster": np.nan,
        "eoff_reversal_date": "",
        "eoff_ephemeris_state": "UNAVAILABLE",
        "eoff_ephemeris_date": "",
        "eoff_astro_events": "",
        "eoff_active_aspects": "",
        "eoff_retrograde_planets": "",
        "eoff_retrograde_transition_events": "",
        "eoff_stationary_planets": "",
        "eoff_ingress_events": "",
        "eoff_moon_declination_deg": np.nan,
        "eoff_moon_declination_extreme_score": 0.0,
        "eoff_moon_phase": "UNKNOWN",
        "eoff_sun_sign": "UNKNOWN",
        "eoff_sun_annual_cycle_bias": "NEUTRAL",
        "eoff_roadmap_json": "[]",
        "eoff_internal_weight_pct": 0.0,
        "eoff_explanation": "Riwayat belum cukup untuk clean-room EOFF.",
        "time_cycle_explanation": "Riwayat belum cukup untuk time-cycle harian.",
        "quick_buy_state": "NO_VALID_BUY_DATE",
        "quick_buy_action": "WAIT",
        "best_buy_date": "",
        "best_buy_date_basis": "",
        "best_buy_window_start": "",
        "best_buy_window_end": "",
        "best_buy_score": 0.0,
        "best_buy_confidence": 0.0,
        "best_buy_entry_low": np.nan,
        "best_buy_entry_high": np.nan,
        "best_buy_trigger": np.nan,
        "best_buy_stop_loss": np.nan,
        "best_buy_tp1": np.nan,
        "best_buy_tp2": np.nan,
        "best_buy_rr1": np.nan,
        "best_buy_rr2": np.nan,
        "best_buy_target_basis": "",
        "best_buy_order_plan": "NO_ORDER",
        "best_buy_reason": "Riwayat belum cukup untuk menyimpulkan tanggal beli.",
        "best_buy_no_trade_condition": "Jangan membeli tanpa konfirmasi struktur harga.",
        "best_buy_summary": "WAIT — belum ada tanggal beli tervalidasi.",
    }
    if len(df) < cfg.min_bars:
        return unavailable

    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    high_pos = _confirmed_pivot_positions(highs, cfg.pivot_left, cfg.pivot_right, "high")
    low_pos = _confirmed_pivot_positions(lows, cfg.pivot_left, cfg.pivot_right, "low")
    combined = _chronological_same_type_intervals(high_pos, low_pos, cfg.min_period, cfg.max_period)

    pivot_cycle = _robust_cycle(combined)
    pivot_hit, pivot_samples, pivot_error = _chronological_cycle_validation(combined, cfg.validation_tolerance_pct)
    acf_period, acf_strength = _autocorrelation_period(df["Close"], cfg.min_period, cfg.max_period)
    spectral_period, spectral_power = _spectral_period(df["Close"], cfg.min_period, cfg.max_period)
    dominant, agreement = _weighted_consensus([
        (pivot_cycle, 0.50),
        (acf_period, max(0.05, _finite(acf_strength, 0.0)) * 0.30),
        (spectral_period, max(0.05, _finite(spectral_power, 0.0) / 100.0) * 0.20),
    ])
    if not np.isfinite(dominant):
        result = dict(unavailable)
        result["time_cycle_state"] = "NO_STABLE_CYCLE"
        result["time_cycle_explanation"] = "Swing, autocorrelation, dan spektrum belum menghasilkan cycle stabil."
        # EOFF may still provide a shadow diagnostic, but no scanner influence
        # is allowed without a stable objective cycle.
        result.update(analyze_eoff_reconstruction(
            df, config=EOFFConfig(
                enabled=cfg.eoff_enabled,
                ephemeris_enabled=cfg.eoff_ephemeris_enabled,
                min_fib_cluster=cfg.eoff_min_fib_cluster,
                aspect_orb_deg=cfg.eoff_aspect_orb_deg,
                require_astro_fib_confluence=cfg.eoff_require_astro_fib_confluence,
            ),
        ))
        result["eoff_internal_weight_pct"] = 0.0
        return result

    eoff = analyze_eoff_reconstruction(
        df, config=EOFFConfig(
            enabled=cfg.eoff_enabled,
            ephemeris_enabled=cfg.eoff_ephemeris_enabled,
            min_fib_cluster=cfg.eoff_min_fib_cluster,
            aspect_orb_deg=cfg.eoff_aspect_orb_deg,
            require_astro_fib_confluence=cfg.eoff_require_astro_fib_confluence,
        ),
    )

    last_pos = len(df) - 1
    last_low = low_pos[-1] if low_pos else None
    last_high = high_pos[-1] if high_pos else None
    age_low = float(last_pos - last_low) if last_low is not None else np.nan
    age_high = float(last_pos - last_high) if last_high is not None else np.nan
    low_due = _proximity(age_low, dominant)
    high_due = _proximity(age_high, dominant)
    fib_low_target, fib_low = _nearest_fibonacci(age_low)
    fib_high_target, fib_high = _nearest_fibonacci(age_high)

    close = float(df["Close"].iloc[-1])
    atr = float((df["High"] - df["Low"]).rolling(14).mean().iloc[-1])
    ema20 = float(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(df["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
    support_candidates = [float(df["Low"].tail(20).min()), ema20, ema50]
    resistance_candidates = [float(df["High"].tail(20).max()), float(df["High"].tail(55).max())]
    support_distance = min(abs(close - value) for value in support_candidates if np.isfinite(value)) / max(atr, 1e-9)
    resistance_distance = min(abs(value - close) for value in resistance_candidates if np.isfinite(value)) / max(atr, 1e-9)
    support_score = max(0.0, min(100.0, 100.0 - 45.0 * support_distance))
    resistance_score = max(0.0, min(100.0, 100.0 - 45.0 * resistance_distance))
    trend_score = 100.0 if close > ema20 > ema50 > ema200 else 80.0 if close > ema50 > ema200 else 60.0 if close > ema50 else 35.0

    lunar = _lunar_phase(df.index[-1]) if cfg.lunar_enabled else {
        "lunar_phase": "DISABLED", "lunar_days_to_major_marker": np.nan,
        "lunar_marker_score": 0.0, "lunar_phase_fraction": np.nan,
    }
    lunar_hit, lunar_lift, lunar_samples = _lunar_pivot_validation(df.index, sorted(set(high_pos + low_pos))) if cfg.lunar_enabled else (np.nan, np.nan, 0)
    # The legacy lunar check measures proximity to any pivot and has no
    # directional/return label.  Keep it visible for audit, but allow astronomy
    # to influence decisions only through the stricter EOFF directional gate.
    lunar_component = 0.0

    bullish_timing = 0.47 * max(low_due, fib_low) + 0.31 * support_score + 0.22 * trend_score
    bearish_timing = 0.51 * max(high_due, fib_high) + 0.38 * resistance_score + 0.11 * (100.0 - trend_score)
    expansion_phase = 100.0 * max(0.0, 1.0 - abs(age_low - 0.40 * dominant) / max(0.45 * dominant, 2.0)) if np.isfinite(age_low) else 50.0
    continuation_timing = 0.50 * expansion_phase + 0.35 * trend_score + 0.15 * (100.0 - high_due)
    bullish_timing = float(max(0.0, min(100.0, bullish_timing)))
    bearish_timing = float(max(0.0, min(100.0, bearish_timing)))
    continuation_timing = float(max(0.0, min(100.0, continuation_timing)))

    # Clean-room EOFF participates only after the public-style confluence gate:
    # at least four Fibonacci time projections, an astronomical marker, and
    # sufficient chronological evidence. The outer scanner weight remains
    # capped separately at 10% for core/swing and 5% for Multibagger.
    eoff_active = bool(eoff.get("eoff_signal_active"))
    eoff_history = int(_finite(eoff.get("eoff_public_directional_events"), 0.0))
    strength_multiplier = {
        "VERY_STRONG": 1.0, "STRONG": 0.72, "MEDIUM": 0.40, "LOW": 0.0,
    }.get(str(eoff.get("eoff_strength_label") or "LOW"), 0.0)
    eoff_internal_weight = min(0.35, 0.35 * strength_multiplier * min(1.0, eoff_history / 16.0)) if eoff_active else 0.0
    eoff_direction = str(eoff.get("eoff_direction_bias") or "NEUTRAL")
    eoff_score = _finite(eoff.get("eoff_reconstruction_score"), 50.0)
    if eoff_internal_weight > 0.0:
        if eoff_direction == "BULLISH":
            bullish_timing = (1.0 - eoff_internal_weight) * bullish_timing + eoff_internal_weight * eoff_score
            continuation_timing = (1.0 - 0.65 * eoff_internal_weight) * continuation_timing + 0.65 * eoff_internal_weight * eoff_score
            bearish_timing = (1.0 - eoff_internal_weight) * bearish_timing + eoff_internal_weight * max(0.0, 100.0 - eoff_score)
        elif eoff_direction == "BEARISH":
            bearish_timing = (1.0 - eoff_internal_weight) * bearish_timing + eoff_internal_weight * eoff_score
            bullish_timing = (1.0 - eoff_internal_weight) * bullish_timing + eoff_internal_weight * max(0.0, 100.0 - eoff_score)
            continuation_timing = (1.0 - eoff_internal_weight) * continuation_timing + eoff_internal_weight * max(0.0, 100.0 - eoff_score)

    bullish_timing = float(max(0.0, min(100.0, bullish_timing)))
    bearish_timing = float(max(0.0, min(100.0, bearish_timing)))
    continuation_timing = float(max(0.0, min(100.0, continuation_timing)))

    if bullish_timing >= bearish_timing + 8.0:
        direction = "BULLISH_REVERSAL_WINDOW"
        bars_due = dominant - age_low if np.isfinite(age_low) else dominant
        phase = "LATE_CORRECTION" if bars_due <= max(3.0, 0.15 * dominant) else "CORRECTION"
    elif bearish_timing >= bullish_timing + 8.0:
        direction = "BEARISH_REVERSAL_RISK"
        bars_due = dominant - age_high if np.isfinite(age_high) else dominant
        phase = "LATE_EXPANSION" if bars_due <= max(3.0, 0.15 * dominant) else "EXPANSION"
    else:
        direction = "NEUTRAL_TIME_CLUSTER"
        low_remaining = dominant - age_low if np.isfinite(age_low) else dominant
        high_remaining = dominant - age_high if np.isfinite(age_high) else dominant
        bars_due = min(low_remaining, high_remaining)
        phase = "TRANSITION"
    while bars_due < -max(2.0, 0.15 * dominant):
        bars_due += dominant
    bars_due = max(0.0, bars_due)
    tolerance_bars = max(2, int(round(cfg.window_tolerance_pct * dominant)))
    window_start, window_end = _business_window(df.index[-1], bars_due, tolerance_bars)

    sample_score = min(100.0, 100.0 * pivot_samples / max(1, cfg.min_validation_events))
    validation_score = 50.0 if not np.isfinite(pivot_hit) else pivot_hit
    confidence = 0.30 * agreement + 0.35 * validation_score + 0.20 * sample_score + 0.15 * min(100.0, max(0.0, 100.0 * _finite(acf_strength, 0.0)))
    confidence = float(max(0.0, min(100.0, confidence)))
    fib_score = max(fib_low, fib_high)
    time_score = 0.31 * agreement + 0.28 * validation_score + 0.24 * max(bullish_timing, bearish_timing, continuation_timing) + 0.17 * fib_score
    if eoff_internal_weight > 0.0:
        time_score = (1.0 - eoff_internal_weight) * time_score + eoff_internal_weight * eoff_score
        directional_hit = _finite(eoff.get("eoff_public_forward_hit_rate"), 0.0)
        eoff_confidence_proxy = min(100.0, 35.0 + 0.55 * directional_hit + 2.0 * min(12, eoff_history))
        confidence = (1.0 - 0.50 * eoff_internal_weight) * confidence + 0.50 * eoff_internal_weight * eoff_confidence_proxy
    time_score = float(max(0.0, min(100.0, time_score)))
    confidence = float(max(0.0, min(100.0, confidence)))
    price_time = float(max(0.0, min(100.0, 0.60 * max(bullish_timing, continuation_timing) + 0.40 * confidence)))

    time_state = (
        "VALIDATED"
        if confidence >= cfg.validation_min_confidence and pivot_samples >= cfg.min_validation_events
        else "LIMITED_EVIDENCE"
    )
    quick_buy = _build_quick_buy_decision(
        df,
        direction=direction,
        phase=phase,
        time_state=time_state,
        time_score=time_score,
        confidence=confidence,
        bullish_timing=bullish_timing,
        continuation_timing=continuation_timing,
        price_time=price_time,
        trend_score=trend_score,
        window_start=window_start,
        window_end=window_end,
        bars_due=bars_due,
        eoff=eoff,
        min_rr1=cfg.quick_min_rr1,
        require_final_eod=cfg.require_final_eod,
    )

    explanation = (
        f"Dominant cycle {dominant:.1f} bar; pivot {pivot_cycle:.1f} bila tersedia, "
        f"ACF {acf_period:.1f}, spectral {spectral_period:.1f}; historical hit "
        f"{pivot_hit:.1f}%/{pivot_samples} event; direction {direction}; window {window_start}–{window_end}; "
        f"EOFF {eoff.get('eoff_strength_label', 'LOW')} active={eoff_active} internal-weight={eoff_internal_weight*100:.1f}%."
    )
    return {
        "time_cycle_version": TIME_CYCLE_VERSION,
        "time_cycle_state": time_state,
        "time_cycle_score": round(time_score, 1),
        "time_cycle_confidence": round(confidence, 1),
        "time_cycle_direction_bias": direction,
        "time_cycle_phase": phase,
        "dominant_cycle_bars": round(dominant, 1),
        "pivot_cycle_bars": round(pivot_cycle, 1) if np.isfinite(pivot_cycle) else np.nan,
        "autocorr_cycle_bars": round(acf_period, 1) if np.isfinite(acf_period) else np.nan,
        "autocorr_strength": round(_finite(acf_strength, np.nan), 3),
        "spectral_cycle_bars": round(spectral_period, 1) if np.isfinite(spectral_period) else np.nan,
        "spectral_power_pct": round(_finite(spectral_power, np.nan), 1),
        "cycle_agreement_score": round(agreement, 1),
        "cycle_historical_hit_rate": round(pivot_hit, 1) if np.isfinite(pivot_hit) else np.nan,
        "cycle_validation_samples": int(pivot_samples),
        "cycle_median_error_pct": round(pivot_error, 1) if np.isfinite(pivot_error) else np.nan,
        "bullish_timing_score": round(bullish_timing, 1),
        "continuation_timing_score": round(continuation_timing, 1),
        "bearish_timing_score": round(bearish_timing, 1),
        "price_time_confluence_score": round(price_time, 1),
        "fibonacci_time_score": round(fib_score, 1),
        "fibonacci_low_target": fib_low_target,
        "fibonacci_high_target": fib_high_target,
        "cycle_age_from_low": age_low,
        "cycle_age_from_high": age_high,
        "bars_to_reversal_window": round(bars_due, 1),
        "next_reversal_window_start": window_start,
        "next_reversal_window_end": window_end,
        "lunar_phase": lunar["lunar_phase"],
        "lunar_days_to_major_marker": round(_finite(lunar["lunar_days_to_major_marker"], np.nan), 2),
        "lunar_marker_score": round(_finite(lunar["lunar_marker_score"], 0.0), 1),
        "lunar_historical_hit_rate": round(lunar_hit, 1) if np.isfinite(lunar_hit) else np.nan,
        "lunar_historical_lift": round(lunar_lift, 2) if np.isfinite(lunar_lift) else np.nan,
        "lunar_validation_samples": int(lunar_samples),
        "eoff_internal_weight_pct": round(100.0 * eoff_internal_weight, 2),
        **eoff,
        **quick_buy,
        "time_cycle_explanation": explanation,
    }


def setup_time_alignment(analysis: Mapping[str, Any], setup: str) -> float:
    name = str(setup or "").upper()
    if name in {"PULLBACK_CONTINUATION", "REVERSAL_ACCUMULATION"}:
        return _finite(analysis.get("bullish_timing_score"), 50.0)
    if name in {"BREAKOUT_RETEST"}:
        return _finite(analysis.get("continuation_timing_score"), 50.0)
    if name in {"UNICORN_SNIPER_ICT", "UNICORN_ICT"}:
        return max(
            _finite(analysis.get("bullish_timing_score"), 50.0),
            _finite(analysis.get("continuation_timing_score"), 50.0),
        )
    return _finite(analysis.get("price_time_confluence_score"), 50.0)


def enrich_core_signals_with_time_cycle(
    signals: pd.DataFrame | None,
    prepared: Mapping[str, pd.DataFrame],
    *,
    enabled: bool = True,
    max_weight: float = 0.10,
    min_confidence: float = 55.0,
    config: TimeCycleConfig | None = None,
) -> pd.DataFrame:
    if signals is None:
        return pd.DataFrame()
    out = signals.copy()
    defaults = {
        "time_cycle_score": np.nan,
        "time_cycle_alignment_score": 50.0,
        "time_cycle_confidence": 0.0,
        "time_cycle_effective_weight_pct": 0.0,
        "time_cycle_direction_bias": "NEUTRAL",
        "time_cycle_phase": "UNKNOWN",
        "dominant_cycle_bars": np.nan,
        "next_reversal_window_start": "",
        "next_reversal_window_end": "",
        "bars_to_reversal_window": np.nan,
        "cycle_historical_hit_rate": np.nan,
        "cycle_validation_samples": 0,
        "lunar_phase": "UNKNOWN",
        "lunar_days_to_major_marker": np.nan,
        "time_cycle_state": "DISABLED" if not enabled else "UNAVAILABLE",
        "time_cycle_explanation": "",
        "eoff_state": "DISABLED" if not enabled else "UNAVAILABLE",
        "eoff_reconstruction_score": np.nan,
        "eoff_strength_label": "LOW",
        "eoff_signal_active": False,
        "eoff_direction_bias": "NEUTRAL",
        "eoff_time_power_score": 0.0,
        "eoff_price_power_score": 0.0,
        "eoff_pattern_score": 0.0,
        "eoff_momentum_score": 0.0,
        "eoff_astro_score": 0.0,
        "eoff_fib_cluster_count": 0,
        "eoff_fib_unique_anchor_count": 0,
        "eoff_historical_hit_rate": np.nan,
        "eoff_historical_baseline_rate": np.nan,
        "eoff_historical_lift": np.nan,
        "eoff_confluence_historical_hit_rate": np.nan,
        "eoff_confluence_historical_events": 0,
        "eoff_confluence_historical_lift": np.nan,
        "eoff_public_validation_state": "INSUFFICIENT_EVENTS",
        "eoff_public_validation_method": "PUBLIC_FIXED_FAMILY_CHRONOLOGICAL_FORWARD_TEST",
        "eoff_public_directional_events": 0,
        "eoff_public_reversal_hit_rate": np.nan,
        "eoff_public_baseline_rate": np.nan,
        "eoff_public_lift": np.nan,
        "eoff_public_forward_hit_rate": np.nan,
        "eoff_public_median_directional_return_pct": np.nan,
        "eoff_historical_events": 0,
        "eoff_bars_to_cluster": np.nan,
        "eoff_reversal_date": "",
        "eoff_ephemeris_state": "UNAVAILABLE",
        "eoff_ephemeris_date": "",
        "eoff_astro_events": "",
        "eoff_active_aspects": "",
        "eoff_retrograde_planets": "",
        "eoff_stationary_planets": "",
        "eoff_ingress_events": "",
        "eoff_moon_declination_deg": np.nan,
        "eoff_moon_declination_extreme_score": 0.0,
        "eoff_moon_phase": "UNKNOWN",
        "eoff_sun_sign": "UNKNOWN",
        "eoff_sun_annual_cycle_bias": "NEUTRAL",
        "eoff_roadmap_json": "[]",
        "eoff_internal_weight_pct": 0.0,
        "eoff_explanation": "",
        "quick_buy_state": "NO_VALID_BUY_DATE",
        "quick_buy_action": "WAIT",
        "best_buy_date": "",
        "best_buy_date_basis": "",
        "best_buy_window_start": "",
        "best_buy_window_end": "",
        "best_buy_score": 0.0,
        "best_buy_confidence": 0.0,
        "best_buy_entry_low": np.nan,
        "best_buy_entry_high": np.nan,
        "best_buy_trigger": np.nan,
        "best_buy_stop_loss": np.nan,
        "best_buy_tp1": np.nan,
        "best_buy_tp2": np.nan,
        "best_buy_rr1": np.nan,
        "best_buy_rr2": np.nan,
        "best_buy_target_basis": "",
        "best_buy_order_plan": "NO_ORDER",
        "best_buy_reason": "",
        "best_buy_no_trade_condition": "",
        "best_buy_summary": "",
    }
    for column, default in defaults.items():
        if column not in out:
            out[column] = default
    if not enabled or out.empty or "ticker" not in out:
        return out
    cache: dict[str, dict[str, Any]] = {}
    cap = max(0.0, min(0.10, float(max_weight)))
    for idx, row in out.iterrows():
        ticker = str(row.get("ticker") or "")
        frame = prepared.get(ticker)
        if frame is None:
            frame = prepared.get(ticker.replace(".JK", ""))
        if ticker not in cache:
            cache[ticker] = analyze_time_cycle(frame, config=config)
        analysis = cache[ticker]
        setup = str(row.get("setup") or "")
        alignment = setup_time_alignment(analysis, setup)
        confidence = _finite(analysis.get("time_cycle_confidence"), 0.0)
        samples = int(_finite(analysis.get("cycle_validation_samples"), 0.0))
        state = str(analysis.get("time_cycle_state") or "UNAVAILABLE")
        evidence_multiplier = min(1.0, samples / max(1, (config or TimeCycleConfig()).min_validation_events))
        effective = cap * (confidence / 100.0) * evidence_multiplier if confidence >= min_confidence and state == "VALIDATED" else 0.0
        for key in defaults:
            if key in analysis:
                out.at[idx, key] = analysis[key]
        out.at[idx, "time_cycle_alignment_score"] = round(alignment, 1)
        out.at[idx, "time_cycle_effective_weight_pct"] = round(100.0 * effective, 2)
    return out


def enrich_swing_specialty_with_time_cycle(
    frame: pd.DataFrame | None,
    prepared: Mapping[str, pd.DataFrame],
    *,
    enabled: bool = True,
    max_weight: float = 0.10,
    min_confidence: float = 55.0,
    config: TimeCycleConfig | None = None,
) -> pd.DataFrame:
    """Attach guarded daily time-cycle evidence to swing specialty rows.

    Intended for daily Sniper/ICT only. Intraday strategies must not call this
    function.
    """
    if frame is None:
        return pd.DataFrame()
    out = frame.copy()
    defaults = {
        'time_cycle_alignment_score': 50.0, 'time_cycle_effective_weight_pct': 0.0,
        'time_cycle_score': np.nan, 'time_cycle_confidence': 0.0,
        'time_cycle_direction_bias': 'NEUTRAL', 'time_cycle_phase': 'UNKNOWN',
        'dominant_cycle_bars': np.nan, 'next_reversal_window_start': '',
        'next_reversal_window_end': '', 'bars_to_reversal_window': np.nan,
        'cycle_historical_hit_rate': np.nan, 'cycle_validation_samples': 0,
        'lunar_phase': 'UNKNOWN', 'lunar_days_to_major_marker': np.nan,
        'time_cycle_state': 'DISABLED' if not enabled else 'UNAVAILABLE',
        'time_cycle_explanation': '',
        'eoff_state': 'DISABLED' if not enabled else 'UNAVAILABLE',
        'eoff_reconstruction_score': np.nan, 'eoff_strength_label': 'LOW',
        'eoff_signal_active': False, 'eoff_direction_bias': 'NEUTRAL',
        'eoff_time_power_score': 0.0, 'eoff_price_power_score': 0.0,
        'eoff_pattern_score': 0.0, 'eoff_momentum_score': 0.0,
        'eoff_astro_score': 0.0, 'eoff_fib_cluster_count': 0,
        'eoff_fib_unique_anchor_count': 0, 'eoff_historical_hit_rate': np.nan,
        'eoff_historical_baseline_rate': np.nan, 'eoff_historical_lift': np.nan,
        'eoff_confluence_historical_hit_rate': np.nan, 'eoff_confluence_historical_events': 0,
        'eoff_confluence_historical_lift': np.nan,
        'eoff_public_validation_state': 'INSUFFICIENT_EVENTS',
        'eoff_public_validation_method': 'PUBLIC_FIXED_FAMILY_CHRONOLOGICAL_FORWARD_TEST',
        'eoff_public_directional_events': 0,
        'eoff_public_reversal_hit_rate': np.nan, 'eoff_public_baseline_rate': np.nan,
        'eoff_public_lift': np.nan, 'eoff_public_forward_hit_rate': np.nan,
        'eoff_public_median_directional_return_pct': np.nan,
        'eoff_historical_events': 0, 'eoff_bars_to_cluster': np.nan,
        'eoff_reversal_date': '', 'eoff_ephemeris_state': 'UNAVAILABLE', 'eoff_ephemeris_date': '',
        'eoff_astro_events': '', 'eoff_active_aspects': '',
        'eoff_retrograde_planets': '', 'eoff_stationary_planets': '',
        'eoff_ingress_events': '', 'eoff_moon_declination_deg': np.nan,
        'eoff_moon_declination_extreme_score': 0.0, 'eoff_moon_phase': 'UNKNOWN',
        'eoff_sun_sign': 'UNKNOWN', 'eoff_sun_annual_cycle_bias': 'NEUTRAL',
        'eoff_roadmap_json': '[]', 'eoff_internal_weight_pct': 0.0,
        'eoff_explanation': '',
        'quick_buy_state': 'NO_VALID_BUY_DATE', 'quick_buy_action': 'WAIT',
        'best_buy_date': '', 'best_buy_date_basis': '',
        'best_buy_window_start': '', 'best_buy_window_end': '',
        'best_buy_score': 0.0, 'best_buy_confidence': 0.0,
        'best_buy_entry_low': np.nan, 'best_buy_entry_high': np.nan,
        'best_buy_trigger': np.nan, 'best_buy_stop_loss': np.nan,
        'best_buy_tp1': np.nan, 'best_buy_tp2': np.nan,
        'best_buy_rr1': np.nan, 'best_buy_rr2': np.nan,
        'best_buy_target_basis': '',
        'best_buy_order_plan': 'NO_ORDER', 'best_buy_reason': '',
        'best_buy_no_trade_condition': '', 'best_buy_summary': '',
    }
    for column, default in defaults.items():
        if column not in out:
            out[column] = default
    if not enabled or out.empty or 'ticker' not in out:
        return out
    cap = max(0.0, min(0.10, float(max_weight)))
    cache: dict[str, dict[str, Any]] = {}
    for idx, row in out.iterrows():
        ticker = str(row.get('ticker') or '')
        source = prepared.get(ticker)
        if source is None:
            source = prepared.get(ticker.replace('.JK', ''))
        if ticker not in cache:
            cache[ticker] = analyze_time_cycle(source, config=config)
        analysis = cache[ticker]
        alignment = max(
            _finite(analysis.get('bullish_timing_score'), 50.0),
            _finite(analysis.get('continuation_timing_score'), 50.0),
        )
        confidence = _finite(analysis.get('time_cycle_confidence'), 0.0)
        samples = int(_finite(analysis.get('cycle_validation_samples'), 0.0))
        state = str(analysis.get('time_cycle_state') or 'UNAVAILABLE')
        min_events = max(1, (config or TimeCycleConfig()).min_validation_events)
        effective = cap * confidence / 100.0 * min(1.0, samples / min_events) if confidence >= min_confidence and state == 'VALIDATED' else 0.0
        for key in defaults:
            if key in analysis:
                out.at[idx, key] = analysis[key]
        out.at[idx, 'time_cycle_alignment_score'] = round(alignment, 1)
        out.at[idx, 'time_cycle_effective_weight_pct'] = round(100.0 * effective, 2)
    return out


def make_time_cycle_chart(frame: pd.DataFrame, analysis: Mapping[str, Any], ticker: str = ""):
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return None
    df = _clean_frame(frame)
    if df.empty:
        return None
    recent = df.tail(260)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=recent.index, open=recent.get("Open", recent["Close"]), high=recent["High"],
        low=recent["Low"], close=recent["Close"], name="Price",
    ))
    start = str(analysis.get("next_reversal_window_start") or "")
    end = str(analysis.get("next_reversal_window_end") or "")
    if start and end:
        fig.add_vrect(x0=start, x1=end, opacity=0.18, line_width=0, annotation_text="Time window", annotation_position="top left")
    fig.update_layout(
        title=f"{ticker} · Objective Time-Cycle Intelligence",
        template="plotly_dark", height=610, xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=40, t=55, b=20), hovermode="x unified",
    )
    return fig


__all__ = [
    "TIME_CYCLE_VERSION", "TimeCycleConfig", "analyze_time_cycle",
    "setup_time_alignment", "enrich_core_signals_with_time_cycle",
    "enrich_swing_specialty_with_time_cycle", "make_time_cycle_chart",
    "EOFFConfig", "EOFF_VERSION", "analyze_eoff_reconstruction",
]

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
    EOFFConfig, EOFF_VERSION, analyze_eoff_reconstruction, setup_eoff_alignment,
)

import numpy as np
import pandas as pd

TIME_CYCLE_VERSION = "6.5.0-eoff-time-cycle"
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
        "eoff_fib_cluster_count": 0,
        "eoff_fib_unique_anchor_count": 0,
        "eoff_historical_hit_rate": np.nan,
        "eoff_historical_baseline_rate": np.nan,
        "eoff_historical_lift": np.nan,
        "eoff_confluence_historical_hit_rate": np.nan,
        "eoff_confluence_historical_events": 0,
        "eoff_confluence_historical_lift": np.nan,
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
        "eoff_explanation": "Riwayat belum cukup untuk clean-room EOFF.",
        "time_cycle_explanation": "Riwayat belum cukup untuk time-cycle harian.",
    }
    if len(df) < cfg.min_bars:
        return unavailable

    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    high_pos = _confirmed_pivot_positions(highs, cfg.pivot_left, cfg.pivot_right, "high")
    low_pos = _confirmed_pivot_positions(lows, cfg.pivot_left, cfg.pivot_right, "low")
    high_intervals = _filtered_intervals(high_pos, cfg.min_period, cfg.max_period)
    low_intervals = _filtered_intervals(low_pos, cfg.min_period, cfg.max_period)
    combined = np.concatenate([high_intervals, low_intervals]) if high_intervals.size or low_intervals.size else np.array([], dtype=float)

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
    lunar_valid = bool(lunar_samples >= 8 and np.isfinite(lunar_lift) and lunar_lift >= 1.05)
    lunar_component = float(lunar["lunar_marker_score"]) if lunar_valid else 0.0

    bullish_timing = 0.42 * max(low_due, fib_low) + 0.28 * support_score + 0.20 * trend_score + 0.10 * lunar_component
    bearish_timing = 0.46 * max(high_due, fib_high) + 0.34 * resistance_score + 0.10 * (100.0 - trend_score) + 0.10 * lunar_component
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
    eoff_history = int(_finite(eoff.get("eoff_historical_events"), 0.0))
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

    sample_score = min(100.0, 12.0 * pivot_samples)
    validation_score = 50.0 if not np.isfinite(pivot_hit) else pivot_hit
    confidence = 0.30 * agreement + 0.35 * validation_score + 0.20 * sample_score + 0.15 * min(100.0, max(0.0, 100.0 * _finite(acf_strength, 0.0)))
    confidence = float(max(0.0, min(100.0, confidence)))
    fib_score = max(fib_low, fib_high)
    time_score = 0.28 * agreement + 0.25 * validation_score + 0.22 * max(bullish_timing, bearish_timing, continuation_timing) + 0.15 * fib_score + 0.10 * lunar_component
    if eoff_internal_weight > 0.0:
        time_score = (1.0 - eoff_internal_weight) * time_score + eoff_internal_weight * eoff_score
        eoff_confidence_proxy = min(100.0, 45.0 + 0.45 * _finite(eoff.get("eoff_historical_hit_rate"), 0.0) + 2.0 * min(12, eoff_history))
        confidence = (1.0 - 0.50 * eoff_internal_weight) * confidence + 0.50 * eoff_internal_weight * eoff_confidence_proxy
    time_score = float(max(0.0, min(100.0, time_score)))
    confidence = float(max(0.0, min(100.0, confidence)))
    price_time = float(max(0.0, min(100.0, 0.60 * max(bullish_timing, continuation_timing) + 0.40 * confidence)))

    explanation = (
        f"Dominant cycle {dominant:.1f} bar; pivot {pivot_cycle:.1f} bila tersedia, "
        f"ACF {acf_period:.1f}, spectral {spectral_period:.1f}; historical hit "
        f"{pivot_hit:.1f}%/{pivot_samples} event; direction {direction}; window {window_start}–{window_end}; "
        f"EOFF {eoff.get('eoff_strength_label', 'LOW')} active={eoff_active} internal-weight={eoff_internal_weight*100:.1f}%."
    )
    return {
        "time_cycle_version": TIME_CYCLE_VERSION,
        "time_cycle_state": "VALIDATED" if confidence >= 60.0 and pivot_samples >= 5 else "LIMITED_EVIDENCE",
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
        evidence_multiplier = min(1.0, samples / 12.0)
        effective = cap * (confidence / 100.0) * evidence_multiplier if confidence >= min_confidence and state in {"VALIDATED", "LIMITED_EVIDENCE"} else 0.0
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
        'eoff_historical_events': 0, 'eoff_bars_to_cluster': np.nan,
        'eoff_reversal_date': '', 'eoff_ephemeris_state': 'UNAVAILABLE', 'eoff_ephemeris_date': '',
        'eoff_astro_events': '', 'eoff_active_aspects': '',
        'eoff_retrograde_planets': '', 'eoff_stationary_planets': '',
        'eoff_ingress_events': '', 'eoff_moon_declination_deg': np.nan,
        'eoff_moon_declination_extreme_score': 0.0, 'eoff_moon_phase': 'UNKNOWN',
        'eoff_sun_sign': 'UNKNOWN', 'eoff_sun_annual_cycle_bias': 'NEUTRAL',
        'eoff_roadmap_json': '[]', 'eoff_internal_weight_pct': 0.0,
        'eoff_explanation': '',
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
        effective = cap * confidence / 100.0 * min(1.0, samples / 12.0) if confidence >= min_confidence else 0.0
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

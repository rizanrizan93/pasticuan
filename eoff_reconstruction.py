"""Clean-room Eye-of-Future-style reconstruction for IDX daily analysis.

This module implements only public/reproducible concepts discussed in the
project: geocentric ephemeris, lunar phase/declination, major aspects,
ingress/retrograde markers, Sun annual cycle, multi-anchor Fibonacci time
clusters, and a price-time-pattern-momentum ensemble.

It does *not* claim to reproduce Astronacci's proprietary Eye of Future.
Every component is auditable and fail-soft.  Astronomical markers never create
a trade on their own and never override structural invalidation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence
import json
import math

import numpy as np
import pandas as pd

try:  # Optional so the scanner core can still import if dependency is absent.
    import ephem  # type: ignore
except Exception:  # pragma: no cover - explicitly tested through monkeypatch.
    ephem = None

EOFF_VERSION = "6.5.0-clean-room-eoff-reconstruction"
MAJOR_ASPECTS = (0.0, 60.0, 90.0, 120.0, 180.0)
FIB_NUMBERS = (5, 8, 13, 21, 34, 55, 89, 144, 233)
FIB_RATIOS = (0.382, 0.5, 0.618, 1.0, 1.272, 1.618, 2.0, 2.618)
ZODIAC_SIGNS = (
    "ARIES", "TAURUS", "GEMINI", "CANCER", "LEO", "VIRGO",
    "LIBRA", "SCORPIO", "SAGITTARIUS", "CAPRICORN", "AQUARIUS", "PISCES",
)
PLANET_NAMES = (
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
)
# Pairs/aspects explicitly surfaced in the project's public-method research.
PUBLISHED_CANDIDATE_ASPECTS = {
    ("Sun", "Jupiter", 120.0),
    ("Sun", "Jupiter", 90.0),
    ("Sun", "Neptune", 90.0),
    ("Venus", "Neptune", 90.0),
    ("Venus", "Pluto", 90.0),
}


@dataclass(frozen=True)
class EOFFConfig:
    enabled: bool = True
    ephemeris_enabled: bool = True
    min_fib_cluster: int = 4
    fib_window_bars: int = 2
    aspect_orb_deg: float = 3.0
    station_speed_deg_day: float = 0.12
    historical_min_events: int = 8
    historical_window_bars: int = 3
    max_anchor_pivots: int = 14
    future_horizon_bars: int = 34
    require_astro_fib_confluence: bool = True


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _clip(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, _finite(value, low))))


def _clean_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    mapping: dict[Any, str] = {}
    lowercase = {str(column).lower(): column for column in out.columns}
    for canonical in ("Open", "High", "Low", "Close", "Volume"):
        source = lowercase.get(canonical.lower())
        if source is not None:
            mapping[source] = canonical
    out = out.rename(columns=mapping)
    if not {"High", "Low", "Close"}.issubset(out.columns):
        return pd.DataFrame()
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["High", "Low", "Close"])
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()]
    if out.index.tz is not None:
        out.index = out.index.tz_convert("Asia/Jakarta").tz_localize(None)
    return out[~out.index.duplicated(keep="last")].sort_index()


def _angle_deg(value: Any) -> float:
    return (float(value) * 180.0 / math.pi) % 360.0


def _signed_angle_delta(current: float, previous: float) -> float:
    return float((current - previous + 180.0) % 360.0 - 180.0)


def _angular_separation(first: float, second: float) -> float:
    return float(abs((first - second + 180.0) % 360.0 - 180.0))


def _stamp_key(timestamp: pd.Timestamp | str) -> str:
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize().strftime("%Y/%m/%d 12:00:00")


@lru_cache(maxsize=8192)
def _body_position(body_name: str, date_key: str) -> tuple[float, float]:
    if ephem is None:
        return (np.nan, np.nan)
    body_class = getattr(ephem, body_name, None)
    if body_class is None:
        return (np.nan, np.nan)
    body = body_class()
    body.compute(ephem.Date(date_key))
    ecliptic = ephem.Ecliptic(body)
    return (_angle_deg(ecliptic.lon), float(body.dec) * 180.0 / math.pi)


def _zodiac(longitude: float) -> tuple[str, float]:
    if not np.isfinite(longitude):
        return ("UNKNOWN", np.nan)
    normalized = longitude % 360.0
    sign_index = int(normalized // 30.0)
    return (ZODIAC_SIGNS[sign_index], normalized % 30.0)


def _ephemeris_snapshot(timestamp: pd.Timestamp, config: EOFFConfig) -> dict[str, Any]:
    unavailable = {
        "ephemeris_available": False,
        "ephemeris_state": "DISABLED" if not config.ephemeris_enabled else "DEPENDENCY_MISSING",
        "planet_longitudes": {},
        "planet_declinations": {},
        "planet_daily_motion": {},
        "retrograde_planets": [],
        "stationary_planets": [],
        "ingress_events": [],
        "active_aspects": [],
        "aspect_cluster_score": 0.0,
        "published_aspect_score": 0.0,
        "moon_declination_deg": np.nan,
        "moon_declination_extreme_score": 0.0,
        "moon_declination_turning": False,
        "moon_phase_angle_deg": np.nan,
        "moon_phase_name": "UNKNOWN",
        "moon_phase_score": 0.0,
        "sun_sign": "UNKNOWN",
        "sun_annual_cycle_bias": "NEUTRAL",
        "sun_annual_cycle_score": 50.0,
        "astro_event_count": 0,
        "astro_cluster_score": 0.0,
        "astro_events": [],
    }
    if not config.ephemeris_enabled or ephem is None:
        return unavailable

    current_key = _stamp_key(timestamp)
    previous_key = _stamp_key(pd.Timestamp(timestamp) - pd.Timedelta(days=1))
    next_key = _stamp_key(pd.Timestamp(timestamp) + pd.Timedelta(days=1))
    longitudes: dict[str, float] = {}
    declinations: dict[str, float] = {}
    motion: dict[str, float] = {}
    retrograde: list[str] = []
    stationary: list[str] = []
    ingress: list[str] = []
    for name in PLANET_NAMES:
        longitude, declination = _body_position(name, current_key)
        prev_longitude, _ = _body_position(name, previous_key)
        next_longitude, _ = _body_position(name, next_key)
        longitudes[name] = longitude
        declinations[name] = declination
        if np.isfinite(prev_longitude) and np.isfinite(next_longitude):
            daily_motion = _signed_angle_delta(next_longitude, prev_longitude) / 2.0
        else:
            daily_motion = np.nan
        motion[name] = daily_motion
        if name not in {"Sun", "Moon"} and np.isfinite(daily_motion):
            if daily_motion < -config.station_speed_deg_day:
                retrograde.append(name)
            if abs(daily_motion) <= config.station_speed_deg_day:
                stationary.append(name)
        if np.isfinite(prev_longitude) and np.isfinite(next_longitude):
            prev_sign, _ = _zodiac(prev_longitude)
            current_sign, current_degree = _zodiac(longitude)
            next_sign, _ = _zodiac(next_longitude)
            if current_sign != prev_sign or current_sign != next_sign or current_degree <= max(0.8, abs(daily_motion) * 1.5):
                ingress.append(f"{name}:{current_sign}")

    aspects: list[dict[str, Any]] = []
    published_score = 0.0
    all_names = list(PLANET_NAMES)
    for i, first in enumerate(all_names):
        for second in all_names[i + 1:]:
            first_lon = longitudes.get(first, np.nan)
            second_lon = longitudes.get(second, np.nan)
            if not np.isfinite(first_lon) or not np.isfinite(second_lon):
                continue
            separation = _angular_separation(first_lon, second_lon)
            target = min(MAJOR_ASPECTS, key=lambda angle: abs(separation - angle))
            orb = abs(separation - target)
            if orb > config.aspect_orb_deg:
                continue
            quality = _clip(100.0 * (1.0 - orb / max(config.aspect_orb_deg, 0.1)))
            published = (first, second, target) in PUBLISHED_CANDIDATE_ASPECTS or (second, first, target) in PUBLISHED_CANDIDATE_ASPECTS
            if published:
                published_score = max(published_score, quality)
            aspects.append({
                "pair": f"{first}-{second}", "aspect_deg": target,
                "separation_deg": round(separation, 3), "orb_deg": round(orb, 3),
                "quality": round(quality, 1), "published_candidate": published,
            })
    aspects.sort(key=lambda row: (bool(row["published_candidate"]), float(row["quality"])), reverse=True)
    aspect_score = _clip(sum(float(row["quality"]) * (1.25 if row["published_candidate"] else 0.55) for row in aspects[:8]) / 3.5)

    moon_declination = declinations.get("Moon", np.nan)
    moon_prev_decl = _body_position("Moon", previous_key)[1]
    moon_next_decl = _body_position("Moon", next_key)[1]
    moon_turning = bool(
        np.isfinite(moon_prev_decl) and np.isfinite(moon_declination) and np.isfinite(moon_next_decl)
        and (moon_declination - moon_prev_decl) * (moon_next_decl - moon_declination) <= 0
    )
    declination_extreme = _clip((abs(moon_declination) - 16.0) / 12.5 * 100.0) if np.isfinite(moon_declination) else 0.0
    if moon_turning:
        declination_extreme = min(100.0, declination_extreme + 18.0)

    sun_lon = longitudes.get("Sun", np.nan)
    moon_lon = longitudes.get("Moon", np.nan)
    phase_angle = (moon_lon - sun_lon) % 360.0 if np.isfinite(sun_lon) and np.isfinite(moon_lon) else np.nan
    phase_targets = {"NEW_MOON": 0.0, "FIRST_QUARTER": 90.0, "FULL_MOON": 180.0, "LAST_QUARTER": 270.0}
    if np.isfinite(phase_angle):
        phase_name = min(phase_targets, key=lambda key: _angular_separation(phase_angle, phase_targets[key]))
        phase_distance = _angular_separation(phase_angle, phase_targets[phase_name])
        major_distance = min(_angular_separation(phase_angle, 0.0), _angular_separation(phase_angle, 180.0))
        phase_score = _clip(100.0 * (1.0 - major_distance / 24.0))
    else:
        phase_name, phase_score = "UNKNOWN", 0.0

    sun_sign, _ = _zodiac(sun_lon)
    if sun_sign in {"CAPRICORN", "AQUARIUS", "PISCES", "ARIES", "TAURUS"}:
        annual_bias, annual_score = "TRADING_BIAS", 68.0
    elif sun_sign in {"GEMINI", "CANCER", "LEO", "VIRGO", "LIBRA", "SCORPIO", "SAGITTARIUS"}:
        annual_bias, annual_score = "DEFENSIVE_BIAS", 42.0
    else:
        annual_bias, annual_score = "NEUTRAL", 50.0

    events: list[str] = []
    if phase_score >= 55.0:
        events.append(phase_name)
    if declination_extreme >= 55.0:
        events.append("MOON_DECLINATION_EXTREME")
    events.extend(f"ASPECT:{row['pair']}:{int(row['aspect_deg'])}" for row in aspects[:5] if float(row["quality"]) >= 45.0)
    events.extend(f"RETROGRADE:{name}" for name in retrograde)
    events.extend(f"STATION:{name}" for name in stationary)
    events.extend(f"INGRESS:{name}" for name in ingress)
    astro_cluster = _clip(
        0.25 * phase_score + 0.20 * declination_extreme + 0.28 * aspect_score
        + 0.12 * published_score + 0.08 * min(100.0, 25.0 * len(stationary))
        + 0.07 * min(100.0, 12.5 * len(ingress))
    )
    return {
        "ephemeris_available": True,
        "ephemeris_state": "READY",
        "planet_longitudes": {key: round(value, 4) if np.isfinite(value) else np.nan for key, value in longitudes.items()},
        "planet_declinations": {key: round(value, 4) if np.isfinite(value) else np.nan for key, value in declinations.items()},
        "planet_daily_motion": {key: round(value, 5) if np.isfinite(value) else np.nan for key, value in motion.items()},
        "retrograde_planets": retrograde,
        "stationary_planets": stationary,
        "ingress_events": ingress,
        "active_aspects": aspects,
        "aspect_cluster_score": round(aspect_score, 1),
        "published_aspect_score": round(published_score, 1),
        "moon_declination_deg": round(moon_declination, 3) if np.isfinite(moon_declination) else np.nan,
        "moon_declination_extreme_score": round(declination_extreme, 1),
        "moon_declination_turning": moon_turning,
        "moon_phase_angle_deg": round(phase_angle, 3) if np.isfinite(phase_angle) else np.nan,
        "moon_phase_name": phase_name,
        "moon_phase_score": round(phase_score, 1),
        "sun_sign": sun_sign,
        "sun_annual_cycle_bias": annual_bias,
        "sun_annual_cycle_score": annual_score,
        "astro_event_count": len(events),
        "astro_cluster_score": round(astro_cluster, 1),
        "astro_events": events,
    }


def _confirmed_pivots(values: np.ndarray, left: int = 3, right: int = 3, mode: str = "low") -> list[int]:
    positions: list[int] = []
    if len(values) < left + right + 5:
        return positions
    for index in range(left, len(values) - right):
        window = values[index - left:index + right + 1]
        value = values[index]
        if not np.isfinite(value) or not np.isfinite(window).all():
            continue
        if mode == "high":
            condition = value >= float(np.max(window)) and value > float(np.max(values[index-left:index]))
        else:
            condition = value <= float(np.min(window)) and value < float(np.min(values[index-left:index]))
        if condition:
            positions.append(index)
    return positions


def _projection_candidates(
    pivot_positions: Sequence[int],
    current_position: int,
    config: EOFFConfig,
) -> list[dict[str, Any]]:
    pivots = sorted({int(position) for position in pivot_positions if 0 <= int(position) <= current_position})
    pivots = pivots[-config.max_anchor_pivots:]
    candidates: list[dict[str, Any]] = []
    for anchor in pivots:
        for fib in FIB_NUMBERS:
            target = anchor + int(fib)
            if current_position - config.fib_window_bars <= target <= current_position + config.future_horizon_bars:
                candidates.append({"anchor": anchor, "target": target, "basis": f"FIB_{fib}", "ratio": float(fib)})
    # Dynamic ratios project each historical swing duration from its ending pivot.
    for first, second in zip(pivots[:-1], pivots[1:]):
        duration = second - first
        if duration < 5:
            continue
        for ratio in FIB_RATIOS:
            target = second + int(round(duration * ratio))
            if current_position - config.fib_window_bars <= target <= current_position + config.future_horizon_bars:
                candidates.append({
                    "anchor": second, "target": target,
                    "basis": f"SWING_{duration}x{ratio:.3f}", "ratio": ratio,
                })
    return candidates


def _best_projection_cluster(candidates: Sequence[Mapping[str, Any]], current_position: int, config: EOFFConfig) -> dict[str, Any]:
    if not candidates:
        return {
            "fib_cluster_count": 0, "fib_unique_anchor_count": 0,
            "fib_cluster_target_bar": np.nan, "fib_cluster_bars_ahead": np.nan,
            "fib_cluster_score": 0.0, "fib_projection_details": [],
        }
    best_target = current_position
    best_members: list[Mapping[str, Any]] = []
    for target in range(current_position - config.fib_window_bars, current_position + config.future_horizon_bars + 1):
        members = [row for row in candidates if abs(int(row["target"]) - target) <= config.fib_window_bars]
        unique = {(int(row["anchor"]), str(row["basis"])) for row in members}
        if len(unique) > len({(int(row["anchor"]), str(row["basis"])) for row in best_members}):
            best_target, best_members = target, members
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in best_members:
        key = (int(row["anchor"]), str(row["basis"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    anchor_count = len({int(row["anchor"]) for row in deduped})
    count = len(deduped)
    score = _clip(12.0 * count + 5.0 * min(6, anchor_count))
    return {
        "fib_cluster_count": count,
        "fib_unique_anchor_count": anchor_count,
        "fib_cluster_target_bar": int(best_target),
        "fib_cluster_bars_ahead": int(best_target - current_position),
        "fib_cluster_score": round(score, 1),
        "fib_projection_details": deduped[:24],
    }


def _astro_event_is_active(ephemeris: Mapping[str, Any]) -> bool:
    return bool(
        ephemeris.get("ephemeris_available")
        and (
            _finite(ephemeris.get("moon_phase_score"), 0.0) >= 55.0
            or _finite(ephemeris.get("moon_declination_extreme_score"), 0.0) >= 55.0
            or _finite(ephemeris.get("aspect_cluster_score"), 0.0) >= 40.0
            or len(ephemeris.get("stationary_planets", [])) > 0
            or len(ephemeris.get("ingress_events", [])) > 0
        )
    )


def _historical_confluence_validation(
    index: pd.DatetimeIndex,
    pivots: Sequence[int],
    fib_event_positions: Sequence[int],
    baseline_rate: float,
    config: EOFFConfig,
) -> tuple[float, int, float]:
    """Validate the combined astronomy + Fibonacci event, not astrology alone."""
    if not config.ephemeris_enabled or ephem is None or not fib_event_positions:
        return (np.nan, 0, np.nan)
    pivot_set = {int(position) for position in pivots}
    hits: list[float] = []
    # Recent bounded history keeps Streamlit-free runtime predictable. Body
    # positions are globally cached, so later tickers reuse the same dates.
    for current in list(fib_event_positions)[-120:]:
        if current < 0 or current >= len(index):
            continue
        snapshot = _ephemeris_snapshot(index[current], config)
        if not _astro_event_is_active(snapshot):
            continue
        hit = any(abs(position - current) <= config.historical_window_bars for position in pivot_set)
        hits.append(float(hit))
    if not hits:
        return (np.nan, 0, np.nan)
    precision = 100.0 * float(np.mean(hits))
    lift = precision / baseline_rate if np.isfinite(baseline_rate) and baseline_rate > 0 else np.nan
    return (precision, len(hits), lift)


def _historical_fib_validation(
    pivots: Sequence[int], total_bars: int, config: EOFFConfig,
) -> tuple[float, int, float, float, float, list[int]]:
    """Chronological event precision versus an unconditional pivot baseline.

    At each historical bar only pivots already confirmable at that date are
    allowed as anchors. Consecutive cluster bars are deduplicated into one
    event. The label may look forward only for evaluation, never as a feature.
    """
    pivot_set = sorted({int(position) for position in pivots if 0 <= int(position) < total_bars})
    if len(pivot_set) < 10 or total_bars < 180:
        return (np.nan, 0, np.nan, np.nan, np.nan, [])
    event_hits: list[float] = []
    event_counts: list[float] = []
    event_positions: list[int] = []
    last_event = -10_000
    start = max(100, pivot_set[7] + 3)
    end = max(start, total_bars - config.historical_window_bars)
    for current in range(start, end):
        prior = [position for position in pivot_set if position <= current - 3]
        if len(prior) < 8:
            continue
        candidates = _projection_candidates(prior, current, config)
        members = [
            row for row in candidates
            if abs(int(row["target"]) - current) <= config.historical_window_bars
        ]
        count = len({(int(row["anchor"]), str(row["basis"])) for row in members})
        if count < config.min_fib_cluster:
            continue
        if current - last_event <= config.historical_window_bars:
            continue
        last_event = current
        hit = any(abs(position - current) <= config.historical_window_bars for position in pivot_set)
        event_hits.append(float(hit))
        event_counts.append(float(count))
        event_positions.append(current)
    evaluated_bars = max(1, end - start)
    covered = set()
    for position in pivot_set:
        for bar in range(max(start, position - config.historical_window_bars), min(end, position + config.historical_window_bars + 1)):
            covered.add(bar)
    baseline = 100.0 * len(covered) / evaluated_bars
    if not event_hits:
        return (np.nan, 0, np.nan, baseline, np.nan, [])
    precision = 100.0 * float(np.mean(event_hits))
    lift = precision / baseline if baseline > 0 else np.nan
    return (precision, len(event_hits), float(np.median(event_counts)), baseline, lift, event_positions)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = pd.to_numeric(series, errors="coerce").diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _unfilled_gap_context(df: pd.DataFrame, lookback: int = 90) -> dict[str, Any]:
    recent = df.tail(max(10, lookback)).copy()
    gaps: list[dict[str, Any]] = []
    for index in range(1, len(recent)):
        previous_high = float(recent["High"].iloc[index - 1])
        previous_low = float(recent["Low"].iloc[index - 1])
        current_high = float(recent["High"].iloc[index])
        current_low = float(recent["Low"].iloc[index])
        future = recent.iloc[index + 1:]
        if current_low > previous_high:
            lower, upper = previous_high, current_low
            filled = bool(not future.empty and float(future["Low"].min()) <= lower)
            if not filled:
                gaps.append({"type": "GAP_UP_SUPPORT", "low": lower, "high": upper, "bar": index})
        if current_high < previous_low:
            lower, upper = current_high, previous_low
            filled = bool(not future.empty and float(future["High"].max()) >= upper)
            if not filled:
                gaps.append({"type": "GAP_DOWN_RESISTANCE", "low": lower, "high": upper, "bar": index})
    close = float(recent["Close"].iloc[-1])
    if not gaps:
        return {"nearest_gap_type": "NONE", "nearest_gap_low": np.nan, "nearest_gap_high": np.nan, "nearest_gap_distance": np.nan}
    nearest = min(gaps, key=lambda row: min(abs(close - float(row["low"])), abs(close - float(row["high"]))))
    distance = min(abs(close - float(nearest["low"])), abs(close - float(nearest["high"])))
    return {
        "nearest_gap_type": nearest["type"],
        "nearest_gap_low": float(nearest["low"]),
        "nearest_gap_high": float(nearest["high"]),
        "nearest_gap_distance": float(distance),
    }


def _alternating_wave_targets(
    df: pd.DataFrame, high_pivots: Sequence[int], low_pivots: Sequence[int], atr: float,
) -> dict[str, float | str]:
    points = sorted([(int(pos), "H") for pos in high_pivots] + [(int(pos), "L") for pos in low_pivots])[-8:]
    if len(points) < 3:
        return {"wave_structure": "UNAVAILABLE", "wave_target_100": np.nan, "wave_target_1618": np.nan, "wave_score": 0.0}
    last_three = points[-3:]
    labels = "".join(label for _, label in last_three)
    prices = [float(df["High"].iloc[pos]) if label == "H" else float(df["Low"].iloc[pos]) for pos, label in last_three]
    close = float(df["Close"].iloc[-1])
    if labels == "LHL":
        impulse = max(0.0, prices[1] - prices[0])
        target_100 = prices[2] + impulse
        target_1618 = prices[2] + 1.618 * impulse
        direction = "BULLISH_ABC"
    elif labels == "HLH":
        impulse = max(0.0, prices[0] - prices[1])
        target_100 = prices[2] - impulse
        target_1618 = prices[2] - 1.618 * impulse
        direction = "BEARISH_ABC"
    else:
        return {"wave_structure": labels, "wave_target_100": np.nan, "wave_target_1618": np.nan, "wave_score": 0.0}
    proximity = min(abs(close - target_100), abs(close - target_1618)) / max(atr, 1e-9)
    score = _clip(100.0 - 30.0 * proximity)
    return {
        "wave_structure": direction,
        "wave_target_100": float(target_100),
        "wave_target_1618": float(target_1618),
        "wave_score": score,
    }


def _price_pattern_momentum(df: pd.DataFrame, high_pivots: Sequence[int], low_pivots: Sequence[int]) -> dict[str, Any]:
    close = float(df["Close"].iloc[-1])
    high = float(df["High"].iloc[-1])
    low = float(df["Low"].iloc[-1])
    open_price = float(df.get("Open", df["Close"]).iloc[-1])
    atr_series = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1).rolling(14).mean()
    atr = _finite(atr_series.iloc[-1], max(high - low, 1e-9))
    ema20 = float(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(df["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
    rsi14 = _finite(_rsi(df["Close"], 14).iloc[-1], 50.0)
    macd = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    macd_hist = macd - macd.ewm(span=9, adjust=False).mean()
    hist_now = _finite(macd_hist.iloc[-1], 0.0)
    hist_prev = _finite(macd_hist.iloc[-3], hist_now)

    prior_high20 = float(df["High"].iloc[-21:-1].max()) if len(df) > 21 else high
    prior_low20 = float(df["Low"].iloc[-21:-1].min()) if len(df) > 21 else low
    sell_side_sweep = bool(low < prior_low20 and close > prior_low20)
    buy_side_sweep = bool(high > prior_high20 and close < prior_high20)
    body = abs(close - open_price)
    lower_wick = max(0.0, min(open_price, close) - low)
    upper_wick = max(0.0, high - max(open_price, close))
    bullish_rejection = bool(lower_wick >= max(body, 0.15 * atr) * 1.25 and close >= (high + low) / 2.0)
    bearish_rejection = bool(upper_wick >= max(body, 0.15 * atr) * 1.25 and close <= (high + low) / 2.0)

    last_low_value = float(df["Low"].iloc[low_pivots[-1]]) if low_pivots else float(df["Low"].tail(55).min())
    last_high_value = float(df["High"].iloc[high_pivots[-1]]) if high_pivots else float(df["High"].tail(55).max())
    swing_low = min(last_low_value, last_high_value)
    swing_high = max(last_low_value, last_high_value)
    swing_range = max(swing_high - swing_low, atr)
    retracement = (close - swing_low) / swing_range
    fib_levels = {
        "fib_382": swing_high - 0.382 * swing_range,
        "fib_500": swing_high - 0.500 * swing_range,
        "fib_618": swing_high - 0.618 * swing_range,
        "fib_786": swing_high - 0.786 * swing_range,
        "ext_1272": swing_low + 1.272 * swing_range,
        "ext_1618": swing_low + 1.618 * swing_range,
    }
    nearby_fib = min(abs(close - value) / max(atr, 1e-9) for value in fib_levels.values())
    fib_price_score = _clip(100.0 - 45.0 * nearby_fib)
    harmonic_levels = [
        swing_low + 0.618 * swing_range, swing_low + 0.786 * swing_range,
        swing_low + 1.272 * swing_range, swing_low + 1.618 * swing_range,
    ]
    harmonic_distance = min(abs(close - value) for value in harmonic_levels) / max(atr, 1e-9)
    harmonic_price_score = _clip(100.0 - 40.0 * harmonic_distance)

    envelope_width = max(1.25 * atr, 0.025 * max(close, 1.0))
    ma_envelope_lower = ema20 - envelope_width
    ma_envelope_upper = ema20 + envelope_width
    envelope_distance = min(abs(close - ma_envelope_lower), abs(close - ma_envelope_upper)) / max(atr, 1e-9)
    ma_envelope_score = _clip(100.0 - 42.0 * envelope_distance)
    gap = _unfilled_gap_context(df)
    gap_distance_atr = _finite(gap.get("nearest_gap_distance"), np.nan) / max(atr, 1e-9)
    gap_confluence_score = _clip(100.0 - 42.0 * gap_distance_atr) if np.isfinite(gap_distance_atr) else 0.0
    wave = _alternating_wave_targets(df, high_pivots, low_pivots, atr)

    support_candidates = [float(df["Low"].tail(20).min()), float(df["Low"].tail(55).min()), ema20, ema50, ma_envelope_lower]
    resistance_candidates = [float(df["High"].tail(20).max()), float(df["High"].tail(55).max()), ma_envelope_upper]
    if gap.get("nearest_gap_type") == "GAP_UP_SUPPORT":
        support_candidates.extend([_finite(gap.get("nearest_gap_low"), np.nan), _finite(gap.get("nearest_gap_high"), np.nan)])
    elif gap.get("nearest_gap_type") == "GAP_DOWN_RESISTANCE":
        resistance_candidates.extend([_finite(gap.get("nearest_gap_low"), np.nan), _finite(gap.get("nearest_gap_high"), np.nan)])
    support_distance = min(abs(close - value) for value in support_candidates if np.isfinite(value)) / max(atr, 1e-9)
    resistance_distance = min(abs(value - close) for value in resistance_candidates if np.isfinite(value)) / max(atr, 1e-9)
    support_score = _clip(100.0 - 40.0 * support_distance)
    resistance_score = _clip(100.0 - 40.0 * resistance_distance)

    trend_bull = 100.0 if close > ema20 > ema50 > ema200 else 78.0 if close > ema50 > ema200 else 58.0 if close > ema50 else 28.0
    trend_bear = 100.0 - trend_bull
    bullish_momentum = _clip(
        50.0 + 1.4 * (rsi14 - 50.0) + (18.0 if hist_now > hist_prev else -8.0)
    )
    bearish_momentum = _clip(100.0 - bullish_momentum)
    bullish_exhaustion = _clip(
        (80.0 if rsi14 <= 34.0 else 50.0 if rsi14 <= 42.0 else 20.0)
        + (20.0 if hist_now > hist_prev else 0.0)
        + (25.0 if sell_side_sweep else 0.0)
        + (15.0 if bullish_rejection else 0.0)
    )
    bearish_exhaustion = _clip(
        (80.0 if rsi14 >= 66.0 else 50.0 if rsi14 >= 58.0 else 20.0)
        + (20.0 if hist_now < hist_prev else 0.0)
        + (25.0 if buy_side_sweep else 0.0)
        + (15.0 if bearish_rejection else 0.0)
    )
    double_bottom = bool(
        len(low_pivots) >= 2
        and abs(float(df["Low"].iloc[low_pivots[-1]]) - float(df["Low"].iloc[low_pivots[-2]])) <= 1.25 * atr
        and low_pivots[-1] - low_pivots[-2] >= 5
    )
    double_top = bool(
        len(high_pivots) >= 2
        and abs(float(df["High"].iloc[high_pivots[-1]]) - float(df["High"].iloc[high_pivots[-2]])) <= 1.25 * atr
        and high_pivots[-1] - high_pivots[-2] >= 5
    )
    atr20 = _finite(atr_series.tail(20).mean(), atr)
    atr60 = _finite(atr_series.tail(60).mean(), atr20)
    compression_score = _clip(100.0 * (1.0 - atr20 / max(atr60 * 1.15, 1e-9)))
    pattern_bull = _clip(
        0.24 * support_score + 0.18 * fib_price_score + 0.12 * harmonic_price_score
        + 0.10 * ma_envelope_score + 0.08 * gap_confluence_score + 0.08 * _finite(wave.get("wave_score"), 0.0)
        + 12.0 * sell_side_sweep + 10.0 * bullish_rejection + 8.0 * double_bottom
    )
    pattern_bear = _clip(
        0.24 * resistance_score + 0.18 * fib_price_score + 0.12 * harmonic_price_score
        + 0.10 * ma_envelope_score + 0.08 * gap_confluence_score + 0.08 * _finite(wave.get("wave_score"), 0.0)
        + 12.0 * buy_side_sweep + 10.0 * bearish_rejection + 8.0 * double_top
    )
    bullish_direction = _clip(0.30 * trend_bull + 0.25 * bullish_momentum + 0.25 * pattern_bull + 0.20 * bullish_exhaustion)
    bearish_direction = _clip(0.30 * trend_bear + 0.25 * bearish_momentum + 0.25 * pattern_bear + 0.20 * bearish_exhaustion)
    if bullish_direction >= bearish_direction + 8.0:
        direction = "BULLISH"
    elif bearish_direction >= bullish_direction + 8.0:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"
    return {
        "close": close, "atr": atr, "ema20": ema20, "ema50": ema50, "ema200": ema200,
        "rsi14": round(rsi14, 2), "macd_hist": hist_now,
        "sell_side_sweep": sell_side_sweep, "buy_side_sweep": buy_side_sweep,
        "bullish_rejection": bullish_rejection, "bearish_rejection": bearish_rejection,
        "support_score": round(support_score, 1), "resistance_score": round(resistance_score, 1),
        "fib_price_score": round(fib_price_score, 1), "price_retracement_fraction": round(retracement, 4),
        "trend_bull_score": round(trend_bull, 1), "trend_bear_score": round(trend_bear, 1),
        "bullish_momentum_score": round(bullish_momentum, 1), "bearish_momentum_score": round(bearish_momentum, 1),
        "bullish_exhaustion_score": round(bullish_exhaustion, 1), "bearish_exhaustion_score": round(bearish_exhaustion, 1),
        "bullish_pattern_score": round(pattern_bull, 1), "bearish_pattern_score": round(pattern_bear, 1),
        "bullish_direction_score": round(bullish_direction, 1), "bearish_direction_score": round(bearish_direction, 1),
        "eoff_direction_bias": direction,
        "swing_low": round(swing_low, 4), "swing_high": round(swing_high, 4),
        "harmonic_price_score": round(harmonic_price_score, 1),
        "ma_envelope_lower": round(ma_envelope_lower, 4), "ma_envelope_upper": round(ma_envelope_upper, 4),
        "ma_envelope_score": round(ma_envelope_score, 1),
        "nearest_gap_type": str(gap.get("nearest_gap_type") or "NONE"),
        "nearest_gap_low": _finite(gap.get("nearest_gap_low"), np.nan),
        "nearest_gap_high": _finite(gap.get("nearest_gap_high"), np.nan),
        "gap_confluence_score": round(gap_confluence_score, 1),
        "double_bottom": double_bottom, "double_top": double_top,
        "compression_score": round(compression_score, 1),
        "wave_structure": str(wave.get("wave_structure") or "UNAVAILABLE"),
        "wave_target_100": _finite(wave.get("wave_target_100"), np.nan),
        "wave_target_1618": _finite(wave.get("wave_target_1618"), np.nan),
        "wave_score": round(_finite(wave.get("wave_score"), 0.0), 1),
        **{key: round(value, 4) for key, value in fib_levels.items()},
    }


def _timestamp_from_bar(index: pd.DatetimeIndex, target_bar: int) -> pd.Timestamp:
    if target_bar < len(index):
        return pd.Timestamp(index[max(0, target_bar)])
    ahead = target_bar - (len(index) - 1)
    return pd.Timestamp(index[-1]).normalize() + pd.offsets.BDay(max(0, ahead))


def _date_from_bar(index: pd.DatetimeIndex, target_bar: int) -> str:
    return _timestamp_from_bar(index, target_bar).date().isoformat()


def _price_roadmap(df: pd.DataFrame, context: Mapping[str, Any], cluster: Mapping[str, Any], strength: str) -> list[dict[str, Any]]:
    close = _finite(context.get("close"), np.nan)
    atr = max(_finite(context.get("atr"), 0.0), 1e-9)
    direction = str(context.get("eoff_direction_bias") or "NEUTRAL")
    target_bar = int(_finite(cluster.get("fib_cluster_target_bar"), len(df) - 1))
    window_date = _date_from_bar(df.index, target_bar)
    if direction == "BULLISH":
        support = min(_finite(context.get("fib_618"), close), _finite(context.get("ema50"), close))
        first_target = max(_finite(context.get("swing_high"), close + atr), close + 1.5 * atr)
        second_target = max(_finite(context.get("ext_1272"), first_target + atr), first_target)
        invalidation = min(_finite(context.get("swing_low"), close - 2.0 * atr), close - 1.2 * atr)
        stages = [
            ("ACCUMULATION_OR_REVERSAL", window_date, support, close, invalidation),
            ("PRIMARY_TARGET", window_date, first_target, first_target, invalidation),
            ("EXTENSION_TARGET", window_date, second_target, second_target, invalidation),
        ]
    elif direction == "BEARISH":
        resistance = max(_finite(context.get("fib_382"), close), _finite(context.get("ema20"), close))
        first_target = min(_finite(context.get("swing_low"), close - atr), close - 1.5 * atr)
        second_target = first_target - 1.2 * atr
        invalidation = max(_finite(context.get("swing_high"), close + 2.0 * atr), close + 1.2 * atr)
        stages = [
            ("DISTRIBUTION_OR_TOP", window_date, close, resistance, invalidation),
            ("PRIMARY_DOWNSIDE", window_date, first_target, first_target, invalidation),
            ("EXTENDED_DOWNSIDE", window_date, second_target, second_target, invalidation),
        ]
    else:
        lower = close - 1.25 * atr
        upper = close + 1.25 * atr
        stages = [("WAIT_DIRECTION_CONFIRMATION", window_date, lower, upper, np.nan)]
    confidence_map = {"VERY_STRONG": 85.0, "STRONG": 74.0, "MEDIUM": 62.0, "LOW": 45.0}
    return [
        {
            "phase": phase, "date_window_anchor": date,
            "price_zone_low": round(min(zone_low, zone_high), 4),
            "price_zone_high": round(max(zone_low, zone_high), 4),
            "invalidation": round(invalidation, 4) if np.isfinite(invalidation) else np.nan,
            "scenario_confidence": confidence_map.get(strength, 45.0),
        }
        for phase, date, zone_low, zone_high, invalidation in stages
    ]


def analyze_eoff_reconstruction(
    frame: pd.DataFrame | None,
    *,
    config: EOFFConfig | None = None,
) -> dict[str, Any]:
    cfg = config or EOFFConfig()
    unavailable = {
        "eoff_version": EOFF_VERSION,
        "eoff_state": "DISABLED" if not cfg.enabled else "INSUFFICIENT_HISTORY",
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
        "eoff_fib_cluster_score": 0.0,
        "eoff_historical_hit_rate": np.nan,
        "eoff_historical_events": 0,
        "eoff_historical_median_cluster": np.nan,
        "eoff_historical_baseline_rate": np.nan,
        "eoff_historical_lift": np.nan,
        "eoff_confluence_historical_hit_rate": np.nan,
        "eoff_confluence_historical_events": 0,
        "eoff_confluence_historical_lift": np.nan,
        "eoff_bars_to_cluster": np.nan,
        "eoff_reversal_date": "",
        "eoff_ephemeris_state": "DISABLED" if not cfg.ephemeris_enabled else "UNAVAILABLE",
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
        "eoff_roadmap": [],
        "eoff_roadmap_json": "[]",
        "eoff_explanation": "Riwayat belum cukup untuk rekonstruksi Eye-of-Future clean-room.",
    }
    if not cfg.enabled:
        return unavailable
    df = _clean_frame(frame)
    if len(df) < 180:
        return unavailable

    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    high_pivots = _confirmed_pivots(highs, mode="high")
    low_pivots = _confirmed_pivots(lows, mode="low")
    all_pivots = sorted(set(high_pivots + low_pivots))
    current = len(df) - 1
    candidates = _projection_candidates(all_pivots, current, cfg)
    cluster = _best_projection_cluster(candidates, current, cfg)
    historical_hit, historical_events, historical_median, historical_baseline, historical_lift, fib_event_positions = _historical_fib_validation(
        all_pivots, len(df), cfg,
    )
    confluence_hit, confluence_events, confluence_lift = _historical_confluence_validation(
        df.index, all_pivots, fib_event_positions, historical_baseline, cfg,
    )
    projected_bar = int(_finite(cluster.get("fib_cluster_target_bar"), current))
    projected_timestamp = _timestamp_from_bar(df.index, projected_bar)
    ephemeris = _ephemeris_snapshot(projected_timestamp, cfg)
    context = _price_pattern_momentum(df, high_pivots, low_pivots)

    fib_score = _clip(cluster.get("fib_cluster_score"))
    cycle_evidence = 50.0 if not np.isfinite(historical_hit) else historical_hit
    astro_score = _clip(ephemeris.get("astro_cluster_score"))
    time_power = _clip(
        0.45 * fib_score + 0.20 * cycle_evidence + 0.25 * astro_score
        + 0.10 * _clip(ephemeris.get("sun_annual_cycle_score"), 0.0, 100.0)
    )
    direction = str(context.get("eoff_direction_bias") or "NEUTRAL")
    if direction == "BULLISH":
        pattern_score = _clip(context.get("bullish_pattern_score"))
        momentum_score = _clip(0.55 * _finite(context.get("bullish_momentum_score"), 50.0) + 0.45 * _finite(context.get("bullish_exhaustion_score"), 50.0))
        price_score = _clip(
            0.30 * _finite(context.get("support_score"), 50.0)
            + 0.20 * _finite(context.get("fib_price_score"), 50.0)
            + 0.12 * _finite(context.get("harmonic_price_score"), 50.0)
            + 0.10 * _finite(context.get("ma_envelope_score"), 50.0)
            + 0.08 * _finite(context.get("gap_confluence_score"), 0.0)
            + 0.08 * _finite(context.get("wave_score"), 0.0)
            + 0.12 * _finite(context.get("trend_bull_score"), 50.0)
        )
    elif direction == "BEARISH":
        pattern_score = _clip(context.get("bearish_pattern_score"))
        momentum_score = _clip(0.55 * _finite(context.get("bearish_momentum_score"), 50.0) + 0.45 * _finite(context.get("bearish_exhaustion_score"), 50.0))
        price_score = _clip(
            0.30 * _finite(context.get("resistance_score"), 50.0)
            + 0.20 * _finite(context.get("fib_price_score"), 50.0)
            + 0.12 * _finite(context.get("harmonic_price_score"), 50.0)
            + 0.10 * _finite(context.get("ma_envelope_score"), 50.0)
            + 0.08 * _finite(context.get("gap_confluence_score"), 0.0)
            + 0.08 * _finite(context.get("wave_score"), 0.0)
            + 0.12 * _finite(context.get("trend_bear_score"), 50.0)
        )
    else:
        pattern_score = max(_clip(context.get("bullish_pattern_score")), _clip(context.get("bearish_pattern_score")))
        momentum_score = 50.0
        price_score = max(_clip(context.get("support_score")), _clip(context.get("resistance_score")))

    reconstruction = _clip(0.36 * time_power + 0.24 * price_score + 0.20 * pattern_score + 0.20 * momentum_score)
    astro_event_active = _astro_event_is_active(ephemeris)
    fib_active = int(cluster.get("fib_cluster_count", 0)) >= max(1, cfg.min_fib_cluster)
    if cfg.require_astro_fib_confluence:
        historical_ok = bool(
            confluence_events >= cfg.historical_min_events
            and np.isfinite(confluence_hit)
            and np.isfinite(confluence_lift)
            and confluence_lift >= 1.05
            and confluence_hit >= max(20.0, historical_baseline + 2.0)
        )
    else:
        historical_ok = bool(
            historical_events >= cfg.historical_min_events
            and np.isfinite(historical_hit)
            and np.isfinite(historical_lift)
            and historical_lift >= 1.05
            and historical_hit >= max(20.0, historical_baseline + 2.0)
        )
    signal_active = bool(fib_active and historical_ok and (astro_event_active or not cfg.require_astro_fib_confluence))
    # Classification describes confluence strength, not a guaranteed probability.
    if signal_active and reconstruction >= 82.0 and int(cluster.get("fib_cluster_count", 0)) >= cfg.min_fib_cluster + 3:
        strength = "VERY_STRONG"
    elif signal_active and reconstruction >= 72.0:
        strength = "STRONG"
    elif fib_active and reconstruction >= 60.0:
        strength = "MEDIUM"
    else:
        strength = "LOW"
    state = "ACTIVE_GUARDED" if signal_active else "SHADOW_NO_CONFLUENCE" if ephemeris.get("ephemeris_available") else "EPHEMERIS_UNAVAILABLE"
    target_bar = projected_bar
    reversal_date = _date_from_bar(df.index, target_bar)
    roadmap = _price_roadmap(df, context, cluster, strength)

    aspects_text = "; ".join(
        f"{row['pair']} {int(float(row['aspect_deg']))}° orb {float(row['orb_deg']):.2f}°"
        for row in ephemeris.get("active_aspects", [])[:8]
    )
    explanation = (
        f"Clean-room EOFF {strength}: fib cluster {int(cluster.get('fib_cluster_count', 0))} proyeksi/"
        f"{int(cluster.get('fib_unique_anchor_count', 0))} anchor; time power {time_power:.1f}; "
        f"astro {astro_score:.1f}; price {price_score:.1f}; pattern {pattern_score:.1f}; "
        f"momentum {momentum_score:.1f}; direction {direction}; target window {reversal_date}; "
        f"ephemeris evaluated on {projected_timestamp.date().isoformat()}."
    )
    return {
        "eoff_version": EOFF_VERSION,
        "eoff_state": state,
        "eoff_reconstruction_score": round(reconstruction, 1),
        "eoff_strength_label": strength,
        "eoff_signal_active": signal_active,
        "eoff_direction_bias": direction,
        "eoff_time_power_score": round(time_power, 1),
        "eoff_price_power_score": round(price_score, 1),
        "eoff_pattern_score": round(pattern_score, 1),
        "eoff_momentum_score": round(momentum_score, 1),
        "eoff_astro_score": round(astro_score, 1),
        "eoff_fib_cluster_count": int(cluster.get("fib_cluster_count", 0)),
        "eoff_fib_unique_anchor_count": int(cluster.get("fib_unique_anchor_count", 0)),
        "eoff_fib_cluster_score": round(fib_score, 1),
        "eoff_fib_projection_details": cluster.get("fib_projection_details", []),
        "eoff_historical_hit_rate": round(historical_hit, 1) if np.isfinite(historical_hit) else np.nan,
        "eoff_historical_events": int(historical_events),
        "eoff_historical_median_cluster": round(historical_median, 1) if np.isfinite(historical_median) else np.nan,
        "eoff_historical_baseline_rate": round(historical_baseline, 1) if np.isfinite(historical_baseline) else np.nan,
        "eoff_historical_lift": round(historical_lift, 2) if np.isfinite(historical_lift) else np.nan,
        "eoff_confluence_historical_hit_rate": round(confluence_hit, 1) if np.isfinite(confluence_hit) else np.nan,
        "eoff_confluence_historical_events": int(confluence_events),
        "eoff_confluence_historical_lift": round(confluence_lift, 2) if np.isfinite(confluence_lift) else np.nan,
        "eoff_bars_to_cluster": int(_finite(cluster.get("fib_cluster_bars_ahead"), 0.0)),
        "eoff_reversal_date": reversal_date,
        "eoff_ephemeris_state": str(ephemeris.get("ephemeris_state") or "UNAVAILABLE"),
        "eoff_ephemeris_date": projected_timestamp.date().isoformat(),
        "eoff_astro_events": "; ".join(str(value) for value in ephemeris.get("astro_events", [])),
        "eoff_active_aspects": aspects_text,
        "eoff_retrograde_planets": "; ".join(str(value) for value in ephemeris.get("retrograde_planets", [])),
        "eoff_stationary_planets": "; ".join(str(value) for value in ephemeris.get("stationary_planets", [])),
        "eoff_ingress_events": "; ".join(str(value) for value in ephemeris.get("ingress_events", [])),
        "eoff_moon_declination_deg": ephemeris.get("moon_declination_deg"),
        "eoff_moon_declination_extreme_score": ephemeris.get("moon_declination_extreme_score"),
        "eoff_moon_declination_turning": bool(ephemeris.get("moon_declination_turning")),
        "eoff_moon_phase": ephemeris.get("moon_phase_name"),
        "eoff_moon_phase_angle_deg": ephemeris.get("moon_phase_angle_deg"),
        "eoff_sun_sign": ephemeris.get("sun_sign"),
        "eoff_sun_annual_cycle_bias": ephemeris.get("sun_annual_cycle_bias"),
        "eoff_roadmap": roadmap,
        "eoff_roadmap_json": json.dumps(roadmap, ensure_ascii=False),
        "eoff_explanation": explanation,
        **{key: value for key, value in context.items() if key not in {"close", "atr"}},
    }


def setup_eoff_alignment(analysis: Mapping[str, Any], setup: str) -> float:
    direction = str(analysis.get("eoff_direction_bias") or "NEUTRAL")
    score = _finite(analysis.get("eoff_reconstruction_score"), 50.0)
    name = str(setup or "").upper()
    bullish_setup = name in {
        "PULLBACK_CONTINUATION", "BREAKOUT_RETEST", "REVERSAL_ACCUMULATION",
        "UNICORN_SNIPER_ICT", "UNICORN_ICT", "SNIPER",
    }
    if bullish_setup and direction == "BEARISH":
        return max(0.0, 100.0 - score)
    if bullish_setup and direction == "BULLISH":
        return score
    return 50.0 + 0.35 * (score - 50.0)


__all__ = [
    "EOFF_VERSION", "EOFFConfig", "analyze_eoff_reconstruction",
    "setup_eoff_alignment", "ephem",
]

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
from datetime import timedelta as _calendar_timedelta

import numpy as np
import pandas as pd

try:  # Optional so the scanner core can still import if dependency is absent.
    import ephem  # type: ignore
except Exception:  # pragma: no cover - explicitly tested through monkeypatch.
    ephem = None

EOFF_VERSION = "6.6.7-public-prior-astro-weighting"
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
ADAPTIVE_ASTRO_FAMILIES = (
    "MOON_DECLINATION", "INGRESS", "RETROGRADE", "SUN_ANNUAL",
)
# Clean-room public-prior reconstruction.  These are NOT claimed to be the
# proprietary Astronacci weights.  They encode the relative prominence found
# in public Astronacci material and the Goeyardi-Hady-Ghozali research: lunar
# phase and planetary aspects are primary timing families; declination and the
# Sun cycle are material secondary families; ingress and retrograde are event
# modifiers.  The basket always sums to 100%.
PUBLIC_ASTRO_PRIOR_WEIGHTS = {
    "MOON_PHASE": 0.25,
    "PLANETARY_ASPECT": 0.25,
    "MOON_DECLINATION": 0.15,
    "INGRESS": 0.10,
    "RETROGRADE": 0.10,
    "SUN_ANNUAL": 0.15,
}
# Walk-forward evidence modulates the public prior but can no longer erase a
# family.  Weak evidence reduces a factor to 75% of its prior; validated skill
# may raise it to at most 125%.  The normalized basket remains 100%.
ADAPTIVE_FAMILY_CAPS = {
    "MOON_DECLINATION": 0.15,
    "INGRESS": 0.10,
    "RETROGRADE": 0.10,
    "SUN_ANNUAL": 0.15,
}


@dataclass(frozen=True)
class EOFFConfig:
    enabled: bool = True
    ephemeris_enabled: bool = True
    min_fib_cluster: int = 4
    fib_window_bars: int = 2
    aspect_orb_deg: float = 3.0
    station_speed_deg_day: float = 0.12
    historical_min_events: int = 12
    historical_window_bars: int = 3
    directional_forward_bars: int = 5
    directional_min_return_pct: float = 0.005
    directional_min_lift: float = 1.10
    max_anchor_pivots: int = 14
    future_horizon_bars: int = 34
    require_astro_fib_confluence: bool = True
    adaptive_astro_enabled: bool = True
    adaptive_min_train_events: int = 10
    adaptive_min_oos_events: int = 6
    adaptive_min_reversal_lift: float = 1.05
    adaptive_min_forward_hit_pct: float = 52.0
    adaptive_min_reversal_edge_pct: float = 3.0
    adaptive_max_total_share: float = 0.50
    public_prior_min_multiplier: float = 0.75
    public_prior_max_multiplier: float = 1.25
    secondary_prior_min_active_factors: int = 3
    secondary_prior_min_astro_score: float = 60.0
    secondary_prior_min_price_score: float = 65.0
    secondary_prior_min_pattern_score: float = 55.0
    secondary_prior_min_momentum_score: float = 55.0


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


def _normalized_calendar_date(timestamp: pd.Timestamp | str):
    """Return a plain Python date to avoid NumPy generic-timedelta warnings.

    Streamlit Cloud may install a newer NumPy than the local development
    runtime.  Performing +/- one day on a Timestamp backed by a generic
    ``datetime64`` unit emits thousands of warnings during ephemeris scans.
    Calendar arithmetic does not require nanosecond precision, so converting
    once to ``datetime.date`` is both exact and version-stable.
    """
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.date()


def _stamp_key(timestamp: pd.Timestamp | str) -> str:
    return _normalized_calendar_date(timestamp).strftime("%Y/%m/%d 12:00:00")


def _shifted_stamp_key(timestamp: pd.Timestamp | str, days: int) -> str:
    shifted = _normalized_calendar_date(timestamp) + _calendar_timedelta(days=int(days))
    return shifted.strftime("%Y/%m/%d 12:00:00")


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
        "retrograde_transition_events": [],
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
    previous_key = _shifted_stamp_key(timestamp, -1)
    next_key = _shifted_stamp_key(timestamp, 1)
    longitudes: dict[str, float] = {}
    declinations: dict[str, float] = {}
    motion: dict[str, float] = {}
    retrograde: list[str] = []
    stationary: list[str] = []
    retrograde_transitions: list[str] = []
    ingress: list[str] = []
    for name in PLANET_NAMES:
        longitude, declination = _body_position(name, current_key)
        prev_longitude, _ = _body_position(name, previous_key)
        next_longitude, _ = _body_position(name, next_key)
        longitudes[name] = longitude
        declinations[name] = declination
        if np.isfinite(prev_longitude) and np.isfinite(next_longitude):
            previous_step = _signed_angle_delta(longitude, prev_longitude)
            next_step = _signed_angle_delta(next_longitude, longitude)
            daily_motion = (previous_step + next_step) / 2.0
        else:
            previous_step = np.nan
            next_step = np.nan
            daily_motion = np.nan
        motion[name] = daily_motion
        if name not in {"Sun", "Moon"} and np.isfinite(daily_motion):
            if daily_motion < -config.station_speed_deg_day:
                retrograde.append(name)
            if abs(daily_motion) <= config.station_speed_deg_day:
                stationary.append(name)
            if np.isfinite(previous_step) and np.isfinite(next_step) and previous_step * next_step < 0:
                transition = "RETROGRADE_START" if next_step < 0 else "DIRECT_START"
                retrograde_transitions.append(f"{name}:{transition}")
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
    events.extend(f"RETROGRADE_TRANSITION:{name}" for name in retrograde_transitions)
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
        "retrograde_transition_events": retrograde_transitions,
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


def _public_supported_astro_event(ephemeris: Mapping[str, Any]) -> bool:
    """Fixed public core: lunar phase and major planetary aspects only."""
    return bool(
        ephemeris.get("ephemeris_available")
        and (
            _finite(ephemeris.get("moon_phase_score"), 0.0) >= 55.0
            or _finite(ephemeris.get("aspect_cluster_score"), 0.0) >= 40.0
        )
    )


def _adaptive_family_signal(
    ephemeris: Mapping[str, Any], family: str, direction: str,
) -> dict[str, Any]:
    """Return the current event state without assigning an unconditional weight.

    These families are timing/intensity features.  They never determine market
    direction; direction is supplied by the causal price/pattern/momentum layer.
    """
    name = str(family or "").upper()
    side = str(direction or "NEUTRAL").upper()
    if not ephemeris.get("ephemeris_available"):
        return {"active": False, "score": 50.0, "detail": "EPHEMERIS_UNAVAILABLE"}
    if name == "MOON_DECLINATION":
        extreme = _clip(ephemeris.get("moon_declination_extreme_score"))
        turning = bool(ephemeris.get("moon_declination_turning"))
        active = bool(extreme >= 55.0 or turning)
        score = max(extreme, 68.0 if turning else 0.0) if active else 50.0
        detail = "EXTREME_TURNING" if turning else "EXTREME" if active else "INACTIVE"
    elif name == "INGRESS":
        events = list(ephemeris.get("ingress_events", []) or [])
        active = bool(events)
        score = _clip(58.0 + 9.0 * min(4, len(events))) if active else 50.0
        detail = ";".join(str(value) for value in events[:6]) or "INACTIVE"
    elif name == "RETROGRADE":
        transitions = list(ephemeris.get("retrograde_transition_events", []) or [])
        stationary = list(ephemeris.get("stationary_planets", []) or [])
        retrograde = list(ephemeris.get("retrograde_planets", []) or [])
        active = bool(transitions or stationary or retrograde)
        if transitions or stationary:
            score = _clip(64.0 + 12.0 * min(3, len(transitions)) + 6.0 * min(3, len(stationary)))
        elif retrograde:
            # Ongoing retrograde is a lower-intensity background condition;
            # station/transition dates remain the high-intensity marker.
            score = _clip(54.0 + 4.0 * min(4, len(retrograde)))
        else:
            score = 50.0
        detail = ";".join([*(str(value) for value in transitions[:4]), *(f"STATION:{value}" for value in stationary[:4]), *(f"RETROGRADE:{value}" for value in retrograde[:4])]) or "INACTIVE"
    elif name == "SUN_ANNUAL":
        bias = str(ephemeris.get("sun_annual_cycle_bias") or "NEUTRAL").upper()
        aligned = (side == "BULLISH" and bias == "TRADING_BIAS") or (side == "BEARISH" and bias == "DEFENSIVE_BIAS")
        opposed = (side == "BULLISH" and bias == "DEFENSIVE_BIAS") or (side == "BEARISH" and bias == "TRADING_BIAS")
        active = bool(side in {"BULLISH", "BEARISH"} and bias != "NEUTRAL")
        score = 68.0 if aligned else 32.0 if opposed else 50.0
        detail = f"{bias}:{'ALIGNED' if aligned else 'OPPOSED' if opposed else 'NEUTRAL'}"
    else:
        return {"active": False, "score": 50.0, "detail": "UNKNOWN_FAMILY"}
    return {"active": bool(active), "score": round(_clip(score), 1), "detail": detail}


def _record_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "events": 0.0, "reversal_hit_rate": np.nan, "baseline_rate": np.nan,
            "lift": np.nan, "forward_hit_rate": np.nan,
            "median_directional_return_pct": np.nan,
        }
    reversal = np.asarray([_finite(row.get("reversal_hit"), np.nan) for row in records], dtype=float)
    baseline = np.asarray([_finite(row.get("baseline_rate"), np.nan) for row in records], dtype=float)
    forward = np.asarray([_finite(row.get("forward_hit"), np.nan) for row in records], dtype=float)
    returns = np.asarray([_finite(row.get("directional_return_pct"), np.nan) for row in records], dtype=float)
    valid_reversal = reversal[np.isfinite(reversal)]
    valid_baseline = baseline[np.isfinite(baseline)]
    valid_forward = forward[np.isfinite(forward)]
    valid_returns = returns[np.isfinite(returns)]
    hit_rate = 100.0 * float(np.mean(valid_reversal)) if valid_reversal.size else np.nan
    baseline_rate = float(np.mean(valid_baseline)) if valid_baseline.size else np.nan
    lift = hit_rate / baseline_rate if np.isfinite(hit_rate) and np.isfinite(baseline_rate) and baseline_rate > 0 else np.nan
    return {
        "events": float(min(len(valid_reversal), len(valid_forward))),
        "reversal_hit_rate": hit_rate,
        "baseline_rate": baseline_rate,
        "lift": lift,
        "forward_hit_rate": 100.0 * float(np.mean(valid_forward)) if valid_forward.size else np.nan,
        "median_directional_return_pct": float(np.median(valid_returns)) if valid_returns.size else np.nan,
    }


def _factor_gate(metrics: Mapping[str, Any], config: EOFFConfig, minimum_events: int) -> bool:
    events = int(_finite(metrics.get("events"), 0.0))
    hit = _finite(metrics.get("reversal_hit_rate"), np.nan)
    baseline = _finite(metrics.get("baseline_rate"), np.nan)
    lift = _finite(metrics.get("lift"), np.nan)
    forward = _finite(metrics.get("forward_hit_rate"), np.nan)
    median_return = _finite(metrics.get("median_directional_return_pct"), np.nan)
    return bool(
        events >= max(1, int(minimum_events))
        and np.isfinite(hit) and np.isfinite(baseline) and np.isfinite(lift)
        and lift >= config.adaptive_min_reversal_lift
        and hit >= baseline + config.adaptive_min_reversal_edge_pct
        and np.isfinite(forward) and forward >= config.adaptive_min_forward_hit_pct
        and np.isfinite(median_return) and median_return > 0.0
    )


def _walk_forward_factor_metrics(
    records: Sequence[Mapping[str, Any]], config: EOFFConfig, cap_fraction: float,
) -> dict[str, Any]:
    """Expanding-window gate: the current event is never used to approve itself."""
    ordered = sorted((dict(row) for row in records), key=lambda row: int(_finite(row.get("position"), 0.0)))
    oos: list[dict[str, Any]] = []
    gate_history: list[bool] = []
    minimum_train = max(4, int(config.adaptive_min_train_events))
    for index in range(minimum_train, len(ordered)):
        prior = ordered[:index]
        prior_metrics = _record_metrics(prior)
        approved_before_event = _factor_gate(prior_metrics, config, minimum_train)
        gate_history.append(approved_before_event)
        if approved_before_event:
            oos.append(ordered[index])
    total_metrics = _record_metrics(ordered)
    oos_metrics = _record_metrics(oos)
    validated = bool(
        _factor_gate(oos_metrics, config, max(1, int(config.adaptive_min_oos_events)))
        and bool(gate_history and gate_history[-1])
    )
    if validated:
        oos_events = max(1.0, _finite(oos_metrics.get("events"), 0.0))
        lift = _finite(oos_metrics.get("lift"), 1.0)
        forward = _finite(oos_metrics.get("forward_hit_rate"), 50.0)
        median_return = _finite(oos_metrics.get("median_directional_return_pct"), 0.0)
        evidence = min(1.0, oos_events / max(1.0, 2.0 * config.adaptive_min_oos_events))
        lift_skill = min(1.0, max(0.0, (lift - config.adaptive_min_reversal_lift) / 0.50))
        forward_skill = min(1.0, max(0.0, (forward - 50.0) / 20.0))
        return_skill = min(1.0, max(0.0, median_return / 3.0))
        quality = 0.25 + 0.75 * (0.45 * lift_skill + 0.35 * forward_skill + 0.20 * return_skill)
        weight_fraction = max(0.0, min(float(cap_fraction), float(cap_fraction) * evidence * quality))
        state = "WALK_FORWARD_VALIDATED"
    else:
        weight_fraction = 0.0
        state = "SHADOW_INSUFFICIENT_OOS" if int(_finite(oos_metrics.get("events"), 0.0)) < config.adaptive_min_oos_events else "REJECTED_NO_OOS_SKILL"
    return {
        "state": state,
        "validated": validated,
        "total_events": int(_finite(total_metrics.get("events"), 0.0)),
        "oos_events": int(_finite(oos_metrics.get("events"), 0.0)),
        "oos_reversal_hit_rate": _finite(oos_metrics.get("reversal_hit_rate"), np.nan),
        "oos_baseline_rate": _finite(oos_metrics.get("baseline_rate"), np.nan),
        "oos_lift": _finite(oos_metrics.get("lift"), np.nan),
        "oos_forward_hit_rate": _finite(oos_metrics.get("forward_hit_rate"), np.nan),
        "oos_median_directional_return_pct": _finite(oos_metrics.get("median_directional_return_pct"), np.nan),
        "weight_fraction": weight_fraction,
        "weight_pct": 100.0 * weight_fraction,
        "method": "EXPANDING_WINDOW_PRE_EVENT_GATE_OOS",
    }


def _public_prior_multiplier(validation: Mapping[str, Any], config: EOFFConfig) -> float:
    """Return a bounded evidence multiplier while preserving the public prior.

    Insufficient evidence leaves the prior unchanged.  Verified OOS skill can
    modestly raise it, while demonstrated no-skill reduces—but never deletes—
    the factor.  This is a clean-room governance choice, not a proprietary
    Astronacci formula.
    """
    low = max(0.0, float(config.public_prior_min_multiplier))
    high = max(low, float(config.public_prior_max_multiplier))
    state = str(validation.get("state") or "SHADOW_INSUFFICIENT_OOS").upper()
    if bool(validation.get("validated")):
        lift = _finite(validation.get("oos_lift"), 1.0)
        forward = _finite(validation.get("oos_forward_hit_rate"), 50.0)
        median_return = _finite(validation.get("oos_median_directional_return_pct"), 0.0)
        lift_skill = min(1.0, max(0.0, (lift - 1.0) / 0.75))
        forward_skill = min(1.0, max(0.0, (forward - 50.0) / 20.0))
        return_skill = min(1.0, max(0.0, median_return / 3.0))
        quality = 0.45 * lift_skill + 0.35 * forward_skill + 0.20 * return_skill
        return min(high, max(low, 1.0 + (high - 1.0) * quality))
    if state == "REJECTED_NO_OOS_SKILL":
        return low
    return 1.0


def _normalized_public_astro_weights(
    adaptive_validation: Mapping[str, Mapping[str, Any]], config: EOFFConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    multipliers = {"MOON_PHASE": 1.0, "PLANETARY_ASPECT": 1.0}
    for family in ADAPTIVE_ASTRO_FAMILIES:
        multipliers[family] = _public_prior_multiplier(adaptive_validation.get(family, {}), config)
    raw = {
        family: PUBLIC_ASTRO_PRIOR_WEIGHTS[family] * multipliers.get(family, 1.0)
        for family in PUBLIC_ASTRO_PRIOR_WEIGHTS
    }
    total = sum(raw.values()) or 1.0
    return ({family: value / total for family, value in raw.items()}, multipliers)


def _adaptive_astro_walk_forward(
    df: pd.DataFrame, fib_event_positions: Sequence[int], config: EOFFConfig,
) -> dict[str, dict[str, Any]]:
    unavailable = {
        family: {
            "state": "DISABLED" if not config.adaptive_astro_enabled else "SHADOW_INSUFFICIENT_OOS",
            "validated": False, "total_events": 0, "oos_events": 0,
            "oos_reversal_hit_rate": np.nan, "oos_baseline_rate": np.nan,
            "oos_lift": np.nan, "oos_forward_hit_rate": np.nan,
            "oos_median_directional_return_pct": np.nan, "weight_fraction": 0.0,
            "weight_pct": 0.0, "method": "EXPANDING_WINDOW_PRE_EVENT_GATE_OOS",
        }
        for family in ADAPTIVE_ASTRO_FAMILIES
    }
    if not config.adaptive_astro_enabled or not config.ephemeris_enabled or ephem is None or not fib_event_positions:
        return unavailable
    bullish_reversals, bearish_reversals = _public_five_candle_reversals(df)
    window = max(1, int(config.historical_window_bars))
    horizon = max(1, int(config.directional_forward_bars))
    evaluated_end = len(df) - horizon
    if evaluated_end <= 60:
        return unavailable
    bullish_covered = {bar for pivot in bullish_reversals for bar in range(max(60, pivot-window), min(evaluated_end, pivot+window+1))}
    bearish_covered = {bar for pivot in bearish_reversals for bar in range(max(60, pivot-window), min(evaluated_end, pivot+window+1))}
    evaluated_bars = max(1, evaluated_end - 60)
    baselines = {
        "BULLISH": 100.0 * len(bullish_covered) / evaluated_bars,
        "BEARISH": 100.0 * len(bearish_covered) / evaluated_bars,
    }
    records: dict[str, list[dict[str, Any]]] = {family: [] for family in ADAPTIVE_ASTRO_FAMILIES}
    for current in list(fib_event_positions)[-180:]:
        current = int(current)
        if current < 60 or current >= evaluated_end:
            continue
        direction = _causal_public_direction(df, current)
        if direction not in {"BULLISH", "BEARISH"}:
            continue
        snapshot = _ephemeris_snapshot(df.index[current], config)
        matching = bullish_reversals if direction == "BULLISH" else bearish_reversals
        reversal_hit = float(any(abs(position-current) <= window for position in matching))
        start_close = _finite(df["Close"].iloc[current], np.nan)
        end_close = _finite(df["Close"].iloc[current+horizon], np.nan)
        if not np.isfinite(start_close) or not np.isfinite(end_close) or start_close <= 0:
            continue
        raw_return = end_close / start_close - 1.0
        directional_return = raw_return if direction == "BULLISH" else -raw_return
        label = {
            "position": current,
            "reversal_hit": reversal_hit,
            "baseline_rate": baselines[direction],
            "forward_hit": float(directional_return >= max(0.0, config.directional_min_return_pct)),
            "directional_return_pct": 100.0 * directional_return,
        }
        for family in ADAPTIVE_ASTRO_FAMILIES:
            signal = _adaptive_family_signal(snapshot, family, direction)
            if signal["active"]:
                records[family].append({**label, "factor_score": signal["score"]})
    return {
        family: _walk_forward_factor_metrics(records[family], config, ADAPTIVE_FAMILY_CAPS[family])
        for family in ADAPTIVE_ASTRO_FAMILIES
    }


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


def _public_five_candle_reversals(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Implement the five-candle reversal definition in the public paper.

    Bullish: two successively lower lows into the centre candle followed by
    two successively higher lows.  Bearish uses the mirror image on highs.
    """
    lows = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float)
    bullish: list[int] = []
    bearish: list[int] = []
    for index in range(2, len(df) - 2):
        if np.isfinite(lows[index - 2:index + 3]).all() and (
            lows[index - 2] > lows[index - 1] > lows[index]
            and lows[index] < lows[index + 1] < lows[index + 2]
        ):
            bullish.append(index)
        if np.isfinite(highs[index - 2:index + 3]).all() and (
            highs[index - 2] < highs[index - 1] < highs[index]
            and highs[index] > highs[index + 1] > highs[index + 2]
        ):
            bearish.append(index)
    return bullish, bearish


def _causal_public_direction(df: pd.DataFrame, current: int) -> str:
    """Infer direction using data available at ``current`` only."""
    history = df.iloc[:current + 1]
    if len(history) < 55:
        return "NEUTRAL"
    close = _finite(history["Close"].iloc[-1], np.nan)
    rolling_low = _finite(history["Low"].tail(20).min(), np.nan)
    rolling_high = _finite(history["High"].tail(20).max(), np.nan)
    span = max(rolling_high - rolling_low, 1e-9)
    location = _clip(100.0 * (close - rolling_low) / span)
    rsi = _finite(_rsi(history["Close"], 14).iloc[-1], 50.0)
    ema20 = history["Close"].ewm(span=20, adjust=False).mean()
    ema50 = history["Close"].ewm(span=50, adjust=False).mean()
    slope = _finite(ema20.iloc[-1] / max(ema20.iloc[-4], 1e-9) - 1.0, 0.0)
    bullish = _clip(
        0.45 * (100.0 - location)
        + 0.30 * _clip((48.0 - rsi) * 3.0 + 50.0)
        + 0.15 * (65.0 if close <= ema20.iloc[-1] else 35.0)
        + 0.10 * _clip(50.0 + 2_500.0 * slope)
    )
    bearish = _clip(
        0.45 * location
        + 0.30 * _clip((rsi - 52.0) * 3.0 + 50.0)
        + 0.15 * (65.0 if close >= ema20.iloc[-1] else 35.0)
        + 0.10 * _clip(50.0 - 2_500.0 * slope)
    )
    # A directional margin is mandatory; astronomy supplies timing, not side.
    if bullish >= bearish + 8.0 and close <= _finite(ema50.iloc[-1], close) * 1.08:
        return "BULLISH"
    if bearish >= bullish + 8.0 and close >= _finite(ema50.iloc[-1], close) * 0.92:
        return "BEARISH"
    return "NEUTRAL"


def _public_directional_validation(
    df: pd.DataFrame,
    fib_event_positions: Sequence[int],
    config: EOFFConfig,
) -> dict[str, Any]:
    """Chronologically validate direction, reversal structure, and return.

    Features at an event use only information known at that bar.  Future bars
    are used solely as labels.  This is a chronological forward test, not a
    claim of independent out-of-sample proof.
    """
    unavailable = {
        "events": 0,
        "reversal_hit_rate": np.nan,
        "baseline_rate": np.nan,
        "lift": np.nan,
        "forward_hit_rate": np.nan,
        "median_directional_return_pct": np.nan,
        "bullish_events": 0,
        "bearish_events": 0,
        "validation_state": "INSUFFICIENT_EVENTS",
        "method": "PUBLIC_FIXED_FAMILY_CHRONOLOGICAL_FORWARD_TEST",
    }
    if not fib_event_positions:
        return unavailable
    bullish_reversals, bearish_reversals = _public_five_candle_reversals(df)
    window = max(1, int(config.historical_window_bars))
    horizon = max(1, int(config.directional_forward_bars))
    evaluated_end = len(df) - horizon
    if evaluated_end <= 60:
        return unavailable
    bullish_covered = {
        bar for pivot in bullish_reversals
        for bar in range(max(60, pivot - window), min(evaluated_end, pivot + window + 1))
    }
    bearish_covered = {
        bar for pivot in bearish_reversals
        for bar in range(max(60, pivot - window), min(evaluated_end, pivot + window + 1))
    }
    evaluated_bars = max(1, evaluated_end - 60)
    bullish_baseline = 100.0 * len(bullish_covered) / evaluated_bars
    bearish_baseline = 100.0 * len(bearish_covered) / evaluated_bars
    reversal_hits: list[float] = []
    forward_hits: list[float] = []
    directional_returns: list[float] = []
    directions: list[str] = []
    for current in list(fib_event_positions)[-120:]:
        current = int(current)
        if current < 60 or current >= evaluated_end:
            continue
        if config.require_astro_fib_confluence:
            snapshot = _ephemeris_snapshot(df.index[current], config)
            if not _public_supported_astro_event(snapshot):
                continue
        direction = _causal_public_direction(df, current)
        if direction == "NEUTRAL":
            continue
        matching = bullish_reversals if direction == "BULLISH" else bearish_reversals
        reversal_hits.append(float(any(abs(position - current) <= window for position in matching)))
        start_close = _finite(df["Close"].iloc[current], np.nan)
        end_close = _finite(df["Close"].iloc[current + horizon], np.nan)
        if not np.isfinite(start_close) or not np.isfinite(end_close) or start_close <= 0:
            continue
        raw_return = end_close / start_close - 1.0
        directional_return = raw_return if direction == "BULLISH" else -raw_return
        directional_returns.append(100.0 * directional_return)
        forward_hits.append(float(directional_return >= max(0.0, config.directional_min_return_pct)))
        directions.append(direction)
    if not reversal_hits or not forward_hits:
        return unavailable
    bullish_events = directions.count("BULLISH")
    bearish_events = directions.count("BEARISH")
    total = bullish_events + bearish_events
    baseline = (
        bullish_events * bullish_baseline + bearish_events * bearish_baseline
    ) / max(1, total)
    hit_rate = 100.0 * float(np.mean(reversal_hits))
    lift = hit_rate / baseline if baseline > 0 else np.nan
    events = min(len(reversal_hits), len(forward_hits))
    return {
        "events": events,
        "reversal_hit_rate": hit_rate,
        "baseline_rate": baseline,
        "lift": lift,
        "forward_hit_rate": 100.0 * float(np.mean(forward_hits)),
        "median_directional_return_pct": float(np.median(directional_returns)),
        "bullish_events": bullish_events,
        "bearish_events": bearish_events,
        "validation_state": "CHRONOLOGICAL_FORWARD_TEST",
        "method": "PUBLIC_FIXED_FAMILY_CHRONOLOGICAL_FORWARD_TEST",
    }


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
        "eoff_astro_diagnostic_score": 0.0,
        "eoff_core_astro_score": 0.0,
        "eoff_adaptive_astro_score": 50.0,
        "eoff_adaptive_total_weight_pct": 50.0,
        "eoff_secondary_prior_share_pct": 50.0,
        "eoff_adaptive_active_factors": "",
        "eoff_adaptive_validation_state": "PUBLIC_PRIOR_NO_OOS_MODULATION",
        "eoff_astro_weight_policy": "PUBLIC_RECONSTRUCTION_PRIOR_WITH_OOS_MODULATION",
        "eoff_astro_prior_weights_json": json.dumps(PUBLIC_ASTRO_PRIOR_WEIGHTS, sort_keys=True),
        "eoff_phase_base_weight_pct": 25.0,
        "eoff_aspect_base_weight_pct": 25.0,
        "eoff_declination_base_weight_pct": 15.0,
        "eoff_ingress_base_weight_pct": 10.0,
        "eoff_retrograde_base_weight_pct": 10.0,
        "eoff_sun_base_weight_pct": 15.0,
        "eoff_phase_weight_pct": 25.0,
        "eoff_aspect_weight_pct": 25.0,
        "eoff_validation_path": "NONE",
        "eoff_retrograde_transition_events": "",
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
        "eoff_public_validation_state": "INSUFFICIENT_EVENTS",
        "eoff_public_validation_method": "PUBLIC_FIXED_FAMILY_CHRONOLOGICAL_FORWARD_TEST",
        "eoff_public_directional_events": 0,
        "eoff_public_reversal_hit_rate": np.nan,
        "eoff_public_baseline_rate": np.nan,
        "eoff_public_lift": np.nan,
        "eoff_public_forward_hit_rate": np.nan,
        "eoff_public_median_directional_return_pct": np.nan,
        "eoff_declination_validation_state": "SHADOW_INSUFFICIENT_OOS",
        "eoff_declination_oos_events": 0,
        "eoff_declination_oos_lift": np.nan,
        "eoff_declination_oos_forward_hit_rate": np.nan,
        "eoff_declination_oos_median_return_pct": np.nan,
        "eoff_declination_weight_pct": 15.0,
        "eoff_declination_current_active": False,
        "eoff_declination_current_score": 50.0,
        "eoff_ingress_validation_state": "SHADOW_INSUFFICIENT_OOS",
        "eoff_ingress_oos_events": 0,
        "eoff_ingress_oos_lift": np.nan,
        "eoff_ingress_oos_forward_hit_rate": np.nan,
        "eoff_ingress_oos_median_return_pct": np.nan,
        "eoff_ingress_weight_pct": 10.0,
        "eoff_ingress_current_active": False,
        "eoff_ingress_current_score": 50.0,
        "eoff_retrograde_validation_state": "SHADOW_INSUFFICIENT_OOS",
        "eoff_retrograde_oos_events": 0,
        "eoff_retrograde_oos_lift": np.nan,
        "eoff_retrograde_oos_forward_hit_rate": np.nan,
        "eoff_retrograde_oos_median_return_pct": np.nan,
        "eoff_retrograde_weight_pct": 10.0,
        "eoff_retrograde_current_active": False,
        "eoff_retrograde_current_score": 50.0,
        "eoff_sun_validation_state": "SHADOW_INSUFFICIENT_OOS",
        "eoff_sun_oos_events": 0,
        "eoff_sun_oos_lift": np.nan,
        "eoff_sun_oos_forward_hit_rate": np.nan,
        "eoff_sun_oos_median_return_pct": np.nan,
        "eoff_sun_weight_pct": 15.0,
        "eoff_sun_current_active": False,
        "eoff_sun_current_score": 50.0,
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
    public_validation = _public_directional_validation(df, fib_event_positions, cfg)
    adaptive_validation = _adaptive_astro_walk_forward(df, fib_event_positions, cfg)
    projected_bar = int(_finite(cluster.get("fib_cluster_target_bar"), current))
    projected_timestamp = _timestamp_from_bar(df.index, projected_bar)
    ephemeris = _ephemeris_snapshot(projected_timestamp, cfg)
    context = _price_pattern_momentum(df, high_pivots, low_pivots)

    fib_score = _clip(cluster.get("fib_cluster_score"))
    public_hit = _finite(public_validation.get("reversal_hit_rate"), np.nan)
    cycle_evidence = 50.0 if not np.isfinite(public_hit) else public_hit
    astro_diagnostic_score = _clip(ephemeris.get("astro_cluster_score"))
    direction = str(context.get("eoff_direction_bias") or "NEUTRAL")
    core_astro_active = _public_supported_astro_event(ephemeris)
    effective_weights, prior_multipliers = _normalized_public_astro_weights(adaptive_validation, cfg)
    phase_raw = _clip(ephemeris.get("moon_phase_score"))
    aspect_raw = _clip(ephemeris.get("aspect_cluster_score"))
    factor_scores: dict[str, float] = {
        "MOON_PHASE": phase_raw if phase_raw >= 55.0 else 50.0,
        "PLANETARY_ASPECT": max(50.0, aspect_raw) if aspect_raw >= 40.0 else 50.0,
    }
    current_adaptive: dict[str, dict[str, Any]] = {}
    active_factor_names: list[str] = []
    validated_active_factors: list[str] = []
    for family in ADAPTIVE_ASTRO_FAMILIES:
        current_signal = _adaptive_family_signal(ephemeris, family, direction)
        validation = dict(adaptive_validation.get(family, {}))
        fixed_weight = effective_weights.get(family, PUBLIC_ASTRO_PRIOR_WEIGHTS[family])
        current_adaptive[family] = {
            **validation,
            **{f"current_{key}": value for key, value in current_signal.items()},
            "base_weight": PUBLIC_ASTRO_PRIOR_WEIGHTS[family],
            "weight_multiplier": prior_multipliers.get(family, 1.0),
            "applied_weight": fixed_weight,
        }
        factor_scores[family] = _clip(current_signal.get("score"), 50.0) if bool(current_signal.get("active")) else 50.0
        if bool(current_signal.get("active")):
            active_factor_names.append(family)
            if bool(validation.get("validated")):
                validated_active_factors.append(family)
    core_weight = effective_weights["MOON_PHASE"] + effective_weights["PLANETARY_ASPECT"]
    core_astro_score = (
        effective_weights["MOON_PHASE"] * factor_scores["MOON_PHASE"]
        + effective_weights["PLANETARY_ASPECT"] * factor_scores["PLANETARY_ASPECT"]
    ) / max(core_weight, 1e-9)
    secondary_families = tuple(ADAPTIVE_ASTRO_FAMILIES)
    applied_weight_total = sum(effective_weights[family] for family in secondary_families)
    adaptive_astro_score = sum(effective_weights[family] * factor_scores[family] for family in secondary_families) / max(applied_weight_total, 1e-9)
    astro_score = _clip(sum(effective_weights[family] * factor_scores[family] for family in PUBLIC_ASTRO_PRIOR_WEIGHTS))
    best_adaptive_hit = max(
        [_finite(value.get("oos_reversal_hit_rate"), np.nan) for value in current_adaptive.values() if value.get("validated")]
        or [np.nan]
    )
    if not np.isfinite(public_hit) and np.isfinite(best_adaptive_hit):
        cycle_evidence = best_adaptive_hit
    time_power = _clip(
        0.50 * fib_score + 0.25 * cycle_evidence + 0.25 * astro_score
    )
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
    fib_active = int(cluster.get("fib_cluster_count", 0)) >= max(1, cfg.min_fib_cluster)
    public_events = int(_finite(public_validation.get("events"), 0.0))
    public_baseline = _finite(public_validation.get("baseline_rate"), np.nan)
    public_lift = _finite(public_validation.get("lift"), np.nan)
    public_forward_hit = _finite(public_validation.get("forward_hit_rate"), np.nan)
    public_median_return = _finite(public_validation.get("median_directional_return_pct"), np.nan)
    core_historical_ok = bool(
        public_events >= cfg.historical_min_events
        and np.isfinite(public_hit)
        and np.isfinite(public_baseline)
        and np.isfinite(public_lift)
        and public_lift >= cfg.directional_min_lift
        and public_hit >= max(35.0, public_baseline + 5.0)
        and np.isfinite(public_forward_hit)
        and public_forward_hit >= 50.0
        and np.isfinite(public_median_return)
        and public_median_return > 0.0
    )
    adaptive_event_active = bool(active_factor_names)
    core_path = bool(core_astro_active and core_historical_ok)
    adaptive_path = bool(validated_active_factors)
    secondary_prior_path = bool(
        len(active_factor_names) >= max(1, int(cfg.secondary_prior_min_active_factors))
        and astro_score >= cfg.secondary_prior_min_astro_score
        and price_score >= cfg.secondary_prior_min_price_score
        and pattern_score >= cfg.secondary_prior_min_pattern_score
        and momentum_score >= cfg.secondary_prior_min_momentum_score
    )
    if core_path and adaptive_path:
        validation_path = "CORE_PUBLIC_PLUS_OOS_MODULATED_PRIOR"
    elif core_path:
        validation_path = "CORE_PUBLIC_PHASE_ASPECT_WITH_PUBLIC_PRIOR"
    elif adaptive_path:
        validation_path = "OOS_VALIDATED_SECONDARY_WITH_PUBLIC_PRIOR"
    elif secondary_prior_path:
        validation_path = "PUBLIC_PRIOR_SECONDARY_CLUSTER_GUARDED"
    else:
        validation_path = "NONE"
    astro_event_active = bool(core_astro_active or adaptive_event_active)
    historical_ok = bool(core_path or adaptive_path or secondary_prior_path or not cfg.require_astro_fib_confluence)
    signal_active = bool(
        fib_active
        and direction in {"BULLISH", "BEARISH"}
        and historical_ok
        and (astro_event_active or not cfg.require_astro_fib_confluence)
    )
    # Classification describes confluence strength, not a guaranteed probability.
    if signal_active and reconstruction >= 82.0 and int(cluster.get("fib_cluster_count", 0)) >= cfg.min_fib_cluster + 3:
        strength = "VERY_STRONG"
    elif signal_active and reconstruction >= 72.0:
        strength = "STRONG"
    elif fib_active and reconstruction >= 60.0:
        strength = "MEDIUM"
    else:
        strength = "LOW"
    state = "ACTIVE_PUBLIC_PRIOR_OOS_MODULATED" if signal_active and (adaptive_path or secondary_prior_path) else "ACTIVE_PUBLIC_RESEARCH_GUARDED" if signal_active else "SHADOW_NO_DIRECTIONAL_VALIDATION" if ephemeris.get("ephemeris_available") else "EPHEMERIS_UNAVAILABLE"
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
        f"astro {astro_score:.1f} (core {core_astro_score:.1f}, adaptive {adaptive_astro_score:.1f}, "
        f"secondary public-prior share {100.0*applied_weight_total:.1f}%); price {price_score:.1f}; pattern {pattern_score:.1f}; "
        f"momentum {momentum_score:.1f}; direction {direction}; target window {reversal_date}; "
        f"public directional validation {public_hit:.1f}%/{public_events} event, "
        f"forward hit {public_forward_hit:.1f}%; validation path {validation_path}; "
        f"adaptive active {','.join(active_factor_names) or 'NONE'}; ephemeris evaluated on {projected_timestamp.date().isoformat()}."
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
        "eoff_astro_diagnostic_score": round(astro_diagnostic_score, 1),
        "eoff_core_astro_score": round(core_astro_score, 1),
        "eoff_adaptive_astro_score": round(adaptive_astro_score, 1),
        "eoff_adaptive_total_weight_pct": round(100.0 * applied_weight_total, 2),
        "eoff_secondary_prior_share_pct": round(100.0 * applied_weight_total, 2),
        "eoff_adaptive_active_factors": "; ".join(active_factor_names),
        "eoff_adaptive_validation_state": "OOS_MODULATED" if validated_active_factors else "PUBLIC_PRIOR_NO_OOS_MODULATION",
        "eoff_astro_weight_policy": "PUBLIC_RECONSTRUCTION_PRIOR_WITH_OOS_MODULATION",
        "eoff_astro_prior_weights_json": json.dumps(PUBLIC_ASTRO_PRIOR_WEIGHTS, sort_keys=True),
        "eoff_phase_base_weight_pct": round(100.0 * PUBLIC_ASTRO_PRIOR_WEIGHTS["MOON_PHASE"], 2),
        "eoff_aspect_base_weight_pct": round(100.0 * PUBLIC_ASTRO_PRIOR_WEIGHTS["PLANETARY_ASPECT"], 2),
        "eoff_declination_base_weight_pct": round(100.0 * PUBLIC_ASTRO_PRIOR_WEIGHTS["MOON_DECLINATION"], 2),
        "eoff_ingress_base_weight_pct": round(100.0 * PUBLIC_ASTRO_PRIOR_WEIGHTS["INGRESS"], 2),
        "eoff_retrograde_base_weight_pct": round(100.0 * PUBLIC_ASTRO_PRIOR_WEIGHTS["RETROGRADE"], 2),
        "eoff_sun_base_weight_pct": round(100.0 * PUBLIC_ASTRO_PRIOR_WEIGHTS["SUN_ANNUAL"], 2),
        "eoff_phase_weight_pct": round(100.0 * effective_weights["MOON_PHASE"], 2),
        "eoff_aspect_weight_pct": round(100.0 * effective_weights["PLANETARY_ASPECT"], 2),
        "eoff_validation_path": validation_path,
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
        "eoff_public_validation_state": str(public_validation.get("validation_state") or "INSUFFICIENT_EVENTS"),
        "eoff_public_validation_method": str(public_validation.get("method") or "PUBLIC_FIXED_FAMILY_CHRONOLOGICAL_FORWARD_TEST"),
        "eoff_public_directional_events": public_events,
        "eoff_public_reversal_hit_rate": round(public_hit, 1) if np.isfinite(public_hit) else np.nan,
        "eoff_public_baseline_rate": round(public_baseline, 1) if np.isfinite(public_baseline) else np.nan,
        "eoff_public_lift": round(public_lift, 2) if np.isfinite(public_lift) else np.nan,
        "eoff_public_forward_hit_rate": round(public_forward_hit, 1) if np.isfinite(public_forward_hit) else np.nan,
        "eoff_public_median_directional_return_pct": round(public_median_return, 2) if np.isfinite(public_median_return) else np.nan,
        "eoff_declination_validation_state": str(current_adaptive["MOON_DECLINATION"].get("state") or "SHADOW_INSUFFICIENT_OOS"),
        "eoff_declination_oos_events": int(_finite(current_adaptive["MOON_DECLINATION"].get("oos_events"), 0.0)),
        "eoff_declination_oos_lift": round(_finite(current_adaptive["MOON_DECLINATION"].get("oos_lift"), np.nan), 2),
        "eoff_declination_oos_forward_hit_rate": round(_finite(current_adaptive["MOON_DECLINATION"].get("oos_forward_hit_rate"), np.nan), 1),
        "eoff_declination_oos_median_return_pct": round(_finite(current_adaptive["MOON_DECLINATION"].get("oos_median_directional_return_pct"), np.nan), 2),
        "eoff_declination_weight_pct": round(100.0 * effective_weights["MOON_DECLINATION"], 2),
        "eoff_declination_current_active": bool(current_adaptive["MOON_DECLINATION"].get("current_active")),
        "eoff_declination_current_score": round(_finite(current_adaptive["MOON_DECLINATION"].get("current_score"), 50.0), 1),
        "eoff_ingress_validation_state": str(current_adaptive["INGRESS"].get("state") or "SHADOW_INSUFFICIENT_OOS"),
        "eoff_ingress_oos_events": int(_finite(current_adaptive["INGRESS"].get("oos_events"), 0.0)),
        "eoff_ingress_oos_lift": round(_finite(current_adaptive["INGRESS"].get("oos_lift"), np.nan), 2),
        "eoff_ingress_oos_forward_hit_rate": round(_finite(current_adaptive["INGRESS"].get("oos_forward_hit_rate"), np.nan), 1),
        "eoff_ingress_oos_median_return_pct": round(_finite(current_adaptive["INGRESS"].get("oos_median_directional_return_pct"), np.nan), 2),
        "eoff_ingress_weight_pct": round(100.0 * effective_weights["INGRESS"], 2),
        "eoff_ingress_current_active": bool(current_adaptive["INGRESS"].get("current_active")),
        "eoff_ingress_current_score": round(_finite(current_adaptive["INGRESS"].get("current_score"), 50.0), 1),
        "eoff_retrograde_validation_state": str(current_adaptive["RETROGRADE"].get("state") or "SHADOW_INSUFFICIENT_OOS"),
        "eoff_retrograde_oos_events": int(_finite(current_adaptive["RETROGRADE"].get("oos_events"), 0.0)),
        "eoff_retrograde_oos_lift": round(_finite(current_adaptive["RETROGRADE"].get("oos_lift"), np.nan), 2),
        "eoff_retrograde_oos_forward_hit_rate": round(_finite(current_adaptive["RETROGRADE"].get("oos_forward_hit_rate"), np.nan), 1),
        "eoff_retrograde_oos_median_return_pct": round(_finite(current_adaptive["RETROGRADE"].get("oos_median_directional_return_pct"), np.nan), 2),
        "eoff_retrograde_weight_pct": round(100.0 * effective_weights["RETROGRADE"], 2),
        "eoff_retrograde_current_active": bool(current_adaptive["RETROGRADE"].get("current_active")),
        "eoff_retrograde_current_score": round(_finite(current_adaptive["RETROGRADE"].get("current_score"), 50.0), 1),
        "eoff_sun_validation_state": str(current_adaptive["SUN_ANNUAL"].get("state") or "SHADOW_INSUFFICIENT_OOS"),
        "eoff_sun_oos_events": int(_finite(current_adaptive["SUN_ANNUAL"].get("oos_events"), 0.0)),
        "eoff_sun_oos_lift": round(_finite(current_adaptive["SUN_ANNUAL"].get("oos_lift"), np.nan), 2),
        "eoff_sun_oos_forward_hit_rate": round(_finite(current_adaptive["SUN_ANNUAL"].get("oos_forward_hit_rate"), np.nan), 1),
        "eoff_sun_oos_median_return_pct": round(_finite(current_adaptive["SUN_ANNUAL"].get("oos_median_directional_return_pct"), np.nan), 2),
        "eoff_sun_weight_pct": round(100.0 * effective_weights["SUN_ANNUAL"], 2),
        "eoff_sun_current_active": bool(current_adaptive["SUN_ANNUAL"].get("current_active")),
        "eoff_sun_current_score": round(_finite(current_adaptive["SUN_ANNUAL"].get("current_score"), 50.0), 1),
        "eoff_bars_to_cluster": int(_finite(cluster.get("fib_cluster_bars_ahead"), 0.0)),
        "eoff_reversal_date": reversal_date,
        "eoff_ephemeris_state": str(ephemeris.get("ephemeris_state") or "UNAVAILABLE"),
        "eoff_ephemeris_date": projected_timestamp.date().isoformat(),
        "eoff_astro_events": "; ".join(str(value) for value in ephemeris.get("astro_events", [])),
        "eoff_active_aspects": aspects_text,
        "eoff_retrograde_planets": "; ".join(str(value) for value in ephemeris.get("retrograde_planets", [])),
        "eoff_stationary_planets": "; ".join(str(value) for value in ephemeris.get("stationary_planets", [])),
        "eoff_retrograde_transition_events": "; ".join(str(value) for value in ephemeris.get("retrograde_transition_events", [])),
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
    "EOFF_VERSION", "EOFFConfig", "ADAPTIVE_ASTRO_FAMILIES", "PUBLIC_ASTRO_PRIOR_WEIGHTS",
    "analyze_eoff_reconstruction", "setup_eoff_alignment", "ephem",
]

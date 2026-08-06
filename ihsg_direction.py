from __future__ import annotations

"""Point-in-time IHSG direction research engine.

The engine is intentionally dependency-light and deterministic:

* it builds causal trend, momentum, volatility, drawdown, and universe-breadth
  features;
* estimates 1/5/20-session direction from spaced historical analogues;
* validates each horizon with chronological walk-forward predictions;
* abstains when data, probability separation, or out-of-sample evidence is
  insufficient;
* emits a risk-budget cap, never a score bonus or an automatic order.

The probabilities are empirical analogue frequencies.  They are labelled as
validated only when the chronological test beats its own historical-climatology
baseline.  This module does not claim certainty or guaranteed returns.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
import json
import math

import numpy as np
import pandas as pd


IHSG_DIRECTION_VERSION = "1.0.0-chronological-analogue"
IHSG_DIRECTION_SCHEMA_VERSION = "ihsg_direction_v1"
IHSG_TICKER = "^JKSE"

FEATURE_COLUMNS: tuple[str, ...] = (
    "close_vs_ema20",
    "close_vs_ema50",
    "close_vs_ema200",
    "ema20_vs_ema50",
    "ema50_vs_ema200",
    "ema50_slope20",
    "roc5",
    "roc20",
    "roc60",
    "rsi14_scaled",
    "macd_hist_pct",
    "realized_vol20",
    "atr_pct",
    "drawdown63",
    "drawdown252",
    "breadth_ema20",
    "breadth_ema50",
    "breadth_ema200",
    "breadth_positive_20",
    "breadth_advancers",
    "breadth_mean_roc20",
)

HORIZON_LABELS: dict[int, str] = {1: "1D", 5: "5D", 20: "20D"}
HORIZON_CONSENSUS_WEIGHTS: dict[int, float] = {1: 0.20, 5: 0.35, 20: 0.45}
NEUTRAL_RETURN_FLOORS: dict[int, float] = {1: 0.0025, 5: 0.0080, 20: 0.0200}


@dataclass(frozen=True)
class IHSGDirectionConfig:
    horizons: tuple[int, ...] = (1, 5, 20)
    min_history_bars: int = 260
    min_train_bars: int = 220
    min_analogues: int = 30
    max_analogues: int = 96
    analogue_spacing_bars: int = 5
    min_feature_training_coverage: float = 0.72
    min_features: int = 8
    validation_points: int = 64
    min_validation_predictions: int = 32
    min_directional_validation_predictions: int = 12
    min_brier_skill: float = 0.02
    min_directional_probability: float = 0.44
    min_probability_edge: float = 0.075
    min_production_confidence: float = 58.0
    min_breadth_members: int = 8
    breadth_forward_fill_limit: int = 3
    max_data_age_days: int = 5
    neutral_volatility_fraction: float = 0.18
    structural_prior_strength: float = 5.0


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value).strip()


def _clip(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    number = _finite(value, lower)
    return float(np.clip(number, lower, upper))


def _normalise_ohlcv(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty or "Close" not in frame:
        return pd.DataFrame()
    out = frame.copy()
    index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~index.isna()].copy()
    index = pd.DatetimeIndex(index[~index.isna()])
    if index.tz is not None:
        index = index.tz_convert("Asia/Jakarta").tz_localize(None)
    out.index = index.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    close = pd.to_numeric(out["Close"], errors="coerce")
    out = out.loc[close.notna() & close.gt(0)].copy()
    if out.empty:
        return out
    for column in ("Open", "High", "Low"):
        if column not in out:
            out[column] = out["Close"]
        else:
            out[column] = out[column].fillna(out["Close"])
    if "Volume" not in out:
        out["Volume"] = np.nan
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").clip(lower=0)
    return out


def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(
        span=span, adjust=False, min_periods=max(3, span // 2)
    ).mean()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = pd.to_numeric(series, errors="coerce").diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    relative = gain / loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + relative)).where(loss.ne(0), 100.0).fillna(50.0)


def _atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    true_range = pd.concat(
        [(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def _breadth_frame(
    prepared: Mapping[str, pd.DataFrame] | None,
    index: pd.DatetimeIndex,
    config: IHSGDirectionConfig,
) -> pd.DataFrame:
    columns = (
        "breadth_ema20",
        "breadth_ema50",
        "breadth_ema200",
        "breadth_positive_20",
        "breadth_advancers",
        "breadth_mean_roc20",
        "breadth_member_count",
    )
    histories = prepared or {}
    if not histories:
        return pd.DataFrame(index=index, columns=columns, dtype=float)

    metric_names = columns[:-1]
    sums = {name: np.zeros(len(index), dtype=float) for name in metric_names}
    counts = {name: np.zeros(len(index), dtype=float) for name in metric_names}
    current_members = np.zeros(len(index), dtype=float)
    limit = max(0, int(config.breadth_forward_fill_limit))

    for _, raw in histories.items():
        local = _normalise_ohlcv(raw)
        if len(local) < 60:
            continue
        close = local["Close"]
        local_metrics = pd.DataFrame(
            {
                "close": close,
                "ema20": _ema(close, 20),
                "ema50": _ema(close, 50),
                "ema200": _ema(close, 200),
                "roc20": close.pct_change(20),
                "daily_return": close.pct_change(),
            },
            index=local.index,
        )
        aligned = local_metrics.reindex(index)
        if limit:
            aligned = aligned.ffill(limit=limit)
        active = aligned["close"].notna()
        current_members += active.to_numpy(dtype=float)
        values: dict[str, pd.Series] = {
            "breadth_ema20": (aligned["close"] > aligned["ema20"]).where(
                active & aligned["ema20"].notna()
            ).astype(float),
            "breadth_ema50": (aligned["close"] > aligned["ema50"]).where(
                active & aligned["ema50"].notna()
            ).astype(float),
            "breadth_ema200": (aligned["close"] > aligned["ema200"]).where(
                active & aligned["ema200"].notna()
            ).astype(float),
            "breadth_positive_20": aligned["roc20"].gt(0).where(
                active & aligned["roc20"].notna()
            ).astype(float),
            "breadth_advancers": aligned["daily_return"].gt(0).where(
                active & aligned["daily_return"].notna()
            ).astype(float),
            "breadth_mean_roc20": aligned["roc20"].clip(-0.35, 0.35).where(active),
        }
        for name, series in values.items():
            array = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(array)
            sums[name] += np.where(valid, array, 0.0)
            counts[name] += valid.astype(float)

    output = pd.DataFrame(index=index)
    minimum = max(1, int(config.min_breadth_members))
    for name in metric_names:
        with np.errstate(divide="ignore", invalid="ignore"):
            value = sums[name] / counts[name]
        value[counts[name] < minimum] = np.nan
        output[name] = value
    output["breadth_member_count"] = current_members
    return output.reindex(columns=columns)


def build_ihsg_feature_frame(
    benchmark: pd.DataFrame | None,
    prepared: Mapping[str, pd.DataFrame] | None = None,
    config: IHSGDirectionConfig | None = None,
) -> pd.DataFrame:
    """Build a causal daily feature frame for IHSG and the scanned universe."""
    cfg = config or IHSGDirectionConfig()
    frame = _normalise_ohlcv(benchmark)
    if frame.empty:
        return pd.DataFrame()
    close = frame["Close"]
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    macd = _ema(close, 12) - _ema(close, 26)
    macd_signal = _ema(macd, 9)
    log_return = np.log(close).diff()
    annualised_volatility = log_return.rolling(20, min_periods=15).std() * np.sqrt(252.0)
    features = pd.DataFrame(
        {
            "Close": close,
            "EMA20": ema20,
            "EMA50": ema50,
            "EMA200": ema200,
            "close_vs_ema20": close / ema20.replace(0, np.nan) - 1.0,
            "close_vs_ema50": close / ema50.replace(0, np.nan) - 1.0,
            "close_vs_ema200": close / ema200.replace(0, np.nan) - 1.0,
            "ema20_vs_ema50": ema20 / ema50.replace(0, np.nan) - 1.0,
            "ema50_vs_ema200": ema50 / ema200.replace(0, np.nan) - 1.0,
            "ema50_slope20": ema50 / ema50.shift(20).replace(0, np.nan) - 1.0,
            "roc5": close.pct_change(5),
            "roc20": close.pct_change(20),
            "roc60": close.pct_change(60),
            "rsi14_scaled": (_rsi(close, 14) - 50.0) / 50.0,
            "macd_hist_pct": (macd - macd_signal) / close.replace(0, np.nan),
            "realized_vol20": annualised_volatility,
            "atr_pct": _atr(frame, 14) / close.replace(0, np.nan),
            "drawdown63": close / close.rolling(63, min_periods=40).max() - 1.0,
            "drawdown252": close / close.rolling(252, min_periods=120).max() - 1.0,
        },
        index=frame.index,
    )
    features = features.join(_breadth_frame(prepared, features.index, cfg), how="left")
    return features.replace([np.inf, -np.inf], np.nan)


def _weighted_component(components: Sequence[tuple[Any, float]]) -> float:
    values: list[float] = []
    weights: list[float] = []
    for value, weight in components:
        number = _finite(value, np.nan)
        if np.isfinite(number) and weight > 0:
            values.append(float(np.clip(number, -1.0, 1.0)))
            weights.append(float(weight))
    if not values:
        return 0.0
    return float(np.average(values, weights=weights))


def _structural_score(row: Mapping[str, Any]) -> float:
    trend = _weighted_component(
        (
            (np.tanh(_finite(row.get("close_vs_ema50"), 0.0) / 0.05), 1.0),
            (np.tanh(_finite(row.get("close_vs_ema200"), 0.0) / 0.10), 0.8),
            (np.tanh(_finite(row.get("ema50_vs_ema200"), 0.0) / 0.07), 1.0),
            (np.tanh(_finite(row.get("ema50_slope20"), 0.0) / 0.035), 0.8),
        )
    )
    momentum = _weighted_component(
        (
            (np.tanh(_finite(row.get("roc5"), 0.0) / 0.035), 0.5),
            (np.tanh(_finite(row.get("roc20"), 0.0) / 0.075), 1.0),
            (np.tanh(_finite(row.get("roc60"), 0.0) / 0.15), 0.8),
            (_finite(row.get("rsi14_scaled"), 0.0), 0.5),
            (np.tanh(_finite(row.get("macd_hist_pct"), 0.0) / 0.008), 0.6),
        )
    )
    breadth = _weighted_component(
        (
            (2.0 * (_finite(row.get("breadth_ema20"), 0.5) - 0.5), 0.5),
            (2.0 * (_finite(row.get("breadth_ema50"), 0.5) - 0.5), 1.0),
            (2.0 * (_finite(row.get("breadth_ema200"), 0.5) - 0.5), 0.7),
            (2.0 * (_finite(row.get("breadth_positive_20"), 0.5) - 0.5), 0.8),
            (np.tanh(_finite(row.get("breadth_mean_roc20"), 0.0) / 0.06), 0.6),
        )
    )
    drawdown = _finite(row.get("drawdown63"), 0.0)
    volatility = _finite(row.get("realized_vol20"), 0.20)
    risk = np.clip((drawdown + 0.04) / 0.12, -1.0, 1.0)
    if volatility > 0.35:
        risk -= min(0.7, (volatility - 0.35) / 0.25)
    return float(np.clip(0.42 * trend + 0.30 * momentum + 0.20 * breadth + 0.08 * risk, -1.0, 1.0))


def _neutral_band(horizon: int, annualised_volatility: Any, config: IHSGDirectionConfig) -> float:
    horizon = max(1, int(horizon))
    floor = NEUTRAL_RETURN_FLOORS.get(horizon, 0.0025 * math.sqrt(horizon))
    annual = max(0.0, _finite(annualised_volatility, 0.20))
    daily = annual / math.sqrt(252.0)
    volatility_band = float(config.neutral_volatility_fraction) * daily * math.sqrt(horizon)
    return float(max(floor, volatility_band))


def _neutral_bands(
    rows: pd.DataFrame,
    horizon: int,
    config: IHSGDirectionConfig,
) -> np.ndarray:
    horizon = max(1, int(horizon))
    floor = NEUTRAL_RETURN_FLOORS.get(horizon, 0.0025 * math.sqrt(horizon))
    annual = pd.to_numeric(
        rows.get("realized_vol20", pd.Series(0.20, index=rows.index)),
        errors="coerce",
    ).fillna(0.20).clip(lower=0.0).to_numpy(dtype=float)
    volatility_band = (
        float(config.neutral_volatility_fraction)
        * annual
        / math.sqrt(252.0)
        * math.sqrt(horizon)
    )
    return np.maximum(float(floor), volatility_band)


def _classify_return(value: Any, band: float) -> str:
    number = _finite(value, np.nan)
    if not np.isfinite(number):
        return "UNKNOWN"
    if number > band:
        return "UP"
    if number < -band:
        return "DOWN"
    return "SIDEWAYS"


def _spaced_positions(
    ordered_positions: Iterable[int],
    maximum: int,
    spacing: int,
) -> list[int]:
    selected: list[int] = []
    minimum_gap = max(1, int(spacing))
    for value in ordered_positions:
        position = int(value)
        if all(abs(position - prior) >= minimum_gap for prior in selected):
            selected.append(position)
            if len(selected) >= maximum:
                break
    return selected


def _analogue_distribution(
    features: pd.DataFrame,
    target_position: int,
    horizon: int,
    config: IHSGDirectionConfig,
    *,
    structural_prior: bool = True,
) -> dict[str, Any]:
    empty = {
        "prob_up": np.nan,
        "prob_sideways": np.nan,
        "prob_down": np.nan,
        "analogue_count": 0,
        "effective_analogue_count": 0.0,
        "median_distance": np.nan,
        "features_used": 0,
        "feature_names": "",
        "expected_return": np.nan,
        "return_p25": np.nan,
        "return_p75": np.nan,
        "base_prob_up": np.nan,
        "base_prob_sideways": np.nan,
        "base_prob_down": np.nan,
        "neutral_band": np.nan,
    }
    if features.empty or target_position <= 0 or "Close" not in features:
        return empty
    horizon = max(1, int(horizon))
    train_end = int(target_position) - horizon
    if train_end < max(60, int(config.min_train_bars)):
        return empty
    target = features.iloc[int(target_position)]
    candidate = features.iloc[: train_end + 1].copy()
    close = pd.to_numeric(features["Close"], errors="coerce")
    candidate_positions = np.arange(len(candidate), dtype=int)
    future_positions = candidate_positions + horizon
    forward_returns = (
        close.iloc[future_positions].to_numpy(dtype=float)
        / close.iloc[candidate_positions].to_numpy(dtype=float)
        - 1.0
    )
    label_valid = np.isfinite(forward_returns)
    if int(label_valid.sum()) < int(config.min_analogues):
        return empty
    candidate = candidate.iloc[label_valid].copy()
    candidate_positions = candidate_positions[label_valid]
    forward_returns = forward_returns[label_valid]

    feature_names: list[str] = []
    for name in FEATURE_COLUMNS:
        if name not in candidate or not np.isfinite(_finite(target.get(name), np.nan)):
            continue
        coverage = float(pd.to_numeric(candidate[name], errors="coerce").notna().mean())
        if coverage >= float(config.min_feature_training_coverage):
            feature_names.append(name)
    if len(feature_names) < int(config.min_features):
        return empty

    matrix = candidate[feature_names].apply(pd.to_numeric, errors="coerce")
    medians = matrix.median(axis=0, skipna=True)
    absolute_deviation = matrix.sub(medians, axis=1).abs().median(axis=0, skipna=True)
    scale = (1.4826 * absolute_deviation).replace(0, np.nan)
    fallback_scale = matrix.std(axis=0, skipna=True).replace(0, np.nan)
    scale = scale.fillna(fallback_scale)
    usable = [
        name for name in feature_names
        if np.isfinite(_finite(scale.get(name), np.nan)) and _finite(scale.get(name), 0.0) > 1e-12
    ]
    if len(usable) < int(config.min_features):
        return empty
    matrix = matrix[usable].fillna(medians[usable])
    target_values = pd.to_numeric(target[usable], errors="coerce")
    normalised = matrix.sub(target_values, axis=1).div(scale[usable], axis=1)
    distance = np.sqrt(np.square(normalised).mean(axis=1)).to_numpy(dtype=float)
    valid_distance = np.isfinite(distance)
    if int(valid_distance.sum()) < int(config.min_analogues):
        return empty
    distance = distance[valid_distance]
    positions = candidate_positions[valid_distance]
    returns = forward_returns[valid_distance]
    training_feature_rows = features.iloc[positions]
    training_bands = _neutral_bands(training_feature_rows, horizon, config)
    training_labels = np.asarray(
        [_classify_return(value, band) for value, band in zip(returns, training_bands)],
        dtype=object,
    )
    ordered = np.argsort(distance, kind="stable")
    maximum = min(int(config.max_analogues), len(ordered))
    selected_positions = _spaced_positions(
        positions[ordered], maximum, int(config.analogue_spacing_bars)
    )
    if len(selected_positions) < int(config.min_analogues):
        return empty
    position_lookup = {int(position): index for index, position in enumerate(positions)}
    selected_indices = np.asarray([position_lookup[position] for position in selected_positions], dtype=int)
    selected_distance = distance[selected_indices]
    selected_returns = returns[selected_indices]
    distance_scale = float(np.nanmedian(selected_distance))
    if not np.isfinite(distance_scale) or distance_scale <= 1e-12:
        distance_scale = 1.0
    weights = np.exp(-0.5 * np.square(selected_distance / distance_scale))
    weights = np.clip(weights, 1e-6, None)

    selected_feature_rows = features.iloc[selected_positions]
    bands = _neutral_bands(selected_feature_rows, horizon, config)
    labels = np.asarray(
        [_classify_return(value, band) for value, band in zip(selected_returns, bands)],
        dtype=object,
    )
    weighted_counts = {
        state: float(weights[labels == state].sum())
        for state in ("UP", "SIDEWAYS", "DOWN")
    }
    base_counts = {
        state: float(np.sum(training_labels == state))
        for state in ("UP", "SIDEWAYS", "DOWN")
    }
    priors = {"UP": 1.5, "SIDEWAYS": 1.5, "DOWN": 1.5}
    if structural_prior:
        score = _structural_score(target)
        strength = max(0.0, float(config.structural_prior_strength))
        priors["UP"] += strength * max(0.0, score)
        priors["DOWN"] += strength * max(0.0, -score)
        priors["SIDEWAYS"] += 0.40 * strength * max(0.0, 1.0 - abs(score))
    posterior = {
        state: weighted_counts[state] + priors[state]
        for state in ("UP", "SIDEWAYS", "DOWN")
    }
    denominator = sum(posterior.values())
    base_denominator = sum(base_counts.values()) + 3.0
    probabilities = {state: posterior[state] / denominator for state in posterior}
    base_probabilities = {
        state: (base_counts[state] + 1.0) / base_denominator
        for state in base_counts
    }
    effective = float(np.square(weights.sum()) / np.square(weights).sum())
    weighted_mean = float(np.average(selected_returns, weights=weights))
    current_band = _neutral_band(horizon, target.get("realized_vol20"), config)
    return {
        "prob_up": probabilities["UP"],
        "prob_sideways": probabilities["SIDEWAYS"],
        "prob_down": probabilities["DOWN"],
        "analogue_count": int(len(selected_positions)),
        "effective_analogue_count": effective,
        "median_distance": float(np.nanmedian(selected_distance)),
        "features_used": int(len(usable)),
        "feature_names": "|".join(usable),
        "expected_return": weighted_mean,
        "return_p25": float(np.nanquantile(selected_returns, 0.25)),
        "return_p75": float(np.nanquantile(selected_returns, 0.75)),
        "base_prob_up": base_probabilities["UP"],
        "base_prob_sideways": base_probabilities["SIDEWAYS"],
        "base_prob_down": base_probabilities["DOWN"],
        "neutral_band": current_band,
    }


def _raw_direction(distribution: Mapping[str, Any], config: IHSGDirectionConfig) -> str:
    probabilities = {
        "UP": _finite(distribution.get("prob_up"), np.nan),
        "SIDEWAYS": _finite(distribution.get("prob_sideways"), np.nan),
        "DOWN": _finite(distribution.get("prob_down"), np.nan),
    }
    if not all(np.isfinite(value) for value in probabilities.values()):
        return "ABSTAIN"
    ordered = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    state, top = ordered[0]
    edge = top - ordered[1][1]
    if edge < float(config.min_probability_edge):
        return "ABSTAIN"
    if state in {"UP", "DOWN"} and top < float(config.min_directional_probability):
        return "ABSTAIN"
    return state


def _multiclass_brier(probabilities: Mapping[str, Any], actual: str) -> float:
    mapping = {
        "UP": _finite(probabilities.get("prob_up"), np.nan),
        "SIDEWAYS": _finite(probabilities.get("prob_sideways"), np.nan),
        "DOWN": _finite(probabilities.get("prob_down"), np.nan),
    }
    if actual not in mapping or not all(np.isfinite(value) for value in mapping.values()):
        return np.nan
    return float(sum((probability - float(state == actual)) ** 2 for state, probability in mapping.items()))


def _wilson_lower_bound(successes: int, total: int, z: float = 1.2815515655446004) -> float:
    if total <= 0:
        return np.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total)
    return float((centre - radius) / denominator)


def _walk_forward_validation(
    features: pd.DataFrame,
    horizon: int,
    config: IHSGDirectionConfig,
) -> dict[str, Any]:
    empty = {
        "validation_state": "LIMITED_EVIDENCE",
        "validation_predictions": 0,
        "directional_predictions": 0,
        "directional_accuracy": np.nan,
        "directional_accuracy_ci_low": np.nan,
        "coverage": 0.0,
        "brier_score": np.nan,
        "baseline_brier_score": np.nan,
        "brier_skill": np.nan,
    }
    horizon = max(1, int(horizon))
    start = max(int(config.min_train_bars) + horizon, 100)
    end = len(features) - horizon - 1
    if end <= start:
        return empty
    positions = np.arange(start, end + 1, dtype=int)
    maximum = max(1, int(config.validation_points))
    if len(positions) > maximum:
        positions = np.unique(np.linspace(start, end, maximum, dtype=int))
    close = pd.to_numeric(features["Close"], errors="coerce")
    records: list[dict[str, Any]] = []
    for position in positions:
        distribution = _analogue_distribution(
            features, int(position), horizon, config, structural_prior=True
        )
        prediction = _raw_direction(distribution, config)
        current_close = _finite(close.iloc[position], np.nan)
        future_close = _finite(close.iloc[position + horizon], np.nan)
        if not np.isfinite(current_close) or not np.isfinite(future_close) or current_close <= 0:
            continue
        realised = future_close / current_close - 1.0
        band = _neutral_band(horizon, features.iloc[position].get("realized_vol20"), config)
        actual = _classify_return(realised, band)
        if not np.isfinite(_finite(distribution.get("prob_up"), np.nan)):
            continue
        model_brier = _multiclass_brier(distribution, actual)
        baseline_brier = _multiclass_brier(
            {
                "prob_up": distribution.get("base_prob_up"),
                "prob_sideways": distribution.get("base_prob_sideways"),
                "prob_down": distribution.get("base_prob_down"),
            },
            actual,
        )
        records.append(
            {
                "prediction": prediction,
                "actual": actual,
                "model_brier": model_brier,
                "baseline_brier": baseline_brier,
            }
        )
    if not records:
        return empty
    frame = pd.DataFrame(records)
    actionable = frame[frame["prediction"].isin(["UP", "DOWN"])]
    correct = int((actionable["prediction"] == actionable["actual"]).sum())
    directional_count = int(len(actionable))
    directional_accuracy = correct / directional_count if directional_count else np.nan
    model_brier = float(pd.to_numeric(frame["model_brier"], errors="coerce").mean())
    baseline_brier = float(pd.to_numeric(frame["baseline_brier"], errors="coerce").mean())
    brier_skill = 1.0 - model_brier / baseline_brier if baseline_brier > 1e-12 else np.nan
    enough = (
        len(frame) >= int(config.min_validation_predictions)
        and directional_count >= int(config.min_directional_validation_predictions)
    )
    ci_low = _wilson_lower_bound(correct, directional_count)
    if not enough:
        state = "LIMITED_EVIDENCE"
    elif (
        np.isfinite(brier_skill)
        and brier_skill >= float(config.min_brier_skill)
        and np.isfinite(directional_accuracy)
        and directional_accuracy >= 0.50
        and np.isfinite(ci_low)
        and ci_low >= 0.40
    ):
        state = "OOS_POSITIVE"
    elif np.isfinite(brier_skill) and brier_skill >= -0.03:
        state = "OOS_MIXED"
    else:
        state = "OOS_NEGATIVE"
    return {
        "validation_state": state,
        "validation_predictions": int(len(frame)),
        "directional_predictions": directional_count,
        "directional_accuracy": directional_accuracy,
        "directional_accuracy_ci_low": ci_low,
        "coverage": directional_count / len(frame) if len(frame) else 0.0,
        "brier_score": model_brier,
        "baseline_brier_score": baseline_brier,
        "brier_skill": brier_skill,
    }


def _regime_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    close_200 = _finite(row.get("close_vs_ema200"), np.nan)
    ema_cross = _finite(row.get("ema50_vs_ema200"), np.nan)
    slope = _finite(row.get("ema50_slope20"), np.nan)
    roc20 = _finite(row.get("roc20"), np.nan)
    breadth50 = _finite(row.get("breadth_ema50"), np.nan)
    drawdown = _finite(row.get("drawdown63"), 0.0)
    volatility = _finite(row.get("realized_vol20"), 0.20)
    score = _structural_score(row)
    if not all(np.isfinite(value) for value in (close_200, ema_cross, slope, roc20)):
        regime = "UNKNOWN"
        reason = "EMA200/momentum history belum lengkap"
    elif close_200 > 0 and ema_cross > 0 and slope > 0:
        if roc20 > 0 and (not np.isfinite(breadth50) or breadth50 >= 0.52):
            regime = "BULL_CONFIRMED"
            reason = "IHSG di atas EMA200, EMA50 menanjak, momentum dan breadth mendukung"
        else:
            regime = "BULL_FRAGILE"
            reason = "Struktur panjang bullish tetapi momentum/breadth belum seragam"
    elif close_200 < 0 and ema_cross < 0 and slope < 0:
        if roc20 < 0 and (not np.isfinite(breadth50) or breadth50 <= 0.45):
            regime = "BEAR_CONFIRMED"
            reason = "IHSG di bawah EMA200, EMA50 turun, momentum dan breadth lemah"
        else:
            regime = "BEAR_RALLY"
            reason = "Struktur panjang bearish dengan pantulan momentum/breadth"
    elif close_200 < 0 and roc20 > 0:
        regime = "BEAR_RALLY"
        reason = "Momentum memantul tetapi IHSG masih di bawah EMA200"
    else:
        regime = "TRANSITION"
        reason = "Trend, momentum, dan breadth belum satu arah"
    crash_risk = bool(drawdown <= -0.10 and volatility >= 0.30)
    if crash_risk:
        reason += "; drawdown dan volatilitas berada pada zona defensif"
    return {
        "regime": regime,
        "regime_score": round(100.0 * score, 2),
        "regime_reason": reason,
        "crash_risk": crash_risk,
    }


def _confidence_score(
    distribution: Mapping[str, Any],
    validation: Mapping[str, Any],
    feature_coverage: float,
    config: IHSGDirectionConfig,
) -> float:
    probabilities = sorted(
        [
            _finite(distribution.get("prob_up"), 0.0),
            _finite(distribution.get("prob_sideways"), 0.0),
            _finite(distribution.get("prob_down"), 0.0),
        ],
        reverse=True,
    )
    margin = max(0.0, probabilities[0] - probabilities[1])
    sample = min(
        1.0,
        _finite(distribution.get("effective_analogue_count"), 0.0)
        / max(1.0, 0.75 * float(config.min_analogues)),
    )
    compactness = 1.0 / (1.0 + max(0.0, _finite(distribution.get("median_distance"), 5.0)))
    validation_state = _text(validation.get("validation_state")).upper()
    validation_component = {
        "OOS_POSITIVE": 1.0,
        "OOS_MIXED": 0.55,
        "LIMITED_EVIDENCE": 0.35,
        "OOS_NEGATIVE": 0.10,
    }.get(validation_state, 0.20)
    score = 100.0 * (
        0.23 * sample
        + 0.22 * min(1.0, margin / 0.22)
        + 0.16 * _clip(feature_coverage)
        + 0.14 * min(1.0, 2.5 * compactness)
        + 0.25 * validation_component
    )
    return float(np.clip(score, 0.0, 100.0))


def _data_state(
    features: pd.DataFrame,
    config: IHSGDirectionConfig,
    now: Any | None,
) -> tuple[str, int]:
    if features.empty:
        return "BENCHMARK_UNAVAILABLE", 999999
    if len(features) < int(config.min_history_bars):
        return "INSUFFICIENT_HISTORY", 999999
    current = pd.to_datetime(now, errors="coerce")
    if pd.isna(current):
        current = pd.Timestamp.now(tz="Asia/Jakarta")
    current = pd.Timestamp(current)
    if current.tzinfo is not None:
        current = current.tz_convert("Asia/Jakarta").tz_localize(None)
    age = max(0, int((current.normalize() - pd.Timestamp(features.index[-1]).normalize()).days))
    if age > int(config.max_data_age_days):
        return "STALE_BENCHMARK", age
    return "READY", age


def _empty_horizon_rows(config: IHSGDirectionConfig, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horizon": HORIZON_LABELS.get(int(horizon), f"{int(horizon)}D"),
                "horizon_bars": int(horizon),
                "raw_direction": "ABSTAIN",
                "prediction_state": "ABSTAIN",
                "prob_up_pct": np.nan,
                "prob_sideways_pct": np.nan,
                "prob_down_pct": np.nan,
                "confidence_pct": 0.0,
                "validation_state": reason,
                "actionable": False,
            }
            for horizon in config.horizons
        ]
    )


def _consensus(horizons: pd.DataFrame) -> dict[str, Any]:
    if horizons is None or horizons.empty:
        return {
            "consensus_direction": "NO_EDGE",
            "consensus_confidence": 0.0,
            "consensus_prob_up": np.nan,
            "consensus_prob_sideways": np.nan,
            "consensus_prob_down": np.nan,
            "actionable_horizons": 0,
        }
    usable = horizons[horizons["prediction_state"].isin(["UP", "SIDEWAYS", "DOWN"])].copy()
    if usable.empty:
        usable = horizons[horizons["raw_direction"].isin(["UP", "SIDEWAYS", "DOWN"])].copy()
        production = False
    else:
        production = True
    if usable.empty:
        return {
            "consensus_direction": "NO_EDGE",
            "consensus_confidence": 0.0,
            "consensus_prob_up": np.nan,
            "consensus_prob_sideways": np.nan,
            "consensus_prob_down": np.nan,
            "actionable_horizons": 0,
        }
    weights = np.asarray(
        [
            HORIZON_CONSENSUS_WEIGHTS.get(int(value), 1.0)
            * max(0.20, _finite(confidence, 0.0) / 100.0)
            for value, confidence in zip(usable["horizon_bars"], usable["confidence_pct"])
        ],
        dtype=float,
    )
    if weights.sum() <= 0:
        weights = np.ones(len(usable), dtype=float)
    probabilities = {
        "UP": float(np.average(pd.to_numeric(usable["prob_up_pct"], errors="coerce") / 100.0, weights=weights)),
        "SIDEWAYS": float(np.average(pd.to_numeric(usable["prob_sideways_pct"], errors="coerce") / 100.0, weights=weights)),
        "DOWN": float(np.average(pd.to_numeric(usable["prob_down_pct"], errors="coerce") / 100.0, weights=weights)),
    }
    ordered = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    direction = ordered[0][0] if ordered[0][1] - ordered[1][1] >= 0.06 else "MIXED"
    if not production:
        direction = "NO_EDGE"
    return {
        "consensus_direction": direction,
        "consensus_confidence": round(float(np.average(usable["confidence_pct"], weights=weights)), 1),
        "consensus_prob_up": round(100.0 * probabilities["UP"], 1),
        "consensus_prob_sideways": round(100.0 * probabilities["SIDEWAYS"], 1),
        "consensus_prob_down": round(100.0 * probabilities["DOWN"], 1),
        "actionable_horizons": int(horizons["actionable"].fillna(False).astype(bool).sum()),
    }


def _risk_budget(
    regime: str,
    consensus_direction: str,
    crash_risk: bool,
    data_state: str,
) -> tuple[float, str]:
    if data_state != "READY":
        return 0.50, "DEFENSIVE_DATA_UNCERTAIN"
    base = {
        "BULL_CONFIRMED": 1.00,
        "BULL_FRAGILE": 0.80,
        "TRANSITION": 0.70,
        "BEAR_RALLY": 0.55,
        "BEAR_CONFIRMED": 0.35,
        "UNKNOWN": 0.50,
    }.get(regime, 0.50)
    if consensus_direction == "DOWN":
        base = min(base, 0.45)
    elif consensus_direction in {"MIXED", "NO_EDGE"}:
        base = min(base, 0.70)
    elif consensus_direction == "UP" and regime == "BEAR_CONFIRMED":
        base = min(base, 0.50)
    if crash_risk:
        base = min(base, 0.25)
    base = float(np.clip(base, 0.20, 1.00))
    if base >= 0.95:
        action = "NORMAL_RISK_CAP"
    elif base >= 0.70:
        action = "REDUCED_RISK_CAP"
    elif base >= 0.45:
        action = "DEFENSIVE_RISK_CAP"
    else:
        action = "CAPITAL_PRESERVATION"
    return base, action


def analyze_ihsg_direction(
    benchmark: pd.DataFrame | None,
    prepared: Mapping[str, pd.DataFrame] | None = None,
    *,
    config: IHSGDirectionConfig | None = None,
    now: Any | None = None,
    eod_final: bool = True,
) -> dict[str, Any]:
    """Estimate IHSG direction with chronological validation and abstention."""
    cfg = config or IHSGDirectionConfig()
    features = build_ihsg_feature_frame(benchmark, prepared, cfg)
    data_state, data_age_days = _data_state(features, cfg, now)
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    if features.empty:
        horizons = _empty_horizon_rows(cfg, data_state)
        return {
            "version": IHSG_DIRECTION_VERSION,
            "schema_version": IHSG_DIRECTION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of": "",
            "data_state": data_state,
            "data_age_days": data_age_days,
            "eod_final": bool(eod_final),
            "regime": "UNKNOWN",
            "regime_score": 0.0,
            "regime_reason": "Benchmark IHSG tidak tersedia",
            "crash_risk": False,
            "consensus_direction": "NO_EDGE",
            "consensus_confidence": 0.0,
            "risk_budget_multiplier": 0.50,
            "risk_action": "DEFENSIVE_DATA_UNCERTAIN",
            "feature_coverage_pct": 0.0,
            "breadth_member_count": 0,
            "horizons": horizons,
            "validation": horizons.copy(),
            "feature_snapshot": {},
            "limitations": [
                "Tidak ada prediksi tanpa benchmark yang cukup.",
                "Arah IHSG bukan jaminan arah setiap saham.",
            ],
        }
    if data_state != "READY":
        last = features.iloc[-1]
        regime = _regime_profile(last)
        feature_available = sum(
            np.isfinite(_finite(last.get(name), np.nan)) for name in FEATURE_COLUMNS
        )
        horizons = _empty_horizon_rows(cfg, data_state)
        risk_multiplier, risk_action = _risk_budget(
            regime["regime"], "NO_EDGE", bool(regime["crash_risk"]), data_state
        )
        return {
            "version": IHSG_DIRECTION_VERSION,
            "schema_version": IHSG_DIRECTION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of": pd.Timestamp(features.index[-1]).date().isoformat(),
            "data_state": data_state,
            "data_age_days": int(data_age_days),
            "eod_final": bool(eod_final),
            "benchmark_close": round(_finite(last.get("Close"), np.nan), 2),
            **regime,
            "consensus_direction": "NO_EDGE",
            "consensus_confidence": 0.0,
            "consensus_prob_up": np.nan,
            "consensus_prob_sideways": np.nan,
            "consensus_prob_down": np.nan,
            "actionable_horizons": 0,
            "risk_budget_multiplier": round(risk_multiplier, 2),
            "risk_budget_pct": round(100.0 * risk_multiplier, 1),
            "risk_action": risk_action,
            "feature_coverage_pct": round(
                100.0 * feature_available / len(FEATURE_COLUMNS), 1
            ),
            "breadth_member_count": int(
                max(0.0, _finite(last.get("breadth_member_count"), 0.0))
            ),
            "breadth_ema20_pct": round(
                100.0 * _finite(last.get("breadth_ema20"), np.nan), 1
            ),
            "breadth_ema50_pct": round(
                100.0 * _finite(last.get("breadth_ema50"), np.nan), 1
            ),
            "breadth_ema200_pct": round(
                100.0 * _finite(last.get("breadth_ema200"), np.nan), 1
            ),
            "feature_hash": "",
            "horizons": horizons,
            "validation": horizons.copy(),
            "feature_snapshot": {},
            "config": asdict(cfg),
            "limitations": [
                f"Prediksi dihentikan karena data_state={data_state}.",
                "Arah IHSG bukan jaminan arah setiap saham.",
                "Risk budget hanya boleh mengurangi eksposur.",
            ],
        }
    last = features.iloc[-1]
    feature_available = sum(np.isfinite(_finite(last.get(name), np.nan)) for name in FEATURE_COLUMNS)
    feature_coverage = feature_available / len(FEATURE_COLUMNS)
    regime = _regime_profile(last)
    horizon_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    target_position = len(features) - 1
    for horizon in cfg.horizons:
        horizon = max(1, int(horizon))
        distribution = _analogue_distribution(features, target_position, horizon, cfg)
        validation = _walk_forward_validation(features, horizon, cfg)
        raw_direction = _raw_direction(distribution, cfg)
        confidence = _confidence_score(distribution, validation, feature_coverage, cfg)
        production_allowed = (
            data_state == "READY"
            and bool(eod_final)
            and validation.get("validation_state") == "OOS_POSITIVE"
            and confidence >= float(cfg.min_production_confidence)
            and raw_direction in {"UP", "SIDEWAYS", "DOWN"}
        )
        prediction_state = raw_direction if production_allowed else "ABSTAIN"
        reason_parts = []
        if data_state != "READY":
            reason_parts.append(data_state)
        if not eod_final:
            reason_parts.append("WAIT_FINAL_EOD")
        if validation.get("validation_state") != "OOS_POSITIVE":
            reason_parts.append(_text(validation.get("validation_state")) or "NO_OOS_EDGE")
        if confidence < float(cfg.min_production_confidence):
            reason_parts.append("LOW_CONFIDENCE")
        if raw_direction == "ABSTAIN":
            reason_parts.append("PROBABILITY_NOT_SEPARATED")
        row = {
            "horizon": HORIZON_LABELS.get(horizon, f"{horizon}D"),
            "horizon_bars": horizon,
            "raw_direction": raw_direction,
            "prediction_state": prediction_state,
            "prob_up_pct": round(100.0 * _finite(distribution.get("prob_up"), np.nan), 1),
            "prob_sideways_pct": round(100.0 * _finite(distribution.get("prob_sideways"), np.nan), 1),
            "prob_down_pct": round(100.0 * _finite(distribution.get("prob_down"), np.nan), 1),
            "confidence_pct": round(confidence, 1),
            "expected_return_pct": round(100.0 * _finite(distribution.get("expected_return"), np.nan), 2),
            "return_p25_pct": round(100.0 * _finite(distribution.get("return_p25"), np.nan), 2),
            "return_p75_pct": round(100.0 * _finite(distribution.get("return_p75"), np.nan), 2),
            "neutral_band_pct": round(100.0 * _finite(distribution.get("neutral_band"), np.nan), 2),
            "analogue_count": int(_finite(distribution.get("analogue_count"), 0.0)),
            "effective_analogue_count": round(
                _finite(distribution.get("effective_analogue_count"), 0.0), 1
            ),
            "features_used": int(_finite(distribution.get("features_used"), 0.0)),
            "median_distance": round(_finite(distribution.get("median_distance"), np.nan), 3),
            "validation_state": _text(validation.get("validation_state")),
            "validation_predictions": int(_finite(validation.get("validation_predictions"), 0.0)),
            "directional_validation_predictions": int(
                _finite(validation.get("directional_predictions"), 0.0)
            ),
            "directional_accuracy_pct": round(
                100.0 * _finite(validation.get("directional_accuracy"), np.nan), 1
            ),
            "directional_accuracy_ci_low_pct": round(
                100.0 * _finite(validation.get("directional_accuracy_ci_low"), np.nan), 1
            ),
            "validation_coverage_pct": round(
                100.0 * _finite(validation.get("coverage"), 0.0), 1
            ),
            "brier_score": round(_finite(validation.get("brier_score"), np.nan), 4),
            "baseline_brier_score": round(
                _finite(validation.get("baseline_brier_score"), np.nan), 4
            ),
            "brier_skill_pct": round(
                100.0 * _finite(validation.get("brier_skill"), np.nan), 1
            ),
            "actionable": bool(production_allowed),
            "abstain_reason": "|".join(dict.fromkeys(reason_parts)),
            "feature_names": _text(distribution.get("feature_names")),
        }
        horizon_rows.append(row)
        validation_rows.append(
            {
                key: row[key]
                for key in (
                    "horizon",
                    "horizon_bars",
                    "validation_state",
                    "validation_predictions",
                    "directional_validation_predictions",
                    "directional_accuracy_pct",
                    "directional_accuracy_ci_low_pct",
                    "validation_coverage_pct",
                    "brier_score",
                    "baseline_brier_score",
                    "brier_skill_pct",
                )
            }
        )
    horizons = pd.DataFrame(horizon_rows)
    consensus = _consensus(horizons)
    risk_multiplier, risk_action = _risk_budget(
        regime["regime"],
        consensus["consensus_direction"],
        bool(regime["crash_risk"]),
        data_state,
    )
    breadth_count = int(max(0.0, _finite(last.get("breadth_member_count"), 0.0)))
    as_of = pd.Timestamp(features.index[-1]).date().isoformat()
    snapshot = {
        name: (
            round(_finite(last.get(name)), 6)
            if np.isfinite(_finite(last.get(name), np.nan))
            else None
        )
        for name in FEATURE_COLUMNS
    }
    feature_hash = sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "version": IHSG_DIRECTION_VERSION,
        "schema_version": IHSG_DIRECTION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "as_of": as_of,
        "data_state": data_state,
        "data_age_days": int(data_age_days),
        "eod_final": bool(eod_final),
        "benchmark_close": round(_finite(last.get("Close"), np.nan), 2),
        **regime,
        **consensus,
        "risk_budget_multiplier": round(risk_multiplier, 2),
        "risk_budget_pct": round(100.0 * risk_multiplier, 1),
        "risk_action": risk_action,
        "feature_coverage_pct": round(100.0 * feature_coverage, 1),
        "breadth_member_count": breadth_count,
        "breadth_ema20_pct": round(100.0 * _finite(last.get("breadth_ema20"), np.nan), 1),
        "breadth_ema50_pct": round(100.0 * _finite(last.get("breadth_ema50"), np.nan), 1),
        "breadth_ema200_pct": round(100.0 * _finite(last.get("breadth_ema200"), np.nan), 1),
        "feature_hash": feature_hash,
        "horizons": horizons,
        "validation": pd.DataFrame(validation_rows),
        "feature_snapshot": snapshot,
        "config": asdict(cfg),
        "limitations": [
            "Probabilitas adalah frekuensi historical analogue, bukan kepastian.",
            "Breadth memakai universe yang diunggah dan dapat mengandung survivorship/selection bias.",
            "Corporate action, makro intraday, rupiah, obligasi, dan arus asing belum menjadi input otomatis.",
            "Risk budget hanya boleh mengurangi eksposur; ranking saham tetap quality-first.",
        ],
    }


def ihsg_snapshot_frame(forecast: Mapping[str, Any] | None) -> pd.DataFrame:
    if not isinstance(forecast, Mapping):
        return pd.DataFrame()
    horizons = forecast.get("horizons")
    if not isinstance(horizons, pd.DataFrame) or horizons.empty:
        return pd.DataFrame()
    out = horizons.copy()
    shared = {
        "ticker": IHSG_TICKER,
        "as_of": _text(forecast.get("as_of")),
        "data_state": _text(forecast.get("data_state")),
        "eod_final": bool(forecast.get("eod_final")),
        "benchmark_close": _finite(forecast.get("benchmark_close"), np.nan),
        "regime": _text(forecast.get("regime")),
        "regime_score": _finite(forecast.get("regime_score"), np.nan),
        "consensus_direction": _text(forecast.get("consensus_direction")),
        "consensus_confidence": _finite(forecast.get("consensus_confidence"), np.nan),
        "risk_budget_multiplier": _finite(forecast.get("risk_budget_multiplier"), np.nan),
        "risk_action": _text(forecast.get("risk_action")),
        "feature_coverage_pct": _finite(forecast.get("feature_coverage_pct"), np.nan),
        "breadth_member_count": int(_finite(forecast.get("breadth_member_count"), 0.0)),
        "breadth_ema50_pct": _finite(forecast.get("breadth_ema50_pct"), np.nan),
        "feature_hash": _text(forecast.get("feature_hash")),
        "model_version": _text(forecast.get("version")) or IHSG_DIRECTION_VERSION,
        "payload": {
            "limitations": list(forecast.get("limitations", []) or []),
            "feature_snapshot": dict(forecast.get("feature_snapshot", {}) or {}),
            "risk_cap_applied": bool(forecast.get("risk_cap_applied")),
            "risk_cap_policy": _text(forecast.get("risk_cap_policy")),
            "effective_risk_per_trade_pct": _finite(
                forecast.get("effective_risk_per_trade_pct"), np.nan
            ),
            "effective_multibagger_budget_idr": _finite(
                forecast.get("effective_multibagger_budget_idr"), np.nan
            ),
        },
    }
    for key, value in shared.items():
        out[key] = [value] * len(out)
    return out


def _outcome_id(signal_date: str, horizon: int, model_version: str) -> str:
    payload = f"{IHSG_TICKER}|IHSG_DIRECTION_{int(horizon)}D|{signal_date}|{model_version}"
    return sha256(payload.encode("utf-8")).hexdigest()


def _prediction_outcome_rows(
    forecast: Mapping[str, Any] | None,
    as_of: Any | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(forecast, Mapping):
        return []
    horizons = forecast.get("horizons")
    if not isinstance(horizons, pd.DataFrame) or horizons.empty:
        return []
    timestamp = pd.to_datetime(as_of or forecast.get("generated_at"), errors="coerce", utc=True)
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.now(tz="UTC")
    signal_date = _text(forecast.get("as_of")) or timestamp.tz_convert("Asia/Jakarta").date().isoformat()
    model = _text(forecast.get("version")) or IHSG_DIRECTION_VERSION
    rows: list[dict[str, Any]] = []
    for _, horizon_row in horizons.iterrows():
        horizon = max(1, int(_finite(horizon_row.get("horizon_bars"), 1.0)))
        direction = _text(horizon_row.get("prediction_state")).upper() or "ABSTAIN"
        raw_direction = _text(horizon_row.get("raw_direction")).upper() or "ABSTAIN"
        probability_map = {
            "UP": _finite(horizon_row.get("prob_up_pct"), np.nan),
            "SIDEWAYS": _finite(horizon_row.get("prob_sideways_pct"), np.nan),
            "DOWN": _finite(horizon_row.get("prob_down_pct"), np.nan),
        }
        signal_score = max(
            (value for value in probability_map.values() if np.isfinite(value)),
            default=np.nan,
        )
        rows.append(
            {
                "outcome_id": _outcome_id(signal_date, horizon, model),
                "ticker": IHSG_TICKER,
                "signal_family": f"IHSG_DIRECTION_{horizon}D",
                "signal_timestamp": timestamp.isoformat(),
                "signal_date": signal_date,
                "anchor_id": f"{signal_date}|{horizon}D|{_text(forecast.get('feature_hash'))[:16]}",
                "liquidity_bucket": "INDEX",
                "predicted_state": direction,
                "predicted_direction": direction,
                "signal_score": signal_score,
                "signal_confidence": _finite(horizon_row.get("confidence_pct"), np.nan),
                "prediction_window_start": signal_date,
                "prediction_window_end": "",
                "entry_reference": _finite(forecast.get("benchmark_close"), np.nan),
                "horizon_bars": horizon,
                "outcome_status": "OPEN",
                "model_version": model,
                "payload": {
                    "raw_direction": raw_direction,
                    "prob_up_pct": probability_map["UP"],
                    "prob_sideways_pct": probability_map["SIDEWAYS"],
                    "prob_down_pct": probability_map["DOWN"],
                    "neutral_band_pct": _finite(horizon_row.get("neutral_band_pct"), np.nan),
                    "validation_state": _text(horizon_row.get("validation_state")),
                    "brier_skill_pct": _finite(horizon_row.get("brier_skill_pct"), np.nan),
                    "regime": _text(forecast.get("regime")),
                    "risk_budget_multiplier": _finite(
                        forecast.get("risk_budget_multiplier"), np.nan
                    ),
                    "feature_hash": _text(forecast.get("feature_hash")),
                },
            }
        )
    return rows


def _resolve_ihsg_outcome(
    record: Mapping[str, Any],
    benchmark: pd.DataFrame,
) -> dict[str, Any]:
    local = dict(record)
    frame = _normalise_ohlcv(benchmark)
    signal_date = pd.to_datetime(
        local.get("signal_date") or local.get("signal_timestamp"), errors="coerce"
    )
    if frame.empty or pd.isna(signal_date):
        return local
    horizon = max(1, int(_finite(local.get("horizon_bars"), 1.0)))
    future = frame.loc[
        frame.index.normalize() > pd.Timestamp(signal_date).normalize()
    ].head(horizon)
    if len(future) < horizon:
        return local
    entry = _finite(local.get("entry_reference"), np.nan)
    if not np.isfinite(entry) or entry <= 0:
        prior = frame.loc[
            frame.index.normalize() <= pd.Timestamp(signal_date).normalize(), "Close"
        ].tail(1)
        entry = _finite(prior.iloc[0] if not prior.empty else np.nan, np.nan)
    if not np.isfinite(entry) or entry <= 0:
        return local
    close = _finite(future["Close"].iloc[horizon - 1], np.nan)
    if not np.isfinite(close):
        return local
    realised = close / entry - 1.0
    local[f"forward_return_{horizon}d"] = realised
    if horizon not in {1, 5, 10, 20}:
        payload = dict(local.get("payload", {}) or {})
        payload["actual_forward_return"] = realised
        local["payload"] = payload
    highs = pd.to_numeric(future["High"], errors="coerce")
    lows = pd.to_numeric(future["Low"], errors="coerce")
    local["maximum_favourable_excursion"] = (
        float(highs.max() / entry - 1.0) if highs.notna().any() else np.nan
    )
    local["maximum_adverse_excursion"] = (
        float(lows.min() / entry - 1.0) if lows.notna().any() else np.nan
    )
    local["actual_high_date"] = (
        pd.Timestamp(highs.idxmax()).date().isoformat() if highs.notna().any() else ""
    )
    local["actual_low_date"] = (
        pd.Timestamp(lows.idxmin()).date().isoformat() if lows.notna().any() else ""
    )
    payload = dict(local.get("payload", {}) or {})
    band = max(0.0, _finite(payload.get("neutral_band_pct"), 0.0) / 100.0)
    actual_direction = _classify_return(realised, band)
    predicted = _text(local.get("predicted_direction")).upper()
    local["hit"] = None if predicted == "ABSTAIN" else bool(predicted == actual_direction)
    payload["actual_direction"] = actual_direction
    payload["actual_forward_return"] = realised
    local["payload"] = payload
    local["outcome_status"] = "RESOLVED"
    resolved = pd.Timestamp(future.index[-1])
    if resolved.tzinfo is None:
        resolved = resolved.tz_localize("Asia/Jakarta")
    local["resolved_at"] = resolved.tz_convert("UTC").isoformat()
    return local


def update_ihsg_outcomes(
    existing: pd.DataFrame | None,
    forecast: Mapping[str, Any] | None,
    benchmark: pd.DataFrame | None,
    *,
    as_of: Any | None = None,
) -> pd.DataFrame:
    """Append/deduplicate IHSG forecasts and resolve them after forward bars."""
    records: dict[str, dict[str, Any]] = {}
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        for row in existing.to_dict("records"):
            outcome_id = _text(row.get("outcome_id"))
            if outcome_id:
                records[outcome_id] = dict(row)
    for row in _prediction_outcome_rows(forecast, as_of=as_of):
        records.setdefault(row["outcome_id"], row)
    clean_benchmark = _normalise_ohlcv(benchmark)
    if not clean_benchmark.empty:
        for outcome_id, row in list(records.items()):
            if _text(row.get("outcome_status")).upper() == "RESOLVED":
                continue
            if not _text(row.get("signal_family")).upper().startswith("IHSG_DIRECTION_"):
                continue
            records[outcome_id] = _resolve_ihsg_outcome(row, clean_benchmark)
    columns = [
        "outcome_id",
        "ticker",
        "signal_family",
        "signal_timestamp",
        "signal_date",
        "anchor_id",
        "liquidity_bucket",
        "predicted_state",
        "predicted_direction",
        "signal_score",
        "signal_confidence",
        "prediction_window_start",
        "prediction_window_end",
        "entry_reference",
        "horizon_bars",
        "outcome_status",
        "resolved_at",
        "actual_low_date",
        "actual_high_date",
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_10d",
        "forward_return_20d",
        "maximum_favourable_excursion",
        "maximum_adverse_excursion",
        "hit",
        "model_version",
        "payload",
    ]
    frame = pd.DataFrame(list(records.values()))
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame[columns].sort_values(
        ["signal_timestamp", "ticker", "signal_family"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def make_ihsg_direction_chart(
    benchmark: pd.DataFrame | None,
    forecast: Mapping[str, Any] | None = None,
    bars: int = 320,
):
    """Create a compact IHSG trend chart; Plotly is imported lazily."""
    frame = _normalise_ohlcv(benchmark)
    if frame.empty:
        return None
    import plotly.graph_objects as go

    local = frame.tail(max(60, int(bars))).copy()
    local["EMA20"] = _ema(frame["Close"], 20).reindex(local.index)
    local["EMA50"] = _ema(frame["Close"], 50).reindex(local.index)
    local["EMA200"] = _ema(frame["Close"], 200).reindex(local.index)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=local.index,
            open=local["Open"],
            high=local["High"],
            low=local["Low"],
            close=local["Close"],
            name="IHSG",
            increasing_line_color="#20c997",
            decreasing_line_color="#ff5c6c",
        )
    )
    for column, color in (
        ("EMA20", "#f6c85f"),
        ("EMA50", "#6f9ceb"),
        ("EMA200", "#ad75f4"),
    ):
        fig.add_trace(
            go.Scatter(
                x=local.index,
                y=local[column],
                name=column,
                line={"color": color, "width": 1.4},
            )
        )
    title = "IHSG — trend dan regime"
    if isinstance(forecast, Mapping):
        title += (
            f" · {_text(forecast.get('regime'))}"
            f" · consensus {_text(forecast.get('consensus_direction'))}"
        )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=590,
        margin={"l": 20, "r": 40, "t": 55, "b": 20},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.02},
        hovermode="x unified",
    )
    return fig


__all__ = [
    "IHSG_DIRECTION_VERSION",
    "IHSG_DIRECTION_SCHEMA_VERSION",
    "IHSG_TICKER",
    "IHSGDirectionConfig",
    "FEATURE_COLUMNS",
    "build_ihsg_feature_frame",
    "analyze_ihsg_direction",
    "ihsg_snapshot_frame",
    "update_ihsg_outcomes",
    "make_ihsg_direction_chart",
]

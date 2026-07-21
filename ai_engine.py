"""Local hybrid AI for IDX Super Scanner.

The module intentionally avoids paid APIs and heavy model runtimes.  It combines:
- regularised logistic learning,
- nearest-neighbour similarity,
- Bayesian strategy/regime priors,
- chronological holdout calibration,
- feature-drift detection,
- an outcome memory that can be exported/imported.

The rule engine remains authoritative for structural validity.  AI only adjusts
ranking/conviction when enough empirical data exists.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import io
import json
import math
import os

import numpy as np
import pandas as pd

AI_VERSION = "6.6.4-strict-validated-hybrid-ai"

NUMERIC_FEATURES = (
    "structure_score",
    "timing_score",
    "flow_score",
    "liquidity_score",
    "target_quality_score",
    "data_quality_score",
    "validation_score",
    "rr1",
    "rr2",
    "stop_pct",
    "atr_pct",
    "volume_ratio",
    "rsi14",
    "adx14",
    "cmf20",
    "roc60",
    "distance_52w_high",
    "relative_strength60",
    "silent_accumulation_score",
    "body_atr",
    "close_location",
)

STRATEGIES = (
    "PULLBACK_CONTINUATION",
    "BREAKOUT_RETEST",
    "REVERSAL_ACCUMULATION",
    "SNIPER",
    "BPJS",
    "BSJP",
    "PRE_ARA",
    "ARA_CONTINUATION",
)

REGIMES = ("RISK_ON", "NEUTRAL", "RISK_OFF", "UNKNOWN")


@dataclass
class LocalAIConfig:
    enabled: bool = True
    mode: str = "HYBRID_GUARDED"  # HYBRID_GUARDED | SHADOW_ONLY | RULE_ONLY
    max_weight: float = 0.35
    min_training_events: int = 30
    min_strategy_events: int = 18
    min_positive_events: int = 5
    calibration_fraction: float = 0.25
    min_calibration_events: int = 20
    min_evaluation_events: int = 20
    min_feature_coverage: float = 0.60
    min_brier_skill: float = 0.02
    # Retained for backward-compatible configuration parsing only.  A model
    # that has not passed both chronological OOS gates is now shadow-only.
    max_unvalidated_weight: float = 0.0
    recency_half_life_days: float = 540.0
    max_iterations: int = 650
    learning_rate: float = 0.06
    l2: float = 0.12
    knn_k: int = 21
    memory_entry_window_bars: int = 5
    memory_horizon_bars: int = 20
    memory_max_rows: int = 10000
    conservative_same_bar_resolution: bool = True
    roundtrip_cost_pct: float = 0.0065
    max_entry_gap_atr: float = 0.75

    def replace(self, **changes: Any) -> "LocalAIConfig":
        values = asdict(self)
        values.update(changes)
        return LocalAIConfig(**values)


@dataclass
class _Scaler:
    median: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.median) / self.scale, -8.0, 8.0)


@dataclass
class _LogisticModel:
    coef: np.ndarray
    intercept: float
    scaler: _Scaler
    feature_names: list[str]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = _matrix(frame, self.feature_names)
        # Live ranking rows legitimately have strategy-specific missing fields.
        # Training used median imputation, so inference must use the identical
        # transformation instead of propagating NaN through every coefficient.
        x = np.where(np.isfinite(x), x, self.scaler.median)
        z = self.scaler.transform(x) @ self.coef + self.intercept
        return _sigmoid(z)


@dataclass
class _PlattModel:
    slope: float = 1.0
    intercept: float = 0.0

    def predict(self, probability: np.ndarray) -> np.ndarray:
        p = np.clip(probability, 1e-5, 1 - 1e-5)
        logits = np.log(p / (1 - p))
        return _sigmoid(self.slope * logits + self.intercept)


def _safe_number(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return _safe_text(value).upper() in {"1", "TRUE", "YES", "Y", "READY", "OK"}


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(value: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(value, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def _normalize_strategy(value: Any) -> str:
    text = _safe_text(value).upper().replace("UNICORN_SNIPER_ICT", "SNIPER")
    aliases = {
        "SNIPER_ICT": "SNIPER",
        "PRE-ARA": "PRE_ARA",
        "ARA_CONTINUATION_TECHNICAL": "ARA_CONTINUATION",
    }
    return aliases.get(text, text if text in STRATEGIES else "UNKNOWN")


def _normalize_regime(value: Any) -> str:
    text = _safe_text(value).upper()
    return text if text in REGIMES else "UNKNOWN"


def _feature_names() -> list[str]:
    names = list(NUMERIC_FEATURES)
    names.extend([f"strategy::{name}" for name in STRATEGIES])
    names.extend([f"regime::{name}" for name in REGIMES])
    names.extend(["decision::ORDER_READY", "decision::SETUP_READY", "decision::ENTRY_PLAN"])
    return names


def _canonical_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=_feature_names())
    out = pd.DataFrame(index=frame.index)
    for name in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[name], errors="coerce") if name in frame else pd.Series(np.nan, index=frame.index)
        out[name] = values
    strategy = frame.get("strategy", frame.get("setup", pd.Series("UNKNOWN", index=frame.index))).map(_normalize_strategy)
    regime = frame.get("market_regime", pd.Series("UNKNOWN", index=frame.index)).map(_normalize_regime)
    decision = frame.get("decision_state", pd.Series("SETUP_READY", index=frame.index)).astype(str).str.upper()
    for name in STRATEGIES:
        out[f"strategy::{name}"] = strategy.eq(name).astype(float)
    for name in REGIMES:
        out[f"regime::{name}"] = regime.eq(name).astype(float)
    for name in ("ORDER_READY", "SETUP_READY", "ENTRY_PLAN"):
        out[f"decision::{name}"] = decision.eq(name).astype(float)
    return out


def _matrix(frame: pd.DataFrame, names: Iterable[str]) -> np.ndarray:
    canonical = _canonical_feature_frame(frame)
    result = canonical.reindex(columns=list(names)).apply(pd.to_numeric, errors="coerce")
    return result.to_numpy(dtype=float)


def _feature_coverage(frame: pd.DataFrame) -> np.ndarray:
    """Return signal-time numeric feature coverage for each candidate."""
    if frame is None or frame.empty:
        return np.asarray([], dtype=float)
    raw = _matrix(frame, NUMERIC_FEATURES)
    if raw.shape[1] == 0:
        return np.zeros(len(frame), dtype=float)
    return np.mean(np.isfinite(raw), axis=1)


def _recency_weights(frame: pd.DataFrame, half_life_days: float) -> np.ndarray:
    """Down-weight stale market regimes without using any future information."""
    if frame is None or frame.empty:
        return np.asarray([], dtype=float)
    dates = pd.to_datetime(frame.get("signal_date"), errors="coerce")
    if not isinstance(dates, pd.Series) or dates.notna().sum() < 2 or half_life_days <= 0:
        return np.ones(len(frame), dtype=float)
    anchor = dates.max()
    age = (anchor - dates).dt.total_seconds().div(86400.0)
    fallback_age = float(age.dropna().median()) if age.notna().any() else 0.0
    age = age.fillna(fallback_age).clip(lower=0.0)
    weights = np.exp(-math.log(2.0) * age.to_numpy(dtype=float) / float(half_life_days))
    return np.clip(weights, 0.15, 1.0)


def _outcome_quality_weights(frame: pd.DataFrame) -> np.ndarray:
    if frame is None or frame.empty:
        return np.asarray([], dtype=float)
    quality = frame.get("outcome_quality", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    return quality.map({
        "BROKER_CONFIRMED": 1.20,
        "CHRONOLOGICAL_BACKTEST": 1.00,
        "DAILY_APPROX": 0.75,
    }).fillna(0.70).to_numpy(dtype=float)


def _fit_scaler(x: np.ndarray) -> _Scaler:
    median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(x), x, median)
    q75 = np.nanpercentile(filled, 75, axis=0)
    q25 = np.nanpercentile(filled, 25, axis=0)
    scale = (q75 - q25) / 1.349
    std = np.nanstd(filled, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return _Scaler(median=median.astype(float), scale=scale.astype(float))


def _fit_logistic(
    frame: pd.DataFrame,
    target: pd.Series,
    config: LocalAIConfig,
    minimum_events: int | None = None,
) -> _LogisticModel | None:
    names = _feature_names()
    x_raw = _matrix(frame, names)
    y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    valid_y = np.isfinite(y)
    x_raw, y = x_raw[valid_y], y[valid_y]
    minimum = int(config.min_training_events if minimum_events is None else minimum_events)
    if len(y) < max(12, minimum) or len(np.unique(y)) < 2:
        return None
    valid_frame = frame.loc[valid_y].copy()
    scaler = _fit_scaler(x_raw)
    x = scaler.transform(np.where(np.isfinite(x_raw), x_raw, scaler.median))
    coef = np.zeros(x.shape[1], dtype=float)
    weights = _recency_weights(valid_frame, config.recency_half_life_days) * _outcome_quality_weights(valid_frame)
    weights = weights / max(1e-9, float(np.mean(weights)))
    prevalence = float(np.clip(np.average(y, weights=weights), 1e-4, 1 - 1e-4))
    intercept = float(math.log(prevalence / (1 - prevalence)))
    lr = float(config.learning_rate)
    for step in range(int(config.max_iterations)):
        pred = _sigmoid(x @ coef + intercept)
        error = (pred - y) * weights
        # Probability estimation should preserve prevalence; class balancing
        # would force the raw model toward 50/50. L2 is applied on the same
        # mean-loss scale as the data gradient.
        grad_coef = x.T @ error / len(y) + config.l2 * coef
        grad_intercept = float(np.mean(error))
        adaptive_lr = lr / (1.0 + 0.003 * step)
        coef -= adaptive_lr * np.clip(grad_coef, -5.0, 5.0)
        intercept -= adaptive_lr * float(np.clip(grad_intercept, -2.0, 2.0))
    return _LogisticModel(coef=coef, intercept=intercept, scaler=scaler, feature_names=names)


def _fit_platt(raw_probability: np.ndarray, target: np.ndarray) -> _PlattModel:
    if len(target) < 12 or len(np.unique(target)) < 2:
        return _PlattModel()
    x = _logit(raw_probability)
    y = target.astype(float)
    slope, intercept = 1.0, 0.0
    for step in range(350):
        pred = _sigmoid(slope * x + intercept)
        error = pred - y
        lr = 0.035 / (1.0 + 0.004 * step)
        slope -= lr * float(np.mean(error * x) + 0.02 * (slope - 1.0))
        intercept -= lr * float(np.mean(error))
        slope = float(np.clip(slope, 0.15, 4.0))
        intercept = float(np.clip(intercept, -4.0, 4.0))
    return _PlattModel(slope=slope, intercept=intercept)


def _beta_probability(successes: float, total: float, prior_success: float = 6.0, prior_failure: float = 6.0) -> tuple[float, float]:
    alpha = successes + prior_success
    beta = max(0, total - successes) + prior_failure
    mean = alpha / (alpha + beta)
    variance = alpha * beta / (((alpha + beta) ** 2) * (alpha + beta + 1))
    return float(mean), float(math.sqrt(max(0.0, variance)))


def _strategy_regime_prior(
    history: pd.DataFrame,
    strategy: str,
    regime: str,
    target_col: str,
    config: LocalAIConfig,
) -> tuple[float, float, float]:
    if history is None or history.empty or target_col not in history:
        return 0.5, 0.14, 0.0
    local = history.copy()
    local["_strategy"] = local.get("strategy", local.get("setup", "UNKNOWN")).map(_normalize_strategy)
    local["_regime"] = local.get("market_regime", "UNKNOWN").map(_normalize_regime)
    target = pd.to_numeric(local[target_col], errors="coerce")
    local = local[target.notna()].copy()
    local["_target"] = target[target.notna()].astype(int)
    exact = local[(local["_strategy"] == strategy) & (local["_regime"] == regime)]
    strategy_only = local[local["_strategy"] == strategy]
    chosen = exact if len(exact) >= 8 else strategy_only if len(strategy_only) >= 8 else local
    weights = _recency_weights(chosen, config.recency_half_life_days) * _outcome_quality_weights(chosen)
    successes = float(np.sum(weights * chosen["_target"].to_numpy(dtype=float))) if len(chosen) else 0.0
    effective_total = float(np.sum(weights)) if len(chosen) else 0.0
    mean, std = _beta_probability(successes, effective_total)
    return mean, std, effective_total


def _knn_prediction(
    train_frame: pd.DataFrame,
    target: pd.Series,
    current_frame: pd.DataFrame,
    config: LocalAIConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if train_frame is None or train_frame.empty:
        return np.full(len(current_frame), 0.5), np.full(len(current_frame), np.nan)
    names = _feature_names()
    x_train_raw = _matrix(train_frame, names)
    y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)
    x_train_raw, y = x_train_raw[valid], y[valid]
    if len(y) < 12:
        return np.full(len(current_frame), 0.5), np.full(len(current_frame), np.nan)
    scaler = _fit_scaler(x_train_raw)
    x_train = scaler.transform(np.where(np.isfinite(x_train_raw), x_train_raw, scaler.median))
    x_now_raw = _matrix(current_frame, names)
    x_now = scaler.transform(np.where(np.isfinite(x_now_raw), x_now_raw, scaler.median))
    probabilities, expected = [], []
    r_values = pd.to_numeric(train_frame.loc[valid].get("r_multiple", pd.Series(np.nan, index=np.flatnonzero(valid))), errors="coerce").to_numpy(dtype=float)
    valid_frame = train_frame.loc[valid].copy()
    recency = _recency_weights(valid_frame, config.recency_half_life_days) * _outcome_quality_weights(valid_frame)
    strategy_train = valid_frame.get("strategy", valid_frame.get("setup", "UNKNOWN")).map(_normalize_strategy).to_numpy()
    strategy_now = current_frame.get("strategy", current_frame.get("setup", "UNKNOWN")).map(_normalize_strategy).to_numpy()
    for idx, vector in enumerate(x_now):
        distances = np.sqrt(np.mean((x_train - vector) ** 2, axis=1))
        same = strategy_train == strategy_now[idx]
        distances = distances * np.where(same, 0.72, 1.15)
        k = min(max(7, int(config.knn_k)), len(distances))
        nearest = np.argpartition(distances, k - 1)[:k]
        weights = recency[nearest] / np.maximum(0.15, distances[nearest])
        probability = float(np.average(y[nearest], weights=weights))
        valid_r = np.isfinite(r_values[nearest])
        expected_r = float(np.average(r_values[nearest][valid_r], weights=weights[valid_r])) if valid_r.any() else np.nan
        probabilities.append(probability)
        expected.append(expected_r)
    return np.asarray(probabilities), np.asarray(expected)


def _chronological_split(history: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if history is None or history.empty:
        return pd.DataFrame(), pd.DataFrame()
    ordered = history.copy()
    ordered["_date"] = pd.to_datetime(ordered.get("signal_date"), errors="coerce")
    ordered = ordered.sort_values("_date", na_position="first").drop(columns="_date")
    dates = pd.to_datetime(ordered.get("signal_date"), errors="coerce")
    unique_dates = pd.Index(dates.dropna().sort_values().unique())
    if len(unique_dates) >= 2:
        split_date_position = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * (1.0 - fraction))))
        holdout_dates = set(unique_dates[split_date_position:])
        holdout_mask = dates.isin(holdout_dates)
        return ordered.loc[~holdout_mask].copy(), ordered.loc[holdout_mask].copy()
    split = max(1, min(len(ordered) - 1, int(len(ordered) * (1.0 - fraction))))
    return ordered.iloc[:split].copy(), ordered.iloc[split:].copy()


def _calibration_evaluation_split(
    holdout: pd.DataFrame,
    config: LocalAIConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep final dates untouched for honest post-calibration metrics."""
    minimum = int(config.min_calibration_events + config.min_evaluation_events)
    if holdout is None or len(holdout) < minimum:
        return holdout.copy(), pd.DataFrame()
    calibration, evaluation = _chronological_split(holdout, 0.45)
    if len(calibration) < config.min_calibration_events or len(evaluation) < config.min_evaluation_events:
        return holdout.copy(), pd.DataFrame()
    return calibration, evaluation


def _calibration_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    baseline_probability: float | None = None,
) -> dict[str, float]:
    if len(target) == 0:
        return {
            "brier": np.nan, "log_loss": np.nan, "accuracy": np.nan,
            "baseline_brier": np.nan, "brier_skill": np.nan,
        }
    p = np.clip(probability, 1e-5, 1 - 1e-5)
    y = target.astype(float)
    baseline = float(np.clip(np.mean(y) if baseline_probability is None else baseline_probability, 1e-5, 1 - 1e-5))
    brier = float(np.mean((p - y) ** 2))
    baseline_brier = float(np.mean((baseline - y) ** 2))
    return {
        "brier": brier,
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "baseline_brier": baseline_brier,
        "brier_skill": float(1.0 - brier / baseline_brier) if baseline_brier > 1e-9 else np.nan,
    }


def _model_probability(
    history: pd.DataFrame,
    current: pd.DataFrame,
    target_col: str,
    config: LocalAIConfig,
) -> tuple[np.ndarray, dict[str, Any], _LogisticModel | None]:
    if history is None or history.empty or target_col not in history:
        return np.full(len(current), 0.5), {"state": "NO_HISTORY", "sample_size": 0}, None
    target = pd.to_numeric(history[target_col], errors="coerce")
    usable = history[target.notna()].copy()
    usable[target_col] = target[target.notna()].astype(int)
    usable = usable[usable[target_col].isin([0, 1])].copy()
    positives = int(usable[target_col].sum()) if len(usable) else 0
    if len(usable) < config.min_training_events or min(positives, len(usable) - positives) < config.min_positive_events:
        return np.full(len(current), 0.5), {
            "state": "INSUFFICIENT_CLASS_BALANCE",
            "sample_size": int(len(usable)),
            "positives": positives,
        }, None
    train, holdout = _chronological_split(usable, config.calibration_fraction)
    minimum_fit = max(12, int(config.min_training_events * max(0.55, 1.0 - config.calibration_fraction)))
    model = _fit_logistic(train, train[target_col], config, minimum_events=minimum_fit)
    if model is None:
        return np.full(len(current), 0.5), {
            "state": "MODEL_NOT_FIT", "sample_size": int(len(usable)),
            "train_size": int(len(train)), "holdout_size": int(len(holdout)),
        }, None
    calibration, evaluation = _calibration_evaluation_split(holdout, config)
    platt = _PlattModel()
    metrics = _calibration_metrics(np.asarray([]), np.asarray([]))
    calibrated = False
    if len(calibration) >= config.min_calibration_events and calibration[target_col].nunique() >= 2:
        raw_calibration = model.predict_proba(calibration)
        platt = _fit_platt(raw_calibration, calibration[target_col].to_numpy(dtype=float))
        calibrated = True
    if calibrated and len(evaluation) >= config.min_evaluation_events and evaluation[target_col].nunique() >= 2:
        evaluation_probability = platt.predict(model.predict_proba(evaluation))
        baseline_probability = float(train[target_col].mean())
        metrics = _calibration_metrics(
            evaluation_probability,
            evaluation[target_col].to_numpy(dtype=float),
            baseline_probability=baseline_probability,
        )
    brier_skill = _safe_number(metrics.get("brier_skill"), np.nan)
    if np.isfinite(brier_skill):
        state = "VALIDATED_LOCAL_LOGISTIC" if brier_skill > config.min_brier_skill else "MODEL_NO_OOS_SKILL"
    elif calibrated:
        state = "CALIBRATION_PENDING_EVALUATION"
    else:
        state = "LOCAL_LOGISTIC_UNCALIBRATED"
    raw = model.predict_proba(current)
    predicted = platt.predict(raw)
    audit = {
        "state": state,
        "sample_size": int(len(usable)),
        "positives": positives,
        "train_size": int(len(train)),
        "calibration_size": int(len(calibration)),
        "evaluation_size": int(len(evaluation)),
        "effective_sample_size": round(float((
            _recency_weights(usable, config.recency_half_life_days)
            * _outcome_quality_weights(usable)
        ).sum()), 1),
        "history_feature_coverage": round(float(np.nanmean(_feature_coverage(usable))), 3),
        **metrics,
    }
    return predicted, audit, model


def _feature_contributions(model: _LogisticModel | None, row: pd.DataFrame, limit: int = 4) -> str:
    if model is None or row.empty:
        return ""
    x_raw = _matrix(row, model.feature_names)
    x = model.scaler.transform(np.where(np.isfinite(x_raw), x_raw, model.scaler.median))[0]
    values = x * model.coef
    order = np.argsort(np.abs(values))[::-1]
    parts: list[str] = []
    for index in order[:limit]:
        if abs(values[index]) < 0.02:
            continue
        name = model.feature_names[index].replace("strategy::", "strategy ").replace("regime::", "regime ").replace("decision::", "state ")
        parts.append(f"{name} {'+' if values[index] > 0 else '-'}")
    return " • ".join(parts)


def _drift_score(model: _LogisticModel | None, current: pd.DataFrame) -> np.ndarray:
    if model is None or current.empty:
        return np.full(len(current), np.nan)
    x_raw = _matrix(current, model.feature_names)
    z = model.scaler.transform(np.where(np.isfinite(x_raw), x_raw, model.scaler.median))
    numeric_count = len(NUMERIC_FEATURES)
    return np.nanmean(np.abs(z[:, :numeric_count]), axis=1)


def _confidence_level(
    sample: int,
    brier: float,
    brier_skill: float,
    drift: float,
    disagreement: float,
    feature_coverage: float,
) -> tuple[str, float]:
    if sample <= 0:
        return "NONE", 0.0
    sample_score = min(1.0, math.log1p(max(0, sample)) / math.log1p(250))
    calibration_score = 0.0 if not np.isfinite(brier) else max(0.0, min(1.0, 1.0 - brier / 0.30))
    performance_score = 0.0 if not np.isfinite(brier_skill) else max(0.0, min(1.0, brier_skill / 0.25))
    drift_score = 0.65 if not np.isfinite(drift) else max(0.0, min(1.0, 1.0 - max(0.0, drift - 0.8) / 3.0))
    agreement_score = max(0.0, min(1.0, 1.0 - disagreement / 0.35))
    coverage_score = max(0.0, min(1.0, feature_coverage))
    score = (
        0.30 * sample_score
        + 0.20 * calibration_score
        + 0.20 * performance_score
        + 0.12 * drift_score
        + 0.10 * agreement_score
        + 0.08 * coverage_score
    )
    level = "HIGH" if score >= 0.75 else "MEDIUM" if score >= 0.52 else "LOW"
    return level, float(score)


def _expected_r_score(expected_r: float) -> float:
    if not np.isfinite(expected_r):
        return 50.0
    return float(np.clip(100.0 * (expected_r + 0.45) / 2.25, 0.0, 100.0))


def _binary_outcome(values: pd.Series) -> pd.Series:
    """Parse CSV booleans without converting unresolved/unknown rows to loss."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    text = values.fillna("").astype(str).str.strip().str.upper()
    mapped = text.map({
        "TRUE": 1.0, "YES": 1.0, "Y": 1.0, "WIN": 1.0,
        "FALSE": 0.0, "NO": 0.0, "N": 0.0, "LOSS": 0.0,
    })
    return numeric.where(numeric.notna(), mapped)


def _rule_to_ai_training_frame(events: pd.DataFrame, source: str = "UNKNOWN") -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    out = events.copy()

    def series(name: str, default: Any=np.nan) -> pd.Series:
        return out[name] if name in out else pd.Series(default, index=out.index)

    strategy_source = series("strategy", np.nan)
    if strategy_source.isna().all():
        strategy_source = series("setup", "UNKNOWN")
    out["strategy"] = strategy_source.map(_normalize_strategy)
    if "decision_state" not in out:
        out["decision_state"] = "ORDER_READY"
    quality = pd.to_numeric(series("quality_score", 50.0), errors="coerce").fillna(50.0)
    out["structure_score"] = pd.to_numeric(series("structural_quality_score"), errors="coerce").fillna(quality)
    out["timing_score"] = pd.to_numeric(series("timing_score"), errors="coerce").fillna(88.0)
    silent = pd.to_numeric(series("silent_accumulation_score"), errors="coerce").fillna(50.0)
    cmf_v = pd.to_numeric(series("cmf20"), errors="coerce").fillna(0.0)
    vol = pd.to_numeric(series("volume_ratio"), errors="coerce").fillna(1.0)
    flow_fallback = (0.55 * silent + 25 * np.clip(cmf_v + 0.1, 0, 0.4) + 15 * np.clip(vol / 2.0, 0, 1)).clip(0, 100)
    out["flow_score"] = pd.to_numeric(series("flow_score"), errors="coerce").fillna(flow_fallback)
    adtv = pd.to_numeric(series("adtv20_idr"), errors="coerce").fillna(0.0)
    liquidity_fallback = pd.Series(np.clip(22.0 * np.log10(np.maximum(1.0, adtv) / 1e7), 0, 100), index=out.index)
    out["liquidity_score"] = pd.to_numeric(series("liquidity_score"), errors="coerce").fillna(liquidity_fallback)
    rr1_source = series("rr1_plan") if "rr1_plan" in out else series("rr1")
    rr2_source = series("rr2_plan") if "rr2_plan" in out else series("rr2")
    rr1 = pd.to_numeric(rr1_source, errors="coerce")
    rr2 = pd.to_numeric(rr2_source, errors="coerce")
    out["rr1"] = rr1
    out["rr2"] = rr2
    target_fallback = pd.Series(np.clip(35 + 18 * rr1.fillna(0) + 8 * rr2.fillna(0), 0, 100), index=out.index)
    out["target_quality_score"] = pd.to_numeric(series("target_quality_score"), errors="coerce").fillna(target_fallback)
    out["data_quality_score"] = pd.to_numeric(series("data_quality_score"), errors="coerce").fillna(82.0)
    out["validation_score"] = pd.to_numeric(series("validation_score"), errors="coerce").fillna(65.0)
    for name in ("stop_pct", "atr_pct", "volume_ratio", "rsi14", "adx14", "cmf20", "roc60", "distance_52w_high", "relative_strength60", "silent_accumulation_score", "body_atr", "close_location"):
        if name not in out:
            out[name] = np.nan
    out["fill_target"] = _binary_outcome(series("filled", np.nan))
    out["success_target"] = _binary_outcome(series("tp1_hit", np.nan))
    out.loc[~out["fill_target"].isin([0, 1]), "fill_target"] = np.nan
    out.loc[~out["success_target"].isin([0, 1]), "success_target"] = np.nan
    out["outcome_source"] = source
    if "outcome_quality" not in out:
        out["outcome_quality"] = "CHRONOLOGICAL_BACKTEST" if source == "WALK_FORWARD" else "DAILY_APPROX"
    out["signal_date"] = pd.to_datetime(series("signal_date", pd.NaT), errors="coerce")
    return out


def _event_identity(frame: pd.DataFrame) -> pd.Series:
    """Create a stable de-duplication key without treating duplicated evidence as sample growth."""
    existing = frame.get("signal_id", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    ticker = frame.get("ticker", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    strategy = frame.get("strategy", frame.get("setup", pd.Series("UNKNOWN", index=frame.index))).map(_normalize_strategy)
    date_source = frame.get("signal_date", pd.Series(pd.NaT, index=frame.index))
    date = pd.to_datetime(date_source, errors="coerce").dt.strftime("%Y-%m-%d").fillna("UNKNOWN_DATE")
    planned_entry = pd.to_numeric(frame.get("planned_entry", pd.Series(np.nan, index=frame.index)), errors="coerce")
    observed_entry = pd.to_numeric(frame.get("entry", pd.Series(np.nan, index=frame.index)), errors="coerce")
    entry_numeric = planned_entry.where(planned_entry.notna(), observed_entry)
    stop_loss = pd.to_numeric(frame.get("stop_loss", pd.Series(np.nan, index=frame.index)), errors="coerce")
    stop_plan = pd.to_numeric(frame.get("stop", pd.Series(np.nan, index=frame.index)), errors="coerce")
    stop_numeric = stop_loss.where(stop_loss.notna(), stop_plan)
    entry = entry_numeric.round(4).astype(str)
    stop = stop_numeric.round(4).astype(str)
    tp1 = pd.to_numeric(frame.get("tp1", pd.Series(np.nan, index=frame.index)), errors="coerce").round(4).astype(str)
    derived = ticker + "|" + strategy + "|" + date + "|" + entry + "|" + stop + "|" + tp1
    derived_is_usable = ticker.ne("") & date.ne("UNKNOWN_DATE") & (entry_numeric.notna() | stop_numeric.notna())
    return derived.where(derived_is_usable, existing.where(existing.ne(""), derived))


def _prepare_training_history(
    validation_events: pd.DataFrame | None,
    memory_events: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    parts: list[pd.DataFrame] = []
    validation = _rule_to_ai_training_frame(
        validation_events if validation_events is not None else pd.DataFrame(),
        source="WALK_FORWARD",
    )
    if not validation.empty:
        validation["_source_priority"] = 0
        parts.append(validation)
    memory = _rule_to_ai_training_frame(
        memory_events if memory_events is not None else pd.DataFrame(),
        source="OUTCOME_MEMORY",
    )
    if not memory.empty:
        memory["_source_priority"] = 1
        parts.append(memory)
    if not parts:
        return pd.DataFrame(), {"raw_events": 0, "deduplicated_events": 0, "duplicates_removed": 0}
    history = pd.concat(parts, ignore_index=True, sort=False)
    raw_count = len(history)
    history["_event_identity"] = _event_identity(history)
    history = history.sort_values(["_source_priority", "signal_date"], na_position="first")
    history = history.drop_duplicates("_event_identity", keep="last").reset_index(drop=True)
    return history, {
        "raw_events": int(raw_count),
        "deduplicated_events": int(len(history)),
        "duplicates_removed": int(raw_count - len(history)),
    }


def enrich_profit_ranking_with_ai(
    ranking: pd.DataFrame,
    validation_events: pd.DataFrame | None = None,
    memory_events: pd.DataFrame | None = None,
    config: LocalAIConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach empirically gated local-AI probabilities and optional re-ranking.

    No resolved evidence means no AI influence. Unvalidated probability models
    and Bayesian priors remain visible as diagnostics but receive zero ranking
    weight. Both fill and TP1|fill models must beat their chronological naive
    baselines before AI can alter conviction.
    """
    cfg = config or LocalAIConfig()
    if ranking is None or ranking.empty:
        return pd.DataFrame() if ranking is None else ranking.copy(), pd.DataFrame()
    result = ranking.copy().reset_index(drop=True)
    strategy_source = result.get("strategy", result.get("setup", pd.Series("UNKNOWN", index=result.index)))
    result["strategy"] = strategy_source.map(_normalize_strategy)
    score_source = result.get("profit_conviction_score", pd.Series(50.0, index=result.index))
    rule_score = pd.to_numeric(score_source, errors="coerce").fillna(50.0)
    history, history_stats = _prepare_training_history(validation_events, memory_events)
    candidate_coverage = _feature_coverage(result)

    if (not cfg.enabled) or cfg.mode.upper() == "RULE_ONLY":
        for column in (
            "ai_fill_probability_pct", "ai_tp1_probability_pct",
            "ai_trade_success_probability_pct", "ai_probability_lower_pct",
            "ai_probability_upper_pct", "ai_trade_probability_lower_pct",
            "ai_trade_probability_upper_pct", "ai_tp1_probability_lower_pct",
            "ai_tp1_probability_upper_pct", "ai_expected_r", "ai_drift_score",
            "ai_brier_skill", "ai_fill_brier_skill", "ai_tp1_brier_skill",
        ):
            result[column] = np.nan
        result["ai_model_state"] = "DISABLED_RULE_ONLY"
        result["ai_confidence"] = "NONE"
        result["ai_confidence_score"] = 0.0
        result["ai_effective_weight_pct"] = 0.0
        result["ai_feature_coverage_pct"] = np.round(100 * candidate_coverage, 1)
        result["ai_strategy_sample_size"] = 0
        result["ai_can_influence_ranking"] = False
        result["ai_gate_reasons"] = "RULE_ONLY"
        result["hybrid_conviction_score"] = rule_score.round(1)
        result["ai_adjusted_conviction"] = rule_score.round(1)
        result["ai_explanation"] = "AI lokal dinonaktifkan; ranking murni rule engine."
        result["ai_feature_contributions"] = ""
        result["ai_version"] = AI_VERSION
        result["profit_rank"] = np.arange(1, len(result) + 1)
        return result, pd.DataFrame([
            {"metric": "AI state", "value": "RULE_ONLY"},
            {"metric": "AI version", "value": AI_VERSION},
        ])

    fill_prob, fill_audit, fill_model = _model_probability(history, result, "fill_target", cfg)
    fill_target = pd.to_numeric(history.get("fill_target", pd.Series(dtype=float)), errors="coerce")
    filled_history = history.loc[fill_target.eq(1)].copy() if not history.empty else pd.DataFrame()
    success_prob, success_audit, success_model = _model_probability(filled_history, result, "success_target", cfg)
    success_target = filled_history.get("success_target", pd.Series(dtype=float))
    knn_success, knn_r = _knn_prediction(filled_history, success_target, result, cfg)

    fill_sample = int(fill_audit.get("sample_size", 0))
    success_sample = int(success_audit.get("sample_size", 0))
    fill_brier = _safe_number(fill_audit.get("brier"), np.nan)
    success_brier = _safe_number(success_audit.get("brier"), np.nan)
    fill_brier_skill = _safe_number(fill_audit.get("brier_skill"), np.nan)
    success_brier_skill = _safe_number(success_audit.get("brier_skill"), np.nan)
    fill_audit_state = _safe_text(fill_audit.get("state")) or "NO_HISTORY"
    success_audit_state = _safe_text(success_audit.get("state")) or "NO_HISTORY"
    component_states = (fill_audit_state, success_audit_state)
    component_no_skill = "MODEL_NO_OOS_SKILL" in component_states
    joint_validated = all(state == "VALIDATED_LOCAL_LOGISTIC" for state in component_states)
    finite_briers = [value for value in (fill_brier, success_brier) if np.isfinite(value)]
    finite_skills = [value for value in (fill_brier_skill, success_brier_skill) if np.isfinite(value)]
    joint_brier = max(finite_briers) if len(finite_briers) == 2 else np.nan
    joint_brier_skill = min(finite_skills) if len(finite_skills) == 2 else np.nan
    drift = _drift_score(success_model, result)

    final_fill: list[float] = []
    final_success: list[float] = []
    expected_r: list[float] = []
    confidence_levels: list[str] = []
    confidence_scores: list[float] = []
    model_states: list[str] = []
    explanations: list[str] = []
    contributions: list[str] = []
    trade_lower_bounds: list[float] = []
    trade_upper_bounds: list[float] = []
    success_lower_bounds: list[float] = []
    success_upper_bounds: list[float] = []
    effective_weights: list[float] = []
    strategy_samples: list[int] = []
    gate_reasons_all: list[str] = []
    can_influence: list[bool] = []

    normalized_history_strategy = (
        filled_history.get("strategy", pd.Series("UNKNOWN", index=filled_history.index)).map(_normalize_strategy)
        if not filled_history.empty else pd.Series(dtype=str)
    )
    for idx, row in result.iterrows():
        strategy = _normalize_strategy(row.get("strategy"))
        regime = _normalize_regime(row.get("market_regime"))
        prior_fill, prior_fill_std, fill_prior_n = _strategy_regime_prior(
            history, strategy, regime, "fill_target", cfg,
        )
        prior_success, prior_success_std, success_prior_n = _strategy_regime_prior(
            filled_history, strategy, regime, "success_target", cfg,
        )
        fill_model_allowed = fill_model is not None and fill_audit_state != "MODEL_NO_OOS_SKILL"
        success_model_allowed = success_model is not None and success_audit_state != "MODEL_NO_OOS_SKILL"
        logistic_fill = float(fill_prob[idx]) if fill_model_allowed and idx < len(fill_prob) and np.isfinite(fill_prob[idx]) else 0.5
        logistic_success = float(success_prob[idx]) if success_model_allowed and idx < len(success_prob) and np.isfinite(success_prob[idx]) else 0.5
        similarity_available = len(filled_history) >= 12 and idx < len(knn_success) and np.isfinite(knn_success[idx])
        similarity_success = float(knn_success[idx]) if similarity_available else 0.5
        # A probability that can alter ranking must match what was evaluated.
        # Calibrated logistic components are used directly only after each
        # component passes its own chronological OOS test. Bayesian priors are
        # the conservative shadow estimate while validation is pending; KNN is
        # retained only as a similarity/expected-R diagnostic.
        p_fill = logistic_fill if fill_audit_state == "VALIDATED_LOCAL_LOGISTIC" else prior_fill
        p_success = logistic_success if success_audit_state == "VALIDATED_LOCAL_LOGISTIC" else prior_success
        comparison = [prior_success]
        if success_model_allowed:
            comparison.append(logistic_success)
        if similarity_available:
            comparison.append(similarity_success)
        disagreement = float(np.std(comparison)) if len(comparison) > 1 else float(prior_success_std)
        drift_value = float(drift[idx]) if idx < len(drift) and np.isfinite(drift[idx]) else np.nan
        coverage_value = float(candidate_coverage[idx]) if idx < len(candidate_coverage) else 0.0
        confidence, confidence_score = _confidence_level(
            min(fill_sample, success_sample), joint_brier, joint_brier_skill,
            drift_value, disagreement, coverage_value,
        )
        strategy_sample = int(
            pd.to_numeric(
                filled_history.loc[normalized_history_strategy.eq(strategy), "success_target"],
                errors="coerce",
            ).notna().sum()
        ) if not filled_history.empty and "success_target" in filled_history else 0

        fill_effective_n = max(1.0, fill_prior_n + 0.35 * fill_sample)
        success_effective_n = max(1.0, success_prior_n + 0.35 * success_sample)
        fill_se = math.sqrt(max(1e-8, p_fill * (1.0 - p_fill) / (fill_effective_n + 12.0)))
        success_se = math.sqrt(max(1e-8, p_success * (1.0 - p_success) / (success_effective_n + 12.0)))
        fill_disagreement = abs(logistic_fill - prior_fill) if fill_model_allowed else prior_fill_std
        fill_lower = float(np.clip(p_fill - 1.64 * fill_se - 0.10 * fill_disagreement, 0.0, 1.0))
        fill_upper = float(np.clip(p_fill + 1.64 * fill_se + 0.10 * fill_disagreement, 0.0, 1.0))
        success_lower = float(np.clip(p_success - 1.64 * success_se - 0.10 * disagreement, 0.0, 1.0))
        success_upper = float(np.clip(p_success + 1.64 * success_se + 0.10 * disagreement, 0.0, 1.0))
        trade_lower = fill_lower * success_lower
        trade_upper = fill_upper * success_upper

        similar_r = float(knn_r[idx]) if idx < len(knn_r) and np.isfinite(knn_r[idx]) else np.nan
        strategy_r = (
            pd.to_numeric(
                filled_history.loc[normalized_history_strategy.eq(strategy), "r_multiple"],
                errors="coerce",
            ).dropna()
            if not filled_history.empty and "r_multiple" in filled_history else pd.Series(dtype=float)
        )
        base_r = float(strategy_r.median()) if len(strategy_r) else np.nan
        e_r = similar_r if np.isfinite(similar_r) else base_r
        trade_probability = p_fill * p_success
        # RR/target quality already lives in the authoritative rule score.
        # Expected R remains diagnostic until it has a separate OOS gate.
        ai_edge = 100.0 * trade_probability

        gate_reasons: list[str] = []
        if success_sample <= 0:
            gate_reasons.append("NO_RESOLVED_HISTORY")
        if strategy_sample < cfg.min_strategy_events:
            gate_reasons.append("STRATEGY_SAMPLE_LOW")
        if coverage_value < cfg.min_feature_coverage:
            gate_reasons.append("FEATURE_COVERAGE_LOW")
        if fill_audit_state == "MODEL_NO_OOS_SKILL":
            gate_reasons.append("FILL_MODEL_NO_OOS_SKILL")
        if success_audit_state == "MODEL_NO_OOS_SKILL":
            gate_reasons.append("TP1_MODEL_NO_OOS_SKILL")
        if not component_no_skill and not joint_validated:
            gate_reasons.append("MODEL_VALIDATION_PENDING")
        if np.isfinite(drift_value) and drift_value > 2.8:
            gate_reasons.append("HIGH_FEATURE_DRIFT")

        global_support = min(1.0, min(fill_sample, success_sample) / max(60.0, 2.0 * cfg.min_training_events))
        strategy_support = min(1.0, strategy_sample / max(1.0, float(cfg.min_strategy_events)))
        coverage_support = float(np.clip(
            (coverage_value - cfg.min_feature_coverage) / max(1e-9, 1.0 - cfg.min_feature_coverage),
            0.0,
            1.0,
        ))
        drift_multiplier = 0.0 if np.isfinite(drift_value) and drift_value > 3.8 else 0.55 if np.isfinite(drift_value) and drift_value > 2.8 else 0.75 if np.isfinite(drift_value) and drift_value > 2.0 else 1.0
        if joint_validated and np.isfinite(joint_brier_skill) and joint_brier_skill > cfg.min_brier_skill:
            maximum_weight = min(float(cfg.max_weight), 0.35)
            performance_support = float(np.clip(joint_brier_skill / 0.25, 0.0, 1.0))
        elif component_no_skill or success_sample <= 0:
            maximum_weight = 0.0
            performance_support = 0.0
        else:
            # Validation pending is not evidence of skill.  Keep Bayesian and
            # logistic estimates visible in shadow form, but fail closed until
            # both components pass their untouched chronological evaluation.
            maximum_weight = 0.0
            performance_support = 0.0
        effective_weight = (
            maximum_weight
            * confidence_score
            * global_support
            * strategy_support
            * coverage_support
            * drift_multiplier
            * performance_support
        )
        if cfg.mode.upper() == "SHADOW_ONLY":
            effective_weight = 0.0
            gate_reasons.append("SHADOW_ONLY")
        influence = bool(effective_weight >= 0.001)
        hybrid = (
            (1.0 - effective_weight) * float(rule_score.iloc[idx]) + effective_weight * ai_edge
            if influence else float(rule_score.iloc[idx])
        )

        if success_sample <= 0:
            state = "NO_EMPIRICAL_HISTORY"
        elif component_no_skill:
            state = "MODEL_REJECTED_NO_OOS_SKILL"
        elif joint_validated and influence:
            state = "VALIDATED_HYBRID"
        elif influence:
            state = "LIMITED_BAYESIAN_INFLUENCE"
        else:
            state = "EVIDENCE_SHADOW_ONLY"

        probability_available = fill_sample > 0 and success_sample > 0
        final_fill.append(p_fill if probability_available else np.nan)
        final_success.append(p_success if probability_available else np.nan)
        expected_r.append(e_r if probability_available else np.nan)
        confidence_levels.append(confidence)
        confidence_scores.append(confidence_score)
        trade_lower_bounds.append(trade_lower if probability_available else np.nan)
        trade_upper_bounds.append(trade_upper if probability_available else np.nan)
        success_lower_bounds.append(success_lower if probability_available else np.nan)
        success_upper_bounds.append(success_upper if probability_available else np.nan)
        effective_weights.append(effective_weight)
        strategy_samples.append(strategy_sample)
        model_states.append(state)
        gate_text = " | ".join(dict.fromkeys(gate_reasons))
        gate_reasons_all.append(gate_text or "ALL_AI_GATES_PASSED")
        can_influence.append(influence)
        contributions.append(_feature_contributions(success_model, result.iloc[[idx]]))

        if probability_available:
            explanation = f"P(fill) {p_fill*100:.0f}%; P(TP1|fill) {p_success*100:.0f}%"
            explanation += f"; expected R {e_r:.2f}" if np.isfinite(e_r) else "; expected R belum stabil"
        else:
            explanation = "Probabilitas AI belum diterbitkan karena belum ada outcome teresolusi yang cukup"
        risk_flags: list[str] = []
        if confidence in {"NONE", "LOW"}:
            risk_flags.append("sample AI belum kuat")
        if np.isfinite(drift_value) and drift_value > 2.0:
            risk_flags.append("fitur di luar distribusi latihan")
        if probability_available and trade_lower < 0.30:
            risk_flags.append("lower-bound peluang trade lemah")
        if gate_text:
            risk_flags.append(f"gate: {gate_text}")
        if risk_flags:
            explanation += "; " + "; ".join(risk_flags)
        explanations.append(explanation)
        result.loc[idx, "hybrid_conviction_score"] = round(float(np.clip(hybrid, 0.0, 100.0)), 1)

    fill_array = np.asarray(final_fill, dtype=float)
    success_array = np.asarray(final_success, dtype=float)
    result["ai_fill_probability_pct"] = np.round(100.0 * fill_array, 1)
    result["ai_tp1_probability_pct"] = np.round(100.0 * success_array, 1)
    result["ai_trade_success_probability_pct"] = np.round(100.0 * fill_array * success_array, 1)
    result["ai_probability_lower_pct"] = np.round(100.0 * np.asarray(trade_lower_bounds), 1)
    result["ai_probability_upper_pct"] = np.round(100.0 * np.asarray(trade_upper_bounds), 1)
    result["ai_trade_probability_lower_pct"] = result["ai_probability_lower_pct"]
    result["ai_trade_probability_upper_pct"] = result["ai_probability_upper_pct"]
    result["ai_tp1_probability_lower_pct"] = np.round(100.0 * np.asarray(success_lower_bounds), 1)
    result["ai_tp1_probability_upper_pct"] = np.round(100.0 * np.asarray(success_upper_bounds), 1)
    result["ai_expected_r"] = np.round(np.asarray(expected_r, dtype=float), 3)
    result["ai_confidence"] = confidence_levels
    result["ai_confidence_score"] = np.round(100.0 * np.asarray(confidence_scores), 1)
    result["ai_model_state"] = model_states
    result["ai_effective_weight_pct"] = np.round(100.0 * np.asarray(effective_weights), 2)
    result["ai_feature_coverage_pct"] = np.round(100.0 * candidate_coverage, 1)
    result["ai_strategy_sample_size"] = strategy_samples
    result["ai_fill_brier_skill"] = fill_brier_skill
    result["ai_tp1_brier_skill"] = success_brier_skill
    result["ai_brier_skill"] = joint_brier_skill
    result["ai_can_influence_ranking"] = can_influence
    result["ai_gate_reasons"] = gate_reasons_all
    result["ai_drift_score"] = np.round(drift, 3)
    result["ai_explanation"] = explanations
    result["ai_feature_contributions"] = contributions
    result["ai_version"] = AI_VERSION
    result["ai_adjusted_conviction"] = result["hybrid_conviction_score"]
    sort_column = "hybrid_conviction_score" if cfg.mode.upper() == "HYBRID_GUARDED" else "profit_conviction_score"
    readiness = {"ORDER_READY": 0, "SETUP_READY": 1, "ENTRY_PLAN": 2}
    decision_source = result.get("decision_state", pd.Series("SETUP_READY", index=result.index))
    result["_ai_readiness"] = decision_source.map(readiness).fillna(9)
    result = result.sort_values(
        [sort_column, "_ai_readiness", "target_quality_score", "liquidity_score"],
        ascending=[False, True, False, False],
    ).drop(columns="_ai_readiness").reset_index(drop=True)
    result["profit_rank"] = np.arange(1, len(result) + 1)

    audit = pd.DataFrame([
        {"metric": "AI version", "value": AI_VERSION},
        {"metric": "AI mode", "value": cfg.mode},
        {"metric": "Raw outcome rows", "value": history_stats["raw_events"]},
        {"metric": "Deduplicated outcome rows", "value": history_stats["deduplicated_events"]},
        {"metric": "Duplicate rows removed", "value": history_stats["duplicates_removed"]},
        {"metric": "Filled training events", "value": int(len(filled_history))},
        {"metric": "Fill model", "value": fill_audit.get("state")},
        {"metric": "Success model", "value": success_audit.get("state")},
        {"metric": "Success train/calibration/evaluation", "value": f"{success_audit.get('train_size', 0)}/{success_audit.get('calibration_size', 0)}/{success_audit.get('evaluation_size', 0)}"},
        {"metric": "Fill Brier / baseline", "value": f"{fill_audit.get('brier')} / {fill_audit.get('baseline_brier')}"},
        {"metric": "TP1 Brier / baseline", "value": f"{success_audit.get('brier')} / {success_audit.get('baseline_brier')}"},
        {"metric": "Fill OOS Brier skill", "value": fill_audit.get("brier_skill")},
        {"metric": "TP1 OOS Brier skill", "value": success_audit.get("brier_skill")},
        {"metric": "OOS Brier skill", "value": joint_brier_skill},
        {"metric": "Joint validation state", "value": "VALIDATED" if joint_validated else "REJECTED_NO_SKILL" if component_no_skill else "PENDING"},
        {"metric": "Candidates AI may influence", "value": int(sum(can_influence))},
        {"metric": "Maximum effective AI weight", "value": round(100.0 * max(effective_weights, default=0.0), 2)},
        {"metric": "Configured maximum AI weight", "value": round(100.0 * min(cfg.max_weight, 0.35), 1)},
        {"metric": "Storage", "value": "Local cache + manual CSV export/import"},
    ])
    return result, audit


def _memory_root() -> Path:
    base = os.getenv("IDX_SCANNER_CACHE_DIR", "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".cache" / "idx_super_scanner"
    path = root / "local_ai"
    path.mkdir(parents=True, exist_ok=True)
    return path


def memory_path() -> Path:
    return _memory_root() / "outcome_memory.csv"


def parse_memory_csv(source: bytes | io.BytesIO | pd.DataFrame | None) -> pd.DataFrame:
    if source is None:
        return pd.DataFrame()
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        payload = source.getvalue() if hasattr(source, "getvalue") else source
        frame = pd.read_csv(io.BytesIO(payload) if isinstance(payload, (bytes, bytearray)) else payload)
    if frame.empty:
        return frame
    required = {"signal_id", "ticker", "strategy", "signal_date", "entry", "stop_loss", "tp1"}
    if not required.issubset(frame.columns):
        missing = sorted(required.difference(frame.columns))
        raise ValueError(f"AI memory kehilangan kolom: {', '.join(missing)}")
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    return frame


def load_memory(uploaded: pd.DataFrame | bytes | io.BytesIO | None = None) -> pd.DataFrame:
    disk = pd.DataFrame()
    path = memory_path()
    if path.exists():
        try:
            disk = pd.read_csv(path)
            disk["signal_date"] = pd.to_datetime(disk.get("signal_date"), errors="coerce")
        except Exception:
            disk = pd.DataFrame()
    imported = parse_memory_csv(uploaded) if uploaded is not None else pd.DataFrame()
    if disk.empty:
        return imported
    if imported.empty:
        return disk
    merged = pd.concat([disk, imported], ignore_index=True, sort=False)
    return merged.drop_duplicates("signal_id", keep="last").reset_index(drop=True)


def save_memory(frame: pd.DataFrame, max_rows: int = 10000) -> None:
    if frame is None:
        return
    output = frame.copy().tail(int(max_rows))
    try:
        output.to_csv(memory_path(), index=False)
    except Exception:
        pass


def _signal_identifier(ticker: str, strategy: str, signal_date: Any, entry: Any, stop: Any, tp1: Any) -> str:
    text = "|".join([
        _safe_text(ticker).upper(), _normalize_strategy(strategy), str(pd.Timestamp(signal_date).date()),
        f"{_safe_number(entry, 0):.4f}", f"{_safe_number(stop, 0):.4f}", f"{_safe_number(tp1, 0):.4f}",
    ])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _register_current(memory: pd.DataFrame, ranking: pd.DataFrame, prepared: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    existing = memory.copy() if memory is not None else pd.DataFrame()
    rows = []
    for _, row in ranking.iterrows():
        ticker = _safe_text(row.get("ticker")).upper()
        frame = prepared.get(ticker)
        if frame is None or frame.empty:
            continue
        signal_date = pd.Timestamp(frame.index[-1]).tz_localize(None) if getattr(pd.Timestamp(frame.index[-1]), "tzinfo", None) else pd.Timestamp(frame.index[-1])
        signal_id = _signal_identifier(ticker, row.get("strategy"), signal_date, row.get("entry"), row.get("stop_loss"), row.get("tp1"))
        record = row.to_dict()
        record.update({
            "signal_id": signal_id,
            "signal_date": signal_date,
            "memory_state": "OPEN",
            "filled": np.nan,
            "tp1_hit": np.nan,
            "r_multiple": np.nan,
            "result": "OPEN",
            "outcome_quality": "DAILY_APPROX",
        })
        rows.append(record)
    additions = pd.DataFrame(rows)
    if existing.empty:
        return additions
    if additions.empty:
        return existing
    merged = pd.concat([existing, additions], ignore_index=True, sort=False)
    return merged.drop_duplicates("signal_id", keep="first").reset_index(drop=True)


def _resolve_one(row: pd.Series, frame: pd.DataFrame, cfg: LocalAIConfig) -> dict[str, Any]:
    result = row.to_dict()
    if _safe_text(row.get("memory_state")).upper() not in {"", "OPEN"}:
        return result
    signal_date = pd.to_datetime(row.get("signal_date"), errors="coerce")
    if pd.isna(signal_date) or frame is None or frame.empty:
        return result
    signal_date = pd.Timestamp(signal_date)
    if signal_date.tzinfo is not None:
        signal_date = signal_date.tz_convert("Asia/Jakarta").tz_localize(None)
    local = frame.copy()
    local.index = pd.to_datetime(local.index, errors="coerce")
    if getattr(local.index, "tz", None) is not None:
        local.index = local.index.tz_convert("Asia/Jakarta").tz_localize(None)
    local = local[local.index > signal_date].copy()
    if local.empty:
        return result
    entry = _safe_number(row.get("entry"), np.nan)
    stop = _safe_number(row.get("stop_loss"), np.nan)
    tp1 = _safe_number(row.get("tp1"), np.nan)
    tp2 = _safe_number(row.get("tp2"), np.nan)
    if not all(np.isfinite(x) and x > 0 for x in (entry, stop, tp1)) or entry <= stop:
        result.update({"memory_state": "INVALID", "result": "INVALID_LEVELS"})
        return result
    entry_window = local.iloc[: int(cfg.memory_entry_window_bars)]
    fill_position = None
    fill_price = np.nan
    fill_at_open = False
    trigger = _safe_number(row.get("trigger_price"), np.nan)
    buy_stop = np.isfinite(trigger) and trigger > entry * 0.995
    for position, (_, candle) in enumerate(entry_window.iterrows()):
        open_v, high_v, low_v = (_safe_number(candle.get(name), np.nan) for name in ("Open", "High", "Low"))
        if buy_stop:
            level = trigger
            atr_value = entry * max(0.0, _safe_number(row.get("atr_pct"), 0.0))
            if np.isfinite(open_v) and atr_value > 0 and open_v > level + cfg.max_entry_gap_atr * atr_value:
                result.update({
                    "memory_state": "RESOLVED", "filled": False, "tp1_hit": False,
                    "result": "NO_FILL_GAP_CANCEL", "r_multiple": 0.0,
                    "no_fill_reason": "Gap di atas toleransi entry",
                })
                return result
            if high_v >= level:
                fill_position = position
                fill_at_open = bool(np.isfinite(open_v) and open_v >= level)
                fill_price = max(open_v, level) if np.isfinite(open_v) else level
                break
        else:
            if np.isfinite(open_v) and open_v <= stop:
                result.update({
                    "memory_state": "RESOLVED", "filled": False, "tp1_hit": False,
                    "result": "NO_FILL_INVALIDATED_GAP", "r_multiple": 0.0,
                    "no_fill_reason": "Gap di bawah stop sebelum limit fill",
                })
                return result
            if low_v <= entry <= high_v or (np.isfinite(open_v) and open_v <= entry):
                fill_position = position
                fill_at_open = bool(np.isfinite(open_v) and open_v <= entry)
                fill_price = min(entry, open_v) if np.isfinite(open_v) else entry
                break
    if fill_position is None:
        if len(local) >= int(cfg.memory_entry_window_bars):
            result.update({"memory_state": "RESOLVED", "filled": False, "tp1_hit": False, "result": "NO_FILL", "r_multiple": 0.0})
        return result
    future = local.iloc[fill_position: fill_position + int(cfg.memory_horizon_bars)]
    exit_price = _safe_number(future.iloc[-1].get("Close"), fill_price)
    exit_date = future.index[-1]
    outcome = "TIME_EXIT"
    tp1_hit: Any = False
    tp2_hit: Any = False
    outcome_ambiguous = False
    for bar_offset, (candle_date, candle) in enumerate(future.iterrows()):
        open_v = _safe_number(candle.get("Open"), np.nan)
        low_v = _safe_number(candle.get("Low"), np.nan)
        high_v = _safe_number(candle.get("High"), np.nan)
        stop_touched = bool(np.isfinite(low_v) and low_v <= stop)
        tp1_touched = bool(np.isfinite(high_v) and high_v >= tp1)
        # If an intrabar order and an exit level appear in the same daily bar,
        # OHLC cannot reveal which happened first. Preserve the confirmed fill,
        # but never fabricate a TP1 win/loss label for model training.
        if bar_offset == 0 and not fill_at_open and (stop_touched or tp1_touched):
            exit_price, exit_date, outcome = np.nan, candle_date, "AMBIGUOUS_FILL_BAR"
            tp1_hit, tp2_hit, outcome_ambiguous = np.nan, np.nan, True
            break
        if np.isfinite(open_v) and open_v <= stop:
            exit_price, exit_date, outcome = open_v, candle_date, "LOSS_GAP"
            break
        both = stop_touched and tp1_touched
        if both and cfg.conservative_same_bar_resolution:
            exit_price, exit_date, outcome = stop, candle_date, "LOSS_AMBIGUOUS_BAR"
            tp1_hit, tp2_hit, outcome_ambiguous = np.nan, np.nan, True
            break
        if stop_touched:
            exit_price, exit_date, outcome = stop, candle_date, "LOSS"
            break
        if tp1_touched:
            exit_price, exit_date, outcome, tp1_hit = tp1, candle_date, "WIN_TP1", True
            tp2_hit = bool(np.isfinite(tp2) and high_v >= tp2)
            break
    risk_pct = (fill_price - stop) / fill_price if fill_price > stop else np.nan
    net_return = exit_price / fill_price - 1.0 - max(0.0, cfg.roundtrip_cost_pct)
    r_multiple = net_return / risk_pct if np.isfinite(risk_pct) and risk_pct > 0 else np.nan
    if outcome == "TIME_EXIT" and len(future) < int(cfg.memory_horizon_bars):
        return result
    result.update({
        "memory_state": "RESOLVED",
        "filled": True,
        "fill_date": future.index[0],
        "fill_price": fill_price,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
        "outcome_ambiguous": outcome_ambiguous,
        "result": outcome,
        "r_multiple": round(float(r_multiple), 4) if np.isfinite(r_multiple) else np.nan,
        "resolved_at": pd.Timestamp(frame.index[-1]),
    })
    return result


def update_outcome_memory(
    ranking: pd.DataFrame,
    prepared: Mapping[str, pd.DataFrame],
    existing: pd.DataFrame | None = None,
    config: LocalAIConfig | None = None,
) -> pd.DataFrame:
    cfg = config or LocalAIConfig()
    memory = _register_current(existing if existing is not None else pd.DataFrame(), ranking, prepared)
    if memory.empty:
        return memory
    resolved = []
    for _, row in memory.iterrows():
        ticker = _safe_text(row.get("ticker")).upper()
        resolved.append(_resolve_one(row, prepared.get(ticker, pd.DataFrame()), cfg))
    output = pd.DataFrame(resolved).drop_duplicates("signal_id", keep="last")
    output = output.sort_values("signal_date", na_position="last").tail(int(cfg.memory_max_rows)).reset_index(drop=True)
    save_memory(output, cfg.memory_max_rows)
    return output


def resolved_memory_events(memory: pd.DataFrame | None) -> pd.DataFrame:
    if memory is None or memory.empty:
        return pd.DataFrame()
    result = memory[memory.get("memory_state", "").astype(str).str.upper().eq("RESOLVED")].copy()
    if result.empty:
        return result
    result["setup"] = result.get("strategy", "UNKNOWN")
    return result


def memory_summary(memory: pd.DataFrame | None) -> pd.DataFrame:
    if memory is None or memory.empty:
        return pd.DataFrame([{"metric": "Memory rows", "value": 0}, {"metric": "Resolved", "value": 0}])
    state = memory.get("memory_state", pd.Series("UNKNOWN", index=memory.index)).astype(str).str.upper()
    resolved = memory[state.eq("RESOLVED")]
    filled_state = _binary_outcome(
        resolved.get("filled", pd.Series(np.nan, index=resolved.index)),
    ) if not resolved.empty else pd.Series(dtype=float)
    filled = resolved[filled_state.eq(1)] if not resolved.empty else pd.DataFrame()
    tp1_state = _binary_outcome(
        filled.get("tp1_hit", pd.Series(np.nan, index=filled.index)),
    ) if not filled.empty else pd.Series(dtype=float)
    labelled = filled[tp1_state.isin([0, 1])] if not filled.empty else pd.DataFrame()
    wins = int(tp1_state.eq(1).sum()) if not filled.empty else 0
    return pd.DataFrame([
        {"metric": "Memory rows", "value": int(len(memory))},
        {"metric": "Open signals", "value": int(state.eq("OPEN").sum())},
        {"metric": "Resolved", "value": int(len(resolved))},
        {"metric": "Filled", "value": int(len(filled))},
        {"metric": "TP1-labelled outcomes", "value": int(len(labelled))},
        {"metric": "Ambiguous filled outcomes", "value": int(len(filled) - len(labelled))},
        {"metric": "TP1 wins", "value": wins},
        {"metric": "Empirical hit rate", "value": round(100 * wins / len(labelled), 1) if len(labelled) else np.nan},
        {"metric": "Persistence", "value": "Best-effort local cache; export CSV before redeploy/sleep"},
    ])


__all__ = [
    "AI_VERSION",
    "LocalAIConfig",
    "enrich_profit_ranking_with_ai",
    "parse_memory_csv",
    "load_memory",
    "save_memory",
    "update_outcome_memory",
    "resolved_memory_events",
    "memory_summary",
    "memory_path",
]


def _percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    count = int(numeric.notna().sum())
    if count < 3:
        return pd.Series(50.0, index=series.index)
    ranks = numeric.rank(method="average")
    ranks = 100.0 * (ranks - 1.0) / max(1.0, count - 1.0)
    if not higher_is_better:
        ranks = 100.0 - ranks
    return ranks.fillna(50.0)


def _grouped_percentile_score(
    series: pd.Series,
    groups: pd.Series,
    higher_is_better: bool,
) -> pd.Series:
    """Compare issuers inside a usable economic peer group."""
    global_score = _percentile_score(series, higher_is_better)
    result = global_score.copy()
    for _, index in groups.groupby(groups).groups.items():
        local_index = pd.Index(index)
        if pd.to_numeric(series.loc[local_index], errors="coerce").notna().sum() >= 3:
            result.loc[local_index] = _percentile_score(series.loc[local_index], higher_is_better)
    return result


def enrich_multibagger_with_peer_ai(
    candidates: pd.DataFrame,
    enabled: bool = True,
    max_weight: float = 0.20,
) -> pd.DataFrame:
    """Add free unsupervised peer intelligence to Multibagger candidates.

    This layer does not pretend to forecast a 10x outcome. It detects the
    strongest cross-sectional quality/growth/forward-impact combinations,
    penalises accounting/project outliers, and exposes uncertainty.
    """
    if candidates is None or candidates.empty:
        return candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    out = candidates.copy()
    if not enabled:
        out["ai_multibagger_peer_score"] = 50.0
        out["ai_multibagger_forward_score"] = 50.0
        out["ai_multibagger_absolute_score"] = 50.0
        out["ai_multibagger_outlier_risk"] = 0.0
        out["ai_multibagger_confidence"] = "NONE"
        out["ai_multibagger_effective_weight_pct"] = 0.0
        out["ai_multibagger_cluster"] = "AI_DISABLED"
        out["ai_multibagger_peer_group"] = "AI_DISABLED"
        out["ai_multibagger_peer_count"] = 0
        out["ai_multibagger_gate_reasons"] = "AI_DISABLED"
        return out

    directions = {
        "revenue_growth": True,
        "earnings_growth": True,
        "roe": True,
        "roa": True,
        "net_margin": True,
        "cash_conversion_ttm": True,
        "positive_ocf_ratio": True,
        "positive_earnings_ratio": True,
        "margin_stability": True,
        "fcf_yield": True,
        "roic_proxy": True,
        "interest_coverage": True,
        "cash_to_debt": True,
        "debt_equity": False,
        "net_debt_ebitda": False,
        "share_dilution_yoy": False,
        "project_pipeline_score": True,
        "management_quality_score": True,
        "future_fundamental_impact_score": True,
        "fundamental_consensus_score": True,
        "fundamental_coverage": True,
        "silent_accumulation_score": True,
    }
    weights = {
        "revenue_growth": 0.09,
        "earnings_growth": 0.09,
        "roe": 0.06,
        "roa": 0.04,
        "net_margin": 0.05,
        "cash_conversion_ttm": 0.08,
        "positive_ocf_ratio": 0.06,
        "positive_earnings_ratio": 0.04,
        "margin_stability": 0.04,
        "fcf_yield": 0.04,
        "roic_proxy": 0.06,
        "interest_coverage": 0.04,
        "cash_to_debt": 0.03,
        "debt_equity": 0.04,
        "net_debt_ebitda": 0.04,
        "share_dilution_yoy": 0.03,
        "project_pipeline_score": 0.05,
        "management_quality_score": 0.05,
        "future_fundamental_impact_score": 0.04,
        "fundamental_consensus_score": 0.03,
        "fundamental_coverage": 0.02,
        "silent_accumulation_score": 0.02,
    }
    model_source = out.get("fundamental_model", pd.Series("GENERAL", index=out.index)).fillna("GENERAL").astype(str).str.upper()
    base_group = pd.Series(np.where(model_source.str.contains("BANK|FINANCIAL"), "BANK", "GENERAL"), index=out.index)
    sector = out.get("sector", pd.Series("", index=out.index)).fillna("").astype(str).str.upper().str.strip()
    sector_key = sector.str.replace(r"[^A-Z0-9]+", "_", regex=True).str.strip("_")
    sector_count = sector_key.map(sector_key[sector_key.ne("")].value_counts()).fillna(0).astype(int)
    usable_sector = base_group.eq("GENERAL") & sector_key.ne("") & sector_count.ge(5)
    peer_group = base_group.where(~usable_sector, "SECTOR::" + sector_key)
    peer_count = peer_group.map(peer_group.value_counts()).astype(int)
    percentile = pd.DataFrame(index=out.index)
    availability = pd.DataFrame(index=out.index)
    for column, direction in directions.items():
        values = out[column] if column in out else pd.Series(np.nan, index=out.index)
        percentile[column] = _grouped_percentile_score(values, peer_group, direction)
        availability[column] = pd.to_numeric(values, errors="coerce").notna().astype(float)
    total_weight = sum(weights.values())
    peer_score = sum(weights[name] * percentile[name] for name in weights) / total_weight
    coverage = sum(weights[name] * availability[name] for name in weights) / total_weight

    # Robust multivariate outlier detection. Extreme combinations such as very
    # high growth with negative cash conversion/debt stress receive a penalty.
    matrix = percentile[list(weights)].to_numpy(dtype=float)
    center = np.nanmedian(matrix, axis=0)
    mad = np.nanmedian(np.abs(matrix - center), axis=0) * 1.4826
    mad = np.where(mad > 1e-6, mad, 20.0)
    z = (matrix - center) / mad
    outlier = np.sqrt(np.nanmean(np.clip(z, -8, 8) ** 2, axis=1))
    contradiction = np.zeros(len(out), dtype=float)
    growth = (percentile["revenue_growth"] + percentile["earnings_growth"]) / 2
    cash_quality = (percentile["cash_conversion_ttm"] + percentile["positive_ocf_ratio"] + percentile["fcf_yield"]) / 3
    solvency = (percentile["debt_equity"] + percentile["net_debt_ebitda"] + percentile["interest_coverage"]) / 3
    contradiction += np.maximum(0, growth.to_numpy() - cash_quality.to_numpy() - 25) / 2.5
    contradiction += np.maximum(0, growth.to_numpy() - solvency.to_numpy() - 30) / 3.0
    outlier_risk = np.clip(12 * np.maximum(0, outlier - 1.5) + contradiction, 0, 35)

    revenue_uplift = pd.to_numeric(out.get("future_revenue_uplift_base_pct", pd.Series(np.nan, index=out.index)), errors="coerce")
    ebitda_uplift = pd.to_numeric(out.get("future_ebitda_uplift_base_pct", pd.Series(np.nan, index=out.index)), errors="coerce")
    debt_change = pd.to_numeric(out.get("future_net_debt_change_pct", pd.Series(np.nan, index=out.index)), errors="coerce")
    success_probability = pd.to_numeric(out.get("project_success_probability_pct", pd.Series(np.nan, index=out.index)), errors="coerce")
    impact = pd.to_numeric(out.get("future_fundamental_impact_score", pd.Series(50.0, index=out.index)), errors="coerce").fillna(50.0)
    forward_score = (
        0.30 * impact
        + 0.22 * np.clip(50 + revenue_uplift.fillna(0) * 1.4, 0, 100)
        + 0.20 * np.clip(50 + ebitda_uplift.fillna(0) * 1.2, 0, 100)
        + 0.18 * success_probability.fillna(50)
        + 0.10 * np.clip(65 - debt_change.fillna(0) * 0.8, 0, 100)
    )
    score10 = pd.to_numeric(out.get("fundamental_score_10", pd.Series(np.nan, index=out.index)), errors="coerce")
    score100 = pd.to_numeric(out.get("fundamental_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    business_quality = (10.0 * score10).where(score10.notna(), score100).fillna(50.0).clip(0.0, 100.0)
    consensus = pd.to_numeric(out.get("fundamental_consensus_score", pd.Series(np.nan, index=out.index)), errors="coerce").fillna(50.0).clip(0.0, 100.0)
    fundamental_coverage = pd.to_numeric(out.get("fundamental_coverage", pd.Series(np.nan, index=out.index)), errors="coerce").fillna(0.0).clip(0.0, 100.0)
    official = out.get("fundamental_official_verified", pd.Series(False, index=out.index)).map(_truthy)
    grade = out.get("fundamental_data_grade", pd.Series("D", index=out.index)).fillna("D").astype(str).str.upper()
    grade_score = grade.map({"A": 100.0, "B": 75.0, "C": 45.0, "D": 15.0}).fillna(15.0)
    absolute_score = (
        0.50 * business_quality
        + 0.20 * consensus
        + 0.15 * fundamental_coverage
        + 0.10 * grade_score
        + 0.05 * official.astype(float) * 100.0
    )
    combined = np.clip(0.50 * peer_score + 0.30 * absolute_score + 0.20 * forward_score - outlier_risk, 0, 100)
    peer_support = np.where(
        peer_count.ge(5), np.clip((peer_count.astype(float) - 4.0) / 8.0, 0.0, 1.0), 0.0,
    )
    grade_support = grade.map({"A": 1.0, "B": 0.70, "C": 0.15, "D": 0.0}).fillna(0.0)
    # Peer intelligence may be displayed for every row, but it cannot alter
    # capital conviction without an official verified filing anchor.
    official_support = np.where(official, 1.0, 0.0)
    statement_current = out.get("statement_current", pd.Series(False, index=out.index)).map(_truthy)
    source_count = pd.to_numeric(
        out.get("fundamental_source_count", pd.Series(0, index=out.index)), errors="coerce",
    ).fillna(0)
    conflicts = out.get("fundamental_conflicts", pd.Series("", index=out.index)).map(_safe_text).ne("")
    severe = out.get("severe_fundamental_flags", pd.Series(False, index=out.index)).map(_truthy)
    integrity_support = np.where(statement_current & source_count.ge(2) & ~conflicts & ~severe, 1.0, 0.0)
    confidence_score = np.clip(
        coverage * (1.0 - outlier_risk / 70.0) * peer_support * grade_support
        * official_support * integrity_support,
        0,
        1,
    )
    effective_weight = np.clip(float(max_weight), 0, 0.25) * confidence_score
    clusters = np.where(
        (combined >= 78) & (cash_quality >= 65) & (solvency >= 55), "QUALITY_GROWTH_LEADER",
        np.where((combined >= 68) & (growth >= 70), "EMERGING_GROWTH", np.where(outlier_risk >= 22, "HIGH_UNCERTAINTY_OUTLIER", "PEER_AVERAGE")),
    )
    confidence = np.where(confidence_score >= 0.75, "HIGH", np.where(confidence_score >= 0.52, "MEDIUM", "LOW"))
    out["ai_multibagger_peer_score"] = np.round(combined, 1)
    out["ai_multibagger_forward_score"] = np.round(forward_score, 1)
    out["ai_multibagger_absolute_score"] = np.round(absolute_score, 1)
    out["ai_multibagger_outlier_risk"] = np.round(outlier_risk, 1)
    out["ai_multibagger_data_coverage_pct"] = np.round(100 * coverage, 1)
    out["ai_multibagger_confidence"] = confidence
    out["ai_multibagger_effective_weight_pct"] = np.round(100 * effective_weight, 1)
    out["ai_multibagger_cluster"] = clusters
    out["ai_multibagger_peer_group"] = peer_group
    out["ai_multibagger_peer_count"] = peer_count
    out["ai_multibagger_gate_reasons"] = [
        " | ".join(reason for reason in (
            "PEER_SAMPLE_LOW" if count < 5 else "",
            "OFFICIAL_XBRL_NOT_VERIFIED" if not verified else "",
            "FUNDAMENTAL_GRADE_LOW" if data_grade not in {"A", "B"} else "",
            "STATEMENT_NOT_CURRENT" if not current else "",
            "FUNDAMENTAL_SOURCE_QUORUM_LOW" if sources < 2 else "",
            "FUNDAMENTAL_CONFLICT_OR_SEVERE" if conflict or severe_flag else "",
            "OUTLIER_RISK_HIGH" if risk >= 22 else "",
        ) if reason) or "PEER_AI_GATES_PASSED"
        for count, verified, data_grade, current, sources, conflict, severe_flag, risk in zip(
            peer_count, official, grade, statement_current, source_count, conflicts, severe, outlier_risk,
        )
    ]
    out["ai_multibagger_reason"] = [
        f"peer+absolute {p:.1f}; absolute {a:.1f}; forward {f:.1f}; outlier risk {o:.1f}; coverage {c*100:.0f}%"
        for p, a, f, o, c in zip(combined, absolute_score, forward_score, outlier_risk, coverage)
    ]
    return out


__all__.append("enrich_multibagger_with_peer_ai")

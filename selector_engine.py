"""Point-in-time cross-sectional stock selection for IDX Super Scanner.

The selector answers a different question from the setup detectors:

1. Which stocks have the strongest relative opportunity versus IHSG?
2. Only after selection, is there a valid setup and executable trade plan?

The module evaluates four frozen challengers (rule, independent selector,
relative-strength baseline, and a regularised AI challenger) on chronological
unseen dates.  SciPy is the primary statistical-validation backend on Python
3.12, with a deterministic NumPy fail-safe.  The AI challenger is shadow-only
unless the lower confidence bounds remain positive after costs, it beats the
strongest paired baseline after multiple-horizon correction, and drawdown
stays inside the configured guard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Mapping
from itertools import combinations
import math

import numpy as np
import pandas as pd

try:  # SciPy 1.18 is pinned in requirements for the Python 3.12 runtime.
    import scipy as _scipy
    from scipy import stats as _scipy_stats
except Exception:  # Fail-soft keeps diagnostics available during a bad deploy.
    _scipy = None
    _scipy_stats = None

SELECTOR_VERSION = "1.6.0-absolute-dominant-400-universe"
SELECTOR_OUTCOME_VERSION = "selector_outcomes_v1.1-impact-aware"
SELECTOR_HORIZONS = (5, 20, 60)
SELECTOR_MODELS = (
    "RULE_ENGINE",
    "INDEPENDENT_SELECTOR",
    "AI_CHALLENGER",
    "RELATIVE_STRENGTH",
)

_FEATURE_CACHE: dict[str, tuple[str, pd.DataFrame]] = {}
_SILENT_PROFILE_CACHE: dict[str, tuple[str, dict[str, Any]]] = {}
_PANEL_CACHE: dict[str, pd.DataFrame] = {}
_EVALUATION_CACHE: dict[str, tuple[pd.DataFrame, dict[int, dict[str, Any]]]] = {}

FEATURE_COLUMNS = (
    "close_ema20",
    "ema20_ema50",
    "ema50_ema200",
    "ema20_slope10",
    "adx14_scaled",
    "roc20",
    "roc60",
    "roc120",
    "relative_strength60",
    "cmf20",
    "cmf60",
    "obv_slope20",
    "adl_slope20",
    "volume_ratio",
    "atr_pct",
    "distance_ema20_atr",
    "distance_52w_high",
    "positive_day_ratio20",
    "trend_efficiency20",
    "trend_efficiency60",
    "jump_concentration20",
    "drawdown60",
    "log_adtv20",
    "accumulation_proxy",
)


@dataclass(frozen=True)
class SelectorConfig:
    horizons: tuple[int, ...] = SELECTOR_HORIZONS
    min_history_bars: int = 220
    training_lookback_bars: int = 620
    anchor_step_bars: int = 5
    min_cross_section: int = 8
    min_training_rows: int = 180
    min_evaluation_rows: int = 60
    min_evaluation_dates: int = 12
    min_evaluation_tickers: int = 25
    calibration_fraction: float = 0.20
    evaluation_fraction: float = 0.20
    ridge_l2: float = 1.5
    logistic_l2: float = 0.12
    logistic_iterations: int = 160
    logistic_learning_rate: float = 0.045
    roundtrip_cost_pct: float = 0.0065
    market_impact_enabled: bool = True
    market_impact_very_illiquid_pct: float = 0.0150
    market_impact_illiquid_pct: float = 0.0075
    market_impact_medium_pct: float = 0.0035
    market_impact_liquid_pct: float = 0.0015
    market_impact_very_liquid_pct: float = 0.0005
    max_ai_weight: float = 0.30
    min_brier_skill: float = 0.0
    min_net_expectancy_pct: float = 0.0
    max_promotion_drawdown_pct: float = 20.0
    statistical_confidence_level: float = 0.90
    statistical_bootstrap_resamples: int = 999
    statistical_permutation_resamples: int = 1999
    statistical_significance_level: float = 0.10
    statistical_seed: int = 740
    cscv_slices: int = 6
    max_backtest_overfit_probability_pct: float = 50.0
    top_fraction: float = 0.20
    min_top_k: int = 3
    max_model_rows: int = 12000
    min_feature_coverage_pct: float = 75.0
    min_model_feature_coverage_pct: float = 80.0

    def replace(self, **changes: Any) -> "SelectorConfig":
        values = asdict(self)
        values.update(changes)
        return SelectorConfig(**values)


@dataclass
class _Scaler:
    median: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(values), values, self.median)
        return np.clip((filled - self.median) / self.scale, -8.0, 8.0)


@dataclass
class _LinearModel:
    coef: np.ndarray
    intercept: float
    scaler: _Scaler
    logistic: bool = False

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = _feature_matrix(frame)
        raw = self.scaler.transform(values) @ self.coef + self.intercept
        return _sigmoid(raw) if self.logistic else raw


@dataclass
class _PlattModel:
    slope: float = 1.0
    intercept: float = 0.0

    def predict(self, probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), 1e-5, 1.0 - 1e-5)
        logits = np.log(p / (1.0 - p))
        return _sigmoid(self.slope * logits + self.intercept)


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


def _clip(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    return float(np.clip(_finite(value, low), low, high))


def _liquidity_cost_from_log_adtv(
    log_adtv20: Any,
    config: SelectorConfig,
) -> tuple[str, float]:
    """Return a conservative one-roundtrip market-impact allowance.

    The values are configurable bucket priors, not an assertion that every
    order experiences the same impact. They prevent an illiquid winner from
    receiving the same after-cost label as a highly liquid stock.
    """
    log_value = _finite(log_adtv20, np.nan)
    adtv = math.expm1(log_value) if np.isfinite(log_value) else 0.0
    if adtv < 2_000_000_000.0:
        bucket = "VERY_ILLIQUID"
        cost = config.market_impact_very_illiquid_pct
    elif adtv < 10_000_000_000.0:
        bucket = "ILLIQUID"
        cost = config.market_impact_illiquid_pct
    elif adtv < 50_000_000_000.0:
        bucket = "MEDIUM"
        cost = config.market_impact_medium_pct
    elif adtv < 250_000_000_000.0:
        bucket = "LIQUID"
        cost = config.market_impact_liquid_pct
    else:
        bucket = "VERY_LIQUID"
        cost = config.market_impact_very_liquid_pct
    return bucket, max(0.0, float(cost)) if config.market_impact_enabled else 0.0


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=float), -35.0, 35.0)))


def _normalise_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert("Asia/Jakarta").tz_localize(None)
    return out[~out.index.isna()].sort_index()


def _frame_cache_signature(frame: pd.DataFrame, lookback: int = 720) -> str:
    """Cheap revision-aware signature for incremental in-process caches."""
    if frame is None or frame.empty:
        return "EMPTY"
    columns = [
        column for column in (
            "Open", "High", "Low", "Close", "Volume", "BENCH_CLOSE",
        ) if column in frame
    ]
    local = frame.loc[:, columns].tail(max(5, int(lookback))).copy()
    try:
        hashed = pd.util.hash_pandas_object(local, index=True).to_numpy(dtype=np.uint64)
        checksum = int(hashed.sum(dtype=np.uint64))
    except Exception:
        checksum = hash((
            len(local), str(local.index[-1]),
            tuple(_finite(local.iloc[-1].get(column), 0.0) for column in columns),
        ))
    return f"{len(frame)}|{frame.index[-1]}|{checksum}"


def _cached_technical_features(ticker: Any, frame: pd.DataFrame) -> pd.DataFrame:
    key = str(ticker).upper()
    signature = _frame_cache_signature(frame)
    cached = _FEATURE_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1].copy()
    features = _technical_feature_frame(frame)
    _FEATURE_CACHE[key] = (signature, features.copy())
    if len(_FEATURE_CACHE) > 2000:
        _FEATURE_CACHE.pop(next(iter(_FEATURE_CACHE)))
    return features


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _technical_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    local = _normalise_index(frame)
    close = _series(local, "Close")
    ema20 = _series(local, "EMA20")
    ema50 = _series(local, "EMA50")
    ema200 = _series(local, "EMA200")
    atr14 = _series(local, "ATR14")
    cmf20 = _series(local, "CMF20")
    cmf60 = _series(local, "CMF60")
    obv20 = _series(local, "OBV_SLOPE20")
    adl20 = _series(local, "ADL_SLOPE20")
    close_location = _series(local, "CLOSE_LOCATION")
    volume_ratio = _series(local, "VOL_RATIO")
    daily_return = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    absolute_return = daily_return.abs()
    rolling_path20 = absolute_return.rolling(20, min_periods=15).sum()
    rolling_path60 = absolute_return.rolling(60, min_periods=40).sum()

    out = pd.DataFrame(index=local.index)
    out["close_ema20"] = close / ema20.replace(0, np.nan) - 1.0
    out["ema20_ema50"] = ema20 / ema50.replace(0, np.nan) - 1.0
    out["ema50_ema200"] = ema50 / ema200.replace(0, np.nan) - 1.0
    out["ema20_slope10"] = ema20 / ema20.shift(10).replace(0, np.nan) - 1.0
    out["adx14_scaled"] = (_series(local, "ADX14") - 12.0) / 25.0
    out["roc20"] = _series(local, "ROC20")
    out["roc60"] = _series(local, "ROC60")
    out["roc120"] = _series(local, "ROC120")
    out["relative_strength60"] = _series(local, "REL_STRENGTH60")
    out["cmf20"] = cmf20
    out["cmf60"] = cmf60
    out["obv_slope20"] = obv20
    out["adl_slope20"] = adl20
    out["volume_ratio"] = volume_ratio
    out["atr_pct"] = _series(local, "ATR_PCT")
    out["distance_ema20_atr"] = (close - ema20) / atr14.replace(0, np.nan)
    out["distance_52w_high"] = _series(local, "DIST_52W_HIGH")
    # Momentum quality: gradual, persistent advances are preferable to a score
    # dominated by one jump.  All windows are point-in-time.
    out["positive_day_ratio20"] = daily_return.gt(0).rolling(20, min_periods=15).mean()
    out["trend_efficiency20"] = (
        (close / close.shift(20).replace(0, np.nan) - 1.0).abs()
        / rolling_path20.replace(0, np.nan)
    ).clip(0.0, 1.0)
    out["trend_efficiency60"] = (
        (close / close.shift(60).replace(0, np.nan) - 1.0).abs()
        / rolling_path60.replace(0, np.nan)
    ).clip(0.0, 1.0)
    out["jump_concentration20"] = (
        absolute_return.rolling(20, min_periods=15).max()
        / rolling_path20.replace(0, np.nan)
    ).clip(0.0, 1.0)
    out["drawdown60"] = (
        close / close.rolling(60, min_periods=40).max().replace(0, np.nan) - 1.0
    ).clip(-1.0, 0.0)
    out["log_adtv20"] = np.log1p(_series(local, "ADTV20").clip(lower=0.0))
    # Historical price-volume proxy.  It is fully point-in-time and does not
    # call the more expensive current Silent Accumulation profile per anchor.
    accumulation_inputs = pd.concat(
        [
            cmf20.rename("cmf20"),
            cmf60.rename("cmf60"),
            obv20.rename("obv20"),
            adl20.rename("adl20"),
            close_location.rename("close_location"),
            volume_ratio.rename("volume_ratio"),
        ],
        axis=1,
    )
    out["flow_feature_coverage_pct"] = (
        100.0 * accumulation_inputs.notna().mean(axis=1)
    )
    accumulation_proxy = np.clip(
        0.50
        + 0.80 * cmf20.fillna(0.0)
        + 0.45 * cmf60.fillna(0.0)
        + 0.12 * np.tanh(5.0 * obv20.fillna(0.0))
        + 0.10 * np.tanh(5.0 * adl20.fillna(0.0))
        + 0.08 * (close_location.fillna(0.5) - 0.5)
        + 0.04 * np.tanh(volume_ratio.fillna(1.0) - 1.0),
        0.0,
        1.0,
    )
    # A neutral-looking price-volume proxy is not evidence.  Keep it missing
    # until at least half of its causal inputs are actually observed.
    out["accumulation_proxy"] = accumulation_proxy.where(
        out["flow_feature_coverage_pct"].ge(50.0)
    )
    out["close"] = close
    out["benchmark_close"] = _series(local, "BENCH_CLOSE")
    out["net_stock_return_placeholder"] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame.reindex(columns=FEATURE_COLUMNS).apply(
        pd.to_numeric, errors="coerce",
    )
    # Median imputation alone can make "not observed" look like a genuine
    # cross-sectional median.  Append one binary indicator per feature so the
    # challenger can learn the distinction while the production coverage gate
    # remains authoritative.
    missing = numeric.isna().astype(float)
    return np.concatenate(
        [
            numeric.to_numpy(dtype=float),
            missing.to_numpy(dtype=float),
        ],
        axis=1,
    )


def _fit_scaler(values: np.ndarray) -> _Scaler:
    median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(values), values, median)
    q75 = np.nanpercentile(filled, 75, axis=0)
    q25 = np.nanpercentile(filled, 25, axis=0)
    scale = (q75 - q25) / 1.349
    std = np.nanstd(filled, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return _Scaler(median.astype(float), scale.astype(float))


def _fit_ridge(
    frame: pd.DataFrame,
    target: pd.Series,
    l2: float,
) -> _LinearModel | None:
    values = _feature_matrix(frame)
    y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)
    values, y = values[valid], y[valid]
    if len(y) < 24:
        return None
    scaler = _fit_scaler(values)
    x = scaler.transform(values)
    x_mean = np.mean(x, axis=0)
    y_mean = float(np.mean(y))
    centered = x - x_mean
    penalty = np.eye(centered.shape[1], dtype=float) * max(1e-6, float(l2))
    try:
        coef = np.linalg.solve(centered.T @ centered + penalty, centered.T @ (y - y_mean))
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(centered.T @ centered + penalty) @ centered.T @ (y - y_mean)
    intercept = y_mean - float(x_mean @ coef)
    return _LinearModel(coef.astype(float), intercept, scaler, logistic=False)


def _fit_logistic(
    frame: pd.DataFrame,
    target: pd.Series,
    config: SelectorConfig,
) -> _LinearModel | None:
    values = _feature_matrix(frame)
    y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)
    values, y = values[valid], y[valid]
    if len(y) < 24 or len(np.unique(y)) < 2:
        return None
    scaler = _fit_scaler(values)
    x = scaler.transform(values)
    coef = np.zeros(x.shape[1], dtype=float)
    base = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
    intercept = math.log(base / (1.0 - base))
    learning_rate = max(1e-4, float(config.logistic_learning_rate))
    configured_iterations = max(50, int(config.logistic_iterations))
    iterations = min(
        configured_iterations,
        90 if len(y) > 8000 else 120 if len(y) > 4000 else configured_iterations,
    )
    for _ in range(iterations):
        probability = _sigmoid(x @ coef + intercept)
        error = probability - y
        grad_coef = x.T @ error / len(y) + float(config.logistic_l2) * coef
        grad_intercept = float(np.mean(error))
        coef -= learning_rate * np.clip(grad_coef, -4.0, 4.0)
        intercept -= learning_rate * float(np.clip(grad_intercept, -2.0, 2.0))
    return _LinearModel(coef, float(intercept), scaler, logistic=True)


def _fit_platt(probability: np.ndarray, target: pd.Series) -> _PlattModel:
    p = np.clip(np.asarray(probability, dtype=float), 1e-5, 1.0 - 1e-5)
    y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    if valid.sum() < 15 or len(np.unique(y[valid])) < 2:
        return _PlattModel()
    x = np.log(p[valid] / (1.0 - p[valid]))
    y = y[valid]
    slope, intercept = 1.0, 0.0
    for _ in range(180):
        predicted = _sigmoid(slope * x + intercept)
        error = predicted - y
        slope -= 0.025 * float(np.mean(error * x) + 0.02 * slope)
        intercept -= 0.025 * float(np.mean(error))
    return _PlattModel(float(np.clip(slope, 0.05, 5.0)), float(np.clip(intercept, -5.0, 5.0)))


def _percentile(values: pd.Series, neutral: float = 50.0) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(neutral, index=values.index, dtype=float)
    ranked = 100.0 * numeric.rank(method="average", pct=True)
    return ranked.fillna(neutral)


def _group_percentile(
    values: pd.Series,
    groups: pd.Series | None,
    neutral: float = 50.0,
) -> pd.Series:
    """Vectorised equivalent of per-date ``_percentile``.

    Missing values remain neutral, and groups with zero/one observed value are
    deliberately neutral for every member. This preserves the fail-neutral
    behaviour of the reference implementation while avoiding hundreds of
    small DataFrame copies on 400-ticker panels.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if groups is None:
        return _percentile(numeric, neutral=neutral)
    keys = pd.Series(groups, index=values.index)
    counts = numeric.notna().groupby(keys, sort=False).transform("sum")
    ranked = numeric.groupby(keys, sort=False).rank(method="average", pct=True).mul(100.0)
    ranked = ranked.fillna(neutral)
    return ranked.where(counts.gt(1), neutral).astype(float)


def _score_cross_section(
    frame: pd.DataFrame,
    min_feature_coverage_pct: float = 75.0,
) -> pd.DataFrame:
    """Score all dates in one vectorised pass.

    The prior implementation looped over every ``as_of`` date and repeatedly
    built dozens of small Series/DataFrames. On a 400-ticker, ~620-bar selector
    panel that dominated scan time. The formulas and gates are unchanged; only
    their execution strategy is different.
    """
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    out = frame.copy()
    group_keys = (
        out["as_of"]
        if "as_of" in out
        else pd.Series("__ALL__", index=out.index, dtype="object")
    )
    feature_values = out.reindex(columns=FEATURE_COLUMNS).apply(
        pd.to_numeric, errors="coerce",
    )
    out["technical_feature_coverage_pct"] = (
        100.0 * feature_values.notna().mean(axis=1)
    ).round(1)
    coverage_weight = (
        out["technical_feature_coverage_pct"] / 100.0
    ).clip(0.0, 1.0)

    def pct(column: str | pd.Series) -> pd.Series:
        values = out[column] if isinstance(column, str) else column
        return _group_percentile(values, group_keys)

    trend = (
        0.27 * pct("close_ema20")
        + 0.25 * pct("ema20_ema50")
        + 0.22 * pct("ema50_ema200")
        + 0.14 * pct("ema20_slope10")
        + 0.12 * pct("adx14_scaled")
    )
    roc_ranks = pd.DataFrame({
        "20D": pct("roc20"),
        "60D": pct("roc60"),
        "120D": pct("roc120"),
    }, index=out.index)
    momentum = roc_ranks.mean(axis=1)
    momentum_consistency = (
        roc_ranks.mean(axis=1) - 0.35 * roc_ranks.std(axis=1, ddof=0)
    ).clip(0.0, 100.0)
    jump = -pd.to_numeric(out["jump_concentration20"], errors="coerce")
    momentum_continuity = (
        0.30 * pct("positive_day_ratio20")
        + 0.25 * pct("trend_efficiency20")
        + 0.25 * pct("trend_efficiency60")
        + 0.10 * pct(jump)
        + 0.10 * pct("drawdown60")
    )
    high52 = pct("distance_52w_high")
    momentum_quality = (
        0.30 * momentum
        + 0.25 * momentum_consistency
        + 0.25 * momentum_continuity
        + 0.20 * high52
    )
    relative = pct("relative_strength60")

    sector = out.get(
        "sector", pd.Series("", index=out.index, dtype="object"),
    ).fillna("").astype(str).str.strip()
    sector_keys = pd.MultiIndex.from_arrays([group_keys, sector])
    sector_peer_count = pd.Series(1, index=out.index).groupby(
        sector_keys, sort=False,
    ).transform("sum").where(sector.ne(""), 0)
    sector_raw = (
        0.55 * pd.to_numeric(out["relative_strength60"], errors="coerce")
        + 0.30 * pd.to_numeric(out["roc60"], errors="coerce")
        + 0.15 * pd.to_numeric(out["roc20"], errors="coerce")
    )
    eligible_sector = sector.ne("") & sector_peer_count.ge(5)
    sector_relative = pd.Series(50.0, index=out.index, dtype=float)
    if eligible_sector.any():
        eligible_keys = pd.MultiIndex.from_arrays([
            group_keys.loc[eligible_sector], sector.loc[eligible_sector],
        ])
        sector_relative.loc[eligible_sector] = (
            sector_raw.loc[eligible_sector]
            .groupby(eligible_keys, sort=False)
            .rank(method="average", pct=True)
            .mul(100.0)
        )

    flow = (
        0.28 * pct("cmf20")
        + 0.22 * pct("cmf60")
        + 0.20 * pct("obv_slope20")
        + 0.15 * pct("adl_slope20")
        + 0.15 * pct("accumulation_proxy")
    )
    distance = pd.to_numeric(out["distance_ema20_atr"], errors="coerce")
    atr_pct = pd.to_numeric(out["atr_pct"], errors="coerce")
    extension = (100.0 - 32.0 * (distance - 0.75).clip(lower=0.0)).clip(0.0, 100.0)
    structure = (
        45.0
        + 700.0 * pd.to_numeric(out["close_ema20"], errors="coerce").fillna(0.0)
        + 700.0 * pd.to_numeric(out["ema20_ema50"], errors="coerce").fillna(0.0)
    ).clip(0.0, 100.0)
    volatility = np.where(
        atr_pct.between(0.012, 0.060),
        100.0,
        np.where(
            atr_pct < 0.012,
            (atr_pct / 0.012 * 100.0).clip(0.0, 100.0),
            ((0.11 - atr_pct) / 0.05 * 100.0).clip(0.0, 100.0),
        ),
    )
    entry = (extension + structure + pd.Series(volatility, index=out.index)) / 3.0
    liquidity = pct("log_adtv20")
    absolute_rule = (
        35.0
        + 350.0 * pd.to_numeric(out["close_ema20"], errors="coerce").fillna(0.0)
        + 300.0 * pd.to_numeric(out["ema20_ema50"], errors="coerce").fillna(0.0)
        + 250.0 * pd.to_numeric(out["ema50_ema200"], errors="coerce").fillna(0.0)
        + 45.0 * pd.to_numeric(out["relative_strength60"], errors="coerce").fillna(0.0)
        + 20.0 * pd.to_numeric(out["cmf20"], errors="coerce").fillna(0.0)
    ).clip(0.0, 100.0)

    def coverage_adjust(score: pd.Series | np.ndarray) -> pd.Series:
        numeric = pd.to_numeric(pd.Series(score, index=out.index), errors="coerce").fillna(50.0)
        return (50.0 + coverage_weight * (numeric - 50.0)).clip(0.0, 100.0)

    trend = coverage_adjust(trend)
    momentum = coverage_adjust(momentum)
    momentum_consistency = coverage_adjust(momentum_consistency)
    momentum_continuity = coverage_adjust(momentum_continuity)
    high52 = coverage_adjust(high52)
    momentum_quality = coverage_adjust(momentum_quality)
    relative = coverage_adjust(relative)
    sector_relative = coverage_adjust(sector_relative)
    flow = coverage_adjust(flow)
    entry = coverage_adjust(entry)
    liquidity = coverage_adjust(liquidity)
    absolute_rule = coverage_adjust(absolute_rule)

    out["trend_score"] = trend.round(2)
    out["momentum_score"] = momentum.round(2)
    out["momentum_consistency_score"] = momentum_consistency.round(2)
    out["momentum_continuity_score"] = momentum_continuity.round(2)
    out["high_52w_proximity_score"] = high52.round(2)
    out["momentum_quality_score"] = momentum_quality.round(2)
    out["relative_strength_score"] = relative.round(2)
    out["sector_peer_count"] = sector_peer_count.fillna(0).astype(int)
    out["sector_relative_strength_score"] = sector_relative.round(2)
    out["sector_relative_state"] = np.where(
        eligible_sector, "SECTOR_NEUTRAL_ACTIVE", "INSUFFICIENT_SECTOR_PEERS",
    )
    out["flow_score"] = flow.round(2)
    out["entry_geometry_score"] = entry.round(2)
    out["liquidity_score"] = liquidity.round(2)
    out["rule_engine_score"] = absolute_rule.round(2)

    roc20 = pd.to_numeric(out["roc20"], errors="coerce")
    roc60 = pd.to_numeric(out["roc60"], errors="coerce")
    roc120 = pd.to_numeric(out["roc120"], errors="coerce")
    absolute_momentum = (
        0.45 * (50.0 + 250.0 * roc20).clip(0.0, 100.0)
        + 0.35 * (50.0 + 150.0 * roc60).clip(0.0, 100.0)
        + 0.20 * (50.0 + 100.0 * roc120).clip(0.0, 100.0)
    )
    cmf20 = pd.to_numeric(out["cmf20"], errors="coerce")
    cmf60 = pd.to_numeric(out["cmf60"], errors="coerce")
    accumulation = pd.to_numeric(out["accumulation_proxy"], errors="coerce")
    accumulation_score = np.where(
        accumulation.abs().le(1.5), 50.0 + 100.0 * accumulation, accumulation,
    )
    absolute_flow = (
        0.45 * (50.0 + 100.0 * cmf20).clip(0.0, 100.0)
        + 0.30 * (50.0 + 80.0 * cmf60).clip(0.0, 100.0)
        + 0.25 * pd.Series(accumulation_score, index=out.index).clip(0.0, 100.0)
    )
    log_adtv = pd.to_numeric(out["log_adtv20"], errors="coerce")
    absolute_liquidity = (
        50.0 + 20.0 * (log_adtv - np.log(2_000_000_000.0))
    ).clip(0.0, 100.0)
    relative_abs = (
        50.0 + 200.0 * pd.to_numeric(out["relative_strength60"], errors="coerce")
    ).clip(0.0, 100.0)
    absolute_selector = (
        0.38 * absolute_rule
        + 0.22 * absolute_momentum
        + 0.17 * absolute_flow
        + 0.11 * entry
        + 0.07 * absolute_liquidity
        + 0.05 * relative_abs
    )

    peer_count = out["ticker"].groupby(group_keys, sort=False).transform("nunique") if "ticker" in out else pd.Series(1, index=out.index)
    relative_weight = pd.Series(
        np.select(
            [peer_count.lt(20), peer_count.lt(50), peer_count.lt(100)],
            [0.0, 0.05, 0.10],
            default=0.10,
        ),
        index=out.index,
        dtype=float,
    )
    universe_state = np.select(
        [peer_count.lt(20), peer_count.lt(50), peer_count.lt(100)],
        [
            "ABSOLUTE_ONLY_SMALL_UNIVERSE",
            "LIMITED_RELATIVE_OVERLAY",
            "MODERATE_RELATIVE_OVERLAY",
        ],
        default="ABSOLUTE_DOMINANT_RELATIVE_OVERLAY",
    )
    relative_selector = (
        0.28 * trend
        + 0.22 * momentum_quality
        + 0.12 * relative
        + 0.08 * sector_relative
        + 0.15 * flow
        + 0.10 * entry
        + 0.05 * liquidity
    )
    out["absolute_momentum_score"] = coverage_adjust(absolute_momentum).round(2)
    out["absolute_flow_score"] = coverage_adjust(absolute_flow).round(2)
    out["absolute_liquidity_score"] = coverage_adjust(absolute_liquidity).round(2)
    out["absolute_selector_score"] = coverage_adjust(absolute_selector).round(2)
    out["relative_selector_overlay_score"] = coverage_adjust(relative_selector).round(2)
    out["cross_sectional_peer_count"] = peer_count.astype(int)
    out["relative_overlay_weight_pct"] = (100.0 * relative_weight).round(1)
    out["selector_universe_state"] = universe_state
    out["score_inflation_guard_active"] = relative_weight.lt(0.10)
    out["independent_selector_score"] = (
        (1.0 - relative_weight) * out["absolute_selector_score"]
        + relative_weight * out["relative_selector_overlay_score"]
    ).round(2)
    out["selector_rank_eligible"] = out[
        "technical_feature_coverage_pct"
    ].ge(float(min_feature_coverage_pct))
    out["selector_data_state"] = np.select(
        [
            out["technical_feature_coverage_pct"].ge(90.0),
            out["selector_rank_eligible"],
        ],
        ["FEATURES_COMPLETE", "FEATURES_SUFFICIENT"],
        default="DATA_PENDING_FEATURES",
    )
    out["selector_missing_feature_count"] = (
        len(FEATURE_COLUMNS) - feature_values.notna().sum(axis=1)
    ).astype(int)
    masks = feature_values.notna().to_numpy()
    columns = feature_values.columns.to_numpy(dtype=object)
    out["selector_missing_features"] = [
        " | ".join(columns[~mask].tolist()) for mask in masks
    ]
    return out.reset_index(drop=True)

def build_selector_panel(
    prepared: Mapping[str, pd.DataFrame],
    config: SelectorConfig | None = None,
    sector_map: Mapping[str, Any] | None = None,
    feature_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build a point-in-time training panel with unseen forward labels."""
    cfg = config or SelectorConfig()
    signature_parts = [
        f"{ticker}:{_frame_cache_signature(frame, cfg.training_lookback_bars + max(cfg.horizons) + 20)}"
        for ticker, frame in sorted(prepared.items(), key=lambda item: str(item[0]))
        if frame is not None and not frame.empty
    ]
    signature_parts.append(str({
        "horizons": cfg.horizons,
        "lookback": cfg.training_lookback_bars,
        "step": cfg.anchor_step_bars,
        "cost": cfg.roundtrip_cost_pct,
        "market_impact": (
            cfg.market_impact_enabled,
            cfg.market_impact_very_illiquid_pct,
            cfg.market_impact_illiquid_pct,
            cfg.market_impact_medium_pct,
            cfg.market_impact_liquid_pct,
            cfg.market_impact_very_liquid_pct,
        ),
        "cross": cfg.min_cross_section,
        "sectors": sorted(
            (str(key).upper(), _text(value))
            for key, value in (sector_map or {}).items()
        ),
    }))
    cache_key = sha256("|".join(signature_parts).encode("utf-8")).hexdigest()
    cached_panel = _PANEL_CACHE.get(cache_key)
    if cached_panel is not None:
        return cached_panel.copy()
    maximum_horizon = max(int(value) for value in cfg.horizons)
    blocks: list[pd.DataFrame] = []
    for ticker, raw in prepared.items():
        if raw is None or raw.empty or len(raw) < cfg.min_history_bars + maximum_horizon:
            continue
        features = (feature_frames or {}).get(str(ticker).upper())
        if features is None:
            features = _cached_technical_features(ticker, raw)
        start = max(
            cfg.min_history_bars - 1,
            len(features) - int(cfg.training_lookback_bars) - maximum_horizon,
        )
        stop = len(features) - maximum_horizon
        positions = np.arange(start, stop, max(1, int(cfg.anchor_step_bars)), dtype=int)
        if positions.size == 0:
            continue
        selected = features.iloc[positions]
        close = pd.to_numeric(selected.get("close"), errors="coerce").to_numpy(dtype=float)
        valid_close = np.isfinite(close)
        if not valid_close.any():
            continue
        positions = positions[valid_close]
        selected = selected.iloc[np.flatnonzero(valid_close)]
        close = close[valid_close]
        block = selected.reindex(columns=FEATURE_COLUMNS).apply(
            pd.to_numeric, errors="coerce",
        ).reset_index(drop=True)
        block.insert(0, "sector", _text(
            (sector_map or {}).get(
                str(ticker).upper(),
                (sector_map or {}).get(str(ticker), ""),
            )
        ))
        block.insert(0, "as_of", pd.DatetimeIndex(selected.index).normalize().to_numpy())
        block.insert(0, "ticker", str(ticker).upper())

        log_adtv = pd.to_numeric(block.get("log_adtv20"), errors="coerce").to_numpy(dtype=float)
        adtv = np.expm1(log_adtv)
        buckets = np.select(
            [
                ~np.isfinite(adtv) | (adtv < 2_000_000_000.0),
                adtv < 10_000_000_000.0,
                adtv < 50_000_000_000.0,
                adtv < 250_000_000_000.0,
            ],
            ["VERY_ILLIQUID", "ILLIQUID", "MEDIUM", "LIQUID"],
            default="VERY_LIQUID",
        )
        if cfg.market_impact_enabled:
            market_impact = np.select(
                [
                    ~np.isfinite(adtv) | (adtv < 2_000_000_000.0),
                    adtv < 10_000_000_000.0,
                    adtv < 50_000_000_000.0,
                    adtv < 250_000_000_000.0,
                ],
                [
                    cfg.market_impact_very_illiquid_pct,
                    cfg.market_impact_illiquid_pct,
                    cfg.market_impact_medium_pct,
                    cfg.market_impact_liquid_pct,
                ],
                default=cfg.market_impact_very_liquid_pct,
            ).astype(float)
        else:
            market_impact = np.zeros(len(block), dtype=float)
        total_cost = max(0.0, float(cfg.roundtrip_cost_pct)) + market_impact
        block["liquidity_bucket"] = buckets
        block["estimated_market_impact_cost_pct"] = market_impact
        block["estimated_total_cost_pct"] = total_cost

        benchmark_close = pd.to_numeric(
            selected.get("benchmark_close"), errors="coerce",
        ).to_numpy(dtype=float)
        all_close = pd.to_numeric(features.get("close"), errors="coerce").to_numpy(dtype=float)
        all_benchmark = pd.to_numeric(
            features.get("benchmark_close"), errors="coerce",
        ).to_numpy(dtype=float)
        for horizon in cfg.horizons:
            future_positions = positions + int(horizon)
            future_close = all_close[future_positions]
            future_benchmark = all_benchmark[future_positions]
            stock_return = np.divide(
                future_close, close,
                out=np.full(len(close), np.nan, dtype=float),
                where=np.isfinite(future_close) & np.isfinite(close) & (close > 0),
            ) - 1.0
            benchmark_return = np.divide(
                future_benchmark, benchmark_close,
                out=np.full(len(close), np.nan, dtype=float),
                where=(
                    np.isfinite(future_benchmark) & np.isfinite(benchmark_close)
                    & (future_benchmark > 0) & (benchmark_close > 0)
                ),
            ) - 1.0
            net_stock = np.where(np.isfinite(stock_return), stock_return - total_cost, np.nan)
            net_excess = np.where(
                np.isfinite(stock_return) & np.isfinite(benchmark_return),
                stock_return - benchmark_return - total_cost,
                np.nan,
            )
            block[f"stock_return_{horizon}d"] = stock_return
            block[f"benchmark_return_{horizon}d"] = benchmark_return
            block[f"net_stock_return_{horizon}d"] = net_stock
            block[f"net_excess_return_{horizon}d"] = net_excess
            block[f"outperform_{horizon}d"] = np.where(
                np.isfinite(net_excess), (net_excess > 0.0).astype(float), np.nan,
            )
        blocks.append(block)
    panel = pd.concat(blocks, ignore_index=True, sort=False) if blocks else pd.DataFrame()
    if panel.empty:
        return panel
    counts = panel.groupby("as_of")["ticker"].transform("nunique")
    panel = panel[counts >= max(2, int(cfg.min_cross_section))].copy()
    result = _score_cross_section(
        panel,
        min_feature_coverage_pct=cfg.min_model_feature_coverage_pct,
    ).sort_values(
        ["as_of", "ticker"], kind="stable",
    ).reset_index(drop=True)
    result.attrs["selector_panel_cache_key"] = cache_key
    _PANEL_CACHE[cache_key] = result.copy()
    if len(_PANEL_CACHE) > 1:
        _PANEL_CACHE.pop(next(iter(_PANEL_CACHE)))
    return result


def _chronological_three_way(
    frame: pd.DataFrame,
    config: SelectorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame is None or frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    dates = sorted(pd.to_datetime(frame["as_of"], errors="coerce").dropna().unique())
    if len(dates) < 5:
        return frame.iloc[0:0], frame.iloc[0:0], frame.copy()
    eval_count = max(1, int(math.ceil(len(dates) * config.evaluation_fraction)))
    cal_count = max(1, int(math.ceil(len(dates) * config.calibration_fraction)))
    if eval_count + cal_count >= len(dates):
        eval_count = max(1, len(dates) // 4)
        cal_count = max(1, len(dates) // 4)
    train_dates = set(dates[: len(dates) - cal_count - eval_count])
    calibration_dates = set(dates[len(dates) - cal_count - eval_count: len(dates) - eval_count])
    evaluation_dates = set(dates[len(dates) - eval_count:])
    return (
        frame[frame["as_of"].isin(train_dates)].copy(),
        frame[frame["as_of"].isin(calibration_dates)].copy(),
        frame[frame["as_of"].isin(evaluation_dates)].copy(),
    )


def _spearman_numpy(left: pd.Series, right: pd.Series) -> float:
    """Deterministic tie-aware fallback used only when SciPy is unavailable."""
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan
    x_rank = x[valid].rank(method="average").to_numpy(dtype=float)
    y_rank = y[valid].rank(method="average").to_numpy(dtype=float)
    x_centered = x_rank - float(np.mean(x_rank))
    y_centered = y_rank - float(np.mean(y_rank))
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if not np.isfinite(denominator) or denominator <= 1e-12:
        return np.nan
    return float(np.dot(x_centered, y_centered) / denominator)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    """Use SciPy's audited statistic with a NumPy fail-safe."""
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan
    if _scipy_stats is not None:
        try:
            result = _scipy_stats.spearmanr(
                x[valid].to_numpy(dtype=float),
                y[valid].to_numpy(dtype=float),
                nan_policy="omit",
            )
            statistic = _finite(getattr(result, "statistic", result[0]), np.nan)
            if np.isfinite(statistic):
                return statistic
        except Exception:
            pass
    return _spearman_numpy(x[valid], y[valid])


def _statistical_backend() -> str:
    return (
        f"SCIPY_{getattr(_scipy, '__version__', 'UNKNOWN')}"
        if _scipy_stats is not None else
        "NUMPY_DETERMINISTIC_FALLBACK"
    )


def _bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    config: SelectorConfig,
    seed_offset: int = 0,
) -> tuple[float, float, str]:
    """Return a deterministic confidence interval for the mean.

    SciPy BCa is primary.  Degenerate samples and deployment import failures
    fall back to a deterministic percentile bootstrap so the scanner can
    remain operational, but the backend is exposed in every audit row.
    """
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 3:
        return np.nan, np.nan, "INSUFFICIENT_SAMPLE"
    if np.nanmax(sample) - np.nanmin(sample) <= 1e-12:
        value = float(np.mean(sample))
        return value, value, "CONSTANT_SAMPLE"
    confidence = float(np.clip(config.statistical_confidence_level, 0.50, 0.999))
    resamples = max(199, int(config.statistical_bootstrap_resamples))
    rng = np.random.default_rng(int(config.statistical_seed) + int(seed_offset))
    if _scipy_stats is not None:
        try:
            result = _scipy_stats.bootstrap(
                (sample,),
                np.mean,
                vectorized=False,
                paired=False,
                confidence_level=confidence,
                n_resamples=resamples,
                method="BCa",
                rng=rng,
            )
            low = _finite(result.confidence_interval.low, np.nan)
            high = _finite(result.confidence_interval.high, np.nan)
            if np.isfinite(low) and np.isfinite(high):
                return low, high, "SCIPY_BCA"
        except Exception:
            pass
    draws = rng.choice(sample, size=(resamples, len(sample)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(draws, alpha)),
        float(np.quantile(draws, 1.0 - alpha)),
        "NUMPY_PERCENTILE_FALLBACK",
    )


def _paired_permutation_pvalue(
    differences: list[float] | np.ndarray,
    config: SelectorConfig,
    seed_offset: int = 0,
) -> tuple[float, str]:
    """One-sided paired sign-flip test that mean advantage is above zero."""
    sample = np.asarray(differences, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 3:
        return np.nan, "INSUFFICIENT_SAMPLE"
    resamples = max(199, int(config.statistical_permutation_resamples))
    rng = np.random.default_rng(int(config.statistical_seed) + 10_000 + int(seed_offset))
    if _scipy_stats is not None:
        try:
            result = _scipy_stats.permutation_test(
                (sample,),
                lambda values: float(np.mean(values)),
                permutation_type="samples",
                vectorized=False,
                n_resamples=resamples,
                alternative="greater",
                rng=rng,
            )
            pvalue = _finite(result.pvalue, np.nan)
            if np.isfinite(pvalue):
                return pvalue, "SCIPY_PAIRED_SIGN_FLIP"
        except Exception:
            pass
    observed = float(np.mean(sample))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, len(sample)))
    simulated = (signs * sample).mean(axis=1)
    pvalue = (1.0 + float(np.sum(simulated >= observed))) / (resamples + 1.0)
    return pvalue, "NUMPY_PAIRED_SIGN_FLIP_FALLBACK"


def _max_drawdown(returns: list[float]) -> float:
    if not returns:
        return np.nan
    equity = np.cumprod(1.0 + np.clip(np.asarray(returns, dtype=float), -0.95, 3.0))
    peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / np.where(peak > 0, peak, 1.0)
    return float(100.0 * np.nanmax(drawdown)) if len(drawdown) else np.nan


def _cscv_backtest_overfit_probability(
    model_date_excess: Mapping[str, Mapping[str, float]],
    config: SelectorConfig,
) -> dict[str, Any]:
    """Estimate strategy-family overfit risk with CSCV.

    This follows the PBO paper's core construction: split common OOS dates
    into an even number of contiguous slices, choose the in-sample winner on
    every half-slice combination, then measure how often that winner ranks
    below the median on the complementary dates.
    """
    models = [model for model in SELECTOR_MODELS if model in model_date_excess]
    common_dates: set[str] | None = None
    for model in models:
        dates = {
            date for date, value in model_date_excess.get(model, {}).items()
            if np.isfinite(_finite(value, np.nan))
        }
        common_dates = dates if common_dates is None else common_dates & dates
    ordered_dates = sorted(common_dates or ())
    slices = max(4, int(config.cscv_slices))
    if slices % 2:
        slices += 1
    if len(models) < 3 or len(ordered_dates) < slices * 2:
        return {
            "cscv_pbo_pct": np.nan,
            "cscv_trials": 0,
            "cscv_slices": slices,
            "cscv_common_dates": len(ordered_dates),
            "cscv_state": "INSUFFICIENT_CSCV_SAMPLE",
        }
    matrix = np.asarray([
        [
            _finite(model_date_excess[model].get(date), np.nan)
            for model in models
        ]
        for date in ordered_dates
    ], dtype=float)
    blocks = [np.asarray(block, dtype=int) for block in np.array_split(
        np.arange(len(ordered_dates)), slices,
    ) if len(block)]
    lambdas: list[float] = []
    degradation: list[float] = []
    winner_counts = {model: 0 for model in models}
    for chosen_blocks in combinations(range(len(blocks)), len(blocks) // 2):
        train_index = np.concatenate([blocks[index] for index in chosen_blocks])
        test_blocks = [
            index for index in range(len(blocks)) if index not in chosen_blocks
        ]
        test_index = np.concatenate([blocks[index] for index in test_blocks])
        train_performance = np.nanmean(matrix[train_index], axis=0)
        test_performance = np.nanmean(matrix[test_index], axis=0)
        if not (
            np.isfinite(train_performance).all()
            and np.isfinite(test_performance).all()
        ):
            continue
        winner = int(np.argmax(train_performance))
        winner_counts[models[winner]] += 1
        ranks = pd.Series(test_performance).rank(
            method="average", ascending=True,
        ).to_numpy(dtype=float)
        relative_rank = float(ranks[winner] / (len(models) + 1.0))
        relative_rank = float(np.clip(relative_rank, 1e-6, 1.0 - 1e-6))
        lambdas.append(math.log(relative_rank / (1.0 - relative_rank)))
        degradation.append(
            float(test_performance[winner] - train_performance[winner])
        )
    if not lambdas:
        return {
            "cscv_pbo_pct": np.nan,
            "cscv_trials": 0,
            "cscv_slices": slices,
            "cscv_common_dates": len(ordered_dates),
            "cscv_state": "INSUFFICIENT_CSCV_SAMPLE",
        }
    pbo = 100.0 * float(np.mean(np.asarray(lambdas) <= 0.0))
    return {
        "cscv_pbo_pct": round(pbo, 2),
        "cscv_trials": len(lambdas),
        "cscv_slices": slices,
        "cscv_common_dates": len(ordered_dates),
        "cscv_median_winner_degradation_pct": round(
            float(np.median(degradation)), 4,
        ),
        "cscv_most_selected_model": max(
            winner_counts, key=winner_counts.get,
        ),
        "cscv_state": (
            "PBO_GUARD_PASSED"
            if pbo <= float(config.max_backtest_overfit_probability_pct)
            else "PBO_GUARD_FAILED"
        ),
    }


def _evaluate_ranking(
    frame: pd.DataFrame,
    score_column: str,
    probability: pd.Series,
    horizon: int,
    baseline_probability: float,
    config: SelectorConfig,
) -> dict[str, Any]:
    target_column = f"net_excess_return_{horizon}d"
    stock_column = f"net_stock_return_{horizon}d"
    binary_column = f"outperform_{horizon}d"
    local = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    required_schema = {'as_of', 'ticker', target_column, stock_column, binary_column}
    if local.empty or not required_schema.issubset(local.columns):
        return {
            'evaluation_rows': 0, 'evaluation_dates': 0, 'evaluation_tickers': 0,
            'brier_score': np.nan, 'baseline_brier_score': np.nan,
            'brier_skill_pct': np.nan, 'brier_skill_ci_low_pct': np.nan,
            'brier_skill_ci_high_pct': np.nan, 'net_excess_expectancy_pct': np.nan,
            'net_excess_expectancy_ci_low_pct': np.nan,
            'net_excess_expectancy_ci_high_pct': np.nan,
            'net_absolute_expectancy_pct': np.nan, 'topk_hit_rate_pct': np.nan,
            'spearman_ic': np.nan, 'max_drawdown_pct': np.nan,
            'statistical_backend': _statistical_backend(),
            'statistical_confidence_level_pct': round(100.0 * float(config.statistical_confidence_level), 1),
            'statistical_bootstrap_resamples': int(config.statistical_bootstrap_resamples),
            'expectancy_ci_method': 'NOT_EVALUATED', 'brier_ci_method': 'NOT_EVALUATED',
            '_date_excess_pct': {}, 'evaluation_state': 'INSUFFICIENT_SCHEMA_OR_ROWS',
        }
    local["_score"] = pd.to_numeric(local.get(score_column), errors="coerce")
    local["_probability"] = pd.to_numeric(probability, errors="coerce")
    local["_target"] = pd.to_numeric(local.get(target_column), errors="coerce")
    local["_stock"] = pd.to_numeric(local.get(stock_column), errors="coerce")
    local["_binary"] = pd.to_numeric(local.get(binary_column), errors="coerce")
    valid_probability = local["_probability"].notna() & local["_binary"].isin([0.0, 1.0])
    if valid_probability.any():
        brier = float(np.mean((local.loc[valid_probability, "_probability"] - local.loc[valid_probability, "_binary"]) ** 2))
        baseline = float(np.mean((baseline_probability - local.loc[valid_probability, "_binary"]) ** 2))
        brier_skill = 1.0 - brier / baseline if baseline > 1e-12 else np.nan
    else:
        brier = baseline = brier_skill = np.nan

    unique_dates = sorted(pd.to_datetime(local["as_of"], errors="coerce").dropna().unique())
    stride = max(1, int(math.ceil(horizon / max(1, config.anchor_step_bars))))
    selected_dates = {pd.Timestamp(value).normalize() for value in unique_dates[::stride]}
    top_returns: list[float] = []
    top_excess: list[float] = []
    hit_values: list[float] = []
    correlations: list[float] = []
    date_excess_pct: dict[str, float] = {}
    date_brier_improvement: list[float] = []
    for date_value, group in local.groupby("as_of", sort=True):
        date_stamp = pd.Timestamp(date_value).normalize()
        if date_stamp not in selected_dates:
            continue
        probability_rows = group["_probability"].notna() & group["_binary"].isin([0.0, 1.0])
        if probability_rows.any() and np.isfinite(baseline) and baseline > 1e-12:
            model_loss = float(np.mean(
                (
                    group.loc[probability_rows, "_probability"]
                    - group.loc[probability_rows, "_binary"]
                ) ** 2
            ))
            baseline_loss = float(np.mean(
                (baseline_probability - group.loc[probability_rows, "_binary"]) ** 2
            ))
            date_brier_improvement.append(
                100.0 * (baseline_loss - model_loss) / baseline
            )
        eligible = group.dropna(subset=["_score", "_target"]).sort_values("_score", ascending=False)
        if eligible.empty:
            continue
        top_k = max(config.min_top_k, int(math.ceil(len(eligible) * config.top_fraction)))
        top = eligible.head(min(len(eligible), top_k))
        excess_value = float(top["_target"].mean())
        top_excess.append(excess_value)
        top_returns.append(float(top["_stock"].mean()))
        hit_values.append(float(top["_binary"].mean()))
        correlations.append(_spearman(eligible["_score"], eligible["_target"]))
        date_excess_pct[date_stamp.date().isoformat()] = 100.0 * excess_value
    seed_offset = int.from_bytes(
        sha256(f"{score_column}|{horizon}".encode("utf-8")).digest()[:4],
        "big",
    )
    expectancy_low, expectancy_high, expectancy_method = _bootstrap_mean_ci(
        100.0 * np.asarray(top_excess, dtype=float),
        config,
        seed_offset=seed_offset,
    )
    brier_low, brier_high, brier_method = _bootstrap_mean_ci(
        date_brier_improvement,
        config,
        seed_offset=seed_offset + 1,
    )
    return {
        "evaluation_rows": int(local["_target"].notna().sum()),
        "evaluation_dates": int(len(top_excess)),
        "evaluation_tickers": int(local.loc[local["_target"].notna(), "ticker"].nunique()),
        "brier_score": round(brier, 6) if np.isfinite(brier) else np.nan,
        "baseline_brier_score": round(baseline, 6) if np.isfinite(baseline) else np.nan,
        "brier_skill_pct": round(100.0 * brier_skill, 2) if np.isfinite(brier_skill) else np.nan,
        "brier_skill_ci_low_pct": round(brier_low, 3) if np.isfinite(brier_low) else np.nan,
        "brier_skill_ci_high_pct": round(brier_high, 3) if np.isfinite(brier_high) else np.nan,
        "net_excess_expectancy_pct": round(100.0 * float(np.mean(top_excess)), 3) if top_excess else np.nan,
        "net_excess_expectancy_ci_low_pct": round(expectancy_low, 3) if np.isfinite(expectancy_low) else np.nan,
        "net_excess_expectancy_ci_high_pct": round(expectancy_high, 3) if np.isfinite(expectancy_high) else np.nan,
        "net_absolute_expectancy_pct": round(100.0 * float(np.mean(top_returns)), 3) if top_returns else np.nan,
        "topk_hit_rate_pct": round(100.0 * float(np.mean(hit_values)), 1) if hit_values else np.nan,
        "spearman_ic": round(float(np.nanmean(correlations)), 4) if any(np.isfinite(correlations)) else np.nan,
        "max_drawdown_pct": round(_max_drawdown(top_returns), 2),
        "statistical_backend": _statistical_backend(),
        "statistical_confidence_level_pct": round(
            100.0 * float(config.statistical_confidence_level), 1,
        ),
        "statistical_bootstrap_resamples": int(config.statistical_bootstrap_resamples),
        "expectancy_ci_method": expectancy_method,
        "brier_ci_method": brier_method,
        "_date_excess_pct": date_excess_pct,
    }


def _model_probability_from_score(
    train_score: pd.Series,
    train_target: pd.Series,
    evaluation_score: pd.Series,
) -> np.ndarray:
    score = pd.to_numeric(train_score, errors="coerce")
    target = pd.to_numeric(train_target, errors="coerce")
    valid = score.notna() & target.isin([0.0, 1.0])
    if valid.sum() < 15 or target[valid].nunique() < 2:
        return np.full(len(evaluation_score), float(target[valid].mean()) if valid.any() else 0.5)
    x = score[valid].to_numpy(dtype=float) / 100.0
    y = target[valid].to_numpy(dtype=float)
    slope, intercept = 2.0, math.log(np.clip(y.mean(), 1e-4, 1 - 1e-4) / np.clip(1 - y.mean(), 1e-4, 1))
    for _ in range(300):
        probability = _sigmoid(slope * (x - 0.5) + intercept)
        error = probability - y
        slope -= 0.04 * float(np.mean(error * (x - 0.5)) + 0.01 * slope)
        intercept -= 0.04 * float(np.mean(error))
    current = pd.to_numeric(evaluation_score, errors="coerce").fillna(50.0).to_numpy(dtype=float) / 100.0
    return _sigmoid(slope * (current - 0.5) + intercept)


def _bounded_model_sample(frame: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    """Deterministically cap model fitting while retaining the full OOS audit."""
    if frame is None or frame.empty or len(frame) <= max(24, int(maximum_rows)):
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    ordered = frame.sort_values(["as_of", "ticker"], kind="stable").reset_index(drop=True)
    positions = np.linspace(
        0, len(ordered) - 1, num=max(24, int(maximum_rows)), dtype=int,
    )
    return ordered.iloc[np.unique(positions)].copy()


def evaluate_selector_challengers(
    panel: pd.DataFrame,
    config: SelectorConfig | None = None,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    """Chronologically compare all challengers and return promotion metadata."""
    cfg = config or SelectorConfig()
    audit_rows: list[dict[str, Any]] = []
    fitted: dict[int, dict[str, Any]] = {}
    if panel is None or panel.empty:
        return pd.DataFrame(), fitted
    required_panel = {'ticker', 'as_of'}
    if not required_panel.issubset(panel.columns):
        audit = pd.DataFrame([{
            'horizon': f'{horizon}D', 'horizon_bars': int(horizon),
            'model': model, 'selector_version': SELECTOR_VERSION,
            'training_rows': 0, 'calibration_rows': 0, 'evaluation_rows': 0,
            'promotion_state': 'INSUFFICIENT_PANEL_SCHEMA',
        } for horizon in cfg.horizons for model in SELECTOR_MODELS])
        return audit, fitted
    panel_key = _text(panel.attrs.get("selector_panel_cache_key"))
    if not panel_key:
        hashed = pd.util.hash_pandas_object(
            panel.reindex(columns=[
                column for column in (
                    "ticker", "as_of", *FEATURE_COLUMNS,
                    *(f"net_excess_return_{horizon}d" for horizon in cfg.horizons),
                ) if column in panel
            ]),
            index=False,
        ).to_numpy(dtype=np.uint64)
        panel_key = f"{len(panel)}|{int(hashed.sum(dtype=np.uint64))}"
    evaluation_key = sha256(
        f"{panel_key}|{asdict(cfg)}|{SELECTOR_VERSION}".encode("utf-8")
    ).hexdigest()
    cached_evaluation = _EVALUATION_CACHE.get(evaluation_key)
    if cached_evaluation is not None:
        return cached_evaluation[0].copy(), dict(cached_evaluation[1])

    for horizon in cfg.horizons:
        labelled = panel.dropna(subset=[
            f"net_excess_return_{horizon}d",
            f"outperform_{horizon}d",
        ]).copy()
        if "technical_feature_coverage_pct" in labelled:
            labelled = labelled[
                pd.to_numeric(
                    labelled["technical_feature_coverage_pct"],
                    errors="coerce",
                ).ge(cfg.min_model_feature_coverage_pct)
            ].copy()
        train, calibration, evaluation = _chronological_three_way(labelled, cfg)
        model_train = _bounded_model_sample(train, cfg.max_model_rows)
        baseline_probability = float(
            pd.to_numeric(train.get(f"outperform_{horizon}d"), errors="coerce").mean()
        ) if not train.empty else 0.5
        baseline_probability = float(np.clip(baseline_probability if np.isfinite(baseline_probability) else 0.5, 0.02, 0.98))

        ridge = _fit_ridge(
            model_train, model_train.get(f"net_excess_return_{horizon}d"), cfg.ridge_l2,
        ) if len(train) >= cfg.min_training_rows else None
        logistic = _fit_logistic(
            model_train, model_train.get(f"outperform_{horizon}d"), cfg,
        ) if len(train) >= cfg.min_training_rows else None
        platt = _PlattModel()
        if logistic is not None and not calibration.empty:
            platt = _fit_platt(
                logistic.predict(calibration),
                calibration[f"outperform_{horizon}d"],
            )
        ai_probability = (
            platt.predict(logistic.predict(evaluation))
            if logistic is not None and not evaluation.empty
            else np.full(len(evaluation), np.nan)
        )
        if ridge is not None and not evaluation.empty:
            evaluation = evaluation.copy()
            evaluation["_ai_score"] = ridge.predict(evaluation)
            evaluation["_ai_rank_score"] = (
                evaluation.groupby("as_of")["_ai_score"].rank(method="average", pct=True) * 100.0
            )
        else:
            evaluation = evaluation.copy()
            evaluation["_ai_rank_score"] = np.nan

        model_columns = {
            "RULE_ENGINE": "rule_engine_score",
            "INDEPENDENT_SELECTOR": "independent_selector_score",
            "AI_CHALLENGER": "_ai_rank_score",
            "RELATIVE_STRENGTH": "relative_strength_score",
        }
        horizon_rows: list[dict[str, Any]] = []
        model_date_excess: dict[str, dict[str, float]] = {}
        combined_for_calibration = pd.concat([train, calibration], ignore_index=True, sort=False)
        for model_name, score_column in model_columns.items():
            if model_name == "AI_CHALLENGER":
                probability = pd.Series(ai_probability, index=evaluation.index)
            else:
                probability = pd.Series(
                    _model_probability_from_score(
                        combined_for_calibration.get(score_column, pd.Series(dtype=float)),
                        combined_for_calibration.get(f"outperform_{horizon}d", pd.Series(dtype=float)),
                        evaluation.get(score_column, pd.Series(index=evaluation.index, dtype=float)),
                    ),
                    index=evaluation.index,
                )
            metrics = _evaluate_ranking(
                evaluation,
                score_column,
                probability,
                int(horizon),
                baseline_probability,
                cfg,
            )
            model_date_excess[model_name] = dict(metrics.pop("_date_excess_pct", {}))
            record = {
                "horizon": f"{horizon}D",
                "horizon_bars": int(horizon),
                "model": model_name,
                "selector_version": SELECTOR_VERSION,
                "training_rows": int(len(train)),
                "model_fit_rows": int(len(model_train)),
                "calibration_rows": int(len(calibration)),
                **metrics,
            }
            horizon_rows.append(record)

        ai_row = next(row for row in horizon_rows if row["model"] == "AI_CHALLENGER")
        baseline_rows = [row for row in horizon_rows if row["model"] != "AI_CHALLENGER"]
        best_baseline_row = max(
            baseline_rows,
            key=lambda row: _finite(row.get("net_excess_expectancy_pct"), -np.inf),
        )
        best_baseline_name = _text(best_baseline_row.get("model")) or "UNKNOWN"
        ai_dates = model_date_excess.get("AI_CHALLENGER", {})
        baseline_dates = model_date_excess.get(best_baseline_name, {})
        paired_dates = sorted(set(ai_dates) & set(baseline_dates))
        paired_advantage = [
            _finite(ai_dates[date], np.nan) - _finite(baseline_dates[date], np.nan)
            for date in paired_dates
        ]
        paired_advantage = [value for value in paired_advantage if np.isfinite(value)]
        advantage_low, advantage_high, advantage_ci_method = _bootstrap_mean_ci(
            paired_advantage,
            cfg,
            seed_offset=50_000 + int(horizon),
        )
        advantage_pvalue, advantage_test_method = _paired_permutation_pvalue(
            paired_advantage,
            cfg,
            seed_offset=60_000 + int(horizon),
        )
        # Family-wise correction is deliberately conservative.  It prevents
        # trying 5D/20D/60D from tripling the AI's chance of a lucky promotion.
        adjusted_pvalue = (
            min(1.0, advantage_pvalue * max(1, len(cfg.horizons)))
            if np.isfinite(advantage_pvalue) else np.nan
        )
        sufficient = bool(
            ai_row["training_rows"] >= cfg.min_training_rows
            and ai_row["evaluation_rows"] >= cfg.min_evaluation_rows
            and ai_row["evaluation_dates"] >= cfg.min_evaluation_dates
            and ai_row["evaluation_tickers"] >= cfg.min_evaluation_tickers
            and len(paired_advantage) >= cfg.min_evaluation_dates
            and ridge is not None and logistic is not None
        )
        ai_expectancy = _finite(ai_row.get("net_excess_expectancy_pct"), -np.inf)
        best_baseline_expectancy = max(
            (_finite(row.get("net_excess_expectancy_pct"), -np.inf) for row in baseline_rows),
            default=-np.inf,
        )
        brier_ci_low = _finite(ai_row.get("brier_skill_ci_low_pct"), -np.inf)
        expectancy_ci_low = _finite(
            ai_row.get("net_excess_expectancy_ci_low_pct"), -np.inf,
        )
        pbo_audit = _cscv_backtest_overfit_probability(
            model_date_excess, cfg,
        )
        pbo_value = _finite(pbo_audit.get("cscv_pbo_pct"), np.nan)
        pbo_gate = bool(
            np.isfinite(pbo_value)
            and pbo_value <= cfg.max_backtest_overfit_probability_pct
        )
        advantage_gate = bool(
            _finite(advantage_low, -np.inf) > 0.0
            and np.isfinite(adjusted_pvalue)
            and adjusted_pvalue <= cfg.statistical_significance_level
        )
        promoted = bool(
            sufficient
            and brier_ci_low > 100.0 * cfg.min_brier_skill
            and expectancy_ci_low > cfg.min_net_expectancy_pct
            and _finite(ai_row.get("max_drawdown_pct"), np.inf) <= cfg.max_promotion_drawdown_pct
            and ai_expectancy > best_baseline_expectancy
            and advantage_gate
            and pbo_gate
        )
        promotion_state = (
            "PROMOTED_AI_CHAMPION"
            if promoted else
            "INSUFFICIENT_EVIDENCE"
            if not sufficient else
            "REJECTED_BRIER_CI_NOT_POSITIVE"
            if brier_ci_low <= 100.0 * cfg.min_brier_skill else
            "REJECTED_EXPECTANCY_CI_NOT_POSITIVE"
            if expectancy_ci_low <= cfg.min_net_expectancy_pct else
            "REJECTED_DRAWDOWN"
            if _finite(ai_row.get("max_drawdown_pct"), np.inf) > cfg.max_promotion_drawdown_pct else
            "REJECTED_BACKTEST_OVERFIT_RISK"
            if not pbo_gate else
            "REJECTED_DID_NOT_BEAT_BASELINE_POINT_ESTIMATE"
            if ai_expectancy <= best_baseline_expectancy else
            "REJECTED_PAIRED_BASELINE_TEST"
        )
        gate_reasons: list[str] = []
        if not sufficient:
            gate_reasons.append("sample walk-forward belum mencukupi")
        if brier_ci_low <= 100.0 * cfg.min_brier_skill:
            gate_reasons.append("lower CI Brier skill belum positif")
        if expectancy_ci_low <= cfg.min_net_expectancy_pct:
            gate_reasons.append("lower CI expectancy net belum positif")
        if _finite(ai_row.get("max_drawdown_pct"), np.inf) > cfg.max_promotion_drawdown_pct:
            gate_reasons.append("drawdown melampaui guard")
        if not pbo_gate:
            gate_reasons.append(
                "CSCV PBO belum tersedia/masih di atas batas"
            )
        if ai_expectancy <= best_baseline_expectancy:
            gate_reasons.append(f"point estimate belum mengalahkan {best_baseline_name}")
        if not advantage_gate:
            gate_reasons.append("paired advantage belum signifikan setelah koreksi 3 horizon")
        promotion_gate_reason = (
            "Semua statistical champion gates lolos"
            if promoted else
            " • ".join(dict.fromkeys(gate_reasons))
        )
        ranked_models = sorted(
            horizon_rows,
            key=lambda row: (
                _finite(row.get("net_excess_expectancy_pct"), -np.inf),
                _finite(row.get("spearman_ic"), -np.inf),
                -_finite(row.get("max_drawdown_pct"), np.inf),
            ),
            reverse=True,
        )
        ranks = {row["model"]: rank for rank, row in enumerate(ranked_models, start=1)}
        for row in horizon_rows:
            row["challenger_rank"] = ranks[row["model"]]
            row["walkforward_best_model"] = ranked_models[0]["model"]
            row["strongest_baseline_model"] = best_baseline_name
            row["ai_vs_baseline_paired_dates"] = int(len(paired_advantage))
            row["ai_vs_baseline_advantage_pct"] = round(
                float(np.mean(paired_advantage)), 3,
            ) if paired_advantage else np.nan
            row["ai_vs_baseline_advantage_ci_low_pct"] = (
                round(advantage_low, 3) if np.isfinite(advantage_low) else np.nan
            )
            row["ai_vs_baseline_advantage_ci_high_pct"] = (
                round(advantage_high, 3) if np.isfinite(advantage_high) else np.nan
            )
            row["ai_vs_baseline_pvalue"] = (
                round(advantage_pvalue, 6) if np.isfinite(advantage_pvalue) else np.nan
            )
            row["ai_vs_baseline_pvalue_adjusted"] = (
                round(adjusted_pvalue, 6) if np.isfinite(adjusted_pvalue) else np.nan
            )
            row["multiple_testing_correction"] = (
                f"BONFERRONI_{max(1, len(cfg.horizons))}_HORIZONS"
            )
            row["paired_advantage_ci_method"] = advantage_ci_method
            row["paired_advantage_test_method"] = advantage_test_method
            row.update(pbo_audit)
            row["max_cscv_pbo_pct"] = float(
                cfg.max_backtest_overfit_probability_pct
            )
            # The independent selector is the frozen production baseline.
            # A statistically interesting but insufficient AI result remains
            # a challenger and must never be labelled as promoted.
            row["promoted_model"] = (
                "AI_CHALLENGER" if promoted else "INDEPENDENT_SELECTOR"
            )
            row["ai_promotion_state"] = promotion_state
            row["ai_promotion_gate_reason"] = promotion_gate_reason
            row["ai_can_influence"] = bool(promoted and row["model"] == "AI_CHALLENGER")
            audit_rows.append(row)

        # Models fitted on all resolved history are used only after the untouched
        # evaluation decision above has been frozen.
        full_fit = _bounded_model_sample(labelled, cfg.max_model_rows)
        full_ridge = _fit_ridge(
            full_fit, full_fit[f"net_excess_return_{horizon}d"], cfg.ridge_l2,
        )
        full_logistic = _fit_logistic(
            full_fit, full_fit[f"outperform_{horizon}d"], cfg,
        )
        fitted[int(horizon)] = {
            "promoted": promoted,
            "promotion_state": promotion_state,
            "ridge": full_ridge,
            "logistic": full_logistic,
            "platt": platt,
            "ai_weight": cfg.max_ai_weight if promoted else 0.0,
            "best_model": "AI_CHALLENGER" if promoted else "INDEPENDENT_SELECTOR",
            "statistical_backend": _statistical_backend(),
            "promotion_gate_reason": promotion_gate_reason,
        }
    audit_frame = pd.DataFrame(audit_rows)
    _EVALUATION_CACHE[evaluation_key] = (audit_frame.copy(), dict(fitted))
    if len(_EVALUATION_CACHE) > 4:
        _EVALUATION_CACHE.pop(next(iter(_EVALUATION_CACHE)))
    return audit_frame, fitted


def current_silent_profiles(
    prepared: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Compute each current Silent Accumulation profile once per scan."""
    try:
        from scanner import silent_accumulation_profile
    except Exception:
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    for ticker, frame in prepared.items():
        key = str(ticker).upper()
        embedded = getattr(frame, 'attrs', {}).get('_silent_accumulation_profile')
        if isinstance(embedded, Mapping) and embedded:
            profiles[key] = dict(embedded)
            continue
        signature = _frame_cache_signature(frame, lookback=180)
        cached = _SILENT_PROFILE_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            profiles[key] = dict(cached[1])
            continue
        try:
            profile = dict(silent_accumulation_profile(frame))
        except Exception:
            profile = {}
        profiles[key] = profile
        _SILENT_PROFILE_CACHE[key] = (signature, dict(profile))
        if len(_SILENT_PROFILE_CACHE) > 2000:
            _SILENT_PROFILE_CACHE.pop(next(iter(_SILENT_PROFILE_CACHE)))
    return profiles


def _current_feature_rows(
    prepared: Mapping[str, pd.DataFrame],
    config: SelectorConfig,
    sector_map: Mapping[str, Any] | None = None,
    feature_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 20:
            continue
        features = (feature_frames or {}).get(str(ticker).upper())
        if features is None:
            features = _cached_technical_features(ticker, frame)
        if features.empty:
            continue
        current = features.iloc[-1]
        record = {
            "ticker": str(ticker).upper(),
            "as_of": pd.Timestamp(features.index[-1]).normalize(),
            "last_price": _finite(current.get("close"), np.nan),
            "sector": _text(
                (sector_map or {}).get(
                    str(ticker).upper(),
                    (sector_map or {}).get(str(ticker), ""),
                )
            ),
        }
        record.update({name: _finite(current.get(name), np.nan) for name in FEATURE_COLUMNS})
        bucket, market_impact = _liquidity_cost_from_log_adtv(
            current.get("log_adtv20"), config,
        )
        record["liquidity_bucket"] = bucket
        record["estimated_market_impact_cost_pct"] = market_impact
        record["estimated_total_cost_pct"] = (
            max(0.0, float(config.roundtrip_cost_pct)) + market_impact
        )
        rows.append(record)
    return _score_cross_section(
        pd.DataFrame(rows),
        min_feature_coverage_pct=config.min_feature_coverage_pct,
    )


def _reason_fields(row: Mapping[str, Any]) -> dict[str, str]:
    positives: list[str] = []
    risks: list[str] = []
    codes: list[str] = []
    if _finite(row.get("trend_score"), 0.0) >= 70.0:
        positives.append(f"trend {float(row.get('trend_score')):.0f}/100")
        codes.append("STRONG_TREND")
    else:
        risks.append(f"trend hanya {_finite(row.get('trend_score'), 0.0):.0f}/100")
    if _finite(row.get("relative_strength_score"), 0.0) >= 70.0:
        positives.append(f"relative strength persentil {_finite(row.get('relative_strength_score')):.0f}")
        codes.append("LEADING_RELATIVE_STRENGTH")
    elif _finite(row.get("relative_strength60"), 0.0) <= 0.0:
        risks.append("relative strength 60D belum mengungguli IHSG")
    silent = _finite(row.get("silent_accumulation_score"), 0.0)
    if silent >= 70.0:
        positives.append(f"Silent Accumulation {silent:.0f}/100")
        codes.append("SILENT_ACCUMULATION_STRONG")
    elif silent < 45.0:
        risks.append(f"Silent Accumulation lemah {silent:.0f}/100")
    momentum_quality = _finite(row.get("momentum_quality_score"), 0.0)
    if momentum_quality >= 70.0:
        positives.append(f"momentum konsisten {momentum_quality:.0f}/100")
        codes.append("SMOOTH_PERSISTENT_MOMENTUM")
    elif momentum_quality < 40.0:
        risks.append(f"kualitas momentum rendah {momentum_quality:.0f}/100")
    jump_concentration = _finite(row.get("jump_concentration20"), np.nan)
    if np.isfinite(jump_concentration) and jump_concentration > 0.45:
        risks.append("momentum terlalu bergantung pada satu lonjakan harian")
        codes.append("JUMP_DOMINATED_MOMENTUM")
    distance = _finite(row.get("distance_ema20_atr"), np.nan)
    if np.isfinite(distance) and distance > 1.75:
        risks.append(f"harga extended {distance:.2f} ATR di atas EMA20")
        codes.append("EXTENDED_FROM_EMA20")
    state = _text(row.get("selector_model_state"))
    if state == "PROMOTED_AI_CHAMPION":
        positives.append("AI selector lolos champion gate")
        codes.append("AI_SELECTOR_PROMOTED")
    else:
        risks.append(f"AI selector {state.lower().replace('_', ' ')}")
    if not positives:
        positives.append("belum ada keunggulan seleksi yang kuat")
    if not risks:
        risks.append("tidak ada risiko teknikal utama yang terdeteksi")
    return {
        "selected_reason": " • ".join(positives),
        "selection_risks": " • ".join(risks),
        "selection_reason_codes": " | ".join(dict.fromkeys(codes)) or "NO_STRONG_SELECTION_EDGE",
    }


def build_cross_sectional_selector(
    prepared: Mapping[str, pd.DataFrame],
    config: SelectorConfig | None = None,
    silent_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    sector_map: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return current stock selector, challenger audit, and training panel."""
    cfg = config or SelectorConfig()
    feature_frames: dict[str, pd.DataFrame] = {}
    for ticker, frame in prepared.items():
        if frame is None or frame.empty:
            continue
        key = str(ticker).upper()
        feature_frames[key] = _cached_technical_features(ticker, frame)
    current = _current_feature_rows(
        prepared, cfg, sector_map, feature_frames=feature_frames,
    )
    panel = build_selector_panel(
        prepared, cfg, sector_map, feature_frames=feature_frames,
    )
    audit, fitted = evaluate_selector_challengers(panel, cfg)
    if current.empty:
        return current, audit, panel
    profiles = silent_profiles or {}
    current["silent_accumulation_score"] = [
        _finite(profiles.get(str(ticker).upper(), {}).get("silent_accumulation_score"), 50.0)
        for ticker in current["ticker"]
    ]
    current["silent_accumulation_state"] = [
        _text(profiles.get(str(ticker).upper(), {}).get("silent_accumulation_state")) or "UNAVAILABLE"
        for ticker in current["ticker"]
    ]
    current["silent_accumulation_confidence"] = [
        _finite(
            profiles.get(str(ticker).upper(), {}).get(
                "silent_accumulation_confidence"
            ),
            0.0,
        )
        for ticker in current["ticker"]
    ]
    raw_silent = pd.to_numeric(
        current["silent_accumulation_score"], errors="coerce",
    ).fillna(50.0).clip(0.0, 100.0)
    silent_weight = (
        pd.to_numeric(
            current["silent_accumulation_confidence"], errors="coerce",
        ).fillna(0.0).clip(0.0, 100.0) / 100.0
    )
    effective_silent = 50.0 + silent_weight * (raw_silent - 50.0)
    distribution = current["silent_accumulation_state"].astype(str).str.upper().eq(
        "DISTRIBUTION_RISK"
    )
    current["effective_silent_accumulation_score"] = (
        effective_silent.where(~distribution, effective_silent.clip(upper=25.0))
        .clip(0.0, 100.0)
        .round(1)
    )

    for horizon in cfg.horizons:
        meta = fitted.get(int(horizon), {})
        ridge = meta.get("ridge")
        logistic = meta.get("logistic")
        platt = meta.get("platt", _PlattModel())
        expected = ridge.predict(current) if ridge is not None else np.full(len(current), np.nan)
        probability = (
            platt.predict(logistic.predict(current))
            if logistic is not None else np.full(len(current), np.nan)
        )
        ai_alpha_score = _percentile(pd.Series(expected, index=current.index))
        ai_weight = float(meta.get("ai_weight", 0.0))
        base = pd.to_numeric(current["independent_selector_score"], errors="coerce")
        horizon_score = ((1.0 - ai_weight) * base + ai_weight * ai_alpha_score).where(base.notna())
        current[f"selector_expected_excess_return_{horizon}d_pct"] = np.round(100.0 * expected, 3)
        current[f"selector_outperform_probability_{horizon}d_pct"] = np.round(100.0 * probability, 1)
        current[f"selector_score_{horizon}d"] = np.round(horizon_score, 1)
        current[f"selector_ai_weight_{horizon}d_pct"] = round(100.0 * ai_weight, 1)
        current[f"selector_model_state_{horizon}d"] = meta.get("promotion_state", "INSUFFICIENT_EVIDENCE")
        current[f"selector_champion_{horizon}d"] = meta.get("best_model", "INDEPENDENT_SELECTOR")

    model_states = [
        fitted.get(int(horizon), {}).get("promotion_state", "INSUFFICIENT_EVIDENCE")
        for horizon in cfg.horizons
    ]
    current["selector_model_state"] = (
        "PROMOTED_AI_CHAMPION"
        if any(state == "PROMOTED_AI_CHAMPION" for state in model_states)
        else "SHADOW_CHALLENGER"
        if any(state != "INSUFFICIENT_EVIDENCE" for state in model_states)
        else "INSUFFICIENT_EVIDENCE"
    )
    selector_5 = pd.to_numeric(current.get("selector_score_5d"), errors="coerce").fillna(current["independent_selector_score"])
    selector_20 = pd.to_numeric(current.get("selector_score_20d"), errors="coerce").fillna(current["independent_selector_score"])
    selector_60 = pd.to_numeric(current.get("selector_score_60d"), errors="coerce").fillna(current["independent_selector_score"])
    horizon_swing = 0.40 * selector_5 + 0.40 * selector_20 + 0.20 * selector_60
    horizon_long = 0.35 * selector_20 + 0.65 * selector_60
    technical = (
        0.32 * pd.to_numeric(current["trend_score"], errors="coerce").fillna(50.0)
        + 0.16 * pd.to_numeric(current["relative_strength_score"], errors="coerce").fillna(50.0)
        + 0.08 * pd.to_numeric(
            current["sector_relative_strength_score"], errors="coerce",
        ).fillna(50.0)
        + 0.20 * pd.to_numeric(current["momentum_quality_score"], errors="coerce").fillna(50.0)
        + 0.14 * pd.to_numeric(current["flow_score"], errors="coerce").fillna(50.0)
        + 0.10 * pd.to_numeric(current["entry_geometry_score"], errors="coerce").fillna(50.0)
    )
    silent = pd.to_numeric(
        current["effective_silent_accumulation_score"], errors="coerce",
    ).fillna(50.0)
    swing_quality = (
        0.45 * pd.to_numeric(current["momentum_continuity_score"], errors="coerce").fillna(50.0)
        + 0.30 * pd.to_numeric(current["high_52w_proximity_score"], errors="coerce").fillna(50.0)
        + 0.25 * pd.to_numeric(current["momentum_consistency_score"], errors="coerce").fillna(50.0)
    )
    current["technical_selection_score"] = np.round(technical, 1)
    current["swing_momentum_quality_score"] = np.round(swing_quality, 1)
    absolute_swing = (
        0.42 * pd.to_numeric(current["absolute_selector_score"], errors="coerce")
        + 0.20 * pd.to_numeric(current["absolute_momentum_score"], errors="coerce")
        + 0.16 * pd.to_numeric(current["absolute_flow_score"], errors="coerce")
        + 0.10 * pd.to_numeric(current["entry_geometry_score"], errors="coerce")
        + 0.07 * pd.to_numeric(current["absolute_liquidity_score"], errors="coerce")
        + 0.05 * silent
    )
    relative_swing_overlay = 0.44 * horizon_swing + 0.44 * technical + 0.12 * swing_quality
    relative_weight = pd.to_numeric(current["relative_overlay_weight_pct"], errors="coerce").fillna(0.0).clip(0.0, 10.0) / 100.0
    current["absolute_swing_score"] = np.round(absolute_swing, 1)
    current["relative_swing_overlay_score"] = np.round(relative_swing_overlay, 1)
    current["swing_selection_score"] = np.round(
        (1.0 - relative_weight) * absolute_swing + relative_weight * relative_swing_overlay,
        1,
    )
    current["multibagger_timing_selector_score"] = np.round(
        0.60 * horizon_long + 0.40 * technical,
        1,
    )
    feature_confidence = (
        pd.to_numeric(
            current["technical_feature_coverage_pct"], errors="coerce",
        ).fillna(0.0).clip(0.0, 100.0) / 100.0
    )
    for column in (
        "technical_selection_score",
        "swing_momentum_quality_score",
        "absolute_swing_score",
        "relative_swing_overlay_score",
        "swing_selection_score",
        "multibagger_timing_selector_score",
    ):
        value = pd.to_numeric(current[column], errors="coerce")
        current[column] = (
            50.0 + feature_confidence * (value - 50.0)
        ).where(value.notna()).clip(0.0, 100.0).round(1)
    reason_rows = [_reason_fields(row) for row in current.to_dict("records")]
    reason_frame = pd.DataFrame(reason_rows, index=current.index)
    for column in reason_frame:
        current[column] = reason_frame[column]
    current["_selector_eligible_sort"] = current[
        "selector_rank_eligible"
    ].fillna(False).astype(bool)
    current = current.sort_values(
        [
            "_selector_eligible_sort", "swing_selection_score",
            "effective_silent_accumulation_score",
            "relative_strength_score", "sector_relative_strength_score",
            "log_adtv20",
        ],
        ascending=[False, False, False, False, False, False],
        kind="stable",
    ).drop(columns="_selector_eligible_sort").reset_index(drop=True)
    current["selection_rank"] = np.arange(1, len(current) + 1)
    current["production_selection_rank"] = 0
    eligible_index = current.index[
        current["selector_rank_eligible"].fillna(False).astype(bool)
    ]
    current.loc[eligible_index, "production_selection_rank"] = np.arange(
        1, len(eligible_index) + 1,
    )
    current["production_selection_rank"] = current[
        "production_selection_rank"
    ].astype(int)
    current["selector_version"] = SELECTOR_VERSION
    return current, audit, panel


def attach_setups_to_selector(
    selector: pd.DataFrame,
    core_signals: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach the best setup after technical selection without hiding no-setup names."""
    if selector is None or selector.empty:
        return pd.DataFrame() if selector is None else selector.copy()
    out = selector.copy()
    signals = core_signals.copy() if isinstance(core_signals, pd.DataFrame) else pd.DataFrame()
    if not signals.empty and "ticker" in signals:
        status_order = {
            "EXECUTION_READY": 0,
            "READY_FOR_STOCKBIT_VERIFY": 1,
            "SIGNAL_READY": 2,
            "ENTRY_PLAN_READY": 3,
            "READY_FOR_PRICE_VERIFY": 4,
            "WATCHLIST_ENTRY": 5,
            "REJECT": 9,
        }
        signals["_setup_status_rank"] = signals.get(
            "status", pd.Series("REJECT", index=signals.index),
        ).map(status_order).fillna(8)
        setup_score_source = signals.get(
            "analyst_fusion_score",
            signals.get(
                "quality_score", pd.Series(0.0, index=signals.index),
            ),
        )
        signals["_setup_score"] = pd.to_numeric(
            setup_score_source, errors="coerce",
        ).fillna(0.0)
        best = signals.sort_values(
            ["_setup_status_rank", "_setup_score"],
            ascending=[True, False],
            kind="stable",
        ).drop_duplicates("ticker", keep="first")
        columns = [
            "ticker", "setup", "status", "action", "entry_type",
            "entry_low", "entry_high", "entry", "trigger",
            "stockbit_trigger_price", "stockbit_limit_price",
            "stockbit_order_price", "stop_loss", "tp1", "tp2", "rr1", "rr2",
            "entry_plan_min_rr1", "entry_plan_min_rr2",
            "blockers", "reason", "quality_score", "analyst_fusion_score",
            "autopilot_verified", "order_instruction", "execution_timing",
            "best_buy_date", "best_buy_window_start", "best_buy_window_end",
        ]
        best = best.loc[:, [column for column in columns if column in best]]
        best = best.rename(columns={
            "setup": "active_setup",
            "status": "setup_status",
            "action": "setup_action",
            "reason": "setup_reason",
            "blockers": "setup_blockers",
        })
        out = out.merge(best, on="ticker", how="left")
    for column, default in (
        ("active_setup", "NO_SETUP"),
        ("setup_status", "NO_SETUP"),
        ("setup_action", "WAIT_SETUP"),
        ("setup_reason", ""),
        ("setup_blockers", ""),
    ):
        if column not in out:
            out[column] = default
        out[column] = out[column].fillna(default)
    out["setup_detected"] = out["active_setup"].ne("NO_SETUP")
    out["not_entry_reason"] = np.where(
        out["setup_detected"],
        out["setup_blockers"].where(out["setup_blockers"].astype(str).str.len().gt(0), "Setup belum melewati execution gate"),
        "Belum ada setup valid; tetap di radar seleksi",
    )
    trigger = pd.to_numeric(
        out.get("stockbit_trigger_price", out.get("trigger", pd.Series(np.nan, index=out.index))),
        errors="coerce",
    )
    out["trigger_waiting"] = [
        f"Tunggu trigger di {value:,.0f}" if np.isfinite(value) else "Tunggu setup dan trigger terkonfirmasi"
        for value in trigger
    ]
    stop = pd.to_numeric(out.get("stop_loss", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["invalidation_reason"] = [
        f"Invalid bila penutupan/struktur menembus SL {value:,.0f}" if np.isfinite(value) else "Invalidation belum tersedia sebelum setup valid"
        for value in stop
    ]
    selection_risks = out.get(
        "selection_risks", pd.Series("", index=out.index),
    )
    out["primary_risk"] = [
        (_text(value).split(" • ")[0] if _text(value) else _text(selection_risk).split(" • ")[0])
        for value, selection_risk in zip(out["setup_blockers"], selection_risks)
    ]
    setup_rank = out["setup_status"].map({
        "EXECUTION_READY": 0,
        "READY_FOR_STOCKBIT_VERIFY": 1,
        "SIGNAL_READY": 2,
        "ENTRY_PLAN_READY": 3,
        "READY_FOR_PRICE_VERIFY": 4,
        "WATCHLIST_ENTRY": 5,
        "NO_SETUP": 8,
        "REJECT": 9,
    }).fillna(8)
    out["_setup_rank"] = setup_rank
    out["_selector_eligible_sort"] = out.get(
        "selector_rank_eligible", pd.Series(True, index=out.index),
    ).fillna(False).astype(bool)
    if "effective_silent_accumulation_score" not in out:
        out["effective_silent_accumulation_score"] = pd.to_numeric(
            out.get(
                "silent_accumulation_score",
                pd.Series(50.0, index=out.index),
            ),
            errors="coerce",
        ).fillna(50.0)
    out = out.sort_values(
        [
            "_selector_eligible_sort", "swing_selection_score",
            "effective_silent_accumulation_score",
            "technical_selection_score", "_setup_rank",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).drop(columns=["_setup_rank", "_selector_eligible_sort"]).reset_index(drop=True)
    out["swing_selection_rank"] = np.arange(1, len(out) + 1)
    executable = (
        out.get(
            "selector_rank_eligible", pd.Series(True, index=out.index),
        ).fillna(False).astype(bool)
        & out["setup_detected"]
        & out["setup_status"].isin({
        "EXECUTION_READY", "READY_FOR_STOCKBIT_VERIFY", "SIGNAL_READY",
        "ENTRY_PLAN_READY", "READY_FOR_PRICE_VERIFY",
        })
    )
    out["execution_rank"] = np.nan
    if executable.any():
        order = out.loc[executable].sort_values(
            [
                "setup_status", "swing_selection_score",
                "effective_silent_accumulation_score",
            ],
            ascending=[True, False, False],
            kind="stable",
        ).index
        out.loc[order, "execution_rank"] = np.arange(1, len(order) + 1)
    return out


def selector_snapshot_frame(selector: pd.DataFrame | None) -> pd.DataFrame:
    if selector is None or selector.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in selector.iterrows():
        for horizon in SELECTOR_HORIZONS:
            ticker = _text(row.get("ticker")).upper()
            as_of = pd.to_datetime(row.get("as_of"), errors="coerce")
            if not ticker or pd.isna(as_of):
                continue
            snapshot_id = sha256(
                f"{ticker}|{pd.Timestamp(as_of).date()}|{horizon}|{SELECTOR_VERSION}".encode("utf-8")
            ).hexdigest()[:32]
            rows.append({
                "snapshot_id": snapshot_id,
                "ticker": ticker,
                "as_of": pd.Timestamp(as_of),
                "horizon": f"{horizon}D",
                "horizon_bars": horizon,
                "selection_rank": row.get("selection_rank"),
                "production_selection_rank": row.get(
                    "production_selection_rank"
                ),
                "selector_rank_eligible": row.get(
                    "selector_rank_eligible"
                ),
                "selector_data_state": row.get("selector_data_state"),
                "technical_feature_coverage_pct": row.get(
                    "technical_feature_coverage_pct"
                ),
                "selector_missing_feature_count": row.get(
                    "selector_missing_feature_count"
                ),
                "selector_missing_features": row.get(
                    "selector_missing_features"
                ),
                "swing_selection_score": row.get("swing_selection_score"),
                "multibagger_timing_selector_score": row.get("multibagger_timing_selector_score"),
                "technical_selection_score": row.get("technical_selection_score"),
                "silent_accumulation_score": row.get("silent_accumulation_score"),
                "effective_silent_accumulation_score": row.get(
                    "effective_silent_accumulation_score"
                ),
                "silent_accumulation_confidence": row.get(
                    "silent_accumulation_confidence"
                ),
                "relative_strength_score": row.get("relative_strength_score"),
                "sector": row.get("sector"),
                "sector_peer_count": row.get("sector_peer_count"),
                "sector_relative_strength_score": row.get(
                    "sector_relative_strength_score"
                ),
                "liquidity_bucket": row.get("liquidity_bucket"),
                "estimated_market_impact_cost_pct": row.get(
                    "estimated_market_impact_cost_pct"
                ),
                "estimated_total_cost_pct": row.get(
                    "estimated_total_cost_pct"
                ),
                "expected_excess_return_pct": row.get(f"selector_expected_excess_return_{horizon}d_pct"),
                "outperform_probability_pct": row.get(f"selector_outperform_probability_{horizon}d_pct"),
                "selector_score": row.get(f"selector_score_{horizon}d"),
                "ai_weight_pct": row.get(f"selector_ai_weight_{horizon}d_pct"),
                "model_state": row.get(f"selector_model_state_{horizon}d"),
                "champion_model": row.get(f"selector_champion_{horizon}d"),
                "selected_reason": row.get("selected_reason"),
                "selection_risks": row.get("selection_risks"),
                "model_version": SELECTOR_VERSION,
            })
    return pd.DataFrame(rows)


def update_selector_outcomes(
    existing: pd.DataFrame | None,
    selector: pd.DataFrame | None,
    prepared: Mapping[str, pd.DataFrame],
    config: SelectorConfig | None = None,
) -> pd.DataFrame:
    """Register current predictions and resolve prior excess-return outcomes."""
    cfg = config or SelectorConfig()
    memory = existing.copy() if isinstance(existing, pd.DataFrame) else pd.DataFrame()
    snapshots = selector_snapshot_frame(selector)
    if not snapshots.empty:
        snapshots = snapshots[
            pd.to_numeric(snapshots["horizon_bars"], errors="coerce").isin(
                [int(value) for value in cfg.horizons]
            )
        ].copy()
    additions: list[dict[str, Any]] = []
    for _, row in snapshots.iterrows():
        outcome_id = sha256(
            f"{row['snapshot_id']}|{SELECTOR_OUTCOME_VERSION}".encode("utf-8")
        ).hexdigest()[:32]
        additions.append({
            "outcome_id": outcome_id,
            "snapshot_id": row["snapshot_id"],
            "ticker": row["ticker"],
            "signal_date": pd.Timestamp(row["as_of"]).normalize(),
            "horizon": row["horizon"],
            "horizon_bars": int(row["horizon_bars"]),
            "predicted_excess_return_pct": row.get("expected_excess_return_pct"),
            "outperform_probability_pct": row.get("outperform_probability_pct"),
            "selector_score": row.get("selector_score"),
            "estimated_market_impact_cost_pct": row.get(
                "estimated_market_impact_cost_pct"
            ),
            "estimated_total_cost_pct": row.get(
                "estimated_total_cost_pct"
            ),
            "liquidity_bucket": row.get("liquidity_bucket"),
            "model_state": row.get("model_state"),
            "champion_model": row.get("champion_model"),
            "outcome_status": "OPEN",
            "model_version": SELECTOR_VERSION,
        })
    additions_frame = pd.DataFrame(additions)
    if memory.empty:
        memory = additions_frame
    elif not additions_frame.empty:
        # Outcome updates are intentionally idempotent.  Filter records that are
        # already present before concatenating; besides avoiding unnecessary
        # work, this prevents pandas from coercing an all-empty duplicate frame
        # through its deprecated generic timedelta path.
        if "outcome_id" in memory.columns and "outcome_id" in additions_frame.columns:
            known_outcome_ids = set(
                memory["outcome_id"].fillna("").astype(str)
            )
            additions_frame = additions_frame.loc[
                ~additions_frame["outcome_id"].fillna("").astype(str).isin(
                    known_outcome_ids
                )
            ]
        if not additions_frame.empty:
            memory = pd.concat(
                [memory, additions_frame], ignore_index=True, sort=False
            )
            memory = memory.drop_duplicates("outcome_id", keep="first")
    if memory.empty:
        return memory

    resolved: list[dict[str, Any]] = []
    for _, raw_row in memory.iterrows():
        record = raw_row.to_dict()
        if _text(record.get("outcome_status")).upper() == "RESOLVED":
            resolved.append(record)
            continue
        ticker = _text(record.get("ticker")).upper()
        frame = prepared.get(ticker)
        signal_date = pd.to_datetime(record.get("signal_date"), errors="coerce")
        horizon = int(_finite(record.get("horizon_bars"), 0.0))
        if frame is None or frame.empty or pd.isna(signal_date) or horizon <= 0:
            resolved.append(record)
            continue
        local = _normalise_index(frame)
        positions = np.flatnonzero(local.index.normalize() == pd.Timestamp(signal_date).normalize())
        if not len(positions) or positions[-1] + horizon >= len(local):
            resolved.append(record)
            continue
        position = int(positions[-1])
        start = local.iloc[position]
        future = local.iloc[position + horizon]
        stock_return = _finite(future.get("Close"), np.nan) / _finite(start.get("Close"), np.nan) - 1.0
        benchmark_start = _finite(start.get("BENCH_CLOSE"), np.nan)
        benchmark_end = _finite(future.get("BENCH_CLOSE"), np.nan)
        benchmark_return = benchmark_end / benchmark_start - 1.0 if benchmark_start > 0 and benchmark_end > 0 else np.nan
        stored_cost = _finite(
            record.get("estimated_total_cost_pct"), np.nan,
        )
        total_cost = (
            stored_cost
            if np.isfinite(stored_cost)
            else max(0.0, float(cfg.roundtrip_cost_pct))
        )
        net_excess = (
            stock_return - benchmark_return - total_cost
            if np.isfinite(stock_return) and np.isfinite(benchmark_return) else np.nan
        )
        record.update({
            "outcome_status": "RESOLVED",
            "resolved_at": pd.Timestamp(local.index[-1]),
            "stock_return_pct": round(100.0 * stock_return, 4) if np.isfinite(stock_return) else np.nan,
            "benchmark_return_pct": round(100.0 * benchmark_return, 4) if np.isfinite(benchmark_return) else np.nan,
            "realised_total_cost_pct": round(100.0 * total_cost, 4),
            "net_excess_return_pct": round(100.0 * net_excess, 4) if np.isfinite(net_excess) else np.nan,
            "outperformed_after_cost": bool(net_excess > 0.0) if np.isfinite(net_excess) else np.nan,
        })
        resolved.append(record)
    return pd.DataFrame(resolved).drop_duplicates(
        "outcome_id", keep="last",
    ).sort_values("signal_date", na_position="last").reset_index(drop=True)


__all__ = [
    "SELECTOR_VERSION",
    "SELECTOR_OUTCOME_VERSION",
    "SELECTOR_HORIZONS",
    "SELECTOR_MODELS",
    "SelectorConfig",
    "build_selector_panel",
    "evaluate_selector_challengers",
    "current_silent_profiles",
    "build_cross_sectional_selector",
    "attach_setups_to_selector",
    "selector_snapshot_frame",
    "update_selector_outcomes",
]

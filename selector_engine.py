"""Point-in-time cross-sectional stock selection for IDX Super Scanner.

The selector answers a different question from the setup detectors:

1. Which stocks have the strongest relative opportunity versus IHSG?
2. Only after selection, is there a valid setup and executable trade plan?

The module is deliberately dependency-light.  It evaluates four frozen
challengers (rule, independent selector, relative-strength baseline, and a
regularised AI challenger) on chronological unseen dates.  The AI challenger
is shadow-only unless it wins after costs, has positive Brier skill and
expectancy, and stays inside a drawdown guard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Mapping
import math

import numpy as np
import pandas as pd


SELECTOR_VERSION = "1.0.0-cross-sectional-excess-return"
SELECTOR_OUTCOME_VERSION = "selector_outcomes_v1"
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
    max_ai_weight: float = 0.30
    min_brier_skill: float = 0.0
    min_net_expectancy_pct: float = 0.0
    max_promotion_drawdown_pct: float = 20.0
    top_fraction: float = 0.20
    min_top_k: int = 3
    max_model_rows: int = 12000

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
    close_location = _series(local, "CLOSE_LOCATION").fillna(0.5)
    volume_ratio = _series(local, "VOL_RATIO")

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
    out["log_adtv20"] = np.log1p(_series(local, "ADTV20").clip(lower=0.0))
    # Historical price-volume proxy.  It is fully point-in-time and does not
    # call the more expensive current Silent Accumulation profile per anchor.
    out["accumulation_proxy"] = np.clip(
        0.50
        + 0.80 * cmf20.fillna(0.0)
        + 0.45 * cmf60.fillna(0.0)
        + 0.12 * np.tanh(5.0 * obv20.fillna(0.0))
        + 0.10 * np.tanh(5.0 * adl20.fillna(0.0))
        + 0.08 * (close_location - 0.5)
        + 0.04 * np.tanh(volume_ratio.fillna(1.0) - 1.0),
        0.0,
        1.0,
    )
    out["close"] = close
    out["benchmark_close"] = _series(local, "BENCH_CLOSE")
    out["net_stock_return_placeholder"] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.reindex(columns=FEATURE_COLUMNS).apply(
        pd.to_numeric, errors="coerce",
    ).to_numpy(dtype=float)


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


def _score_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    out = frame.copy()
    groups = out.groupby("as_of", sort=False) if "as_of" in out else [(None, out)]
    scored: list[pd.DataFrame] = []
    for _, local in groups:
        local = local.copy()
        trend = (
            0.27 * _percentile(local["close_ema20"])
            + 0.25 * _percentile(local["ema20_ema50"])
            + 0.22 * _percentile(local["ema50_ema200"])
            + 0.14 * _percentile(local["ema20_slope10"])
            + 0.12 * _percentile(local["adx14_scaled"])
        )
        momentum = (
            _percentile(local["roc20"])
            + _percentile(local["roc60"])
            + _percentile(local["roc120"])
        ) / 3.0
        relative = _percentile(local["relative_strength60"])
        flow = (
            0.28 * _percentile(local["cmf20"])
            + 0.22 * _percentile(local["cmf60"])
            + 0.20 * _percentile(local["obv_slope20"])
            + 0.15 * _percentile(local["adl_slope20"])
            + 0.15 * _percentile(local["accumulation_proxy"])
        )
        distance = pd.to_numeric(local["distance_ema20_atr"], errors="coerce")
        atr_pct = pd.to_numeric(local["atr_pct"], errors="coerce")
        extension = (100.0 - 32.0 * (distance - 0.75).clip(lower=0.0)).clip(0.0, 100.0)
        structure = (
            45.0
            + 700.0 * pd.to_numeric(local["close_ema20"], errors="coerce").fillna(0.0)
            + 700.0 * pd.to_numeric(local["ema20_ema50"], errors="coerce").fillna(0.0)
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
        entry = (extension + structure + pd.Series(volatility, index=local.index)) / 3.0
        liquidity = _percentile(local["log_adtv20"])
        absolute_rule = (
            35.0
            + 350.0 * pd.to_numeric(local["close_ema20"], errors="coerce").fillna(0.0)
            + 300.0 * pd.to_numeric(local["ema20_ema50"], errors="coerce").fillna(0.0)
            + 250.0 * pd.to_numeric(local["ema50_ema200"], errors="coerce").fillna(0.0)
            + 45.0 * pd.to_numeric(local["relative_strength60"], errors="coerce").fillna(0.0)
            + 20.0 * pd.to_numeric(local["cmf20"], errors="coerce").fillna(0.0)
        ).clip(0.0, 100.0)
        local["trend_score"] = trend.round(2)
        local["momentum_score"] = momentum.round(2)
        local["relative_strength_score"] = relative.round(2)
        local["flow_score"] = flow.round(2)
        local["entry_geometry_score"] = pd.to_numeric(entry, errors="coerce").fillna(50.0).round(2)
        local["liquidity_score"] = liquidity.round(2)
        local["rule_engine_score"] = pd.to_numeric(absolute_rule, errors="coerce").fillna(50.0).round(2)
        local["independent_selector_score"] = (
            0.30 * trend
            + 0.20 * momentum
            + 0.20 * relative
            + 0.15 * flow
            + 0.10 * pd.to_numeric(entry, errors="coerce").fillna(50.0)
            + 0.05 * liquidity
        ).round(2)
        scored.append(local)
    return pd.concat(scored, ignore_index=True, sort=False)


def build_selector_panel(
    prepared: Mapping[str, pd.DataFrame],
    config: SelectorConfig | None = None,
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
        "cross": cfg.min_cross_section,
    }))
    cache_key = sha256("|".join(signature_parts).encode("utf-8")).hexdigest()
    cached_panel = _PANEL_CACHE.get(cache_key)
    if cached_panel is not None:
        return cached_panel.copy()
    maximum_horizon = max(int(value) for value in cfg.horizons)
    rows: list[dict[str, Any]] = []
    for ticker, raw in prepared.items():
        if raw is None or raw.empty or len(raw) < cfg.min_history_bars + maximum_horizon:
            continue
        features = _cached_technical_features(ticker, raw)
        start = max(
            cfg.min_history_bars - 1,
            len(features) - int(cfg.training_lookback_bars) - maximum_horizon,
        )
        stop = len(features) - maximum_horizon
        for position in range(start, stop, max(1, int(cfg.anchor_step_bars))):
            current = features.iloc[position]
            if not np.isfinite(_finite(current.get("close"), np.nan)):
                continue
            record: dict[str, Any] = {
                "ticker": str(ticker).upper(),
                "as_of": pd.Timestamp(features.index[position]).normalize(),
            }
            record.update({name: _finite(current.get(name), np.nan) for name in FEATURE_COLUMNS})
            close = _finite(current.get("close"), np.nan)
            benchmark_close = _finite(current.get("benchmark_close"), np.nan)
            for horizon in cfg.horizons:
                future = features.iloc[position + int(horizon)]
                stock_return = (
                    _finite(future.get("close"), np.nan) / close - 1.0
                    if close > 0 else np.nan
                )
                future_benchmark = _finite(future.get("benchmark_close"), np.nan)
                benchmark_return = (
                    future_benchmark / benchmark_close - 1.0
                    if benchmark_close > 0 and future_benchmark > 0 else np.nan
                )
                net_stock = stock_return - cfg.roundtrip_cost_pct if np.isfinite(stock_return) else np.nan
                net_excess = (
                    stock_return - benchmark_return - cfg.roundtrip_cost_pct
                    if np.isfinite(stock_return) and np.isfinite(benchmark_return) else np.nan
                )
                record[f"stock_return_{horizon}d"] = stock_return
                record[f"benchmark_return_{horizon}d"] = benchmark_return
                record[f"net_stock_return_{horizon}d"] = net_stock
                record[f"net_excess_return_{horizon}d"] = net_excess
                record[f"outperform_{horizon}d"] = (
                    float(net_excess > 0.0) if np.isfinite(net_excess) else np.nan
                )
            rows.append(record)
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    counts = panel.groupby("as_of")["ticker"].transform("nunique")
    panel = panel[counts >= max(2, int(cfg.min_cross_section))].copy()
    result = _score_cross_section(panel).sort_values(
        ["as_of", "ticker"], kind="stable",
    ).reset_index(drop=True)
    _PANEL_CACHE[cache_key] = result.copy()
    if len(_PANEL_CACHE) > 2:
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


def _spearman(left: pd.Series, right: pd.Series) -> float:
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan
    return float(x[valid].corr(y[valid], method="spearman"))


def _max_drawdown(returns: list[float]) -> float:
    if not returns:
        return np.nan
    equity = np.cumprod(1.0 + np.clip(np.asarray(returns, dtype=float), -0.95, 3.0))
    peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / np.where(peak > 0, peak, 1.0)
    return float(100.0 * np.nanmax(drawdown)) if len(drawdown) else np.nan


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
    local = frame.copy()
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
    selected_dates = set(unique_dates[::stride])
    top_returns: list[float] = []
    top_excess: list[float] = []
    hit_values: list[float] = []
    correlations: list[float] = []
    for date_value, group in local.groupby("as_of", sort=True):
        if pd.Timestamp(date_value).to_datetime64() not in selected_dates:
            continue
        eligible = group.dropna(subset=["_score", "_target"]).sort_values("_score", ascending=False)
        if eligible.empty:
            continue
        top_k = max(config.min_top_k, int(math.ceil(len(eligible) * config.top_fraction)))
        top = eligible.head(min(len(eligible), top_k))
        top_excess.append(float(top["_target"].mean()))
        top_returns.append(float(top["_stock"].mean()))
        hit_values.append(float(top["_binary"].mean()))
        correlations.append(_spearman(eligible["_score"], eligible["_target"]))
    return {
        "evaluation_rows": int(local["_target"].notna().sum()),
        "evaluation_dates": int(len(top_excess)),
        "evaluation_tickers": int(local.loc[local["_target"].notna(), "ticker"].nunique()),
        "brier_score": round(brier, 6) if np.isfinite(brier) else np.nan,
        "baseline_brier_score": round(baseline, 6) if np.isfinite(baseline) else np.nan,
        "brier_skill_pct": round(100.0 * brier_skill, 2) if np.isfinite(brier_skill) else np.nan,
        "net_excess_expectancy_pct": round(100.0 * float(np.mean(top_excess)), 3) if top_excess else np.nan,
        "net_absolute_expectancy_pct": round(100.0 * float(np.mean(top_returns)), 3) if top_returns else np.nan,
        "topk_hit_rate_pct": round(100.0 * float(np.mean(hit_values)), 1) if hit_values else np.nan,
        "spearman_ic": round(float(np.nanmean(correlations)), 4) if any(np.isfinite(correlations)) else np.nan,
        "max_drawdown_pct": round(_max_drawdown(top_returns), 2),
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

    for horizon in cfg.horizons:
        labelled = panel.dropna(subset=[
            f"net_excess_return_{horizon}d",
            f"outperform_{horizon}d",
        ]).copy()
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
        sufficient = bool(
            ai_row["training_rows"] >= cfg.min_training_rows
            and ai_row["evaluation_rows"] >= cfg.min_evaluation_rows
            and ai_row["evaluation_dates"] >= cfg.min_evaluation_dates
            and ai_row["evaluation_tickers"] >= cfg.min_evaluation_tickers
            and ridge is not None and logistic is not None
        )
        ai_expectancy = _finite(ai_row.get("net_excess_expectancy_pct"), -np.inf)
        best_baseline_expectancy = max(
            (_finite(row.get("net_excess_expectancy_pct"), -np.inf) for row in baseline_rows),
            default=-np.inf,
        )
        promoted = bool(
            sufficient
            and _finite(ai_row.get("brier_skill_pct"), -np.inf) > 100.0 * cfg.min_brier_skill
            and ai_expectancy > cfg.min_net_expectancy_pct
            and _finite(ai_row.get("max_drawdown_pct"), np.inf) <= cfg.max_promotion_drawdown_pct
            and ai_expectancy > best_baseline_expectancy
        )
        promotion_state = (
            "PROMOTED_AI_CHAMPION"
            if promoted else
            "INSUFFICIENT_EVIDENCE"
            if not sufficient else
            "REJECTED_NON_POSITIVE_BRIER"
            if _finite(ai_row.get("brier_skill_pct"), -np.inf) <= 100.0 * cfg.min_brier_skill else
            "REJECTED_NON_POSITIVE_EXPECTANCY"
            if ai_expectancy <= cfg.min_net_expectancy_pct else
            "REJECTED_DRAWDOWN"
            if _finite(ai_row.get("max_drawdown_pct"), np.inf) > cfg.max_promotion_drawdown_pct else
            "REJECTED_DID_NOT_BEAT_BASELINES"
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
            # The independent selector is the frozen production baseline.
            # A statistically interesting but insufficient AI result remains
            # a challenger and must never be labelled as promoted.
            row["promoted_model"] = (
                "AI_CHALLENGER" if promoted else "INDEPENDENT_SELECTOR"
            )
            row["ai_promotion_state"] = promotion_state
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
        }
    return pd.DataFrame(audit_rows), fitted


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


def _current_feature_rows(prepared: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 20:
            continue
        features = _cached_technical_features(ticker, frame)
        if features.empty:
            continue
        current = features.iloc[-1]
        record = {
            "ticker": str(ticker).upper(),
            "as_of": pd.Timestamp(features.index[-1]).normalize(),
            "last_price": _finite(current.get("close"), np.nan),
        }
        record.update({name: _finite(current.get(name), np.nan) for name in FEATURE_COLUMNS})
        rows.append(record)
    return _score_cross_section(pd.DataFrame(rows))


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return current stock selector, challenger audit, and training panel."""
    cfg = config or SelectorConfig()
    current = _current_feature_rows(prepared)
    panel = build_selector_panel(prepared, cfg)
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
        base = pd.to_numeric(current["independent_selector_score"], errors="coerce").fillna(50.0)
        horizon_score = (1.0 - ai_weight) * base + ai_weight * ai_alpha_score
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
        0.38 * pd.to_numeric(current["trend_score"], errors="coerce").fillna(50.0)
        + 0.24 * pd.to_numeric(current["relative_strength_score"], errors="coerce").fillna(50.0)
        + 0.18 * pd.to_numeric(current["momentum_score"], errors="coerce").fillna(50.0)
        + 0.12 * pd.to_numeric(current["flow_score"], errors="coerce").fillna(50.0)
        + 0.08 * pd.to_numeric(current["entry_geometry_score"], errors="coerce").fillna(50.0)
    )
    silent = pd.to_numeric(current["silent_accumulation_score"], errors="coerce").fillna(50.0)
    current["technical_selection_score"] = np.round(technical, 1)
    current["swing_selection_score"] = np.round(
        0.55 * horizon_swing + 0.25 * technical + 0.20 * silent,
        1,
    )
    current["multibagger_timing_selector_score"] = np.round(
        0.55 * horizon_long + 0.25 * silent + 0.20 * technical,
        1,
    )
    reason_rows = [_reason_fields(row) for row in current.to_dict("records")]
    reason_frame = pd.DataFrame(reason_rows, index=current.index)
    for column in reason_frame:
        current[column] = reason_frame[column]
    current = current.sort_values(
        ["swing_selection_score", "silent_accumulation_score", "relative_strength_score", "log_adtv20"],
        ascending=[False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    current["selection_rank"] = np.arange(1, len(current) + 1)
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
            "ticker", "setup", "status", "action", "entry", "trigger",
            "stockbit_trigger_price", "stop_loss", "tp1", "tp2", "rr1", "rr2",
            "blockers", "reason", "quality_score", "analyst_fusion_score",
            "autopilot_verified", "order_instruction", "best_buy_date",
            "best_buy_window_start", "best_buy_window_end",
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
    out["primary_risk"] = [
        (_text(value).split(" • ")[0] if _text(value) else _text(selection_risk).split(" • ")[0])
        for value, selection_risk in zip(out["setup_blockers"], out["selection_risks"])
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
    out = out.sort_values(
        ["swing_selection_score", "silent_accumulation_score", "technical_selection_score", "_setup_rank"],
        ascending=[False, False, False, True],
        kind="stable",
    ).drop(columns="_setup_rank").reset_index(drop=True)
    out["swing_selection_rank"] = np.arange(1, len(out) + 1)
    executable = out["setup_detected"] & out["setup_status"].isin({
        "EXECUTION_READY", "READY_FOR_STOCKBIT_VERIFY", "SIGNAL_READY",
        "ENTRY_PLAN_READY", "READY_FOR_PRICE_VERIFY",
    })
    out["execution_rank"] = np.nan
    if executable.any():
        order = out.loc[executable].sort_values(
            ["setup_status", "swing_selection_score", "silent_accumulation_score"],
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
                "swing_selection_score": row.get("swing_selection_score"),
                "multibagger_timing_selector_score": row.get("multibagger_timing_selector_score"),
                "technical_selection_score": row.get("technical_selection_score"),
                "silent_accumulation_score": row.get("silent_accumulation_score"),
                "relative_strength_score": row.get("relative_strength_score"),
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
            "model_state": row.get("model_state"),
            "champion_model": row.get("champion_model"),
            "outcome_status": "OPEN",
            "model_version": SELECTOR_VERSION,
        })
    additions_frame = pd.DataFrame(additions)
    if memory.empty:
        memory = additions_frame
    elif not additions_frame.empty:
        memory = pd.concat([memory, additions_frame], ignore_index=True, sort=False)
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
        net_excess = (
            stock_return - benchmark_return - cfg.roundtrip_cost_pct
            if np.isfinite(stock_return) and np.isfinite(benchmark_return) else np.nan
        )
        record.update({
            "outcome_status": "RESOLVED",
            "resolved_at": pd.Timestamp(local.index[-1]),
            "stock_return_pct": round(100.0 * stock_return, 4) if np.isfinite(stock_return) else np.nan,
            "benchmark_return_pct": round(100.0 * benchmark_return, 4) if np.isfinite(benchmark_return) else np.nan,
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

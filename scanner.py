"""IDX Super Scanner core engine.

This module contains market-data acquisition, indicators, core setup detection,
validation, execution-state logic, portfolio analytics, and charting. Specialty
intraday scanners live in :mod:`scanner_specialty`.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class MarketContext:
    regime: str = 'UNKNOWN'
    benchmark_close: float | None = None
    benchmark_roc20: float | None = None
    breadth_ema50: float | None = None
    breadth_ema200: float | None = None
    reason: str = 'Benchmark tidak tersedia'

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
    entry_type: str = 'CONDITIONAL'
    trigger: float | None = None
    stop_loss: float | None = None
    stop_basis: str = 'STRUCTURE_RAW'
    stop_structure: str = ''
    stop_structure_valid: bool = False
    stop_distance_atr: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    tp1_basis: str = 'PRICE_STRUCTURE'
    tp2_basis: str = 'PRICE_STRUCTURE'
    target_model: str = 'PRICE_STRUCTURE'
    target_structure: str = ''
    rr1: float | None = None
    rr2: float | None = None
    distance_atr: float | None = None
    zone_age_bars: int | None = None
    valid_until: Any = None
    invalidated: bool = False
    action: str = 'NO_SETUP'
    reason: str = ''
    evidence: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    # Strategy-specific diagnostics. These fields are optional and remain NaN/None
    # for setups that do not use them. Keeping them on the canonical plan makes
    # cross-strategy ranking auditable instead of reconstructing edge quality
    # from a single headline score.
    structural_quality_score: float | None = None
    confirmation_quality_score: float | None = None
    supply_demand_score: float | None = None
    failure_risk_score: float | None = None
    extension_atr: float | None = None
    setup_detail: str = ''

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['evidence'] = ' • '.join(self.evidence)
        result['blockers'] = ' • '.join(self.blockers)
        return result

@dataclass
class DownloadReport:
    requested: list[str]
    downloaded: list[str]
    failed: dict[str, str]
    benchmark_ok: bool = False
    provider: str = 'Yahoo Finance via yfinance'
    adjusted_prices: bool = True
    downloaded_at: Any = None
    warnings: dict[str, str] = field(default_factory=dict)
    source_tiers: dict[str, str] = field(default_factory=dict)
from dataclasses import dataclass
import math

def idx_tick_size(price: float) -> int:
    """Return the IDX regular-market price fraction effective since May 2016."""
    if not math.isfinite(price) or price <= 0:
        return 1
    if price < 200:
        return 1
    if price < 500:
        return 2
    if price < 2000:
        return 5
    if price < 5000:
        return 10
    return 25

def round_idx_price(price: float | None, direction: str='nearest') -> float | None:
    if price is None or not math.isfinite(float(price)) or float(price) <= 0:
        return None
    value = float(price)
    tick = idx_tick_size(value)
    scaled = value / tick
    if direction == 'up':
        rounded = math.ceil(scaled - 1e-12) * tick
    elif direction == 'down':
        rounded = math.floor(scaled + 1e-12) * tick
    else:
        rounded = round(scaled) * tick
    tick2 = idx_tick_size(float(rounded))
    scaled2 = rounded / tick2
    if direction == 'up':
        rounded = math.ceil(scaled2 - 1e-12) * tick2
    elif direction == 'down':
        rounded = math.floor(scaled2 + 1e-12) * tick2
    else:
        rounded = round(scaled2) * tick2
    return float(max(1, rounded))

def idx_ara_pct(reference_price: float) -> float:
    if reference_price <= 200:
        return 0.35
    if reference_price <= 5000:
        return 0.25
    return 0.2

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
    return abs(value / tick - round(value / tick)) <= 1e-09

def idx_daily_price_band(reference_price: float) -> tuple[float | None, float | None]:
    """Conservative regular-board ARB/ARA band rounded to valid IDX ticks."""
    if not math.isfinite(reference_price) or reference_price <= 0:
        return (None, None)
    lower = round_idx_price(reference_price * (1 - idx_arb_pct(reference_price)), 'up')
    upper = round_idx_price(reference_price * (1 + idx_ara_pct(reference_price)), 'down')
    return (lower, upper)

def within_idx_daily_price_band(price: float | None, reference_price: float) -> bool:
    if not is_valid_idx_price(price):
        return False
    lower, upper = idx_daily_price_band(reference_price)
    return bool(lower is not None and upper is not None and (lower <= float(price) <= upper))

def near_upper_auto_rejection(previous_close: float, close: float, high: float) -> bool:
    if previous_close <= 0:
        return False
    daily_return = close / previous_close - 1.0
    locked_at_high = abs(high - close) <= idx_tick_size(close) * 0.51
    return bool(locked_at_high and daily_return >= 0.9 * idx_ara_pct(previous_close))
import numpy as np
import pandas as pd
OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df['Close'].shift(1)
    return pd.concat([df['High'] - df['Low'], (df['High'] - prev).abs(), (df['Low'] - prev).abs()], axis=1).max(axis=1)

def atr(df: pd.DataFrame, length: int=14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

def rsi(close: pd.Series, length: int=14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss.ne(0), 100.0).fillna(50.0)

def adx(df: pd.DataFrame, length: int=14) -> pd.Series:
    up = df['High'].diff()
    down = -df['Low'].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_ = atr(df, length).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean().fillna(0.0)

def cmf(df: pd.DataFrame, length: int=20) -> pd.Series:
    spread = (df['High'] - df['Low']).replace(0, np.nan)
    multiplier = (df['Close'] - df['Low'] - (df['High'] - df['Close'])) / spread
    money_flow = multiplier.fillna(0.0) * df['Volume']
    return money_flow.rolling(length).sum() / df['Volume'].rolling(length).sum().replace(0, np.nan)

def mfi(df: pd.DataFrame, length: int=14) -> pd.Series:
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    raw = typical * df['Volume']
    direction = typical.diff()
    positive = raw.where(direction > 0, 0.0).rolling(length).sum()
    negative = raw.where(direction < 0, 0.0).rolling(length).sum()
    ratio = positive / negative.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50.0)

def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df['Close'].diff()).fillna(0.0)
    return (sign * df['Volume']).cumsum()

def confirmed_pivot(series: pd.Series, left: int=3, right: int=3, mode: str='high') -> pd.Series:
    window = left + right + 1
    roll = series.rolling(window, center=True, min_periods=window)
    raw = series.eq(roll.max()) if mode == 'high' else series.eq(roll.min())
    return series.where(raw).shift(right)

def prepare_indicators(df: pd.DataFrame, benchmark: pd.DataFrame | None=None) -> pd.DataFrame:
    out = df.copy()
    for col in OHLCV:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out = out.dropna(subset=['Open', 'High', 'Low', 'Close']).sort_index()
    out['Volume'] = out['Volume'].fillna(0.0).clip(lower=0)
    for length in (10, 20, 50, 100, 200):
        out[f'EMA{length}'] = ema(out['Close'], length)
    out['ATR14'] = atr(out, 14)
    out['ATR_PCT'] = out['ATR14'] / out['Close'].replace(0, np.nan)
    out['RSI14'] = rsi(out['Close'], 14)
    out['ADX14'] = adx(out, 14)
    macd = ema(out['Close'], 12) - ema(out['Close'], 26)
    signal = ema(macd, 9)
    out['MACD'] = macd
    out['MACD_HIST'] = macd - signal
    out['CMF20'] = cmf(out, 20)
    out['MFI14'] = mfi(out, 14)
    out['OBV'] = obv(out)
    out['OBV_SLOPE10'] = out['OBV'].diff(10) / out['Volume'].rolling(20).mean().replace(0, np.nan)
    out['VOL_MA20'] = out['Volume'].rolling(20).mean()
    out['VOL_RATIO'] = out['Volume'] / out['VOL_MA20'].replace(0, np.nan)
    out['VALUE'] = out['Close'] * out['Volume']
    out['ADTV20'] = out['VALUE'].rolling(20).mean()
    out['ZERO_VOL20'] = out['Volume'].eq(0).rolling(20).mean()
    typical = (out['High'] + out['Low'] + out['Close']) / 3
    out['VWAP20'] = (typical * out['Volume']).rolling(20).sum() / out['Volume'].rolling(20).sum().replace(0, np.nan)
    for length in (20, 60, 120):
        out[f'ROC{length}'] = out['Close'].pct_change(length)
    out['HIGH20_PREV'] = out['High'].shift(1).rolling(20).max()
    out['HIGH55_PREV'] = out['High'].shift(1).rolling(55).max()
    out['HIGH252'] = out['High'].rolling(252, min_periods=120).max()
    out['LOW20_PREV'] = out['Low'].shift(1).rolling(20).min()
    out['LOW55_PREV'] = out['Low'].shift(1).rolling(55).min()
    out['DIST_52W_HIGH'] = out['Close'] / out['HIGH252'].replace(0, np.nan) - 1
    out['PIVOT_HIGH'] = confirmed_pivot(out['High'], 3, 3, 'high')
    out['PIVOT_LOW'] = confirmed_pivot(out['Low'], 3, 3, 'low')
    out['LAST_PIVOT_HIGH'] = out['PIVOT_HIGH'].ffill()
    out['LAST_PIVOT_LOW'] = out['PIVOT_LOW'].ffill()
    body = (out['Close'] - out['Open']).abs()
    candle_range = (out['High'] - out['Low']).replace(0, np.nan)
    out['BODY_ATR'] = body / out['ATR14'].replace(0, np.nan)
    out['CLOSE_LOCATION'] = (out['Close'] - out['Low']) / candle_range
    out['BULL_CANDLE'] = out['Close'] > out['Open']
    out['BEAR_CANDLE'] = out['Close'] < out['Open']
    out['BULL_REJECTION'] = (out['CLOSE_LOCATION'] > 0.65) & (out['Close'] > out['Open'])
    out['RANGE_CONTRACTION20'] = out['ATR14'] / out['ATR14'].rolling(60).median().replace(0, np.nan)
    out['BULL_FVG'] = (out['Low'] > out['High'].shift(2)) & (out['Close'].shift(1) > out['Open'].shift(1)) & (out['BODY_ATR'].shift(1) >= 0.65) & (out['VOL_RATIO'].shift(1) >= 1.15)
    out['FVG_LOW'] = out['High'].shift(2).where(out['BULL_FVG'])
    out['FVG_HIGH'] = out['Low'].where(out['BULL_FVG'])
    if benchmark is not None and (not benchmark.empty) and ('Close' in benchmark):
        bench_close = benchmark['Close'].reindex(out.index).ffill()
        out['BENCH_CLOSE'] = bench_close
        out['BENCH_EMA50'] = ema(bench_close, 50)
        out['BENCH_EMA200'] = ema(bench_close, 200)
        out['BENCH_ROC20'] = bench_close.pct_change(20)
        out['REL_STRENGTH60'] = out['ROC60'] - bench_close.pct_change(60)
    else:
        out['BENCH_CLOSE'] = np.nan
        out['BENCH_EMA50'] = np.nan
        out['BENCH_EMA200'] = np.nan
        out['BENCH_ROC20'] = np.nan
        out['REL_STRENGTH60'] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)
import math
from typing import Callable
import numpy as np
import pandas as pd

def _finite(value: object, default: float=0.0) -> float:
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
        return float('inf')
    if zone_low <= close <= zone_high:
        return 0.0
    if close > zone_high:
        return (close - zone_high) / atr
    return (zone_low - close) / atr

def _price_structure_target_pair(
    df: pd.DataFrame,
    entry: float,
    *,
    setup: str = 'GENERIC',
    explicit_levels: list[tuple[float, str]] | None = None,
    projection_origin: float | None = None,
    projection_height: float | None = None,
    price_ceiling: float | None = None,
    lookback: int = 252,
) -> dict[str, Any]:
    """Return TP1/TP2 from observable price structure, never R multiples.

    Candidate levels are drawn from confirmed pivot highs, prior rolling highs,
    setup-specific anchors, and measured-move projections whose height comes
    from an actual swing/range.  ``price_ceiling`` is used for intraday setups
    where the IDX ARA boundary is a hard market constraint.
    """
    result = {
        'tp1': None, 'tp2': None, 'tp1_basis': 'STRUCTURE_UNAVAILABLE',
        'tp2_basis': 'STRUCTURE_UNAVAILABLE', 'target_structure': '',
        'target_structure_valid': False,
    }
    entry = _finite(entry, np.nan)
    if not np.isfinite(entry) or entry <= 0 or df is None or df.empty:
        return result
    ceiling = _finite(price_ceiling, np.nan)
    tick = idx_tick_size(entry)
    minimum_gap = max(2.0 * tick, 0.0025 * entry)
    candidates: dict[float, str] = {}

    def add_level(value: Any, basis: str) -> None:
        number = _finite(value, np.nan)
        if not np.isfinite(number):
            return
        rounded = round_idx_price(number, 'down')
        if rounded is None or rounded < entry + minimum_gap:
            return
        if np.isfinite(ceiling) and rounded > ceiling:
            rounded = round_idx_price(ceiling, 'down')
        if rounded is None or rounded < entry + minimum_gap:
            return
        candidates.setdefault(float(rounded), basis)

    for value, basis in explicit_levels or []:
        add_level(value, basis)

    recent = df.iloc[-max(20, min(int(lookback), len(df))):]
    if 'PIVOT_HIGH' in recent:
        for value in pd.to_numeric(recent['PIVOT_HIGH'], errors='coerce').dropna().tolist():
            add_level(value, 'CONFIRMED_PIVOT_HIGH')
    if 'LAST_PIVOT_HIGH' in recent:
        add_level(recent['LAST_PIVOT_HIGH'].iloc[-1], 'LAST_CONFIRMED_PIVOT_HIGH')
    for column, basis in (
        ('HIGH20_PREV', 'PRIOR_20D_HIGH'),
        ('HIGH55_PREV', 'PRIOR_55D_HIGH'),
        ('HIGH252', 'PRIOR_52W_HIGH'),
    ):
        if column in recent:
            add_level(recent[column].iloc[-1], basis)

    height = _finite(projection_height, np.nan)
    origin = _finite(projection_origin, np.nan)
    if not np.isfinite(height) or height <= tick:
        # Prefer a completed impulse leg (pivot low -> later pivot high).
        pivot_height = np.nan
        if 'PIVOT_HIGH' in recent and 'PIVOT_LOW' in recent:
            highs = pd.to_numeric(recent['PIVOT_HIGH'], errors='coerce').dropna()
            lows = pd.to_numeric(recent['PIVOT_LOW'], errors='coerce').dropna()
            if not highs.empty and not lows.empty:
                high_time = highs.index[-1]
                prior_lows = lows.loc[lows.index < high_time]
                if not prior_lows.empty:
                    pivot_height = float(highs.iloc[-1] - prior_lows.iloc[-1])
                    if not np.isfinite(origin):
                        origin = float(highs.iloc[-1])
        range20 = _finite(recent['High'].tail(min(20, len(recent))).max(), np.nan) - _finite(recent['Low'].tail(min(20, len(recent))).min(), np.nan)
        range55 = _finite(recent['High'].tail(min(55, len(recent))).max(), np.nan) - _finite(recent['Low'].tail(min(55, len(recent))).min(), np.nan)
        valid_heights = [v for v in (pivot_height, range20, range55) if np.isfinite(v) and v > tick]
        height = min(valid_heights) if valid_heights else np.nan
    if not np.isfinite(origin):
        explicit_values = [_finite(v, np.nan) for v, _ in (explicit_levels or [])]
        usable = [v for v in explicit_values if np.isfinite(v)]
        origin = max(usable) if usable else entry
    if np.isfinite(height) and height > tick:
        # Fibonacci/measured-move projections are derived from the actual
        # swing/range height, not from stop distance or a desired RR.
        for ratio, label in ((0.618, 'MEASURED_MOVE_0.618'), (1.0, 'MEASURED_MOVE_1.000'), (1.618, 'MEASURED_MOVE_1.618')):
            add_level(origin + ratio * height, f'{setup}_{label}')
        # If the origin is below entry, project the same structural amplitude
        # from entry so two usable targets still represent the observed range.
        if origin < entry:
            for ratio, label in ((0.618, 'ENTRY_RANGE_PROJECTION_0.618'), (1.0, 'ENTRY_RANGE_PROJECTION_1.000'), (1.618, 'ENTRY_RANGE_PROJECTION_1.618')):
                add_level(entry + ratio * height, f'{setup}_{label}')

    levels = sorted(candidates)
    if len(levels) < 2:
        return result
    tp1 = levels[0]
    tp2_candidates = [value for value in levels[1:] if value >= tp1 + max(2.0 * idx_tick_size(tp1), 0.0025 * entry)]
    if not tp2_candidates:
        return result
    tp2 = tp2_candidates[0]
    result.update({
        'tp1': tp1, 'tp2': tp2,
        'tp1_basis': candidates[tp1], 'tp2_basis': candidates[tp2],
        'target_structure': ' | '.join(f'{value:.0f}:{candidates[value]}' for value in levels[:8]),
        'target_structure_valid': True,
    })
    return result

def _structural_stop_selection(
    df: pd.DataFrame, entry: float, proposed_stop: float, atr_value: float, setup: str
) -> dict[str, Any]:
    """Validate a stop against observable support rather than a risk multiple.

    The detector's proposed stop remains authoritative when it is already below
    a meaningful support with enough market-noise clearance.  If it is only one
    tick below entry, the function selects the nearest lower confirmed support
    and places the stop one IDX tick beneath it.  When no support exists, the
    setup is invalidated instead of fabricating an ATR/R fallback.
    """
    entry = _finite(entry, np.nan)
    proposed = _finite(proposed_stop, np.nan)
    atr_value = _finite(atr_value, np.nan)
    result = {
        'stop_loss': np.nan, 'stop_basis': 'STRUCTURE_UNAVAILABLE',
        'stop_structure': '', 'stop_structure_valid': False,
        'stop_distance_atr': np.nan,
    }
    if df is None or df.empty or not np.isfinite(entry) or entry <= 0:
        return result
    recent = df.tail(min(90, len(df)))
    levels: list[tuple[float, str]] = []

    def add(value: Any, basis: str) -> None:
        number = _finite(value, np.nan)
        if np.isfinite(number) and 0 < number < entry:
            levels.append((number, basis))

    if 'PIVOT_LOW' in recent:
        for value in pd.to_numeric(recent['PIVOT_LOW'], errors='coerce').dropna().tail(8):
            add(value, 'CONFIRMED_PIVOT_LOW')
    if 'LAST_PIVOT_LOW' in recent:
        add(recent['LAST_PIVOT_LOW'].iloc[-1], 'LAST_CONFIRMED_PIVOT_LOW')
    for column, basis in (
        ('LOW20_PREV', 'PRIOR_20D_LOW'), ('LOW55_PREV', 'PRIOR_55D_LOW'),
        ('EMA20', 'EMA20_SUPPORT'), ('EMA50', 'EMA50_SUPPORT'),
    ):
        if column in recent:
            add(recent[column].iloc[-1], basis)
    # Completed local swing lows are valid structure, but only use lows that
    # are not the current incomplete/noisy bar.
    low_series = pd.to_numeric(recent['Low'], errors='coerce')
    for window in (5, 10, 20):
        if len(low_series) > window:
            add(low_series.iloc[-window-1:-1].min(), f'LOCAL_{window}D_SWING_LOW')

    tick = idx_tick_size(entry)
    minimum_clearance = max(2.0 * tick, 0.30 * atr_value if np.isfinite(atr_value) and atr_value > 0 else 2.0 * tick)
    proposed_distance = entry - proposed if np.isfinite(proposed) else np.nan
    # Raw stop is accepted only when it is not a micro-stop and it sits at or
    # below at least one observable support.
    supports_above_stop = [(value, basis) for value, basis in levels if value >= proposed] if np.isfinite(proposed) else []
    if np.isfinite(proposed) and proposed > 0 and proposed < entry and proposed_distance >= minimum_clearance and supports_above_stop:
        chosen_support, chosen_basis = min(supports_above_stop, key=lambda item: item[0] - proposed)
        stop = round_idx_price(proposed, 'down')
        result.update({
            'stop_loss': stop, 'stop_basis': f'DETECTOR_{chosen_basis}',
            'stop_structure': f'{chosen_support:.2f}:{chosen_basis}',
            'stop_structure_valid': bool(stop and stop < entry),
            'stop_distance_atr': (entry - stop) / atr_value if stop and np.isfinite(atr_value) and atr_value > 0 else np.nan,
        })
        return result

    # Find the closest support that provides adequate clearance; stop is one
    # tick below it.  This is still price structure, not a desired stop %.
    candidates: list[tuple[float, str, float]] = []
    for support, basis in levels:
        stop = round_idx_price(support - idx_tick_size(support), 'down')
        if stop is None or stop <= 0 or stop >= entry:
            continue
        distance = entry - stop
        if distance >= minimum_clearance:
            candidates.append((stop, basis, support))
    if candidates:
        stop, basis, support = max(candidates, key=lambda item: item[0])
        result.update({
            'stop_loss': stop, 'stop_basis': basis,
            'stop_structure': f'{support:.2f}:{basis}',
            'stop_structure_valid': True,
            'stop_distance_atr': (entry - stop) / atr_value if np.isfinite(atr_value) and atr_value > 0 else np.nan,
        })
    return result


def _plan_prices(
    plan: SetupPlan,
    df: pd.DataFrame,
    atr_value: float,
    raw_entry: float,
    raw_stop: float,
    *,
    structural_levels: list[tuple[float, str]] | None = None,
    projection_origin: float | None = None,
    projection_height: float | None = None,
) -> SetupPlan:
    if plan.entry_low is None or plan.entry_high is None or plan.entry_low >= plan.entry_high:
        plan.invalidated = True
        plan.reason = 'Zona entry tidak memiliki rentang harga yang valid'
        return plan
    plan.entry_low = round_idx_price(plan.entry_low, 'down')
    plan.entry_high = round_idx_price(plan.entry_high, 'up')
    plan.entry = round_idx_price(raw_entry, 'up')
    plan.trigger = round_idx_price(plan.trigger if plan.trigger is not None else raw_entry, 'up')
    stop_selection = _structural_stop_selection(
        df, float(plan.entry) if plan.entry is not None else np.nan, raw_stop, atr_value, plan.setup
    )
    plan.stop_loss = stop_selection['stop_loss'] if np.isfinite(_finite(stop_selection.get('stop_loss'), np.nan)) else None
    plan.stop_basis = str(stop_selection.get('stop_basis') or 'STRUCTURE_UNAVAILABLE')
    plan.stop_structure = str(stop_selection.get('stop_structure') or '')
    plan.stop_structure_valid = bool(stop_selection.get('stop_structure_valid'))
    plan.stop_distance_atr = _finite(stop_selection.get('stop_distance_atr'), np.nan)
    if plan.entry is None or plan.stop_loss is None or plan.stop_loss >= plan.entry or not plan.stop_structure_valid:
        plan.invalidated = True
        plan.blockers.append('Stop-loss tidak memiliki invalidasi support/swing yang terverifikasi')
        plan.reason = 'Struktur tidak menghasilkan structural stop yang valid; tidak memakai fallback risiko/ATR sintetis'
        return plan
    targets = _price_structure_target_pair(
        df, float(plan.entry), setup=plan.setup, explicit_levels=structural_levels,
        projection_origin=projection_origin, projection_height=projection_height,
    )
    if not targets['target_structure_valid']:
        plan.invalidated = True
        plan.blockers.append('Struktur harga tidak menghasilkan dua target resistance/proyeksi yang valid')
        plan.reason = 'Target tidak difabrikasi dengan R-multiple; setup menunggu struktur target yang valid'
        plan.tp1_basis = 'STRUCTURE_UNAVAILABLE'
        plan.tp2_basis = 'STRUCTURE_UNAVAILABLE'
        plan.target_model = 'PRICE_STRUCTURE_ONLY'
        return plan
    plan.tp1 = targets['tp1']
    plan.tp2 = targets['tp2']
    plan.tp1_basis = targets['tp1_basis']
    plan.tp2_basis = targets['tp2_basis']
    plan.target_model = 'PRICE_STRUCTURE_ONLY'
    plan.target_structure = targets['target_structure']
    risk = plan.entry - plan.stop_loss
    plan.rr1 = round((plan.tp1 - plan.entry) / risk, 2)
    plan.rr2 = round((plan.tp2 - plan.entry) / risk, 2)
    return plan

def _detect_pullback_v21(df: pd.DataFrame, ticker: str) -> SetupPlan:
    name = 'PULLBACK_CONTINUATION'
    plan = SetupPlan(ticker=ticker, setup=name, detected=False, setup_score=0.0)
    if len(df) < 205:
        plan.reason = 'Data tren jangka panjang belum cukup'
        return plan
    row = df.iloc[-1]
    prev = df.iloc[-2]
    close = _finite(row['Close'])
    atr_v = _finite(row['ATR14'])
    if close <= 0 or atr_v <= 0:
        plan.reason = 'ATR/harga tidak valid'
        return plan
    ema20, ema50, ema200 = (_finite(row[x]) for x in ('EMA20', 'EMA50', 'EMA200'))
    trend = ema20 > ema50 > ema200 and close > ema50 and (ema20 > _finite(df['EMA20'].iloc[-11]))
    momentum = _finite(row['ROC60'], -1) > 0.04 and _finite(row['DIST_52W_HIGH'], -1) > -0.18
    support = ema20 if close >= ema20 - 0.45 * atr_v else max(ema50, _finite(row['VWAP20']))
    recent = df.iloc[-5:]
    touched = bool((recent['Low'] <= recent['EMA20'] + 0.35 * recent['ATR14']).any())
    held = close >= ema50 - 0.25 * atr_v
    pullback = trend and touched and held
    confirmation = bool(row['BULL_REJECTION']) or (close > _finite(prev['High']) and close > _finite(row['Open']))
    vol_contract = _finite(recent['Volume'].iloc[:-1].mean()) < 0.92 * _finite(df['VOL_MA20'].iloc[-1], 1)
    relative = _finite(row['REL_STRENGTH60'], 0) > 0
    score = 0.0
    score += 25 if trend else 0
    score += 15 if momentum else 0
    score += 18 if pullback else 8 if trend and held else 0
    score += 10 if vol_contract else 4
    score += 12 if confirmation else 3
    score += 10 if _finite(row['CMF20']) > -0.03 else 0
    score += 10 if relative else 4
    plan.setup_score = min(100.0, score)
    plan.detected = bool(trend and momentum and pullback)
    if not plan.detected:
        plan.reason = 'Belum memenuhi kombinasi uptrend, momentum, dan pullback ke value area'
        return plan
    touch_mask = (df['Low'] <= df['EMA20'] + 0.35 * df['ATR14']) & (df['Close'] >= df['EMA50'])
    created = _last_true_index(touch_mask, 10) or df.index[-1]
    zone_low = support - 0.35 * atr_v
    zone_high = support + 0.4 * atr_v
    recent_low = _finite(df['Low'].iloc[-7:].min())
    pivot_low = _finite(row['LAST_PIVOT_LOW'], recent_low)
    structural_low = min(recent_low, pivot_low) if pivot_low > close - 4 * atr_v else recent_low
    raw_stop = structural_low - 0.2 * atr_v
    if confirmation and _distance_to_zone(close, zone_low, zone_high, atr_v) <= 0.5:
        raw_entry = max(close, _finite(row['High']) + idx_tick_size(close))
        plan.entry_type = 'BUY_STOP_CONFIRMATION'
        plan.action = 'READY_TRIGGER'
    else:
        raw_entry = (zone_low + zone_high) / 2
        plan.entry_type = 'LIMIT_ON_PULLBACK_THEN_CONFIRM'
        plan.action = 'WAIT_PULLBACK_CONFIRMATION'
    plan.signal_date = df.index[-1]
    plan.zone_created_date = created
    plan.zone_age_bars = _bars_since(df, created)
    plan.valid_until = pd.Timestamp(df.index[-1]) + pd.offsets.BDay(10)
    plan.entry_low, plan.entry_high = (zone_low, zone_high)
    plan.trigger = _finite(row['High']) + idx_tick_size(close)
    plan.distance_atr = round(_distance_to_zone(close, zone_low, zone_high, atr_v), 2)
    plan.evidence = ['EMA20 > EMA50 > EMA200', 'Momentum 3 bulan positif', 'Pullback menyentuh value area']
    if vol_contract:
        plan.evidence.append('Volume mengecil saat pullback')
    if confirmation:
        plan.evidence.append('Ada reclaim/rejection bullish')
    plan.reason = 'Kelanjutan tren setelah pullback terkontrol'
    swing_window = df.iloc[-80:]
    confirmed_highs = pd.to_numeric(swing_window.get('PIVOT_HIGH'), errors='coerce').dropna() if 'PIVOT_HIGH' in swing_window else pd.Series(dtype=float)
    swing_high = float(confirmed_highs.iloc[-1]) if not confirmed_highs.empty else _finite(swing_window['High'].tail(20).max(), close)
    impulse_low = _finite(swing_window['Low'].tail(40).min(), support)
    if not confirmed_highs.empty and 'PIVOT_LOW' in swing_window:
        high_time = confirmed_highs.index[-1]
        prior_lows = pd.to_numeric(swing_window.loc[swing_window.index < high_time, 'PIVOT_LOW'], errors='coerce').dropna()
        if not prior_lows.empty:
            impulse_low = float(prior_lows.iloc[-1])
    structural_levels = [
        (swing_high, 'PRIOR_IMPULSE_HIGH'),
        (_finite(row.get('HIGH55_PREV'), np.nan), 'PRIOR_55D_HIGH'),
        (_finite(row.get('HIGH252'), np.nan), 'PRIOR_52W_HIGH'),
    ]
    return _plan_prices(
        plan, df, atr_v, raw_entry, raw_stop, structural_levels=structural_levels,
        projection_origin=swing_high, projection_height=max(swing_high - impulse_low, zone_high - zone_low),
    )

def _detect_breakout_v21(df: pd.DataFrame, ticker: str) -> SetupPlan:
    name = 'BREAKOUT_RETEST'
    plan = SetupPlan(ticker=ticker, setup=name, detected=False, setup_score=0.0)
    if len(df) < 205:
        plan.reason = 'Data belum cukup'
        return plan
    row = df.iloc[-1]
    close, atr_v = (_finite(row['Close']), _finite(row['ATR14']))
    if close <= 0 or atr_v <= 0:
        plan.reason = 'ATR/harga tidak valid'
        return plan
    breakout_mask = (df['Close'] > df['HIGH55_PREV'] + 0.05 * df['ATR14']) & (df['VOL_RATIO'] >= 1.25) & (df['BODY_ATR'] >= 0.4) & (df['Close'] > df['Open'])
    breakout_date = _last_true_index(breakout_mask, 18)
    if breakout_date is None:
        plan.reason = 'Belum ada breakout 55-hari dengan volume dan displacement'
        return plan
    pos = int(np.flatnonzero(df.index == breakout_date)[-1])
    breakout_row = df.iloc[pos]
    resistance = _finite(breakout_row['HIGH55_PREV'])
    breakout_atr = _finite(breakout_row['ATR14'], atr_v)
    post = df.iloc[pos + 1:] if pos + 1 < len(df) else df.iloc[0:0]
    retest_mask = (post['Low'] <= resistance + 0.45 * post['ATR14']) & (post['Low'] >= resistance - 1.0 * post['ATR14']) & (post['Close'] >= resistance - 0.1 * post['ATR14'])
    retest_date = _last_true_index(retest_mask, min(12, len(post))) if not post.empty else None
    invalidated = bool((post['Close'] < resistance - 1.15 * post['ATR14']).any()) if not post.empty else False
    confirmation = False
    retest_low = resistance - 0.6 * atr_v
    if retest_date is not None:
        rpos = int(np.flatnonzero(df.index == retest_date)[-1])
        retest_low = _finite(df['Low'].iloc[max(pos + 1, rpos - 2):rpos + 1].min(), retest_low)
        latest_retest = df.iloc[rpos]
        confirmation = bool(latest_retest['BULL_REJECTION']) or _finite(latest_retest['Close']) > resistance
    trend = _finite(row['EMA20']) > _finite(row['EMA50']) > _finite(row['EMA200'])
    relative = _finite(row['REL_STRENGTH60'], 0) > 0
    breakout_quality = min(1.0, _finite(breakout_row['VOL_RATIO']) / 2.0)
    score = 20 * float(trend) + 25 * breakout_quality + 15 * min(1.0, _finite(breakout_row['BODY_ATR']))
    score += 22 if retest_date is not None else 6
    score += 10 if confirmation else 2
    score += 8 if relative else 3
    plan.setup_score = round(min(100.0, score), 1)
    plan.detected = not invalidated
    plan.invalidated = invalidated
    if invalidated:
        plan.reason = 'Breakout sudah gagal: penutupan menembus bawah level invalidasi'
        return plan
    zone_low = resistance - 0.35 * atr_v
    zone_high = resistance + 0.35 * atr_v
    in_retest_area = _distance_to_zone(close, zone_low, zone_high, atr_v) <= 0.45
    if retest_date is not None and confirmation and in_retest_area:
        raw_entry = max(close, _finite(row['High']) + idx_tick_size(close))
        plan.entry_type = 'BUY_STOP_AFTER_RETEST'
        plan.action = 'READY_TRIGGER'
    else:
        raw_entry = resistance + 0.1 * atr_v
        plan.entry_type = 'LIMIT_RETEST_WITH_RECLAIM'
        plan.action = 'WAIT_RETEST'
    raw_stop = min(retest_low, resistance - 0.7 * atr_v) - 0.15 * atr_v
    plan.signal_date = breakout_date
    plan.zone_created_date = breakout_date
    plan.zone_age_bars = _bars_since(df, breakout_date)
    plan.valid_until = pd.Timestamp(breakout_date) + pd.offsets.BDay(25)
    plan.entry_low, plan.entry_high = (zone_low, zone_high)
    plan.trigger = max(resistance, _finite(row['High'])) + idx_tick_size(close)
    plan.distance_atr = round(_distance_to_zone(close, zone_low, zone_high, atr_v), 2)
    plan.evidence = ['Breakout high 55-hari', f"Volume breakout {_finite(breakout_row['VOL_RATIO']):.2f}x", 'Displacement bullish']
    if retest_date is not None:
        plan.evidence.append('Retest level breakout terdeteksi')
    if confirmation:
        plan.evidence.append('Retest ditutup dengan reclaim')
    plan.reason = 'Breakout tervalidasi; eksekusi hanya setelah retest/reclaim'
    base_start = max(0, pos - 25)
    breakout_base_low = _finite(df['Low'].iloc[base_start:pos + 1].min(), resistance - atr_v)
    post_breakout_high = _finite(df['High'].iloc[pos:].max(), resistance)
    structural_levels = [
        (post_breakout_high, 'POST_BREAKOUT_SWING_HIGH'),
        (_finite(row.get('HIGH252'), np.nan), 'PRIOR_52W_HIGH'),
    ]
    return _plan_prices(
        plan, df, atr_v, raw_entry, raw_stop, structural_levels=structural_levels,
        projection_origin=resistance, projection_height=max(resistance - breakout_base_low, zone_high - zone_low),
    )

def _detect_reversal_v21(df: pd.DataFrame, ticker: str) -> SetupPlan:
    name = 'REVERSAL_ACCUMULATION'
    plan = SetupPlan(ticker=ticker, setup=name, detected=False, setup_score=0.0)
    if len(df) < 205:
        plan.reason = 'Data belum cukup'
        return plan
    row = df.iloc[-1]
    close, atr_v = (_finite(row['Close']), _finite(row['ATR14']))
    if close <= 0 or atr_v <= 0:
        plan.reason = 'ATR/harga tidak valid'
        return plan
    prior = df.iloc[-150:-30]
    base = df.iloc[-30:]
    prior_high = _finite(prior['High'].max(), close)
    base_low, base_high = (_finite(base['Low'].min(), close), _finite(base['High'].max(), close))
    decline = base_low / prior_high - 1 if prior_high > 0 else 0
    base_width = (base_high - base_low) / close
    based = decline <= -0.12 and base_width <= 0.32
    contraction = _finite(row['RANGE_CONTRACTION20'], 2) <= 0.95
    accumulation = _finite(base['CMF20'].iloc[-10:].mean()) > 0.02 and _finite(row['OBV_SLOPE10']) > 0
    sweep_mask = (df['Low'] < df['LOW20_PREV']) & (df['Close'] > df['LOW20_PREV']) & (df['CLOSE_LOCATION'] > 0.58)
    sweep_date = _last_true_index(sweep_mask, 25)
    if sweep_date is not None:
        spos = int(np.flatnonzero(df.index == sweep_date)[-1])
        sweep_low = _finite(df.iloc[spos]['Low'])
    else:
        spos, sweep_low = (len(df) - 30, base_low)
    choch_mask = (df['Close'] > df['LAST_PIVOT_HIGH'] + 0.05 * df['ATR14']) & (df['Close'] > df['EMA20']) & (df['VOL_RATIO'] >= 1.05)
    post_choch = choch_mask.iloc[spos:] if sweep_date is not None else choch_mask.iloc[-15:]
    choch_date = _last_true_index(post_choch, len(post_choch)) if len(post_choch) else None
    choch = choch_date is not None
    invalidated = close < sweep_low - 0.2 * atr_v
    score = 0.0
    score += 18 if based else 5
    score += 12 if contraction else 3
    score += 20 if accumulation else 8 if _finite(row['CMF20']) > 0 else 0
    score += 20 if sweep_date is not None else 0
    score += 22 if choch else 4
    score += 8 if close > _finite(row['EMA50']) else 2
    plan.setup_score = min(100.0, score)
    plan.detected = bool(based and accumulation and (sweep_date is not None) and (not invalidated))
    plan.invalidated = invalidated
    if not plan.detected:
        plan.reason = 'Belum ada rangkaian decline–base–akumulasi–liquidity sweep yang lengkap'
        return plan
    structure_level = _finite(row['LAST_PIVOT_HIGH'], base_high)
    if choch_date is not None:
        cpos = int(np.flatnonzero(df.index == choch_date)[-1])
        structure_level = _finite(df.iloc[cpos]['LAST_PIVOT_HIGH'], structure_level)
    zone_low = structure_level - 0.45 * atr_v
    zone_high = structure_level + 0.35 * atr_v
    in_zone = _distance_to_zone(close, zone_low, zone_high, atr_v) <= 0.45
    confirmation = bool(row['BULL_REJECTION']) or close > _finite(df['High'].iloc[-2])
    if choch and in_zone and confirmation:
        raw_entry = max(close, _finite(row['High']) + idx_tick_size(close))
        plan.entry_type = 'BUY_STOP_AFTER_CHOCH'
        plan.action = 'READY_TRIGGER'
    elif choch:
        raw_entry = (zone_low + zone_high) / 2
        plan.entry_type = 'LIMIT_ON_CHOCH_RETEST'
        plan.action = 'WAIT_RETEST'
    else:
        raw_entry = structure_level + idx_tick_size(structure_level)
        plan.entry_type = 'BUY_STOP_AFTER_CHOCH'
        plan.action = 'WAIT_CHOCH'
    raw_stop = sweep_low - 0.2 * atr_v
    plan.signal_date = sweep_date
    plan.zone_created_date = choch_date or sweep_date
    plan.zone_age_bars = _bars_since(df, plan.zone_created_date)
    plan.valid_until = pd.Timestamp(sweep_date) + pd.offsets.BDay(30)
    plan.entry_low, plan.entry_high = (zone_low, zone_high)
    plan.trigger = structure_level + idx_tick_size(structure_level)
    plan.distance_atr = round(_distance_to_zone(close, zone_low, zone_high, atr_v), 2)
    plan.evidence = ['Penurunan diikuti base', 'Proxy CMF/OBV menguat', 'Sell-side liquidity sweep']
    if contraction:
        plan.evidence.append('Volatilitas berkontraksi')
    if choch:
        plan.evidence.append('CHOCH/BOS bullish terkonfirmasi')
    plan.reason = 'Reversal hanya dapat dieksekusi setelah perubahan struktur bullish'
    structural_levels = [
        (base_high, 'BASE_RANGE_HIGH'),
        (prior_high, 'PRIOR_SUPPLY_HIGH'),
        (_finite(row.get('HIGH252'), np.nan), 'PRIOR_52W_HIGH'),
    ]
    return _plan_prices(
        plan, df, atr_v, raw_entry, raw_stop, structural_levels=structural_levels,
        projection_origin=structure_level, projection_height=max(base_high - base_low, zone_high - zone_low),
    )

def _detect_unicorn_v21(df: pd.DataFrame, ticker: str) -> SetupPlan:
    name = 'UNICORN_SNIPER_ICT'
    plan = SetupPlan(ticker=ticker, setup=name, detected=False, setup_score=0.0)
    if len(df) < 120:
        plan.reason = 'Data struktur belum cukup'
        return plan
    row = df.iloc[-1]
    close, atr_v = (_finite(row['Close']), _finite(row['ATR14']))
    if close <= 0 or atr_v <= 0:
        plan.reason = 'ATR/harga tidak valid'
        return plan
    sweep_mask = (df['Low'] < df['LOW20_PREV']) & (df['Close'] > df['LOW20_PREV']) & (df['CLOSE_LOCATION'] >= 0.55)
    sweep_date = _last_true_index(sweep_mask, 35)
    if sweep_date is None:
        plan.reason = 'Belum ada sell-side liquidity sweep bullish'
        return plan
    spos = int(np.flatnonzero(df.index == sweep_date)[-1])
    sweep_low = _finite(df.iloc[spos]['Low'])
    bos_mask = (df['Close'] > df['LAST_PIVOT_HIGH'] + 0.05 * df['ATR14']) & (df['BODY_ATR'] >= 0.55) & (df['Close'] > df['Open'])
    bos_post = bos_mask.iloc[spos + 1:]
    bos_date = _last_true_index(bos_post, min(20, len(bos_post))) if len(bos_post) else None
    if bos_date is None:
        plan.setup_score = 28.0
        plan.reason = 'Liquidity sweep ada, tetapi displacement/BOS belum terkonfirmasi'
        return plan
    bpos = int(np.flatnonzero(df.index == bos_date)[-1])
    fvg_window = df.iloc[max(spos + 1, bpos - 2):min(len(df), bpos + 6)]
    fvg_hits = fvg_window[fvg_window['BULL_FVG'].fillna(False)]
    if fvg_hits.empty:
        plan.setup_score = 50.0
        plan.reason = 'Sweep dan BOS ada, tetapi FVG displacement tidak valid'
        return plan
    fvg_date = fvg_hits.index[-1]
    fpos = int(np.flatnonzero(df.index == fvg_date)[-1])
    fvg_low = _finite(df.loc[fvg_date, 'FVG_LOW'])
    fvg_high = _finite(df.loc[fvg_date, 'FVG_HIGH'])
    if fvg_high <= fvg_low:
        plan.reason = 'FVG tidak valid'
        return plan
    search_ob = df.iloc[spos:max(spos + 1, fpos)]
    bear = search_ob[search_ob['BEAR_CANDLE'].fillna(False)]
    ob_overlap = False
    if not bear.empty:
        ob_row = bear.iloc[-1]
        ob_low, ob_high = (_finite(ob_row['Low']), max(_finite(ob_row['Open']), _finite(ob_row['Close'])))
        overlap_low, overlap_high = (max(fvg_low, ob_low), min(fvg_high, ob_high))
        if overlap_high > overlap_low:
            zone_low, zone_high = (overlap_low, overlap_high)
            ob_overlap = True
        else:
            zone_low, zone_high = (fvg_low, fvg_high)
    else:
        zone_low, zone_high = (fvg_low, fvg_high)
    after_fvg = df.iloc[fpos + 1:]
    invalidated = bool(((after_fvg['Close'] < fvg_low - 0.15 * after_fvg['ATR14']) | (after_fvg['Close'] < sweep_low)).any()) if not after_fvg.empty else False
    dealing_high = _finite(df['High'].iloc[spos:bpos + 1].max(), close)
    equilibrium = (sweep_low + dealing_high) / 2
    discount = (zone_low + zone_high) / 2 <= equilibrium
    volume_ok = _finite(df.loc[bos_date, 'VOL_RATIO'], 0) >= 1.05
    confirmation = bool(row['BULL_REJECTION']) or close > _finite(df['High'].iloc[-2])
    distance = _distance_to_zone(close, zone_low, zone_high, atr_v)
    score = 20 + 25 + 15 + 15
    score += 10 if ob_overlap else 3
    score += 10 if discount else 2
    score += 5 if volume_ok else 0
    plan.setup_score = min(100.0, float(score))
    plan.detected = not invalidated
    plan.invalidated = invalidated
    if invalidated:
        plan.reason = 'FVG/low sweep sudah ditutup tembus; zona tidak lagi valid'
        return plan
    in_zone = distance <= 0.35
    if in_zone and confirmation:
        raw_entry = max(close, _finite(row['High']) + idx_tick_size(close))
        plan.entry_type = 'BUY_STOP_FVG_RECLAIM'
        plan.action = 'READY_TRIGGER'
    else:
        raw_entry = (zone_low + zone_high) / 2
        plan.entry_type = 'LIMIT_FVG_THEN_RECLAIM'
        plan.action = 'WAIT_FVG_RETRACE'
    raw_stop = min(sweep_low, zone_low - 0.45 * atr_v) - 0.1 * atr_v
    plan.signal_date = sweep_date
    plan.zone_created_date = fvg_date
    plan.zone_age_bars = _bars_since(df, fvg_date)
    plan.valid_until = pd.Timestamp(fvg_date) + pd.offsets.BDay(30)
    plan.entry_low, plan.entry_high = (zone_low, zone_high)
    plan.trigger = max(zone_high, _finite(row['High'])) + idx_tick_size(close)
    plan.distance_atr = round(distance, 2)
    plan.evidence = ['Sell-side liquidity sweep', 'Bullish BOS dengan displacement', 'Bullish FVG valid']
    if ob_overlap:
        plan.evidence.append('FVG overlap dengan order-block proxy')
    if discount:
        plan.evidence.append('Zona berada di discount dealing range')
    plan.reason = 'SMC/ICT dipakai sebagai timing confluence, bukan bukti standalone'
    bos_leg_high = _finite(df['High'].iloc[bpos:min(len(df), fpos + 6)].max(), dealing_high)
    structural_levels = [
        (dealing_high, 'DEALING_RANGE_BUY_SIDE_LIQUIDITY'),
        (bos_leg_high, 'BOS_LEG_HIGH'),
        (_finite(row.get('HIGH252'), np.nan), 'PRIOR_52W_HIGH'),
    ]
    return _plan_prices(
        plan, df, atr_v, raw_entry, raw_stop, structural_levels=structural_levels,
        projection_origin=dealing_high, projection_height=max(dealing_high - sweep_low, zone_high - zone_low),
    )

def detect_all_setups(df: pd.DataFrame, ticker: str) -> list[SetupPlan]:
    return [detector(df, ticker) for detector in SETUP_DETECTORS]
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable
import pandas as pd
TICKER_COLUMNS = ('ticker', 'tickers', 'symbol', 'symbols', 'kode', 'code', 'emiten', 'stock')

def _daily_ohlcv_cache_path(ticker: str) -> Path:
    safe = re.sub('[^A-Z0-9._-]', '_', str(ticker).upper())
    root = _cache_root() / 'ohlcv_daily'
    root.mkdir(parents=True, exist_ok=True)
    return root / f'{safe}.csv'

def _load_daily_ohlcv_cache_v431(ticker: str) -> pd.DataFrame:
    try:
        path = _daily_ohlcv_cache_path(ticker)
        if not path.is_file():
            return pd.DataFrame()
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        return _clean_ohlcv(frame, strict=True)
    except Exception:
        return pd.DataFrame()

def _daily_ohlcv_cache_meta_path(ticker: str) -> Path:
    return _daily_ohlcv_cache_path(ticker).with_suffix('.meta.json')

def _load_daily_ohlcv_cache_source_family(ticker: str) -> str:
    import json as _json
    try:
        path = _daily_ohlcv_cache_meta_path(ticker)
        if not path.is_file():
            return 'UNKNOWN'
        payload = _json.loads(path.read_text(encoding='utf-8'))
        family = str(payload.get('source_family', 'UNKNOWN')).strip().upper()
        return family or 'UNKNOWN'
    except Exception:
        return 'UNKNOWN'

def normalize_idx_ticker(value: object) -> str | None:
    text = str(value).strip().upper()
    if not text or text in {'NAN', 'NONE', 'NULL', 'TICKER'}:
        return None
    text = re.sub('\\s+', '', text)
    text = text.replace('.IDX', '').replace('IDX:', '')
    if text.endswith('.JK'):
        base = text[:-3]
    else:
        base = text
    if not re.fullmatch('[A-Z0-9]{3,8}', base):
        return None
    return f'{base}.JK'

def parse_ticker_csv(source: bytes | BinaryIO | pd.DataFrame, max_tickers: int=1200) -> list[str]:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        payload = BytesIO(source) if isinstance(source, bytes) else source
        try:
            frame = pd.read_csv(payload, sep=None, engine='python')
        except UnicodeDecodeError:
            if hasattr(payload, 'seek'):
                payload.seek(0)
            frame = pd.read_csv(payload, encoding='latin-1', sep=None, engine='python')
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

def _clean_ohlcv(frame: pd.DataFrame, strict: bool=False) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.columns = [str(c).title() for c in out.columns]
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all((c in out.columns for c in required)):
        return pd.DataFrame()
    out = out[required]
    out.index = pd.to_datetime(out.index, errors='coerce')
    if getattr(out.index, 'tz', None) is not None:
        out.index = out.index.tz_convert('Asia/Jakarta').tz_localize(None)
    for col in required:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out = out.dropna(subset=['Open', 'High', 'Low', 'Close'])
    out = out[~out.index.duplicated(keep='last')].sort_index()
    out['Volume'] = out['Volume'].fillna(0.0).clip(lower=0)
    valid = out[['Open', 'High', 'Low', 'Close']].gt(0).all(axis=1) & out['High'].ge(out[['Open', 'Low', 'Close']].max(axis=1)) & out['Low'].le(out[['Open', 'High', 'Close']].min(axis=1))
    if strict:
        out = out[valid]
    return out

def ohlcv_quality_issues(frame: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if frame is None or frame.empty:
        return ['OHLCV kosong']
    if frame.index.has_duplicates:
        issues.append('Tanggal duplikat')
    if not frame.index.is_monotonic_increasing:
        issues.append('Tanggal tidak terurut')
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all((column in frame for column in required)):
        return ['Kolom OHLCV tidak lengkap']
    if frame[required].isna().any().any():
        issues.append('OHLCV mengandung nilai kosong')
    if (frame['Volume'] < 0).any():
        issues.append('Volume negatif')
    valid = frame[['Open', 'High', 'Low', 'Close']].gt(0).all(axis=1) & frame['High'].ge(frame[['Open', 'Low', 'Close']].max(axis=1)) & frame['Low'].le(frame[['Open', 'High', 'Close']].min(axis=1))
    if not bool(valid.all()):
        issues.append('Bar OHLC tidak konsisten')
    jumps = frame['Close'].pct_change().abs()
    if bool(jumps.gt(0.8).any()):
        issues.append('Lompatan adjusted price >80%; corporate action/data wajib diverifikasi')
    if len(frame) >= 20 and float(frame['Volume'].tail(20).eq(0).mean()) > 0.1:
        issues.append('Lebih dari 10% bar terakhir bervolume nol')
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

def _merge_ohlcv_history(base: pd.DataFrame, update: pd.DataFrame) -> pd.DataFrame:
    """Merge provider updates without discarding previously verified history."""
    left = _clean_ohlcv(base, strict=True)
    right = _clean_ohlcv(update, strict=True)
    if left.empty:
        return right
    if right.empty:
        return left
    return _clean_ohlcv(pd.concat([left, right], axis=0), strict=True)

def _align_secondary_ohlcv_to_cached(secondary: pd.DataFrame, cached: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Scale raw secondary bars to the cached adjusted-price basis when safe.

    iTick exposes standard OHLCV and separate adjustment-factor products. To
    avoid spending extra free-tier calls, overlapping closes are used to infer
    a stable scale factor. Unstable/no overlap is left raw and explicitly
    warned so corporate-action gates remain fail-closed.
    """
    incoming = _clean_ohlcv(secondary, strict=True)
    base = _clean_ohlcv(cached, strict=True)
    if incoming.empty or base.empty:
        return (incoming, 'secondary history tidak memiliki overlap adjusted; corporate action wajib diverifikasi')
    overlap = base[['Close']].rename(columns={'Close': 'base'}).join(incoming[['Close']].rename(columns={'Close': 'secondary'}), how='inner').dropna()
    if len(overlap) < 3:
        return (incoming, 'overlap adjusted kurang dari 3 bar; corporate action wajib diverifikasi')
    ratios = (overlap['base'] / overlap['secondary']).replace([np.inf, -np.inf], np.nan).dropna()
    if ratios.empty:
        return (incoming, 'rasio adjusted provider tidak dapat dihitung')
    ratio = float(ratios.median())
    dispersion = float((ratios / ratio - 1.0).abs().median()) if ratio else np.inf
    if not np.isfinite(ratio) or ratio <= 0 or dispersion > 0.03:
        return (incoming, 'basis harga antar-provider tidak stabil; data sekunder dipakai tanpa penyesuaian')
    aligned = incoming.copy()
    for column in ('Open', 'High', 'Low', 'Close'):
        aligned[column] = aligned[column] * ratio
    return (_clean_ohlcv(aligned, strict=True), f'secondary OHLC diselaraskan ke cache dengan faktor {ratio:.6f}')

def _itick_period_limit(period: str, interval: str) -> int:
    if interval == '1d':
        return {'2y': 540, '3y': 800, '5y': 1350}.get(str(period).lower(), 800)
    return 1000 if interval == '5m' else 500

def _reserve_itick_free_call(max_calls_per_minute: int=4) -> bool:
    """Process-local/persistent guard below iTick's published 5 calls/minute."""
    import json as _json
    import time as _time
    path = _cache_root() / 'itick_free_rate_budget.json'
    now = float(_time.time())
    timestamps: list[float] = []
    try:
        if path.is_file():
            payload = _json.loads(path.read_text(encoding='utf-8'))
            timestamps = [float(value) for value in payload if now - float(value) < 60.0]
    except Exception:
        timestamps = []
    if len(timestamps) >= max(1, int(max_calls_per_minute)):
        return False
    timestamps.append(now)
    tmp = path.with_suffix('.tmp')
    try:
        tmp.write_text(_json.dumps(timestamps), encoding='utf-8')
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return True

def fetch_itick_ohlcv(tickers: Iterable[str], api_token: str, period: str='3y', interval: str='1d', timeout: int=12, max_tickers: int=40) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Optional free-tier IDX OHLCV fallback using iTick's documented API.

    The adapter is deliberately bounded and rate-budgeted. A missing token or
    exhausted free quota never blocks the rest of the scanner.
    """
    columns = ['ticker', 'status', 'bars', 'source', 'error']
    token = str(api_token or '').strip()
    names = [normalize_idx_ticker(value) for value in tickers]
    names = [value for value in dict.fromkeys(names) if value][:max(0, int(max_tickers))]
    if not names or not token:
        return ({}, pd.DataFrame(columns=columns))
    ktype = {'1d': '8', '5m': '2', '15m': '3'}.get(interval)
    if not ktype:
        raise ValueError(f'Interval iTick belum didukung: {interval}')
    histories: dict[str, pd.DataFrame] = {}
    reports: list[dict[str, Any]] = []
    for ticker in names:
        if not _reserve_itick_free_call():
            reports.append({'ticker': ticker, 'status': 'RATE_BUDGET_EXHAUSTED', 'bars': 0, 'source': 'ITICK_FREE', 'error': 'Batas internal 4 panggilan/menit tercapai'})
            continue
        code = ticker[:-3] if ticker.endswith('.JK') else ticker
        try:
            import requests
            response = requests.get('https://api.itick.org/stock/kline', params={'region': 'ID', 'code': code, 'kType': ktype, 'limit': str(_itick_period_limit(period, interval))}, headers={'accept': 'application/json', 'token': token}, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, Mapping) or int(payload.get('code', -1)) != 0:
                raise RuntimeError(_safe_text(payload.get('msg')) if isinstance(payload, Mapping) else 'respons tidak valid')
            values = payload.get('data', [])
            if not isinstance(values, list) or not values:
                raise RuntimeError('data kline kosong')
            raw = pd.DataFrame(values).rename(columns={'t': 'Date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
            stamps = pd.to_datetime(raw['Date'], unit='ms', utc=True, errors='coerce')
            raw.index = stamps.dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
            if interval == '1d':
                raw.index = raw.index.normalize()
            frame = _clean_ohlcv(raw, strict=True)
            if frame.empty:
                raise RuntimeError('OHLCV iTick tidak lolos validasi')
            frame.attrs['source_interval'] = interval
            frame.attrs['provider'] = 'ITICK_FREE'
            histories[ticker] = frame
            reports.append({'ticker': ticker, 'status': 'OK', 'bars': len(frame), 'source': 'ITICK_FREE', 'error': ''})
        except Exception as exc:
            safe_error = str(exc).replace(token, '***')
            reports.append({'ticker': ticker, 'status': 'FAILED', 'bars': 0, 'source': 'ITICK_FREE', 'error': f'{type(exc).__name__}: {safe_error[:140]}'})
    return (histories, pd.DataFrame(reports, columns=columns))

def _idx_eod_patch_frame(row: pd.Series) -> pd.DataFrame:
    if row is None or row.empty:
        return pd.DataFrame()
    frame = pd.DataFrame([{'Open': row.get('Open'), 'High': row.get('High'), 'Low': row.get('Low'), 'Close': row.get('Close'), 'Volume': row.get('Volume')}], index=[pd.Timestamp(row.get('Date')).normalize()])
    return _clean_ohlcv(frame, strict=True)

def _download_ohlcv_v431(tickers: Iterable[str], period: str='3y', batch_size: int=30, itick_api_token: str='') -> tuple[dict[str, pd.DataFrame], DownloadReport]:
    """Free-first resilient daily OHLCV orchestration.

    Order: current verified cache -> bounded Yahoo refresh -> official IDX EOD
    patch -> optional iTick free historical fallback -> stale cache research.
    """
    import yfinance as yf
    requested = list(dict.fromkeys(tickers))
    histories: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}
    warnings: dict[str, str] = {}
    source_tiers: dict[str, str] = {}
    cached_histories = {ticker: _load_daily_ohlcv_cache(ticker) for ticker in requested}
    for ticker, cached in cached_histories.items():
        if _daily_cache_is_current(cached):
            histories[ticker] = cached
            source_tiers[ticker] = f'CACHE_FRESH_VERIFIED_{_load_daily_ohlcv_cache_source_family(ticker)}'
            warnings[ticker] = f'Cache EOD current; bar terakhir {pd.Timestamp(cached.index[-1]).date().isoformat()}'
    provider_targets = [ticker for ticker in requested if ticker not in histories]
    skip_individual_retry: set[str] = set()
    for start in range(0, len(provider_targets), max(1, int(batch_size))):
        batch = provider_targets[start:start + max(1, int(batch_size))]
        try:
            raw = yf.download(batch, period=period, interval='1d', group_by='ticker', auto_adjust=True, repair=False, actions=False, threads=4, progress=False, timeout=25)
            for ticker in batch:
                extracted = _extract_batch(raw, ticker, len(batch))
                quality = ohlcv_quality_issues(extracted)
                live = _clean_ohlcv(extracted, strict=True)
                if not live.empty:
                    merged = _merge_ohlcv_history(cached_histories.get(ticker, pd.DataFrame()), live)
                    histories[ticker] = merged
                    source_tiers[ticker] = 'LIVE_YAHOO'
                    _write_daily_ohlcv_cache(ticker, merged, 'YAHOO')
                    if quality:
                        warnings[ticker] = ' • '.join(quality)
                else:
                    failed[ticker] = 'Data batch Yahoo kosong'
            if raw is None or raw.empty:
                skip_individual_retry.update(batch)
        except Exception as exc:
            for ticker in batch:
                failed[ticker] = f'Batch Yahoo gagal: {type(exc).__name__}'
            message = f'{type(exc).__name__} {exc}'.upper()
            if 'RATE' in message or 'TOO MANY' in message or 'INVALID CRUMB' in message:
                skip_individual_retry.update(batch)
    missing = [ticker for ticker in provider_targets if ticker not in histories and ticker not in skip_individual_retry]

    def retry_one(ticker: str) -> tuple[str, pd.DataFrame, str | None]:
        try:
            frame = yf.Ticker(ticker).history(period=period, interval='1d', auto_adjust=True, repair=False, actions=False, timeout=20)
            extracted = _clean_ohlcv(frame)
            clean = _clean_ohlcv(extracted, strict=True)
            quality = ' • '.join(ohlcv_quality_issues(extracted))
            audit = quality or None if not clean.empty else 'Data individual Yahoo kosong'
            return (ticker, clean, audit)
        except Exception as exc:
            return (ticker, pd.DataFrame(), f'{type(exc).__name__}: {str(exc)[:100]}')
    if missing:
        with ThreadPoolExecutor(max_workers=min(2, len(missing))) as pool:
            futures = [pool.submit(retry_one, ticker) for ticker in missing]
            for future in as_completed(futures):
                ticker, live, error = future.result()
                if not live.empty:
                    merged = _merge_ohlcv_history(cached_histories.get(ticker, pd.DataFrame()), live)
                    histories[ticker] = merged
                    source_tiers[ticker] = 'LIVE_YAHOO_RETRY'
                    _write_daily_ohlcv_cache(ticker, merged, 'YAHOO')
                    failed.pop(ticker, None)
                    if error:
                        warnings[ticker] = error
                else:
                    failed[ticker] = error or 'Yahoo tidak menyediakan data'
    unresolved = [ticker for ticker in requested if ticker not in histories]
    if unresolved:
        try:
            official, _ = fetch_idx_official_eod_quotes(unresolved, reference_date=_expected_last_completed_daily_date(), lookback_days=7, timeout=8)
        except Exception:
            official = pd.DataFrame()
        if official is not None and (not official.empty):
            for ticker, group in official.groupby('ticker', sort=False):
                cached = cached_histories.get(str(ticker), pd.DataFrame())
                if cached.empty:
                    continue
                latest_row = group.sort_values('Date').iloc[-1]
                patch = _idx_eod_patch_frame(latest_row)
                if patch.empty:
                    continue
                merged = _merge_ohlcv_history(cached, patch)
                if len(merged) < len(cached):
                    continue
                histories[str(ticker)] = merged
                source_tiers[str(ticker)] = 'LIVE_IDX_EOD_PATCH'
                _write_daily_ohlcv_cache(str(ticker), merged, 'IDX_OFFICIAL')
                failed.pop(str(ticker), None)
                gap_days = abs((pd.Timestamp(patch.index[-1]) - pd.Timestamp(cached.index[-1])).days)
                warnings[str(ticker)] = f'Yahoo gagal; bar EOD terakhir ditambal dari IDX resmi (gap kalender {gap_days} hari)'
    unresolved = [ticker for ticker in requested if ticker not in histories]
    if unresolved and str(itick_api_token or '').strip():
        secondary, secondary_report = fetch_itick_ohlcv(unresolved, api_token=itick_api_token, period=period, interval='1d', max_tickers=len(unresolved))
        for ticker, frame in secondary.items():
            cached = cached_histories.get(ticker, pd.DataFrame())
            aligned, note = _align_secondary_ohlcv_to_cached(frame, cached)
            merged = _merge_ohlcv_history(cached, aligned)
            if merged.empty:
                continue
            histories[ticker] = merged
            source_tiers[ticker] = 'LIVE_ITICK_FREE_FALLBACK'
            _write_daily_ohlcv_cache(ticker, merged, 'ITICK')
            failed.pop(ticker, None)
            warnings[ticker] = f'Yahoo/IDX history fallback memakai iTick free • {note}'
        if not secondary_report.empty:
            for _, row in secondary_report.loc[secondary_report['status'].ne('OK')].iterrows():
                failed.setdefault(str(row['ticker']), _safe_text(row['error']) or _safe_text(row['status']))
    unresolved = [ticker for ticker in requested if ticker not in histories]
    for ticker in unresolved:
        cached = cached_histories.get(ticker, pd.DataFrame())
        if cached.empty:
            continue
        histories[ticker] = cached
        source_tiers[ticker] = 'CACHE_FALLBACK'
        cache_date = pd.Timestamp(cached.index[-1]).date().isoformat()
        warnings[ticker] = f'OHLCV memakai cache fallback stale; bar terakhir {cache_date}'
        failed.pop(ticker, None)
    for ticker in requested:
        source_tiers.setdefault(ticker, 'UNAVAILABLE' if ticker not in histories else 'LIVE_YAHOO')
        if ticker not in histories:
            failed.setdefault(ticker, 'Semua provider OHLCV gratis tidak tersedia')
    report = DownloadReport(requested, sorted(histories), failed, provider='Free multi-source: cache → Yahoo → IDX official → iTick optional', adjusted_prices=True, downloaded_at=pd.Timestamp.now(tz='Asia/Jakarta').isoformat(), warnings=warnings, source_tiers=source_tiers)
    return (histories, report)

def _download_benchmark_v431(period: str='3y') -> pd.DataFrame:
    """Cache-first JKSE benchmark; never discard valid history on Yahoo outage."""
    import yfinance as yf
    cached = _load_daily_ohlcv_cache('^JKSE')
    if _daily_cache_is_current(cached):
        return cached
    try:
        frame = yf.Ticker('^JKSE').history(period=period, interval='1d', auto_adjust=True, repair=False, actions=False, timeout=20)
        clean = _clean_ohlcv(frame, strict=True)
        if not clean.empty:
            merged = _merge_ohlcv_history(cached, clean)
            _write_daily_ohlcv_cache('^JKSE', merged, 'YAHOO')
            return merged
    except Exception:
        pass
    return cached
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

def _linear_score(value: float, bad: float, good: float, higher_is_better: bool=True) -> float | None:
    if not np.isfinite(value):
        return None
    if good == bad:
        return 50.0
    ratio = (value - bad) / (good - bad)
    if not higher_is_better:
        ratio = 1 - ratio
    return float(np.clip(100 * ratio, 0, 100))

def score_fundamentals(info: dict[str, Any]) -> dict[str, Any]:
    sector_text = str(info.get('sector') or '')
    industry_text = str(info.get('industry') or '')
    is_financial = 'financial' in sector_text.lower() or 'bank' in industry_text.lower()
    debt_equity_raw = _num(info.get('debtToEquity'))
    debt_equity = debt_equity_raw / 100 if np.isfinite(debt_equity_raw) else np.nan
    total_cash = _num(info.get('totalCash'))
    total_debt = _num(info.get('totalDebt'))
    market_cap = _num(info.get('marketCap'))
    fcf = _num(info.get('freeCashflow'))
    ocf = _num(info.get('operatingCashflow'))
    cash_to_debt = total_cash / total_debt if np.isfinite(total_cash) and total_debt > 0 else 5.0 if np.isfinite(total_cash) and total_debt == 0 else np.nan
    fcf_yield = fcf / market_cap if np.isfinite(fcf) and market_cap > 0 else np.nan
    officers = info.get('companyOfficers') or []
    chief = {}
    for officer in officers if isinstance(officers, list) else []:
        title = str(officer.get('title') or '').lower() if isinstance(officer, dict) else ''
        if any(token in title for token in ('chief executive', 'president director', 'direktur utama', 'ceo')):
            chief = officer
            break
    if not chief and isinstance(officers, list) and officers:
        chief = officers[0] if isinstance(officers[0], dict) else {}
    metrics = {
        'revenue_growth': _num(info.get('revenueGrowth')),
        'earnings_growth': _num(info.get('earningsGrowth')),
        'gross_margin': _num(info.get('grossMargins')),
        'operating_margin': _num(info.get('operatingMargins')),
        'net_margin': _num(info.get('profitMargins')),
        'roe': _num(info.get('returnOnEquity')),
        'roa': _num(info.get('returnOnAssets')),
        'debt_equity': debt_equity,
        'current_ratio': _num(info.get('currentRatio')),
        'cash_to_debt': cash_to_debt,
        'operating_cash_flow': ocf,
        'free_cash_flow': fcf,
        'fcf_yield': fcf_yield,
        'trailing_pe': _num(info.get('trailingPE')),
        'forward_pe': _num(info.get('forwardPE')),
        'price_to_book': _num(info.get('priceToBook')),
        'peg_ratio': _num(info.get('pegRatio')),
        'market_cap': market_cap,
        'total_revenue': _num(info.get('totalRevenue')),
        'ebitda': _num(info.get('ebitda')),
        'enterprise_value': _num(info.get('enterpriseValue')),
        'sector': sector_text or industry_text,
        'company_name': info.get('shortName') or info.get('longName') or '',
        'company_website': info.get('website') or '',
        'business_summary': info.get('longBusinessSummary') or '',
        'ceo_name': chief.get('name') or '',
        'ceo_title': chief.get('title') or '',
        'ceo_age': _num(chief.get('age')),
        'governance_overall_risk': _num(info.get('overallRisk')),
        'governance_board_risk': _num(info.get('boardRisk')),
        'governance_audit_risk': _num(info.get('auditRisk')),
        'governance_shareholder_rights_risk': _num(info.get('shareHolderRightsRisk')),
        'fundamental_model': 'FINANCIAL' if is_financial else 'GENERAL',
    }
    weighted: list[tuple[float, float]] = []
    applicable_weight = 0.0

    def add(score: float | None, weight: float) -> None:
        nonlocal applicable_weight
        applicable_weight += weight
        if score is not None and np.isfinite(score):
            weighted.append((float(score), weight))
    add(_linear_score(metrics['revenue_growth'], -0.05, 0.2), 14)
    add(_linear_score(metrics['earnings_growth'], -0.1, 0.25), 14)
    add(_linear_score(metrics['roe'], 0.05, 0.22), 10)
    add(_linear_score(metrics['roa'], 0.01, 0.1), 7)
    add(_linear_score(metrics['gross_margin'], 0.1, 0.45), 6)
    add(_linear_score(metrics['operating_margin'], 0.02, 0.2), 7)
    add(_linear_score(metrics['net_margin'], 0.01, 0.15), 6)
    if not is_financial:
        add(_linear_score(metrics['debt_equity'], 2.0, 0.3, higher_is_better=True), 8)
        add(_linear_score(metrics['current_ratio'], 0.8, 2.0), 5)
        add(_linear_score(metrics['cash_to_debt'], 0.1, 1.2), 5)
        add(100.0 if np.isfinite(ocf) and ocf > 0 else 0.0 if np.isfinite(ocf) else None, 6)
        add(100.0 if np.isfinite(fcf) and fcf > 0 else 0.0 if np.isfinite(fcf) else None, 6)
        add(_linear_score(metrics['fcf_yield'], 0.0, 0.08), 3)
    peg = metrics['peg_ratio']
    peg_score = None
    if np.isfinite(peg):
        peg_score = 100.0 if 0 < peg <= 1.5 else 65.0 if peg <= 2.5 else 20.0 if peg > 0 else 0.0
    add(peg_score, 3)
    score = sum((value * weight for value, weight in weighted)) / sum((weight for _, weight in weighted)) if weighted else np.nan
    coverage = sum((weight for _, weight in weighted)) / applicable_weight if applicable_weight else 0.0
    red_flags: list[str] = []
    if np.isfinite(metrics['revenue_growth']) and metrics['revenue_growth'] < 0:
        red_flags.append('Revenue menyusut')
    if np.isfinite(metrics['earnings_growth']) and metrics['earnings_growth'] < 0:
        red_flags.append('Laba menyusut')
    if not is_financial:
        if np.isfinite(ocf) and ocf <= 0:
            red_flags.append('OCF negatif')
        if np.isfinite(fcf) and fcf <= 0:
            red_flags.append('FCF negatif')
        if np.isfinite(debt_equity) and debt_equity > 2:
            red_flags.append('DER tinggi')
    if np.isfinite(metrics['net_margin']) and metrics['net_margin'] <= 0:
        red_flags.append('Margin bersih negatif')
    metrics.update({'fundamental_score': round(float(score), 1) if np.isfinite(score) else np.nan, 'fundamental_coverage': round(100 * coverage, 1), 'fundamental_reliability': 'HIGH' if coverage >= 0.7 else 'MEDIUM' if coverage >= 0.45 else 'LOW', 'fundamental_red_flags': ' • '.join(red_flags)})
    return metrics

def fetch_fundamentals(tickers: Iterable[str], max_workers: int=2) -> pd.DataFrame:
    """Fetch Yahoo snapshots with a bounded provider circuit breaker.

    Yahoo authentication/rate-limit failures are provider-wide, not ticker-wide.
    A single probe is therefore used before bounded parallel chunks. Once a
    provider-wide failure is detected, remaining symbols are marked unresolved
    without generating dozens of identical HTTP/crumb requests.
    """
    names = list(dict.fromkeys(tickers))
    if not names:
        return pd.DataFrame()
    health = _yahoo_provider_health_status()
    if health:
        message = _yahoo_cooldown_message(health)
        return pd.DataFrame([
            _fundamental_failure_row(ticker, 'PROVIDER_COOLDOWN', message)
            for ticker in names
        ])

    rows: list[dict[str, Any]] = []
    first = fetch_one_fundamental(names[0])
    rows.append(first)
    first_code = _safe_text(first.get('fundamental_error_code')).upper()
    if _is_yahoo_blocking_error(first_code):
        _mark_yahoo_provider_unavailable(first_code, first.get('fundamental_error', ''))
        rows.extend(
            _fundamental_failure_row(
                ticker, 'PROVIDER_COOLDOWN',
                f'Yahoo dihentikan sementara setelah {first_code} pada {names[0]}',
            )
            for ticker in names[1:]
        )
        return pd.DataFrame(rows)
    if _finite(first.get('fundamental_coverage'), 0) > 0:
        _clear_yahoo_provider_unavailable()

    remaining = names[1:]
    workers = min(max(1, int(max_workers)), max(1, len(remaining)))
    chunk_size = max(2, workers * 2)
    for offset in range(0, len(remaining), chunk_size):
        chunk = remaining[offset:offset + chunk_size]
        with ThreadPoolExecutor(max_workers=min(workers, len(chunk))) as pool:
            futures = {pool.submit(fetch_one_fundamental, ticker): ticker for ticker in chunk}
            chunk_rows = [future.result() for future in as_completed(futures)]
        rows.extend(chunk_rows)
        blocking = next((
            row for row in chunk_rows
            if _is_yahoo_blocking_error(_safe_text(row.get('fundamental_error_code')).upper())
        ), None)
        if blocking is not None:
            code = _safe_text(blocking.get('fundamental_error_code')).upper()
            _mark_yahoo_provider_unavailable(code, blocking.get('fundamental_error', ''))
            pending = remaining[offset + len(chunk):]
            rows.extend(
                _fundamental_failure_row(
                    ticker, 'PROVIDER_COOLDOWN',
                    f'Yahoo dihentikan sementara setelah {code}',
                )
                for ticker in pending
            )
            break
    return pd.DataFrame(rows)

def attach_fundamentals(signals: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    if fundamentals.empty:
        result = signals.copy()
        result['fundamental_score'] = np.nan
        result['fundamental_coverage'] = 0.0
        result['fundamental_reliability'] = 'NONE'
        result['fundamental_red_flags'] = ''
        result['fundamental_error'] = 'Fundamental tidak diambil/tersedia'
        result['composite_score'] = result['quality_score']
        return result
    result = signals.merge(fundamentals, on='ticker', how='left')
    usable = result['fundamental_coverage'].fillna(0) >= 60
    result['composite_score'] = result['quality_score']
    result.loc[usable, 'composite_score'] = (0.78 * result.loc[usable, 'quality_score'] + 0.22 * result.loc[usable, 'fundamental_score']).round(1)
    return result
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
    for encoding in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            if hasattr(payload, 'seek'):
                payload.seek(0)
            return pd.read_csv(payload, sep=None, engine='python', encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc
    raise ValueError(f'CSV tidak dapat dibaca: {last_error}')

def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'ya', 'aktif', 'active'}

def _column(frame: pd.DataFrame, name: str, default: object) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index)

def _context_append_blocker(frame: pd.DataFrame, index: object, message: str) -> None:
    prior = str(frame.at[index, 'blockers'] or '').strip()
    if message not in prior:
        frame.at[index, 'blockers'] = f'{prior} • {message}' if prior else message
        count = pd.to_numeric(frame.at[index, 'blocker_count'], errors='coerce')
        frame.at[index, 'blocker_count'] = int(count) + 1 if pd.notna(count) else 1

def _downgrade(frame: pd.DataFrame, index: object, message: str, reject: bool=False) -> None:
    if reject:
        frame.at[index, 'status'] = 'REJECT'
    elif frame.at[index, 'status'] == 'EXECUTION_READY':
        frame.at[index, 'status'] = 'WATCHLIST_ENTRY'
    _context_append_blocker(frame, index, message)

def parse_broker_summary_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    """Aggregate Stockbit/exported broker summary without claiming beneficial ownership."""
    frame = _read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {'ticker', 'date', 'broker_code'}
    if not required.issubset(frame.columns):
        raise ValueError('Broker-summary CSV wajib memiliki ticker, date, dan broker_code')
    buy_col = 'buy_value' if 'buy_value' in frame else 'buy_volume' if 'buy_volume' in frame else None
    sell_col = 'sell_value' if 'sell_value' in frame else 'sell_volume' if 'sell_volume' in frame else None
    if buy_col is None or sell_col is None:
        raise ValueError('Broker-summary CSV memerlukan buy_value/sell_value atau buy_volume/sell_volume')
    frame['ticker'] = frame['ticker'].map(normalize_idx_ticker)
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
    frame['buy'] = pd.to_numeric(frame[buy_col], errors='coerce').fillna(0).clip(lower=0)
    frame['sell'] = pd.to_numeric(frame[sell_col], errors='coerce').fillna(0).clip(lower=0)
    frame['net'] = frame['buy'] - frame['sell']
    frame['gross'] = frame['buy'] + frame['sell']
    rows: list[dict[str, object]] = []
    for ticker, group in frame.dropna(subset=['ticker', 'date']).groupby('ticker', sort=False):
        dates = sorted(group['date'].dt.normalize().unique())[-10:]
        recent = group[group['date'].dt.normalize().isin(dates)]
        net = float(recent['net'].sum())
        gross = float(recent['gross'].sum())
        ratio = net / gross if gross > 0 else np.nan
        broker_net = recent.groupby('broker_code')['net'].sum().sort_values(ascending=False)
        label = 'ACCUMULATION_PROXY' if ratio >= 0.08 else 'DISTRIBUTION_PROXY' if ratio <= -0.08 else 'NEUTRAL'
        rows.append({'ticker': ticker, 'broksum_asof': recent['date'].max(), 'broksum_days': len(dates), 'broksum_net': net, 'broksum_net_ratio': ratio, 'broksum_signal': label, 'top_net_buy_brokers': ', '.join(map(str, broker_net.head(3).index.tolist())), 'top_net_sell_brokers': ', '.join(map(str, broker_net.tail(3).index.tolist())), 'broksum_note': 'Proxy kode broker; bukan identitas beneficial owner'})
    return pd.DataFrame(rows)

def attach_broker_summary(signals: pd.DataFrame, broksum: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if broksum is None or broksum.empty:
        out['broksum_signal'] = 'UNAVAILABLE'
        out['broksum_note'] = 'Data broker summary tidak dilampirkan'
        return out
    out = out.merge(broksum, on='ticker', how='left')
    out['broksum_signal'] = out['broksum_signal'].fillna('UNAVAILABLE')
    for idx in out.index[out['broksum_signal'].eq('DISTRIBUTION_PROXY')]:
        _downgrade(out, idx, 'Broker-summary menunjukkan distribution proxy')
    out['status_rank'] = out['status'].map({'EXECUTION_READY': 0, 'WATCHLIST_ENTRY': 1, 'REJECT': 2})
    return out
import math
import numpy as np
import pandas as pd

def size_stockbit_order(entry: float, stop_loss: float, config: ScanConfig | None=None) -> dict[str, float | int | str]:
    """Size a regular-market order so the fee/slippage-adjusted loss stays capped."""
    cfg = config or ScanConfig()
    values = (entry, stop_loss, cfg.account_size_idr, cfg.risk_per_trade_pct)
    if not all((math.isfinite(float(value)) for value in values)):
        return {'sizing_status': 'INVALID_LEVELS', 'suggested_lots': 0}
    if entry <= 0 or stop_loss <= 0 or stop_loss >= entry:
        return {'sizing_status': 'INVALID_LEVELS', 'suggested_lots': 0}
    if cfg.account_size_idr <= 0 or cfg.risk_per_trade_pct <= 0 or cfg.max_position_pct <= 0:
        return {
            'sizing_status': 'ACCOUNT_SIZE_NOT_CONFIGURED',
            'risk_budget_idr': 0.0, 'risk_per_share_net': np.nan, 'risk_per_lot_idr': np.nan,
            'lots_by_risk': 0, 'lots_by_capital': 0, 'suggested_lots': 0, 'shares': 0,
            'capital_required_idr': 0.0, 'position_pct': 0.0, 'max_loss_idr': 0.0,
            'max_loss_pct_account': 0.0, 'portfolio_max_positions': int(cfg.max_positions),
            'portfolio_risk_cap_idr': 0.0,
        }
    if not is_valid_idx_price(entry) or not is_valid_idx_price(stop_loss):
        return {'sizing_status': 'INVALID_TICK', 'suggested_lots': 0}
    half_slippage = cfg.order_slippage_pct / 2
    effective_buy = entry * (1 + cfg.buy_fee_pct + half_slippage)
    effective_stop_proceeds = stop_loss * (1 - cfg.sell_fee_pct - half_slippage)
    risk_per_share = effective_buy - effective_stop_proceeds
    capital_per_lot = 100 * effective_buy
    risk_per_lot = 100 * risk_per_share
    risk_budget = cfg.account_size_idr * cfg.risk_per_trade_pct
    position_cap = cfg.account_size_idr * cfg.max_position_pct
    if risk_per_lot <= 0 or capital_per_lot <= 0:
        return {'sizing_status': 'INVALID_RISK', 'suggested_lots': 0}
    lots_by_risk = math.floor(risk_budget / risk_per_lot)
    lots_by_capital = math.floor(position_cap / capital_per_lot)
    lots = max(0, min(lots_by_risk, lots_by_capital))
    capital_required = lots * capital_per_lot
    max_loss = lots * risk_per_lot
    status = 'OK' if lots >= 1 else 'ACCOUNT_TOO_SMALL_FOR_ONE_LOT'
    return {'sizing_status': status, 'risk_budget_idr': round(risk_budget, 0), 'risk_per_share_net': round(risk_per_share, 4), 'risk_per_lot_idr': round(risk_per_lot, 0), 'lots_by_risk': int(lots_by_risk), 'lots_by_capital': int(lots_by_capital), 'suggested_lots': int(lots), 'shares': int(lots * 100), 'capital_required_idr': round(capital_required, 0), 'position_pct': round(100 * capital_required / cfg.account_size_idr, 2), 'max_loss_idr': round(max_loss, 0), 'max_loss_pct_account': round(100 * max_loss / cfg.account_size_idr, 3), 'portfolio_max_positions': int(cfg.max_positions), 'portfolio_risk_cap_idr': round(cfg.account_size_idr * cfg.max_portfolio_risk_pct, 0)}
from typing import Any, Mapping
import pandas as pd

def make_signal_chart(frame: pd.DataFrame, signal: Mapping[str, Any], bars: int=180) -> go.Figure:
    import plotly.graph_objects as go
    data = frame.iloc[-bars:].copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='OHLC', increasing_line_color='#20c997', decreasing_line_color='#ff5c6c'))
    for column, color, width in (('EMA20', '#f6c85f', 1.3), ('EMA50', '#6f9ceb', 1.3), ('EMA200', '#ad75f4', 1.5)):
        if column in data:
            fig.add_trace(go.Scatter(x=data.index, y=data[column], name=column, line=dict(color=color, width=width)))
    zone_low, zone_high = (signal.get('entry_low'), signal.get('entry_high'))
    if pd.notna(zone_low) and pd.notna(zone_high):
        fig.add_hrect(y0=float(zone_low), y1=float(zone_high), fillcolor='rgba(38, 166, 154, 0.17)', line_width=0, annotation_text='Entry zone', annotation_position='top left')
    levels = (('entry', 'Entry', '#22d3ee', 'dash'), ('stop_loss', 'SL', '#ff5c6c', 'solid'), ('tp1', 'TP1', '#f6c85f', 'dot'), ('tp2', 'TP2', '#20c997', 'dot'))
    for key, label, color, dash in levels:
        value = signal.get(key)
        if pd.notna(value):
            fig.add_hline(y=float(value), line_color=color, line_dash=dash, line_width=1.25, annotation_text=f'{label} {float(value):,.0f}', annotation_position='right')
    fig.update_layout(title=f"{signal.get('ticker', '')} · {signal.get('setup', '')}", template='plotly_dark', height=620, margin=dict(l=20, r=80, t=55, b=20), xaxis_rangeslider_visible=False, legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0), hovermode='x unified')
    return fig
from dataclasses import asdict
from typing import Any
import numpy as np
import pandas as pd

def _number(value: Any, default: float=float('nan')) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default

class ScanEngine:

    def __init__(self, config: ScanConfig | None=None):
        self.config = config or ScanConfig()

    def _market_context(self, prepared: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None) -> tuple[MarketContext, pd.DataFrame | None]:
        bench_ind: pd.DataFrame | None = None
        above50: list[bool] = []
        above200: list[bool] = []
        for frame in prepared.values():
            if not frame.empty:
                last = frame.iloc[-1]
                if pd.notna(last.get('EMA50')):
                    above50.append(bool(last['Close'] > last['EMA50']))
                if pd.notna(last.get('EMA200')):
                    above200.append(bool(last['Close'] > last['EMA200']))
        breadth50 = 100 * float(np.mean(above50)) if above50 else None
        breadth200 = 100 * float(np.mean(above200)) if above200 else None
        context = MarketContext(breadth_ema50=breadth50, breadth_ema200=breadth200)
        if benchmark is None or benchmark.empty or len(benchmark) < 205:
            context.reason = 'Data IHSG tidak tersedia/cukup; sinyal tidak boleh langsung dieksekusi'
            return (context, bench_ind)
        today = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None).normalize()
        benchmark_age = max(0, (today - pd.Timestamp(benchmark.index[-1]).normalize()).days)
        if benchmark_age > self.config.max_absolute_data_age_days:
            context.reason = f'Data IHSG berumur {benchmark_age} hari; regime tidak dapat dipercaya'
            return (context, bench_ind)
        bench_ind = prepare_indicators(benchmark)
        last = bench_ind.iloc[-1]
        close = _number(last['Close'])
        ema50, ema200 = (_number(last['EMA50']), _number(last['EMA200']))
        roc20 = _number(last['ROC20'])
        context.benchmark_close = close
        context.benchmark_roc20 = roc20
        risk_on = close > ema50 > ema200 and roc20 > 0 and (breadth50 is None or breadth50 >= 52)
        risk_off = close < ema200 and roc20 < 0 or (breadth50 is not None and breadth50 < 35)
        if risk_on:
            context.regime = 'RISK_ON'
            context.reason = 'IHSG di atas EMA50/200, momentum positif, breadth mendukung'
        elif risk_off:
            context.regime = 'RISK_OFF'
            context.reason = 'IHSG/breadth menunjukkan risiko pasar tinggi'
        else:
            context.regime = 'NEUTRAL'
            context.reason = 'Sinyal IHSG dan breadth belum seragam'
        return (context, bench_ind)

    def _tradeability(self, frame: pd.DataFrame, asof: pd.Timestamp) -> tuple[list[str], dict[str, float | int | str]]:
        cfg = self.config
        blockers: list[str] = []
        last = frame.iloc[-1]
        previous = frame.iloc[-2] if len(frame) >= 2 else last
        close = _number(last['Close'], 0)
        atr_pct = _number(last.get('ATR_PCT'), 0)
        adtv = _number(last.get('ADTV20'), 0)
        zero_vol = _number(last.get('ZERO_VOL20'), 1)
        value_today = _number(last.get('VALUE'), 0)
        lag = max(0, (pd.Timestamp(asof).normalize() - pd.Timestamp(frame.index[-1]).normalize()).days)
        now_jakarta = pd.Timestamp.now(tz='Asia/Jakarta')
        today = now_jakarta.tz_localize(None).normalize()
        absolute_age = max(0, (today - pd.Timestamp(frame.index[-1]).normalize()).days)
        current_bar_incomplete = bool(
            pd.Timestamp(frame.index[-1]).date() == now_jakarta.date()
            and idx_regular_decision_window(now_jakarta)
        )
        if len(frame) < cfg.min_bars:
            blockers.append(f'Riwayat hanya {len(frame)} bar (<{cfg.min_bars})')
        if close < cfg.min_price:
            blockers.append(f'Harga Rp{close:,.0f} di bawah minimum')
        if adtv < cfg.min_adtv_idr:
            blockers.append(f'ADTV20 Rp{adtv / 1000000000.0:.2f} miliar di bawah gate')
        if zero_vol > cfg.max_zero_volume_ratio:
            blockers.append(f'Hari volume nol {zero_vol:.0%} terlalu tinggi')
        if atr_pct < cfg.min_atr_pct:
            blockers.append(f'ATR {atr_pct:.1%} terlalu rendah/stagnan')
        if atr_pct > cfg.max_atr_pct:
            blockers.append(f'ATR {atr_pct:.1%} terlalu ekstrem')
        if lag > cfg.max_data_lag_days:
            blockers.append(f'Data tertinggal {lag} hari dari universe')
        if absolute_age > cfg.max_absolute_data_age_days:
            blockers.append(f'Data absolut sudah berumur {absolute_age} hari')
        if current_bar_incomplete:
            blockers.append('Daily bar hari ini belum dianggap final')
        if adtv > 0 and value_today < 0.15 * adtv:
            blockers.append('Nilai transaksi bar terakhir sangat rendah')
        if len(frame) >= 2 and near_upper_auto_rejection(_number(previous['Close']), close, _number(last['High'])):
            blockers.append('Harga dekat/terkunci ARA; risiko mengejar harga')
        smart_score, up_down = _silent_accumulation_metrics(frame)
        metrics: dict[str, float | int | str] = {
            'last_price': close,
            'last_date': pd.Timestamp(frame.index[-1]).date().isoformat(),
            'data_lag_days': lag,
            'absolute_data_age_days': absolute_age,
            'current_bar_incomplete': int(current_bar_incomplete),
            'adtv20_idr': adtv,
            'atr_pct': atr_pct,
            'zero_volume_ratio20': zero_vol,
            'volume_ratio': _number(last.get('VOL_RATIO')),
            'rsi14': _number(last.get('RSI14')),
            'adx14': _number(last.get('ADX14')),
            'cmf20': _number(last.get('CMF20')),
            'roc60': _number(last.get('ROC60')),
            'distance_52w_high': _number(last.get('DIST_52W_HIGH')),
            'relative_strength60': _number(last.get('REL_STRENGTH60')),
            'silent_accumulation_score': smart_score,
            'up_down_value_ratio20': up_down,
            'eod_reference_date': _expected_last_completed_daily_date().date().isoformat(),
            'daily_bar_final': int(not current_bar_incomplete),
        }
        return (blockers, metrics)

    def _finalize(self, plan: SetupPlan, frame: pd.DataFrame, context: MarketContext, trade_blockers: list[str], metrics: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        result = plan.to_dict()
        close = float(metrics['last_price'])
        atr_value = _number(frame.iloc[-1].get('ATR14'), 0)
        score_adjustment = {'RISK_ON': 3.0, 'NEUTRAL': 0.0, 'RISK_OFF': -8.0, 'UNKNOWN': -5.0}
        quality_score = round(max(0.0, min(100.0, plan.setup_score + score_adjustment[context.regime])), 1)
        blockers = list(trade_blockers)
        status = 'REJECT'
        if plan.detected and (not plan.invalidated):
            if plan.zone_age_bars is not None and plan.zone_age_bars > cfg.max_zone_age_bars:
                blockers.append(f'Zona berumur {plan.zone_age_bars} bar; sudah kedaluwarsa')
            if plan.valid_until is not None and pd.Timestamp(frame.index[-1]) > pd.Timestamp(plan.valid_until):
                blockers.append('Masa berlaku setup sudah habis')
            if plan.distance_atr is not None and plan.distance_atr > cfg.watch_distance_atr:
                blockers.append(f'Harga {plan.distance_atr:.2f} ATR dari zona; terlalu jauh')
                plan.action = 'TOO_EXTENDED_WAIT_NEW_BASE'
                result['action'] = plan.action
            if plan.entry_low is not None and close < plan.entry_low - 0.75 * atr_value:
                blockers.append('Harga sudah menutup jauh di bawah zona entry')
            if plan.entry and plan.stop_loss:
                stop_pct = (plan.entry - plan.stop_loss) / plan.entry
                result['stop_pct'] = stop_pct
                if stop_pct > cfg.max_stop_pct:
                    blockers.append(f'Jarak SL {stop_pct:.1%} melebihi batas')
            else:
                result['stop_pct'] = np.nan
                blockers.append('Level entry/SL tidak valid')
            levels = (plan.entry, plan.stop_loss, plan.tp1, plan.tp2)
            if not all((is_valid_idx_price(level) for level in levels)):
                blockers.append('Satu atau lebih level order tidak sesuai fraksi harga IDX')
            if all((level is not None for level in levels)):
                if not float(plan.stop_loss) < float(plan.entry) < float(plan.tp1) < float(plan.tp2):
                    blockers.append('Urutan SL < entry < TP1 < TP2 tidak valid')
            if plan.entry is not None and (not within_idx_daily_price_band(plan.entry, close)):
                blockers.append('Entry berada di luar rentang auto-rejection sesi berikutnya')
            if (plan.rr1 or 0) < cfg.min_rr1 or (plan.rr2 or 0) < cfg.min_rr2:
                blockers.append('Risk/reward di bawah minimum')
            if quality_score < cfg.min_score:
                blockers.append(f'Quality score {quality_score:.0f} di bawah {cfg.min_score:.0f}')
            if context.regime == 'RISK_OFF':
                blockers.append('Regime IHSG RISK_OFF')
            elif context.regime == 'UNKNOWN':
                blockers.append('Regime IHSG tidak dapat diverifikasi')
            ready_action = plan.action in {'READY_TRIGGER', 'READY_LIMIT'}
            close_enough = plan.distance_atr is not None and plan.distance_atr <= _ready_distance_atr_for_setup(plan.setup, cfg)
            if not ready_action:
                blockers.append('Retest/reclaim/entry trigger belum lengkap')
            elif quality_score < cfg.execution_score:
                blockers.append(f'Quality score {quality_score:.0f} belum mencapai execution threshold {cfg.execution_score:.0f}')
            if plan.distance_atr is None:
                blockers.append('Jarak ke zona tidak dapat dihitung')
            if ready_action and close_enough and (not blockers) and (quality_score >= cfg.execution_score) and (context.regime in {'RISK_ON', 'NEUTRAL'}):
                status = 'EXECUTION_READY'
            else:
                status = 'WATCHLIST_ENTRY'
        smart = _finite(metrics.get('silent_accumulation_score'), 0.0)
        strict_flow = str(plan.setup) in {'REVERSAL_ACCUMULATION', 'UNICORN_SNIPER_ICT'}
        result['flow_gate_mode'] = 'HARD' if strict_flow else 'CONFIDENCE_ONLY'
        result['flow_confirmation'] = 'CONFIRMED' if smart >= 60 else 'NEUTRAL' if smart >= 45 else 'WEAK'
        if strict_flow and smart < 60 and plan.detected:
            message = f'Flow proxy {smart:.0f}/100 di bawah minimum 60 untuk {plan.setup}'
            if message not in blockers:
                blockers.append(message)
            if status == 'EXECUTION_READY':
                status = 'WATCHLIST_ENTRY'
        result['quality_score'] = quality_score
        result['status'] = status
        result['blockers'] = ' • '.join(blockers)
        result['blocker_count'] = len(blockers)
        result['market_regime'] = context.regime
        result['market_reason'] = context.reason
        result['breadth_ema50'] = context.breadth_ema50
        result['breadth_ema200'] = context.breadth_ema200
        result.update(metrics)
        if quality_score >= 88 and (not blockers):
            grade = 'A'
        elif quality_score >= 78 and len(blockers) <= 1:
            grade = 'B+'
        elif quality_score >= 70:
            grade = 'B'
        else:
            grade = 'C'
        result['grade'] = grade
        result['status_rank'] = STATUS_ORDER[status]
        return result

    def scan(self, histories: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None=None) -> dict[str, Any]:
        if not histories:
            return {'signals': pd.DataFrame(), 'universe': pd.DataFrame(), 'prepared': {}, 'market_context': MarketContext()}
        prepared: dict[str, pd.DataFrame] = {}
        for ticker, frame in histories.items():
            if frame is not None and (not frame.empty):
                prepared[ticker] = prepare_indicators(frame, benchmark)
        if not prepared:
            return {'signals': pd.DataFrame(), 'universe': pd.DataFrame(), 'prepared': {}, 'market_context': MarketContext()}
        context, _ = self._market_context(prepared, benchmark)
        asof_candidates = [pd.Timestamp(frame.index[-1]) for frame in prepared.values()]
        if benchmark is not None and (not benchmark.empty):
            asof_candidates.append(pd.Timestamp(benchmark.index[-1]))
        asof = max(asof_candidates)
        signal_rows: list[dict[str, Any]] = []
        universe_rows: list[dict[str, Any]] = []
        for ticker, frame in prepared.items():
            trade_blockers, metrics = self._tradeability(frame, asof)
            plans = detect_all_setups(frame, ticker)
            finalized = [self._finalize(plan, frame, context, trade_blockers, metrics) for plan in plans]
            detected = [row for row in finalized if row['detected']]
            signal_rows.extend(detected)
            candidates = detected or finalized
            best = sorted(candidates, key=lambda x: (x['status_rank'], -x['quality_score']))[0]
            universe_rows.append({'ticker': ticker, 'best_setup': best['setup'] if best['detected'] else 'NO_SETUP', 'status': best['status'], 'quality_score': best['quality_score'], 'grade': best['grade'], 'reason': best['reason'], 'blockers': best['blockers'], **metrics})
        signals = pd.DataFrame(signal_rows)
        universe = pd.DataFrame(universe_rows)
        if not signals.empty:
            signals = signals.sort_values(['status_rank', 'quality_score', 'rr2', 'adtv20_idr'], ascending=[True, False, False, False], na_position='last').reset_index(drop=True)
        if not universe.empty:
            universe['status_rank'] = universe['status'].map(STATUS_ORDER)
            universe = universe.sort_values(['status_rank', 'quality_score', 'adtv20_idr'], ascending=[True, False, False]).drop(columns='status_rank').reset_index(drop=True)
        return {'signals': signals, 'universe': universe, 'prepared': prepared, 'market_context': context, 'asof': asof, 'config': asdict(self.config)}
import math
from dataclasses import asdict, dataclass
from typing import Iterable
import numpy as np
import pandas as pd
SETUPS = ('PULLBACK_CONTINUATION', 'BREAKOUT_RETEST', 'REVERSAL_ACCUMULATION', 'UNICORN_SNIPER_ICT')

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
    result: str = 'NO_FILL'
    r_multiple: float | None = None
    holding_bars: int | None = None
    tp1_hit: bool = False
    tp2_hit: bool = False
    time_to_tp1_bars: int | None = None
    no_fill_reason: str = ''
    is_oos: bool = False
    oos_fold: int = 0
    # Local-AI features captured strictly at signal time. These fields make
    # the walk-forward events useful for learning without any look-ahead.
    rr1_plan: float | None = None
    rr2_plan: float | None = None
    stop_pct: float | None = None
    atr_pct: float | None = None
    volume_ratio: float | None = None
    rsi14: float | None = None
    adx14: float | None = None
    cmf20: float | None = None
    roc60: float | None = None
    distance_52w_high: float | None = None
    relative_strength60: float | None = None
    silent_accumulation_score: float | None = None
    adtv20_idr: float | None = None
    body_atr: float | None = None
    close_location: float | None = None
    structural_quality_score: float | None = None

def _broad_candidate_mask(df: pd.DataFrame, setup: str) -> pd.Series:
    """Cheap superset; the actual signal is always rebuilt by the live detector."""
    bullish_confirmation = df['BULL_REJECTION'] | (df['Close'] > df['High'].shift(1)) & (df['Close'] > df['Open'])
    if setup == 'PULLBACK_CONTINUATION':
        trend = (df['EMA20'] > df['EMA50']) & (df['EMA50'] > df['EMA200']) & (df['Close'] > df['EMA50'])
        momentum = (df['ROC60'] > 0.04) & (df['DIST_52W_HIGH'] > -0.18)
        touch = (df['Low'] <= df['EMA20'] + 0.35 * df['ATR14']).rolling(5).max().gt(0)
        return (trend & momentum & touch & bullish_confirmation).fillna(False)
    if setup == 'BREAKOUT_RETEST':
        breakout = (df['Close'] > df['HIGH55_PREV'] + 0.05 * df['ATR14']) & (df['VOL_RATIO'] >= 1.25) & (df['BODY_ATR'] >= 0.4) & (df['Close'] > df['Open'])
        return (breakout.rolling(18).max().gt(0) & bullish_confirmation).fillna(False)
    if setup == 'REVERSAL_ACCUMULATION':
        sweep = (df['Low'] < df['LOW20_PREV']) & (df['Close'] > df['LOW20_PREV']) & (df['CLOSE_LOCATION'] > 0.58)
        return (sweep.rolling(25).max().gt(0) & (df['CMF20'].rolling(10).mean() > -0.02) & bullish_confirmation).fillna(False)
    if setup == 'UNICORN_SNIPER_ICT':
        sweep = (df['Low'] < df['LOW20_PREV']) & (df['Close'] > df['LOW20_PREV']) & (df['CLOSE_LOCATION'] >= 0.55)
        bos = (df['Close'] > df['LAST_PIVOT_HIGH'] + 0.05 * df['ATR14']) & (df['BODY_ATR'] >= 0.55) & (df['Close'] > df['Open'])
        return (sweep.rolling(35).max().gt(0) & bos.rolling(20).max().gt(0) & df['BULL_FVG'].rolling(25).max().gt(0) & bullish_confirmation).fillna(False)
    raise ValueError(f'Unknown setup: {setup}')

def historical_signal_mask(df: pd.DataFrame, setup: str) -> pd.Series:
    """Compatibility helper: candidate dates, not a substitute live signal."""
    mask = _broad_candidate_mask(df, setup)
    if len(mask) > 0:
        mask.iloc[:min(205, len(mask))] = False
    return mask

def _historical_context(df: pd.DataFrame) -> MarketContext:
    row = df.iloc[-1]
    values = [row.get(name) for name in ('BENCH_CLOSE', 'BENCH_EMA50', 'BENCH_EMA200', 'BENCH_ROC20')]
    try:
        close, ema50, ema200, roc20 = (float(value) for value in values)
    except (TypeError, ValueError):
        return MarketContext(regime='UNKNOWN', reason='Benchmark historis tidak tersedia')
    if not all((np.isfinite(value) for value in (close, ema50, ema200, roc20))):
        return MarketContext(regime='UNKNOWN', reason='Benchmark historis tidak tersedia')
    if close > ema50 > ema200 and roc20 > 0:
        regime, reason = ('RISK_ON', 'IHSG historis trend/momentum positif')
    elif close < ema200 and roc20 < 0:
        regime, reason = ('RISK_OFF', 'IHSG historis di bawah EMA200 dan momentum negatif')
    else:
        regime, reason = ('NEUTRAL', 'Regime historis campuran')
    return MarketContext(regime=regime, benchmark_close=close, benchmark_roc20=roc20, reason=reason)

def _historical_gate_inputs_v300(df: pd.DataFrame, config: ScanConfig) -> tuple[list[str], dict[str, float | str]]:
    row = df.iloc[-1]
    prev = df.iloc[-2]

    def number(name: str, default: float=float('nan')) -> float:
        try:
            value = float(row.get(name))
            return value if np.isfinite(value) else default
        except (TypeError, ValueError):
            return default
    close = number('Close', 0)
    adtv = number('ADTV20', 0)
    atr_pct = number('ATR_PCT', 0)
    zero = number('ZERO_VOL20', 1)
    blockers: list[str] = []
    if len(df) < config.min_bars:
        blockers.append('Riwayat tidak cukup')
    if close < config.min_price:
        blockers.append('Harga di bawah minimum')
    if adtv < config.min_adtv_idr:
        blockers.append('ADTV di bawah gate')
    if zero > config.max_zero_volume_ratio:
        blockers.append('Hari volume nol terlalu tinggi')
    if not config.min_atr_pct <= atr_pct <= config.max_atr_pct:
        blockers.append('ATR di luar gate')
    if adtv > 0 and number('VALUE', 0) < 0.15 * adtv:
        blockers.append('Nilai transaksi bar sinyal terlalu rendah')
    if near_upper_auto_rejection(float(prev['Close']), close, float(row['High'])):
        blockers.append('ARA chase')
    metrics: dict[str, float | str] = {'last_price': close, 'last_date': pd.Timestamp(df.index[-1]).date().isoformat(), 'adtv20_idr': adtv, 'atr_pct': atr_pct, 'zero_volume_ratio20': zero, 'volume_ratio': number('VOL_RATIO'), 'rsi14': number('RSI14'), 'adx14': number('ADX14'), 'cmf20': number('CMF20'), 'roc60': number('ROC60'), 'distance_52w_high': number('DIST_52W_HIGH'), 'relative_strength60': number('REL_STRENGTH60'), 'data_lag_days': 0.0, 'absolute_data_age_days': 0.0, 'current_bar_incomplete': 0.0}
    return (blockers, metrics)

def _simulate_order(df: pd.DataFrame, signal_pos: int, ticker: str, setup: str, plan: object, quality_score: float, regime: str, config: ScanConfig) -> BacktestEvent:
    planned_entry = float(plan.entry)
    stop = float(plan.stop_loss)
    tp1 = float(plan.tp1)
    tp2 = float(plan.tp2)
    atr_signal = float(df['ATR14'].iloc[signal_pos])
    signal_row = df.iloc[signal_pos]
    risk = planned_entry - stop
    smart_score, _ = _silent_accumulation_metrics(df.iloc[:signal_pos + 1])
    event = BacktestEvent(
        ticker=ticker, setup=setup, signal_date=df.index[signal_pos], market_regime=regime,
        quality_score=quality_score, order_type=str(plan.entry_type), planned_entry=planned_entry,
        stop=stop, tp1=tp1, tp2=tp2, filled=False,
        rr1_plan=(tp1 - planned_entry) / risk if risk > 0 else np.nan,
        rr2_plan=(tp2 - planned_entry) / risk if risk > 0 else np.nan,
        stop_pct=risk / planned_entry if planned_entry > 0 else np.nan,
        atr_pct=_finite(signal_row.get('ATR_PCT'), np.nan),
        volume_ratio=_finite(signal_row.get('VOL_RATIO'), np.nan),
        rsi14=_finite(signal_row.get('RSI14'), np.nan),
        adx14=_finite(signal_row.get('ADX14'), np.nan),
        cmf20=_finite(signal_row.get('CMF20'), np.nan),
        roc60=_finite(signal_row.get('ROC60'), np.nan),
        distance_52w_high=_finite(signal_row.get('DIST_52W_HIGH'), np.nan),
        relative_strength60=_finite(signal_row.get('REL_STRENGTH60'), np.nan),
        silent_accumulation_score=smart_score,
        adtv20_idr=_finite(signal_row.get('ADTV20'), np.nan),
        body_atr=_finite(signal_row.get('BODY_ATR'), np.nan),
        close_location=_finite(signal_row.get('CLOSE_LOCATION'), np.nan),
        structural_quality_score=_finite(getattr(plan, 'structural_quality_score', np.nan), quality_score),
    )
    start = signal_pos + 1
    end = min(len(df) - 1, signal_pos + config.backtest_entry_window_bars)
    if start > end:
        event.no_fill_reason = 'Tidak ada bar setelah sinyal'
        return event
    fill_pos: int | None = None
    fill_price: float | None = None
    is_stop_order = 'BUY_STOP' in str(plan.entry_type)
    for pos in range(start, end + 1):
        day = df.iloc[pos]
        day_open, day_high, day_low = (float(day[name]) for name in ('Open', 'High', 'Low'))
        if is_stop_order:
            if day_open > planned_entry + config.max_entry_gap_atr * atr_signal:
                event.no_fill_reason = 'Gap di atas toleransi; order dibatalkan'
                return event
            if day_open >= planned_entry:
                fill_pos, fill_price = (pos, day_open)
                break
            if day_high >= planned_entry:
                fill_pos, fill_price = (pos, planned_entry)
                break
        else:
            if day_open < stop:
                event.no_fill_reason = 'Gap di bawah invalidasi sebelum limit fill'
                return event
            if day_open <= planned_entry:
                fill_pos, fill_price = (pos, day_open)
                break
            if day_low <= planned_entry <= day_high:
                fill_pos, fill_price = (pos, planned_entry)
                break
    if fill_pos is None or fill_price is None:
        event.no_fill_reason = 'Entry tidak tersentuh dalam jendela order'
        return event
    if fill_price <= stop:
        event.no_fill_reason = 'Fill tidak valid terhadap stop'
        return event
    rr1_at_fill = (tp1 - fill_price) / (fill_price - stop)
    if rr1_at_fill < config.min_rr1:
        event.no_fill_reason = 'Gap/fill menurunkan RR1 di bawah minimum'
        return event
    event.filled = True
    event.fill_date = df.index[fill_pos]
    event.fill_wait_bars = fill_pos - signal_pos
    event.entry = fill_price
    last_pos = min(len(df) - 1, fill_pos + config.backtest_horizon_bars - 1)
    exit_pos = last_pos
    exit_price = float(df['Close'].iloc[last_pos])
    result = 'TIME_EXIT'
    tp1_pos: int | None = None
    for pos in range(fill_pos, last_pos + 1):
        day_open = float(df['Open'].iloc[pos])
        day_low = float(df['Low'].iloc[pos])
        day_high = float(df['High'].iloc[pos])
        if day_open <= stop:
            exit_price, exit_pos, result = (day_open, pos, 'LOSS_GAP')
            break
        if day_low <= stop:
            exit_price, exit_pos, result = (stop, pos, 'LOSS')
            break
        if day_high >= tp1:
            exit_price, exit_pos, result = (tp1, pos, 'WIN_TP1')
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

def simulate_setup(df: pd.DataFrame, ticker: str, setup: str, config: ScanConfig) -> list[BacktestEvent]:
    detector = DETECTORS[setup]
    candidates = np.flatnonzero(historical_signal_mask(df, setup).to_numpy())
    events: list[BacktestEvent] = []
    next_allowed = 0
    engine = ScanEngine(config)
    for pos in candidates:
        if pos < max(205, next_allowed) or pos + 1 >= len(df):
            continue
        snapshot = df.iloc[:pos + 1]
        plan = detector(snapshot, ticker)
        blockers, metrics = _historical_gate_inputs(snapshot, config)
        context = _historical_context(snapshot)
        finalized = engine._finalize(plan, snapshot, context, blockers, metrics)
        if finalized['status'] != 'EXECUTION_READY':
            continue
        event = _simulate_order(df, pos, ticker, setup, plan, float(finalized['quality_score']), context.regime, config)
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

def _wilson_interval(successes: int, total: int, z: float=1.96) -> tuple[float, float]:
    if total <= 0:
        return (np.nan, np.nan)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))

def run_walkforward_validation(prepared: dict[str, pd.DataFrame], config: ScanConfig | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or ScanConfig()
    all_events: list[dict[str, object]] = []
    for ticker, frame in prepared.items():
        if len(frame) < 225:
            continue
        for setup in SETUPS:
            all_events.extend((asdict(event) for event in simulate_setup(frame, ticker, setup, cfg)))
    events = pd.DataFrame(all_events)
    if not events.empty:
        events = _assign_oos_folds(events, cfg)
    stats = aggregate_backtest(events, cfg)
    return (stats, events)
from urllib.parse import urlparse
import re

def detect_pullback_continuation(df: pd.DataFrame, ticker: str) -> SetupPlan:
    """Detect a trend-continuation pullback using impulse, contraction and re-acceleration.

    The detector deliberately separates *setup validity* from *entry timing*.
    A valid trend and controlled retracement can become an entry plan, while a
    bullish reclaim/rejection is required for READY_TRIGGER.  Risk and RR remain
    disclosures; price-structure invalidation is the only technical rejection.
    """
    plan = SetupPlan(ticker=ticker, setup='PULLBACK_CONTINUATION', detected=False, setup_score=0.0)
    if df is None or df.empty or len(df) < 205:
        plan.reason = 'Data tren jangka panjang belum cukup'
        return plan

    row, prev = df.iloc[-1], df.iloc[-2]
    close = _finite(row.get('Close'), 0.0)
    atr_v = _finite(row.get('ATR14'), 0.0)
    if close <= 0 or atr_v <= 0:
        plan.reason = 'ATR/harga tidak valid'
        return plan

    ema20 = _finite(row.get('EMA20'), 0.0)
    ema50 = _finite(row.get('EMA50'), 0.0)
    ema200 = _finite(row.get('EMA200'), 0.0)
    ema20_old = _finite(df['EMA20'].iloc[-11], ema20) if 'EMA20' in df else ema20
    ema50_old = _finite(df['EMA50'].iloc[-21], ema50) if 'EMA50' in df else ema50
    slope20 = ema20 / ema20_old - 1.0 if ema20_old > 0 else 0.0
    slope50 = ema50 / ema50_old - 1.0 if ema50_old > 0 else 0.0
    trend_stack = bool(ema20 > ema50 > ema200 and close >= ema50 - 0.20 * atr_v)
    trend_rising = bool(slope20 > 0 and slope50 >= -0.005)
    trend_strength = _finite(row.get('ADX14'), 0.0)
    momentum = bool(_finite(row.get('ROC60'), -1.0) > 0.03 and _finite(row.get('DIST_52W_HIGH'), -1.0) > -0.22)
    relative = bool(_finite(row.get('REL_STRENGTH60'), -1.0) > -0.01)

    # Use a completed prior swing high rather than the current recovery candle.
    lookback_start = max(0, len(df) - 80)
    candidate_end = max(lookback_start + 5, len(df) - 3)
    prior_window = df.iloc[lookback_start:candidate_end]
    pivot_highs = pd.to_numeric(prior_window.get('PIVOT_HIGH'), errors='coerce').dropna() if 'PIVOT_HIGH' in prior_window else pd.Series(dtype=float)
    if not pivot_highs.empty:
        swing_high_time = pivot_highs.index[-1]
        swing_high = float(pivot_highs.iloc[-1])
        swing_high_pos = int(np.flatnonzero(df.index == swing_high_time)[-1])
    else:
        high_series = pd.to_numeric(prior_window['High'], errors='coerce')
        rel_pos = int(np.nanargmax(high_series.to_numpy(dtype=float)))
        swing_high_pos = lookback_start + rel_pos
        swing_high_time = df.index[swing_high_pos]
        swing_high = _finite(df['High'].iloc[swing_high_pos], close)

    impulse_start = max(lookback_start, swing_high_pos - 45)
    impulse_slice = df.iloc[impulse_start:swing_high_pos + 1]
    pivot_lows = pd.to_numeric(impulse_slice.get('PIVOT_LOW'), errors='coerce').dropna() if 'PIVOT_LOW' in impulse_slice else pd.Series(dtype=float)
    impulse_low = float(pivot_lows.iloc[-1]) if not pivot_lows.empty else _finite(impulse_slice['Low'].min(), swing_high - 2.0 * atr_v)
    impulse_height = max(0.0, swing_high - impulse_low)
    impulse_valid = bool(impulse_height >= 1.8 * atr_v)

    pullback_slice = df.iloc[min(swing_high_pos + 1, len(df) - 1):]
    if pullback_slice.empty:
        pullback_slice = df.iloc[-8:]
    pullback_low = _finite(pullback_slice['Low'].min(), close)
    retracement = (swing_high - pullback_low) / impulse_height if impulse_height > 0 else np.nan
    depth_ideal = bool(np.isfinite(retracement) and 0.15 <= retracement <= 0.58)
    depth_acceptable = bool(np.isfinite(retracement) and 0.08 <= retracement <= 0.68)

    vwap20 = _finite(row.get('VWAP20'), ema20)
    value_floor = max(ema50, min(ema20, vwap20))
    value_ceiling = max(ema20, vwap20)
    zone_low = value_floor - 0.30 * atr_v
    zone_high = value_ceiling + 0.30 * atr_v
    recent = df.iloc[-8:]
    touched_value = bool((pd.to_numeric(recent['Low'], errors='coerce') <= zone_high).any())
    held_structure = bool(
        _finite(recent['Close'].min(), close) >= ema50 - 0.40 * atr_v
        and close >= value_floor - 0.15 * atr_v
    )

    pullback_volume = _finite(pd.to_numeric(pullback_slice['Volume'], errors='coerce').tail(5).mean(), 0.0)
    impulse_volume = _finite(pd.to_numeric(impulse_slice['Volume'], errors='coerce').tail(min(10, len(impulse_slice))).mean(), 0.0)
    vol_ma20 = _finite(row.get('VOL_MA20'), impulse_volume)
    volume_dryup_ratio = pullback_volume / max(impulse_volume, vol_ma20, 1.0)
    volume_contraction = bool(volume_dryup_ratio <= 0.90)

    recent_range_atr = ((pd.to_numeric(recent['High'], errors='coerce') - pd.to_numeric(recent['Low'], errors='coerce')) / pd.to_numeric(recent['ATR14'], errors='coerce').replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    prior_range = ((pd.to_numeric(df['High'].iloc[-28:-8], errors='coerce') - pd.to_numeric(df['Low'].iloc[-28:-8], errors='coerce')) / pd.to_numeric(df['ATR14'].iloc[-28:-8], errors='coerce').replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    recent_range_mean = _finite(recent_range_atr.tail(4).mean(), 1.0)
    prior_range_mean = _finite(prior_range.mean(), 1.0)
    range_contraction = bool(recent_range_mean <= 0.90 * max(prior_range_mean, 0.5))

    cmf_v = _finite(row.get('CMF20'), -1.0)
    obv_up = _finite(row.get('OBV_SLOPE10'), -1.0) > 0
    flow_ok = bool(cmf_v >= -0.03 and obv_up)
    close_location = _finite(row.get('CLOSE_LOCATION'), 0.5)
    bull_rejection = bool(row.get('BULL_REJECTION', False))
    price_reclaim = bool(close > _finite(prev.get('High'), close) and close > _finite(row.get('Open'), close) and close_location >= 0.62)
    trigger_volume = _finite(row.get('VOL_RATIO'), 0.0) >= 0.85
    confirmation = bool((bull_rejection or price_reclaim) and trigger_volume)
    distribution = bool(_finite(row.get('VOL_RATIO'), 0.0) >= 1.6 and close_location <= 0.35 and cmf_v < -0.08)

    plan.detected = bool(
        trend_stack and trend_rising and momentum and impulse_valid and depth_acceptable
        and touched_value and held_structure and not distribution
    )
    if not plan.detected:
        missing: list[str] = []
        if not trend_stack:
            missing.append('EMA20>EMA50>EMA200/hold EMA50 belum valid')
        if not trend_rising:
            missing.append('slope EMA belum naik')
        if not momentum:
            missing.append('momentum/posisi 52-minggu belum mendukung')
        if not impulse_valid:
            missing.append('impulse sebelum pullback terlalu kecil')
        if not depth_acceptable:
            missing.append('kedalaman retracement tidak sehat')
        if not touched_value:
            missing.append('harga belum menyentuh value area')
        if not held_structure:
            missing.append('pullback gagal bertahan di atas struktur')
        if distribution:
            missing.append('indikasi distribusi pada candle terakhir')
        plan.blockers.extend(missing)
        plan.reason = 'Belum memenuhi impulse–controlled pullback–hold structure: ' + ' • '.join(missing)
        return plan

    score = 0.0
    score += 20.0 if trend_stack and trend_rising else 12.0
    score += 12.0 if trend_strength >= 22 else 8.0 if trend_strength >= 16 else 4.0
    score += 12.0 if momentum and relative else 7.0 if momentum else 3.0
    score += 16.0 if depth_ideal else 10.0
    score += 12.0 if volume_contraction else 5.0
    score += 10.0 if range_contraction else 4.0
    score += 10.0 if flow_ok else 4.0 if cmf_v >= -0.03 else 0.0
    score += 8.0 if confirmation else 3.0
    plan.setup_score = round(min(100.0, score), 1)

    distance = _distance_to_zone(close, zone_low, zone_high, atr_v)
    if confirmation and distance <= 0.45:
        raw_entry = max(close, _finite(row.get('High'), close) + idx_tick_size(close))
        plan.entry_type = 'BUY_STOP_REACCELERATION'
        plan.action = 'READY_TRIGGER'
    else:
        raw_entry = (zone_low + zone_high) / 2.0
        plan.entry_type = 'LIMIT_VALUE_ZONE_THEN_RECLAIM'
        plan.action = 'WAIT_PULLBACK_CONFIRMATION'

    structural_low = pullback_low
    last_pivot_low = _finite(row.get('LAST_PIVOT_LOW'), np.nan)
    if np.isfinite(last_pivot_low) and last_pivot_low < close and last_pivot_low >= close - 4.0 * atr_v:
        structural_low = min(structural_low, last_pivot_low)
    raw_stop = structural_low - 0.18 * atr_v

    plan.signal_date = swing_high_time
    touch_mask = (df['Low'] <= zone_high) & (df['Close'] >= ema50 - 0.40 * df['ATR14'])
    created = _last_true_index(touch_mask, min(15, len(df))) or df.index[-1]
    plan.zone_created_date = created
    plan.zone_age_bars = _bars_since(df, created)
    plan.valid_until = pd.Timestamp(df.index[-1]) + pd.offsets.BDay(8)
    plan.entry_low, plan.entry_high = zone_low, zone_high
    plan.trigger = _finite(row.get('High'), close) + idx_tick_size(close)
    plan.distance_atr = round(distance, 2)
    plan.evidence = [
        'EMA stack dan slope bullish',
        f'Impulse {impulse_height / atr_v:.1f} ATR sebelum pullback',
        f'Retracement {retracement:.0%} dari impulse' if np.isfinite(retracement) else 'Retracement terkontrol',
        'Pullback bertahan di value area',
    ]
    if volume_contraction:
        plan.evidence.append(f'Volume dry-up {volume_dryup_ratio:.2f}x')
    else:
        plan.blockers.append('Volume pullback belum kontraksi ideal')
    if range_contraction:
        plan.evidence.append('Range pullback berkontraksi')
    else:
        plan.blockers.append('Range pullback belum berkontraksi')
    if flow_ok:
        plan.evidence.append('CMF/OBV mendukung')
    else:
        plan.blockers.append('Flow proxy belum ideal')
    if confirmation:
        plan.evidence.append('Bullish re-acceleration/reclaim terkonfirmasi')
    else:
        plan.blockers.append('Menunggu bullish reclaim/re-acceleration')
    plan.reason = 'Trend continuation setelah impulse, retracement terkontrol, dan hold value area'

    structural_levels = [
        (swing_high, 'PRIOR_IMPULSE_HIGH'),
        (_finite(row.get('HIGH55_PREV'), np.nan), 'PRIOR_55D_HIGH'),
        (_finite(row.get('HIGH252'), np.nan), 'PRIOR_52W_HIGH'),
    ]
    plan = _plan_prices(
        plan, df, atr_v, raw_entry, raw_stop,
        structural_levels=structural_levels,
        projection_origin=swing_high,
        projection_height=max(impulse_height, zone_high - zone_low),
    )
    if plan.detected and not plan.invalidated:
        if np.isfinite(_finite(plan.rr2, np.nan)) and _finite(plan.rr2, 0.0) < 1.5:
            plan.blockers.append(f'Overhead supply membatasi RR2 struktur ke {_finite(plan.rr2):.2f}')
            plan.setup_score = max(0.0, plan.setup_score - 8.0)
        if not relative:
            plan.blockers.append('Relative strength belum mengungguli IHSG')
            plan.setup_score = max(0.0, plan.setup_score - 5.0)
    return plan

def detect_breakout_retest(df: pd.DataFrame, ticker: str) -> SetupPlan:
    """Breakout-retest with base, acceptance and false-break diagnostics.

    The detector keeps the original structural requirement (55-day breakout)
    but distinguishes a clean institutional breakout from a one-bar spike. A
    valid breakout needs a constructive base, a strong close through supply,
    post-breakout acceptance, and a controlled retest. Risk warnings reduce the
    edge score instead of fabricating a target or silently promoting the row.
    """
    plan = _detect_breakout_v21(df, ticker)
    if not plan.detected or df.empty:
        return plan
    row = df.iloc[-1]
    close = _finite(row['Close'])
    atr_v = _finite(row['ATR14'])
    if close <= 0 or atr_v <= 0:
        return plan

    signal_date = pd.Timestamp(plan.signal_date) if plan.signal_date is not None else None
    if signal_date is None or signal_date not in df.index:
        return plan
    pos = int(np.flatnonzero(df.index == signal_date)[-1])
    breakout_row = df.iloc[pos]
    resistance = _finite(breakout_row.get('HIGH55_PREV'), _finite(plan.entry_low, close))
    base = df.iloc[max(0, pos - 30):pos]
    post = df.iloc[pos + 1:]

    # Pre-breakout base quality: enough compression, limited depth and repeated
    # closes near the upper half of the range imply supply absorption.
    base_high = _finite(base['High'].max(), resistance) if not base.empty else resistance
    base_low = _finite(base['Low'].min(), resistance - atr_v) if not base.empty else resistance - atr_v
    base_depth = (base_high - base_low) / max(resistance, 1e-9)
    recent_range = pd.to_numeric(base.get('High'), errors='coerce') - pd.to_numeric(base.get('Low'), errors='coerce') if not base.empty else pd.Series(dtype=float)
    prior_range = (pd.to_numeric(df['High'].iloc[max(0, pos - 60):max(0, pos - 30)], errors='coerce') - pd.to_numeric(df['Low'].iloc[max(0, pos - 60):max(0, pos - 30)], errors='coerce'))
    contraction_ratio = _finite(recent_range.tail(10).median(), np.nan) / max(_finite(prior_range.median(), np.nan), 1e-9) if (not recent_range.empty and not prior_range.empty) else np.nan
    upper_half_ratio = float((pd.to_numeric(base['Close'], errors='coerce') >= (base_low + 0.55 * (base_high - base_low))).mean()) if not base.empty else 0.0
    base_quality = 0.0
    base_quality += 35.0 if base_depth <= 0.14 else 24.0 if base_depth <= 0.22 else 10.0 if base_depth <= 0.30 else 0.0
    base_quality += 30.0 if np.isfinite(contraction_ratio) and contraction_ratio <= 0.80 else 18.0 if np.isfinite(contraction_ratio) and contraction_ratio <= 1.0 else 5.0
    base_quality += 20.0 if upper_half_ratio >= 0.60 else 12.0 if upper_half_ratio >= 0.45 else 4.0
    base_quality += 15.0 if len(base) >= 15 else 8.0
    base_quality = float(min(100.0, base_quality))

    # Breakout candle quality: close location, wick control, displacement,
    # volume and relative strength.
    br_range = max(_finite(breakout_row['High']) - _finite(breakout_row['Low']), idx_tick_size(close))
    close_location = (_finite(breakout_row['Close']) - _finite(breakout_row['Low'])) / br_range
    upper_wick = (_finite(breakout_row['High']) - max(_finite(breakout_row['Open']), _finite(breakout_row['Close']))) / br_range
    vol_ratio = _finite(breakout_row.get('VOL_RATIO'), 0.0)
    body_atr = _finite(breakout_row.get('BODY_ATR'), 0.0)
    rs_ok = _finite(breakout_row.get('REL_STRENGTH60'), _finite(row.get('REL_STRENGTH60'), 0.0)) > 0
    breakout_quality = 0.0
    breakout_quality += 25.0 if close_location >= 0.80 else 16.0 if close_location >= 0.65 else 5.0
    breakout_quality += 20.0 if upper_wick <= 0.15 else 12.0 if upper_wick <= 0.30 else 0.0
    breakout_quality += 25.0 if vol_ratio >= 1.8 else 18.0 if vol_ratio >= 1.35 else 8.0
    breakout_quality += 20.0 if body_atr >= 0.8 else 12.0 if body_atr >= 0.5 else 4.0
    breakout_quality += 10.0 if rs_ok else 0.0
    breakout_quality = float(min(100.0, breakout_quality))

    # Post-breakout acceptance and retest quality.
    if post.empty:
        hold_ratio = 1.0
        retest_volume_ratio = np.nan
        false_closes = 0
    else:
        atr_post = pd.to_numeric(post['ATR14'], errors='coerce').replace(0, np.nan)
        hold_ratio = float((pd.to_numeric(post['Close'], errors='coerce') >= resistance - 0.20 * atr_post).mean())
        false_closes = int((pd.to_numeric(post['Close'], errors='coerce') < resistance - 0.45 * atr_post).sum())
        near_retest = post[(pd.to_numeric(post['Low'], errors='coerce') <= resistance + 0.45 * atr_post) & (pd.to_numeric(post['Low'], errors='coerce') >= resistance - 1.0 * atr_post)]
        retest_volume_ratio = _finite(pd.to_numeric(near_retest.get('Volume'), errors='coerce').median(), np.nan) / max(_finite(breakout_row.get('Volume'), np.nan), 1e-9) if not near_retest.empty else np.nan
    current_confirmation = bool(row['BULL_REJECTION']) or (close > _finite(df['High'].iloc[-2]) and close > _finite(row['Open']))
    in_zone = plan.entry_low is not None and plan.entry_high is not None and (_distance_to_zone(close, float(plan.entry_low), float(plan.entry_high), atr_v) <= 0.45)
    retest_seen = 'Retest level breakout terdeteksi' in set(plan.evidence)
    retest_quality = 0.0
    retest_quality += 30.0 if retest_seen else 8.0
    retest_quality += 25.0 if current_confirmation else 8.0
    retest_quality += 20.0 if hold_ratio >= 0.80 else 12.0 if hold_ratio >= 0.60 else 0.0
    retest_quality += 15.0 if np.isfinite(retest_volume_ratio) and retest_volume_ratio <= 0.75 else 8.0 if np.isfinite(retest_volume_ratio) and retest_volume_ratio <= 1.0 else 3.0
    retest_quality += 10.0 if in_zone else 3.0
    retest_quality = float(min(100.0, retest_quality))

    extension_atr = max(0.0, (close - float(plan.entry_high)) / atr_v) if plan.entry_high is not None else 0.0
    failure_penalty = min(100.0, 22.0 * false_closes + 18.0 * max(0.0, extension_atr - 1.0) + (18.0 if upper_wick > 0.35 else 0.0))
    edge = max(0.0, min(100.0, 0.28 * base_quality + 0.34 * breakout_quality + 0.38 * retest_quality - 0.45 * failure_penalty))

    trend = _finite(row['EMA20']) > _finite(row['EMA50']) > _finite(row['EMA200'])
    if not trend:
        plan.detected = False
        plan.setup_score = min(plan.setup_score, 65.0)
        plan.action = 'NO_SETUP'
        plan.reason = 'Breakout ditolak: EMA20 > EMA50 > EMA200 tidak terpenuhi'
        return plan
    if false_closes >= 2:
        plan.detected = False
        plan.invalidated = True
        plan.action = 'NO_SETUP'
        plan.setup_score = min(plan.setup_score, 60.0)
        plan.reason = 'Breakout gagal: acceptance di atas resistance tidak bertahan'
        plan.blockers.append(f'{false_closes} penutupan kembali di bawah level breakout')
        return plan
    if plan.action.startswith('READY') and not (current_confirmation and in_zone):
        plan.action = 'WAIT_CURRENT_RETEST_CONFIRMATION'
        plan.blockers.append('Menunggu retest/reclaim terbaru di area breakout')
    if extension_atr > 1.5:
        plan.action = 'WAIT_RETEST'
        plan.blockers.append(f'Harga sudah {extension_atr:.2f} ATR di atas zona retest')

    plan.structural_quality_score = round(base_quality, 1)
    plan.confirmation_quality_score = round(retest_quality, 1)
    plan.supply_demand_score = round(breakout_quality, 1)
    plan.failure_risk_score = round(failure_penalty, 1)
    plan.extension_atr = round(extension_atr, 2)
    plan.setup_detail = (
        f'Base depth {base_depth:.1%}; contraction {contraction_ratio:.2f}x; '
        f'breakout CLV {close_location:.2f}; hold ratio {hold_ratio:.0%}; '
        f'retest volume {retest_volume_ratio:.2f}x' if np.isfinite(retest_volume_ratio)
        else f'Base depth {base_depth:.1%}; breakout CLV {close_location:.2f}; hold ratio {hold_ratio:.0%}'
    )
    plan.setup_score = round(max(0.0, min(100.0, 0.45 * plan.setup_score + 0.55 * edge)), 1)
    if base_quality >= 70:
        plan.evidence.append('Base pra-breakout rapat dan supply terkompresi')
    if breakout_quality >= 75:
        plan.evidence.append('Breakout candle berkualitas tinggi')
    if retest_quality >= 70:
        plan.evidence.append('Acceptance/retest berkualitas')
    if failure_penalty >= 25:
        plan.blockers.append(f'False-break/extension penalty {failure_penalty:.0f}')
    plan.reason = 'Breakout-retest dinilai dari base, acceptance, retest, flow, dan false-break risk'
    return plan


def detect_reversal_accumulation(df: pd.DataFrame, ticker: str) -> SetupPlan:
    """Reversal accumulation with absorption and structure-change scoring."""
    plan = _detect_reversal_v21(df, ticker)
    if not plan.detected or df.empty:
        return plan
    row = df.iloc[-1]
    close = _finite(row['Close'])
    atr_v = _finite(row['ATR14'])
    base = df.iloc[-30:]
    prior = df.iloc[-150:-30]
    base_low = _finite(base['Low'].min(), close)
    base_high = _finite(base['High'].max(), close)
    prior_high = _finite(prior['High'].max(), base_high) if not prior.empty else base_high
    base_width = (base_high - base_low) / max(close, 1e-9)
    decline = base_low / max(prior_high, 1e-9) - 1.0
    range_contraction = _finite(row.get('RANGE_CONTRACTION20'), 2.0)
    upper_base_closes = float((pd.to_numeric(base['Close'], errors='coerce') >= base_low + 0.55 * (base_high - base_low)).mean())

    base_quality = 0.0
    base_quality += 28.0 if -0.45 <= decline <= -0.12 else 18.0 if decline < -0.08 else 5.0
    base_quality += 32.0 if base_width <= 0.16 else 22.0 if base_width <= 0.24 else 8.0 if base_width <= 0.30 else 0.0
    base_quality += 22.0 if range_contraction <= 0.80 else 13.0 if range_contraction <= 1.0 else 3.0
    base_quality += 18.0 if upper_base_closes >= 0.60 else 10.0 if upper_base_closes >= 0.45 else 2.0
    base_quality = float(min(100.0, base_quality))

    accumulation_score, up_down = _silent_accumulation_metrics(df)
    cmf_now = _finite(row.get('CMF20'), -1.0)
    obv_slope = _finite(row.get('OBV_SLOPE10'), -1.0)
    down_days = base[pd.to_numeric(base['Close'], errors='coerce') < pd.to_numeric(base['Open'], errors='coerce')]
    up_days = base[pd.to_numeric(base['Close'], errors='coerce') >= pd.to_numeric(base['Open'], errors='coerce')]
    down_volume = _finite(pd.to_numeric(down_days.get('Volume'), errors='coerce').median(), np.nan)
    up_volume = _finite(pd.to_numeric(up_days.get('Volume'), errors='coerce').median(), np.nan)
    volume_absorption = up_volume / max(down_volume, 1e-9) if np.isfinite(up_volume) and np.isfinite(down_volume) else np.nan
    absorption = 0.0
    absorption += 30.0 if cmf_now >= 0.08 else 20.0 if cmf_now >= 0.03 else 8.0 if cmf_now > 0 else 0.0
    absorption += 25.0 if obv_slope > 0 else 0.0
    absorption += 25.0 if up_down >= 1.35 else 16.0 if up_down >= 1.10 else 5.0
    absorption += 20.0 if np.isfinite(volume_absorption) and volume_absorption >= 1.25 else 10.0 if np.isfinite(volume_absorption) and volume_absorption >= 1.0 else 3.0
    absorption = float(min(100.0, absorption))

    higher_low = _finite(df['Low'].iloc[-5:].min()) > base_low
    reclaim_ema20 = close > _finite(row.get('EMA20'))
    reclaim_ema50 = close > _finite(row.get('EMA50'))
    break_prev_high = close > _finite(df['High'].iloc[-2])
    choch_evidence = 'CHOCH/BOS bullish terkonfirmasi' in set(plan.evidence)
    structure_change = 0.0
    structure_change += 35.0 if choch_evidence else 8.0
    structure_change += 25.0 if higher_low else 5.0
    structure_change += 20.0 if reclaim_ema20 else 0.0
    structure_change += 10.0 if reclaim_ema50 else 0.0
    structure_change += 10.0 if break_prev_high else 0.0
    structure_change = float(min(100.0, structure_change))

    breakdown_attempts = int((pd.to_numeric(base['Close'], errors='coerce') < base_low + 0.10 * (base_high - base_low)).sum())
    failure_penalty = min(100.0, (25.0 if base_width > 0.24 else 0.0) + 8.0 * max(0, breakdown_attempts - 3) + (20.0 if cmf_now < 0 else 0.0) + (18.0 if close < base_low + 0.30 * (base_high - base_low) else 0.0))
    edge = max(0.0, min(100.0, 0.30 * base_quality + 0.35 * absorption + 0.35 * structure_change - 0.45 * failure_penalty))

    if base_width > 0.28:
        plan.detected = False
        plan.action = 'NO_SETUP'
        plan.setup_score = min(plan.setup_score, 62.0)
        plan.reason = f'Base terlalu lebar ({base_width:.1%}); reversal belum terkontrol'
        return plan
    if plan.action.startswith('READY') and not (higher_low and absorption >= 55.0 and structure_change >= 65.0):
        plan.action = 'WAIT_HIGHER_LOW_AND_FLOW'
        plan.blockers.append('Higher-low, absorption, dan reclaim struktur belum lengkap')
    if failure_penalty >= 45.0:
        plan.action = 'WAIT_CHOCH'
        plan.blockers.append(f'Failure risk reversal tinggi ({failure_penalty:.0f})')

    plan.structural_quality_score = round(base_quality, 1)
    plan.confirmation_quality_score = round(structure_change, 1)
    plan.supply_demand_score = round(absorption, 1)
    plan.failure_risk_score = round(failure_penalty, 1)
    plan.extension_atr = round(max(0.0, _distance_to_zone(close, float(plan.entry_low), float(plan.entry_high), atr_v)) if plan.entry_low is not None and plan.entry_high is not None else 0.0, 2)
    plan.setup_detail = (
        f'Decline {decline:.1%}; base width {base_width:.1%}; accumulation {accumulation_score:.0f}; '
        f'up/down value {up_down:.2f}x; structure change {structure_change:.0f}'
    )
    plan.setup_score = round(max(0.0, min(100.0, 0.42 * plan.setup_score + 0.58 * edge)), 1)
    if base_quality >= 70:
        plan.evidence.append('Base reversal rapat dan terkontrol')
    if absorption >= 70:
        plan.evidence.append('Supply absorption/akumulasi kuat')
    if structure_change >= 70:
        plan.evidence.append('Perubahan struktur bullish berkualitas')
    if failure_penalty >= 25:
        plan.blockers.append(f'Reversal failure penalty {failure_penalty:.0f}')
    plan.reason = 'Reversal dinilai dari base quality, absorption, higher-low, dan perubahan struktur'
    return plan


def detect_unicorn_sniper(df: pd.DataFrame, ticker: str) -> SetupPlan:
    plan = _detect_unicorn_v21(df, ticker)
    if not plan.detected or df.empty:
        return plan
    evidence = set(plan.evidence)
    strict_ob = 'FVG overlap dengan order-block proxy' in evidence
    strict_discount = 'Zona berada di discount dealing range' in evidence
    volume_ok = False
    if plan.zone_created_date is not None and plan.zone_created_date in df.index:
        pos = int(np.flatnonzero(df.index == plan.zone_created_date)[-1])
        start = max(0, pos - 3)
        volume_ok = bool((pd.to_numeric(df['VOL_RATIO'].iloc[start:pos + 1], errors='coerce') >= 1.2).any())
    if not (strict_ob and strict_discount and volume_ok):
        plan.setup_score = min(plan.setup_score, 74.0)
        if plan.action.startswith('READY'):
            plan.action = 'WAIT_STRICT_UNICORN_CONFLUENCE'
        plan.blockers.append('Strict Unicorn memerlukan OB×FVG overlap, discount, dan volume displacement ≥1.20x')
        plan.reason = 'Sweep–BOS–FVG valid, tetapi belum strict Unicorn execution grade'
    return plan
SETUP_DETECTORS = (detect_pullback_continuation, detect_breakout_retest, detect_reversal_accumulation, detect_unicorn_sniper)

def _is_exact_official_idx_url(value: object) -> bool:
    try:
        parsed = urlparse(str(value).strip())
    except Exception:
        return False
    return parsed.scheme == 'https' and (parsed.hostname or '').lower() in {'idx.co.id', 'www.idx.co.id'} and bool(parsed.path and parsed.path != '/')

def parse_market_status_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    """Strict parser retained for audit/import compatibility.

    A row is verified only when all status flags are explicitly present, the
    coverage flag is true, and the source host is exactly idx.co.id.
    """
    frame = _read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {'ticker', 'as_of', 'suspended', 'special_monitoring', 'fca', 'special_notation', 'corporate_action', 'source_url', 'coverage_complete', 'verification_method'}
    if not {'ticker', 'as_of'}.issubset(frame.columns):
        raise ValueError('Market-status CSV wajib memiliki kolom ticker dan as_of')
    out = pd.DataFrame(index=frame.index)
    out['ticker'] = frame['ticker'].map(normalize_idx_ticker)
    out['market_status_asof'] = pd.to_datetime(frame['as_of'], errors='coerce')
    for column in ('suspended', 'special_monitoring', 'fca', 'corporate_action', 'sharia'):
        out[column] = frame[column].map(_truthy) if column in frame else False
    out['special_notation'] = _column(frame, 'special_notation', '').fillna('').astype(str).str.strip()
    out['market_status_source'] = _column(frame, 'source_url', '').fillna('').astype(str).str.strip()
    out['market_status_method'] = _column(frame, 'verification_method', '').fillna('').astype(str).str.strip()
    explicit_columns = required.issubset(frame.columns)
    coverage = _column(frame, 'coverage_complete', False).map(_truthy)
    official = out['market_status_source'].map(_is_exact_official_idx_url)
    out['market_status_verified'] = explicit_columns & coverage & official & out['market_status_asof'].notna() & out['market_status_method'].str.len().gt(0)
    out['market_status_components'] = _column(frame, 'coverage_components', 'MANUAL_IMPORT').fillna('').astype(str)
    return out.dropna(subset=['ticker']).drop_duplicates('ticker', keep='last')
IDX_AUTOMATION_URLS = {'stock_list': 'https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham/', 'watchlist': 'https://www.idx.co.id/id/perusahaan-tercatat/daftar-efek-pemantauan-khusus', 'suspension': 'https://www.idx.co.id/id/berita/suspensi', 'long_suspension': 'https://www.idx.co.id/id/perusahaan-tercatat/suspensi-lebih-dari-6-bulan/', 'corporate_actions': 'https://www.idx.co.id/id/perusahaan-tercatat/aksi-korporasi'}

def _html_text(value: str) -> str:
    clean = re.sub('<script\\b[^>]*>.*?</script>', ' ', value, flags=re.I | re.S)
    clean = re.sub('<style\\b[^>]*>.*?</style>', ' ', clean, flags=re.I | re.S)
    clean = re.sub('<[^>]+>', ' ', clean)
    return re.sub('\\s+', ' ', clean).upper()

def _requested_mentions(text: str, tickers: Iterable[str]) -> set[str]:
    found: set[str] = set()
    for ticker in tickers:
        code = ticker.replace('.JK', '').upper()
        if re.search(f'(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])', text):
            found.add(ticker)
    return found

def fetch_automatic_market_status(tickers: Iterable[str], timeout: int=20) -> pd.DataFrame:
    """Fetch official IDX public pages once and build a conservative blocklist.

    Absence is accepted only when every required page was successfully fetched
    and its semantic marker is present. Any provider failure leaves rows
    unverified, therefore incapable of EXECUTION_READY.
    """
    names = list(dict.fromkeys(tickers))
    if not names:
        return pd.DataFrame()
    now = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None)
    pages, errors = _fetch_official_idx_pages(timeout=timeout)
    text = {key: _html_text(value) for key, value in pages.items()}
    semantic_ok = {'stock_list': 'DAFTAR SAHAM' in text.get('stock_list', '') or 'STOCK LIST' in text.get('stock_list', ''), 'watchlist': 'PEMANTAUAN KHUSUS' in text.get('watchlist', '') or 'SPECIAL MONITORING' in text.get('watchlist', ''), 'suspension': 'SUSPENSI' in text.get('suspension', '') or 'PENGHENTIAN SEMENTARA' in text.get('suspension', ''), 'long_suspension': '6 BULAN' in text.get('long_suspension', '') or '6 MONTH' in text.get('long_suspension', ''), 'corporate_actions': 'AKSI KORPORASI' in text.get('corporate_actions', '') or 'CORPORATE ACTION' in text.get('corporate_actions', '')}
    coverage_complete = len(errors) == 0 and all(semantic_ok.values())
    watchlist = _requested_mentions(text.get('watchlist', ''), names)
    suspended = _requested_mentions(text.get('suspension', '') + ' ' + text.get('long_suspension', ''), names)
    corporate = _requested_mentions(text.get('corporate_actions', ''), names)
    listed = _requested_mentions(text.get('stock_list', ''), names)
    source_join = ' | '.join(IDX_AUTOMATION_URLS.values())
    components = ','.join((key for key, ok in semantic_ok.items() if ok))
    error_text = ' | '.join((f'{key}:{value}' for key, value in errors.items()))
    rows = []
    for ticker in names:
        row_verified = coverage_complete and ticker in listed
        rows.append({'ticker': ticker, 'market_status_asof': now, 'suspended': ticker in suspended, 'special_monitoring': ticker in watchlist, 'fca': ticker in watchlist, 'special_notation': 'X/WATCHLIST' if ticker in watchlist else '', 'corporate_action': ticker in corporate, 'sharia': False, 'market_status_source': source_join, 'market_status_method': 'OFFICIAL_IDX_AUTOMATED_SCREEN', 'market_status_components': components, 'market_status_error': error_text, 'market_status_verified': bool(row_verified)})
    return pd.DataFrame(rows)

def parse_news_review_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    """Strict review parser. Empty/default rows can never mean COMPLETE."""
    frame = _read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if 'ticker' not in frame or 'reviewed_at' not in frame:
        raise ValueError('News-review CSV wajib memiliki kolom ticker dan reviewed_at')
    frame['ticker'] = frame['ticker'].map(normalize_idx_ticker)
    frame['news_reviewed_at'] = pd.to_datetime(frame['reviewed_at'], errors='coerce')
    frame['news_review_status'] = _column(frame, 'review_status', 'MISSING').fillna('MISSING').astype(str).str.upper()
    frame['news_title'] = _column(frame, 'title', '').fillna('').astype(str)
    frame['news_sentiment'] = _column(frame, 'sentiment', 'NEUTRAL').fillna('NEUTRAL').astype(str).str.upper()
    frame['news_materiality'] = _column(frame, 'materiality', 'LOW').fillna('LOW').astype(str).str.upper()
    frame['news_source_url'] = _column(frame, 'source_url', '').fillna('').astype(str)
    frame['news_verified'] = _column(frame, 'verified', False).map(_truthy)
    frame['provider_query_ok'] = _column(frame, 'provider_query_ok', False).map(_truthy)
    frame['items_reviewed'] = pd.to_numeric(_column(frame, 'items_reviewed', np.nan), errors='coerce')
    frame['coverage_start'] = pd.to_datetime(_column(frame, 'coverage_start', pd.NaT), errors='coerce')
    frame['coverage_end'] = pd.to_datetime(_column(frame, 'coverage_end', pd.NaT), errors='coerce')
    frame['news_provider'] = _column(frame, 'provider', '').fillna('').astype(str)
    rows: list[dict[str, object]] = []
    for ticker, group in frame.dropna(subset=['ticker']).groupby('ticker', sort=False):
        group = group.sort_values('news_reviewed_at')
        material = group[group['news_title'].str.len().gt(0)]
        severe = material[(material['news_sentiment'] == 'NEGATIVE') & material['news_materiality'].isin(['HIGH', 'CRITICAL']) & material['news_verified']]
        latest = group.iloc[-1]
        structural_complete = bool(latest['provider_query_ok'] and pd.notna(latest['items_reviewed']) and (latest['items_reviewed'] >= 0) and pd.notna(latest['coverage_start']) and pd.notna(latest['coverage_end']) and bool(str(latest['news_provider']).strip()))
        status = str(latest['news_review_status']).upper()
        if status == 'COMPLETE' and (not structural_complete):
            status = 'INCOMPLETE'
        rows.append({'ticker': ticker, 'news_reviewed_at': latest['news_reviewed_at'], 'news_review_status': status, 'provider_query_ok': bool(latest['provider_query_ok']), 'items_reviewed': int(latest['items_reviewed']) if pd.notna(latest['items_reviewed']) else np.nan, 'coverage_start': latest['coverage_start'], 'coverage_end': latest['coverage_end'], 'news_provider': latest['news_provider'], 'verified_catalyst_count': int(((material['news_sentiment'] == 'POSITIVE') & material['news_verified']).sum()), 'verified_negative_count': int(((material['news_sentiment'] == 'NEGATIVE') & material['news_verified']).sum()), 'severe_negative_news': bool(len(severe)), 'ambiguous_material_news': False, 'catalyst_summary': ' | '.join(material.tail(3)['news_title'].tolist()), 'news_sources': ' | '.join(material.tail(3)['news_source_url'].tolist())})
    return pd.DataFrame(rows)
_NEGATIVE_NEWS_TERMS = ('SUSPENSI', 'SUSPENSION', 'GAGAL BAYAR', 'DEFAULT', 'PAILIT', 'BANKRUPTCY', 'PKPU', 'FRAUD', 'KORUPSI', 'CORRUPTION', 'DELISTING', 'PENIPUAN', 'PENYIDIKAN', 'INVESTIGATION', 'GUGATAN', 'LAWSUIT', 'EKUITAS NEGATIF', 'DISCLAIMER OPINION', 'ADVERSE OPINION', 'REVERSE STOCK')
_EVENT_RISK_TERMS = ('RIGHTS ISSUE', 'HAK MEMESAN EFEK TERLEBIH DAHULU', 'PRIVATE PLACEMENT', 'MERGER', 'AKUISISI', 'ACQUISITION', 'STOCK SPLIT', 'DIVESTASI', 'TENDER OFFER', 'MATERIAL TRANSACTION', 'TRANSAKSI MATERIAL')

def _news_item_fields(item: dict[str, Any]) -> tuple[str, str, str, pd.Timestamp | None]:
    content = item.get('content') if isinstance(item.get('content'), dict) else item
    title = str(content.get('title') or item.get('title') or '').strip()
    summary = str(content.get('summary') or content.get('description') or item.get('summary') or '').strip()
    canonical = content.get('canonicalUrl') if isinstance(content.get('canonicalUrl'), dict) else {}
    click = content.get('clickThroughUrl') if isinstance(content.get('clickThroughUrl'), dict) else {}
    url = str(canonical.get('url') or click.get('url') or content.get('link') or item.get('link') or '').strip()
    raw_date = content.get('pubDate') or item.get('providerPublishTime') or item.get('pubDate')
    published = None
    try:
        if isinstance(raw_date, (int, float)):
            published = pd.to_datetime(raw_date, unit='s', utc=True).tz_convert('Asia/Jakarta').tz_localize(None)
        elif raw_date:
            parsed = pd.to_datetime(raw_date, utc=True, errors='coerce')
            if pd.notna(parsed):
                published = parsed.tz_convert('Asia/Jakarta').tz_localize(None)
    except Exception:
        published = None
    return (title, summary, url, published)

def _assign_oos_folds(events: pd.DataFrame, config: ScanConfig) -> pd.DataFrame:
    out = events.copy()
    out['signal_date'] = pd.to_datetime(out['signal_date'], errors='coerce')
    out = out.sort_values(['signal_date', 'ticker', 'setup']).reset_index(drop=True)
    unique_dates = np.array(sorted(out['signal_date'].dropna().unique()))
    out['is_oos'] = False
    out['oos_fold'] = 0
    out['oos_eligible'] = len(unique_dates) >= config.min_oos_unique_dates
    if not bool(out['oos_eligible'].iloc[0]) if len(out) else True:
        return out
    train_count = max(1, int(math.floor(len(unique_dates) * config.walkforward_min_train_fraction)))
    test_dates = unique_dates[train_count:]
    if not len(test_dates):
        out['oos_eligible'] = False
        return out
    folds = np.array_split(test_dates, min(config.walkforward_folds, len(test_dates)))
    for number, dates in enumerate(folds, start=1):
        mask = out['signal_date'].isin(dates)
        out.loc[mask, 'is_oos'] = True
        out.loc[mask, 'oos_fold'] = number
    return out

def attach_backtest_stats(signals: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    columns = ['signal_events_oos', 'all_signal_events', 'historical_events', 'filled_events', 'entry_fill_rate_5d', 'entry_fill_ci_low', 'entry_fill_ci_high', 'historical_hit_rate', 'bayes_probability', 'tp1_ci_low', 'tp1_ci_high', 'expectancy_r', 'profit_factor', 'max_losing_streak', 'median_fill_bars', 'median_time_to_tp1_bars', 'tp1_time_p25', 'tp1_time_p75', 'sample_reliability', 'validation_scope']
    if signals.empty:
        return signals.copy()
    if stats is None or stats.empty:
        result = signals.copy()
        for column in columns:
            result[column] = np.nan
        result['validation_scope'] = 'MISSING'
        return result
    return signals.merge(stats, on='setup', how='left')

def aggregate_backtest(events: pd.DataFrame, config: ScanConfig) -> pd.DataFrame:
    columns = ['setup', 'signal_events_oos', 'all_signal_events', 'historical_events', 'filled_events', 'entry_fill_rate_5d', 'entry_fill_ci_low', 'entry_fill_ci_high', 'historical_hit_rate', 'bayes_probability', 'tp1_ci_low', 'tp1_ci_high', 'expectancy_r', 'profit_factor', 'max_losing_streak', 'median_fill_bars', 'median_time_to_tp1_bars', 'tp1_time_p25', 'tp1_time_p75', 'sample_reliability', 'validation_scope']
    if events.empty:
        return pd.DataFrame(columns=columns)
    all_counts = events.groupby('setup').size().to_dict()
    has_oos_columns = 'is_oos' in events.columns and 'oos_eligible' in events.columns
    if has_oos_columns:
        eligible = events['oos_eligible'].fillna(False).any()
        has_oos = events['is_oos'].fillna(False).any()
        if not (eligible and has_oos):
            return pd.DataFrame([{'setup': setup, 'signal_events_oos': 0, 'all_signal_events': int(count), 'historical_events': 0, 'filled_events': 0, 'sample_reliability': 'NONE', 'validation_scope': 'INSUFFICIENT_OOS'} for setup, count in all_counts.items()], columns=columns)
        sample = events[events['is_oos'].fillna(False)].copy()
        scope = 'CHRONOLOGICAL_OOS_HOLDOUT'
    else:
        sample = events.copy()
        scope = 'ALL_EVENTS_RESEARCH_ONLY'
    rows: list[dict[str, object]] = []
    for setup, group in sample.groupby('setup', sort=False):
        filled = group[group['filled'].fillna(False)] if 'filled' in group else group
        total_signals = len(group)
        count = len(filled)
        r = pd.to_numeric(filled.get('r_multiple'), errors='coerce').dropna()
        if 'tp1_hit' in filled:
            win_mask = filled['tp1_hit'].fillna(False).astype(bool)
        elif 'result' in filled:
            win_mask = filled['result'].eq('WIN_TP1')
        else:
            win_mask = r > 0
        wins = int(win_mask.sum())
        fill_low, fill_high = _wilson_interval(count, total_signals)
        hit_low, hit_high = _wilson_interval(wins, count)
        bayes = (wins + config.beta_prior_wins) / (count + config.beta_prior_wins + config.beta_prior_losses)
        gross_win = r[r > 0].sum()
        gross_loss = -r[r <= 0].sum()
        pf = gross_win / gross_loss if gross_loss > 0 else np.nan
        fill_wait = pd.to_numeric(filled.get('fill_wait_bars'), errors='coerce').dropna() if 'fill_wait_bars' in filled else pd.Series(dtype=float)
        tp_time = pd.to_numeric(filled.loc[win_mask, 'time_to_tp1_bars'], errors='coerce').dropna() if 'time_to_tp1_bars' in filled else pd.Series(dtype=float)
        reliability = 'HIGH' if count >= 50 else 'MEDIUM' if count >= 30 else 'LOW'
        rows.append({'setup': setup, 'signal_events_oos': total_signals if scope == 'CHRONOLOGICAL_OOS_HOLDOUT' else 0, 'all_signal_events': int(all_counts.get(setup, total_signals)), 'historical_events': count, 'filled_events': count, 'entry_fill_rate_5d': round(100 * count / total_signals, 1) if total_signals else np.nan, 'entry_fill_ci_low': round(100 * fill_low, 1), 'entry_fill_ci_high': round(100 * fill_high, 1), 'historical_hit_rate': round(100 * wins / count, 1) if count else np.nan, 'bayes_probability': round(100 * bayes, 1), 'tp1_ci_low': round(100 * hit_low, 1), 'tp1_ci_high': round(100 * hit_high, 1), 'expectancy_r': round(float(r.mean()), 3) if len(r) else np.nan, 'profit_factor': round(float(pf), 2) if np.isfinite(pf) else np.nan, 'max_losing_streak': _max_losing_streak(r.tolist()), 'median_fill_bars': round(float(fill_wait.median()), 1) if len(fill_wait) else np.nan, 'median_time_to_tp1_bars': round(float(tp_time.median()), 1) if len(tp_time) else np.nan, 'tp1_time_p25': round(float(tp_time.quantile(0.25)), 1) if len(tp_time) else np.nan, 'tp1_time_p75': round(float(tp_time.quantile(0.75)), 1) if len(tp_time) else np.nan, 'sample_reliability': reliability, 'validation_scope': scope})
    return pd.DataFrame(rows, columns=columns)
DETECTORS = {'PULLBACK_CONTINUATION': detect_pullback_continuation, 'BREAKOUT_RETEST': detect_breakout_retest, 'REVERSAL_ACCUMULATION': detect_reversal_accumulation, 'UNICORN_SNIPER_ICT': detect_unicorn_sniper}
DETECTOR_BY_SETUP = {name: detector for name, detector in DETECTORS.items()}

def _silent_accumulation_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    if frame is None or len(frame) < 25:
        return (0.0, np.nan)
    row = frame.iloc[-1]
    recent = frame.iloc[-20:].copy()
    value = pd.to_numeric(recent.get('VALUE'), errors='coerce').fillna(0.0)
    direction = pd.to_numeric(recent['Close'], errors='coerce').diff()
    up_value = float(value[direction > 0].sum())
    down_value = float(value[direction < 0].sum())
    up_down = up_value / down_value if down_value > 0 else 3.0 if up_value > 0 else np.nan
    cmf_value = _finite(row.get('CMF20'), -1)
    obv_slope = _finite(row.get('OBV_SLOPE10'), -1)
    close = _finite(row.get('Close'), 0)
    vwap = _finite(row.get('VWAP20'), float('inf'))
    relative = _finite(row.get('REL_STRENGTH60'), -1)
    vol_ratio = _finite(row.get('VOL_RATIO'), 0)
    score = 0.0
    score += 25 if cmf_value >= 0.05 else 15 if cmf_value > 0 else 0
    score += 20 if obv_slope > 0 else 0
    score += 15 if close >= vwap else 0
    score += 20 if np.isfinite(up_down) and up_down >= 1.15 else 10 if np.isfinite(up_down) and up_down >= 1.0 else 0
    score += 15 if relative > 0 else 0
    score += 5 if 0.7 <= vol_ratio <= 2.5 else 0
    return (min(100.0, score), up_down)
def _historical_gate_inputs(df: pd.DataFrame, config: ScanConfig):
    blockers, metrics = _historical_gate_inputs_v300(df, config)
    smart_score, up_down = _silent_accumulation_metrics(df)
    metrics['silent_accumulation_score'] = smart_score
    metrics['up_down_value_ratio20'] = up_down
    if config.real_money_mode and smart_score < 60:
        blockers.append('Silent-accumulation proxy di bawah gate')
    return (blockers, metrics)

def classify_provider_error(exc: BaseException) -> str:
    """Return a stable error category without hiding the original exception text."""
    name = type(exc).__name__.upper()
    text = str(exc).lower()
    if any(token in text for token in ('invalid crumb', 'unauthorized', 'http 401', 'status code 401', 'unable to access this feature')):
        return 'PROVIDER_AUTH'
    if '429' in text or 'rate limit' in text or 'too many requests' in text:
        return 'PROVIDER_RATE_LIMIT'
    if 'timeout' in name or 'timed out' in text or 'read timed out' in text:
        return 'PROVIDER_TIMEOUT'
    if any(token in name for token in ('CONNECTION', 'DNS', 'GAIERROR')) or any(token in text for token in ('name resolution', 'connection refused', 'connection aborted')):
        return 'PROVIDER_CONNECTION_ERROR'
    if any(token in name for token in ('JSON', 'DECODE', 'PARSER')) or 'unexpected schema' in text:
        return 'INVALID_RESPONSE_SCHEMA'
    if '404' in text or 'no data' in text or 'not found' in text:
        return 'NO_SYMBOL_DATA'
    if any(token in name for token in ('KEYERROR', 'ATTRIBUTEERROR', 'TYPEERROR', 'ASSERTIONERROR')):
        return 'PROGRAMMING_ERROR'
    return 'PROVIDER_ERROR'


_YAHOO_BLOCKING_ERROR_CODES = frozenset({
    'PROVIDER_AUTH', 'PROVIDER_RATE_LIMIT', 'PROVIDER_CONNECTION_ERROR',
})


def _is_yahoo_blocking_error(code: object) -> bool:
    return _safe_text(code).upper() in _YAHOO_BLOCKING_ERROR_CODES


def _fundamental_failure_row(ticker: str, code: str, message: object) -> dict[str, Any]:
    return {
        'ticker': ticker,
        'fundamental_score': np.nan,
        'fundamental_coverage': 0.0,
        'fundamental_reliability': 'NONE',
        'fundamental_red_flags': '',
        'latest_statement_date': pd.NaT,
        'statement_age_days': np.nan,
        'fundamental_error_code': _safe_text(code).upper() or 'PROVIDER_ERROR',
        'fundamental_error': _safe_text(message)[:180],
        'fundamental_provider': 'Yahoo Finance via yfinance',
        'fundamental_fetched_at': pd.Timestamp.now(tz='Asia/Jakarta').isoformat(),
    }


def _yahoo_provider_health_path() -> Path:
    return _cache_root() / 'yahoo_provider_health.json'


def _yahoo_provider_health_status() -> dict[str, Any]:
    path = _yahoo_provider_health_path()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        blocked_until = pd.to_datetime(payload.get('blocked_until'), errors='coerce', utc=True)
        now = pd.Timestamp.now(tz='UTC')
        if pd.isna(blocked_until) or blocked_until <= now:
            path.unlink(missing_ok=True)
            return {}
        payload['blocked_until'] = blocked_until.isoformat()
        return payload
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return {}


def _yahoo_cooldown_message(health: Mapping[str, Any]) -> str:
    code = _safe_text(health.get('error_code')).upper() or 'PROVIDER_UNAVAILABLE'
    until = pd.to_datetime(health.get('blocked_until'), errors='coerce', utc=True)
    until_text = until.tz_convert('Asia/Jakarta').strftime('%Y-%m-%d %H:%M WIB') if pd.notna(until) else 'sementara'
    return f'Yahoo circuit breaker aktif ({code}) sampai {until_text}'


def _mark_yahoo_provider_unavailable(code: object, reason: object) -> None:
    normalized = _safe_text(code).upper() or 'PROVIDER_ERROR'
    if not _is_yahoo_blocking_error(normalized):
        return
    cooldown_minutes = 45 if normalized in {'PROVIDER_AUTH', 'PROVIDER_RATE_LIMIT'} else 15
    now = pd.Timestamp.now(tz='UTC')
    payload = {
        'error_code': normalized,
        'reason': _safe_text(reason)[:240],
        'detected_at': now.isoformat(),
        'blocked_until': (now + pd.Timedelta(minutes=cooldown_minutes)).isoformat(),
    }
    path = _yahoo_provider_health_path()
    tmp = path.with_suffix('.tmp')
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _clear_yahoo_provider_unavailable() -> None:
    try:
        _yahoo_provider_health_path().unlink(missing_ok=True)
    except Exception:
        pass


def fetch_one_fundamental(ticker: str) -> dict[str, Any]:
    import yfinance as yf
    now = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None)
    try:
        obj = yf.Ticker(ticker)
        info = obj.get_info() or {}
        row = score_fundamentals(info)
        latest_statement = pd.NaT
        raw_statement = info.get('mostRecentQuarter') or info.get('lastFiscalYearEnd')
        if raw_statement is not None:
            try:
                if isinstance(raw_statement, (int, float, np.integer, np.floating)):
                    latest_statement = pd.to_datetime(raw_statement, unit='s', utc=True).tz_convert('Asia/Jakarta').tz_localize(None)
                else:
                    latest_statement = _as_jakarta_naive_timestamp(raw_statement)
            except Exception:
                latest_statement = pd.NaT
        statement_error = ''
        statement_age = int((now.normalize() - latest_statement.normalize()).days) if pd.notna(latest_statement) else np.nan
        row.update({'ticker': ticker, 'latest_statement_date': latest_statement, 'statement_age_days': statement_age, 'fundamental_error': statement_error, 'fundamental_provider': 'Yahoo Finance via yfinance', 'fundamental_fetched_at': pd.Timestamp.now(tz='Asia/Jakarta').isoformat()})
        return row
    except Exception as exc:
        code = classify_provider_error(exc)
        return _fundamental_failure_row(
            ticker, code, f'{type(exc).__name__}: {str(exc)[:140]}',
        )


# Canonical statement schema used by every historical-fundamental provider.
# Values are expected in the reporting currency after ``unit_multiplier`` has
# been applied. Duration facts (revenue, profit, cash flow) must describe one
# stand-alone quarter; ``parse_fundamental_history_csv`` converts YTD rows.
FUNDAMENTAL_HISTORY_COLUMNS: tuple[str, ...] = (
    'ticker', 'period_end', 'period_type', 'statement_basis', 'source_family',
    'source_name', 'source_url', 'currency', 'revenue', 'gross_profit',
    'operating_income', 'ebit', 'ebitda', 'net_income',
    'operating_cash_flow', 'capex', 'total_assets', 'total_liabilities',
    'equity', 'total_debt', 'cash', 'shares_outstanding', 'interest_expense',
    'car', 'npl_gross', 'ldr', 'source_verified', 'validation_flags',
)

_FUNDAMENTAL_VALUE_ALIASES: dict[str, tuple[str, ...]] = {
    'revenue': (
        'revenue', 'revenues', 'totalrevenue', 'sales', 'netsales',
        'salesandrevenue', 'netrevenue', 'totaloperatingrevenue',
        'pendapatan', 'pendapatanusaha',
    ),
    'gross_profit': ('grossprofit', 'labakotor'),
    'operating_income': (
        'operatingincome', 'operatingprofit', 'profitlossfromoperations',
        'incomefromoperations', 'labau saha', 'labausaha',
    ),
    'ebit': ('ebit', 'earningsbeforeinterestandtax'),
    'ebitda': ('ebitda', 'normalizedebitda'),
    'net_income': (
        'netincome', 'netincomecommonstockholders', 'profitloss',
        'profitlossattributabletoownersofparententity',
        'profitlossfromcontinuingoperations', 'lababersih',
    ),
    'operating_cash_flow': (
        'operatingcashflow', 'cashflowfromoperations',
        'cashflowsfromusedinoperatingactivities', 'netcashflowsfromusedinoperatingactivities',
        'ocf', 'aruskasoperasi',
    ),
    'capex': (
        'capex', 'capitalexpenditure', 'capitalexpenditures',
        'paymentstoacquirepropertyplantandequipment',
        'acquisitionofpropertyplantandequipment', 'purchaseoffixedassets',
        'belanjamodal',
    ),
    'total_assets': ('totalassets', 'assets', 'totalaset', 'aset'),
    'total_liabilities': ('totalliabilities', 'totalliabilitiesnetminorityinterest', 'liabilities', 'totalliabilitas', 'liabilitas'),
    'equity': ('stockholdersequity', 'shareholdersequity', 'totalequitygrossminorityinterest', 'totalequity', 'equity', 'ekuitas'),
    'total_debt': ('totaldebt', 'totalborrowings', 'borrowings', 'interestbearingdebt', 'utangberbunga'),
    'cash': (
        'cashcashequivalentsandshortterminvestments', 'cashandcashequivalents',
        'cashandbank', 'cash', 'kasdansetarakas',
    ),
    'shares_outstanding': ('sharesoutstanding', 'ordinarysharesnumber', 'dilutedaverageshares', 'sahamberedar'),
    'interest_expense': ('interestexpense', 'interestexpensenonoperating', 'bebanbunga'),
    'car': ('car', 'capitaladequacyratio', 'rasio kecukupan modal', 'rasiokecukupanmodal'),
    'npl_gross': ('nplgross', 'grossnpl', 'nonperformingloangross'),
    'ldr': ('ldr', 'loantodepositratio'),
}


def _fundamental_empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=list(FUNDAMENTAL_HISTORY_COLUMNS))


def _fundamental_column_key(value: object) -> str:
    import re
    return re.sub(r'[^a-z0-9]+', '', str(value or '').strip().lower())


def _fundamental_number(value: object) -> float:
    """Parse common Indonesian/English financial-number representations."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else np.nan
    text = str(value).strip()
    if not text or text.lower() in {'-', '—', 'na', 'n/a', 'nan', 'none'}:
        return np.nan
    negative = text.startswith('(') and text.endswith(')')
    text = text.strip('()').replace('Rp', '').replace('IDR', '').replace('\u00a0', '').replace(' ', '')
    if ',' in text and '.' in text:
        # 1.234.567,89 (ID) versus 1,234,567.89 (EN).
        if text.rfind(',') > text.rfind('.'):
            text = text.replace('.', '').replace(',', '.')
        else:
            text = text.replace(',', '')
    elif ',' in text:
        pieces = text.split(',')
        text = ''.join(pieces) if all(len(piece) == 3 for piece in pieces[1:]) else text.replace(',', '.')
    elif text.count('.') > 1:
        text = text.replace('.', '')
    try:
        number = float(text)
        return -number if negative else number
    except ValueError:
        return np.nan


def _official_idx_reference(value: object) -> bool:
    """Accept only HTTPS references whose host is IDX itself."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(str(value or '').strip())
        host = (parsed.hostname or '').lower()
        return parsed.scheme == 'https' and (host == 'idx.co.id' or host.endswith('.idx.co.id'))
    except Exception:
        return False


def _normalize_period_type(value: object, period_end: object) -> str:
    text = str(value or '').strip().upper().replace(' ', '')
    aliases = {
        'QUARTERLY': 'Q', 'QUARTER': 'Q', 'TRIWULAN': 'Q', 'QTR': 'Q',
        'ANNUAL': 'FY', 'YEARLY': 'FY', 'TAHUNAN': 'FY', 'YEAR': 'FY',
        'Q4': 'Q4', 'Q3': 'Q3', 'Q2': 'Q2', 'Q1': 'Q1', 'FY': 'FY', 'Q': 'Q',
    }
    normalized = aliases.get(text, text)
    if normalized in {'Q1', 'Q2', 'Q3', 'Q4', 'Q', 'FY'}:
        return normalized
    parsed = pd.to_datetime(period_end, errors='coerce')
    if pd.notna(parsed):
        return f'Q{((int(parsed.month) - 1) // 3) + 1}'
    return 'Q'


def _convert_ytd_to_standalone(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert cumulative interim duration facts into stand-alone quarters."""
    if frame.empty:
        return frame
    out = frame.copy()
    duration_columns = [
        'revenue', 'gross_profit', 'operating_income', 'ebit', 'ebitda',
        'net_income', 'operating_cash_flow', 'capex', 'interest_expense',
    ]
    basis = out['statement_basis'].astype(str).str.upper()
    cumulative_mask = basis.isin({'YTD', 'YTD_CUMULATIVE', 'CUMULATIVE'})
    if not cumulative_mask.any():
        return out
    out['_fiscal_year'] = pd.to_datetime(out['period_end'], errors='coerce').dt.year
    group_columns = ['ticker', 'source_family', '_fiscal_year']
    for _, index in out[cumulative_mask].groupby(group_columns, dropna=False).groups.items():
        ordered = out.loc[index].sort_values('period_end')
        prior: dict[str, float] = {}
        for row_index, row in ordered.iterrows():
            period = str(row.get('period_type') or '').upper()
            month = pd.Timestamp(row.get('period_end')).month if pd.notna(row.get('period_end')) else 0
            if period == 'FY':
                out.at[row_index, 'statement_basis'] = 'ANNUAL'
                continue
            first_quarter = period == 'Q1' or (period == 'Q' and month <= 3)
            missing_predecessor = bool(not prior and not first_quarter)
            for column in duration_columns:
                current = _num(row.get(column))
                if not np.isfinite(current):
                    continue
                if first_quarter:
                    standalone = current
                else:
                    previous = prior.get(column, np.nan)
                    standalone = current - previous if np.isfinite(previous) else np.nan
                out.at[row_index, column] = standalone
                prior[column] = current
            out.at[row_index, 'statement_basis'] = 'STANDALONE_QUARTER_FROM_YTD'
            if missing_predecessor:
                old_flags = str(out.at[row_index, 'validation_flags'] or '').strip()
                out.at[row_index, 'validation_flags'] = ' • '.join(
                    part for part in (old_flags, 'YTD_PREDECESSOR_MISSING') if part
                )
    return out.drop(columns=['_fiscal_year'])


def _statement_identity_flags(row: pd.Series) -> str:
    assets = _num(row.get('total_assets'))
    liabilities = _num(row.get('total_liabilities'))
    equity = _num(row.get('equity'))
    flags: list[str] = []
    if np.isfinite(assets) and np.isfinite(liabilities) and np.isfinite(equity) and abs(assets) > 0:
        gap = abs(assets - liabilities - equity) / abs(assets)
        if gap > 0.05:
            flags.append(f'ACCOUNTING_IDENTITY_GAP_{100 * gap:.1f}%')
    if np.isfinite(assets) and assets <= 0:
        flags.append('NONPOSITIVE_ASSETS')
    if np.isfinite(row.get('shares_outstanding', np.nan)) and float(row['shares_outstanding']) <= 0:
        flags.append('NONPOSITIVE_SHARES')
    return ' • '.join(flags)


def normalize_fundamental_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize, validate, and de-duplicate canonical statement rows."""
    if frame is None or frame.empty:
        return _fundamental_empty_history()
    out = frame.copy()
    for column in FUNDAMENTAL_HISTORY_COLUMNS:
        if column not in out:
            if column == 'source_verified':
                out[column] = False
            else:
                out[column] = '' if column in {'period_type', 'statement_basis', 'source_family', 'source_name', 'source_url', 'currency', 'validation_flags'} else np.nan
    out['ticker'] = out['ticker'].map(normalize_idx_ticker)
    out['period_end'] = pd.to_datetime(out['period_end'], errors='coerce').dt.tz_localize(None)
    out['period_type'] = [
        _normalize_period_type(period, date)
        for period, date in zip(out['period_type'], out['period_end'])
    ]
    out['statement_basis'] = out['statement_basis'].fillna('').astype(str).str.upper().replace('', 'STANDALONE_QUARTER')
    out['source_family'] = out['source_family'].fillna('').astype(str).str.upper().replace('', 'USER_UPLOAD')
    out['source_name'] = out['source_name'].fillna('').astype(str)
    out['source_url'] = out['source_url'].fillna('').astype(str)
    out['currency'] = out['currency'].fillna('IDR').astype(str).str.upper().replace('', 'IDR')
    for column in _FUNDAMENTAL_VALUE_ALIASES:
        out[column] = out[column].map(_fundamental_number)
    out['source_verified'] = out['source_verified'].map(_truthy).astype(bool)
    out = out.dropna(subset=['ticker', 'period_end'])
    out = _convert_ytd_to_standalone(out)
    generated_flags = out.apply(_statement_identity_flags, axis=1)
    prior_flags = out['validation_flags'].fillna('').astype(str)
    out['validation_flags'] = [
        ' • '.join(dict.fromkeys(piece for piece in (old, new) if piece))
        for old, new in zip(prior_flags, generated_flags)
    ]
    # Prefer an explicitly verified row when the same provider supplies a
    # duplicate period, while preserving distinct providers for consensus.
    out = out.sort_values(['ticker', 'source_family', 'period_end', 'source_verified'])
    out = out.drop_duplicates(['ticker', 'source_family', 'period_end', 'period_type'], keep='last')
    return out[list(FUNDAMENTAL_HISTORY_COLUMNS)].reset_index(drop=True)


def parse_fundamental_history_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    """Parse normalized IDX/XBRL-derived history without trusting its label.

    A row is marked as official only when ``source_url`` is an HTTPS URL on an
    IDX-owned host. This does not claim that the numeric transcription itself
    was independently audited; accounting and cross-provider checks remain
    mandatory.
    """
    raw = _read_csv(source)
    if raw.empty:
        return _fundamental_empty_history()
    lookup = {_fundamental_column_key(column): column for column in raw.columns}

    def pick(*aliases: str) -> object | None:
        return next((lookup[_fundamental_column_key(alias)] for alias in aliases if _fundamental_column_key(alias) in lookup), None)

    ticker_column = pick('ticker', 'symbol', 'kode', 'kode_emiten', 'stock_code')
    date_column = pick('period_end', 'report_date', 'fiscal_date', 'tanggal_laporan', 'date')
    if ticker_column is None or date_column is None:
        raise ValueError('Fundamental history wajib memiliki kolom ticker dan period_end/report_date')
    out = pd.DataFrame({'ticker': raw[ticker_column], 'period_end': raw[date_column]})
    period_column = pick('period_type', 'frequency', 'quarter', 'periode')
    basis_column = pick('statement_basis', 'basis', 'reporting_basis')
    source_column = pick('source_name', 'provider', 'source', 'sumber')
    source_url_column = pick('source_url', 'document_url', 'idx_url', 'url')
    currency_column = pick('currency', 'mata_uang')
    multiplier_column = pick('unit_multiplier', 'multiplier', 'satuan_pengali')
    shares_multiplier_column = pick('shares_multiplier', 'share_unit_multiplier', 'pengali_saham')
    out['period_type'] = raw[period_column] if period_column is not None else ''
    out['statement_basis'] = raw[basis_column] if basis_column is not None else 'STANDALONE_QUARTER'
    out['source_name'] = raw[source_column] if source_column is not None else 'User statement upload'
    out['source_url'] = raw[source_url_column] if source_url_column is not None else ''
    out['currency'] = raw[currency_column] if currency_column is not None else 'IDR'
    official_reference = out['source_url'].map(_official_idx_reference)
    out['source_family'] = np.where(official_reference, 'IDX_OFFICIAL_REFERENCE', 'USER_UPLOAD')
    # A referenced URL is evidence provenance, not cryptographic verification.
    out['source_verified'] = False
    multiplier = raw[multiplier_column].map(_fundamental_number) if multiplier_column is not None else pd.Series(1.0, index=raw.index)
    multiplier = multiplier.where(multiplier.gt(0), 1.0).fillna(1.0)
    shares_multiplier = raw[shares_multiplier_column].map(_fundamental_number) if shares_multiplier_column is not None else multiplier.copy()
    shares_multiplier = shares_multiplier.where(shares_multiplier.gt(0), 1.0).fillna(1.0)
    for canonical, aliases in _FUNDAMENTAL_VALUE_ALIASES.items():
        column = pick(canonical, *aliases)
        values = raw[column].map(_fundamental_number) if column is not None else pd.Series(np.nan, index=raw.index)
        # Prudential ratios are already ratios/percentages and must not be scaled.
        if canonical in {'car', 'npl_gross', 'ldr'}:
            out[canonical] = values
        elif canonical == 'shares_outstanding':
            out[canonical] = values * shares_multiplier
        else:
            out[canonical] = values * multiplier
    out['validation_flags'] = ''
    normalized = normalize_fundamental_history(out)
    if normalized.empty:
        raise ValueError('Tidak ada baris fundamental historis yang valid setelah normalisasi')
    return normalized


def _statement_value(statement: pd.DataFrame, aliases: Iterable[str], period: object) -> float:
    if statement is None or statement.empty or period not in statement.columns:
        return np.nan
    index_lookup = {_fundamental_column_key(index): index for index in statement.index}
    for alias in aliases:
        actual = index_lookup.get(_fundamental_column_key(alias))
        if actual is not None:
            return _fundamental_number(statement.at[actual, period])
    return np.nan


def _yahoo_statement_rows(ticker: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import yfinance as yf
    rows: list[dict[str, Any]] = []
    try:
        obj = yf.Ticker(ticker)
        errors: list[str] = []
        error_codes: list[str] = []

        def safe_statement(attribute: str) -> pd.DataFrame:
            try:
                value = getattr(obj, attribute)
                return value if isinstance(value, pd.DataFrame) else pd.DataFrame()
            except Exception as exc:
                error_codes.append(classify_provider_error(exc))
                errors.append(f'{attribute}: {type(exc).__name__}: {str(exc)[:80]}')
                return pd.DataFrame()

        for period_type, attributes in (
            ('Q', ('quarterly_income_stmt', 'quarterly_balance_sheet', 'quarterly_cashflow')),
            ('FY', ('income_stmt', 'balance_sheet', 'cashflow')),
        ):
            statements = tuple(safe_statement(attribute) for attribute in attributes)
            income, balance, cashflow = statements
            periods: set[object] = set()
            for statement in statements:
                if statement is not None and not statement.empty:
                    periods.update(statement.columns)
            for period in sorted(periods):
                row: dict[str, Any] = {
                    'ticker': ticker, 'period_end': period, 'period_type': period_type,
                    'statement_basis': 'ANNUAL' if period_type == 'FY' else 'STANDALONE_QUARTER',
                    'source_family': 'YAHOO', 'source_name': 'Yahoo Finance statements via yfinance',
                    'source_url': '', 'currency': 'IDR', 'source_verified': False,
                    'validation_flags': '',
                }
                for canonical, aliases in _FUNDAMENTAL_VALUE_ALIASES.items():
                    source_statement = (
                        cashflow if canonical in {'operating_cash_flow', 'capex'}
                        else balance if canonical in {'total_assets', 'total_liabilities', 'equity', 'total_debt', 'cash', 'shares_outstanding'}
                        else income
                    )
                    value = _statement_value(source_statement, aliases, period)
                    if canonical == 'shares_outstanding' and not np.isfinite(value):
                        value = _statement_value(income, aliases, period)
                    row[canonical] = value
                rows.append(row)
        status = 'PARTIAL' if rows and errors else 'OK' if rows else 'NO_DATA'
        blocking_code = next((code for code in error_codes if _is_yahoo_blocking_error(code)), '')
        return rows, {
            'ticker': ticker, 'provider': 'YAHOO', 'status': status,
            'rows': len(rows), 'error': ' | '.join(errors),
            'error_code': blocking_code or (error_codes[0] if error_codes else ''),
        }
    except Exception as exc:
        code = classify_provider_error(exc)
        return [], {
            'ticker': ticker, 'provider': 'YAHOO', 'status': 'FAILED',
            'rows': 0, 'error': f'{type(exc).__name__}: {str(exc)[:140]}',
            'error_code': code,
        }


def _cached_fundamental_history(cache_name: str, tickers: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load durable statement history and its requested-ticker slice.

    Financial statements are immutable historical facts.  Keeping normalized
    rows on disk is safer than depending only on Streamlit's in-memory cache,
    which disappears after a redeploy or container restart.  Staleness remains
    visible through ``statement_age_days`` and never becomes synthetic data.
    """
    all_history = normalize_fundamental_history(_load_cache(cache_name))
    names = list(dict.fromkeys(tickers))
    selected = (
        all_history.loc[all_history['ticker'].isin(names)].copy()
        if not all_history.empty else _fundamental_empty_history()
    )
    return all_history, selected


def _persist_fundamental_history(cache_name: str, cached_all: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    current = normalize_fundamental_history(current)
    combined = combine_fundamental_history(cached_all, current)
    if not current.empty:
        _write_cache(cache_name, combined)
    return combined


def _history_report_with_cache_fallback(
    reports: pd.DataFrame, names: Iterable[str], cached: pd.DataFrame, provider: str,
) -> pd.DataFrame:
    """Replace provider failures with explicit cache-fallback audit states."""
    columns = ['ticker', 'provider', 'status', 'rows', 'error', 'error_code']
    report = reports.copy() if reports is not None else pd.DataFrame(columns=columns)
    if report.empty:
        report = pd.DataFrame([
            {'ticker': ticker, 'provider': provider, 'status': 'NO_DATA', 'rows': 0, 'error': '', 'error_code': ''}
            for ticker in names
        ])
    for ticker in names:
        count = int(len(cached.loc[cached['ticker'].eq(ticker)])) if not cached.empty else 0
        mask = report.get('ticker', pd.Series(dtype=str)).astype(str).eq(str(ticker))
        if not mask.any():
            report = pd.concat([report, pd.DataFrame([{
                'ticker': ticker, 'provider': provider,
                'status': 'CACHE_FALLBACK' if count else 'NO_DATA',
                'rows': count, 'error': 'Persistent statement cache' if count else '',
                'error_code': '',
            }])], ignore_index=True, sort=False)
            continue
        status = report.loc[mask, 'status'].astype(str).str.upper()
        live_ok = status.isin({'OK', 'PARTIAL'}).any() and pd.to_numeric(report.loc[mask, 'rows'], errors='coerce').fillna(0).gt(0).any()
        if count and not live_ok:
            report.loc[mask, 'status'] = 'CACHE_FALLBACK'
            report.loc[mask, 'rows'] = count
            existing = report.loc[mask, 'error'].fillna('').astype(str)
            report.loc[mask, 'error'] = existing.map(lambda value: (value + ' | ' if value else '') + 'Persistent statement cache digunakan')
    return report


def select_yahoo_fundamental_tickers(
    tickers: Iterable[str],
    idx_history: pd.DataFrame,
    max_tickers: int=40,
    crosscheck_top_n: int=8,
    min_official_periods: int=4,
) -> tuple[str, ...]:
    """Choose Yahoo statement work after inspecting official IDX coverage.

    Ticker order is treated as ranking order. Issuers without enough verified
    official periods are fallbacks; a small leading cohort with sufficient IDX
    history is retained solely for provider-consensus checks.
    """
    names = list(dict.fromkeys(normalize_idx_ticker(value) for value in tickers))
    names = [value for value in names if value]
    history = normalize_fundamental_history(idx_history)
    official_counts: dict[str, int] = {}
    if not history.empty:
        official = history.loc[
            history['source_family'].eq('IDX_OFFICIAL_XBRL')
            & history['source_verified'].map(_truthy)
        ]
        if not official.empty:
            official_counts = official.groupby('ticker').size().astype(int).to_dict()
    threshold = max(1, int(min_official_periods))
    unresolved = [ticker for ticker in names if int(official_counts.get(ticker, 0)) < threshold]
    verified = [ticker for ticker in names if int(official_counts.get(ticker, 0)) >= threshold]
    selected = list(dict.fromkeys(verified[:max(0, int(crosscheck_top_n))] + unresolved))
    return tuple(selected[:max(0, int(max_tickers))])


def fetch_yahoo_fundamental_history(tickers: Iterable[str], max_workers: int=2, max_tickers: int=40) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch bounded Yahoo statements with circuit breaker and durable cache."""
    names = list(dict.fromkeys(tickers))[:max(0, int(max_tickers))]
    report_columns = ['ticker', 'provider', 'status', 'rows', 'error', 'error_code']
    if not names:
        return _fundamental_empty_history(), pd.DataFrame(columns=report_columns)
    cached_all, cached = _cached_fundamental_history('yahoo_fundamental_history', names)
    health = _yahoo_provider_health_status()
    if health:
        message = _yahoo_cooldown_message(health)
        reports = pd.DataFrame([
            {
                'ticker': ticker, 'provider': 'YAHOO', 'status': 'COOLDOWN',
                'rows': 0, 'error': message, 'error_code': 'PROVIDER_COOLDOWN',
            }
            for ticker in names
        ])
        reports = _history_report_with_cache_fallback(reports, names, cached, 'YAHOO')
        return cached, reports

    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    first_rows, first_report = _yahoo_statement_rows(names[0])
    rows.extend(first_rows)
    reports.append(first_report)
    first_code = _safe_text(first_report.get('error_code')).upper()
    if _is_yahoo_blocking_error(first_code):
        _mark_yahoo_provider_unavailable(first_code, first_report.get('error', ''))
        reports.extend({
            'ticker': ticker, 'provider': 'YAHOO', 'status': 'COOLDOWN',
            'rows': 0, 'error': f'Yahoo dihentikan sementara setelah {first_code}',
            'error_code': 'PROVIDER_COOLDOWN',
        } for ticker in names[1:])
        current = normalize_fundamental_history(pd.DataFrame(rows))
        merged_all = _persist_fundamental_history('yahoo_fundamental_history', cached_all, current)
        selected = merged_all.loc[merged_all['ticker'].isin(names)].copy() if not merged_all.empty else _fundamental_empty_history()
        report = _history_report_with_cache_fallback(pd.DataFrame(reports, columns=report_columns), names, cached, 'YAHOO')
        return selected, report
    if first_rows:
        _clear_yahoo_provider_unavailable()

    remaining = names[1:]
    workers = min(max(1, int(max_workers)), max(1, len(remaining)))
    chunk_size = max(2, workers * 2)
    for offset in range(0, len(remaining), chunk_size):
        chunk = remaining[offset:offset + chunk_size]
        with ThreadPoolExecutor(max_workers=min(workers, len(chunk))) as pool:
            futures = {pool.submit(_yahoo_statement_rows, ticker): ticker for ticker in chunk}
            chunk_results = [future.result() for future in as_completed(futures)]
        blocking_report = None
        for statement_rows, report in chunk_results:
            rows.extend(statement_rows)
            reports.append(report)
            if blocking_report is None and _is_yahoo_blocking_error(report.get('error_code')):
                blocking_report = report
        if blocking_report is not None:
            code = _safe_text(blocking_report.get('error_code')).upper()
            _mark_yahoo_provider_unavailable(code, blocking_report.get('error', ''))
            pending = remaining[offset + len(chunk):]
            reports.extend({
                'ticker': ticker, 'provider': 'YAHOO', 'status': 'COOLDOWN',
                'rows': 0, 'error': f'Yahoo dihentikan sementara setelah {code}',
                'error_code': 'PROVIDER_COOLDOWN',
            } for ticker in pending)
            break
    current = normalize_fundamental_history(pd.DataFrame(rows))
    merged_all = _persist_fundamental_history('yahoo_fundamental_history', cached_all, current)
    selected = merged_all.loc[merged_all['ticker'].isin(names)].copy() if not merged_all.empty else _fundamental_empty_history()
    report = _history_report_with_cache_fallback(pd.DataFrame(reports, columns=report_columns), names, cached, 'YAHOO')
    return selected, report


def _twelve_statement_records(payload: object, endpoint: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    candidates = (
        endpoint, endpoint.replace('_', ' '), 'data', 'values', 'statements',
        'income_statement' if endpoint == 'income_statement' else endpoint,
    )
    for key in candidates:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for nested in ('quarterly', 'annual', 'data', 'values'):
                nested_value = value.get(nested)
                if isinstance(nested_value, list):
                    return [item for item in nested_value if isinstance(item, dict)]
    return []


def _record_value(record: Mapping[str, Any], aliases: Iterable[str]) -> float:
    lookup = {_fundamental_column_key(key): value for key, value in record.items()}
    for alias in aliases:
        key = _fundamental_column_key(alias)
        if key in lookup:
            return _fundamental_number(lookup[key])
    return np.nan


def fetch_twelve_data_fundamental_history(tickers: Iterable[str], api_key: str, max_tickers: int=20, timeout: int=15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch a bounded Twelve Data statement shortlist.

    Fundamental endpoints require an eligible Twelve Data plan. A failed or
    unsupported XIDX response is reported and never replaced by synthetic data.
    """
    names = list(dict.fromkeys(tickers))[:max(0, int(max_tickers))]
    report_columns = ['ticker', 'provider', 'status', 'rows', 'error', 'error_code']
    if not names:
        return _fundamental_empty_history(), pd.DataFrame(columns=report_columns)
    cached_all, cached = _cached_fundamental_history('twelve_fundamental_history', names)
    if not _safe_text(api_key):
        reports = pd.DataFrame([
            {'ticker': ticker, 'provider': 'TWELVE_DATA', 'status': 'DISABLED', 'rows': 0,
             'error': 'API key tidak dikonfigurasi', 'error_code': 'PROVIDER_DISABLED'}
            for ticker in names
        ])
        return cached, _history_report_with_cache_fallback(reports, names, cached, 'TWELVE_DATA')
    import requests
    all_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    endpoint_mapping = {
        'income_statement': {'revenue', 'gross_profit', 'operating_income', 'ebit', 'ebitda', 'net_income', 'shares_outstanding', 'interest_expense'},
        'balance_sheet': {'total_assets', 'total_liabilities', 'equity', 'total_debt', 'cash', 'shares_outstanding'},
        'cash_flow': {'operating_cash_flow', 'capex'},
    }
    for ticker in names:
        symbol = ticker[:-3] if ticker.upper().endswith('.JK') else ticker
        by_period: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
        errors: list[str] = []
        for endpoint, canonical_fields in endpoint_mapping.items():
            try:
                response = requests.get(
                    f'https://api.twelvedata.com/{endpoint}',
                    params={'symbol': symbol, 'exchange': 'XIDX', 'period': 'quarterly', 'outputsize': 12, 'format': 'JSON', 'apikey': api_key},
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                records = _twelve_statement_records(payload, endpoint)
                if not records:
                    message = _safe_text(payload.get('message')) if isinstance(payload, dict) else 'empty response'
                    errors.append(f'{endpoint}: {message or "no rows"}')
                    continue
                for record in records:
                    period_raw = next((record.get(key) for key in ('fiscal_date', 'fiscalDate', 'date', 'period_end', 'report_date') if record.get(key) is not None), None)
                    period_end = pd.to_datetime(period_raw, errors='coerce')
                    if pd.isna(period_end):
                        continue
                    period_label = next((record.get(key) for key in ('period', 'frequency', 'quarter') if record.get(key) is not None), 'Q')
                    period_type = _normalize_period_type(period_label, period_end)
                    key = (period_end.tz_localize(None) if getattr(period_end, 'tzinfo', None) is not None else period_end, period_type)
                    row = by_period.setdefault(key, {
                        'ticker': ticker, 'period_end': key[0], 'period_type': period_type,
                        'statement_basis': 'STANDALONE_QUARTER', 'source_family': 'TWELVE_DATA',
                        'source_name': 'Twelve Data fundamentals API', 'source_url': 'https://twelvedata.com/fundamentals',
                        'currency': str(record.get('currency') or 'IDR').upper(), 'source_verified': False,
                        'validation_flags': '',
                    })
                    for canonical in canonical_fields:
                        value = _record_value(record, (canonical, *_FUNDAMENTAL_VALUE_ALIASES[canonical]))
                        if np.isfinite(value):
                            row[canonical] = value
            except Exception as exc:
                safe_error = str(exc).replace(str(api_key), '***')
                errors.append(f'{endpoint}: {type(exc).__name__}: {safe_error[:100]}')
        rows = list(by_period.values())
        all_rows.extend(rows)
        reports.append({
            'ticker': ticker, 'provider': 'TWELVE_DATA',
            'status': 'PARTIAL' if rows and errors else 'OK' if rows else 'FAILED', 'rows': len(rows),
            'error': ' | '.join(errors)[:420],
            'error_code': 'PROVIDER_ERROR' if errors and not rows else '',
        })
    current = normalize_fundamental_history(pd.DataFrame(all_rows))
    merged_all = _persist_fundamental_history('twelve_fundamental_history', cached_all, current)
    selected = merged_all.loc[merged_all['ticker'].isin(names)].copy() if not merged_all.empty else _fundamental_empty_history()
    report = _history_report_with_cache_fallback(pd.DataFrame(reports), names, cached, 'TWELVE_DATA')
    return selected, report


IDX_FINANCIAL_REPORT_ENDPOINTS: tuple[str, ...] = (
    'https://www.idx.co.id/primary/ListedCompany/GetFinancialReport',
    'https://www.idx.co.id/umbraco/Surface/ListedCompany/GetFinancialReport',
)
IDX_FINANCIAL_STATEMENTS_PAGE = (
    'https://www.idx.co.id/id/perusahaan-tercatat/laporan-keuangan-dan-tahunan'
)
IDX_FINANCIAL_REPORT_PERIODS: tuple[tuple[str, str], ...] = (
    ('tw1', 'Q1'), ('tw2', 'Q2'), ('tw3', 'Q3'),
    ('audit', 'FY'), ('tahunan', 'FY'),
)


def _idx_report_results(payload: object) -> tuple[list[dict[str, Any]], int]:
    """Normalize the two response shapes used by the official IDX web page."""
    if not isinstance(payload, Mapping):
        return ([], 0)
    raw = payload.get('Results', payload.get('results', []))
    rows = [item for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []
    count = _fundamental_number(payload.get('ResultCount', payload.get('resultCount', len(rows))))
    return ([dict(item) for item in rows], int(count) if np.isfinite(count) else len(rows))


def _idx_attachment_url(value: object) -> str:
    """Resolve an attachment while refusing redirects/hosts outside IDX."""
    from urllib.parse import urljoin, urlparse
    raw = _safe_text(value)
    if not raw:
        return ''
    url = urljoin('https://www.idx.co.id/', raw)
    parsed = urlparse(url)
    host = (parsed.hostname or '').lower()
    if parsed.scheme != 'https' or not (host == 'idx.co.id' or host.endswith('.idx.co.id')):
        return ''
    return url


def _idx_manifest_rows(
    tickers: Iterable[str],
    years: Iterable[int],
    timeout: int=12,
    page_size: int=1200,
    max_pages: int=3,
    request_get: Any | None=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover official filing attachments used by IDX's public web page.

    The endpoint is intentionally treated as an undocumented page dependency,
    not as a guaranteed public API. Calls are bounded and every failure is
    returned in the audit report instead of being converted into synthetic
    statement data.
    """
    if request_get is None:
        import requests
        getter = requests.get
    else:
        getter = request_get
    names = [normalize_idx_ticker(value) for value in tickers]
    names = [value for value in dict.fromkeys(names) if value]
    wanted = {value[:-3] if value.endswith('.JK') else value for value in names}
    if not wanted:
        return (pd.DataFrame(), pd.DataFrame())
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; IDXSuperScanner/5.5; research-client)',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'id-ID,id;q=0.9,en;q=0.7',
        'Referer': IDX_FINANCIAL_STATEMENTS_PAGE + '/',
    }
    manifests: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    endpoint_index = 0
    today = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None).normalize()
    for year in sorted({int(value) for value in years}, reverse=True):
        for period_code, period_type in IDX_FINANCIAL_REPORT_PERIODS:
            period_month = {'Q1': 3, 'Q2': 6, 'Q3': 9, 'FY': 12}[period_type]
            period_day = {3: 31, 6: 30, 9: 30, 12: 31}[period_month]
            filing_period_end = pd.Timestamp(year=year, month=period_month, day=period_day)
            if filing_period_end > today:
                continue
            found_codes: set[str] = set()
            last_error = ''
            endpoint_used = ''
            rows_seen = 0
            for candidate_index in range(endpoint_index, len(IDX_FINANCIAL_REPORT_ENDPOINTS)):
                endpoint = IDX_FINANCIAL_REPORT_ENDPOINTS[candidate_index]
                endpoint_rows: list[dict[str, Any]] = []
                try:
                    for page in range(max(1, int(max_pages))):
                        response = getter(
                            endpoint,
                            params={
                                'indexFrom': page, 'pageSize': max(10, int(page_size)),
                                'year': year, 'reportType': 'rdf', 'periode': period_code,
                                'kodeEmiten': '', 'EmitenType': 's',
                                'SortColumn': 'KodeEmiten', 'SortOrder': 'asc',
                            },
                            headers=headers, timeout=timeout, allow_redirects=True,
                        )
                        if int(getattr(response, 'status_code', 0)) != 200:
                            raise RuntimeError(f'HTTP {getattr(response, "status_code", "?")}')
                        final_url = _safe_text(getattr(response, 'url', endpoint)) or endpoint
                        if not _official_idx_reference(final_url):
                            raise RuntimeError('redirect keluar domain resmi IDX')
                        payload = response.json()
                        page_rows, total = _idx_report_results(payload)
                        endpoint_rows.extend(page_rows)
                        if not page_rows or len(endpoint_rows) >= total or len(page_rows) < max(10, int(page_size)):
                            break
                    endpoint_used = endpoint
                    endpoint_index = candidate_index
                    break
                except Exception as exc:
                    last_error = f'{type(exc).__name__}: {str(exc)[:140]}'
                    endpoint_rows = []
                    continue
            rows_seen = len(endpoint_rows)
            for result in endpoint_rows:
                raw_code = _safe_text(result.get('KodeEmiten', result.get('kodeEmiten'))).upper()
                if raw_code not in wanted:
                    continue
                found_codes.add(raw_code)
                raw_attachments = result.get('Attachments', result.get('attachments', []))
                attachments = raw_attachments if isinstance(raw_attachments, list) else []
                for attachment in attachments:
                    if not isinstance(attachment, Mapping):
                        continue
                    filename = _safe_text(attachment.get('File_Name', attachment.get('fileName')))
                    file_path = attachment.get('File_Path', attachment.get('filePath'))
                    url = _idx_attachment_url(file_path)
                    if not filename or not url:
                        continue
                    lower = filename.lower()
                    if not (
                        lower.endswith(('.zip', '.xbrl', '.xml', '.xhtml', '.html'))
                        and ('xbrl' in lower or 'instance' in lower or lower.endswith(('.xbrl', '.xml')))
                    ):
                        continue
                    rank = 0 if 'instance' in lower and lower.endswith('.zip') else 1 if 'inlinexbrl' in lower else 2
                    manifests.append({
                        'ticker': f'{raw_code}.JK', 'year': year,
                        'period_code': period_code, 'period_type': period_type,
                        'period_end': filing_period_end,
                        'filename': filename, 'attachment_url': url,
                        'attachment_rank': rank, 'report_modified': result.get('File_Modified'),
                        'manifest_endpoint': endpoint_used,
                    })
            audit.append({
                'ticker': 'ALL_REQUESTED', 'provider': 'IDX_OFFICIAL_XBRL_MANIFEST',
                'scope': f'{year}/{period_code}',
                'status': 'OK' if found_codes else 'NO_MATCH' if endpoint_used else 'FAILED',
                'rows': rows_seen, 'matched_tickers': len(found_codes),
                'error': last_error if not endpoint_used else '',
                'stability': 'UNDOCUMENTED_PUBLIC_PAGE_ENDPOINT',
            })
    manifest = pd.DataFrame(manifests)
    if not manifest.empty:
        manifest = (
            manifest.sort_values(['ticker', 'period_end', 'attachment_rank'])
            .drop_duplicates(['ticker', 'period_end', 'period_type', 'attachment_rank'], keep='last')
            .reset_index(drop=True)
        )
    return (manifest, pd.DataFrame(audit))


def _xbrl_local_name(value: object) -> str:
    text = str(value or '')
    if '}' in text:
        text = text.rsplit('}', 1)[-1]
    if ':' in text:
        text = text.rsplit(':', 1)[-1]
    return text


def _xbrl_attribute(element: Any, name: str) -> str:
    wanted = name.lower()
    for key, value in getattr(element, 'attrib', {}).items():
        if _xbrl_local_name(key).lower() == wanted:
            return _safe_text(value)
    return ''


def _xbrl_fact_number(element: Any, inline: bool=False) -> float:
    if _xbrl_attribute(element, 'nil').lower() in {'true', '1'}:
        return np.nan
    text = ''.join(element.itertext()) if hasattr(element, 'itertext') else ''
    value = _fundamental_number(text)
    if not np.isfinite(value):
        return np.nan
    if inline:
        scale = _fundamental_number(_xbrl_attribute(element, 'scale'))
        if np.isfinite(scale):
            value *= 10.0 ** int(scale)
        if _xbrl_attribute(element, 'sign').strip() == '-':
            value = -abs(value)
    return float(value)


def _xbrl_concept_score(concept: str, aliases: Iterable[str]) -> int:
    key = _fundamental_column_key(concept)
    best = 0
    for alias in aliases:
        candidate = _fundamental_column_key(alias)
        if not candidate:
            continue
        if key == candidate:
            best = max(best, 100)
        elif len(candidate) >= 10 and (key.endswith(candidate) or key.startswith(candidate)):
            best = max(best, 82)
    return best


def parse_idx_xbrl_attachment(
    payload: bytes,
    ticker: str,
    period_end: Any,
    period_type: str,
    source_url: str,
    filename: str='',
    max_uncompressed_bytes: int=60_000_000,
) -> pd.DataFrame:
    """Extract a conservative canonical row from an official XBRL/iXBRL file.

    Only numeric facts tied to a current-period XBRL context are considered.
    Interim duration facts prefer the year-to-date context so the existing
    accounting normalizer can safely derive stand-alone quarters.
    """
    import io
    import zipfile
    import xml.etree.ElementTree as ET

    if not isinstance(payload, (bytes, bytearray)) or not payload:
        return _fundamental_empty_history()
    if not _official_idx_reference(source_url):
        raise ValueError('Attachment XBRL harus berasal dari HTTPS domain IDX')
    documents: list[tuple[str, bytes]] = []
    if bytes(payload[:4]) == b'PK\x03\x04':
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            total = 0
            for member in archive.infolist():
                lower = member.filename.lower()
                if member.is_dir() or not lower.endswith(('.xbrl', '.xml', '.xhtml', '.html', '.htm')):
                    continue
                if member.file_size < 0 or member.file_size > max_uncompressed_bytes:
                    continue
                total += member.file_size
                if total > max_uncompressed_bytes:
                    raise ValueError('Ukuran XBRL terkompresi melampaui batas aman')
                documents.append((member.filename, archive.read(member)))
    else:
        documents.append((filename or 'filing.xbrl', bytes(payload)))
    if not documents:
        return _fundamental_empty_history()

    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    units: dict[tuple[str, str], str] = {}
    facts: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for member_name, content in documents:
        try:
            root = ET.fromstring(content)
        except Exception as exc:
            parse_errors.append(f'{member_name}: {type(exc).__name__}')
            continue
        for element in root.iter():
            local = _xbrl_local_name(element.tag).lower()
            if local == 'context':
                context_id = _xbrl_attribute(element, 'id')
                if not context_id:
                    continue
                start = end = instant = pd.NaT
                dimensions = 0
                for child in element.iter():
                    child_name = _xbrl_local_name(child.tag).lower()
                    if child_name == 'startdate':
                        start = pd.to_datetime(child.text, errors='coerce')
                    elif child_name == 'enddate':
                        end = pd.to_datetime(child.text, errors='coerce')
                    elif child_name == 'instant':
                        instant = pd.to_datetime(child.text, errors='coerce')
                    elif child_name in {'explicitmember', 'typedmember'}:
                        dimensions += 1
                contexts[(member_name, context_id)] = {
                    'id': context_id, 'start': start, 'end': end,
                    'instant': instant, 'dimensions': dimensions,
                }
            elif local == 'unit':
                unit_id = _xbrl_attribute(element, 'id')
                measures = [
                    _safe_text(child.text) for child in element.iter()
                    if _xbrl_local_name(child.tag).lower() == 'measure' and _safe_text(child.text)
                ]
                if unit_id:
                    units[(member_name, unit_id)] = ' '.join(measures)
        for element in root.iter():
            local = _xbrl_local_name(element.tag).lower()
            inline = local in {'nonfraction', 'fraction'}
            context_ref = _xbrl_attribute(element, 'contextref')
            if not context_ref:
                continue
            concept = _xbrl_attribute(element, 'name') if inline else _xbrl_local_name(element.tag)
            value = _xbrl_fact_number(element, inline=inline)
            if not concept or not np.isfinite(value):
                continue
            facts.append({
                'member': member_name, 'concept': concept, 'value': value,
                'context': contexts.get((member_name, context_ref), {}),
                'unit': units.get((member_name, _xbrl_attribute(element, 'unitref')), ''),
            })
    if not facts:
        if parse_errors:
            raise ValueError('XBRL tidak dapat diparse: ' + ' | '.join(parse_errors[:3]))
        return _fundamental_empty_history()

    target = pd.Timestamp(period_end).tz_localize(None) if getattr(pd.Timestamp(period_end), 'tzinfo', None) is not None else pd.Timestamp(period_end)
    duration_fields = {
        'revenue', 'gross_profit', 'operating_income', 'ebit', 'ebitda',
        'net_income', 'operating_cash_flow', 'capex', 'interest_expense',
    }
    selected: dict[str, float] = {}
    selected_units: list[str] = []
    for canonical, aliases in _FUNDAMENTAL_VALUE_ALIASES.items():
        candidates: list[tuple[float, float, str]] = []
        for fact in facts:
            concept_score = _xbrl_concept_score(fact['concept'], (canonical, *aliases))
            if concept_score <= 0:
                continue
            context = fact['context']
            context_end = context.get('instant') if pd.notna(context.get('instant', pd.NaT)) else context.get('end')
            if pd.isna(context_end):
                continue
            context_end = pd.Timestamp(context_end)
            date_gap = abs(int((target.normalize() - context_end.normalize()).days))
            if date_gap > 7:
                continue
            score = float(concept_score + 100 - 10 * date_gap)
            context_id = _safe_text(context.get('id')).lower()
            score += 24 if int(context.get('dimensions', 0)) == 0 else 0
            score += 8 if 'current' in context_id else 0
            score += 6 if 'consolidat' in context_id else 0
            score -= 25 if any(token in context_id for token in ('prior', 'previous', 'comparative')) else 0
            start = context.get('start', pd.NaT)
            if canonical in duration_fields:
                if pd.notna(start):
                    start_stamp = pd.Timestamp(start)
                    score += 12
                    if start_stamp.year == target.year and start_stamp.month == 1 and start_stamp.day <= 7:
                        score += 18
                else:
                    score -= 20
            elif pd.notna(context.get('instant', pd.NaT)):
                score += 12
            candidates.append((score, float(fact['value']), _safe_text(fact.get('unit'))))
        if candidates:
            _, value, unit = max(candidates, key=lambda item: item[0])
            selected[canonical] = value
            if unit:
                selected_units.append(unit)
    if sum(np.isfinite(value) for value in selected.values()) < 3:
        return _fundamental_empty_history()
    currency = 'IDR'
    unit_text = ' '.join(selected_units).upper()
    if 'USD' in unit_text and 'IDR' not in unit_text:
        currency = 'USD'
    row: dict[str, Any] = {
        'ticker': ticker, 'period_end': target, 'period_type': period_type,
        'statement_basis': 'ANNUAL' if str(period_type).upper() == 'FY' else 'YTD_CUMULATIVE',
        'source_family': 'IDX_OFFICIAL_XBRL',
        'source_name': 'IDX official public filing (XBRL/iXBRL)',
        'source_url': source_url, 'currency': currency,
        'source_verified': True, 'validation_flags': '',
        **selected,
    }
    return normalize_fundamental_history(pd.DataFrame([row]))


def fetch_idx_fundamental_history(
    tickers: Iterable[str],
    max_tickers: int=20,
    years_back: int=3,
    timeout: int=12,
    max_workers: int=4,
    max_attachment_bytes: int=25_000_000,
    request_get: Any | None=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Automatically fetch a bounded official IDX XBRL statement shortlist.

    No API key or user upload is required. Because discovery follows an
    undocumented endpoint used by the official IDX page, availability is never
    assumed: Yahoo/Twelve/manual histories remain independent fallbacks and the
    provider report exposes every unresolved document.
    """
    from urllib.parse import urlparse
    if request_get is None:
        import requests
        getter = requests.get
    else:
        getter = request_get
    names = [normalize_idx_ticker(value) for value in tickers]
    names = [value for value in dict.fromkeys(names) if value][:max(0, int(max_tickers))]
    columns = ['ticker', 'provider', 'status', 'rows', 'documents_found', 'documents_parsed', 'error', 'stability']
    if not names:
        return (_fundamental_empty_history(), pd.DataFrame(columns=columns))
    cached_all = normalize_fundamental_history(_load_cache('idx_fundamental_history'))
    cached = (
        cached_all.loc[cached_all['ticker'].isin(names)].copy()
        if not cached_all.empty else _fundamental_empty_history()
    )
    current_year = pd.Timestamp.now(tz='Asia/Jakarta').year
    years = range(current_year, current_year - max(1, int(years_back)), -1)
    manifest, manifest_report = _idx_manifest_rows(
        names, years, timeout=timeout, request_get=getter,
    )
    if manifest.empty:
        reports = [{
            'ticker': ticker, 'provider': 'IDX_OFFICIAL_XBRL',
            'status': 'CACHE_FALLBACK' if not cached.loc[cached['ticker'].eq(ticker)].empty else 'UNAVAILABLE',
            'rows': len(cached.loc[cached['ticker'].eq(ticker)]),
            'documents_found': 0, 'documents_parsed': 0,
            'error': 'Manifest IDX tidak tersedia; memakai cache filing terakhir' if not cached.loc[cached['ticker'].eq(ticker)].empty else 'Manifest IDX tidak tersedia atau tidak memiliki attachment XBRL',
            'stability': 'UNDOCUMENTED_PUBLIC_PAGE_ENDPOINT',
        } for ticker in names]
        report = pd.concat([pd.DataFrame(reports), manifest_report], ignore_index=True, sort=False)
        return (cached, report)
    manifest = (
        manifest.sort_values(['ticker', 'period_end', 'attachment_rank'])
        .drop_duplicates(['ticker', 'period_end', 'period_type'], keep='first')
        .reset_index(drop=True)
    )
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; IDXSuperScanner/5.5; research-client)',
        'Accept': 'application/zip, application/xml, text/xml, text/html, */*',
        'Referer': IDX_FINANCIAL_STATEMENTS_PAGE + '/',
    }

    def one(record: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        ticker = _safe_text(record.get('ticker'))
        url = _safe_text(record.get('attachment_url'))
        try:
            response = getter(url, headers=headers, timeout=timeout, allow_redirects=True)
            if int(getattr(response, 'status_code', 0)) != 200:
                raise RuntimeError(f'HTTP {getattr(response, "status_code", "?")}')
            final_url = _safe_text(getattr(response, 'url', url)) or url
            parsed = urlparse(final_url)
            host = (parsed.hostname or '').lower()
            if parsed.scheme != 'https' or not (host == 'idx.co.id' or host.endswith('.idx.co.id')):
                raise RuntimeError('redirect attachment keluar domain resmi IDX')
            content = bytes(getattr(response, 'content', b'') or b'')
            if not content:
                raise RuntimeError('attachment kosong')
            if len(content) > max(1, int(max_attachment_bytes)):
                raise RuntimeError('attachment melampaui batas ukuran aman')
            frame = parse_idx_xbrl_attachment(
                content, ticker=ticker, period_end=record.get('period_end'),
                period_type=_safe_text(record.get('period_type')),
                source_url=final_url, filename=_safe_text(record.get('filename')),
            )
            if frame.empty:
                raise RuntimeError('tidak ada fact canonical yang lolos context validation')
            return (frame, {'ticker': ticker, 'status': 'OK', 'rows': len(frame), 'error': ''})
        except Exception as exc:
            return (_fundamental_empty_history(), {
                'ticker': ticker, 'status': 'FAILED', 'rows': 0,
                'error': f'{type(exc).__name__}: {str(exc)[:180]}',
            })

    frames: list[pd.DataFrame] = []
    document_results: list[dict[str, Any]] = []
    workers = min(max(1, int(max_workers)), len(manifest))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, row.to_dict()) for _, row in manifest.iterrows()]
        for future in as_completed(futures):
            frame, report = future.result()
            if not frame.empty:
                frames.append(frame)
            document_results.append(report)
    current_history = combine_fundamental_history(*frames)
    history = combine_fundamental_history(cached, current_history)
    if not current_history.empty:
        merged_cache = combine_fundamental_history(cached_all, current_history)
        _write_cache('idx_fundamental_history', merged_cache)
    document_report = pd.DataFrame(document_results)
    reports: list[dict[str, Any]] = []
    for ticker in names:
        local_manifest = manifest.loc[manifest['ticker'].eq(ticker)]
        local_documents = document_report.loc[document_report['ticker'].eq(ticker)] if not document_report.empty else pd.DataFrame()
        parsed_count = int(local_documents['status'].eq('OK').sum()) if not local_documents.empty else 0
        row_count = int(len(history.loc[history['ticker'].eq(ticker)]))
        cached_count = int(len(cached.loc[cached['ticker'].eq(ticker)]))
        errors = ' | '.join(local_documents.loc[local_documents['status'].ne('OK'), 'error'].astype(str).head(4).tolist()) if not local_documents.empty else ''
        if parsed_count == len(local_manifest) and parsed_count:
            provider_status = 'OK'
        elif parsed_count:
            provider_status = 'PARTIAL_WITH_CACHE' if cached_count else 'PARTIAL'
        elif cached_count:
            provider_status = 'CACHE_FALLBACK'
        else:
            provider_status = 'NO_DATA'
        reports.append({
            'ticker': ticker, 'provider': 'IDX_OFFICIAL_XBRL',
            'status': provider_status,
            'rows': row_count, 'documents_found': len(local_manifest),
            'documents_parsed': parsed_count, 'error': errors,
            'stability': 'UNDOCUMENTED_PUBLIC_PAGE_ENDPOINT',
        })
    report = pd.concat([pd.DataFrame(reports), manifest_report], ignore_index=True, sort=False)
    return (history, report)


def combine_fundamental_history(*frames: pd.DataFrame) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return _fundamental_empty_history()
    return normalize_fundamental_history(pd.concat(usable, ignore_index=True, sort=False))


def _median_statement_periods(frame: pd.DataFrame, annual: bool=False) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = frame['period_type'].eq('FY') if annual else ~frame['period_type'].eq('FY')
    selected = frame.loc[mask].copy()
    if selected.empty:
        return selected
    numeric_columns = list(_FUNDAMENTAL_VALUE_ALIASES)
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors='coerce')
    aggregated = selected.groupby('period_end', as_index=False)[numeric_columns].median(numeric_only=True)
    return aggregated.sort_values('period_end').reset_index(drop=True)


def _period_change(frame: pd.DataFrame, column: str, lag: int) -> float:
    if len(frame) <= lag:
        return np.nan
    latest = _num(frame[column].iloc[-1])
    prior = _num(frame[column].iloc[-1 - lag])
    return latest / prior - 1.0 if np.isfinite(latest) and np.isfinite(prior) and prior > 0 else np.nan


def _ttm_sum(frame: pd.DataFrame, column: str) -> float:
    if len(frame) < 4:
        return np.nan
    values = pd.to_numeric(frame[column].tail(4), errors='coerce')
    return float(values.sum()) if values.notna().sum() == 4 else np.nan


def _safe_ratio(numerator: object, denominator: object) -> float:
    top = _num(numerator)
    bottom = _num(denominator)
    return top / bottom if np.isfinite(top) and np.isfinite(bottom) and abs(bottom) > 1e-12 else np.nan


def _fundamental_consensus(frame: pd.DataFrame) -> tuple[float, str, int]:
    """Compare overlapping reported facts across independent provider families."""
    families = sorted(set(frame['source_family'].dropna().astype(str)) - {'', 'USER_UPLOAD'})
    if len(families) < 2:
        return np.nan, '', 0
    comparisons: list[float] = []
    conflicts: list[str] = []
    overlap_facts = 0
    comparable = frame.copy()
    comparable['_frequency'] = np.where(comparable['period_type'].eq('FY'), 'FY', 'Q')
    for (period_end, period_type), period_rows in comparable.groupby(['period_end', '_frequency']):
        if period_rows['source_family'].nunique() < 2:
            continue
        for column in ('revenue', 'net_income', 'operating_cash_flow', 'total_assets', 'equity'):
            values = pd.to_numeric(period_rows.groupby('source_family')[column].median(), errors='coerce').dropna()
            if len(values) < 2:
                continue
            scale = max(abs(float(values.median())), 1.0)
            divergence = float((values.max() - values.min()) / scale)
            comparisons.append(abs(divergence))
            overlap_facts += 1
            if abs(divergence) > 0.20:
                conflicts.append(f'{column}@{pd.Timestamp(period_end).date()}/{period_type} {100 * abs(divergence):.0f}%')
    if not comparisons:
        return np.nan, '', 0
    median_divergence = float(np.median(comparisons))
    score = float(np.clip(100.0 - 400.0 * median_divergence, 0.0, 100.0))
    return round(score, 1), ' • '.join(conflicts[:6]), overlap_facts


def _history_features_for_ticker(ticker: str, history: pd.DataFrame, now: Any | None=None) -> dict[str, Any]:
    local = history.loc[history['ticker'].eq(ticker)].copy()
    if local.empty:
        return {'ticker': ticker, 'fundamental_data_grade': 'D', 'fundamental_history_quarters': 0, 'fundamental_history_years': 0}
    current = _as_jakarta_naive_timestamp(now if now is not None else pd.Timestamp.now(tz='Asia/Jakarta'))
    currency_counts = local['currency'].replace('', np.nan).dropna().value_counts()
    primary_currency = str(currency_counts.index[0]) if not currency_counts.empty else 'IDR'
    currency_conflict = len(currency_counts) > 1
    local = local.loc[local['currency'].eq(primary_currency)].copy()
    quarterly = _median_statement_periods(local, annual=False)
    annual = _median_statement_periods(local, annual=True)
    latest_period = local['period_end'].max()
    latest_age = int((current.normalize() - pd.Timestamp(latest_period).normalize()).days) if pd.notna(latest_period) else np.nan
    sources = sorted(set(local['source_family'].dropna().astype(str)) - {'', 'USER_UPLOAD'})
    all_sources = sorted(set(local['source_family'].dropna().astype(str)) - {''})
    official_referenced = bool(
        local['source_family'].isin({'IDX_OFFICIAL_REFERENCE', 'IDX_OFFICIAL_XBRL'}).any()
    )
    official_verified = bool(
        (
            local['source_family'].eq('IDX_OFFICIAL_XBRL')
            & local['source_verified'].fillna(False).astype(bool)
        ).any()
    )
    identity_flags = sorted({
        flag.strip()
        for value in local['validation_flags'].fillna('').astype(str)
        for flag in value.split(' • ')
        if flag.strip()
    })
    consensus, consensus_conflicts, overlap_facts = _fundamental_consensus(local)
    revenue_yoy = _period_change(quarterly, 'revenue', 4)
    earnings_yoy = _period_change(quarterly, 'net_income', 4)
    if not np.isfinite(revenue_yoy):
        revenue_yoy = _period_change(annual, 'revenue', 1)
    if not np.isfinite(earnings_yoy):
        earnings_yoy = _period_change(annual, 'net_income', 1)
    revenue_cagr_3y = np.nan
    if len(annual) >= 4:
        latest_revenue = _num(annual['revenue'].iloc[-1])
        prior_revenue = _num(annual['revenue'].iloc[-4])
        if latest_revenue > 0 and prior_revenue > 0:
            revenue_cagr_3y = (latest_revenue / prior_revenue) ** (1 / 3) - 1
    revenue_ttm = _ttm_sum(quarterly, 'revenue')
    net_income_ttm = _ttm_sum(quarterly, 'net_income')
    ocf_ttm = _ttm_sum(quarterly, 'operating_cash_flow')
    capex_ttm = _ttm_sum(quarterly, 'capex')
    operating_income_ttm = _ttm_sum(quarterly, 'operating_income')
    ebit_ttm = _ttm_sum(quarterly, 'ebit')
    ebitda_ttm = _ttm_sum(quarterly, 'ebitda')
    interest_ttm = _ttm_sum(quarterly, 'interest_expense')
    if np.isfinite(capex_ttm):
        fcf_ttm = ocf_ttm + capex_ttm if capex_ttm < 0 else ocf_ttm - capex_ttm
    else:
        fcf_ttm = np.nan
    latest = quarterly.iloc[-1] if not quarterly.empty else annual.iloc[-1] if not annual.empty else pd.Series(dtype=float)
    prior_year = quarterly.iloc[-5] if len(quarterly) >= 5 else annual.iloc[-2] if len(annual) >= 2 else pd.Series(dtype=float)
    equity_latest = _num(latest.get('equity'))
    equity_prior = _num(prior_year.get('equity'))
    assets_latest = _num(latest.get('total_assets'))
    assets_prior = _num(prior_year.get('total_assets'))
    average_equity = np.nanmean([equity_latest, equity_prior]) if np.isfinite(equity_latest) or np.isfinite(equity_prior) else np.nan
    average_assets = np.nanmean([assets_latest, assets_prior]) if np.isfinite(assets_latest) or np.isfinite(assets_prior) else np.nan
    shares_latest = _num(latest.get('shares_outstanding'))
    shares_prior = _num(prior_year.get('shares_outstanding'))
    debt_latest = _num(latest.get('total_debt'))
    cash_latest = _num(latest.get('cash'))
    positive_ocf_ratio = float(pd.to_numeric(quarterly['operating_cash_flow'].tail(8), errors='coerce').gt(0).mean()) if len(quarterly) else np.nan
    positive_earnings_ratio = float(pd.to_numeric(quarterly['net_income'].tail(8), errors='coerce').gt(0).mean()) if len(quarterly) else np.nan
    quarterly_margins = pd.to_numeric(quarterly['net_income'], errors='coerce') / pd.to_numeric(quarterly['revenue'], errors='coerce').replace(0, np.nan)
    margin_stability = float(np.clip(1.0 - quarterly_margins.tail(8).std() / max(abs(float(quarterly_margins.tail(8).mean())), 0.02), 0.0, 1.0)) if quarterly_margins.tail(8).notna().sum() >= 4 else np.nan
    invested_capital = equity_latest + debt_latest - cash_latest if all(np.isfinite(value) for value in (equity_latest, debt_latest, cash_latest)) else np.nan
    roic_proxy = _safe_ratio(0.78 * ebit_ttm, invested_capital) if np.isfinite(ebit_ttm) else np.nan
    feature_values = {
        'history_revenue_growth': revenue_yoy,
        'history_earnings_growth': earnings_yoy,
        'history_revenue_cagr_3y': revenue_cagr_3y,
        'history_roe': _safe_ratio(net_income_ttm, average_equity),
        'history_roa': _safe_ratio(net_income_ttm, average_assets),
        'history_operating_margin': _safe_ratio(operating_income_ttm, revenue_ttm),
        'history_net_margin': _safe_ratio(net_income_ttm, revenue_ttm),
        'history_cash_conversion': _safe_ratio(ocf_ttm, net_income_ttm) if np.isfinite(net_income_ttm) and net_income_ttm > 0 else np.nan,
        'history_fcf_ttm': fcf_ttm,
        'history_fcf_margin': _safe_ratio(fcf_ttm, revenue_ttm),
        'history_positive_ocf_ratio': positive_ocf_ratio,
        'history_positive_earnings_ratio': positive_earnings_ratio,
        'history_margin_stability': margin_stability,
        'history_share_dilution_yoy': _safe_ratio(shares_latest, shares_prior) - 1.0 if shares_prior > 0 else np.nan,
        'history_debt_equity': _safe_ratio(debt_latest, equity_latest),
        'history_net_debt_ebitda': _safe_ratio(debt_latest - cash_latest, ebitda_ttm),
        'history_interest_coverage': _safe_ratio(ebit_ttm, abs(interest_ttm)) if np.isfinite(interest_ttm) and interest_ttm != 0 else np.nan,
        'history_roic_proxy': roic_proxy,
        'history_revenue_ttm': revenue_ttm,
        'history_net_income_ttm': net_income_ttm,
        'history_ocf_ttm': ocf_ttm,
        'history_capex_ttm': capex_ttm,
        'history_ebitda_ttm': ebitda_ttm,
        'history_equity_latest': equity_latest,
        'history_assets_latest': assets_latest,
        'history_total_debt_latest': debt_latest,
        'history_cash_latest': cash_latest,
        'history_car': _num(latest.get('car')),
        'history_npl_gross': _num(latest.get('npl_gross')),
        'history_ldr': _num(latest.get('ldr')),
    }
    required_inputs = (
        revenue_ttm, net_income_ttm, ocf_ttm, fcf_ttm, assets_latest,
        equity_latest, debt_latest, cash_latest, shares_latest, ebit_ttm,
    )
    history_coverage = 100.0 * sum(np.isfinite(value) for value in required_inputs) / len(required_inputs)
    conflict_parts = [part for part in (consensus_conflicts, 'CURRENCY_CONFLICT' if currency_conflict else '', ' • '.join(identity_flags)) if part]
    severe_conflict = bool(identity_flags or (np.isfinite(consensus) and consensus < 50) or currency_conflict)
    adequate_history = len(quarterly) >= 8 or len(annual) >= 3
    partial_history = len(quarterly) >= 4 or len(annual) >= 2
    source_count = len(sources)
    if source_count >= 2 and adequate_history and history_coverage >= 70 and np.isfinite(consensus) and consensus >= 75 and latest_age <= 210 and not severe_conflict:
        grade = 'A'
    elif ((source_count >= 2 and partial_history) or (official_referenced and adequate_history)) and history_coverage >= 55 and latest_age <= 300 and not severe_conflict:
        grade = 'B'
    elif partial_history and history_coverage >= 40 and latest_age <= 365 and not severe_conflict:
        grade = 'C'
    else:
        grade = 'D'
    return {
        'ticker': ticker,
        **feature_values,
        'fundamental_history_quarters': int(len(quarterly)),
        'fundamental_history_years': int(len(annual)),
        'fundamental_history_coverage': round(history_coverage, 1),
        'fundamental_history_latest_period': latest_period,
        'fundamental_history_age_days': latest_age,
        'fundamental_source_count': source_count,
        'fundamental_all_source_count': len(all_sources),
        'fundamental_source_families': ' • '.join(all_sources),
        'fundamental_official_reference': official_referenced,
        'fundamental_official_verified': official_verified,
        'fundamental_consensus_score': consensus,
        'fundamental_consensus_overlap_facts': overlap_facts,
        'fundamental_conflicts': ' • '.join(conflict_parts),
        'fundamental_data_grade': grade,
        'fundamental_primary_currency': primary_currency,
    }


def build_fundamental_history_features(history: pd.DataFrame, now: Any | None=None) -> pd.DataFrame:
    normalized = normalize_fundamental_history(history)
    if normalized.empty:
        return pd.DataFrame()
    return pd.DataFrame([
        _history_features_for_ticker(ticker, normalized, now=now)
        for ticker in normalized['ticker'].drop_duplicates()
    ])


def _weighted_available_score(items: Iterable[tuple[float | None, float]]) -> tuple[float, float]:
    values = [(float(score), float(weight)) for score, weight in items if score is not None and np.isfinite(score)]
    total_weight = sum(weight for _, weight in values)
    possible_weight = sum(float(weight) for _, weight in items)
    score = sum(score * weight for score, weight in values) / total_weight if total_weight else np.nan
    coverage = 100.0 * total_weight / possible_weight if possible_weight else 0.0
    return score, coverage


def _history_business_score(row: Mapping[str, Any], is_financial: bool=False) -> tuple[float, float, str]:
    growth_inputs = [
        value for value in (
            _num(row.get('history_revenue_growth')),
            _num(row.get('history_revenue_cagr_3y')),
        ) if np.isfinite(value)
    ]
    growth = float(np.mean(growth_inputs)) if growth_inputs else np.nan
    earnings_growth = _num(row.get('history_earnings_growth'))
    cash_conversion = _num(row.get('history_cash_conversion'))
    dilution = _num(row.get('history_share_dilution_yoy'))
    fcf_margin = _num(row.get('history_fcf_margin'))
    positive_ocf = _num(row.get('history_positive_ocf_ratio'))
    positive_earnings = _num(row.get('history_positive_earnings_ratio'))
    margin_stability = _num(row.get('history_margin_stability'))
    debt_equity = _num(row.get('history_debt_equity'))
    net_debt_ebitda = _num(row.get('history_net_debt_ebitda'))
    interest_coverage = _num(row.get('history_interest_coverage'))
    items: list[tuple[float | None, float]] = [
        (_linear_score(growth, -0.05, 0.18), 12),
        (_linear_score(earnings_growth, -0.10, 0.22), 10),
        (_linear_score(_num(row.get('history_roe')), 0.05, 0.20), 10),
        (_linear_score(_num(row.get('history_roa')), 0.01, 0.09), 5),
        (_linear_score(_num(row.get('history_operating_margin')), 0.02, 0.18), 6),
        (_linear_score(_num(row.get('history_net_margin')), 0.01, 0.14), 6),
        (_linear_score(cash_conversion, 0.45, 1.10), 9),
        (_linear_score(fcf_margin, -0.02, 0.10), 6),
        (_linear_score(positive_ocf, 0.45, 1.0), 5),
        (_linear_score(positive_earnings, 0.50, 1.0), 4),
        (_linear_score(margin_stability, 0.20, 0.85), 4),
        (_linear_score(dilution, 0.12, 0.0, higher_is_better=True), 7),
        (_linear_score(_num(row.get('history_roic_proxy')), 0.04, 0.18), 6),
    ]
    if not is_financial:
        items.extend([
            (_linear_score(debt_equity, 2.0, 0.35, higher_is_better=True), 5),
            (_linear_score(net_debt_ebitda, 4.0, 0.5, higher_is_better=True), 3),
            (_linear_score(interest_coverage, 1.5, 8.0), 2),
        ])
    score, coverage = _weighted_available_score(items)
    flags: list[str] = []
    if np.isfinite(cash_conversion) and cash_conversion < 0.6:
        flags.append('Konversi laba ke kas lemah')
    if np.isfinite(dilution) and dilution > 0.08:
        flags.append('Dilusi saham >8% YoY')
    if np.isfinite(positive_ocf) and positive_ocf < 0.5:
        flags.append('OCF sering negatif')
    if np.isfinite(earnings_growth) and earnings_growth < 0:
        flags.append('Laba historis menyusut')
    return (round(float(score), 1) if np.isfinite(score) else np.nan, round(coverage, 1), ' • '.join(flags))


def enrich_fundamentals_with_history(fundamentals: pd.DataFrame, history: pd.DataFrame, now: Any | None=None) -> pd.DataFrame:
    """Merge snapshot and historical evidence without rewarding missing data."""
    base = fundamentals.copy() if fundamentals is not None else pd.DataFrame()
    features = build_fundamental_history_features(history, now=now)
    if base.empty and features.empty:
        return pd.DataFrame()
    if base.empty:
        base = pd.DataFrame({'ticker': features['ticker']})
    if features.empty:
        out = base.copy()
        out['fundamental_data_grade'] = 'D'
        out['fundamental_source_count'] = 1
        out['fundamental_source_families'] = 'YAHOO'
        out['fundamental_score_10'] = pd.to_numeric(out.get('fundamental_score'), errors='coerce') / 10.0
        out['fundamental_score_model'] = 'SNAPSHOT_ONLY_V1'
        return out
    out = base.merge(features, on='ticker', how='outer')
    snapshot_score = (
        pd.to_numeric(out['fundamental_score'], errors='coerce')
        if 'fundamental_score' in out else pd.Series(np.nan, index=out.index, dtype=float)
    )
    snapshot_coverage = (
        pd.to_numeric(out['fundamental_coverage'], errors='coerce')
        if 'fundamental_coverage' in out else pd.Series(0.0, index=out.index, dtype=float)
    )
    out['fundamental_score_snapshot'] = snapshot_score
    out['fundamental_coverage_snapshot'] = snapshot_coverage.fillna(0.0)
    out['fundamental_score'] = snapshot_score.astype(float)
    out['fundamental_coverage'] = snapshot_coverage.astype(float)
    for index, row in out.iterrows():
        is_financial = _safe_text(row.get('fundamental_model')).upper() == 'FINANCIAL'
        history_score, history_score_coverage, history_flags = _history_business_score(row, is_financial=is_financial)
        snapshot_score = _num(row.get('fundamental_score_snapshot'))
        grade = _safe_text(row.get('fundamental_data_grade')).upper() or 'D'
        if np.isfinite(history_score) and np.isfinite(snapshot_score):
            history_weight = 0.65 if grade in {'A', 'B'} else 0.50 if grade == 'C' else 0.25
            combined_score = history_weight * history_score + (1.0 - history_weight) * snapshot_score
        elif np.isfinite(history_score):
            combined_score = history_score
        else:
            combined_score = snapshot_score
        consensus = _num(row.get('fundamental_consensus_score'))
        conflicts = _safe_text(row.get('fundamental_conflicts'))
        if conflicts and (not np.isfinite(consensus) or consensus < 50 or 'ACCOUNTING_IDENTITY' in conflicts or 'CURRENCY_CONFLICT' in conflicts):
            combined_score = min(combined_score, 55.0) if np.isfinite(combined_score) else combined_score
        history_coverage = _num(row.get('fundamental_history_coverage'))
        snapshot_coverage = _num(row.get('fundamental_coverage_snapshot'))
        combined_coverage = 0.55 * history_coverage + 0.45 * snapshot_coverage if np.isfinite(history_score) else snapshot_coverage
        latest_history = pd.to_datetime(row.get('fundamental_history_latest_period'), errors='coerce')
        latest_snapshot = pd.to_datetime(row.get('latest_statement_date'), errors='coerce')
        latest_statement = max((value for value in (latest_history, latest_snapshot) if pd.notna(value)), default=pd.NaT)
        current = _as_jakarta_naive_timestamp(now if now is not None else pd.Timestamp.now(tz='Asia/Jakarta'))
        age = int((current.normalize() - latest_statement.normalize()).days) if pd.notna(latest_statement) else np.nan
        red_flags = [part for part in (_safe_text(row.get('fundamental_red_flags')), history_flags) if part]
        if conflicts:
            red_flags.append('Konflik data historis')
        overrides = {
            'revenue_growth': row.get('history_revenue_growth'),
            'earnings_growth': row.get('history_earnings_growth'),
            'roe': row.get('history_roe'), 'roa': row.get('history_roa'),
            'operating_margin': row.get('history_operating_margin'),
            'net_margin': row.get('history_net_margin'),
            'debt_equity': row.get('history_debt_equity'),
            'operating_cash_flow': row.get('history_ocf_ttm'),
            'free_cash_flow': row.get('history_fcf_ttm'),
        }
        for column, value in overrides.items():
            if np.isfinite(_num(value)):
                out.at[index, column] = value
        out.at[index, 'fundamental_history_score'] = history_score
        out.at[index, 'fundamental_history_score_coverage'] = history_score_coverage
        out.at[index, 'fundamental_score'] = round(float(combined_score), 1) if np.isfinite(combined_score) else np.nan
        out.at[index, 'fundamental_score_10'] = round(float(combined_score) / 10.0, 2) if np.isfinite(combined_score) else np.nan
        out.at[index, 'fundamental_coverage'] = round(float(np.clip(combined_coverage, 0, 100)), 1)
        out.at[index, 'fundamental_reliability'] = 'HIGH' if grade in {'A', 'B'} else 'MEDIUM' if grade == 'C' else 'LOW'
        out.at[index, 'fundamental_red_flags'] = ' • '.join(dict.fromkeys(flag for flag in red_flags if flag))
        out.at[index, 'latest_statement_date'] = latest_statement
        out.at[index, 'statement_age_days'] = age
        out.at[index, 'fundamental_score_model'] = 'MULTI_SOURCE_HISTORY_V1'
        providers = [_safe_text(row.get('fundamental_provider')), _safe_text(row.get('fundamental_source_families'))]
        out.at[index, 'fundamental_provider'] = ' + '.join(dict.fromkeys(value for value in providers if value))
    return out

def _fast_value(fast: Any, key: str, default: Any=np.nan) -> Any:
    try:
        if hasattr(fast, 'get'):
            value = fast.get(key, default)
        else:
            value = getattr(fast, key, default)
        return value
    except Exception:
        return default

def fetch_execution_snapshots(tickers: Iterable[str], max_workers: int=4) -> pd.DataFrame:
    """Fetch a fresh quote-state check for final candidates.

    This does not place orders and is not a replacement for an exchange/broker
    order book. It only prevents execution when the public quote is stale,
    inconsistent, non-equity, or has an excessive displayed spread.
    """
    import yfinance as yf
    names = list(dict.fromkeys(tickers))
    if not names:
        return pd.DataFrame()
    now = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None)

    def one(ticker: str) -> dict[str, Any]:
        try:
            obj = yf.Ticker(ticker)
            fast = obj.fast_info
            info = obj.get_info() or {}
            last_price = _num(_fast_value(fast, 'last_price', info.get('regularMarketPrice')))
            previous_close = _num(_fast_value(fast, 'previous_close', info.get('regularMarketPreviousClose')))
            bid = _num(info.get('bid'))
            ask = _num(info.get('ask'))
            volume = _num(_fast_value(fast, 'last_volume', info.get('regularMarketVolume')))
            raw_time = info.get('regularMarketTime')
            quote_time = pd.NaT
            if raw_time:
                quote_time = pd.to_datetime(raw_time, unit='s', utc=True).tz_convert('Asia/Jakarta').tz_localize(None)
            spread = (ask - bid) / ((ask + bid) / 2) if np.isfinite(bid) and np.isfinite(ask) and (bid > 0) and (ask >= bid) else np.nan
            exchange = str(info.get('exchange') or _fast_value(fast, 'exchange', '')).upper()
            quote_type = str(info.get('quoteType') or '').upper()
            market_state = str(info.get('marketState') or 'UNKNOWN').upper()
            verified = bool(np.isfinite(last_price) and last_price > 0 and np.isfinite(volume) and (volume > 0) and pd.notna(quote_time) and (quote_type == 'EQUITY') and (exchange in {'JKT', 'IDX', 'JAKARTA'}))
            return {'ticker': ticker, 'quote_checked_at': now, 'quote_time': quote_time, 'quote_last_price': last_price, 'quote_previous_close': previous_close, 'quote_bid': bid, 'quote_ask': ask, 'quote_spread_pct': spread, 'quote_volume': volume, 'quote_market_state': market_state, 'quote_exchange': exchange, 'quote_type': quote_type, 'quote_verified': verified, 'quote_error': ''}
        except Exception as exc:
            return {'ticker': ticker, 'quote_checked_at': now, 'quote_verified': False, 'quote_error': f'{type(exc).__name__}: {str(exc)[:100]}'}
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(names))) as pool:
        futures = [pool.submit(one, ticker) for ticker in names]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)
from pathlib import Path
import json
import os
import tempfile

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
    min_adtv_idr: float = 2000000000.0
    min_atr_pct: float = 0.008
    max_atr_pct: float = 0.1
    max_zero_volume_ratio: float = 0.05
    min_score: float = 72.0
    execution_score: float = 82.0
    min_rr1: float = 1.8
    min_rr2: float = 2.7
    max_stop_pct: float = 0.07
    ready_distance_atr: float = 0.3
    max_entry_gap_atr: float = 0.15
    watch_distance_atr: float = 1.75
    max_zone_age_bars: int = 20
    max_data_lag_days: int = 3
    max_absolute_data_age_days: int = 5
    fee_roundtrip_pct: float = 0.004
    slippage_roundtrip_pct: float = 0.0025
    backtest_horizon_bars: int = 20
    backtest_entry_window_bars: int = 5
    backtest_min_gap_bars: int = 10
    walkforward_min_train_fraction: float = 0.6
    walkforward_folds: int = 4
    min_oos_unique_dates: int = 10
    beta_prior_wins: float = 8.0
    beta_prior_losses: float = 8.0
    fundamental_top_n: int = 80
    fundamental_history_top_n: int = 40
    idx_fundamental_top_n: int = 20
    idx_fundamental_years_back: int = 3
    fundamental_crosscheck_top_n: int = 8
    twelve_fundamental_top_n: int = 20
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
    min_oos_signal_events: int = 30
    min_oos_filled_events: int = 20
    min_oos_fill_rate_pct: float = 25.0
    min_oos_bayes_probability_pct: float = 52.0
    min_oos_tp1_ci_low_pct: float = 35.0
    min_oos_expectancy_r: float = 0.1
    min_oos_profit_factor: float = 1.15
    max_oos_losing_streak: int = 8
    min_execution_confidence: float = 82.0
    min_pending_confidence: float = 68.0
    min_data_completeness: float = 80.0
    min_direct_fundamental_coverage: float = 45.0
    market_status_cache_days: int = 7
    news_cache_days: int = 3
    fundamental_cache_days: int = 21
    execution_policy: str = 'SIGNAL_FIRST'
    provider_retry_count: int = 2
    max_intraday_stale_minutes: int = 20
    require_independent_price_verification: bool = True
    max_independent_price_age_days: int = 5
    max_independent_date_gap_days: int = 0
    max_secondary_price_divergence_pct: float = 0.0075
    min_secondary_overlap_bars: int = 5
    min_secondary_return_correlation: float = 0.97
    max_automatic_price_candidates: int = 40
    max_itick_independent_candidates: int = 4
    automatic_provider_timeout_seconds: int = 10
    idx_summary_lookback_days: int = 7
    google_finance_max_workers: int = 4
    autopilot_enabled: bool = True
    min_autopilot_completeness: float = 85.0
    min_autopilot_confidence: float = 82.0
    min_autopilot_validation_score: float = 70.0
    allow_autopilot_risk_off: bool = False
    account_size_idr: float = 5000000.0
    cash_on_hand_idr: float = 5000000.0
    risk_per_trade_pct: float = 0.005
    max_portfolio_risk_pct: float = 0.015
    max_positions: int = 3
    max_position_pct: float = 0.35
    buy_fee_pct: float = 0.0015
    sell_fee_pct: float = 0.0025
    order_slippage_pct: float = 0.0025
    max_order_pct_adtv: float = 0.005
    max_execution_spread_pct: float = 0.015
    max_specialty_positions: int = 1
    max_specialty_portfolio_risk_pct: float = 0.005
    specialty_risk_per_trade_pct: float = 0.0035
    specialty_max_position_pct: float = 0.20
    daily_radar_limit: int = 15
    profit_order_builder_limit: int = 20
    profit_conviction_min_score: float = 68.0
    multibagger_profit_allocation_pct: float = 1.0
    multibagger_capital_budget_idr: float = 0.0
    multibagger_max_holdings: int = 5
    multibagger_min_capital_conviction: float = 72.0
    multibagger_core_cap_pct: float = 0.35
    multibagger_high_cap_pct: float = 0.25
    multibagger_satellite_cap_pct: float = 0.15
    multibagger_starter_cap_pct: float = 0.08
    automatic_forward_quality_enabled: bool = True
    automatic_forward_quality_top_n: int = 12
    automatic_forward_quality_cache_days: int = 14
    automatic_forward_quality_timeout_seconds: float = 8.0
    automatic_forward_quality_max_documents: int = 5
    automatic_forward_quality_workers: int = 4
    # Free local hybrid AI. Rule-based structural validity remains authoritative;
    # AI adjusts ranking only in proportion to empirical sample quality.
    ai_enabled: bool = True
    ai_mode: str = 'HYBRID_GUARDED'
    ai_max_weight: float = 0.35
    ai_min_training_events: int = 30
    ai_min_strategy_events: int = 18
    ai_knn_k: int = 21
    ai_memory_entry_window_bars: int = 5
    ai_memory_horizon_bars: int = 20
    # Objective Astronacci-style time-cycle intelligence. It only affects
    # daily core/swing ranking and Multibagger timing, never intraday specialty.
    time_cycle_enabled: bool = True
    time_cycle_core_max_weight: float = 0.10
    time_cycle_multibagger_max_weight: float = 0.05
    time_cycle_min_confidence: float = 55.0
    time_cycle_lunar_enabled: bool = True
    time_cycle_min_history_bars: int = 180
    # Clean-room Eye-of-Future reconstruction. These controls affect only the
    # existing daily time-cycle overlay; intraday/BPJS/BSJP/ARA remain excluded.
    eoff_enabled: bool = True
    eoff_ephemeris_enabled: bool = True
    eoff_min_fib_cluster: int = 4
    eoff_aspect_orb_deg: float = 3.0
    eoff_require_astro_fib_confluence: bool = True
    max_avg_down_loss_pct: float = 0.1
    max_avg_down_position_pct: float = 0.28

    def replace(self, **changes: object) -> 'ScanConfig':
        values = self.__dict__.copy()
        values.update(changes)
        return ScanConfig(**values)


def build_source_quorum_audit(
    tickers: Iterable[str],
    source_tiers: Mapping[str, str] | None=None,
    price_validation: pd.DataFrame | None=None,
    fundamental_history: pd.DataFrame | None=None,
    market_status: pd.DataFrame | None=None,
    news_review: pd.DataFrame | None=None,
    validation_stats: pd.DataFrame | None=None,
    intraday_report: Any | None=None,
    broker_summary: pd.DataFrame | None=None,
    orderbook: pd.DataFrame | None=None,
    config: ScanConfig | None=None,
    benchmark_secondary_verified: bool=False,
) -> pd.DataFrame:
    """Explain where two-source validation is real, absent, or inappropriate.

    Provider fallback is deliberately not counted as quorum: a ticker fetched
    from Yahoo *or* iTick still has one observed OHLCV family. Official trading
    restrictions and the user's broker portfolio are authority-first layers;
    two unofficial sources must never overrule the authoritative record.
    """
    cfg = config or ScanConfig()
    names = [normalize_idx_ticker(value) for value in tickers]
    names = [value for value in dict.fromkeys(names) if value]
    denominator = max(1, len(names))

    def percentage(count: int) -> float:
        return round(100.0 * max(0, int(count)) / denominator, 1)

    def state(count: int, *, authority: bool=False, manual: bool=False, method: bool=False) -> str:
        if authority:
            return 'AUTHORITATIVE_VERIFIED' if count >= len(names) and names else 'AUTHORITATIVE_PARTIAL'
        if manual:
            return 'MANUAL_BROKER_CHECK'
        if method:
            return 'METHOD_VALIDATED' if count > 0 else 'METHOD_LIMITED'
        if count >= len(names) and names:
            return 'TWO_SOURCE_VERIFIED'
        if count > 0:
            return 'TWO_SOURCE_PARTIAL'
        return 'SINGLE_OR_UNAVAILABLE'

    primary_families = {
        _primary_source_family_from_tier((source_tiers or {}).get(ticker, ''))
        for ticker in names
    } - {'', 'UNKNOWN', 'UNAVAILABLE'}
    price = price_validation if price_validation is not None else pd.DataFrame()
    price_verified: set[str] = set()
    technical_verified: set[str] = set()
    if not price.empty and 'ticker' in price:
        verified = price.get('independent_price_verified', False)
        if not isinstance(verified, pd.Series):
            verified = pd.Series(False, index=price.index)
        price_verified = set(price.loc[verified.map(_truthy), 'ticker'].dropna().astype(str))
        overlap = pd.to_numeric(price.get('independent_overlap_bars', 0), errors='coerce').fillna(0)
        correlation = pd.to_numeric(price.get('independent_return_correlation', np.nan), errors='coerce')
        technical_mask = verified.map(_truthy) & overlap.ge(cfg.min_secondary_overlap_bars) & correlation.ge(cfg.min_secondary_return_correlation)
        technical_verified = set(price.loc[technical_mask, 'ticker'].dropna().astype(str))

    history = normalize_fundamental_history(fundamental_history if fundamental_history is not None else pd.DataFrame())
    fundamental_verified: set[str] = set()
    fundamental_families: set[str] = set()
    if not history.empty:
        eligible = history.loc[~history['source_family'].isin(['', 'USER_UPLOAD'])].copy()
        fundamental_families = set(eligible['source_family'].dropna().astype(str))
        counts = eligible.groupby('ticker')['source_family'].nunique()
        two_source = set(counts[counts >= 2].index.astype(str))
        verified_official = set(
            eligible.loc[
                eligible['source_family'].eq('IDX_OFFICIAL_XBRL')
                & eligible['source_verified'].map(_truthy),
                'ticker',
            ].dropna().astype(str)
        )
        # Two aggregators are useful cross-checks, but they cannot replace the
        # automatically parsed regulator filing for Multibagger-grade evidence.
        fundamental_verified = two_source & verified_official

    status = market_status if market_status is not None else pd.DataFrame()
    status_verified: set[str] = set()
    if not status.empty and 'ticker' in status:
        mask = status.get('market_status_verified', False)
        if not isinstance(mask, pd.Series):
            mask = pd.Series(False, index=status.index)
        status_verified = set(status.loc[mask.map(_truthy), 'ticker'].dropna().astype(str))

    news = news_review if news_review is not None else pd.DataFrame()
    news_verified: set[str] = set()
    if not news.empty and 'ticker' in news:
        broad = news.get('provider_query_ok', False)
        disclosure = news.get('idx_disclosure_query_ok', False)
        if not isinstance(broad, pd.Series):
            broad = pd.Series(False, index=news.index)
        if not isinstance(disclosure, pd.Series):
            disclosure = pd.Series(False, index=news.index)
        news_verified = set(news.loc[broad.map(_truthy) & disclosure.map(_truthy), 'ticker'].dropna().astype(str))

    validation = validation_stats if validation_stats is not None else pd.DataFrame()
    validation_count = 0
    if not validation.empty:
        scope = validation.get('validation_scope', pd.Series('', index=validation.index)).astype(str)
        events = pd.to_numeric(validation.get('signal_events_oos', 0), errors='coerce').fillna(0)
        validation_count = int((scope.eq('CHRONOLOGICAL_OOS_HOLDOUT') & events.ge(cfg.min_oos_signal_events)).sum())

    intraday_families: set[str] = set()
    if intraday_report is not None:
        tiers = getattr(intraday_report, 'source_tiers', None)
        if isinstance(tiers, Mapping):
            intraday_families = {
                _primary_source_family_from_tier(value) for value in tiers.values()
            } - {'', 'UNKNOWN', 'UNAVAILABLE'}

    rows = [
        {
            'data_layer': 'Technical daily OHLCV', 'policy': 'TWO_SOURCE_QUORUM',
            'observed_sources': ' • '.join(sorted(primary_families)) or 'UNAVAILABLE',
            'verified_tickers': len(technical_verified), 'universe_tickers': len(names),
            'coverage_pct': percentage(len(technical_verified)), 'quorum_state': state(len(technical_verified)),
            'order_effect': 'Research tetap tampil; full-history technical quorum hanya bila overlap return path lulus.',
        },
        {
            'data_layer': 'Execution price / last close', 'policy': 'TWO_SOURCE_QUORUM',
            'observed_sources': 'PRIMARY_OHLCV + IDX/GOOGLE/iTICK/TWELVE',
            'verified_tickers': len(price_verified), 'universe_tickers': len(names),
            'coverage_pct': percentage(len(price_verified)), 'quorum_state': state(len(price_verified)),
            'order_effect': 'Tanpa harga independen: hanya SIGNAL_READY; tidak ada tiket atau kandidat manual.',
        },
        {
            'data_layer': 'Fundamental statements', 'policy': 'TWO_SOURCE_QUORUM',
            'observed_sources': ' • '.join(sorted(fundamental_families)) or 'UNAVAILABLE',
            'verified_tickers': len(fundamental_verified), 'universe_tickers': len(names),
            'coverage_pct': percentage(len(fundamental_verified)), 'quorum_state': state(len(fundamental_verified)),
            'order_effect': 'Multibagger A wajib dua sumber overlap + filing IDX/XBRL otomatis terverifikasi.',
        },
        {
            'data_layer': 'News and issuer disclosure', 'policy': 'TWO_SOURCE_QUORUM',
            'observed_sources': 'GOOGLE_NEWS_OR_YAHOO + IDX_DISCLOSURE',
            'verified_tickers': len(news_verified), 'universe_tickers': len(names),
            'coverage_pct': percentage(len(news_verified)), 'quorum_state': state(len(news_verified)),
            'order_effect': 'Kegagalan provider menurunkan confidence; berita negatif resmi tetap memblokir.',
        },
        {
            'data_layer': 'IDX restrictions / suspension / FCA', 'policy': 'AUTHORITATIVE_SOURCE',
            'observed_sources': 'OFFICIAL_IDX',
            'verified_tickers': len(status_verified), 'universe_tickers': len(names),
            'coverage_pct': percentage(len(status_verified)), 'quorum_state': state(len(status_verified), authority=True),
            'order_effect': 'Satu sumber regulator lebih kuat daripada consensus dua agregator.',
        },
        {
            'data_layer': 'IHSG benchmark and market regime', 'policy': 'TWO_SOURCE_QUORUM',
            'observed_sources': 'YAHOO + SECONDARY_BENCHMARK' if benchmark_secondary_verified else 'YAHOO + INTERNAL_BREADTH',
            'verified_tickers': len(names) if benchmark_secondary_verified else 0,
            'universe_tickers': len(names), 'coverage_pct': 100.0 if benchmark_secondary_verified else 0.0,
            'quorum_state': 'TWO_SOURCE_VERIFIED' if benchmark_secondary_verified else 'SINGLE_PROVIDER_WITH_BREADTH_CHECK',
            'order_effect': 'Breadth adalah cross-check model, bukan provider data kedua.',
        },
        {
            'data_layer': 'Intraday 5m', 'policy': 'TWO_SOURCE_QUORUM',
            'observed_sources': ' • '.join(sorted(intraday_families)) or 'UNAVAILABLE',
            'verified_tickers': 0, 'universe_tickers': len(names), 'coverage_pct': 0.0,
            'quorum_state': 'SINGLE_PROVIDER_OR_MANUAL',
            'order_effect': 'Stockbit bid/offer dan last price wajib dicek manual sebelum submit specialty order.',
        },
        {
            'data_layer': 'Broker summary and order book', 'policy': 'BROKER_SOURCE_OF_TRUTH',
            'observed_sources': 'STOCKBIT_EXPORT' if ((broker_summary is not None and not broker_summary.empty) or (orderbook is not None and not orderbook.empty)) else 'OHLCV_PROXY_ONLY',
            'verified_tickers': 0, 'universe_tickers': len(names), 'coverage_pct': 0.0,
            'quorum_state': state(0, manual=True),
            'order_effect': 'Proxy OHLCV tidak dihitung sebagai broker summary/order book kedua.',
        },
        {
            'data_layer': 'Walk-forward validation', 'policy': 'METHOD_INDEPENDENCE',
            'observed_sources': 'CHRONOLOGICAL_OOS',
            'verified_tickers': validation_count, 'universe_tickers': len(names),
            'coverage_pct': np.nan, 'quorum_state': state(validation_count, method=True),
            'order_effect': 'Yang diperlukan adalah holdout waktu dan sample cukup, bukan duplikasi provider.',
        },
    ]
    return pd.DataFrame(rows)


def _safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    return str(value).strip()

def _append_pipe(frame: pd.DataFrame, index: object, column: str, message: str) -> None:
    if not message:
        return
    if column not in frame:
        frame[column] = ''
    prior = _safe_text(frame.at[index, column])
    pieces = [piece.strip() for piece in prior.split(' • ') if piece.strip()]
    if message not in pieces:
        pieces.append(message)
    frame.at[index, column] = ' • '.join(pieces)

def _set_context_block(frame: pd.DataFrame, index: object, message: str, reject: bool=False) -> None:
    _append_pipe(frame, index, 'critical_blockers', message)
    _append_pipe(frame, index, 'blockers', message)
    current = _safe_text(frame.at[index, 'status'] if 'status' in frame else '')
    if reject:
        frame.at[index, 'status'] = 'REJECT'
    elif current not in {'REJECT', 'BLOCKED_CONTEXT'}:
        frame.at[index, 'status'] = 'BLOCKED_CONTEXT'

def _cache_root() -> Path:
    override = os.environ.get('IDX_SCANNER_CACHE_DIR', '').strip()
    root = Path(override) if override else Path(__file__).resolve().parent / '.scanner_cache'
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        root = Path(tempfile.gettempdir()) / 'idx_super_scanner_cache'
        root.mkdir(parents=True, exist_ok=True)
    return root

def _load_cache(name: str) -> pd.DataFrame:
    path = _cache_root() / f'{name}.json'
    if not path.is_file():
        return pd.DataFrame()
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        rows = payload.get('rows', []) if isinstance(payload, dict) else []
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def _write_cache(name: str, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        return
    path = _cache_root() / f'{name}.json'
    tmp = path.with_suffix('.tmp')
    payload = {'written_at': pd.Timestamp.now(tz='Asia/Jakarta').isoformat(), 'rows': frame.replace({np.nan: None, pd.NaT: None}).to_dict('records')}
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding='utf-8')
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
    stamp = pd.to_datetime(value, errors='coerce')
    if pd.isna(stamp):
        return pd.NaT
    stamp = pd.Timestamp(stamp)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert('Asia/Jakarta').tz_localize(None)
    return stamp

def _cache_row_is_fresh(row: Mapping[str, Any] | None, timestamp_column: str, max_age_days: int, usable, now: pd.Timestamp | None=None) -> bool:
    if row is None or not usable(row):
        return False
    reference = now if now is not None else pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None)
    stamp = _as_jakarta_naive_timestamp(row.get(timestamp_column))
    if pd.isna(stamp):
        return False
    age = (reference.normalize() - stamp.normalize()).days
    return 0 <= age <= max_age_days

def _merge_resilient_rows(current: pd.DataFrame, cached: pd.DataFrame, tickers: Iterable[str], timestamp_column: str, current_usable, cached_max_age_days: int, cache_label: str) -> pd.DataFrame:
    names = list(dict.fromkeys(tickers))
    now = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None)
    current_map = {} if current is None or current.empty else {str(row['ticker']): row.to_dict() for _, row in current.dropna(subset=['ticker']).iterrows()}
    cached_map = {} if cached is None or cached.empty else {str(row['ticker']): row.to_dict() for _, row in cached.dropna(subset=['ticker']).iterrows()}
    rows: list[dict[str, Any]] = []
    for ticker in names:
        cur = current_map.get(ticker)
        if cur is not None and current_usable(cur):
            cur = dict(cur)
            cur['evidence_source_tier'] = 'LIVE'
            rows.append(cur)
            continue
        old = cached_map.get(ticker)
        old_ok = _cache_row_is_fresh(old, timestamp_column, cached_max_age_days, current_usable, now=now)
        if old_ok:
            old = dict(old)
            old['evidence_source_tier'] = cache_label
            rows.append(old)
            continue
        fallback = dict(cur or {'ticker': ticker})
        fallback.setdefault('ticker', ticker)
        fallback['evidence_source_tier'] = 'UNRESOLVED'
        rows.append(fallback)
    return pd.DataFrame(rows)

def fetch_resilient_market_status(tickers: Iterable[str], config: ScanConfig | None=None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    names = list(dict.fromkeys(tickers))
    current = pd.DataFrame()
    for _ in range(max(1, cfg.provider_retry_count)):
        current = fetch_automatic_market_status(names, timeout=8)
        if not current.empty and current.get('market_status_verified', pd.Series(dtype=bool)).map(_truthy).any():
            break
    cached = _load_cache('market_status')
    usable = lambda row: _truthy(row.get('market_status_verified', False))
    resolved = _merge_resilient_rows(current, cached, names, 'market_status_asof', usable, cfg.market_status_cache_days, 'CACHE_FALLBACK')
    if not current.empty:
        live = current[current.get('market_status_verified', False).map(_truthy)].copy()
        if not live.empty:
            prior = cached[~cached.get('ticker', pd.Series(dtype=str)).isin(live['ticker'])] if not cached.empty and 'ticker' in cached else pd.DataFrame()
            _write_cache('market_status', pd.concat([live, prior], ignore_index=True))
    return resolved

def fetch_resilient_news_review(tickers: Iterable[str], lookback_days: int=7, config: ScanConfig | None=None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    names = list(dict.fromkeys(tickers))
    current = pd.DataFrame()
    for _ in range(max(1, cfg.provider_retry_count)):
        current = fetch_automatic_news_review(names, lookback_days=lookback_days)
        if not current.empty and current.get('provider_query_ok', pd.Series(dtype=bool)).map(_truthy).any():
            break
    cached = _load_cache('news_review')

    def usable(row: Mapping[str, Any]) -> bool:
        return _safe_text(row.get('news_review_status')).upper() == 'COMPLETE' and _truthy(row.get('provider_query_ok', False))
    resolved = _merge_resilient_rows(current, cached, names, 'news_reviewed_at', usable, cfg.news_cache_days, 'CACHE_FALLBACK')
    if not current.empty:
        mask = current.get('provider_query_ok', False)
        if not isinstance(mask, pd.Series):
            mask = pd.Series(False, index=current.index)
        live = current[mask.map(_truthy)].copy()
        if not live.empty:
            prior = cached[~cached.get('ticker', pd.Series(dtype=str)).isin(live['ticker'])] if not cached.empty and 'ticker' in cached else pd.DataFrame()
            _write_cache('news_review', pd.concat([live, prior], ignore_index=True))
    return resolved

def fetch_resilient_fundamentals(tickers: Iterable[str], config: ScanConfig | None=None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    names = list(dict.fromkeys(tickers))
    cached = _load_cache('fundamentals')

    def usable(row: Mapping[str, Any]) -> bool:
        return _finite(row.get('fundamental_coverage'), 0) >= 45 and (not _safe_text(row.get('fundamental_error')))
    cached_map = {} if cached is None or cached.empty else {str(row['ticker']): row.to_dict() for _, row in cached.dropna(subset=['ticker']).drop_duplicates('ticker', keep='last').iterrows()}
    now = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None)
    refresh_names = [ticker for ticker in names if not _cache_row_is_fresh(cached_map.get(ticker), 'fundamental_fetched_at', cfg.fundamental_cache_days, usable, now=now)]
    current = fetch_fundamentals(refresh_names) if refresh_names else pd.DataFrame()
    resolved = _merge_resilient_rows(current, cached, names, 'fundamental_fetched_at', usable, cfg.fundamental_cache_days, 'CACHE_FALLBACK')
    if not current.empty:
        live = current[current.apply(lambda row: usable(row.to_dict()), axis=1)].copy()
        if not live.empty:
            prior = cached[~cached.get('ticker', pd.Series(dtype=str)).isin(live['ticker'])] if not cached.empty and 'ticker' in cached else pd.DataFrame()
            _write_cache('fundamentals', pd.concat([live, prior], ignore_index=True))
    return resolved

def apply_validation_gate(signals: pd.DataFrame, config: ScanConfig | None=None) -> pd.DataFrame:
    """Score historical evidence without discarding a live setup for missing data.

    A sufficiently large, clearly negative OOS sample remains a critical block.
    Otherwise the result contributes a confidence score and warning.
    """
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    out['validation_critical_blocker'] = False
    for idx, row in out.iterrows():
        checks = {'scope': _safe_text(row.get('validation_scope')) == 'CHRONOLOGICAL_OOS_HOLDOUT', 'signals': _finite(row.get('signal_events_oos'), 0) >= cfg.min_oos_signal_events, 'filled': _finite(row.get('filled_events'), 0) >= cfg.min_oos_filled_events, 'fill_rate': _finite(row.get('entry_fill_rate_5d'), 0) >= cfg.min_oos_fill_rate_pct, 'probability': _finite(row.get('bayes_probability'), 0) >= cfg.min_oos_bayes_probability_pct, 'ci': _finite(row.get('tp1_ci_low'), 0) >= cfg.min_oos_tp1_ci_low_pct, 'expectancy': _finite(row.get('expectancy_r'), -99) >= cfg.min_oos_expectancy_r, 'profit_factor': _finite(row.get('profit_factor'), 0) >= cfg.min_oos_profit_factor, 'losing_streak': _finite(row.get('max_losing_streak'), 999) <= cfg.max_oos_losing_streak}
        passed = sum(checks.values())
        raw_score = 100 * passed / len(checks)
        filled = _finite(row.get('filled_events'), 0)
        expectancy = _finite(row.get('expectancy_r'), np.nan)
        pf = _finite(row.get('profit_factor'), np.nan)
        bayes = _finite(row.get('bayes_probability'), np.nan)
        demonstrated_negative = bool(filled >= cfg.min_oos_filled_events and (np.isfinite(expectancy) and expectancy < -0.05 or (np.isfinite(pf) and pf < 0.85) or (np.isfinite(bayes) and bayes < 43.0)))
        if all(checks.values()):
            confidence, tier = (100.0, 'ROBUST')
        elif checks['scope'] and filled >= 10 and np.isfinite(expectancy) and (expectancy > 0):
            confidence, tier = (max(65.0, raw_score), 'USABLE')
        elif demonstrated_negative:
            confidence, tier = (min(25.0, raw_score), 'NEGATIVE_EDGE')
        else:
            confidence, tier = (max(45.0, min(64.0, raw_score)), 'LIMITED')
        out.at[idx, 'validation_gate_score'] = round(raw_score, 1)
        out.at[idx, 'validation_confidence'] = round(confidence, 1)
        out.at[idx, 'validation_tier'] = tier
        out.at[idx, 'validation_gate_pass'] = bool(all(checks.values()))
        if demonstrated_negative:
            out.at[idx, 'validation_critical_blocker'] = True
            _set_context_block(out, idx, 'OOS menunjukkan edge negatif yang material')
        elif not all(checks.values()):
            failed = ', '.join((name for name, ok in checks.items() if not ok))
            _append_pipe(out, idx, 'evidence_warnings', f'Validasi historis terbatas: {failed}')
    out['status_rank'] = out['status'].map(STATUS_ORDER).fillna(99)
    return out

def apply_fundamental_gate(signals: pd.DataFrame, config: ScanConfig | None=None) -> pd.DataFrame:
    """Use fundamentals as weighted quality evidence.

    Missing fundamentals no longer erase a short-term technical setup. A truly
    distressed combination (negative margin/cash flow plus leverage) blocks.
    """
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    defaults = {'fundamental_score': np.nan, 'fundamental_coverage': 0.0, 'fundamental_reliability': 'NONE', 'fundamental_red_flags': '', 'fundamental_error': 'Fundamental tidak tersedia', 'statement_age_days': np.nan, 'fundamental_data_grade': 'D', 'fundamental_conflicts': ''}
    for column, default in defaults.items():
        if column not in out:
            out[column] = default
    out['fundamental_critical_blocker'] = False
    for idx, row in out.iterrows():
        score = _finite(row.get('fundamental_score'), np.nan)
        coverage = _finite(row.get('fundamental_coverage'), 0)
        age = _finite(row.get('statement_age_days'), np.nan)
        data_grade = _safe_text(row.get('fundamental_data_grade')).upper() or 'D'
        data_conflicts = _safe_text(row.get('fundamental_conflicts'))
        flags = _safe_text(row.get('fundamental_red_flags'))
        flag_set = {part.strip() for part in flags.split('•') if part.strip()}
        distressed = bool({'Margin bersih negatif', 'OCF negatif'}.issubset(flag_set) or {'DER tinggi', 'OCF negatif'}.issubset(flag_set) or (coverage >= 60 and np.isfinite(score) and (score < 25)))
        if distressed:
            confidence, tier = (0.0, 'DISTRESSED')
            out.at[idx, 'fundamental_critical_blocker'] = True
            _set_context_block(out, idx, 'Fundamental distress: arus kas/profitabilitas/leverage tidak aman')
        elif coverage >= cfg.min_fundamental_coverage and np.isfinite(score) and (score >= cfg.min_fundamental_score) and (np.isfinite(age) and 0 <= age <= cfg.max_statement_age_days):
            confidence, tier = (min(100.0, max(70.0, score)), 'STRONG')
        elif coverage >= 45 and np.isfinite(score):
            confidence, tier = (min(78.0, max(52.0, score)), 'PARTIAL')
            _append_pipe(out, idx, 'evidence_warnings', 'Fundamental parsial atau belum mencapai quality threshold')
        else:
            confidence, tier = (50.0, 'MISSING_NEUTRAL')
            _append_pipe(out, idx, 'evidence_warnings', 'Fundamental belum lengkap; bobot confidence dikurangi')
        if data_conflicts:
            confidence = min(confidence, 55.0)
            tier = 'DATA_CONFLICT'
            _append_pipe(out, idx, 'evidence_warnings', 'Laporan fundamental berbeda antar-sumber; wajib review manual')
        elif data_grade in {'A', 'B'} and tier == 'STRONG':
            tier = f'STRONG_DATA_{data_grade}'
        out.at[idx, 'fundamental_confidence'] = round(confidence, 1)
        out.at[idx, 'fundamental_tier'] = tier
    out['status_rank'] = out['status'].map(STATUS_ORDER).fillna(99)
    return out

def apply_market_status_gate(signals: pd.DataFrame, market_status: pd.DataFrame, config: ScanConfig | None=None, asof: object | None=None) -> pd.DataFrame:
    """Block explicit IDX restrictions; score missing provider coverage."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    reference = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None)
    if market_status is None or market_status.empty:
        market_status = pd.DataFrame({'ticker': out['ticker'].drop_duplicates()})
    out = out.merge(market_status, on='ticker', how='left')
    stamp_source = out['market_status_asof'] if 'market_status_asof' in out else pd.Series(pd.NaT, index=out.index)
    stamp = pd.to_datetime(stamp_source, errors='coerce')
    out['market_status_age_days'] = (reference.normalize() - stamp.dt.normalize()).dt.days
    out['market_status_critical_blocker'] = False
    for idx, row in out.iterrows():
        verified = _truthy(row.get('market_status_verified', False))
        age = _finite(row.get('market_status_age_days'), np.nan)
        fresh = verified and np.isfinite(age) and (0 <= age <= cfg.market_status_cache_days)
        source_tier = _safe_text(row.get('evidence_source_tier')) or ('LIVE' if verified else 'UNRESOLVED')
        out.at[idx, 'market_status_coverage'] = 'AUTO_VERIFIED' if fresh else 'FALLBACK_REQUIRED'
        out.at[idx, 'market_status_confidence'] = 100.0 if fresh and source_tier == 'LIVE' else 82.0 if fresh else 45.0
        if not fresh:
            _append_pipe(out, idx, 'evidence_warnings', 'Status IDX resmi belum lengkap; quote/OHLCV fallback akan digunakan')
        negative = []
        if _truthy(row.get('suspended', False)):
            negative.append('suspensi')
        if _truthy(row.get('fca', False)) or _truthy(row.get('special_monitoring', False)):
            negative.append('FCA/pemantauan khusus')
        notation = _safe_text(row.get('special_notation'))
        if notation:
            negative.append(f'notasi {notation}')
        if _truthy(row.get('corporate_action', False)):
            negative.append('aksi korporasi material')
        if negative:
            out.at[idx, 'market_status_critical_blocker'] = True
            out.at[idx, 'market_status_confidence'] = 0.0
            _set_context_block(out, idx, 'Status IDX negatif: ' + ', '.join(negative), reject=True)
    out['status_rank'] = out['status'].map(STATUS_ORDER).fillna(99)
    return out

def apply_news_gate(signals: pd.DataFrame, news_review: pd.DataFrame, config: ScanConfig | None=None, asof: object | None=None) -> pd.DataFrame:
    """Treat provider failure as reduced confidence, not as proof of bad news."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    reference = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None)
    if news_review is None or news_review.empty:
        news_review = pd.DataFrame({'ticker': out['ticker'].drop_duplicates()})
    out = out.merge(news_review, on='ticker', how='left')
    review_source = out['news_reviewed_at'] if 'news_reviewed_at' in out else pd.Series(pd.NaT, index=out.index)
    start_source = out['coverage_start'] if 'coverage_start' in out else pd.Series(pd.NaT, index=out.index)
    end_source = out['coverage_end'] if 'coverage_end' in out else pd.Series(pd.NaT, index=out.index)
    review_time = pd.to_datetime(review_source, errors='coerce')
    coverage_start = pd.to_datetime(start_source, errors='coerce')
    coverage_end = pd.to_datetime(end_source, errors='coerce')
    out['news_review_age_days'] = (reference.normalize() - review_time.dt.normalize()).dt.days
    out['news_lookback_days'] = (coverage_end.dt.normalize() - coverage_start.dt.normalize()).dt.days
    out['news_critical_blocker'] = False
    for idx, row in out.iterrows():
        status = _safe_text(row.get('news_review_status')).upper() or 'MISSING'
        age = _finite(row.get('news_review_age_days'), np.nan)
        lookback = _finite(row.get('news_lookback_days'), np.nan)
        complete = bool(status == 'COMPLETE' and _truthy(row.get('provider_query_ok', False)) and np.isfinite(age) and (0 <= age <= cfg.news_cache_days) and np.isfinite(lookback) and (lookback >= cfg.min_news_lookback_days))
        positive = _finite(row.get('verified_catalyst_count'), 0)
        negative = _finite(row.get('verified_negative_count'), 0)
        disclosure_field_present = 'idx_disclosure_query_ok' in out.columns
        disclosure_ok = _truthy(row.get('idx_disclosure_query_ok', False)) if disclosure_field_present else True
        confidence = 100.0 if complete and disclosure_ok else 88.0 if complete else 52.0
        if complete and positive > 0 and (negative == 0) and disclosure_ok:
            confidence = 100.0
        elif complete and negative > 0:
            confidence = 75.0
        if complete and (not disclosure_ok):
            _append_pipe(out, idx, 'evidence_warnings', 'Berita luas tersedia; cross-check keterbukaan IDX belum lengkap')
        out.at[idx, 'news_confidence'] = confidence
        if not complete:
            _append_pipe(out, idx, 'evidence_warnings', 'Coverage berita parsial; tidak dianggap sebagai berita negatif')
        if _truthy(row.get('severe_negative_news', False)):
            out.at[idx, 'news_critical_blocker'] = True
            out.at[idx, 'news_confidence'] = 0.0
            _set_context_block(out, idx, 'Berita negatif material terverifikasi', reject=True)
        elif _truthy(row.get('ambiguous_material_news', False)):
            out.at[idx, 'news_critical_blocker'] = True
            out.at[idx, 'news_confidence'] = 15.0
            _set_context_block(out, idx, 'Aksi korporasi material belum direkonsiliasi')
    out['status_rank'] = out['status'].map(STATUS_ORDER).fillna(99)
    return out

def apply_universe_integrity_gate(signals: pd.DataFrame, requested_tickers: Iterable[str], prepared_tickers: Iterable[str], config: ScanConfig | None=None) -> pd.DataFrame:
    """Score breadth quality; do not erase a stock-specific setup."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    requested = list(dict.fromkeys((str(t) for t in requested_tickers if str(t))))
    prepared = set((str(t) for t in prepared_tickers if str(t)))
    requested_count = len(requested)
    prepared_count = sum((1 for ticker in requested if ticker in prepared))
    coverage_pct = 100.0 * prepared_count / requested_count if requested_count else 0.0
    passed = bool(requested_count >= cfg.min_regime_universe_size and coverage_pct >= cfg.min_regime_coverage_pct)
    out['universe_requested_count'] = requested_count
    out['universe_prepared_count'] = prepared_count
    out['universe_coverage_pct'] = round(coverage_pct, 1)
    out['universe_gate_pass'] = passed
    out['universe_confidence'] = 100.0 if passed else 62.0 if prepared_count >= 50 else 48.0
    if not passed:
        _msg = f'Breadth universe terbatas: {requested_count} ticker, coverage {coverage_pct:.1f}%; benchmark IHSG tetap digunakan'
        for idx in out.index:
            _append_pipe(out, idx, 'evidence_warnings', _msg)
    out['status_rank'] = out['status'].map(STATUS_ORDER).fillna(99)
    return out

def enforce_portfolio_execution_budget(signals: pd.DataFrame, config: ScanConfig | None=None, current_positions: int=0, current_open_risk_idr: float=0.0, current_invested_idr: float=0.0, cash_on_hand_idr: float | None=None) -> pd.DataFrame:
    """Rank technically ready orders against actual cash and aggregate risk."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if 'technical_execution_candidate' not in out:
        out['technical_execution_candidate'] = out['status'].eq('EXECUTION_READY')
    out['portfolio_selected'] = False
    out['execution_rank'] = np.nan
    max_risk = cfg.account_size_idr * cfg.max_portfolio_risk_pct
    remaining_risk = max(0.0, max_risk - max(0.0, current_open_risk_idr))
    if cash_on_hand_idr is None:
        cash_value = max(0.0, cfg.account_size_idr - max(0.0, current_invested_idr))
    else:
        cash_value = max(0.0, float(cash_on_hand_idr))
    remaining_cash = cash_value
    slots = max(0, int(cfg.max_positions) - max(0, int(current_positions)))
    candidate_index = out.index[out['status'].eq('EXECUTION_READY')].tolist()
    if not candidate_index:
        out['portfolio_remaining_risk_idr'] = remaining_risk
        out['portfolio_remaining_cash_idr'] = remaining_cash
        return out
    sort_cols = [c for c in ('pre_budget_confidence', 'composite_score', 'quality_score', 'bayes_probability', 'rr2', 'adtv20_idr') if c in out]
    ranked = out.loc[candidate_index].sort_values(sort_cols, ascending=False, na_position='last') if sort_cols else out.loc[candidate_index]
    seen: set[str] = set()
    selected = 0
    for idx, row in ranked.iterrows():
        ticker = _safe_text(row.get('ticker'))
        risk = _finite(row.get('max_loss_idr'), float('inf'))
        capital = _finite(row.get('capital_required_idr'), float('inf'))
        reason = ''
        if ticker in seen:
            reason = 'Hanya satu setup terbaik per ticker'
        elif selected >= slots:
            reason = 'Slot posisi portofolio sudah penuh'
        elif risk > remaining_risk:
            reason = 'Risiko agregat portofolio melampaui batas'
        elif capital > remaining_cash:
            reason = 'Cash on hand tidak cukup'
        if reason:
            _append_pipe(out, idx, 'portfolio_blockers', reason)
            if _safe_text(out.at[idx, 'status']) == 'EXECUTION_READY':
                out.at[idx, 'status'] = 'PENDING_DATA'
            continue
        selected += 1
        seen.add(ticker)
        remaining_risk -= max(0.0, risk)
        remaining_cash -= max(0.0, capital)
        out.at[idx, 'portfolio_selected'] = True
        out.at[idx, 'execution_rank'] = selected
    out['portfolio_remaining_risk_idr'] = remaining_risk
    out['portfolio_remaining_cash_idr'] = remaining_cash
    out['status_rank'] = out['status'].map(STATUS_ORDER).fillna(99)
    return out

def _risk_layer_confidence(row: Mapping[str, Any], cfg: ScanConfig) -> float:
    entry = _finite(row.get('entry'), np.nan)
    stop = _finite(row.get('stop_loss'), np.nan)
    tp1 = _finite(row.get('tp1'), np.nan)
    tp2 = _finite(row.get('tp2'), np.nan)
    levels_ok = all((is_valid_idx_price(v) for v in (entry, stop, tp1, tp2))) and stop < entry < tp1 < tp2
    rr_ok = _finite(row.get('rr1'), 0) >= cfg.min_rr1 and _finite(row.get('rr2'), 0) >= cfg.min_rr2
    stop_ok = _finite(row.get('stop_pct'), 99) <= cfg.max_stop_pct
    sizing_ok = _safe_text(row.get('sizing_status')) == 'OK' and int(_finite(row.get('suggested_lots'), 0)) >= 1
    return 100.0 * sum((levels_ok, rr_ok, stop_ok, sizing_ok)) / 4

def parse_portfolio_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    frame = _read_csv(source)
    frame.columns = [str(column).replace('\ufeff', '').strip().lower().replace(' ', '_') for column in frame.columns]
    aliases = {'symbol': 'ticker', 'kode': 'ticker', 'emiten': 'ticker', 'lot': 'lots', 'jumlah_lot': 'lots', 'qty_lot': 'lots', 'avg': 'avg_price', 'average': 'avg_price', 'average_price': 'avg_price', 'harga_rata_rata': 'avg_price', 'stop': 'stop_loss', 'sl': 'stop_loss'}
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
    required = {'ticker', 'lots', 'avg_price'}
    if not required.issubset(frame.columns):
        found = ', '.join(map(str, frame.columns)) or 'tidak ada'
        raise ValueError(f'Portfolio CSV wajib memiliki kolom ticker, lots, dan avg_price. Kolom terbaca: {found}')
    out = pd.DataFrame()
    out['ticker'] = frame['ticker'].map(normalize_idx_ticker)
    out['lots'] = pd.to_numeric(frame['lots'], errors='coerce').fillna(0).astype(int)
    out['avg_price'] = pd.to_numeric(frame['avg_price'], errors='coerce')
    out['manual_stop_loss'] = pd.to_numeric(frame.get('stop_loss', np.nan), errors='coerce')
    out['manual_tp'] = pd.to_numeric(frame.get('take_profit', frame.get('tp', np.nan)), errors='coerce')
    out['notes'] = frame.get('notes', '').fillna('').astype(str) if 'notes' in frame else ''
    out = out.dropna(subset=['ticker', 'avg_price'])
    out = out[(out['lots'] > 0) & (out['avg_price'] > 0)]
    if out.empty:
        return out
    out['shares'] = out['lots'] * 100
    out['cost_value'] = out['shares'] * out['avg_price']
    grouped = out.groupby('ticker', as_index=False).agg(lots=('lots', 'sum'), shares=('shares', 'sum'), cost_value=('cost_value', 'sum'), manual_stop_loss=('manual_stop_loss', 'last'), manual_tp=('manual_tp', 'last'), notes=('notes', 'last'))
    grouped['avg_price'] = grouped['cost_value'] / grouped['shares']
    return grouped.drop(columns='cost_value')

def _portfolio_fundamental_lookup(fundamentals: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if fundamentals is None or fundamentals.empty or 'ticker' not in fundamentals:
        return {}
    return {str(row['ticker']): row.to_dict() for _, row in fundamentals.iterrows()}

def _portfolio_signal_lookup(signals: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if signals is None or signals.empty or 'ticker' not in signals:
        return {}
    ranked = signals.copy()
    rank_col = 'status_rank' if 'status_rank' in ranked else None
    if rank_col:
        ranked = ranked.sort_values([rank_col, 'quality_score'], ascending=[True, False], na_position='last')
    return {ticker: group.iloc[0].to_dict() for ticker, group in ranked.groupby('ticker', sort=False)}

def analyze_portfolio_positions(portfolio: pd.DataFrame, histories: Mapping[str, pd.DataFrame], fundamentals: pd.DataFrame | None=None, signals: pd.DataFrame | None=None, account_equity_idr: float | None=None, cash_on_hand_idr: float=0.0, config: ScanConfig | None=None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Analyze Stockbit holdings independently from the universe scanner.

    A missing entry signal is neutral for an existing holding. CUT_LOSS is
    reserved for an actual stop breach, a confirmed multi-factor structure
    breakdown, or severe fundamental distress accompanied by price weakness.
    """
    cfg = config or ScanConfig()
    if portfolio is None or portfolio.empty:
        return (pd.DataFrame(), {'positions': 0, 'market_value_idr': 0.0, 'cost_value_idr': 0.0, 'unrealized_pnl_idr': 0.0, 'unrealized_pnl_pct': 0.0, 'open_risk_idr': 0.0, 'open_risk_pct_equity': 0.0, 'cash_on_hand_idr': float(cash_on_hand_idr), 'estimated_equity_idr': float(account_equity_idr or cash_on_hand_idr), 'equity_source': 'INPUT' if account_equity_idr else 'ESTIMATED'})
    fund_map = _portfolio_fundamental_lookup(fundamentals if fundamentals is not None else pd.DataFrame())
    signal_map = _portfolio_signal_lookup(signals if signals is not None else pd.DataFrame())
    provisional_rows: list[dict[str, Any]] = []
    total_market = 0.0
    total_cost = 0.0
    for _, position in portfolio.iterrows():
        ticker = str(position['ticker'])
        raw = histories.get(ticker)
        if raw is None or raw.empty:
            provisional_rows.append({**position.to_dict(), 'last_price': np.nan, 'position_action': 'DATA_REQUIRED', 'action_reason': 'OHLCV posisi tidak tersedia', 'current_value_idr': np.nan, 'cost_value_idr': float(position['shares'] * position['avg_price'])})
            continue
        frame = prepare_indicators(raw)
        if frame.empty:
            provisional_rows.append({**position.to_dict(), 'last_price': np.nan, 'position_action': 'DATA_REQUIRED', 'action_reason': 'Indikator posisi tidak dapat dihitung', 'current_value_idr': np.nan, 'cost_value_idr': float(position['shares'] * position['avg_price'])})
            continue
        row = frame.iloc[-1]
        prev = frame.iloc[-2] if len(frame) >= 2 else row
        close = _finite(row.get('Close'), np.nan)
        atr_v = max(_finite(row.get('ATR14'), 0.0), idx_tick_size(close) * 3)
        shares = int(position['shares'])
        avg = float(position['avg_price'])
        market_value = shares * close
        cost_value = shares * avg
        total_market += market_value
        total_cost += cost_value
        recent20 = frame.iloc[-20:]
        recent120 = frame.iloc[-120:]
        ema20_v = _finite(row.get('EMA20'), close)
        ema50_v = _finite(row.get('EMA50'), close)
        ema200_v = _finite(row.get('EMA200'), close)
        tick = idx_tick_size(close)
        pivot_lows = pd.to_numeric(recent120.get('PIVOT_LOW'), errors='coerce').dropna()
        pivot_lows = pivot_lows[pivot_lows <= close * 1.02]
        latest_pivot_low = _finite(pivot_lows.iloc[-1], np.nan) if len(pivot_lows) else np.nan
        support_candidates = [value for value in (ema20_v, ema50_v, latest_pivot_low, _finite(recent20['Low'].min(), np.nan)) if np.isfinite(value) and value <= close * 1.02]
        structural_support = max(support_candidates) if support_candidates else close - atr_v
        structural_stop = round_idx_price(max(cfg.min_price, structural_support - max(0.5 * atr_v, 2 * tick)), 'down')
        manual_stop = _finite(position.get('manual_stop_loss'), np.nan)
        existing_stop = manual_stop if np.isfinite(manual_stop) and manual_stop > 0 else np.nan
        valid_below_stops = [s for s in (existing_stop, structural_stop) if np.isfinite(s) and s < close]
        suggested_stop = max(valid_below_stops) if valid_below_stops else existing_stop if np.isfinite(existing_stop) else structural_stop
        stop_breached = bool(np.isfinite(existing_stop) and close <= existing_stop or close <= structural_stop)
        last_two = pd.to_numeric(frame['Close'].iloc[-2:], errors='coerce')
        two_close_break = bool(len(last_two) == 2 and (last_two < min(structural_support, ema50_v)).all() and (ema20_v < ema50_v) and (_finite(row.get('CMF20'), 0.0) < 0) and (_finite(row.get('OBV_SLOPE10'), 0.0) < 0))
        confirmed_breakdown = stop_breached or two_close_break
        manual_tp = _finite(position.get('manual_tp'), np.nan)
        recent_high = _finite(recent120['High'].max(), close)
        recent_low = _finite(recent120['Low'].min(), structural_support)
        portfolio_targets = _price_structure_target_pair(
            frame, close, setup='PORTFOLIO',
            explicit_levels=[
                (manual_tp, 'MANUAL_TP'),
                (_finite(row.get('HIGH20_PREV'), np.nan), 'PRIOR_20D_HIGH'),
                (_finite(row.get('HIGH55_PREV'), np.nan), 'PRIOR_55D_HIGH'),
                (_finite(row.get('HIGH252'), np.nan), 'PRIOR_52W_HIGH'),
            ],
            projection_origin=recent_high,
            projection_height=max(recent_high - recent_low, tick),
        )
        suggested_tp1 = portfolio_targets['tp1']
        suggested_tp2 = portfolio_targets['tp2']
        tp1_basis = portfolio_targets['tp1_basis']
        tp2_basis = portfolio_targets['tp2_basis']
        if not portfolio_targets['target_structure_valid']:
            suggested_tp1 = np.nan
            suggested_tp2 = np.nan
        pnl = market_value - cost_value
        pnl_pct = close / avg - 1
        trend_up = close > ema50_v and ema20_v >= ema50_v and (ema50_v >= 0.98 * ema200_v)
        long_term_intact = close > ema200_v
        flow_positive = _finite(row.get('CMF20'), -1) >= 0 and _finite(row.get('OBV_SLOPE10'), -1) > 0
        trend_weak = close < ema50_v or ema20_v < ema50_v or _finite(row.get('CMF20'), 0) < 0
        momentum_hot = _finite(row.get('RSI14'), 50) >= 74 or close > ema20_v + 2.0 * atr_v
        near_support = abs(close - structural_support) <= 1.25 * atr_v
        bullish_confirmation = bool(_truthy(row.get('BULL_REJECTION', False)) or (close > _finite(prev.get('High'), close) and close > _finite(row.get('Open'), close)))
        fund = fund_map.get(ticker, {})
        fund_flags = _safe_text(fund.get('fundamental_red_flags'))
        fund_score = _finite(fund.get('fundamental_score'), np.nan)
        fund_coverage = _finite(fund.get('fundamental_coverage'), 0.0)
        distressed = bool('Margin bersih negatif' in fund_flags and 'OCF negatif' in fund_flags or ('DER tinggi' in fund_flags and 'OCF negatif' in fund_flags) or (fund_coverage >= 60 and np.isfinite(fund_score) and (fund_score < 25)))
        severe_distress_break = distressed and close < ema200_v and (not flow_positive)
        sig = signal_map.get(ticker, {})
        scanner_status = _safe_text(sig.get('status'))
        scanner_setup_ready = scanner_status in {'EXECUTION_READY', 'READY_FOR_PRICE_VERIFY', 'PENDING_DATA'}
        internal_add_setup = bool(long_term_intact and near_support and flow_positive and bullish_confirmation and (not confirmed_breakdown) and (not distressed))
        setup_ready = scanner_setup_ready or internal_add_setup
        add_confirmation = bullish_confirmation or scanner_setup_ready
        provisional_rows.append({**position.to_dict(), 'last_price': close, 'current_value_idr': market_value, 'cost_value_idr': cost_value, 'unrealized_pnl_idr': pnl, 'unrealized_pnl_pct': pnl_pct, 'ema20': ema20_v, 'ema50': ema50_v, 'ema200': ema200_v, 'rsi14': _finite(row.get('RSI14'), np.nan), 'cmf20': _finite(row.get('CMF20'), np.nan), 'structural_support': structural_support, 'existing_stop_loss': existing_stop, 'structural_stop_loss': structural_stop, 'suggested_stop_loss': suggested_stop, 'suggested_tp1': suggested_tp1, 'suggested_tp2': suggested_tp2, 'tp1_basis': tp1_basis, 'tp2_basis': tp2_basis, 'trend_up': trend_up, 'trend_weak': trend_weak, 'long_term_structure_intact': long_term_intact, 'flow_positive': flow_positive, 'near_support': near_support, 'bullish_confirmation': bullish_confirmation, 'stop_breached': stop_breached, 'confirmed_structure_breakdown': confirmed_breakdown, 'momentum_hot': momentum_hot, 'fundamental_distress': distressed, 'scanner_setup': sig.get('setup', ''), 'scanner_status': scanner_status, 'portfolio_add_setup': 'INTERNAL_CONFIRMED' if internal_add_setup else '', 'setup_ready': setup_ready, 'add_confirmation': add_confirmation, 'severe_distress_break': severe_distress_break})
    inferred_equity = total_market + max(0.0, cash_on_hand_idr)
    if account_equity_idr is not None and float(account_equity_idr) > 0:
        estimated_equity = max(float(account_equity_idr), total_market)
        equity_source = 'ACCOUNT_EQUITY_INPUT'
    else:
        estimated_equity = max(inferred_equity, total_market, 1.0)
        equity_source = 'POSITIONS_PLUS_CASH'
    estimated_equity = max(estimated_equity, 1.0)
    rows: list[dict[str, Any]] = []
    open_risk_total = 0.0
    for item in provisional_rows:
        if not np.isfinite(_finite(item.get('last_price'), np.nan)):
            rows.append(item)
            continue
        value = _finite(item.get('current_value_idr'), 0)
        weight = value / estimated_equity
        close = _finite(item.get('last_price'), 0)
        stop = _finite(item.get('suggested_stop_loss'), 0)
        shares = int(item.get('shares', 0))
        open_risk = max(0.0, close - stop) * shares + close * shares * cfg.sell_fee_pct
        open_risk_total += open_risk
        pnl_pct = _finite(item.get('unrealized_pnl_pct'), 0)
        action = 'HOLD'
        reason = 'Struktur utama dan money flow belum memberikan sinyal keluar'
        if _truthy(item.get('stop_breached', False)):
            action = 'CUT_LOSS'
            reason = 'Harga telah menyentuh/menembus stop aktif atau structural stop'
        elif _truthy(item.get('confirmed_structure_breakdown', False)):
            action = 'CUT_LOSS'
            reason = 'Dua penutupan mengonfirmasi breakdown support dengan trend dan flow bearish'
        elif _truthy(item.get('severe_distress_break', False)):
            action = 'CUT_LOSS'
            reason = 'Fundamental distress berat disertai struktur harga jangka panjang yang rusak'
        elif weight > cfg.max_position_pct:
            action = 'REDUCE'
            reason = f'Bobot posisi {weight:.1%} melebihi batas {cfg.max_position_pct:.0%}'
        elif pnl_pct >= 0.15 and _truthy(item.get('momentum_hot', False)):
            action = 'TAKE_PROFIT_PARTIAL'
            reason = 'Profit signifikan dan harga overextended; realisasikan sebagian'
        elif pnl_pct >= 0.08 and (not _truthy(item.get('flow_positive', False))):
            action = 'REDUCE'
            reason = 'Posisi masih profit tetapi money flow melemah'
        elif pnl_pct < 0:
            avg_allowed = bool(pnl_pct >= -cfg.max_avg_down_loss_pct and weight <= cfg.max_avg_down_position_pct and _truthy(item.get('long_term_structure_intact', False)) and _truthy(item.get('near_support', False)) and _truthy(item.get('flow_positive', False)) and _truthy(item.get('add_confirmation', False)) and (not _truthy(item.get('fundamental_distress', False))) and _truthy(item.get('setup_ready', False)) and (cash_on_hand_idr > 0))
            if avg_allowed:
                action = 'AVG_DOWN_ALLOWED'
                reason = 'Loss terbatas, support bertahan, flow dan candle konfirmasi positif, serta bobot masih aman'
            elif not _truthy(item.get('long_term_structure_intact', False)) or _truthy(item.get('fundamental_distress', False)):
                action = 'DO_NOT_AVG_DOWN'
                reason = 'Struktur jangka panjang belum sehat atau fundamental distress'
            elif _truthy(item.get('trend_weak', False)):
                action = 'HOLD_TIGHT_STOP'
                reason = 'Belum breakdown, tetapi trend/flow melemah; jangan tambah dan gunakan structural stop'
            else:
                action = 'HOLD_NO_AVG'
                reason = 'Belum ada konfirmasi lengkap untuk menambah posisi'
        elif _truthy(item.get('trend_weak', False)):
            action = 'HOLD_TIGHT_STOP'
            reason = 'Belum breakdown, tetapi trend menengah melemah'
        avg_lots = 0
        new_average = np.nan
        if action == 'AVG_DOWN_ALLOWED':
            risk_budget = max(0.0, estimated_equity * cfg.risk_per_trade_pct - open_risk)
            per_share_risk = max(idx_tick_size(close), close - stop) + close * (cfg.buy_fee_pct + cfg.sell_fee_pct)
            lots_by_risk = int(risk_budget // (per_share_risk * 100)) if per_share_risk > 0 else 0
            lots_by_cash = int(cash_on_hand_idr // (close * 100 * (1 + cfg.buy_fee_pct)))
            max_extra_value = max(0.0, cfg.max_position_pct * estimated_equity - value)
            lots_by_weight = int(max_extra_value // (close * 100 * (1 + cfg.buy_fee_pct)))
            avg_lots = max(0, min(lots_by_risk, lots_by_cash, lots_by_weight))
            if avg_lots < 1:
                action = 'HOLD_NO_AVG'
                reason = 'Setup mendukung, tetapi cash/risk budget tidak cukup untuk 1 lot'
            else:
                new_shares = shares + avg_lots * 100
                new_average = (item['cost_value_idr'] + avg_lots * 100 * close * (1 + cfg.buy_fee_pct)) / new_shares
        item.update({'position_weight': weight, 'position_weight_pct': weight * 100.0, 'open_risk_idr': open_risk, 'open_risk_pct_equity': open_risk / estimated_equity, 'open_risk_pct_equity_pct': open_risk / estimated_equity * 100.0, 'position_action': action, 'action_reason': reason, 'avg_down_lots': avg_lots, 'avg_down_price': close if avg_lots >= 1 else np.nan, 'new_average_after_avg': new_average, 'equity_basis_idr': estimated_equity, 'equity_source': equity_source})
        rows.append(item)
    result = pd.DataFrame(rows)
    summary = {'positions': int(len(portfolio)), 'market_value_idr': float(total_market), 'cost_value_idr': float(total_cost), 'unrealized_pnl_idr': float(total_market - total_cost), 'unrealized_pnl_pct': float(total_market / total_cost - 1) if total_cost > 0 else 0.0, 'open_risk_idr': float(open_risk_total), 'open_risk_pct_equity': float(open_risk_total / estimated_equity), 'cash_on_hand_idr': float(max(0.0, cash_on_hand_idr)), 'estimated_equity_idr': float(estimated_equity), 'inferred_equity_idr': float(inferred_equity), 'equity_source': equity_source}
    return (result, summary)

def _fetch_official_idx_pages(timeout: int=8) -> tuple[dict[str, str], dict[str, str]]:
    import requests
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; IDXSuperScanner/4.1; research-client)', 'Accept-Language': 'id-ID,id;q=0.9,en;q=0.7'}

    def one(item: tuple[str, str]) -> tuple[str, str | None, str | None]:
        key, url = item
        try:
            response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
            if response.status_code != 200:
                raise RuntimeError(f'HTTP {response.status_code}')
            if not _is_exact_official_idx_url(response.url):
                raise RuntimeError('redirect keluar domain resmi IDX')
            text = response.text or ''
            if len(text) < 1000:
                raise RuntimeError('respons terlalu pendek')
            return (key, text, None)
        except Exception as exc:
            return (key, None, f'{type(exc).__name__}: {str(exc)[:120]}')
    pages: dict[str, str] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(IDX_AUTOMATION_URLS)) as pool:
        for key, text, error in pool.map(one, IDX_AUTOMATION_URLS.items()):
            if text is not None:
                pages[key] = text
            if error is not None:
                errors[key] = error
    return (pages, errors)

def _fetch_idx_disclosure_page(timeout: int=8) -> tuple[str, bool, str]:
    import requests
    url = 'https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi'
    try:
        response = requests.get(url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0 (compatible; IDXSuperScanner/4.1)'}, allow_redirects=True)
        if response.status_code != 200 or not _is_exact_official_idx_url(response.url):
            raise RuntimeError(f'HTTP/redirect {response.status_code}')
        text = _html_text(response.text or '')
        ok = len(text) > 1000 and ('KETERBUKAAN INFORMASI' in text or 'DISCLOSURE' in text)
        return (text, ok, '' if ok else 'semantic marker tidak ditemukan')
    except Exception as exc:
        return ('', False, f'{type(exc).__name__}: {str(exc)[:120]}')
_DATA_LAYER_WEIGHTS: dict[str, float] = {'technical': 0.25, 'risk': 0.15, 'fundamental': 0.15, 'validation': 0.15, 'market_status': 0.1, 'news': 0.1, 'quote': 0.05, 'universe': 0.05}

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
    present = sum((_is_present_value(row.get(name)) for name in names))
    return 100.0 * present / len(names)

def _technical_data_coverage(row: Mapping[str, Any]) -> float:
    return _field_coverage(row, ('ticker', 'setup', 'quality_score', 'last_price', 'entry_low', 'entry_high', 'entry', 'stop_loss', 'tp1', 'tp2', 'volume_ratio', 'adtv20_idr', 'distance_atr', 'market_regime'))

def _risk_data_coverage(row: Mapping[str, Any]) -> float:
    return _field_coverage(row, ('entry', 'stop_loss', 'tp1', 'tp2', 'rr1', 'rr2', 'stop_pct', 'suggested_lots', 'capital_required_idr', 'max_loss_idr'))

def _validation_data_coverage(row: Mapping[str, Any]) -> float:
    return _field_coverage(row, ('validation_scope', 'signal_events_oos', 'filled_events', 'entry_fill_rate_5d', 'bayes_probability', 'tp1_ci_low', 'expectancy_r', 'profit_factor', 'max_losing_streak', 'median_fill_bars', 'median_time_to_tp1_bars'))

def _market_status_data_coverage(row: Mapping[str, Any], cfg: ScanConfig) -> float:
    status = _safe_text(row.get('market_status_coverage')).upper()
    if status == 'AUTO_VERIFIED':
        return 100.0
    components = {part.strip() for part in _safe_text(row.get('market_status_components')).split(',') if part.strip()}
    component_score = min(100.0, 20.0 * len(components))
    data_age = _finite(row.get('absolute_data_age_days'), 999)
    final_ohlcv = data_age <= cfg.max_absolute_data_age_days and (not _truthy(row.get('current_bar_incomplete', False)))
    quote_verified = _truthy(row.get('quote_verified', False))
    if quote_verified:
        return max(55.0, component_score)
    if final_ohlcv:
        return max(35.0, component_score)
    return max(20.0 if _is_present_value(row.get('market_status_asof')) else 0.0, component_score)

def _news_data_coverage(row: Mapping[str, Any], cfg: ScanConfig) -> float:
    status = _safe_text(row.get('news_review_status')).upper()
    provider_ok = _truthy(row.get('provider_query_ok', False))
    age = _finite(row.get('news_review_age_days'), np.nan)
    lookback = _finite(row.get('news_lookback_days'), np.nan)
    structural = _is_present_value(row.get('news_reviewed_at')) and _is_present_value(row.get('coverage_start')) and _is_present_value(row.get('coverage_end'))
    fresh = np.isfinite(age) and 0 <= age <= cfg.news_cache_days
    adequate_window = np.isfinite(lookback) and lookback >= cfg.min_news_lookback_days
    if status == 'COMPLETE' and provider_ok and fresh and adequate_window:
        if 'idx_disclosure_query_ok' in row and (not _truthy(row.get('idx_disclosure_query_ok', False))):
            return 90.0
        return 100.0
    if provider_ok and structural and fresh:
        return 80.0 if adequate_window else 70.0
    if status in {'INCOMPLETE', 'PARTIAL'} and structural:
        return 55.0
    if structural:
        return 40.0
    return 20.0

def _quote_data_coverage(row: Mapping[str, Any], cfg: ScanConfig) -> float:
    verified = _truthy(row.get('quote_verified', False))
    age = _finite(row.get('quote_age_days'), np.nan)
    if verified and np.isfinite(age) and (0 <= age <= 3):
        return 100.0
    data_age = _finite(row.get('absolute_data_age_days'), 999)
    final_ohlcv = data_age <= cfg.max_absolute_data_age_days and (not _truthy(row.get('current_bar_incomplete', False)))
    return 60.0 if final_ohlcv else 25.0

def _universe_data_coverage(row: Mapping[str, Any]) -> float:
    requested = _finite(row.get('universe_requested_count'), 0)
    prepared = _finite(row.get('universe_prepared_count'), 0)
    explicit = _finite(row.get('universe_coverage_pct'), np.nan)
    if np.isfinite(explicit):
        return min(100.0, max(0.0, explicit))
    if requested > 0:
        return min(100.0, max(0.0, 100.0 * prepared / requested))
    return min(100.0, max(0.0, _finite(row.get('universe_confidence'), 0)))

def _layer_data_coverage(row: Mapping[str, Any], cfg: ScanConfig) -> dict[str, float]:
    technical = _technical_data_coverage(row)
    core_trade_fields = ('ticker', 'quality_score', 'last_price', 'entry', 'stop_loss', 'tp1', 'tp2', 'rr1', 'rr2', 'stop_pct', 'distance_atr', 'adtv20_idr', 'volume_ratio', 'market_regime')
    if all((_is_present_value(row.get(name)) for name in core_trade_fields)):
        technical = max(technical, 100.0)
    risk = _risk_data_coverage(row)
    if _risk_layer_confidence(row, cfg) >= 100.0:
        risk = 100.0
    fundamental = min(100.0, max(0.0, _finite(row.get('fundamental_coverage'), 0.0)))
    validation = _validation_data_coverage(row)
    market_status = _market_status_data_coverage(row, cfg)
    news = _news_data_coverage(row, cfg)
    quote = _quote_data_coverage(row, cfg)
    confidence_fallbacks = {'fundamental': _finite(row.get('fundamental_confidence'), 0.0), 'validation': _finite(row.get('validation_confidence'), 0.0), 'market_status': _finite(row.get('market_status_confidence'), 0.0), 'news': _finite(row.get('news_confidence'), 0.0), 'quote': _finite(row.get('quote_confidence'), 0.0)}
    if fundamental <= 0 and confidence_fallbacks['fundamental'] >= 70:
        fundamental = confidence_fallbacks['fundamental']
    if confidence_fallbacks['validation'] >= 70:
        validation = max(validation, confidence_fallbacks['validation'])
    if confidence_fallbacks['market_status'] >= 70:
        market_status = max(market_status, confidence_fallbacks['market_status'])
    if confidence_fallbacks['news'] >= 70:
        news = max(news, confidence_fallbacks['news'])
    if confidence_fallbacks['quote'] >= 70:
        quote = max(quote, confidence_fallbacks['quote'])
    layers = {'technical': technical, 'risk': risk, 'fundamental': fundamental, 'validation': validation, 'market_status': market_status, 'news': news, 'quote': quote, 'universe': _universe_data_coverage(row)}
    return {name: round(min(100.0, max(0.0, value)), 1) for name, value in layers.items()}

def _google_news_rss_items(ticker: str, lookback_days: int, timeout: int=8) -> tuple[list[dict[str, Any]], bool, str]:
    """Retrieve a lightweight Indonesian Google News RSS query.

    This is used before Yahoo news to avoid repeated crumb/authorization errors.
    A successful empty RSS response is still valid coverage for the requested
    window; it is not interpreted as positive news.
    """
    import email.utils
    import requests
    import xml.etree.ElementTree as ET
    from urllib.parse import quote_plus
    code = normalize_idx_ticker(ticker).replace('.JK', '')
    query = quote_plus(f'"{code}" saham when:{max(1, int(lookback_days))}d')
    url = f'https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id'
    try:
        response = requests.get(url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0 (compatible; IDXSuperScanner/4.2.2)'})
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items: list[dict[str, Any]] = []
        for node in root.findall('.//item')[:30]:
            title = (node.findtext('title') or '').strip()
            link = (node.findtext('link') or '').strip()
            description = (node.findtext('description') or '').strip()
            raw_date = (node.findtext('pubDate') or '').strip()
            published = None
            if raw_date:
                try:
                    parsed = email.utils.parsedate_to_datetime(raw_date)
                    published = pd.Timestamp(parsed).tz_convert('Asia/Jakarta').tz_localize(None)
                except Exception:
                    published = pd.to_datetime(raw_date, utc=True, errors='coerce')
                    if pd.notna(published):
                        published = published.tz_convert('Asia/Jakarta').tz_localize(None)
            items.append({'title': title, 'summary': description, 'link': link, 'pubDate': published.isoformat() if published is not None and pd.notna(published) else None})
        return (items, True, '')
    except Exception as exc:
        return ([], False, f'{type(exc).__name__}: {str(exc)[:120]}')

def fetch_automatic_news_review(tickers: Iterable[str], lookback_days: int=7, max_workers: int=3) -> pd.DataFrame:
    """Review current news with Google News RSS, Yahoo fallback, and IDX disclosure.

    Google RSS is the primary broad-news source because it does not depend on a
    Yahoo crumb. Yahoo is queried only if RSS fails. Official IDX disclosure is
    retained as the event-risk cross-check.
    """
    import yfinance as yf
    names = list(dict.fromkeys(tickers))
    if not names:
        return pd.DataFrame()
    now = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None)
    start = now - pd.Timedelta(days=lookback_days)
    disclosure_text, disclosure_ok, disclosure_error = _fetch_idx_disclosure_page(timeout=8)
    disclosure_mentions = _requested_mentions(disclosure_text, names) if disclosure_ok else set()

    def one(ticker: str) -> dict[str, Any]:
        items, rss_ok, rss_error = _google_news_rss_items(ticker, lookback_days, timeout=8)
        yahoo_ok = False
        yahoo_error = ''
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
                yahoo_error = f'{type(exc).__name__}: {str(exc)[:100]}'
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
            body = f'{title} {summary}'.upper()
            is_negative = any((term in body for term in _NEGATIVE_NEWS_TERMS))
            is_event = any((term in body for term in _EVENT_RISK_TERMS))
            severe = severe or is_negative
            ambiguous = ambiguous or is_event
            if is_negative:
                negative_count += 1
            elif any((term in body for term in ('PROFIT', 'LABA', 'GROWTH', 'PERTUMBUHAN', 'CONTRACT', 'KONTRAK'))):
                positive_count += 1
            if title:
                titles.append(title)
            if url:
                sources.append(url)
            reviewed.append(item)
        complete = query_ok
        provider = 'Google News RSS' if rss_ok else 'Yahoo Finance fallback' if yahoo_ok else 'UNAVAILABLE'
        if disclosure_ok:
            provider += ' + official IDX disclosure'
        errors = ' | '.join((value for value in (rss_error, yahoo_error, disclosure_error) if value))
        return {'ticker': ticker, 'news_reviewed_at': now, 'news_review_status': 'COMPLETE' if complete else 'INCOMPLETE', 'provider_query_ok': bool(complete), 'items_reviewed': len(reviewed), 'coverage_start': start, 'coverage_end': now, 'news_provider': provider, 'verified_catalyst_count': positive_count, 'verified_negative_count': negative_count, 'severe_negative_news': severe, 'ambiguous_material_news': ambiguous, 'catalyst_summary': ' | '.join(titles[:3]), 'news_sources': ' | '.join(sources[:3]), 'news_error': errors, 'idx_disclosure_query_ok': bool(disclosure_ok)}
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(names))) as pool:
        futures = [pool.submit(one, ticker) for ticker in names]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)

def _finalize_execution_integrity_v431(signals: pd.DataFrame, config: ScanConfig | None=None) -> pd.DataFrame:
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
    for column, default in (('critical_blockers', ''), ('evidence_warnings', ''), ('market_status_confidence', 45.0), ('news_confidence', 52.0), ('fundamental_confidence', 50.0), ('validation_confidence', 45.0), ('quote_confidence', 68.0), ('universe_confidence', 48.0), ('portfolio_selected', False), ('independent_price_verified', False), ('independent_price_state', 'MISSING_INDEPENDENT'), ('independent_source_family', '')):
        if column not in out:
            out[column] = default
    for idx, row in out.iterrows():
        technical_ready = bool(_truthy(row.get('technical_setup_ready', False)) or _truthy(row.get('technical_execution_candidate', False)) or _safe_text(row.get('status')) == 'EXECUTION_READY')
        technical_conf = min(100.0, max(0.0, _finite(row.get('quality_score'), 0)))
        risk_conf = _risk_layer_confidence(row, cfg)
        market_conf = _finite(row.get('market_status_confidence'), 45)
        news_conf = _finite(row.get('news_confidence'), 52)
        fundamental_conf = _finite(row.get('fundamental_confidence'), 50)
        fundamental_coverage = _finite(row.get('fundamental_coverage'), 0)
        validation_conf = _finite(row.get('validation_confidence'), 45)
        quote_conf = _finite(row.get('quote_confidence'), 68)
        universe_conf = _finite(row.get('universe_confidence'), 48)
        if market_conf < 60 and quote_conf >= 90 and (_finite(row.get('absolute_data_age_days'), 999) <= cfg.max_absolute_data_age_days):
            market_conf = 70.0
            out.at[idx, 'market_status_confidence'] = market_conf
            _append_pipe(out, idx, 'evidence_warnings', 'Status IDX memakai provisional quote/OHLCV fallback')
        confidence_weights = {'technical': (technical_conf, 0.35), 'risk': (risk_conf, 0.2), 'market_status': (market_conf, 0.1), 'news': (news_conf, 0.08), 'fundamental': (fundamental_conf, 0.1), 'validation': (validation_conf, 0.07), 'quote': (quote_conf, 0.05), 'universe': (universe_conf, 0.05)}
        confidence = round(sum((value * weight for value, weight in confidence_weights.values())), 1)
        coverages = _layer_data_coverage(row, cfg)
        completeness = round(sum((coverages[name] * weight for name, weight in _DATA_LAYER_WEIGHTS.items())), 1)
        for name, value in coverages.items():
            out.at[idx, f'{name}_data_coverage'] = value
        out.at[idx, 'data_completeness_score'] = completeness
        out.at[idx, 'data_completeness_tier'] = 'HIGH' if completeness >= 90 else 'SUFFICIENT' if completeness >= cfg.min_data_completeness else 'PARTIAL' if completeness >= 60 else 'LOW'
        missing_layers = [name for name, value in coverages.items() if value < 60]
        direct_fundamental_ok = bool(not cfg.real_money_mode or fundamental_coverage >= cfg.min_direct_fundamental_coverage)
        daily_source_tier = _safe_text(row.get('ohlcv_source_tier')).upper()
        direct_daily_source_ok = bool(not cfg.real_money_mode or daily_source_tier not in {'CACHE_FALLBACK', 'UNAVAILABLE'})
        independent_verified = _truthy(row.get('independent_price_verified', False))
        independent_price_ok = bool(not cfg.real_money_mode or not cfg.require_independent_price_verification or independent_verified)
        if not direct_fundamental_ok and 'fundamental' not in missing_layers:
            missing_layers.append('fundamental')
        if not direct_daily_source_ok and 'live_ohlcv' not in missing_layers:
            missing_layers.append('live_ohlcv')
        if not independent_price_ok and 'independent_price' not in missing_layers:
            missing_layers.append('independent_price')
        out.at[idx, 'data_missing_layers'] = ' | '.join(missing_layers)
        if completeness < cfg.min_data_completeness:
            _append_pipe(out, idx, 'evidence_warnings', f'Data completeness {completeness:.1f}% di bawah minimum {cfg.min_data_completeness:.0f}%')
        if not direct_fundamental_ok:
            _append_pipe(out, idx, 'evidence_warnings', f'Fundamental coverage {fundamental_coverage:.1f}% di bawah minimum direct-order {cfg.min_direct_fundamental_coverage:.0f}%')
        if not direct_daily_source_ok:
            _append_pipe(out, idx, 'evidence_warnings', 'OHLCV daily bukan hasil live; cache hanya boleh dipakai untuk riset/watchlist')
        if not independent_price_ok:
            _append_pipe(out, idx, 'evidence_warnings', 'Direct order menunggu verifikasi harga dari keluarga provider independen')
        critical = _safe_text(row.get('critical_blockers'))
        if _truthy(row.get('validation_critical_blocker', False)):
            critical = critical or 'OOS edge negatif'
        if _truthy(row.get('fundamental_critical_blocker', False)):
            critical = critical or 'Fundamental distress'
        if _truthy(row.get('market_status_critical_blocker', False)):
            critical = critical or 'Status IDX negatif'
        if _truthy(row.get('news_critical_blocker', False)):
            critical = critical or 'Berita material negatif'
        if _truthy(row.get('quote_critical_blocker', False)):
            critical = critical or 'Quote/candle conflict'
        portfolio_ok = _truthy(row.get('portfolio_selected', False))
        if technical_ready and (not portfolio_ok):
            _append_pipe(out, idx, 'portfolio_blockers', 'Belum dipilih oleh budget portofolio')
        base_direct_gates = bool(technical_ready and (not critical) and (risk_conf == 100.0) and portfolio_ok and direct_fundamental_ok and direct_daily_source_ok)
        projected_market_conf = 70.0 if market_conf < 60 else market_conf
        projected_quote_conf = max(100.0, quote_conf)
        projected_completeness = round(min(100.0, completeness + _DATA_LAYER_WEIGHTS['quote'] * (100.0 - coverages['quote'])), 1)
        projected_confidence = round(confidence + 0.1 * (projected_market_conf - market_conf) + 0.05 * (projected_quote_conf - quote_conf), 1)
        ready_except_independent = bool(base_direct_gates and projected_completeness >= cfg.min_data_completeness and (projected_confidence >= cfg.min_execution_confidence))
        direct = bool(base_direct_gates and independent_price_ok and (completeness >= cfg.min_data_completeness) and (confidence >= cfg.min_execution_confidence))
        if direct:
            final_status = 'EXECUTION_READY'
        elif critical:
            final_status = 'REJECT' if _safe_text(row.get('status')) == 'REJECT' else 'BLOCKED_CONTEXT'
        elif ready_except_independent and (not independent_price_ok):
            final_status = 'READY_FOR_PRICE_VERIFY'
        elif technical_ready and (confidence >= cfg.min_pending_confidence or completeness < cfg.min_data_completeness):
            final_status = 'PENDING_DATA'
        else:
            current = _safe_text(row.get('status'))
            final_status = current if current in {'WATCHLIST_ENTRY', 'REJECT'} else 'WATCHLIST_ENTRY'
        out.at[idx, 'status'] = final_status
        out.at[idx, 'execution_integrity_score'] = confidence
        out.at[idx, 'execution_confidence_score'] = confidence
        out.at[idx, 'projected_completeness_with_independent_price'] = projected_completeness
        out.at[idx, 'projected_confidence_with_independent_price'] = projected_confidence
        source_family = _safe_text(row.get('independent_source_family'))
        out.at[idx, 'automated_provider_families'] = 4 if source_family == 'TWELVE_DATA' else 3
        out.at[idx, 'independent_price_provider_families'] = 2 if independent_verified else 1
        out.at[idx, 'price_cross_validation_state'] = 'VERIFIED_INDEPENDENT' if independent_verified else _safe_text(row.get('independent_price_state')) or 'MISSING_INDEPENDENT'
        out.at[idx, 'independent_price_required'] = bool(cfg.real_money_mode and cfg.require_independent_price_verification)
        out.at[idx, 'critical_gate_pass'] = not bool(critical)
        out.at[idx, 'evidence_state'] = 'RESOLVED' if direct else 'ADVERSE' if critical else 'SUFFICIENT_DATA' if completeness >= cfg.min_data_completeness else 'PARTIAL'
        out.at[idx, 'order_instruction'] = 'BUY_LIMIT' if direct else 'DO_NOT_BUY'
        out.at[idx, 'stockbit_order_price'] = row.get('entry') if direct else np.nan
        out.at[idx, 'stockbit_order_lots'] = int(_finite(row.get('suggested_lots'), 0)) if direct else 0
        gate_checks = {'technical': technical_ready, 'risk': risk_conf == 100.0, 'context': not bool(critical), 'portfolio': portfolio_ok, 'fundamental': direct_fundamental_ok, 'live_daily': direct_daily_source_ok, 'independent_price': independent_price_ok, 'completeness': completeness >= cfg.min_data_completeness or (not independent_price_ok and projected_completeness >= cfg.min_data_completeness), 'confidence': confidence >= cfg.min_execution_confidence or (not independent_price_ok and projected_confidence >= cfg.min_execution_confidence)}
        failure_labels = {'technical': 'TECHNICAL_TRIGGER_OR_DISTANCE', 'risk': 'RISK_LEVELS_OR_SIZING', 'context': 'CRITICAL_CONTEXT', 'portfolio': 'PORTFOLIO_BUDGET', 'fundamental': 'FUNDAMENTAL_COVERAGE', 'live_daily': 'DAILY_SOURCE_NOT_LIVE', 'independent_price': 'INDEPENDENT_PRICE_REQUIRED', 'completeness': 'DATA_COMPLETENESS', 'confidence': 'EXECUTION_CONFIDENCE'}
        failures = [failure_labels[name] for name, passed in gate_checks.items() if not passed]
        out.at[idx, 'execution_gate_failures'] = ' | '.join(failures)
        out.at[idx, 'primary_execution_blocker'] = failures[0] if failures else 'NONE'
        out.at[idx, 'execution_readiness_pct'] = round(100.0 * sum(gate_checks.values()) / len(gate_checks), 1)
        out.at[idx, 'automation_decision'] = 'DIRECT_EXECUTION_ELIGIBLE' if direct else 'VERIFY_INDEPENDENT_PRICE' if final_status == 'READY_FOR_PRICE_VERIFY' else 'BLOCKED' if critical else 'RETRY_OR_WATCH'
    out['status_rank'] = out['status'].map(STATUS_ORDER).fillna(99)
    return out

def _normalized_column_name(value: object) -> str:
    return re.sub('[^a-z0-9]', '', str(value).strip().lower())

def _external_number(value: object) -> float:
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return number if np.isfinite(number) else np.nan
    text = str(value).strip().replace('Rp', '').replace('IDR', '').replace(' ', '')
    if not text:
        return np.nan
    if ',' in text and '.' in text:
        if text.rfind(',') > text.rfind('.'):
            text = text.replace('.', '').replace(',', '.')
        else:
            text = text.replace(',', '')
    elif ',' in text:
        tail = text.rsplit(',', 1)[-1]
        text = text.replace(',', '.' if len(tail) <= 2 else '')
    try:
        number = float(text)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan

def _independent_source_family(source: object) -> str:
    text = _safe_text(source).upper()
    if 'YAHOO' in text or 'YFINANCE' in text:
        return 'YAHOO'
    if 'GOOGLE' in text:
        return 'GOOGLE_FINANCE'
    if 'TWELVE' in text:
        return 'TWELVE_DATA'
    if 'IDX' in text or 'BURSA EFEK' in text:
        return 'IDX_OFFICIAL'
    if 'STOCKBIT' in text:
        return 'STOCKBIT'
    return 'MANUAL_EXTERNAL'

def parse_independent_price_file(source: bytes | BinaryIO | pd.DataFrame, filename: str | None=None, default_source: str='MANUAL_EXTERNAL_UPLOAD') -> pd.DataFrame:
    """Normalize an official IDX EOD export or a manual Stockbit quote file.

    Required semantic fields are ticker, date/as-of, and close/last price. The
    parser accepts common Indonesian and English column labels. A source label
    is retained so manual data can never be misrepresented as an automated feed.
    """
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        payload = BytesIO(source) if isinstance(source, bytes) else source
        if hasattr(payload, 'seek'):
            payload.seek(0)
        suffix = Path(filename or getattr(source, 'name', '')).suffix.lower()
        if suffix in {'.xlsx', '.xlsm'}:
            frame = pd.read_excel(payload)
        else:
            try:
                frame = pd.read_csv(payload, sep=None, engine='python')
            except UnicodeDecodeError:
                if hasattr(payload, 'seek'):
                    payload.seek(0)
                frame = pd.read_csv(payload, sep=None, engine='python', encoding='latin-1')
    if frame.empty:
        return pd.DataFrame(columns=['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'independent_source', 'independent_source_family'])
    lookup = {_normalized_column_name(column): column for column in frame.columns}

    def pick(aliases: Iterable[str], required: bool=False) -> object | None:
        selected = next((lookup[name] for name in aliases if name in lookup), None)
        if required and selected is None:
            raise ValueError('Kolom verifikasi harga tidak lengkap: ticker, date/asof, dan close/last_price wajib ada')
        return selected
    ticker_col = pick(('ticker', 'tickers', 'symbol', 'kode', 'kodesaham', 'stockcode', 'code'), True)
    date_col = pick(('date', 'asof', 'timestamp', 'datetime', 'tanggal', 'tanggalperdagangan', 'tradingdate', 'time'), True)
    close_col = pick(('close', 'closingprice', 'lastprice', 'last', 'penutupan', 'harga', 'price', 'lasttrade'), True)
    open_col = pick(('open', 'openprice', 'pembukaan'))
    high_col = pick(('high', 'highprice', 'tertinggi'))
    low_col = pick(('low', 'lowprice', 'terendah'))
    volume_col = pick(('volume', 'vol', 'totalvolume', 'jumlahsaham'))
    source_col = pick(('source', 'provider', 'sumber', 'datasource'))
    out = pd.DataFrame()
    out['ticker'] = frame[ticker_col].map(normalize_idx_ticker)

    def parse_external_date(value: object) -> pd.Timestamp | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        text = str(value).strip()
        iso_first = bool(re.match('^\\d{4}[-/]\\d{1,2}[-/]\\d{1,2}', text))
        parsed = pd.to_datetime(value, errors='coerce', dayfirst=not iso_first)
        return parsed if pd.notna(parsed) else None
    out['Date'] = frame[date_col].map(parse_external_date)
    for output, column in (('Open', open_col), ('High', high_col), ('Low', low_col), ('Close', close_col), ('Volume', volume_col)):
        out[output] = frame[column].map(_external_number) if column is not None else np.nan
    out['independent_source'] = frame[source_col].map(_safe_text).replace('', default_source) if source_col is not None else default_source
    out['independent_source_family'] = out['independent_source'].map(_independent_source_family)
    out = out.dropna(subset=['ticker', 'Date', 'Close'])
    out = out[out['Close'].gt(0)].copy()
    if out.empty:
        return out
    for column in ('Open', 'High', 'Low'):
        out[column] = out[column].where(out[column].gt(0))
    out['Volume'] = out['Volume'].where(out['Volume'].ge(0))
    out = out.sort_values(['ticker', 'independent_source_family', 'Date'])
    return out.drop_duplicates(['ticker', 'independent_source_family', 'Date'], keep='last').reset_index(drop=True)

def fetch_twelve_data_eod(tickers: Iterable[str], api_key: str, outputsize: int=30, max_tickers: int=12, timeout: int=15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch bounded XIDX EOD validation history from Twelve Data.

    Twelve Data lists XIDX coverage as EOD and currently requires an eligible
    paid plan. This adapter is optional and only queries a candidate shortlist.
    API keys are passed as request parameters but are never written to output.
    """
    names = list(dict.fromkeys(tickers))[:max(0, int(max_tickers))]
    columns = ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'independent_source', 'independent_source_family']
    if not names or not _safe_text(api_key):
        return (pd.DataFrame(columns=columns), pd.DataFrame(columns=['ticker', 'status', 'bars', 'error']))
    import requests
    histories: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for ticker in names:
        symbol = ticker[:-3] if ticker.upper().endswith('.JK') else ticker
        try:
            response = requests.get('https://api.twelvedata.com/time_series', params={'symbol': symbol, 'exchange': 'XIDX', 'interval': '1day', 'outputsize': max(5, min(100, int(outputsize))), 'timezone': 'Asia/Jakarta', 'format': 'JSON', 'apikey': api_key}, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            values = payload.get('values', []) if isinstance(payload, dict) else []
            if not values:
                message = _safe_text(payload.get('message')) if isinstance(payload, dict) else 'Respons kosong'
                reports.append({'ticker': ticker, 'status': 'FAILED', 'bars': 0, 'error': message or 'Respons kosong'})
                continue
            frame = pd.DataFrame(values).rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            frame['ticker'] = ticker
            frame['independent_source'] = 'TWELVE_DATA_XIDX_EOD'
            frame['independent_source_family'] = 'TWELVE_DATA'
            frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce')
            for column in ('Open', 'High', 'Low', 'Close', 'Volume'):
                frame[column] = pd.to_numeric(frame.get(column), errors='coerce')
            frame = frame[columns].dropna(subset=['Date', 'Close']).sort_values('Date')
            histories.append(frame)
            reports.append({'ticker': ticker, 'status': 'OK', 'bars': len(frame), 'error': ''})
        except Exception as exc:
            safe_error = str(exc).replace(str(api_key), '***')
            reports.append({'ticker': ticker, 'status': 'FAILED', 'bars': 0, 'error': f'{type(exc).__name__}: {safe_error[:140]}'})
    data = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame(columns=columns)
    return (data, pd.DataFrame(reports))

def _primary_source_family_from_tier(value: object) -> str:
    tier = str(value or '').upper()
    if 'IDX' in tier:
        return 'IDX_OFFICIAL'
    if 'ITICK' in tier:
        return 'ITICK'
    if 'YAHOO' in tier:
        return 'YAHOO'
    return 'UNKNOWN'

def build_independent_price_validation(primary_histories: Mapping[str, pd.DataFrame], independent_data: pd.DataFrame | None, config: ScanConfig | None=None, now: Any | None=None, primary_source_tiers: Mapping[str, str] | None=None) -> pd.DataFrame:
    """Compare latest price and, when available, recent return paths."""
    cfg = config or ScanConfig()
    result_columns = ['ticker', 'independent_price_verified', 'independent_price_state', 'independent_source', 'independent_source_family', 'independent_asof', 'independent_last_price', 'independent_price_age_days', 'independent_date_gap_days', 'independent_price_divergence_pct', 'independent_overlap_bars', 'independent_close_mape_pct', 'independent_return_correlation', 'independent_price_confidence']
    if independent_data is None or independent_data.empty:
        return pd.DataFrame(columns=result_columns)
    required = {'ticker', 'Date', 'Close', 'independent_source', 'independent_source_family'}
    if not required.issubset(independent_data.columns):
        raise ValueError('Schema data harga independen tidak valid')
    now_jkt = _jakarta_timestamp(now).tz_localize(None)
    rows: list[dict[str, Any]] = []
    for ticker, ticker_data in independent_data.groupby('ticker', sort=False):
        primary = _clean_ohlcv(primary_histories.get(str(ticker), pd.DataFrame()), strict=True)
        if primary.empty:
            continue
        primary_close = primary[['Close']].copy()
        primary_close.index = pd.to_datetime(primary_close.index).normalize()
        primary_close = primary_close[~primary_close.index.duplicated(keep='last')]
        primary_last_date = pd.Timestamp(primary_close.index[-1]).normalize()
        primary_last = float(primary_close['Close'].iloc[-1])
        primary_family = _primary_source_family_from_tier((primary_source_tiers or {}).get(str(ticker), ''))
        candidates: list[dict[str, Any]] = []
        for family, group in ticker_data.groupby('independent_source_family', sort=False):
            group = group.copy().dropna(subset=['Date', 'Close']).sort_values('Date')
            if group.empty:
                continue
            group['_day'] = pd.to_datetime(group['Date']).dt.normalize()
            group = group.drop_duplicates('_day', keep='last')
            latest = group.iloc[-1]
            external_date = pd.Timestamp(latest['Date'])
            if external_date.tzinfo is not None:
                external_date = external_date.tz_convert('Asia/Jakarta').tz_localize(None)
            external_last = float(latest['Close'])
            age_days = max(0, int((now_jkt.normalize() - external_date.normalize()).days))
            date_gap = abs(int((primary_last_date - external_date.normalize()).days))
            divergence = abs(external_last / primary_last - 1.0) if primary_last > 0 else np.nan
            tolerance = max(cfg.max_secondary_price_divergence_pct, 2.0 * idx_tick_size(primary_last) / primary_last)
            external_close = group.set_index('_day')[['Close']].rename(columns={'Close': 'external'})
            overlap = primary_close.rename(columns={'Close': 'primary'}).join(external_close, how='inner').dropna()
            overlap_bars = len(overlap)
            mape = float((overlap['external'] / overlap['primary'] - 1.0).abs().median()) if overlap_bars else np.nan
            returns = overlap.pct_change().dropna()
            correlation = float(returns['primary'].corr(returns['external'])) if len(returns) >= 3 and returns['primary'].std() > 0 and (returns['external'].std() > 0) else np.nan
            needs_history = str(family) in {'TWELVE_DATA', 'ITICK'}
            same_provider = bool(str(family) == 'YAHOO' or (primary_family != 'UNKNOWN' and str(family) == primary_family))
            history_ok = bool(not needs_history or (overlap_bars >= cfg.min_secondary_overlap_bars and (np.isfinite(correlation) and correlation >= cfg.min_secondary_return_correlation or (np.isfinite(mape) and mape <= tolerance))))
            fresh = (
                age_days <= cfg.max_independent_price_age_days
                and date_gap <= cfg.max_independent_date_gap_days
            )
            aligned = np.isfinite(divergence) and divergence <= tolerance
            verified = bool(fresh and aligned and history_ok and (not same_provider))
            if same_provider:
                state, confidence = ('SAME_PROVIDER_FAMILY', 0.0)
            elif verified:
                state, confidence = ('VERIFIED_INDEPENDENT', 100.0)
            elif not fresh:
                state, confidence = ('STALE_INDEPENDENT', 20.0)
            elif np.isfinite(divergence) and divergence > max(0.02, 2.0 * tolerance):
                state, confidence = ('PRICE_CONFLICT', 0.0)
            elif not history_ok:
                state, confidence = ('HISTORY_MISMATCH', 10.0)
            else:
                state, confidence = ('PRICE_NOT_ALIGNED', 35.0)
            priority = {'IDX_OFFICIAL': 6, 'STOCKBIT': 5, 'GOOGLE_FINANCE': 4, 'ITICK': 3, 'TWELVE_DATA': 2, 'MANUAL_EXTERNAL': 1}.get(str(family), 0)
            candidates.append({'ticker': str(ticker), 'independent_price_verified': verified, 'independent_price_state': state, 'independent_source': _safe_text(latest.get('independent_source')), 'independent_source_family': str(family), 'independent_asof': external_date, 'independent_last_price': external_last, 'independent_price_age_days': age_days, 'independent_date_gap_days': date_gap, 'independent_price_divergence_pct': divergence, 'independent_overlap_bars': overlap_bars, 'independent_close_mape_pct': mape, 'independent_return_correlation': correlation, 'independent_price_confidence': confidence, '_priority': priority})
        if candidates:
            chosen = sorted(candidates, key=lambda item: (item['independent_price_verified'], item['_priority'], item['independent_price_confidence']), reverse=True)[0]
            chosen.pop('_priority', None)
            rows.append(chosen)
    return pd.DataFrame(rows, columns=result_columns)

def apply_independent_price_gate(signals: pd.DataFrame, validation: pd.DataFrame | None, config: ScanConfig | None=None) -> pd.DataFrame:
    """Attach independent-price evidence without converting absence into bad data."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if validation is None or validation.empty:
        validation = pd.DataFrame({'ticker': out['ticker'].drop_duplicates()})
        validation['independent_price_verified'] = False
        validation['independent_price_state'] = 'MISSING_INDEPENDENT'
        validation['independent_source'] = ''
        validation['independent_source_family'] = ''
        validation['independent_price_confidence'] = 0.0
    authoritative_columns = [column for column in validation.columns if column != 'ticker']
    if authoritative_columns:
        out = out.drop(columns=[column for column in authoritative_columns if column in out.columns])
    out = out.merge(validation, on='ticker', how='left')
    defaults = {'independent_price_verified': False, 'independent_price_state': 'MISSING_INDEPENDENT', 'independent_source': '', 'independent_source_family': '', 'independent_price_confidence': 0.0}
    for column, default in defaults.items():
        if column not in out:
            out[column] = default
        else:
            out[column] = out[column].fillna(default)
    if 'quote_critical_blocker' not in out:
        out['quote_critical_blocker'] = False
    for idx, row in out.iterrows():
        verified = _truthy(row.get('independent_price_verified', False))
        state = _safe_text(row.get('independent_price_state')) or 'MISSING_INDEPENDENT'
        if verified:
            out.at[idx, 'independent_price_verified'] = True
            out.at[idx, 'quote_confidence'] = max(100.0, _finite(row.get('quote_confidence'), 0.0))
        elif state in {'PRICE_CONFLICT', 'HISTORY_MISMATCH'}:
            out.at[idx, 'quote_critical_blocker'] = True
            _set_context_block(out, idx, f'Validasi harga independen gagal: {state}')
        else:
            _append_pipe(out, idx, 'evidence_warnings', f'Harga independen belum terverifikasi: {state}')
    out['status_rank'] = out['status'].map(STATUS_ORDER).fillna(99)
    return out
_AUTOMATIC_INDEPENDENT_COLUMNS = ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'independent_source', 'independent_source_family']
_AUTOMATIC_PROVIDER_REPORT_COLUMNS = ['provider', 'scope', 'status', 'rows', 'asof', 'error']

def _empty_automatic_independent_data() -> pd.DataFrame:
    return pd.DataFrame(columns=_AUTOMATIC_INDEPENDENT_COLUMNS)

def _automatic_provider_report(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_AUTOMATIC_PROVIDER_REPORT_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in _AUTOMATIC_PROVIDER_REPORT_COLUMNS:
        if column not in frame:
            frame[column] = '' if column not in {'rows'} else 0
    return frame[_AUTOMATIC_PROVIDER_REPORT_COLUMNS]

def _load_independent_price_cache(tickers: Iterable[str]) -> pd.DataFrame:
    """Load provider evidence without changing its source-family identity."""
    names = {normalize_idx_ticker(ticker) for ticker in tickers}
    names.discard(None)
    cached = _load_cache('independent_prices_v1')
    if cached.empty or not set(_AUTOMATIC_INDEPENDENT_COLUMNS).issubset(cached.columns):
        return _empty_automatic_independent_data()
    cached = cached[_AUTOMATIC_INDEPENDENT_COLUMNS].copy()
    cached['ticker'] = cached['ticker'].map(normalize_idx_ticker)
    cached['Date'] = pd.to_datetime(cached['Date'], errors='coerce')
    for column in ('Open', 'High', 'Low', 'Close', 'Volume'):
        cached[column] = pd.to_numeric(cached[column], errors='coerce')
    cached = cached[
        cached['ticker'].isin(names)
        & cached['Date'].notna()
        & cached['Close'].gt(0)
        & cached['independent_source_family'].fillna('').astype(str).ne('')
    ]
    return cached.sort_values(['ticker', 'independent_source_family', 'Date']).reset_index(drop=True)

def _write_independent_price_cache(frame: pd.DataFrame) -> None:
    """Atomically retain recent successful evidence across transient outages."""
    if frame is None or frame.empty:
        return
    current = frame.copy()
    if not set(_AUTOMATIC_INDEPENDENT_COLUMNS).issubset(current.columns):
        return
    existing = _load_cache('independent_prices_v1')
    if not existing.empty and set(_AUTOMATIC_INDEPENDENT_COLUMNS).issubset(existing.columns):
        rows = existing[_AUTOMATIC_INDEPENDENT_COLUMNS].to_dict('records')
        rows.extend(current[_AUTOMATIC_INDEPENDENT_COLUMNS].to_dict('records'))
        current = pd.DataFrame(rows, columns=_AUTOMATIC_INDEPENDENT_COLUMNS)
    else:
        current = current[_AUTOMATIC_INDEPENDENT_COLUMNS]
    current['ticker'] = current['ticker'].map(normalize_idx_ticker)
    current['Date'] = pd.to_datetime(current['Date'], errors='coerce')
    current['Close'] = pd.to_numeric(current['Close'], errors='coerce')
    current = current.dropna(subset=['ticker', 'Date', 'Close'])
    current = current[current['Close'].gt(0)]
    if current.empty:
        return
    newest = current['Date'].max()
    cutoff = pd.Timestamp(newest).normalize() - pd.Timedelta(days=45)
    current = current[current['Date'].ge(cutoff)]
    current = current.sort_values(['ticker', 'independent_source_family', 'Date'])
    current = current.drop_duplicates(['ticker', 'independent_source_family', 'Date'], keep='last')
    _write_cache('independent_prices_v1', current.reset_index(drop=True))

def fetch_itick_independent_eod(
    tickers: Iterable[str],
    api_token: str,
    max_tickers: int=4,
    timeout: int=12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the optional iTick free kline as a bounded independent family."""
    names = list(dict.fromkeys(normalize_idx_ticker(ticker) for ticker in tickers))
    names = [ticker for ticker in names if ticker][:max(0, int(max_tickers))]
    if not names or not _safe_text(api_token):
        return (_empty_automatic_independent_data(), _automatic_provider_report([]))
    histories, raw_report = fetch_itick_ohlcv(
        names,
        api_token=api_token,
        period='2y',
        interval='1d',
        timeout=timeout,
        max_tickers=len(names),
    )
    frames: list[pd.DataFrame] = []
    for ticker, history in histories.items():
        clean = _clean_ohlcv(history, strict=True).tail(30)
        if clean.empty:
            continue
        converted = clean.reset_index().rename(columns={clean.index.name or 'index': 'Date'})
        converted['ticker'] = ticker
        converted['independent_source'] = 'ITICK_FREE_KLINE_EOD'
        converted['independent_source_family'] = 'ITICK'
        frames.append(converted[_AUTOMATIC_INDEPENDENT_COLUMNS])
    data = pd.concat(frames, ignore_index=True) if frames else _empty_automatic_independent_data()
    if raw_report is None or raw_report.empty:
        report = _automatic_provider_report([])
    else:
        normalized = pd.DataFrame({
            'provider': 'ITICK_FREE',
            'scope': raw_report.get('ticker', ''),
            'status': raw_report.get('status', 'FAILED'),
            'rows': pd.to_numeric(raw_report.get('bars', 0), errors='coerce').fillna(0).astype(int),
            'asof': pd.NaT,
            'error': raw_report.get('error', ''),
        })
        report = _automatic_provider_report(normalized.to_dict('records'))
    return (data, report)

def _candidate_idx_summary_dates(reference_date: Any | None, lookback_days: int) -> list[pd.Timestamp]:
    anchor = pd.Timestamp(reference_date) if reference_date is not None else pd.Timestamp.now(tz='Asia/Jakarta')
    if anchor.tzinfo is not None:
        anchor = anchor.tz_convert('Asia/Jakarta').tz_localize(None)
    anchor = anchor.normalize()
    dates: list[pd.Timestamp] = []
    # `reference_date` is already the last completed EOD date selected by the
    # scanner.  Starting at offset=1 skipped that exact session (for example,
    # Thursday EOD during Friday pre-market), which made a valid official IDX
    # price appear missing.  Query the anchor first, then walk backwards.
    for offset in range(max(1, int(lookback_days))):
        candidate = anchor - pd.Timedelta(int(offset), unit='D')
        if candidate.weekday() < 5:
            dates.append(candidate)
    return dates

def _parse_idx_api_date(value: object, fallback: pd.Timestamp) -> pd.Timestamp:
    text = _safe_text(value)
    epoch_match = re.search('/Date\\((-?\\d{10,13})', text)
    if epoch_match:
        raw = int(epoch_match.group(1))
        unit = 'ms' if abs(raw) >= 10 ** 12 else 's'
        parsed = pd.to_datetime(raw, unit=unit, utc=True, errors='coerce')
        if pd.notna(parsed):
            return pd.Timestamp(parsed).tz_convert('Asia/Jakarta').tz_localize(None)
    parsed = pd.to_datetime(value, errors='coerce')
    if pd.isna(parsed):
        return fallback
    stamp = pd.Timestamp(parsed)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert('Asia/Jakarta').tz_localize(None)
    return stamp

def _normalize_idx_stock_summary_payload(payload: object, requested: set[str], fallback_date: pd.Timestamp) -> pd.DataFrame:
    raw_rows = payload.get('data', []) if isinstance(payload, Mapping) else []
    if not isinstance(raw_rows, list):
        return _empty_automatic_independent_data()
    rows: list[dict[str, Any]] = []
    for item in raw_rows:
        if not isinstance(item, Mapping):
            continue
        ticker = normalize_idx_ticker(item.get('StockCode') or item.get('Code') or item.get('SecurityCode'))
        if not ticker or ticker not in requested:
            continue
        close = _external_number(item.get('Close', item.get('ClosingPrice')))
        if not np.isfinite(close) or close <= 0:
            continue
        rows.append({'ticker': ticker, 'Date': _parse_idx_api_date(item.get('Date'), fallback_date), 'Open': _external_number(item.get('OpenPrice', item.get('OpeningPrice'))), 'High': _external_number(item.get('High', item.get('HighestPrice'))), 'Low': _external_number(item.get('Low', item.get('LowestPrice'))), 'Close': close, 'Volume': _external_number(item.get('Volume', item.get('TradedVolume'))), 'independent_source': 'IDX_OFFICIAL_STOCK_SUMMARY_API', 'independent_source_family': 'IDX_OFFICIAL'})
    return pd.DataFrame(rows, columns=_AUTOMATIC_INDEPENDENT_COLUMNS)

def fetch_idx_official_eod_quotes(tickers: Iterable[str], reference_date: Any | None=None, lookback_days: int=7, timeout: int=10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch official IDX Stock Summary EOD data without a user upload.

    The public page is opened first so the request session receives the same
    cookies as a normal browser visit. Provider errors remain explicit and are
    handed to the automatic fallback instead of being treated as valid data.
    """
    names = list(dict.fromkeys((normalize_idx_ticker(ticker) for ticker in tickers)))
    names = [ticker for ticker in names if ticker]
    if not names:
        return (_empty_automatic_independent_data(), _automatic_provider_report([]))
    import requests
    page_url = 'https://www.idx.co.id/id/data-pasar/ringkasan-perdagangan/ringkasan-saham'
    endpoint = 'https://www.idx.co.id/primary/TradingSummary/GetStockSummary'
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36', 'Accept': 'application/json, text/plain, */*', 'Accept-Language': 'id-ID,id;q=0.9,en;q=0.7', 'Referer': page_url, 'X-Requested-With': 'XMLHttpRequest'}
    session = requests.Session()
    seed_error = ''
    try:
        seed = session.get(page_url, headers=headers, timeout=timeout, allow_redirects=True)
        if getattr(seed, 'status_code', 200) >= 400:
            seed_error = f'seed HTTP {seed.status_code}'
    except Exception as exc:
        seed_error = f'seed {type(exc).__name__}: {str(exc)[:100]}'
    requested = set(names)
    resolved: set[str] = set()
    frames: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for trade_date in _candidate_idx_summary_dates(reference_date, lookback_days):
        missing = requested - resolved
        if not missing:
            break
        date_text = trade_date.strftime('%Y%m%d')
        try:
            response = session.get(endpoint, params={'date': date_text}, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            final_url = _safe_text(getattr(response, 'url', endpoint)) or endpoint
            if not _is_exact_official_idx_url(final_url):
                raise RuntimeError('redirect keluar domain resmi IDX')
            frame = _normalize_idx_stock_summary_payload(response.json(), missing, trade_date)
            found = set(frame['ticker']) if not frame.empty else set()
            if not frame.empty:
                frames.append(frame)
                resolved.update(found)
            reports.append({'provider': 'IDX_OFFICIAL_STOCK_SUMMARY', 'scope': date_text, 'status': 'OK' if found else 'EMPTY', 'rows': len(found), 'asof': trade_date, 'error': '' if found else seed_error or 'ticker kandidat tidak ditemukan'})
        except Exception as exc:
            reports.append({'provider': 'IDX_OFFICIAL_STOCK_SUMMARY', 'scope': date_text, 'status': 'FAILED', 'rows': 0, 'asof': trade_date, 'error': f'{type(exc).__name__}: {str(exc)[:140]}'})
    data = pd.concat(frames, ignore_index=True) if frames else _empty_automatic_independent_data()
    if not data.empty:
        data = data.sort_values(['ticker', 'Date']).drop_duplicates('ticker', keep='last').reset_index(drop=True)
    return (data, _automatic_provider_report(reports))

def _google_finance_quote_from_html(html_text: str, ticker: str) -> dict[str, Any]:
    """Extract the stable data-* quote attributes from a Google Finance page."""
    import html as html_module
    code = normalize_idx_ticker(ticker).replace('.JK', '')
    tags = re.findall('<[^>]*\\bdata-last-price\\s*=\\s*[\'\\"][^\'\\"]+[\'\\"][^>]*>', html_text or '', flags=re.I)
    candidates: list[tuple[int, dict[str, str]]] = []
    for tag in tags:
        attrs = {key.lower(): html_module.unescape(value) for key, _, value in re.findall('([A-Za-z_:][A-Za-z0-9_:.-]*)\\s*=\\s*([\'\\"])(.*?)\\2', tag, flags=re.S)}
        exchange = _safe_text(attrs.get('data-exchange')).upper()
        symbol = _safe_text(attrs.get('data-symbol') or attrs.get('data-ticker')).upper()
        currency = _safe_text(attrs.get('data-currency-code')).upper()
        if exchange and exchange not in {'IDX', 'JKT', 'JAKARTA'}:
            continue
        if symbol and symbol != code:
            continue
        if currency and currency != 'IDR':
            continue
        score = 4 * int(symbol == code) + 2 * int(exchange in {'IDX', 'JKT', 'JAKARTA'}) + int(currency == 'IDR')
        candidates.append((score, attrs))
    if not candidates:
        raise ValueError('atribut quote Google Finance tidak ditemukan')
    attrs = sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]
    price = _external_number(attrs.get('data-last-price'))
    raw_timestamp = _external_number(attrs.get('data-last-normal-market-timestamp'))
    currency = _safe_text(attrs.get('data-currency-code')).upper()
    exchange = _safe_text(attrs.get('data-exchange')).upper()
    if not np.isfinite(price) or price <= 0:
        raise ValueError('harga Google Finance tidak valid')
    if not np.isfinite(raw_timestamp) or raw_timestamp <= 0:
        raise ValueError('timestamp Google Finance tidak tersedia')
    if raw_timestamp >= 10 ** 12:
        raw_timestamp /= 1000.0
    quote_time = pd.to_datetime(raw_timestamp, unit='s', utc=True, errors='coerce')
    if pd.isna(quote_time):
        raise ValueError('timestamp Google Finance tidak valid')
    return {'ticker': normalize_idx_ticker(ticker), 'price': float(price), 'timestamp': pd.Timestamp(quote_time).tz_convert('Asia/Jakarta').tz_localize(None), 'currency': currency or 'IDR', 'exchange': exchange or 'IDX'}

def fetch_google_finance_quotes(tickers: Iterable[str], max_tickers: int=24, timeout: int=10, max_workers: int=4, retry_count: int=2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch bounded public Google Finance quotes as an independent family.

    This is deliberately a shortlist fetch with low concurrency and cache at
    the application layer. Google Finance is not presented as an official API;
    a markup or access failure simply falls through to the next provider.
    """
    names = list(dict.fromkeys((normalize_idx_ticker(ticker) for ticker in tickers)))
    names = [ticker for ticker in names if ticker][:max(0, int(max_tickers))]
    if not names:
        return (_empty_automatic_independent_data(), _automatic_provider_report([]))
    import requests
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36', 'Accept-Language': 'id-ID,id;q=0.9,en;q=0.7'}

    def one(ticker: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        code = ticker.replace('.JK', '')
        url = f'https://www.google.com/finance/quote/{code}:IDX?hl=id'
        last_error = ''
        for _ in range(max(1, int(retry_count))):
            try:
                response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                response.raise_for_status()
                final_url = _safe_text(getattr(response, 'url', url)) or url
                parsed_url = urlparse(final_url)
                host = (parsed_url.hostname or '').lower()
                if host != 'google.com' and (not host.endswith('.google.com')):
                    raise RuntimeError('redirect keluar domain Google')
                quote = _google_finance_quote_from_html(response.text or '', ticker)
                row = {'ticker': ticker, 'Date': quote['timestamp'], 'Open': np.nan, 'High': np.nan, 'Low': np.nan, 'Close': quote['price'], 'Volume': np.nan, 'independent_source': 'GOOGLE_FINANCE_PUBLIC_QUOTE', 'independent_source_family': 'GOOGLE_FINANCE'}
                report = {'provider': 'GOOGLE_FINANCE', 'scope': ticker, 'status': 'OK', 'rows': 1, 'asof': quote['timestamp'], 'error': ''}
                return (row, report)
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {str(exc)[:140]}'
        return (None, {'provider': 'GOOGLE_FINANCE', 'scope': ticker, 'status': 'FAILED', 'rows': 0, 'asof': pd.NaT, 'error': last_error or 'respons tidak tersedia'})
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    workers = min(max(1, int(max_workers)), len(names))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, ticker) for ticker in names]
        for future in as_completed(futures):
            row, report = future.result()
            if row is not None:
                rows.append(row)
            reports.append(report)
    data = pd.DataFrame(rows, columns=_AUTOMATIC_INDEPENDENT_COLUMNS)
    if not data.empty:
        order = {ticker: index for index, ticker in enumerate(names)}
        data['_order'] = data['ticker'].map(order)
        data = data.sort_values('_order').drop(columns='_order').reset_index(drop=True)
    report = _automatic_provider_report(reports)
    if not report.empty:
        order = {ticker: index for index, ticker in enumerate(names)}
        report['_order'] = report['scope'].map(order).fillna(len(order))
        report = report.sort_values('_order').drop(columns='_order').reset_index(drop=True)
    return (data, report)

def _aligned_automatic_tickers(
    data: pd.DataFrame,
    primary_reference: Mapping[str, tuple[Any, float]] | None,
    config: ScanConfig,
    primary_source_families: Mapping[str, str] | None=None,
) -> set[str]:
    """Return tickers with a usable *independent* aligned provider row.

    A same-session cache hit is not enough when the cached provider later
    becomes the primary OHLCV family.  Treating that row as ``covered`` stopped
    the resolver before Google/iTick/Twelve Data could be tried, after which
    the validation gate correctly rejected the row as ``SAME_PROVIDER_FAMILY``.
    Coverage therefore has to include provider-family independence, not only
    price/date alignment.
    """
    if data is None or data.empty:
        return set()
    if not primary_reference:
        return set(data['ticker'].dropna().astype(str))
    aligned: set[str] = set()
    for ticker, group in data.groupby('ticker', sort=False):
        reference = primary_reference.get(str(ticker))
        if reference is None:
            continue
        primary_family = str((primary_source_families or {}).get(str(ticker), 'UNKNOWN') or 'UNKNOWN').upper()
        provider_family = group.get(
            'independent_source_family', pd.Series('', index=group.index),
        ).fillna('').astype(str).str.upper()
        independent_mask = provider_family.ne('YAHOO')
        if primary_family != 'UNKNOWN':
            independent_mask &= provider_family.ne(primary_family)
        group = group.loc[independent_mask].copy()
        if group.empty:
            continue
        reference_date, reference_close = reference
        reference_stamp = pd.Timestamp(reference_date)
        if reference_stamp.tzinfo is not None:
            reference_stamp = reference_stamp.tz_convert('Asia/Jakarta').tz_localize(None)
        latest = group.sort_values('Date').iloc[-1]
        provider_stamp = pd.Timestamp(latest['Date'])
        if provider_stamp.tzinfo is not None:
            provider_stamp = provider_stamp.tz_convert('Asia/Jakarta').tz_localize(None)
        provider_close = _finite(latest.get('Close'), np.nan)
        reference_close = _finite(reference_close, np.nan)
        if not np.isfinite(provider_close) or not np.isfinite(reference_close) or min(provider_close, reference_close) <= 0:
            continue
        date_gap = abs(int((reference_stamp.normalize() - provider_stamp.normalize()).days))
        tolerance = max(config.max_secondary_price_divergence_pct, 2.0 * idx_tick_size(reference_close) / reference_close)
        divergence = abs(provider_close / reference_close - 1.0)
        if date_gap <= config.max_independent_date_gap_days and divergence <= tolerance:
            aligned.add(str(ticker))
    return aligned

def fetch_automatic_independent_prices(tickers: Iterable[str], reference_date: Any | None=None, twelve_data_api_key: str='', itick_api_token: str='', primary_reference: Mapping[str, tuple[Any, float]] | None=None, primary_source_tiers: Mapping[str, str] | None=None, config: ScanConfig | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve second-source price evidence with no per-scan user upload.

    Provider order is current verified cache, official IDX EOD, Google Finance,
    optional iTick free, then Twelve Data when configured. A ticker counts as
    resolved only when its provider row is from the same completed session and
    aligned with the primary daily snapshot; stale/conflicting rows continue to
    the next fallback.
    """
    cfg = config or ScanConfig()
    names = list(dict.fromkeys((normalize_idx_ticker(ticker) for ticker in tickers)))
    names = [ticker for ticker in names if ticker][:max(0, int(cfg.max_automatic_price_candidates))]
    if not names:
        return (_empty_automatic_independent_data(), _automatic_provider_report([]))
    data_frames: list[pd.DataFrame] = []
    report_frames: list[pd.DataFrame] = []
    primary_families = {ticker: _primary_source_family_from_tier((primary_source_tiers or {}).get(ticker, '')) for ticker in names}
    cached = _load_independent_price_cache(names) if primary_reference else _empty_automatic_independent_data()
    if not cached.empty:
        data_frames.append(cached)
    covered = _aligned_automatic_tickers(
        cached, primary_reference, cfg, primary_source_families=primary_families,
    )
    if covered:
        report_frames.append(_automatic_provider_report([{
            'provider': 'INDEPENDENT_PRICE_CACHE',
            'scope': f'{len(covered)} ticker',
            'status': 'OK',
            'rows': len(covered),
            'asof': pd.to_datetime(cached.loc[cached['ticker'].isin(covered), 'Date'], errors='coerce').max(),
            'error': 'Bukti provider tersimpan masih sama-session dan selaras',
        }]))
    remaining = [ticker for ticker in names if ticker not in covered]
    official_names = [ticker for ticker in remaining if primary_families.get(ticker) != 'IDX_OFFICIAL']
    official, official_report = fetch_idx_official_eod_quotes(official_names, reference_date=reference_date, lookback_days=cfg.idx_summary_lookback_days, timeout=cfg.automatic_provider_timeout_seconds)
    skipped_idx = [ticker for ticker in remaining if ticker not in official_names]
    if skipped_idx:
        report_frames.append(_automatic_provider_report([{'provider': 'IDX_OFFICIAL_STOCK_SUMMARY', 'scope': ticker, 'status': 'SKIPPED_SAME_PRIMARY', 'rows': 0, 'asof': pd.NaT, 'error': 'Primary OHLCV terakhir berasal dari IDX; gunakan keluarga independen lain'} for ticker in skipped_idx]))
    if not official.empty:
        data_frames.append(official)
    if not official_report.empty:
        report_frames.append(official_report)
    covered.update(_aligned_automatic_tickers(
        official, primary_reference, cfg, primary_source_families=primary_families,
    ))
    remaining = [ticker for ticker in names if ticker not in covered]
    google, google_report = fetch_google_finance_quotes(remaining, max_tickers=len(remaining), timeout=cfg.automatic_provider_timeout_seconds, max_workers=cfg.google_finance_max_workers)
    if not google.empty:
        data_frames.append(google)
    if not google_report.empty:
        report_frames.append(google_report)
    covered.update(_aligned_automatic_tickers(
        google, primary_reference, cfg, primary_source_families=primary_families,
    ))
    remaining = [ticker for ticker in names if ticker not in covered]
    itick_names = [ticker for ticker in remaining if primary_families.get(ticker) != 'ITICK']
    if itick_names and _safe_text(itick_api_token):
        itick, itick_report = fetch_itick_independent_eod(
            itick_names,
            api_token=itick_api_token,
            max_tickers=cfg.max_itick_independent_candidates,
            timeout=cfg.automatic_provider_timeout_seconds,
        )
        if not itick.empty:
            data_frames.append(itick)
        if not itick_report.empty:
            report_frames.append(itick_report)
        covered.update(_aligned_automatic_tickers(
            itick, primary_reference, cfg, primary_source_families=primary_families,
        ))
        remaining = [ticker for ticker in names if ticker not in covered]
    if remaining and _safe_text(twelve_data_api_key):
        twelve, twelve_report = fetch_twelve_data_eod(remaining, api_key=twelve_data_api_key, outputsize=30, max_tickers=len(remaining), timeout=cfg.automatic_provider_timeout_seconds)
        if not twelve.empty:
            data_frames.append(twelve)
        if not twelve_report.empty:
            normalized_report = pd.DataFrame({'provider': 'TWELVE_DATA', 'scope': twelve_report.get('ticker', ''), 'status': twelve_report.get('status', 'FAILED'), 'rows': pd.to_numeric(twelve_report.get('bars', 0), errors='coerce').fillna(0).astype(int), 'asof': pd.NaT, 'error': twelve_report.get('error', '')})
            report_frames.append(_automatic_provider_report(normalized_report.to_dict('records')))
    elif remaining:
        report_frames.append(_automatic_provider_report([{'provider': 'TWELVE_DATA', 'scope': f'{len(remaining)} unresolved ticker', 'status': 'OPTIONAL_NOT_CONFIGURED', 'rows': 0, 'asof': pd.NaT, 'error': 'Fallback berbayar tidak dikonfigurasi; tidak diperlukan bila IDX/Google berhasil'}]))
    data = pd.concat(data_frames, ignore_index=True) if data_frames else _empty_automatic_independent_data()
    if not data.empty:
        data = data.drop_duplicates(['ticker', 'independent_source_family', 'Date'], keep='last').reset_index(drop=True)
        _write_independent_price_cache(data)
    report = pd.concat(report_frames, ignore_index=True) if report_frames else _automatic_provider_report([])
    return (data, report)
IDX_DAILY_FINAL_HOUR = 16
IDX_DAILY_FINAL_MINUTE = 20
IDX_REGULAR_DECISION_START_HOUR = 9
IDX_REGULAR_DECISION_START_MINUTE = 0
IDX_CACHE_SCHEMA_VERSION = 3

def _jakarta_timestamp(now: Any | None=None) -> pd.Timestamp:
    stamp = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='Asia/Jakarta')
    if stamp.tzinfo is None:
        return stamp.tz_localize('Asia/Jakarta')
    return stamp.tz_convert('Asia/Jakarta')

def idx_daily_bar_is_final(now: Any | None=None) -> bool:
    """Whether today's IDX daily candle is eligible to be treated as final."""
    stamp = _jakarta_timestamp(now)
    return bool(stamp.weekday() < 5 and (stamp.hour, stamp.minute) >= (IDX_DAILY_FINAL_HOUR, IDX_DAILY_FINAL_MINUTE))

def idx_regular_decision_window(now: Any | None=None) -> bool:
    """Return True only during the weekday window in which today's candle can still change.

    Pre-market scans intentionally return False: before 09:00 WIB the latest completed
    daily candle is yesterday's EOD and may be used to prepare today's limit order.
    """
    stamp = _jakarta_timestamp(now)
    if stamp.weekday() >= 5:
        return False
    minute_of_day = stamp.hour * 60 + stamp.minute
    start = IDX_REGULAR_DECISION_START_HOUR * 60 + IDX_REGULAR_DECISION_START_MINUTE
    final = IDX_DAILY_FINAL_HOUR * 60 + IDX_DAILY_FINAL_MINUTE
    return bool(start <= minute_of_day < final)

def idx_core_waits_for_eod(now: Any | None=None, market_state: str='UNKNOWN', quote_time: Any | None=None, current_bar_incomplete: bool=False) -> bool:
    """Decide whether direct-order status must wait for today's EOD candle.

    The old v4.3.2 rule blocked every weekday before 16:20, including 05:00 WIB.
    This function distinguishes pre-market from an active/incomplete trading day.
    """
    stamp = _jakarta_timestamp(now)
    state = str(market_state or 'UNKNOWN').strip().upper()
    if bool(current_bar_incomplete) or state == 'REGULAR':
        return True
    if not idx_regular_decision_window(stamp):
        return False
    parsed_quote = pd.to_datetime(quote_time, errors='coerce')
    if pd.notna(parsed_quote):
        parsed_quote = pd.Timestamp(parsed_quote)
        if parsed_quote.tzinfo is not None:
            parsed_quote = parsed_quote.tz_convert('Asia/Jakarta').tz_localize(None)
        if parsed_quote.date() == stamp.date():
            return True
        if state in {'CLOSED', 'POST'} and parsed_quote.date() < stamp.date():
            return False
    return state not in {'CLOSED', 'POST'}

def _expected_last_completed_daily_date(now: Any | None=None) -> pd.Timestamp:
    """Return the latest date that may safely be treated as a completed IDX EOD bar."""
    stamp = _jakarta_timestamp(now)
    day = stamp.normalize()
    if idx_daily_bar_is_final(stamp):
        return day.tz_localize(None)
    day = day - pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day = day - pd.Timedelta(days=1)
    return day.tz_localize(None)

def _completed_daily_frame(frame: pd.DataFrame, now: Any | None=None) -> pd.DataFrame:
    """Remove any still-forming daily candle before core scanning or EOD caching."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    source_attrs = dict(getattr(frame, 'attrs', {}) or {})
    clean = _clean_ohlcv(frame, strict=True)
    if clean.empty:
        return clean
    cutoff = _expected_last_completed_daily_date(now)
    normalized = pd.DatetimeIndex(clean.index).normalize()
    clean = clean.loc[normalized <= cutoff].copy()
    if clean.empty:
        return clean
    clean.attrs.update(source_attrs)
    clean.attrs['bar_state'] = 'FINAL_EOD'
    clean.attrs['finalized_for_date'] = cutoff.date().isoformat()
    clean.attrs['last_bar_date'] = pd.Timestamp(clean.index[-1]).date().isoformat()
    return clean

def _load_daily_ohlcv_cache(ticker: str) -> pd.DataFrame:
    frame = _load_daily_ohlcv_cache_v431(ticker)
    if frame.empty:
        return frame
    try:
        import json as _json
        meta_path = _daily_ohlcv_cache_meta_path(ticker)
        if meta_path.is_file():
            payload = _json.loads(meta_path.read_text(encoding='utf-8'))
            frame.attrs.update(payload if isinstance(payload, dict) else {})
    except Exception:
        pass
    return frame

def _cache_meta_proves_final(frame: pd.DataFrame, now: Any | None=None) -> bool:
    if frame is None or frame.empty:
        return False
    attrs = dict(getattr(frame, 'attrs', {}) or {})
    latest = pd.Timestamp(frame.index[-1]).normalize()
    expected = _expected_last_completed_daily_date(now)
    today = _jakarta_timestamp(now).tz_localize(None).normalize()
    if latest > expected or (expected - latest).days > 4:
        return False
    if str(attrs.get('bar_state', '')).upper() == 'FINAL_EOD':
        return True
    written = _as_jakarta_naive_timestamp(attrs.get('written_at'))
    if pd.notna(written):
        if latest < written.normalize():
            return True
        if latest == written.normalize() and (written.hour, written.minute) >= (IDX_DAILY_FINAL_HOUR, IDX_DAILY_FINAL_MINUTE):
            return True
    return bool(latest < today)

def _daily_cache_is_current(frame: pd.DataFrame, now: Any | None=None) -> bool:
    clean = _clean_ohlcv(frame, strict=True)
    if clean.empty:
        return False
    clean.attrs.update(dict(getattr(frame, 'attrs', {}) or {}))
    return _cache_meta_proves_final(clean, now)

def _write_daily_ohlcv_cache(ticker: str, frame: pd.DataFrame, source_family: str='UNKNOWN', now: Any | None=None) -> None:
    """Atomically persist completed EOD bars only; never cache an intraday candle as final."""
    final_frame = _completed_daily_frame(frame, now)
    if final_frame.empty:
        return
    tmp: Path | None = None
    meta_tmp: Path | None = None
    try:
        import json as _json
        stamp = _jakarta_timestamp(now)
        path = _daily_ohlcv_cache_path(ticker)
        tmp = path.with_suffix('.tmp')
        final_frame.to_csv(tmp, index=True, index_label='Date')
        tmp.replace(path)
        meta_path = _daily_ohlcv_cache_meta_path(ticker)
        meta_tmp = meta_path.with_suffix('.tmp')
        payload = {'schema_version': IDX_CACHE_SCHEMA_VERSION, 'source_family': str(source_family or 'UNKNOWN').upper(), 'written_at': stamp.isoformat(), 'last_bar_date': pd.Timestamp(final_frame.index[-1]).date().isoformat(), 'bar_state': 'FINAL_EOD', 'finalized_for_date': _expected_last_completed_daily_date(stamp).date().isoformat(), 'finalization_cutoff_wib': f'{IDX_DAILY_FINAL_HOUR:02d}:{IDX_DAILY_FINAL_MINUTE:02d}'}
        meta_tmp.write_text(_json.dumps(payload), encoding='utf-8')
        meta_tmp.replace(meta_path)
    except Exception:
        for candidate in (tmp, meta_tmp):
            try:
                if candidate is not None:
                    candidate.unlink(missing_ok=True)
            except Exception:
                pass

def download_ohlcv(tickers: Iterable[str], period: str='3y', batch_size: int=30, itick_api_token: str='') -> tuple[dict[str, pd.DataFrame], DownloadReport]:
    histories, report = _download_ohlcv_v431(tickers, period, batch_size, itick_api_token)
    filtered: dict[str, pd.DataFrame] = {}
    for ticker, frame in histories.items():
        completed = _completed_daily_frame(frame)
        if completed.empty:
            report.failed[ticker] = 'Tidak ada completed EOD bar yang aman untuk core scanner'
            report.source_tiers[ticker] = 'UNAVAILABLE'
            continue
        if len(completed) < len(frame):
            prior = report.warnings.get(ticker, '')
            note = 'Candle intraday parsial dibuang; core memakai completed EOD terakhir'
            report.warnings[ticker] = ' • '.join((x for x in (prior, note) if x))
        filtered[ticker] = completed
    report.downloaded = sorted(filtered)
    return (filtered, report)

def download_benchmark(period: str='3y') -> pd.DataFrame:
    return _completed_daily_frame(_download_benchmark_v431(period))

def _ready_distance_atr_for_setup(setup: str, cfg: ScanConfig) -> float:
    limits = {'PULLBACK_CONTINUATION': 0.45, 'BREAKOUT_RETEST': 0.45, 'REVERSAL_ACCUMULATION': 0.35, 'UNICORN_SNIPER_ICT': 0.3}
    return max(float(cfg.ready_distance_atr), limits.get(str(setup), float(cfg.ready_distance_atr)))
def apply_execution_snapshot_gate(signals: pd.DataFrame, snapshots: pd.DataFrame, config: ScanConfig | None=None) -> pd.DataFrame:
    """Confirm quotes without treating normal intraday movement as an EOD conflict."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if snapshots is None or snapshots.empty:
        snapshots = pd.DataFrame({'ticker': out['ticker'].drop_duplicates()})
    out = out.merge(snapshots, on='ticker', how='left')
    now_jkt = _jakarta_timestamp()
    now = now_jkt.tz_localize(None)
    quote_source = out['quote_time'] if 'quote_time' in out else pd.Series(pd.NaT, index=out.index)
    quote_time = pd.to_datetime(quote_source, errors='coerce')
    out['quote_age_days'] = (now.normalize() - quote_time.dt.normalize()).dt.days
    out['quote_critical_blocker'] = False
    out['pending_close'] = False
    for idx, row in out.iterrows():
        verified = _truthy(row.get('quote_verified', False))
        age = _finite(row.get('quote_age_days'), np.nan)
        signal_price = _finite(row.get('last_price'), 0)
        quote_price = _finite(row.get('quote_last_price'), 0)
        atr_pct = max(0.0, _finite(row.get('atr_pct'), 0))
        tolerance = max(0.015, 0.6 * atr_pct)
        divergence = abs(quote_price / signal_price - 1) if signal_price > 0 and quote_price > 0 else np.nan
        market_state = _safe_text(row.get('quote_market_state')).upper() or 'UNKNOWN'
        spread = _finite(row.get('quote_spread_pct'), np.nan)
        data_age = _finite(row.get('absolute_data_age_days'), 999)
        final_ohlcv = data_age <= cfg.max_absolute_data_age_days and (not _truthy(row.get('current_bar_incomplete', False)))
        if verified and np.isfinite(age) and (0 <= age <= 3):
            confidence = 100.0
        else:
            confidence = 68.0 if final_ohlcv else 25.0
            _append_pipe(out, idx, 'evidence_warnings', 'Quote snapshot tidak lengkap; menggunakan final OHLCV sebagai fallback')
        out.at[idx, 'quote_confidence'] = confidence
        out.at[idx, 'intraday_move_from_eod_pct'] = divergence if np.isfinite(divergence) else np.nan
        wait_for_eod = idx_core_waits_for_eod(now=now_jkt, market_state=market_state, quote_time=row.get('quote_time'), current_bar_incomplete=_truthy(row.get('current_bar_incomplete', False)))
        out.at[idx, 'execution_session_phase'] = 'REGULAR_WAIT_EOD' if wait_for_eod else 'PREMARKET_PREVIOUS_EOD' if now_jkt.weekday() < 5 and (not idx_regular_decision_window(now_jkt)) and (not idx_daily_bar_is_final(now_jkt)) else 'POST_CLOSE_FINAL_EOD'
        if wait_for_eod:
            out.at[idx, 'pending_close'] = True
            _append_pipe(out, idx, 'evidence_warnings', 'Sesi aktif/incomplete: setup dipertahankan, keputusan direct-order menunggu candle EOD final')
            if np.isfinite(divergence) and divergence > tolerance:
                _append_pipe(out, idx, 'evidence_warnings', f'Harga intraday bergerak {divergence:.1%} dari EOD; ini bukan konflik data')
            if not np.isfinite(spread) or spread < 0 or spread > 0.015:
                _append_pipe(out, idx, 'evidence_warnings', 'Spread live tidak tersedia atau >1,5%; validasi ulang setelah penutupan')
            continue
        if np.isfinite(divergence) and divergence > tolerance:
            out.at[idx, 'quote_critical_blocker'] = True
            out.at[idx, 'quote_confidence'] = 0.0
            _set_context_block(out, idx, f'Konflik harga quote vs final OHLCV {divergence:.1%}')
        if market_state not in {'CLOSED', 'PRE', 'PREPRE', 'POST', 'UNKNOWN', ''}:
            _append_pipe(out, idx, 'evidence_warnings', f'Market state tidak dikenali: {market_state}')
    out['status_rank'] = out['status'].map(STATUS_ORDER).fillna(99)
    return out

def _finalize_execution_integrity_v433(signals: pd.DataFrame, config: ScanConfig | None=None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    out = _finalize_execution_integrity_v431(signals, cfg)
    if out.empty:
        return out
    for idx, row in out.iterrows():
        setup = _safe_text(row.get('setup'))
        smart = _finite(row.get('silent_accumulation_score'), 0.0)
        strict_flow = setup in {'REVERSAL_ACCUMULATION', 'UNICORN_SNIPER_ICT'}
        confidence = _finite(row.get('execution_confidence_score'), 0.0)
        modifier = 0.0 if strict_flow else 2.0 if smart >= 70 else 0.0 if smart >= 45 else -3.0
        adjusted_confidence = round(min(100.0, max(0.0, confidence + modifier)), 1)
        out.at[idx, 'flow_confidence_modifier'] = modifier
        out.at[idx, 'execution_confidence_score'] = adjusted_confidence
        out.at[idx, 'execution_integrity_score'] = adjusted_confidence
        if _truthy(row.get('pending_close', False)):
            out.at[idx, 'status'] = 'PENDING_CLOSE'
            out.at[idx, 'order_instruction'] = 'DO_NOT_BUY'
            out.at[idx, 'stockbit_order_price'] = np.nan
            out.at[idx, 'stockbit_order_lots'] = 0
            out.at[idx, 'automation_decision'] = 'WAIT_EOD_REFRESH'
            failures = [x for x in _safe_text(row.get('execution_gate_failures')).split(' | ') if x]
            if 'DAILY_BAR_NOT_FINAL' not in failures:
                failures.insert(0, 'DAILY_BAR_NOT_FINAL')
            out.at[idx, 'execution_gate_failures'] = ' | '.join(failures)
            out.at[idx, 'primary_execution_blocker'] = 'DAILY_BAR_NOT_FINAL'
            _append_pipe(out, idx, 'evidence_warnings', 'Refresh setelah 16:20 WIB untuk keputusan final EOD')
            continue
        failures = [x for x in _safe_text(row.get('execution_gate_failures')).split(' | ') if x]
        if adjusted_confidence >= cfg.min_execution_confidence and 'EXECUTION_CONFIDENCE' in failures:
            failures.remove('EXECUTION_CONFIDENCE')
        if modifier < 0 and adjusted_confidence < cfg.min_execution_confidence and (out.at[idx, 'status'] == 'EXECUTION_READY'):
            out.at[idx, 'status'] = 'PENDING_DATA'
            out.at[idx, 'order_instruction'] = 'DO_NOT_BUY'
            out.at[idx, 'stockbit_order_price'] = np.nan
            out.at[idx, 'stockbit_order_lots'] = 0
            out.at[idx, 'automation_decision'] = 'RETRY_OR_WATCH'
            if 'EXECUTION_CONFIDENCE' not in failures:
                failures.append('EXECUTION_CONFIDENCE')
        elif out.at[idx, 'status'] == 'PENDING_DATA' and (not failures) and (adjusted_confidence >= cfg.min_execution_confidence) and _truthy(row.get('portfolio_selected', False)) and (not _safe_text(row.get('critical_blockers'))):
            out.at[idx, 'status'] = 'EXECUTION_READY'
            out.at[idx, 'order_instruction'] = 'BUY_LIMIT'
            out.at[idx, 'stockbit_order_price'] = row.get('entry')
            out.at[idx, 'stockbit_order_lots'] = int(_finite(row.get('suggested_lots'), 0))
            out.at[idx, 'automation_decision'] = 'DIRECT_EXECUTION_ELIGIBLE'
        out.at[idx, 'execution_gate_failures'] = ' | '.join(failures)
        out.at[idx, 'primary_execution_blocker'] = failures[0] if failures else 'NONE'
    out['status_rank'] = out['status'].map(STATUS_ORDER).fillna(99)
    return out
STATUS_ORDER = {
    'EXECUTION_READY': 0,
    'READY_FOR_STOCKBIT_VERIFY': 1,
    'SIGNAL_READY': 2,
    'ENTRY_PLAN_READY': 3,
    'READY_NOT_SELECTED': 4,
    'READY_FOR_PRICE_VERIFY': 5,
    'PENDING_CLOSE': 6,
    'PENDING_DATA': 7,
    'WATCHLIST_ENTRY': 8,
    'BLOCKED_CONTEXT': 9,
    'REJECT': 10,
}
_ANALYST_SOFT_BLOCKER_PREFIXES = ('Retest/reclaim/entry trigger belum lengkap', 'Quality score ', 'Confirmation retest harus terjadi pada bar terakhir', 'volume pullback belum kontraksi', 'relative strength vs IHSG negatif', 'CMF/OBV belum mendukung', 'ADX < 18')
_ANALYST_HARD_BLOCKER_TOKENS = ('Riwayat hanya', 'di bawah minimum', 'ADTV20', 'Hari volume nol', 'ATR ', 'Data tertinggal', 'Data absolut', 'Daily bar hari ini belum dianggap final', 'Nilai transaksi bar terakhir sangat rendah', 'dekat/terkunci ARA', 'Zona berumur', 'Masa berlaku setup sudah habis', 'terlalu jauh', 'menutup jauh di bawah zona entry', 'Jarak SL', 'Level entry/SL tidak valid', 'fraksi harga IDX', 'Urutan SL < entry < TP1 < TP2', 'di luar rentang auto-rejection', 'Risk/reward di bawah minimum', 'Regime IHSG RISK_OFF', 'Regime IHSG tidak dapat diverifikasi', 'Flow proxy ', 'Strict Unicorn memerlukan', 'Base terlalu lebar', 'Breakout ditolak')

def _pipe_parts(value: object) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    return [piece.strip() for piece in text.split(' • ') if piece.strip()]

def _finalize_execution_integrity_v440(signals: pd.DataFrame, config: ScanConfig | None=None) -> pd.DataFrame:
    """Finalize strict and Analyst Fusion decisions in one auditable output."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    prior = signals.copy()
    out = _finalize_execution_integrity_v433(signals, cfg)
    if 'analyst_pre_budget_ready' not in out:
        out = apply_analyst_fusion_gate(out, cfg)
    for idx, row in out.iterrows():
        prior_status = _safe_text(prior.at[idx, 'status']) if idx in prior.index and 'status' in prior else 'WATCHLIST_ENTRY'
        candidate = _truthy(row.get('analyst_pre_budget_ready', False))
        selected = _truthy(row.get('portfolio_selected', False))
        pending_close = _truthy(row.get('pending_close', False))
        hard = _safe_text(row.get('analyst_hard_blockers'))
        strict_ready = _safe_text(row.get('status')) == 'EXECUTION_READY'
        out.at[idx, 'strict_execution_ready'] = strict_ready
        out.at[idx, 'strict_execution_gate_failures'] = _safe_text(row.get('execution_gate_failures'))
        out.at[idx, 'strict_primary_execution_blocker'] = _safe_text(row.get('primary_execution_blocker')) or 'NONE'
        if pending_close:
            if candidate or _truthy(row.get('technical_setup_ready', False)) or prior_status == 'EXECUTION_READY':
                out.at[idx, 'status'] = 'PENDING_CLOSE'
                out.at[idx, 'order_instruction'] = 'DO_NOT_BUY'
                out.at[idx, 'automation_decision'] = 'WAIT_EOD_REFRESH'
                out.at[idx, 'primary_execution_blocker'] = 'DAILY_BAR_NOT_FINAL'
            else:
                restored = prior_status if prior_status in STATUS_ORDER else 'WATCHLIST_ENTRY'
                out.at[idx, 'status'] = restored
                out.at[idx, 'order_instruction'] = 'DO_NOT_BUY'
                out.at[idx, 'automation_decision'] = 'WATCH'
            continue
        if strict_ready:
            out.at[idx, 'execution_mode'] = 'STRICT_VERIFIED'
            out.at[idx, 'requires_stockbit_price_check'] = False
            continue
        if candidate and selected and (not hard):
            out.at[idx, 'status'] = 'EXECUTION_READY'
            out.at[idx, 'execution_mode'] = 'ANALYST_FUSION'
            out.at[idx, 'order_instruction'] = 'BUY_LIMIT'
            out.at[idx, 'stockbit_order_price'] = row.get('entry')
            out.at[idx, 'stockbit_order_lots'] = int(_finite(row.get('suggested_lots'), 0))
            out.at[idx, 'automation_decision'] = 'DIRECT_PLAN_VERIFY_BROKER'
            out.at[idx, 'execution_gate_failures'] = ''
            out.at[idx, 'primary_execution_blocker'] = 'NONE'
            out.at[idx, 'execution_readiness_pct'] = 100.0
            out.at[idx, 'evidence_state'] = 'ANALYST_RESOLVED'
            if _truthy(row.get('requires_stockbit_price_check', True)):
                _append_pipe(out, idx, 'evidence_warnings', 'Sebelum submit, cocokkan harga terakhir dan bid/offer di Stockbit; provider independen otomatis belum terverifikasi')
        elif candidate and (not selected) and (not hard):
            out.at[idx, 'status'] = 'READY_NOT_SELECTED'
            out.at[idx, 'execution_mode'] = 'ANALYST_FUSION_ALTERNATE'
            out.at[idx, 'order_instruction'] = 'DO_NOT_BUY'
            out.at[idx, 'stockbit_order_price'] = np.nan
            out.at[idx, 'stockbit_order_lots'] = 0
            out.at[idx, 'automation_decision'] = 'ALTERNATE_READY'
            out.at[idx, 'primary_execution_blocker'] = 'PORTFOLIO_BUDGET'
        elif hard:
            out.at[idx, 'execution_mode'] = 'BLOCKED'
        else:
            out.at[idx, 'execution_mode'] = 'WATCH'
    out['status_rank'] = out['status'].map(STATUS_ORDER).fillna(99)
    return out

def _risk_disclosure(row: Mapping[str, Any], cfg: ScanConfig) -> tuple[str, list[str]]:
    flags: list[str] = []
    stop_pct = _finite(row.get('stop_pct'), np.nan)
    rr1 = _finite(row.get('rr1'), np.nan)
    rr2 = _finite(row.get('rr2'), np.nan)
    adtv = _finite(row.get('adtv20_idr'), np.nan)
    atr_pct = _finite(row.get('atr_pct'), np.nan)
    regime = _safe_text(row.get('market_regime')).upper()
    if np.isfinite(stop_pct) and stop_pct > cfg.max_stop_pct:
        flags.append(f'SL lebar {stop_pct:.1%}')
    if np.isfinite(rr1) and rr1 < cfg.min_rr1:
        flags.append(f'RR1 rendah {rr1:.2f}')
    if np.isfinite(rr2) and rr2 < cfg.min_rr2:
        flags.append(f'RR2 rendah {rr2:.2f}')
    if np.isfinite(adtv) and adtv < cfg.min_adtv_idr:
        flags.append('Likuiditas di bawah preferensi')
    if np.isfinite(atr_pct) and atr_pct > cfg.max_atr_pct:
        flags.append('Volatilitas ekstrem')
    if regime in {'RISK_OFF', 'UNKNOWN'}:
        flags.append(f'Regime {regime}')
    if _truthy(row.get('fundamental_critical_blocker', False)):
        flags.append('Fundamental distress')
    if _truthy(row.get('news_critical_blocker', False)):
        flags.append('Berita material negatif')
    if _truthy(row.get('quote_critical_blocker', False)):
        flags.append('Harga wajib diverifikasi di Stockbit')
    grade = 'VERY_HIGH' if len(flags) >= 4 else 'HIGH' if len(flags) >= 2 else 'MODERATE' if flags else 'NORMAL'
    return (grade, flags)

def enforce_analyst_portfolio_budget(signals: pd.DataFrame, config: ScanConfig | None=None, current_positions: int=0, current_open_risk_idr: float=0.0, current_invested_idr: float=0.0, cash_on_hand_idr: float | None=None) -> pd.DataFrame:
    """Keep every setup visible while reserving safe budget for autopilot orders.

    Multiple setups for one ticker remain in the table.  Only one can reserve
    cash/risk, preventing two valid detectors from accidentally doubling the
    same Stockbit exposure.
    """
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    if 'analyst_pre_budget_ready' not in out:
        out = apply_analyst_fusion_gate(out, cfg)
    policy = _safe_text(getattr(cfg, 'execution_policy', 'SIGNAL_FIRST')).upper() or 'SIGNAL_FIRST'
    out['portfolio_selected'] = out['analyst_pre_budget_ready'].map(_truthy)
    if policy != 'ACCOUNT_GUARDED':
        out['portfolio_blockers'] = ''
        out['account_risk_gate_applied'] = False
        out['autopilot_budget_pass'] = False
        out['autopilot_budget_blockers'] = ''
        out['autopilot_primary_setup'] = False
        out['execution_policy'] = policy
        out['setup_state'] = np.where(out['analyst_pre_budget_ready'].map(_truthy), 'SETUP_READY', 'WATCH_ONLY')
        out['account_order_state'] = np.where(out['analyst_pre_budget_ready'].map(_truthy), 'USER_MANAGED', 'NOT_APPLICABLE')
        return out
    out['portfolio_blockers'] = ''
    out['account_risk_gate_applied'] = bool(cfg.autopilot_enabled)
    out['autopilot_budget_pass'] = False
    out['autopilot_budget_blockers'] = ''
    out['autopilot_primary_setup'] = False
    risk_cap = max(0.0, cfg.account_size_idr * cfg.max_portfolio_risk_pct)
    remaining_risk = max(0.0, risk_cap - max(0.0, float(current_open_risk_idr)))
    available_cash = cfg.cash_on_hand_idr if cash_on_hand_idr is None else float(cash_on_hand_idr)
    remaining_cash = max(0.0, available_cash)
    remaining_slots = max(0, int(cfg.max_positions) - max(0, int(current_positions)))
    out['portfolio_remaining_risk_idr'] = remaining_risk
    out['portfolio_remaining_cash_idr'] = remaining_cash
    out['portfolio_remaining_slots'] = remaining_slots
    confirmed_mask = out.get('analyst_order_mode', pd.Series('', index=out.index)).astype(str).eq('TRIGGER_CONFIRMED')
    confirmed_mask |= out.get('action', pd.Series('', index=out.index)).astype(str).isin({'READY_TRIGGER', 'READY_LIMIT'})
    # Reserve cash and risk from the price that will actually be entered into
    # Stockbit.  A buy-stop/LIT ticket can have a limit one tick above its
    # trigger, so sizing from the detector entry alone can understate both
    # capital and maximum loss when several candidates share the same account.
    for idx in out.index[confirmed_mask]:
        semantics = _execution_order_semantics(out.loc[idx])
        ticket_price = _finite(semantics.get('stockbit_order_price'), np.nan)
        if not np.isfinite(ticket_price):
            continue
        ticket_sizing = size_stockbit_order(
            ticket_price,
            _finite(out.at[idx, 'stop_loss'] if 'stop_loss' in out else np.nan, np.nan),
            cfg,
        )
        for key, value in ticket_sizing.items():
            out.at[idx, key] = value
    candidate_mask = out['portfolio_selected'].map(_truthy) & confirmed_mask
    independent_mask = out.get('independent_price_verified', pd.Series(False, index=out.index)).map(_truthy)
    context_mask = out.get('critical_blockers', pd.Series('', index=out.index)).fillna('').astype(str).str.strip().eq('')
    for flag in ('validation_critical_blocker', 'fundamental_critical_blocker', 'market_status_critical_blocker', 'news_critical_blocker', 'quote_critical_blocker'):
        context_mask &= ~out.get(flag, pd.Series(False, index=out.index)).map(_truthy)
    regime = out.get('market_regime', pd.Series('UNKNOWN', index=out.index)).fillna('UNKNOWN').astype(str).str.upper()
    regime_mask = regime.ne('UNKNOWN') & (regime.ne('RISK_OFF') | bool(cfg.allow_autopilot_risk_off))
    candidate_mask &= independent_mask & context_mask & regime_mask
    candidates = out.index[candidate_mask].tolist()
    sort_cols = [c for c in ('analyst_fusion_score', 'quality_score', 'validation_gate_score', 'silent_accumulation_score', 'rr2', 'adtv20_idr') if c in out]
    ranked = out.loc[candidates].sort_values(sort_cols, ascending=False, na_position='last') if candidates else out.iloc[0:0]
    out['execution_rank'] = np.nan
    reserved_tickers: set[str] = set()
    for rank, idx in enumerate(ranked.index, start=1):
        out.at[idx, 'execution_rank'] = rank
        ticker = _safe_text(out.at[idx, 'ticker'])
        blockers: list[str] = []
        if ticker in reserved_tickers:
            blockers.append('ALTERNATE_SETUP_SAME_TICKER')
        sizing_ok = _safe_text(out.at[idx, 'sizing_status'] if 'sizing_status' in out else '') == 'OK'
        lots = int(_finite(out.at[idx, 'suggested_lots'] if 'suggested_lots' in out else 0, 0))
        capital = max(0.0, _finite(out.at[idx, 'capital_required_idr'] if 'capital_required_idr' in out else np.nan, np.inf))
        max_loss = max(0.0, _finite(out.at[idx, 'max_loss_idr'] if 'max_loss_idr' in out else np.nan, np.inf))
        if not sizing_ok or lots < 1:
            blockers.append('SIZING_ZERO_OR_INVALID')
        if remaining_slots < 1:
            blockers.append('MAX_POSITIONS_REACHED')
        if not np.isfinite(capital) or capital > remaining_cash:
            blockers.append('INSUFFICIENT_CASH')
        if not np.isfinite(max_loss) or max_loss > remaining_risk:
            blockers.append('PORTFOLIO_RISK_CAP')
        passed = not blockers
        out.at[idx, 'autopilot_budget_pass'] = passed
        out.at[idx, 'autopilot_primary_setup'] = passed
        out.at[idx, 'autopilot_budget_blockers'] = ' | '.join(blockers)
        out.at[idx, 'portfolio_blockers'] = ' • '.join(blockers)
        if passed:
            reserved_tickers.add(ticker)
            remaining_slots -= 1
            remaining_cash = max(0.0, remaining_cash - capital)
            remaining_risk = max(0.0, remaining_risk - max_loss)
        out.at[idx, 'portfolio_remaining_slots'] = remaining_slots
        out.at[idx, 'portfolio_remaining_cash_idr'] = remaining_cash
        out.at[idx, 'portfolio_remaining_risk_idr'] = remaining_risk
    return out

def _execution_order_semantics(row: Mapping[str, Any]) -> dict[str, Any]:
    """Translate detector intent into an order type matching actual order mechanics."""
    entry = _finite(row.get('entry'), np.nan)
    trigger = _finite(row.get('trigger'), entry)
    entry_high = _finite(row.get('entry_high'), np.nan)
    entry_type = _safe_text(row.get('entry_type')).upper()
    last = _finite(row.get('last_price'), np.nan)
    result: dict[str, Any] = {
        'order_instruction': 'WAIT_PRICE_AND_CONFIRM',
        'stockbit_trigger_price': np.nan,
        'stockbit_limit_price': np.nan,
        'stockbit_order_price': np.nan,
        'execution_timing': 'WAIT_FOR_ZONE_OR_CONFIRMATION',
        'stockbit_order_template': 'NONE',
        'stockbit_time_in_force': 'GFD',
        'broker_submission_mode': 'MANUAL_STOCKBIT',
        'broker_price_revalidation_required': True,
        'opening_gap_recheck_required': True,
    }
    buy_stop_style = ('BUY_STOP' in entry_type) or (
        'CONFIRMATION' in entry_type and np.isfinite(entry) and np.isfinite(last) and entry > last
    )
    if buy_stop_style and np.isfinite(trigger):
        trigger_price = round_idx_price(trigger, 'up')
        limit_price = round_idx_price(trigger_price + idx_tick_size(trigger_price), 'up')
        if np.isfinite(entry_high) and entry_high >= trigger_price:
            limit_price = min(limit_price, round_idx_price(entry_high, 'up'))
        result.update({
            'order_instruction': 'BUY_LIT_USER_SIZE',
            'stockbit_trigger_price': trigger_price,
            'stockbit_limit_price': limit_price,
            'stockbit_order_price': limit_price,
            'execution_timing': 'ACTIVATE_ONLY_AFTER_TRIGGER',
            'stockbit_order_template': 'LIT_WITH_SEPARATE_SL_TP',
        })
    elif np.isfinite(entry):
        limit_price = round_idx_price(entry, 'nearest')
        result.update({
            'order_instruction': 'BUY_LIMIT_USER_SIZE',
            'stockbit_limit_price': limit_price,
            'stockbit_order_price': limit_price,
            'execution_timing': 'LIMIT_ORDER_NOW',
            'stockbit_order_template': 'BRACKET_ORDER_LIMIT',
        })
    return result

def _autopilot_order_is_valid(row: Mapping[str, Any], semantics: Mapping[str, Any]) -> bool:
    """Validate that the proposed Stockbit ticket matches its trigger intent."""
    instruction = _safe_text(semantics.get('order_instruction')).upper()
    reference = _finite(
        row.get('independent_last_price'),
        _finite(row.get('quote_last_price'), _finite(row.get('last_price'), np.nan)),
    )
    stop = _finite(row.get('stop_loss'), np.nan)
    tp1 = _finite(row.get('tp1'), np.nan)
    tp2 = _finite(row.get('tp2'), np.nan)
    if not all(is_valid_idx_price(value) for value in (stop, tp1, tp2)):
        return False
    if instruction == 'BUY_LIT_USER_SIZE':
        trigger = _finite(semantics.get('stockbit_trigger_price'), np.nan)
        limit_price = _finite(semantics.get('stockbit_limit_price'), np.nan)
        return bool(
            np.isfinite(reference)
            and is_valid_idx_price(trigger)
            and is_valid_idx_price(limit_price)
            and trigger > reference
            and limit_price >= trigger
            and stop < trigger < tp1 < tp2
            and within_idx_daily_price_band(limit_price, reference)
        )
    if instruction == 'BUY_LIMIT_USER_SIZE':
        limit_price = _finite(semantics.get('stockbit_limit_price'), np.nan)
        return bool(
            np.isfinite(reference)
            and is_valid_idx_price(limit_price)
            and limit_price <= reference
            and stop < limit_price < tp1 < tp2
            and within_idx_daily_price_band(limit_price, reference)
        )
    return False

def _autopilot_gate_evaluation(
    row: Mapping[str, Any],
    cfg: ScanConfig,
    confirmed: bool,
    semantics: Mapping[str, Any],
) -> tuple[bool, float, list[str]]:
    """Return a fail-closed execution decision and auditable blocker list."""
    independent_verified = _truthy(row.get('independent_price_verified', False))
    independent_age = _finite(row.get('independent_price_age_days'), np.inf)
    independent_gap = _finite(row.get('independent_date_gap_days'), np.inf)
    source_tier = _safe_text(row.get('ohlcv_source_tier')).upper()
    absolute_age = _finite(row.get('absolute_data_age_days'), np.inf)
    completeness = _finite(row.get('data_completeness_score'), 0.0)
    confidence = _finite(row.get('execution_confidence_score'), _finite(row.get('execution_integrity_score'), 0.0))
    quality = _finite(row.get('quality_score'), 0.0)
    distance = _finite(row.get('distance_atr'), np.inf)
    stop_pct = _finite(row.get('stop_pct'), np.inf)
    rr1 = _finite(row.get('rr1'), 0.0)
    rr2 = _finite(row.get('rr2'), 0.0)
    adtv = _finite(row.get('adtv20_idr'), 0.0)
    zero_volume = _finite(row.get('zero_volume_ratio20'), _finite(row.get('zero_volume_ratio'), _finite(row.get('zero_vol20'), 0.0)))
    validation_score = _finite(row.get('validation_gate_score'), 0.0)
    validation_tier = _safe_text(row.get('validation_tier')).upper()
    fundamental_coverage = _finite(row.get('fundamental_coverage'), 0.0)
    regime = _safe_text(row.get('market_regime')).upper()
    sizing_ok = _safe_text(row.get('sizing_status')).upper() == 'OK' and int(_finite(row.get('suggested_lots'), 0)) >= 1
    fallback_budget = bool(
        sizing_ok
        and _finite(row.get('capital_required_idr'), np.inf) <= cfg.cash_on_hand_idr
        and _finite(row.get('max_loss_idr'), np.inf) <= cfg.account_size_idr * cfg.max_portfolio_risk_pct
    )
    budget_ok = _truthy(row.get('autopilot_budget_pass', fallback_budget))
    critical_context = bool(
        _safe_text(row.get('critical_blockers'))
        or _truthy(row.get('validation_critical_blocker', False))
        or _truthy(row.get('fundamental_critical_blocker', False))
        or _truthy(row.get('market_status_critical_blocker', False))
        or _truthy(row.get('news_critical_blocker', False))
        or _truthy(row.get('quote_critical_blocker', False))
    )
    checks = {
        'TRIGGER_NOT_CONFIRMED': confirmed,
        'INDEPENDENT_PRICE_REQUIRED': independent_verified,
        'INDEPENDENT_SESSION_MISMATCH': bool(
            independent_verified
            and independent_age <= cfg.max_independent_price_age_days
            and independent_gap <= cfg.max_independent_date_gap_days
        ),
        'DAILY_SOURCE_NOT_LIVE': bool(
            source_tier not in {'', 'CACHE_FALLBACK', 'UNAVAILABLE'}
            and absolute_age <= cfg.max_absolute_data_age_days
            and not _truthy(row.get('current_bar_incomplete', False))
        ),
        'PRICE_CONFLICT': not _truthy(row.get('quote_critical_blocker', False)),
        'DATA_COMPLETENESS': completeness >= max(cfg.min_data_completeness, cfg.min_autopilot_completeness),
        'EXECUTION_CONFIDENCE': confidence >= max(cfg.min_execution_confidence, cfg.min_autopilot_confidence),
        'TECHNICAL_QUALITY_OR_DISTANCE': quality >= cfg.execution_score and distance <= cfg.ready_distance_atr,
        'RISK_LEVELS_OR_SIZING': bool(sizing_ok and stop_pct <= cfg.max_stop_pct and rr1 >= cfg.min_rr1 and rr2 >= cfg.min_rr2),
        'LIQUIDITY': bool(adtv >= cfg.min_adtv_idr and zero_volume <= cfg.max_zero_volume_ratio),
        'CRITICAL_CONTEXT': not critical_context,
        'MARKET_REGIME': bool(regime not in {'', 'UNKNOWN'} and (cfg.allow_autopilot_risk_off or regime != 'RISK_OFF')),
        'VALIDATION': bool(
            not _truthy(row.get('validation_critical_blocker', False))
            and (validation_score >= cfg.min_autopilot_validation_score or validation_tier in {'USABLE', 'ROBUST'})
        ),
        'FUNDAMENTAL_COVERAGE': bool(
            not _truthy(row.get('fundamental_critical_blocker', False))
            and fundamental_coverage >= cfg.min_direct_fundamental_coverage
        ),
        'PORTFOLIO_BUDGET': budget_ok,
        'ORDER_SEMANTICS': _autopilot_order_is_valid(row, semantics),
    }
    failures = [name for name, passed in checks.items() if not passed]
    score = round(100.0 * sum(bool(value) for value in checks.values()) / len(checks), 1)
    return (bool(cfg.autopilot_enabled and not failures), score, failures)


def _signal_first_execution_evaluation(
    row: Mapping[str, Any],
    cfg: ScanConfig,
    confirmed: bool,
    semantics: Mapping[str, Any],
) -> tuple[float, list[str]]:
    """Evaluate non-account safety before asking the user to verify Stockbit.

    Signal-First may publish a technical signal without cash or sizing data, but
    it must never call that row execution-ready.  This gate deliberately keeps
    account budget out while retaining data, market, RR, liquidity, validation,
    fundamental-distress, news, and order-geometry safeguards.
    """
    independent_verified = _truthy(row.get('independent_price_verified', False))
    independent_age = _finite(row.get('independent_price_age_days'), np.inf)
    independent_gap = _finite(row.get('independent_date_gap_days'), np.inf)
    source_tier = _safe_text(row.get('ohlcv_source_tier')).upper()
    absolute_age = _finite(row.get('absolute_data_age_days'), np.inf)
    completeness = _finite(row.get('data_completeness_score'), 0.0)
    confidence = _finite(
        row.get('execution_confidence_score'),
        _finite(row.get('execution_integrity_score'), 0.0),
    )
    quality = _finite(row.get('quality_score'), 0.0)
    distance = _finite(row.get('distance_atr'), np.inf)
    stop_pct = _finite(row.get('stop_pct'), np.inf)
    rr1 = _finite(row.get('rr1'), 0.0)
    rr2 = _finite(row.get('rr2'), 0.0)
    adtv = _finite(row.get('adtv20_idr'), 0.0)
    zero_volume = _finite(
        row.get('zero_volume_ratio20'),
        _finite(row.get('zero_volume_ratio'), _finite(row.get('zero_vol20'), 0.0)),
    )
    validation_score = _finite(row.get('validation_gate_score'), 0.0)
    validation_tier = _safe_text(row.get('validation_tier')).upper()
    fundamental_coverage = _finite(row.get('fundamental_coverage'), 0.0)
    regime = _safe_text(row.get('market_regime')).upper()
    setup = _safe_text(row.get('setup'))
    critical_context = bool(
        _safe_text(row.get('critical_blockers'))
        or _truthy(row.get('validation_critical_blocker', False))
        or _truthy(row.get('fundamental_critical_blocker', False))
        or _truthy(row.get('market_status_critical_blocker', False))
        or _truthy(row.get('news_critical_blocker', False))
        or _truthy(row.get('quote_critical_blocker', False))
    )
    checks = {
        'TRIGGER_NOT_CONFIRMED': confirmed,
        'INDEPENDENT_PRICE_REQUIRED': independent_verified,
        'INDEPENDENT_SESSION_MISMATCH': bool(
            independent_verified
            and independent_age <= cfg.max_independent_price_age_days
            and independent_gap <= cfg.max_independent_date_gap_days
        ),
        'DAILY_SOURCE_NOT_LIVE': bool(
            source_tier not in {'', 'CACHE_FALLBACK', 'UNAVAILABLE'}
            and absolute_age <= cfg.max_absolute_data_age_days
            and not _truthy(row.get('current_bar_incomplete', False))
        ),
        'DATA_COMPLETENESS': completeness >= max(cfg.min_data_completeness, cfg.min_autopilot_completeness),
        'EXECUTION_CONFIDENCE': confidence >= max(cfg.min_execution_confidence, cfg.min_autopilot_confidence),
        'TECHNICAL_QUALITY_OR_DISTANCE': bool(
            quality >= cfg.execution_score
            and distance <= _ready_distance_atr_for_setup(setup, cfg)
        ),
        'RISK_LEVELS': bool(
            np.isfinite(stop_pct)
            and stop_pct <= cfg.max_stop_pct
            and rr1 >= cfg.min_rr1
            and rr2 >= cfg.min_rr2
        ),
        'LIQUIDITY': bool(adtv >= cfg.min_adtv_idr and zero_volume <= cfg.max_zero_volume_ratio),
        'CRITICAL_CONTEXT': not critical_context,
        'MARKET_REGIME': bool(
            regime not in {'', 'UNKNOWN'}
            and (cfg.allow_autopilot_risk_off or regime != 'RISK_OFF')
        ),
        'VALIDATION': bool(
            not _truthy(row.get('validation_critical_blocker', False))
            and (validation_score >= cfg.min_autopilot_validation_score or validation_tier in {'USABLE', 'ROBUST'})
        ),
        'FUNDAMENTAL_COVERAGE': bool(
            not _truthy(row.get('fundamental_critical_blocker', False))
            and fundamental_coverage >= cfg.min_direct_fundamental_coverage
        ),
        'MATERIAL_NEWS': not _truthy(row.get('news_critical_blocker', False)),
        'ORDER_SEMANTICS': _autopilot_order_is_valid(row, semantics),
    }
    failures = [name for name, passed in checks.items() if not passed]
    score = round(100.0 * sum(bool(value) for value in checks.values()) / len(checks), 1)
    return score, failures

def _resolve_autopilot_confluence(out: pd.DataFrame) -> pd.DataFrame:
    """Keep every setup row, but never create two live orders for one ticker."""
    if out.empty or 'ticker' not in out:
        return out
    result = out.copy()
    setup_series = result.get('setup', pd.Series('', index=result.index)).fillna('').astype(str)
    result['confluence_setup_count'] = result.groupby('ticker')['ticker'].transform('size').astype(int)
    setup_map = result.assign(_setup=setup_series).groupby('ticker')['_setup'].agg(lambda values: ' | '.join(dict.fromkeys(value for value in values if value))).to_dict()
    result['confluence_setups'] = result['ticker'].map(setup_map).fillna('')
    ready = result.index[result.get('autopilot_verified', pd.Series(False, index=result.index)).map(_truthy)].tolist()
    if not ready:
        return result
    sort_columns = [column for column in ('autopilot_score', 'quality_score', 'validation_gate_score', 'rr2') if column in result]
    for ticker, group in result.loc[ready].groupby('ticker', sort=False):
        if len(group) <= 1:
            continue
        ranked = group.sort_values(sort_columns, ascending=False, na_position='last') if sort_columns else group
        for idx in ranked.index[1:]:
            result.at[idx, 'autopilot_verified'] = False
            result.at[idx, 'autopilot_primary_setup'] = False
            result.at[idx, 'strict_execution_ready'] = False
            result.at[idx, 'status'] = 'READY_NOT_SELECTED'
            result.at[idx, 'execution_mode'] = 'CONFLUENCE_ALTERNATE'
            result.at[idx, 'order_instruction'] = 'DO_NOT_BUY'
            result.at[idx, 'stockbit_trigger_price'] = np.nan
            result.at[idx, 'stockbit_limit_price'] = np.nan
            result.at[idx, 'stockbit_order_price'] = np.nan
            result.at[idx, 'stockbit_order_lots'] = 0
            result.at[idx, 'automation_decision'] = 'ALTERNATE_SETUP_SAME_TICKER'
            failures = [value for value in _safe_text(result.at[idx, 'autopilot_blockers']).split(' | ') if value]
            if 'ALTERNATE_SETUP_SAME_TICKER' not in failures:
                failures.append('ALTERNATE_SETUP_SAME_TICKER')
            result.at[idx, 'autopilot_blockers'] = ' | '.join(failures)
            result.at[idx, 'primary_execution_blocker'] = 'ALTERNATE_SETUP_SAME_TICKER'
    return result


def finalize_execution_integrity(signals: pd.DataFrame, config: ScanConfig | None=None) -> pd.DataFrame:
    """Publish plans freely, but emit Stockbit tickets only after every hard gate."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = _finalize_execution_integrity_v440(signals, cfg)
    if 'analyst_pre_budget_ready' not in out:
        out = apply_analyst_fusion_gate(out, cfg)
    for column, default in {
        'stockbit_trigger_price': np.nan,
        'stockbit_limit_price': np.nan,
        'execution_timing': '',
        'autopilot_verified': False,
        'autopilot_score': 0.0,
        'autopilot_blockers': '',
        'autopilot_primary_setup': False,
        'stockbit_order_template': 'NONE',
        'stockbit_time_in_force': 'GFD',
        'broker_submission_mode': 'MANUAL_STOCKBIT',
        'broker_price_revalidation_required': True,
        'opening_gap_recheck_required': True,
        'manual_execution_candidate': False,
        'signal_execution_blockers': '',
        'proposed_order_instruction': 'NONE',
    }.items():
        if column not in out:
            out[column] = default
    for idx, row in out.iterrows():
        candidate = _truthy(row.get('analyst_pre_budget_ready', False))
        hard = _safe_text(row.get('analyst_hard_blockers'))
        pending_close = _truthy(row.get('pending_close', False))
        mode = _safe_text(row.get('analyst_order_mode'))
        action = _safe_text(row.get('action'))
        confirmed = mode == 'TRIGGER_CONFIRMED' or action in {'READY_TRIGGER', 'READY_LIMIT'}
        if pending_close:
            if candidate or mode != 'WATCH_ONLY':
                out.at[idx, 'status'] = 'PENDING_CLOSE'
                out.at[idx, 'order_instruction'] = 'DO_NOT_BUY'
                out.at[idx, 'primary_execution_blocker'] = 'DAILY_BAR_NOT_FINAL'
                out.at[idx, 'execution_timing'] = 'WAIT_EOD'
                out.at[idx, 'autopilot_verified'] = False
                out.at[idx, 'strict_execution_ready'] = False
                out.at[idx, 'stockbit_order_lots'] = 0
                out.at[idx, 'autopilot_blockers'] = 'DAILY_BAR_NOT_FINAL'
                out.at[idx, 'setup_state'] = 'PENDING_CLOSE'
                out.at[idx, 'account_order_state'] = 'NOT_EVALUATED'
            continue
        policy = _safe_text(getattr(cfg, 'execution_policy', 'SIGNAL_FIRST')).upper() or 'SIGNAL_FIRST'
        if policy != 'ACCOUNT_GUARDED':
            out.at[idx, 'execution_policy'] = policy
            out.at[idx, 'account_risk_gate_applied'] = False
            out.at[idx, 'autopilot_verified'] = False
            out.at[idx, 'strict_execution_ready'] = False
            out.at[idx, 'stockbit_order_lots'] = 0
            out.at[idx, 'sizing_is_informational'] = True
            out.at[idx, 'requires_stockbit_price_check'] = True
            out.at[idx, 'manual_execution_candidate'] = False
            out.at[idx, 'proposed_order_instruction'] = 'NONE'
            if candidate and not hard:
                out.at[idx, 'setup_state'] = 'SETUP_READY'
                out.at[idx, 'account_order_state'] = 'USER_MANAGED'
                if confirmed:
                    semantics = _execution_order_semantics(row)
                    readiness, safety_failures = _signal_first_execution_evaluation(
                        row, cfg, confirmed, semantics,
                    )
                    out.at[idx, 'execution_readiness_pct'] = readiness
                    out.at[idx, 'signal_execution_blockers'] = ' | '.join(safety_failures)
                    if not safety_failures:
                        proposed = _safe_text(semantics.get('order_instruction')) or 'MANUAL_ORDER'
                        for key, value in semantics.items():
                            out.at[idx, key] = value
                        out.at[idx, 'status'] = 'READY_FOR_STOCKBIT_VERIFY'
                        out.at[idx, 'execution_mode'] = 'MANUAL_STOCKBIT_VERIFY'
                        out.at[idx, 'manual_execution_candidate'] = True
                        out.at[idx, 'proposed_order_instruction'] = proposed
                        out.at[idx, 'order_instruction'] = f'VERIFY_STOCKBIT_THEN_{proposed}'
                        out.at[idx, 'execution_timing'] = 'VERIFY_LIVE_PRICE_SPREAD_GAP'
                        out.at[idx, 'stockbit_order_lots'] = 0
                        out.at[idx, 'automation_decision'] = 'NON_ACCOUNT_GATES_PASSED'
                        out.at[idx, 'execution_gate_failures'] = 'STOCKBIT_PRICE_SPREAD_GAP_CHECK'
                        out.at[idx, 'primary_execution_blocker'] = 'STOCKBIT_BROKER_REVALIDATION_REQUIRED'
                        out.at[idx, 'evidence_state'] = 'READY_FOR_MANUAL_BROKER_VERIFY'
                    else:
                        out.at[idx, 'status'] = 'SIGNAL_READY'
                        out.at[idx, 'execution_mode'] = 'SIGNAL_FIRST_RADAR'
                        out.at[idx, 'order_instruction'] = 'DO_NOT_BUY_YET'
                        out.at[idx, 'stockbit_trigger_price'] = np.nan
                        out.at[idx, 'stockbit_limit_price'] = np.nan
                        out.at[idx, 'stockbit_order_price'] = np.nan
                        out.at[idx, 'stockbit_order_lots'] = 0
                        out.at[idx, 'execution_timing'] = 'WAIT_REQUIRED_NON_ACCOUNT_GATES'
                        out.at[idx, 'automation_decision'] = 'SIGNAL_READY_GATES_PENDING'
                        out.at[idx, 'execution_gate_failures'] = ' | '.join(safety_failures)
                        out.at[idx, 'primary_execution_blocker'] = safety_failures[0]
                        out.at[idx, 'evidence_state'] = 'SIGNAL_ONLY_NOT_EXECUTION_READY'
                else:
                    out.at[idx, 'status'] = 'ENTRY_PLAN_READY'
                    out.at[idx, 'execution_mode'] = 'SIGNAL_FIRST_ENTRY_PLAN'
                    out.at[idx, 'order_instruction'] = 'WAIT_PRICE_AND_CONFIRM'
                    out.at[idx, 'stockbit_trigger_price'] = np.nan
                    out.at[idx, 'stockbit_limit_price'] = np.nan
                    out.at[idx, 'stockbit_order_price'] = np.nan
                    out.at[idx, 'execution_timing'] = 'WAIT_FOR_ZONE_OR_CONFIRMATION'
                    out.at[idx, 'automation_decision'] = 'VALID_PLAN_WAIT_CONFIRMATION'
                    out.at[idx, 'primary_execution_blocker'] = 'TRIGGER_NOT_CONFIRMED'
                    out.at[idx, 'evidence_state'] = 'ENTRY_PLAN_RESOLVED'
            elif hard:
                out.at[idx, 'setup_state'] = 'BLOCKED'
                out.at[idx, 'account_order_state'] = 'NOT_APPLICABLE'
                if _safe_text(out.at[idx, 'status']) not in {'REJECT', 'BLOCKED_CONTEXT'}:
                    out.at[idx, 'status'] = 'BLOCKED_CONTEXT'
                out.at[idx, 'execution_mode'] = 'BLOCKED_INVALID_OR_UNTRADEABLE'
                out.at[idx, 'order_instruction'] = 'DO_NOT_BUY'
                out.at[idx, 'primary_execution_blocker'] = 'HARD_INVALIDATION'
            else:
                out.at[idx, 'setup_state'] = 'WATCH_ONLY'
                out.at[idx, 'account_order_state'] = 'NOT_APPLICABLE'
                out.at[idx, 'execution_mode'] = 'WATCH'
                out.at[idx, 'order_instruction'] = 'DO_NOT_BUY'
            continue
        semantics = _execution_order_semantics(row)
        evaluation_row = row.to_dict()
        ticket_sizing: dict[str, Any] = {}
        ticket_price = _finite(semantics.get('stockbit_order_price'), np.nan)
        if confirmed and np.isfinite(ticket_price):
            ticket_sizing = size_stockbit_order(ticket_price, _finite(row.get('stop_loss'), np.nan), cfg)
            old_capital = max(0.0, _finite(row.get('capital_required_idr'), 0.0))
            old_loss = max(0.0, _finite(row.get('max_loss_idr'), 0.0))
            reserved_cash = _finite(row.get('portfolio_remaining_cash_idr'), np.nan)
            reserved_risk = _finite(row.get('portfolio_remaining_risk_idr'), np.nan)
            budget_pass = _truthy(row.get('autopilot_budget_pass', False))
            if budget_pass and np.isfinite(reserved_cash):
                budget_pass = _finite(ticket_sizing.get('capital_required_idr'), np.inf) <= reserved_cash + old_capital
            if budget_pass and np.isfinite(reserved_risk):
                budget_pass = _finite(ticket_sizing.get('max_loss_idr'), np.inf) <= reserved_risk + old_loss
            evaluation_row.update(ticket_sizing)
            if 'autopilot_budget_pass' in row:
                evaluation_row['autopilot_budget_pass'] = budget_pass
        verified, autopilot_score, autopilot_failures = _autopilot_gate_evaluation(evaluation_row, cfg, confirmed, semantics)
        out.at[idx, 'autopilot_verified'] = verified
        out.at[idx, 'autopilot_score'] = autopilot_score
        out.at[idx, 'autopilot_blockers'] = ' | '.join(autopilot_failures)
        out.at[idx, 'strict_execution_ready'] = verified
        out.at[idx, 'strict_execution_gate_failures'] = ' | '.join(autopilot_failures)
        out.at[idx, 'strict_primary_execution_blocker'] = autopilot_failures[0] if autopilot_failures else 'NONE'
        # Independent public data validates the scanner input, not the live
        # broker queue.  Every ticket still needs a final Stockbit price/spread
        # check immediately before submission.
        out.at[idx, 'requires_stockbit_price_check'] = True
        if candidate and (not hard) and verified:
            out.at[idx, 'status'] = 'EXECUTION_READY'
            out.at[idx, 'execution_mode'] = 'AUTOPILOT_VERIFIED'
            for key, value in semantics.items():
                out.at[idx, key] = value
            for key, value in ticket_sizing.items():
                out.at[idx, key] = value
            out.at[idx, 'stockbit_order_lots'] = int(_finite(ticket_sizing.get('suggested_lots'), 0))
            out.at[idx, 'automation_decision'] = 'AUTOPILOT_VERIFIED'
            out.at[idx, 'execution_gate_failures'] = ''
            out.at[idx, 'primary_execution_blocker'] = 'NONE'
            out.at[idx, 'execution_readiness_pct'] = 100.0
            out.at[idx, 'evidence_state'] = 'AUTOPILOT_VERIFIED'
            out.at[idx, 'portfolio_selected'] = True
            out.at[idx, 'account_risk_gate_applied'] = True
            out.at[idx, 'autopilot_primary_setup'] = True
        elif candidate and (not hard):
            critical_failures = {'PRICE_CONFLICT', 'CRITICAL_CONTEXT', 'MARKET_REGIME'}
            budget_detail = _safe_text(row.get('autopilot_budget_blockers'))
            if confirmed and 'ALTERNATE_SETUP_SAME_TICKER' in budget_detail:
                status, execution_mode, decision = ('READY_NOT_SELECTED', 'CONFLUENCE_ALTERNATE', 'ALTERNATE_SETUP_SAME_TICKER')
            elif not confirmed:
                status, execution_mode, decision = ('ENTRY_PLAN_READY', 'ENTRY_PLAN', 'VALID_PLAN_WAIT_CONFIRMATION')
            elif any(failure in critical_failures for failure in autopilot_failures):
                status, execution_mode, decision = ('BLOCKED_CONTEXT', 'AUTOPILOT_BLOCKED', 'DO_NOT_BUY')
            elif 'INDEPENDENT_PRICE_REQUIRED' in autopilot_failures or 'INDEPENDENT_SESSION_MISMATCH' in autopilot_failures:
                status, execution_mode, decision = ('READY_FOR_PRICE_VERIFY', 'PRICE_VERIFY', 'VERIFY_INDEPENDENT_PRICE')
            else:
                status, execution_mode, decision = ('PENDING_DATA', 'AUTOPILOT_PENDING', 'WAIT_REQUIRED_GATES')
            out.at[idx, 'status'] = status
            out.at[idx, 'execution_mode'] = execution_mode
            out.at[idx, 'order_instruction'] = 'WAIT_PRICE_AND_CONFIRM' if status in {'ENTRY_PLAN_READY', 'READY_FOR_PRICE_VERIFY'} else 'DO_NOT_BUY'
            out.at[idx, 'stockbit_trigger_price'] = np.nan
            out.at[idx, 'stockbit_limit_price'] = np.nan
            out.at[idx, 'stockbit_order_price'] = np.nan
            out.at[idx, 'stockbit_order_lots'] = 0
            out.at[idx, 'execution_timing'] = 'WAIT_FOR_ZONE_OR_CONFIRMATION'
            out.at[idx, 'automation_decision'] = decision
            out.at[idx, 'execution_gate_failures'] = ' | '.join(autopilot_failures)
            out.at[idx, 'primary_execution_blocker'] = autopilot_failures[0] if autopilot_failures else 'NONE'
            out.at[idx, 'execution_readiness_pct'] = autopilot_score
            out.at[idx, 'evidence_state'] = 'ENTRY_PLAN_RESOLVED' if not confirmed else 'AUTOPILOT_GATES_PENDING'
            out.at[idx, 'portfolio_selected'] = True
            out.at[idx, 'account_risk_gate_applied'] = True
        elif hard:
            if _safe_text(out.at[idx, 'status']) not in {'REJECT', 'BLOCKED_CONTEXT'}:
                out.at[idx, 'status'] = 'BLOCKED_CONTEXT'
            out.at[idx, 'execution_mode'] = 'BLOCKED_INVALID_OR_UNTRADEABLE'
            out.at[idx, 'autopilot_verified'] = False
            out.at[idx, 'order_instruction'] = 'DO_NOT_BUY'
            out.at[idx, 'stockbit_order_lots'] = 0
            out.at[idx, 'primary_execution_blocker'] = 'HARD_INVALIDATION'
        else:
            out.at[idx, 'execution_mode'] = 'WATCH'
            out.at[idx, 'autopilot_verified'] = False
            out.at[idx, 'order_instruction'] = 'DO_NOT_BUY'
            out.at[idx, 'stockbit_order_lots'] = 0
            if _safe_text(out.at[idx, 'status']) == 'EXECUTION_READY':
                out.at[idx, 'status'] = 'WATCHLIST_ENTRY'
                out.at[idx, 'primary_execution_blocker'] = autopilot_failures[0] if autopilot_failures else 'SETUP_NOT_ACTIONABLE'
    out = _resolve_autopilot_confluence(out)
    # Release invariant: no upstream/legacy path may leave an executable label
    # without the corresponding verified hard-gate result.
    unsafe_execution = out['status'].eq('EXECUTION_READY') & ~out['autopilot_verified'].map(_truthy)
    if unsafe_execution.any():
        out.loc[unsafe_execution, 'status'] = 'SIGNAL_READY'
        out.loc[unsafe_execution, 'execution_mode'] = 'FAIL_SAFE_SIGNAL_ONLY'
        out.loc[unsafe_execution, 'order_instruction'] = 'DO_NOT_BUY_YET'
        out.loc[unsafe_execution, 'stockbit_trigger_price'] = np.nan
        out.loc[unsafe_execution, 'stockbit_limit_price'] = np.nan
        out.loc[unsafe_execution, 'stockbit_order_price'] = np.nan
        out.loc[unsafe_execution, 'stockbit_order_lots'] = 0
        out.loc[unsafe_execution, 'manual_execution_candidate'] = False
        out.loc[unsafe_execution, 'strict_execution_ready'] = False
        out.loc[unsafe_execution, 'primary_execution_blocker'] = 'EXECUTION_INVARIANT_NOT_VERIFIED'
        out.loc[unsafe_execution, 'signal_execution_blockers'] = 'EXECUTION_INVARIANT_NOT_VERIFIED'
        out.loc[unsafe_execution, 'automation_decision'] = 'FAIL_SAFE_DOWNGRADE'
    out['status_rank'] = out['status'].map(STATUS_ORDER).fillna(99)
    return out

def attach_position_sizing(signals: pd.DataFrame, config: ScanConfig | None=None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    sized_rows: list[dict[str, float | int | str]] = []
    for _, row in out.iterrows():
        sized_rows.append(size_stockbit_order(_finite(row.get('entry'), np.nan), _finite(row.get('stop_loss'), np.nan), cfg))
    sizing = pd.DataFrame(sized_rows, index=out.index)
    for column in sizing.columns:
        out[column] = sizing[column]
    out['sizing_is_informational'] = True
    out['account_risk_gate_applied'] = False
    return out

def _analyst_setup_mode(row: Mapping[str, Any], cfg: ScanConfig) -> tuple[str, list[str]]:
    """Map every structurally actionable detector state to an order-plan mode."""
    setup = _safe_text(row.get('setup'))
    action = _safe_text(row.get('action'))
    evidence = _safe_text(row.get('evidence'))
    quality = _finite(row.get('quality_score'), 0.0)
    distance = _finite(row.get('distance_atr'), 999.0)
    smart = _finite(row.get('silent_accumulation_score'), 0.0)
    if action in {'READY_TRIGGER', 'READY_LIMIT'} or _truthy(row.get('technical_setup_ready', False)):
        return ('TRIGGER_CONFIRMED', ['Trigger detector sudah lengkap'])
    if setup == 'PULLBACK_CONTINUATION' and action in {'WAIT_PULLBACK_CONFIRMATION', 'WAIT_STRICT_FLOW_CONFIRMATION'}:
        if quality >= 70.0 and distance <= 1.25:
            return ('LIMIT_PULLBACK_ZONE', ['Trend dan zona pullback valid; confirmation menjadi syarat eksekusi di chart'])
    if setup == 'BREAKOUT_RETEST' and action in {'WAIT_RETEST', 'WAIT_CURRENT_RETEST_CONFIRMATION'}:
        retest_seen = 'Retest level breakout terdeteksi' in evidence
        if quality >= 70.0 and distance <= 1.2 and retest_seen:
            return ('LIMIT_BREAKOUT_RETEST', ['Breakout valid dan retest telah terobservasi'])
    if setup == 'REVERSAL_ACCUMULATION' and action in {'WAIT_RETEST', 'WAIT_HIGHER_LOW_AND_FLOW'}:
        structure_seen = 'CHOCH/BOS bullish terkonfirmasi' in evidence
        sweep_seen = 'Sell-side liquidity sweep' in evidence
        if quality >= 70.0 and distance <= 1.0 and structure_seen and sweep_seen and (smart >= 45.0):
            return ('LIMIT_CHOCH_RETEST', ['Liquidity sweep dan CHOCH/BOS valid; flow lemah hanya menjadi warning'])
    if setup == 'UNICORN_SNIPER_ICT' and action in {'WAIT_FVG_RETRACE', 'WAIT_STRICT_UNICORN_CONFLUENCE'}:
        core_smc = all((token in evidence for token in ('Sell-side liquidity sweep', 'Bullish BOS dengan displacement', 'Bullish FVG valid')))
        if quality >= 68.0 and distance <= 1.0 and core_smc:
            return ('LIMIT_FVG_RETRACE', ['Sweep–BOS–FVG valid; strict OB/discount/volume menjadi quality warning'])
    return ('WATCH_ONLY', [])

def _signal_first_hard_blockers(row: Mapping[str, Any], cfg: ScanConfig) -> list[str]:
    """Only invalid structure, unusable data, or untradeable status may suppress a signal."""
    hard: list[str] = []
    if not _truthy(row.get('detected', True)) or _truthy(row.get('invalidated', False)):
        hard.append('Setup tidak terdeteksi atau sudah invalid')
    action = _safe_text(row.get('action'))
    if action in {'TOO_EXTENDED_WAIT_NEW_BASE', 'NO_SETUP', 'WAIT_CHOCH'}:
        hard.append('Struktur entry belum valid')
    true_invalid_tokens = ('Riwayat hanya', 'Data absolut', 'Masa berlaku setup sudah habis', 'Harga sudah menutup jauh di bawah zona entry', 'Level entry/SL tidak valid', 'fraksi harga IDX', 'Urutan SL < entry < TP1 < TP2', 'Breakout ditolak', 'Zona berumur', 'Jarak ke zona tidak dapat dihitung')
    for item in _pipe_parts(row.get('blockers')):
        if any((token in item for token in true_invalid_tokens)):
            hard.append(item)
    source_tier = _safe_text(row.get('ohlcv_source_tier')).upper()
    if source_tier in {'UNAVAILABLE', ''}:
        hard.append('OHLCV tidak tersedia')
    if _finite(row.get('absolute_data_age_days'), 999.0) > cfg.max_absolute_data_age_days:
        hard.append('OHLCV terlalu lama')
    if _truthy(row.get('pending_close', False)):
        hard.append('Daily candle belum final')
    if _truthy(row.get('market_status_critical_blocker', False)):
        hard.append('Suspensi/FCA/status perdagangan negatif')
    if _truthy(row.get('quote_critical_blocker', False)):
        hard.append('Konflik quote/candle; setup tidak dapat divalidasi')
    return list(dict.fromkeys(hard))

def apply_analyst_fusion_gate(signals: pd.DataFrame, config: ScanConfig | None=None, minimum_score: float=68.0) -> pd.DataFrame:
    """Calibrated signal-first core gate without account/evidence overblocking."""
    cfg = config or ScanConfig()
    if signals.empty:
        return signals.copy()
    out = signals.copy()
    defaults = {'account_risk_gate_applied': False, 'signal_risk_grade': 'NORMAL', 'signal_risk_warnings': '', 'setup_valid_signal': False, 'analyst_order_mode': 'WATCH_ONLY', 'analyst_fusion_score': 0.0, 'analyst_pre_budget_ready': False, 'analyst_hard_blockers': '', 'analyst_decision_basis': '', 'analyst_candidate_reason': 'WATCH_ONLY', 'requires_stockbit_price_check': True}
    for column, default in defaults.items():
        if column not in out:
            out[column] = default
    mode_scores = {'TRIGGER_CONFIRMED': 100.0, 'LIMIT_PULLBACK_ZONE': 88.0, 'LIMIT_BREAKOUT_RETEST': 88.0, 'LIMIT_CHOCH_RETEST': 84.0, 'LIMIT_FVG_RETRACE': 82.0, 'WATCH_ONLY': 25.0}
    for idx, row in out.iterrows():
        mode, reasons = _analyst_setup_mode(row, cfg)
        hard = _signal_first_hard_blockers(row, cfg)
        quality = np.clip(_finite(row.get('quality_score'), 0.0), 0.0, 100.0)
        smart = np.clip(_finite(row.get('silent_accumulation_score'), 50.0), 0.0, 100.0)
        distance = _finite(row.get('distance_atr'), 999.0)
        distance_score = max(0.0, 100.0 - 30.0 * max(0.0, distance)) if np.isfinite(distance) else 0.0
        score = round(quality * 0.52 + mode_scores.get(mode, 25.0) * 0.28 + distance_score * 0.12 + smart * 0.08, 1)
        candidate = bool(mode != 'WATCH_ONLY' and (not hard) and (score >= minimum_score))
        risk_grade, risk_flags = _risk_disclosure(row, cfg)
        basis = reasons + [f'Calibrated signal score {score:.1f}/100']
        if risk_flags:
            basis.append('Risk disclosure: ' + '; '.join(risk_flags))
        if hard:
            basis.append('Hard invalidation: ' + '; '.join(hard))
        out.at[idx, 'analyst_order_mode'] = mode
        out.at[idx, 'analyst_fusion_score'] = score
        out.at[idx, 'analyst_pre_budget_ready'] = candidate
        out.at[idx, 'setup_valid_signal'] = candidate
        out.at[idx, 'analyst_hard_blockers'] = ' • '.join(hard)
        out.at[idx, 'analyst_decision_basis'] = ' • '.join(basis)
        out.at[idx, 'analyst_candidate_reason'] = 'PASS' if candidate else 'HARD_BLOCK' if hard else 'WATCH_ONLY'
        out.at[idx, 'account_risk_gate_applied'] = False
        out.at[idx, 'signal_risk_grade'] = risk_grade
        out.at[idx, 'signal_risk_warnings'] = ' • '.join(risk_flags)
        out.at[idx, 'requires_stockbit_price_check'] = True
    return out
# Stable public helper API for scanner_specialty.  The specialty module no
# longer imports private implementation names, making the two-module contract
# explicit and safer to refactor.
clean_ohlcv = _clean_ohlcv
extract_download_batch = _extract_batch
safe_number = _finite
jakarta_timestamp = _jakarta_timestamp
pipe_parts = _pipe_parts
price_structure_target_pair = _price_structure_target_pair
read_csv_input = _read_csv
safe_text = _safe_text
silent_accumulation_metrics = _silent_accumulation_metrics
truthy = _truthy

_SPECIALTY_EXPORTS = {
    'download_intraday_ohlcv', 'specialty_intraday_shortlist', 'scan_sniper_entries',
    'scan_bsjp_candidates', 'scan_bpjs_candidates', 'scan_multibagger_candidates',
    'allocate_multibagger_capital',
    'scan_ara_hunter_candidates', 'build_specialty_screens', 'build_daily_opportunity_board', 'build_profit_order_builder',
    'parse_project_management_csv', 'collect_automatic_forward_quality', 'merge_project_management_reviews', 'parse_orderbook_snapshot_csv', 'apply_ara_external_confirmation',
    '_intraday_metrics',
}

def __getattr__(name: str) -> Any:
    """Load specialty exports lazily so either module can be imported first."""
    if name in _SPECIALTY_EXPORTS:
        import importlib
        module = importlib.import_module('scanner_specialty')
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = '6.6.7-fundamental-resilience-batch2'
__all__ = sorted({name for name in globals() if not name.startswith('_')} | {name for name in _SPECIALTY_EXPORTS if not name.startswith('_')})

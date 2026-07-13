from __future__ import annotations

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


def near_upper_auto_rejection(previous_close: float, close: float, high: float) -> bool:
    if previous_close <= 0:
        return False
    daily_return = close / previous_close - 1.0
    locked_at_high = abs(high - close) <= idx_tick_size(close) * 0.51
    return bool(locked_at_high and daily_return >= 0.90 * idx_ara_pct(previous_close))

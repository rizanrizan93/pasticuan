from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import timedelta
from typing import Any
import json

import numpy as np
import pandas as pd
import streamlit as st

from scanner import idx_tick_size, make_signal_chart, normalize_idx_ticker
from time_cycle import TimeCycleConfig, analyze_time_cycle, make_time_cycle_chart

DASHBOARD_VERSION = "7.6.4-evidence-acquisition"
CORE_STRATEGIES = {
    "PULLBACK_CONTINUATION",
    "BREAKOUT_RETEST",
    "REVERSAL_ACCUMULATION",
    "SNIPER",
    "UNICORN_SNIPER_ICT",
}
ACTIONABLE_TRADE_ACTIONS = {
    "BUY_ON_CONFIRMED_TRIGGER",
    "PREPARE_BUY_WAIT_TRIGGER",
    "BUY_ON_RETEST_OR_BUY_STOP_CONFIRMATION",
}
PLAN_DEPENDENT_ACTIONS = ACTIONABLE_TRADE_ACTIONS | {
    "READY_TRIGGER", "WAIT_RETEST", "WAIT_PULLBACK_CONFIRMATION",
    "WAIT_CURRENT_RETEST_CONFIRMATION", "WAIT_STRICT_FLOW_CONFIRMATION",
    "WAIT_HIGHER_LOW_AND_FLOW", "WAIT_STRICT_CONFLUENCE", "WAIT_RETRACE",
    "WAIT_FVG_RETRACE", "WAIT_CHOCH", "WAIT_FOR_PRICE_ZONE_OR_TRIGGER",
}

EOFF_STRONG_LEVELS = {"STRONG", "VERY_STRONG"}
INVALID_BEST_BUY_DATE_TOKENS = {
    "", "N/A", "NA", "NONE", "NAN", "NULL", "UNKNOWN", "UNAVAILABLE",
    "BELUM VALID", "BELUM TERSEDIA", "NO_VALID_DATE", "NO DATE", "—", "-",
}
UNSAFE_RANK_ACTIONS = {"AVOID_NEW_BUY", "NO_ALLOCATION"}
HARD_EXCLUSION_STATES = {
    "REJECT", "BLOCKED", "NO_ALLOCATION", "NOT_QUALIFIED",
    "WAIT_FOR_DATA", "EXPIRED_DATA", "NON_SYARIAH",
}
DECISION_STATUS_FIELDS = (
    "decision", "quick_buy_action", "allocation_action", "review_action",
    "status", "setup_status", "decision_state", "technical_entry_state",
    "research_recommendation_status", "multibagger_status",
    "multibagger_candidate_type", "multibagger_scoring_state",
    "capital_tier", "eligibility_status",
)
FRESHNESS_STATUS_FIELDS = (
    "database_read_state", "fundamental_database_state", "forward_database_state",
    "statement_age_state", "price_freshness_state", "data_freshness_state",
    "ohlcv_freshness_state", "market_data_state",
)

ACTION_PRIORITY = {
    "BUY_ON_CONFIRMED_TRIGGER": 0,
    "PREPARE_BUY_WAIT_TRIGGER": 1,
    "BUY_ON_RETEST_OR_BUY_STOP_CONFIRMATION": 1,
    "WAIT_FINAL_EOD_CONFIRMATION": 2,
    "WAIT_FOR_PRICE_ZONE_OR_TRIGGER": 3,
    "WAIT_FOR_DATE": 4,
    "ACCUMULATE_GRADUALLY": 4,
    "WAIT_FOR_EVIDENCE": 5,
    "RESEARCH_AND_WAIT": 6,
    "RECALCULATE_WINDOW": 7,
    "AVOID_NEW_BUY": 9,
    "NO_ALLOCATION": 9,
}


def _num(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _clip(value: Any, default: float = 50.0) -> float:
    return float(max(0.0, min(100.0, _num(value, default))))


def _fmt_price(value: Any) -> str:
    number = _num(value)
    if not np.isfinite(number):
        return "—"
    return f"Rp{number:,.0f}".replace(",", ".")


def _format_entry_zone(row: Mapping[str, Any]) -> str:
    state = _text(row.get("entry_zone_state")).upper()
    low = _num(row.get("entry_low"))
    high = _num(row.get("entry_high"))
    execution = _num(row.get("execution_price"))
    if state == "RANGE" and np.isfinite(low) and np.isfinite(high) and low < high:
        return f"{_fmt_price(low)} – {_fmt_price(high)}"
    if state == "SINGLE_REFERENCE":
        reference = execution if np.isfinite(execution) else low if np.isfinite(low) else high
        return f"Entry {_fmt_price(reference)}" if np.isfinite(reference) else "—"
    return "—"


def _first_finite(row: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = _num(row.get(key), np.nan)
        if np.isfinite(value):
            return value
    return np.nan


def _entry_zone(row: Mapping[str, Any]) -> tuple[float, float]:
    low = _num(row.get("best_buy_entry_low"))
    high = _num(row.get("best_buy_entry_high"))
    if not np.isfinite(low):
        low = _num(row.get("entry_low", row.get("entry")))
    if not np.isfinite(high):
        high = _num(row.get("entry_high", row.get("entry")))
    if np.isfinite(low) and np.isfinite(high) and low > high:
        low, high = high, low
    return low, high


def _derived_rr(execution_price: float, stop: float, target: float) -> float:
    risk = execution_price - stop
    if not all(np.isfinite(value) for value in (execution_price, stop, target)) or risk <= 0:
        return np.nan
    return float((target - execution_price) / risk)


def _entry_zone_state(low: float, high: float, reference: float) -> str:
    if np.isfinite(low) and np.isfinite(high):
        if low > high:
            return "INVALID_RANGE"
        if abs(high - low) <= 1e-9:
            return "SINGLE_REFERENCE"
        return "RANGE"
    if np.isfinite(reference):
        return "SINGLE_REFERENCE"
    return "MISSING"


def _plan_candidate(row: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Resolve one atomic trade-plan contract without mixing price semantics.

    Detector output contains two different concepts:
    - ``entry`` / entry zone: planned fill price used to build SL, TP and RR;
    - ``trigger``: confirmation/reclaim level, which is not automatically the
      order price for a pullback or retest plan.

    Older dashboards treated every confirmation level as the execution price.
    That produced rows such as entry 1,860, trigger 1,975 and TP1 1,945 even
    though targets and stored RR had been generated from the 1,860 entry.
    This resolver keeps execution price, order trigger and confirmation level
    separate, then recomputes RR from the actual execution reference.
    """
    best = source == "BEST_BUY"
    if best:
        low = _num(row.get("best_buy_entry_low"))
        high = _num(row.get("best_buy_entry_high"))
        raw_entry = np.nan
        confirmation = _num(row.get("best_buy_trigger"))
        order_trigger = confirmation
        stop = _num(row.get("best_buy_stop_loss"))
        tp1 = _num(row.get("best_buy_tp1"))
        tp2 = _num(row.get("best_buy_tp2"))
        order_plan = _text(row.get("best_buy_order_plan")).upper()
        mode = "BEST_BUY_TRIGGER" if order_plan and order_plan != "NO_ORDER" else "BEST_BUY_RESEARCH"
        execution_price = confirmation
        entry_type = order_plan or "BEST_BUY"
        action = _text(row.get("quick_buy_action"))
    else:
        low = _num(row.get("entry_low"))
        high = _num(row.get("entry_high"))
        raw_entry = _num(row.get("entry"))
        entry_type = _text(row.get("entry_type")).upper()
        action = (
            _text(row.get("action"))
            or _text(row.get("setup_action"))
            or _text(row.get("next_action"))
        ).upper()
        order_instruction = _text(row.get("order_instruction")).upper()
        raw_trigger = _first_finite(row, "trigger", "trigger_price")
        stockbit_trigger = _num(row.get("stockbit_trigger_price"))
        stockbit_limit = _num(row.get("stockbit_limit_price"))
        stockbit_order = _num(row.get("stockbit_order_price"))
        buy_stop_style = bool(
            "BUY_STOP" in entry_type
            or "BUY_LIT" in order_instruction
            or (
                np.isfinite(stockbit_trigger)
                and action in {"READY_TRIGGER", "BUY_ON_CONFIRMED_TRIGGER"}
            )
        )
        if buy_stop_style:
            mode = "BUY_STOP"
            order_trigger = stockbit_trigger if np.isfinite(stockbit_trigger) else raw_trigger
            execution_price = next(
                (
                    value for value in (stockbit_order, stockbit_limit, raw_entry, order_trigger)
                    if np.isfinite(value)
                ),
                np.nan,
            )
        else:
            mode = "LIMIT_RETEST"
            order_trigger = np.nan
            execution_price = raw_entry
        confirmation = raw_trigger
        stop = _num(row.get("stop_loss"))
        tp1 = _num(row.get("tp1"))
        tp2 = _num(row.get("tp2"))

    if not np.isfinite(low) and np.isfinite(raw_entry):
        low = raw_entry
    if not np.isfinite(high) and np.isfinite(raw_entry):
        high = raw_entry
    if np.isfinite(low) and np.isfinite(high) and low > high:
        low, high = high, low
    if not np.isfinite(execution_price) and np.isfinite(low) and np.isfinite(high):
        execution_price = (low + high) / 2.0

    rr1 = _derived_rr(execution_price, stop, tp1)
    rr2 = _derived_rr(execution_price, stop, tp2)
    zone_state = _entry_zone_state(low, high, execution_price)
    complete = all(
        np.isfinite(value) and value > 0
        for value in (execution_price, stop, tp1, tp2, rr1, rr2)
    )
    target_geometry = bool(
        complete and stop < execution_price < tp1 < tp2
    )
    zone_geometry = bool(
        zone_state in {"RANGE", "SINGLE_REFERENCE"}
        and (
            not (np.isfinite(low) and np.isfinite(high))
            or low - 1e-9 <= execution_price <= high + max(1.0, idx_tick_size(execution_price))
            or mode == "BUY_STOP"
            or best
        )
    )
    trigger_geometry = True
    trigger_issue = ""
    if mode in {"BUY_STOP", "BEST_BUY_TRIGGER"}:
        trigger_geometry = bool(
            np.isfinite(order_trigger)
            and stop < order_trigger <= execution_price < tp1
        )
        if not trigger_geometry:
            trigger_issue = "ORDER_TRIGGER_GEOMETRY"
    elif np.isfinite(confirmation):
        # A future reclaim/confirmation cannot sit at or beyond the first
        # target.  If it does, the detector's sequence is stale/ambiguous and
        # the row must wait for a rebuilt plan instead of displaying nonsense.
        trigger_geometry = bool(stop < confirmation < tp1)
        if not trigger_geometry:
            trigger_issue = "CONFIRMATION_NOT_BELOW_TP1"

    geometry_valid = bool(target_geometry and zone_geometry and trigger_geometry)
    minimum_rr1 = max(0.0, _num(row.get("entry_plan_min_rr1"), 1.80))
    minimum_rr2 = max(minimum_rr1, _num(row.get("entry_plan_min_rr2"), 2.70))
    rr_valid = bool(
        geometry_valid
        and rr1 >= minimum_rr1 - 1e-9
        and rr2 >= minimum_rr2 - 1e-9
        and rr2 > rr1
    )
    if rr_valid:
        state = f"VALID_{source}_PLAN"
    elif not complete:
        state = f"INCOMPLETE_{source}_PLAN"
    elif trigger_issue:
        state = f"INVALID_{source}_{trigger_issue}"
    elif not target_geometry:
        state = f"INVALID_{source}_TARGET_GEOMETRY"
    elif not zone_geometry:
        state = f"INVALID_{source}_ENTRY_ZONE"
    else:
        state = f"RR_BELOW_MIN_{source}"

    populated = sum(
        np.isfinite(value)
        for value in (
            low, high, execution_price, confirmation, order_trigger,
            stop, tp1, tp2, rr1, rr2,
        )
    )
    plan_id = "|".join(
        str(value) for value in (
            source, mode,
            round(execution_price, 6) if np.isfinite(execution_price) else "NA",
            round(stop, 6) if np.isfinite(stop) else "NA",
            round(tp1, 6) if np.isfinite(tp1) else "NA",
            round(tp2, 6) if np.isfinite(tp2) else "NA",
        )
    )
    return {
        "entry_low": low,
        "entry_high": high,
        "entry_zone_state": zone_state,
        "execution_price": execution_price,
        "trigger": order_trigger,
        "confirmation_level": confirmation,
        "stop_loss": stop,
        "tp1": tp1,
        "tp2": tp2,
        "rr1": rr1,
        "rr2": rr2,
        "execution_plan_source": source,
        "execution_plan_mode": mode,
        "execution_plan_state": state,
        "execution_plan_valid": rr_valid,
        "execution_plan_id": plan_id,
        "_plan_populated_fields": populated,
    }


def _resolve_trade_plan(row: Mapping[str, Any]) -> dict[str, Any]:
    best = _plan_candidate(row, "BEST_BUY")
    base = _plan_candidate(row, "BASE")
    if best["execution_plan_valid"]:
        return best
    if base["execution_plan_valid"]:
        return base

    # Keep raw levels only in diagnostic_* fields.  The main recommendation
    # columns are blanked so a non-actionable plan cannot look like an order.
    selected = (
        best
        if best["_plan_populated_fields"] > base["_plan_populated_fields"]
        else base
    )
    diagnostic = dict(selected)
    return {
        "entry_low": np.nan,
        "entry_high": np.nan,
        "entry_zone_state": "MISSING",
        "execution_price": np.nan,
        "trigger": np.nan,
        "confirmation_level": np.nan,
        "stop_loss": np.nan,
        "tp1": np.nan,
        "tp2": np.nan,
        "rr1": np.nan,
        "rr2": np.nan,
        "execution_plan_source": selected["execution_plan_source"],
        "execution_plan_mode": "NO_VALID_PLAN",
        "execution_plan_state": (
            f"NO_VALID_PLAN:{best['execution_plan_state']}|{base['execution_plan_state']}"
        ),
        "execution_plan_valid": False,
        "execution_plan_id": "",
        "diagnostic_entry_low": diagnostic.get("entry_low"),
        "diagnostic_entry_high": diagnostic.get("entry_high"),
        "diagnostic_execution_price": diagnostic.get("execution_price"),
        "diagnostic_confirmation_level": diagnostic.get("confirmation_level"),
        "diagnostic_trigger": diagnostic.get("trigger"),
        "diagnostic_stop_loss": diagnostic.get("stop_loss"),
        "diagnostic_tp1": diagnostic.get("tp1"),
        "diagnostic_tp2": diagnostic.get("tp2"),
        "diagnostic_rr1": diagnostic.get("rr1"),
        "diagnostic_rr2": diagnostic.get("rr2"),
    }


def resolve_trade_plan(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical atomic trade-plan contract for external surfaces.

    Recommendation surfaces must use this resolver instead of independently
    combining entry, trigger, stop and target columns.
    """
    return _resolve_trade_plan(row)


def _guard_actionable_plan(action: str, plan: Mapping[str, Any]) -> tuple[str, str]:
    if bool(plan.get("execution_plan_valid", False)):
        return action, ""
    normalized = _normalize_state(action)
    warning = (
        "Trade plan tidak ditampilkan karena entry/trigger/SL/TP/RR tidak membentuk satu kontrak eksekusi yang valid."
    )
    diagnostic_values = (
        plan.get("diagnostic_execution_price"),
        plan.get("diagnostic_stop_loss"),
        plan.get("diagnostic_tp1"),
        plan.get("diagnostic_tp2"),
    )
    attempted_plan = sum(bool(np.isfinite(_num(value, np.nan))) for value in diagnostic_values) >= 3
    if normalized in PLAN_DEPENDENT_ACTIONS or "ENTRY_PLAN" in normalized or attempted_plan:
        return "WAIT_FOR_VALID_PLAN", warning
    return action, warning


def _safe_plan_status(source_status: Any, plan_valid: bool) -> str:
    status = _text(source_status)
    if plan_valid:
        return status
    normalized = _normalize_state(status)
    if (
        normalized in {
            "EXECUTION_READY", "READY_FOR_STOCKBIT_VERIFY", "SIGNAL_READY",
            "ENTRY_PLAN_READY", "READY_FOR_PRICE_VERIFY",
        }
        or normalized.startswith("READY_")
    ):
        return "WAIT_FOR_VALID_PLAN"
    return status or "WATCH"


def _quick_action(row: Mapping[str, Any], fallback: str = "REVIEW") -> str:
    action = _text(row.get("quick_buy_action"))
    if action:
        return action
    return _text(row.get("allocation_action")) or _text(row.get("next_action")) or fallback


def _quick_score(row: Mapping[str, Any]) -> float:
    raw = _num(row.get("best_buy_score"), np.nan)
    if np.isfinite(raw) and raw > 0:
        return _clip(raw)
    timing = _num(row.get("time_cycle_alignment_score"), np.nan)
    if not np.isfinite(timing):
        timing = _num(row.get("multibagger_time_cycle_score", row.get("time_cycle_score")), 50.0)
    return _clip(timing)


def _has_valid_best_buy_date(value: Any) -> bool:
    text = _text(value).upper()
    return bool(text and text not in INVALID_BEST_BUY_DATE_TOKENS)


def _is_eoff_strong(value: Any) -> bool:
    normalized = _text(value).upper().replace("-", "_").replace(" ", "_")
    return normalized in EOFF_STRONG_LEVELS


def _timing_priority_label(best_buy_date: Any, eoff_strength: Any, action: Any = "") -> tuple[int, str]:
    action_text = _text(action).upper()
    has_date = _has_valid_best_buy_date(best_buy_date) and action_text != "RECALCULATE_WINDOW"
    strong = _is_eoff_strong(eoff_strength)
    if has_date and strong:
        return 0, "BEST_BUY_DATE + EOFF_STRONG"
    if has_date:
        return 1, "BEST_BUY_DATE"
    if strong:
        return 2, "EOFF_STRONG"
    return 3, "STANDARD"


def _normalize_state(value: Any) -> str:
    text = _text(value).upper().replace("-", "_").replace("/", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is not None and _text(value):
        return value
    source = row.get("source_row")
    if isinstance(source, Mapping):
        return source.get(key)
    return value


def _decision_precedence(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    states = [_normalize_state(_row_value(row, field)) for field in DECISION_STATUS_FIELDS]
    states = [state for state in states if state]
    reasons: list[str] = []

    def contains(token: str) -> bool:
        return any(token in state for state in states)

    if any(state == "REJECT" or state.startswith("REJECT_") or state.endswith("_REJECT") for state in states):
        reasons.append("REJECT")
        return "REJECT", reasons
    if contains("BLOCKED") or contains("AVOID_NEW_BUY"):
        reasons.append("BLOCKED")
        return "BLOCKED", reasons
    if contains("NO_ALLOCATION"):
        reasons.append("NO_ALLOCATION")
        return "NO_ALLOCATION", reasons
    if contains("NOT_QUALIFIED"):
        reasons.append("NOT_QUALIFIED")
        return "NOT_QUALIFIED", reasons
    if contains("DATA_NOT_SCORED") or contains("WAIT_FOR_FUNDAMENTAL_DATA") or contains("INSUFFICIENT_DATA"):
        reasons.append("WAIT_FOR_DATA")
        return "WAIT_FOR_DATA", reasons
    if any(state in ACTIONABLE_TRADE_ACTIONS or state == "BUY_ZONE" for state in states):
        return "BUY_ZONE", reasons
    if any("WAIT" in state or "PREPARE" in state or "ENTRY_PLAN" in state for state in states):
        return "WAIT_FOR_TRIGGER", reasons
    if any("WATCH" in state or "RESEARCH" in state or "READY" in state for state in states):
        return "WATCH", reasons
    return "WATCH", reasons


def _freshness_exclusion(row: Mapping[str, Any]) -> str:
    for field in FRESHNESS_STATUS_FIELDS:
        state = _normalize_state(_row_value(row, field))
        if "EXPIRED" in state:
            return f"EXPIRED_DATA:{field}"
    return ""


def _syariah_exclusion(row: Mapping[str, Any]) -> str:
    explicit = _row_value(row, "is_syariah")
    if explicit is not None and _text(explicit):
        if isinstance(explicit, bool) and not explicit:
            return "NON_SYARIAH"
        if _normalize_state(explicit) in {"FALSE", "NO", "N", "0"}:
            return "NON_SYARIAH"
    status = _normalize_state(_row_value(row, "syariah_status"))
    if status and ("NON_SYARIAH" in status or "NOT_SYARIAH" in status):
        return "NON_SYARIAH"
    return ""


def _eligibility_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    state, reasons = _decision_precedence(row)
    freshness = _freshness_exclusion(row)
    syariah = _syariah_exclusion(row)
    if freshness:
        state = "EXPIRED_DATA"
        reasons.append(freshness)
    if syariah:
        state = "NON_SYARIAH"
        reasons.append(syariah)
    eligible = state not in HARD_EXCLUSION_STATES
    return {
        "is_eligible": bool(eligible),
        "eligibility_status": "ELIGIBLE" if eligible else state,
        "decision_precedence_state": state,
        "eligibility_reasons": " | ".join(dict.fromkeys(reasons)),
        "excluded_from_ranking": not eligible,
    }


def _ranking_reference_date(result: Mapping[str, Any]) -> pd.Timestamp:
    for key in ("ranking_as_of", "scan_as_of", "as_of", "generated_at"):
        value = result.get(key)
        if value is None or not _text(value):
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            return pd.Timestamp(parsed).tz_localize(None).normalize() if pd.Timestamp(parsed).tzinfo else pd.Timestamp(parsed).normalize()
    prepared = result.get("prepared", {}) or {}
    latest: list[pd.Timestamp] = []
    if isinstance(prepared, Mapping):
        for frame in prepared.values():
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                try:
                    stamp = pd.Timestamp(frame.index[-1])
                    latest.append(stamp.tz_localize(None).normalize() if stamp.tzinfo else stamp.normalize())
                except Exception:
                    pass
    if latest:
        return max(latest)
    return pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).normalize()


def _best_buy_date_profile(row: Mapping[str, Any], as_of: pd.Timestamp, max_horizon_days: int = 120) -> tuple[str, str]:
    raw = _text(row.get("best_buy_date"))
    action = _normalize_state(row.get("decision"))
    cycle_state = _normalize_state(row.get("time_cycle_state"))
    if not _has_valid_best_buy_date(raw):
        return "", "NO_VALID_DATE"
    if action == "RECALCULATE_WINDOW":
        return "", "RECALCULATE_WINDOW"
    if cycle_state != "VALIDATED":
        return "", "UNVALIDATED_CYCLE"
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        return "", "INVALID_DATE_FORMAT"
    date = pd.Timestamp(parsed).tz_localize(None).normalize() if pd.Timestamp(parsed).tzinfo else pd.Timestamp(parsed).normalize()
    if date < as_of:
        return "", "EXPIRED"
    if date.weekday() >= 5:
        return "", "NON_TRADING_DATE"
    if date > as_of + timedelta(days=max(1, int(max_horizon_days))):
        return "", "OUTSIDE_HORIZON"
    return date.strftime("%Y-%m-%d"), "VALID"


def _unique_pipe(values: list[Any]) -> str:
    seen: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in seen:
            seen.append(text)
    return " | ".join(seen)



def _normalize_core_strategy(value: Any) -> str:
    key = _text(value).upper().replace("-", "_").replace("/", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    aliases = {
        "PULLBACK": "PULLBACK_CONTINUATION",
        "PULLBACK_CONTINUATION": "PULLBACK_CONTINUATION",
        "BREAKOUT": "BREAKOUT_RETEST",
        "BREAKOUT_RETEST": "BREAKOUT_RETEST",
        "REVERSAL": "REVERSAL_ACCUMULATION",
        "REVERSAL_ACCUMULATION": "REVERSAL_ACCUMULATION",
        "SNIPER": "SNIPER",
        "SNIPER_ICT": "SNIPER",
        "UNICORN": "UNICORN_SNIPER_ICT",
        "UNICORN_SNIPER": "UNICORN_SNIPER_ICT",
        "UNICORN_SNIPER_ICT": "UNICORN_SNIPER_ICT",
    }
    if key in aliases:
        return aliases[key]
    if "PULLBACK" in key:
        return "PULLBACK_CONTINUATION"
    if "BREAKOUT" in key and "RETEST" in key:
        return "BREAKOUT_RETEST"
    if "REVERSAL" in key and "ACCUM" in key:
        return "REVERSAL_ACCUMULATION"
    if "SNIPER" in key:
        return "UNICORN_SNIPER_ICT" if "UNICORN" in key else "SNIPER"
    return key


def _core_candidate_source(result: Mapping[str, Any]) -> pd.DataFrame:
    """Union core candidates from the order builder and raw signals.

    v6.6.3 selected the order builder exclusively whenever it was non-empty.
    The focused builder and raw signals are always unioned so valid daily Swing/Core rows remain available.  This function
    always unions both sources, then prefers the richer builder record when the
    same ticker/strategy appears twice.
    """
    focus = result.get("focus_screens", {}) or {}
    frames: list[pd.DataFrame] = []
    radar = focus.get("core_swing", pd.DataFrame())
    if isinstance(radar, pd.DataFrame) and not radar.empty:
        part = radar.copy()
        part["_candidate_source"] = "STOCK_SELECTOR_RADAR"
        part["_source_priority"] = 0
        frames.append(part)
    builder = focus.get("profit_order_builder", pd.DataFrame())
    if isinstance(builder, pd.DataFrame) and not builder.empty:
        part = builder.copy()
        part["_candidate_source"] = "PROFIT_ORDER_BUILDER"
        part["_source_priority"] = 1
        frames.append(part)
    signals = result.get("signals", pd.DataFrame())
    if isinstance(signals, pd.DataFrame) and not signals.empty:
        part = signals.copy()
        part["_candidate_source"] = "SIGNALS_FALLBACK"
        part["_source_priority"] = 2
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    source = pd.concat(frames, ignore_index=True, sort=False, copy=False)
    source = source.drop(columns=["_normalized_core_strategy"], errors="ignore")
    normalized_strategy = pd.Series(
        [
            _normalize_core_strategy(
                _text(row.get("strategy"))
                or _text(row.get("setup"))
                or _text(row.get("active_setup"))
            )
            for row in source.to_dict(orient="records")
        ],
        index=source.index,
        name="_normalized_core_strategy",
        dtype="object",
    )
    source = pd.concat([source, normalized_strategy], axis=1, copy=False)
    radar_mask = source.get(
        "_candidate_source", pd.Series("", index=source.index),
    ).eq("STOCK_SELECTOR_RADAR")
    no_setup_mask = radar_mask & ~source["_normalized_core_strategy"].isin(CORE_STRATEGIES)
    source.loc[no_setup_mask, "_normalized_core_strategy"] = "WAIT_SETUP"
    source = source[
        source["_normalized_core_strategy"].isin(CORE_STRATEGIES | {"WAIT_SETUP"})
    ].copy()
    if source.empty:
        return source
    source["_ticker_key"] = source.get("ticker", pd.Series(index=source.index, dtype=str)).astype(str).str.upper().str.strip()
    source = source.sort_values(["_source_priority"], ascending=True, kind="stable")
    return source.drop_duplicates(["_ticker_key", "_normalized_core_strategy"], keep="first")

def _build_swing_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = _core_candidate_source(result)
    for _, row in source.iterrows():
        strategy = _text(row.get("_normalized_core_strategy")) or _normalize_core_strategy(
            _text(row.get("strategy")) or _text(row.get("setup"))
        )
        if strategy not in CORE_STRATEGIES | {"WAIT_SETUP"}:
            continue
        score_source = "core_priority_score"
        conviction = _num(row.get(score_source), np.nan)
        if not np.isfinite(conviction):
            score_source = "swing_selection_score"
            conviction = _num(row.get(score_source), np.nan)
        if not np.isfinite(conviction):
            score_source = "hybrid_conviction_score"
            conviction = _num(row.get(score_source), np.nan)
        if not np.isfinite(conviction):
            score_source = "profit_conviction_score"
            conviction = _num(row.get(score_source), np.nan)
        if not np.isfinite(conviction):
            score_source = "analyst_fusion_score"
            conviction = _num(row.get(score_source), np.nan)
        if not np.isfinite(conviction):
            score_source = "composite_score" if np.isfinite(_num(row.get("composite_score"), np.nan)) else "quality_score"
            conviction = _num(row.get(score_source), 0.0)
        # The focus builder has already consumed validated time-cycle once when
        # producing profit/hybrid conviction.  Dashboard ranking must not add it
        # again.  Raw-signal fallbacks also remain unmodified by presentation code.
        final_score = _clip(conviction)
        quick = _quick_score(row)
        cycle = _clip(row.get("time_cycle_alignment_score", row.get("time_cycle_score")), 50.0)
        cycle_state = _normalize_state(row.get("time_cycle_state")) or "UNAVAILABLE"
        cycle_weight = min(0.10, max(0.0, _num(row.get("time_cycle_effective_weight_pct"), 0.0) / 100.0)) if cycle_state == "VALIDATED" else 0.0
        action = _quick_action(row, "WAIT_FOR_EVIDENCE")
        if bool(row.get("narrative_hard_block", False)):
            action = "WAIT_FOR_EVIDENCE"
        if _text(row.get("_candidate_source")) == "STOCK_SELECTOR_RADAR":
            action = (
                _text(row.get("setup_action"))
                if bool(row.get("setup_detected")) and _text(row.get("setup_action"))
                else "WAIT_FOR_EVIDENCE"
            )
        plan = _resolve_trade_plan(row)
        low, high = plan["entry_low"], plan["entry_high"]
        trigger, stop = plan["trigger"], plan["stop_loss"]
        tp1, tp2, rr1 = plan["tp1"], plan["tp2"], plan["rr1"]
        action, plan_warning = _guard_actionable_plan(action, plan)
        reason = (
            _text(row.get("selected_reason"))
            or _text(row.get("selector_selected_reason"))
            or _text(row.get("best_buy_reason"))
            or _text(row.get("conviction_basis"))
            or _text(row.get("reason"))
        )
        if plan_warning:
            reason = " • ".join(value for value in (reason, plan_warning) if value)
        rows.append({
            "ticker": _text(row.get("ticker")),
            "category": "SWING/CORE",
            "strategy": strategy,
            "final_score": round(max(0.0, min(100.0, final_score)), 1),
            "combined_score": round(max(0.0, min(100.0, final_score)), 1),
            "ranking_score_source": score_source,
            "selection_rank": _num(row.get("swing_selection_rank", row.get("selection_rank")), np.nan),
            "technical_selection_score": _num(row.get("technical_selection_score"), np.nan),
            "relative_strength_score": _num(row.get("relative_strength_score", row.get("selector_relative_strength_score")), np.nan),
            "selector_model_state": _text(row.get("selector_model_state")),
            "base_conviction": round(_clip(conviction), 1),
            "multibagger_quality_score": round(_clip(row.get("multibagger_quality_score", row.get("multibagger_score"))), 1),
            "execution_readiness_score": round(_clip(row.get("execution_readiness_score")), 1),
            "multibagger_candidate_type": _text(row.get("multibagger_candidate_type")),
            "research_recommendation_status": _text(row.get("research_recommendation_status")),
            "silent_accumulation_score": round(_clip(row.get("silent_accumulation_score", row.get("accumulation_score")), 0.0), 1),
            "silent_accumulation_state": _text(row.get("silent_accumulation_state")),
            "narrative_flow_effective_score": _num(
                row.get("narrative_flow_effective_score"), np.nan,
            ),
            "narrative_flow_convergence_state": _text(
                row.get("narrative_flow_convergence_state"),
            ),
            "narrative_flow_research_score": _num(
                row.get("narrative_flow_research_score"), np.nan,
            ),
            "narrative_flow_score_state": _text(
                row.get("narrative_flow_score_state"),
            ),
            "operating_narrative_proxy_score": _num(
                row.get("operating_narrative_proxy_score"), np.nan,
            ),
            "narrative_score_state": _text(
                row.get("narrative_score_state"),
            ),
            "issuer_alignment_effective_score": _num(
                row.get("issuer_alignment_effective_score"), np.nan,
            ),
            "issuer_alignment_score_state": _text(
                row.get("issuer_alignment_score_state"),
            ),
            "issuer_alignment_evidence_basis": _text(
                row.get("issuer_alignment_evidence_basis"),
            ),
            "retail_adoption_stage": _text(
                row.get("retail_adoption_stage"),
            ),
            "narrative_hard_block": bool(
                row.get("narrative_hard_block", False),
            ),
            "quick_buy_score": round(quick, 1),
            "time_cycle_score": round(_clip(row.get("time_cycle_score"), cycle), 1),
            "time_cycle_confidence": round(_clip(row.get("time_cycle_confidence"), 0.0), 1),
            "time_cycle_state": cycle_state or "UNAVAILABLE",
            "time_cycle_effective_weight_pct": round(100.0 * cycle_weight, 2),
            "dashboard_time_cycle_adjustment": 0.0,
            "time_cycle_counted_in_final": bool(score_source in {"profit_conviction_score", "hybrid_conviction_score"} and cycle_weight > 0.0),
            "decision": action,
            "best_buy_date": _text(row.get("best_buy_date")),
            "buy_window_start": _text(row.get("best_buy_window_start", row.get("next_reversal_window_start"))),
            "buy_window_end": _text(row.get("best_buy_window_end", row.get("next_reversal_window_end"))),
            "entry_low": low,
            "entry_high": high,
            "entry_zone_state": plan.get("entry_zone_state", "MISSING"),
            "execution_price": plan.get("execution_price", np.nan),
            "trigger": trigger,
            "confirmation_level": plan.get("confirmation_level", np.nan),
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "rr1": rr1,
            "rr2": plan["rr2"],
            "execution_plan_source": plan["execution_plan_source"],
            "execution_plan_mode": plan.get("execution_plan_mode", "NO_VALID_PLAN"),
            "execution_plan_state": plan["execution_plan_state"],
            "execution_plan_valid": plan["execution_plan_valid"],
            "execution_plan_id": plan.get("execution_plan_id", ""),
            "diagnostic_entry_low": plan.get("diagnostic_entry_low", np.nan),
            "diagnostic_entry_high": plan.get("diagnostic_entry_high", np.nan),
            "diagnostic_execution_price": plan.get("diagnostic_execution_price", np.nan),
            "diagnostic_confirmation_level": plan.get("diagnostic_confirmation_level", np.nan),
            "diagnostic_trigger": plan.get("diagnostic_trigger", np.nan),
            "diagnostic_stop_loss": plan.get("diagnostic_stop_loss", np.nan),
            "diagnostic_tp1": plan.get("diagnostic_tp1", np.nan),
            "diagnostic_tp2": plan.get("diagnostic_tp2", np.nan),
            "diagnostic_rr1": plan.get("diagnostic_rr1", np.nan),
            "diagnostic_rr2": plan.get("diagnostic_rr2", np.nan),
            "phase": _text(row.get("time_cycle_phase")) or "UNKNOWN",
            "eoff_strength": _text(row.get("eoff_strength_label")) or "LOW",
            "source_status": _text(row.get("decision_state")) or _text(row.get("setup_status")) or _text(row.get("status")),
            "status": _safe_plan_status(
                _text(row.get("decision_state")) or _text(row.get("setup_status")) or _text(row.get("status")),
                bool(plan["execution_plan_valid"]),
            ),
            "setup_status": _text(row.get("setup_status")) or _text(row.get("status")),
            "decision_state": _text(row.get("decision_state")),
            "allocation_action": _text(row.get("allocation_action")),
            "multibagger_status": _text(row.get("multibagger_status")),
            "multibagger_scoring_state": _text(row.get("multibagger_scoring_state")),
            "database_read_state": _text(row.get("database_read_state")),
            "statement_age_state": _text(row.get("statement_age_state")),
            "price_freshness_state": _text(row.get("price_freshness_state")),
            "data_freshness_state": _text(row.get("data_freshness_state")),
            "is_syariah": row.get("is_syariah"),
            "syariah_status": _text(row.get("syariah_status")),
            "reason": reason,
            "selected_reason": _text(row.get("selected_reason")) or _text(row.get("selector_selected_reason")),
            "not_entry_reason": _text(row.get("not_entry_reason")) or _text(row.get("warnings")),
            "trigger_waiting": _text(row.get("trigger_waiting")),
            "invalidation_reason": _text(row.get("invalidation_reason")),
            "primary_risk": _text(row.get("primary_risk")),
            "no_trade": _text(row.get("not_entry_reason")) or _text(row.get("best_buy_no_trade_condition")) or _text(row.get("warnings")),
            "best_buy_target_basis": _text(row.get("best_buy_target_basis")),
            "candidate_source": _text(row.get("_candidate_source")) or "UNKNOWN",
            "source_row": row.to_dict(),
        })
    return rows


def _build_multibagger_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    focus = result.get("focus_screens", {}) or {}
    frame = focus.get("multibagger", pd.DataFrame())
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        scoring_state = _text(row.get("multibagger_scoring_state")).upper()
        if scoring_state.startswith("DATA_NOT_SCORED"):
            continue
        score_source = "multibagger_selection_score"
        conviction = _num(row.get(score_source), np.nan)
        if not np.isfinite(conviction):
            score_source = "confidence_adjusted_multibagger_score"
            conviction = _num(row.get(score_source), np.nan)
        if not np.isfinite(conviction):
            score_source = "capital_conviction_score"
            conviction = _num(row.get(score_source), np.nan)
        if not np.isfinite(conviction):
            score_source = "multibagger_score"
            conviction = _num(row.get(score_source), 0.0)
        # Multibagger quality/capital conviction is deliberately cycle-free.
        # Time-cycle remains a timing annotation and never upgrades issuer quality.
        final_score = _clip(conviction)
        quick = _quick_score(row)
        cycle = _clip(row.get("multibagger_time_cycle_score", row.get("time_cycle_score")), 50.0)
        cycle_state = _normalize_state(row.get("time_cycle_state")) or "UNAVAILABLE"
        cycle_weight = 0.0
        action = _quick_action(row, "RESEARCH_AND_WAIT")
        if bool(row.get("narrative_hard_block", False)):
            action = "RESEARCH_AND_WAIT"
        plan = _resolve_trade_plan(row)
        low, high = plan["entry_low"], plan["entry_high"]
        trigger, stop = plan["trigger"], plan["stop_loss"]
        tp1, tp2, rr1 = plan["tp1"], plan["tp2"], plan["rr1"]
        action, plan_warning = _guard_actionable_plan(action, plan)
        positive_drivers = _text(row.get("top_positive_drivers"))
        negative_drivers = _text(row.get("top_negative_drivers"))
        reason = _text(row.get("selected_reason")) or _text(row.get("best_buy_reason")) or _text(row.get("allocation_reason")) or _text(row.get("note"))
        evidence_reason = " | ".join(value for value in (
            f"Positif: {positive_drivers}" if positive_drivers else "",
            f"Risiko: {negative_drivers}" if negative_drivers else "",
        ) if value)
        reason = " • ".join(value for value in (reason, evidence_reason) if value)
        if plan_warning:
            reason = " • ".join(value for value in (reason, plan_warning) if value)
        rows.append({
            "ticker": _text(row.get("ticker")),
            "category": "MULTIBAGGER",
            "strategy": (
                _text(row.get("multibagger_lane"))
                or _text(row.get("active_setup"))
                or "MULTIBAGGER"
            ),
            "final_score": round(max(0.0, min(100.0, final_score)), 1),
            "combined_score": round(max(0.0, min(100.0, final_score)), 1),
            "ranking_score_source": score_source,
            "selection_rank": _num(row.get("multibagger_selection_rank"), np.nan),
            "technical_selection_score": _num(row.get("technical_selection_score"), np.nan),
            "relative_strength_score": _num(row.get("selector_relative_strength_score"), np.nan),
            "selector_model_state": _text(row.get("selector_model_state")),
            "base_conviction": round(_clip(conviction), 1),
            "multibagger_quality_score": round(_clip(row.get("multibagger_quality_score", row.get("multibagger_score"))), 1),
            "confidence_adjusted_multibagger_score": _num(row.get("confidence_adjusted_multibagger_score"), np.nan),
            "confidence_adjusted_turnaround_score": _num(row.get("confidence_adjusted_turnaround_score"), np.nan),
            "growth_compounder_score": _num(row.get("growth_compounder_score"), np.nan),
            "growth_compounder_selection_score": _num(row.get("growth_compounder_selection_score"), np.nan),
            "turnaround_recovery_score": _num(row.get("turnaround_recovery_score"), np.nan),
            "turnaround_selection_score": _num(row.get("turnaround_selection_score"), np.nan),
            "turnaround_research_state": _text(row.get("turnaround_research_state")),
            "turnaround_recovery_signals": int(max(0, _num(row.get("turnaround_recovery_signals"), 0))),
            "turnaround_gate_reasons": _text(row.get("turnaround_gate_reasons")),
            "multibagger_lane": _text(row.get("multibagger_lane")) or "GROWTH_COMPOUNDER",
            "research_eligible": bool(row.get("research_eligible", False)),
            "research_eligibility_reason": _text(row.get("research_eligibility_reason")),
            "portfolio_allocation_eligible": bool(
                row.get("portfolio_allocation_eligible", row.get("allocation_eligible", False))
            ),
            "overall_research_confidence": round(_clip(row.get("overall_research_confidence"), 0.0), 1),
            "overall_research_confidence_grade": _text(row.get("overall_research_confidence_grade")),
            "data_confidence_score": round(_clip(row.get("data_confidence_score"), 0.0), 1),
            "fundamental_confidence_score": round(_clip(row.get("fundamental_confidence_score"), 0.0), 1),
            "future_fundamental_confidence_score": round(_clip(row.get("future_fundamental_confidence_score"), 0.0), 1),
            "technical_confidence_score": round(_clip(row.get("technical_confidence_score"), 0.0), 1),
            "eoff_confidence_score": round(_clip(row.get("eoff_confidence_score"), 0.0), 1),
            "top_positive_drivers": positive_drivers,
            "top_negative_drivers": negative_drivers,
            "scoring_reason_codes": _text(row.get("scoring_reason_codes")),
            "execution_readiness_score": round(_clip(row.get("execution_readiness_score")), 1),
            "multibagger_candidate_type": _text(row.get("multibagger_candidate_type")),
            "research_recommendation_status": _text(row.get("research_recommendation_status")),
            "economic_earnings_score": _num(row.get("economic_earnings_score"), np.nan),
            "economic_earnings_state": _text(row.get("economic_earnings_state")),
            "minority_leakage_pct": _num(row.get("minority_leakage_pct"), np.nan),
            "silent_accumulation_score": round(_clip(
                row.get(
                    "effective_silent_accumulation_score",
                    row.get("silent_accumulation_score", row.get("accumulation_score")),
                ),
                0.0,
            ), 1),
            "silent_accumulation_raw_research_score": round(_clip(
                row.get("silent_accumulation_score", row.get("accumulation_score")),
                0.0,
            ), 1),
            "effective_silent_accumulation_score": round(_clip(
                row.get(
                    "effective_silent_accumulation_score",
                    row.get("silent_accumulation_score", row.get("accumulation_score")),
                ),
                0.0,
            ), 1),
            "silent_accumulation_state": _text(row.get("silent_accumulation_state")),
            "narrative_flow_effective_score": _num(
                row.get("narrative_flow_effective_score"), np.nan,
            ),
            "narrative_flow_convergence_state": _text(
                row.get("narrative_flow_convergence_state"),
            ),
            "narrative_flow_research_score": _num(
                row.get("narrative_flow_research_score"), np.nan,
            ),
            "narrative_flow_score_state": _text(
                row.get("narrative_flow_score_state"),
            ),
            "operating_narrative_proxy_score": _num(
                row.get("operating_narrative_proxy_score"), np.nan,
            ),
            "narrative_score_state": _text(
                row.get("narrative_score_state"),
            ),
            "issuer_alignment_effective_score": _num(
                row.get("issuer_alignment_effective_score"), np.nan,
            ),
            "issuer_alignment_score_state": _text(
                row.get("issuer_alignment_score_state"),
            ),
            "issuer_alignment_evidence_basis": _text(
                row.get("issuer_alignment_evidence_basis"),
            ),
            "retail_adoption_stage": _text(
                row.get("retail_adoption_stage"),
            ),
            "narrative_hard_block": bool(
                row.get("narrative_hard_block", False),
            ),
            "accumulation_persistence_score": round(_clip(row.get("accumulation_persistence_score"), 0.0), 1),
            "accumulation_positive_windows_pct": round(_clip(row.get("accumulation_positive_windows_pct"), 0.0), 1),
            "accumulation_longest_run": int(max(0, _num(row.get("accumulation_longest_run"), 0))),
            "accumulation_regime": _text(row.get("accumulation_regime")),
            "silent_accumulation_v4_adjustment": _num(row.get("silent_accumulation_v4_adjustment"), 0.0),
            "absorption_confirmed_days20": int(max(0, _num(row.get("absorption_confirmed_days20"), 0))),
            "failed_absorption_days20": int(max(0, _num(row.get("failed_absorption_days20"), 0))),
            "persistent_bid_score": round(_clip(row.get("persistent_bid_score")), 1),
            "project_stage": _text(row.get("project_stage")),
            "project_stage_probability_pct": _num(row.get("project_stage_probability_pct"), np.nan),
            "quick_buy_score": round(quick, 1),
            "time_cycle_score": round(cycle, 1),
            "time_cycle_confidence": round(_clip(row.get("time_cycle_confidence"), 0.0), 1),
            "time_cycle_state": cycle_state or "UNAVAILABLE",
            "time_cycle_effective_weight_pct": 0.0,
            "dashboard_time_cycle_adjustment": 0.0,
            "time_cycle_counted_in_final": False,
            "decision": action,
            "best_buy_date": _text(row.get("best_buy_date")),
            "buy_window_start": _text(row.get("best_buy_window_start", row.get("next_reversal_window_start"))),
            "buy_window_end": _text(row.get("best_buy_window_end", row.get("next_reversal_window_end"))),
            "entry_low": low,
            "entry_high": high,
            "entry_zone_state": plan.get("entry_zone_state", "MISSING"),
            "execution_price": plan.get("execution_price", np.nan),
            "trigger": trigger,
            "confirmation_level": plan.get("confirmation_level", np.nan),
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "rr1": rr1,
            "rr2": plan["rr2"],
            "execution_plan_source": plan["execution_plan_source"],
            "execution_plan_mode": plan.get("execution_plan_mode", "NO_VALID_PLAN"),
            "execution_plan_state": plan["execution_plan_state"],
            "execution_plan_valid": plan["execution_plan_valid"],
            "execution_plan_id": plan.get("execution_plan_id", ""),
            "diagnostic_entry_low": plan.get("diagnostic_entry_low", np.nan),
            "diagnostic_entry_high": plan.get("diagnostic_entry_high", np.nan),
            "diagnostic_execution_price": plan.get("diagnostic_execution_price", np.nan),
            "diagnostic_confirmation_level": plan.get("diagnostic_confirmation_level", np.nan),
            "diagnostic_trigger": plan.get("diagnostic_trigger", np.nan),
            "diagnostic_stop_loss": plan.get("diagnostic_stop_loss", np.nan),
            "diagnostic_tp1": plan.get("diagnostic_tp1", np.nan),
            "diagnostic_tp2": plan.get("diagnostic_tp2", np.nan),
            "diagnostic_rr1": plan.get("diagnostic_rr1", np.nan),
            "diagnostic_rr2": plan.get("diagnostic_rr2", np.nan),
            "phase": _text(row.get("time_cycle_phase")) or "UNKNOWN",
            "eoff_strength": _text(row.get("eoff_strength_label")) or "LOW",
            "source_status": _text(row.get("capital_tier")) or _text(row.get("multibagger_status")),
            "status": _safe_plan_status(
                _text(row.get("capital_tier")) or _text(row.get("multibagger_status")),
                bool(plan["execution_plan_valid"]),
            ),
            "setup_status": _text(row.get("technical_entry_state")),
            "decision_state": _text(row.get("compounding_state")),
            "allocation_action": _text(row.get("allocation_action")),
            "multibagger_status": _text(row.get("multibagger_status")),
            "multibagger_scoring_state": _text(row.get("multibagger_scoring_state")),
            "database_read_state": _text(row.get("database_read_state")),
            "statement_age_state": _text(row.get("statement_age_state")),
            "price_freshness_state": _text(row.get("price_freshness_state")),
            "data_freshness_state": _text(row.get("data_freshness_state")),
            "is_syariah": row.get("is_syariah"),
            "syariah_status": _text(row.get("syariah_status")),
            "reason": reason,
            "selected_reason": _text(row.get("selected_reason")),
            "not_entry_reason": _text(row.get("not_entry_reason")),
            "trigger_waiting": _text(row.get("trigger_waiting")),
            "invalidation_reason": _text(row.get("invalidation_reason")),
            "primary_risk": _text(row.get("primary_risk")),
            "no_trade": _text(row.get("not_entry_reason")) or _text(row.get("best_buy_no_trade_condition")) or _text(row.get("red_flags")),
            "best_buy_target_basis": _text(row.get("best_buy_target_basis")),
            "source_row": row.to_dict(),
        })
    return rows


def build_top20_ranking(result: Mapping[str, Any], limit: int = 20) -> pd.DataFrame:
    """Build the safe Top-20 research ranking.

    P0 policy:
    - hard-exclude REJECT/BLOCKED/NO_ALLOCATION/NOT_QUALIFIED,
      missing-score, expired-data, and explicit non-syariah rows;
    - rank eligible tickers lexicographically by final score first and Silent
      Accumulation second;
    - Best Buy Date and EOFF are timing annotations only, never ranking
      overrides;
    - a Best Buy Date is exposed only when the cycle is VALIDATED and the
      date is parseable, non-expired, a weekday, and inside the horizon.
    """
    raw = _build_swing_rows(result) + _build_multibagger_rows(result)
    if not raw:
        return pd.DataFrame()
    frame = pd.DataFrame(raw)
    frame = frame[frame["ticker"].astype(str).str.len().gt(0)].copy()
    if frame.empty:
        return frame

    profiles = pd.DataFrame([_eligibility_profile(row) for row in frame.to_dict("records")], index=frame.index)
    frame = pd.concat([frame, profiles], axis=1)
    excluded = frame[frame["excluded_from_ranking"]].copy()
    frame = frame[frame["is_eligible"]].copy()
    if frame.empty:
        empty = pd.DataFrame()
        empty.attrs["excluded_candidates"] = excluded
        empty.attrs["candidate_pool_audit"] = {
            "eligible_total": 0,
            "excluded_total": int(len(excluded)),
            "priority_policy": "FINAL_SCORE_THEN_SILENT_ACCUMULATION",
            "hard_release_gate": "P0_ACTIVE",
        }
        return empty

    as_of = _ranking_reference_date(result)
    date_profiles = [
        _best_buy_date_profile(row, as_of) for row in frame.to_dict("records")
    ]
    frame["best_buy_date_raw"] = frame["best_buy_date"]
    frame["best_buy_date"] = [value[0] for value in date_profiles]
    frame["best_buy_date_state"] = [value[1] for value in date_profiles]
    frame["has_best_buy_date"] = frame["best_buy_date_state"].eq("VALID")
    frame["eoff_strong"] = frame["eoff_strength"].map(_is_eoff_strong)
    timing = frame.apply(
        lambda row: _timing_priority_label(
            row.get("best_buy_date"), row.get("eoff_strength"), row.get("decision")
        ),
        axis=1,
    )
    frame["timing_priority"] = [value[0] for value in timing]
    frame["timing_priority_label"] = [value[1] for value in timing]
    # Final score is category-specific: swing_selection/core_priority for
    # Swing/Core and multibagger_selection_score for Multibagger.
    frame["ranking_priority"] = "FINAL_SCORE → SILENT_ACCUMULATION"
    frame["final_score"] = pd.to_numeric(
        frame.get("final_score", frame.get("combined_score")), errors="coerce"
    )
    frame["combined_score"] = frame["final_score"]
    frame["silent_accumulation_score"] = pd.to_numeric(
        frame.get("silent_accumulation_score"), errors="coerce"
    ).fillna(0.0).clip(0.0, 100.0)
    rr1 = pd.to_numeric(frame.get("rr1"), errors="coerce")
    rr2 = pd.to_numeric(frame.get("rr2"), errors="coerce")
    rr1_quality = (100.0 * rr1 / 2.0).clip(0.0, 100.0)
    rr2_quality = (100.0 * rr2 / 3.0).clip(0.0, 100.0)
    frame["execution_rr_quality_score"] = (
        0.60 * rr1_quality.fillna(0.0) + 0.40 * rr2_quality.fillna(0.0)
    ).round(1)
    readiness = pd.to_numeric(
        frame.get("execution_readiness_score"), errors="coerce",
    )
    status_readiness = frame.get(
        "setup_status", pd.Series("", index=frame.index),
    ).map({
        "EXECUTION_READY": 100.0,
        "READY_FOR_STOCKBIT_VERIFY": 92.0,
        "SIGNAL_READY": 85.0,
        "ENTRY_PLAN_READY": 72.0,
        "READY_FOR_PRICE_VERIFY": 64.0,
        "WATCHLIST_ENTRY": 50.0,
        "NO_SETUP": 0.0,
    }).fillna(35.0)
    frame["execution_readiness_normalized"] = readiness.where(
        readiness.notna(), status_readiness,
    ).clip(0.0, 100.0)
    frame["execution_priority_score"] = (
        0.50 * frame["final_score"].fillna(0.0)
        + 0.20 * frame["silent_accumulation_score"]
        + 0.15 * frame["execution_rr_quality_score"]
        + 0.15 * frame["execution_readiness_normalized"]
    ).round(1)
    plan_valid = frame.get(
        "execution_plan_valid", pd.Series(False, index=frame.index),
    ).fillna(False).astype(bool)
    frame.loc[~plan_valid, "execution_priority_score"] = np.nan
    frame["_action_rank"] = frame["decision"].map(ACTION_PRIORITY).fillna(8)

    sort_columns = [
        "final_score", "silent_accumulation_score", "base_conviction",
        "_action_rank", "ticker",
    ]
    sort_ascending = [False, False, False, True, True]
    frame = frame.sort_values(
        sort_columns, ascending=sort_ascending, na_position="last", kind="stable"
    )
    frame = frame.drop_duplicates(["ticker", "category", "strategy"], keep="first").copy()

    ticker_details = {
        ticker: group.sort_values(
            sort_columns, ascending=sort_ascending, na_position="last", kind="stable"
        ).to_dict("records")
        for ticker, group in frame.groupby("ticker", sort=False)
    }

    out = frame.drop_duplicates(["ticker"], keep="first").copy()
    out["all_rows"] = out["ticker"].map(ticker_details)
    out["category"] = out["all_rows"].map(
        lambda rows: " + ".join(
            category for category in ("MULTIBAGGER", "SWING/CORE")
            if any(_text(item.get("category")) == category for item in rows)
        )
    )
    out["strategy"] = out["all_rows"].map(
        lambda rows: _unique_pipe([item.get("strategy") for item in rows])
    )
    out["candidate_id"] = out["ticker"].map(lambda ticker: f"{_text(ticker)}|TOP20")

    pool = out.copy()
    eligible_multibagger = int(pool["category"].str.contains("MULTIBAGGER", na=False).sum())
    eligible_swing = int(pool["category"].str.contains("SWING/CORE", na=False).sum())
    best_multibagger = _num(
        pool.loc[pool["category"].str.contains("MULTIBAGGER", na=False), "final_score"].max(), np.nan
    )
    best_swing = _num(
        pool.loc[pool["category"].str.contains("SWING/CORE", na=False), "final_score"].max(), np.nan
    )
    top_limit = max(1, int(limit))
    out = out.head(top_limit).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["execution_rank"] = np.nan
    valid_execution = out["execution_priority_score"].notna()
    if valid_execution.any():
        execution_order = out.loc[valid_execution].sort_values(
            [
                "execution_priority_score", "execution_rr_quality_score",
                "final_score", "silent_accumulation_score", "ticker",
            ],
            ascending=[False, False, False, False, True],
            kind="stable",
        ).index
        out.loc[execution_order, "execution_rank"] = np.arange(
            1, len(execution_order) + 1,
        )
    final = out.drop(columns=["_action_rank"], errors="ignore")
    ihsg = result.get("ihsg_direction", {}) if isinstance(result, Mapping) else {}
    if isinstance(ihsg, Mapping):
        final["ihsg_regime"] = _text(ihsg.get("regime")) or "UNKNOWN"
        final["ihsg_consensus"] = _text(ihsg.get("consensus_direction")) or "NO_EDGE"
        final["ihsg_risk_budget_pct"] = round(
            100.0 * _num(ihsg.get("risk_budget_multiplier"), 0.50), 1
        )
        final["ihsg_overlay_policy"] = "RISK_CAP_ONLY_NO_RANK_BONUS"
    cutoff = _num(final["final_score"].min(), np.nan) if not final.empty else np.nan

    excluded_status_counts = (
        excluded.get("eligibility_status", pd.Series(dtype=str)).value_counts().to_dict()
        if not excluded.empty else {}
    )
    final.attrs["excluded_candidates"] = excluded
    final.attrs["candidate_pool_audit"] = {
        "eligible_total": int(len(pool)),
        "excluded_total": int(len(excluded)),
        "excluded_by_status": excluded_status_counts,
        "eligible_multibagger": eligible_multibagger,
        "eligible_swing_core": eligible_swing,
        "top20_multibagger": int(final["category"].str.contains("MULTIBAGGER", na=False).sum()),
        "top20_swing_core": int(final["category"].str.contains("SWING/CORE", na=False).sum()),
        "top20_date_eoff_strong": int(
            (final["has_best_buy_date"] & final["eoff_strong"]).sum()
        ),
        "top20_with_best_buy_date": int(final["has_best_buy_date"].sum()),
        "top20_eoff_strong": int(final["eoff_strong"].sum()),
        "best_multibagger_score": best_multibagger,
        "best_swing_core_score": best_swing,
        "top20_cutoff_score": cutoff,
        "ranking_as_of": as_of.strftime("%Y-%m-%d"),
        "forced_category_quota": False,
        "unique_ticker_ranking": True,
        "priority_policy": "FINAL_SCORE_THEN_SILENT_ACCUMULATION",
        "hard_release_gate": "P0_ACTIVE",
        "timing_is_tiebreaker": False,
        "ihsg_overlay_changes_ranking": False,
    }
    return final


def build_top15_ranking(result: Mapping[str, Any], limit: int = 20) -> pd.DataFrame:
    """Backward-compatible alias.  The dashboard now defaults to Top 20."""
    return build_top20_ranking(result, limit=limit)


def build_multibagger_ranking(
    result: Mapping[str, Any],
    limit: int = 12,
    lane: str | None = None,
) -> pd.DataFrame:
    """Rank Multibagger research independently from portfolio allocation.

    A candidate can be valuable research even when the portfolio allocator has
    no remaining slot or its entry setup is not ready. Critical research and
    data gates still apply; ``NO_ALLOCATION`` is only a capital annotation.
    """
    focus = result.get("focus_screens", {}) or {}
    source = focus.get("multibagger", pd.DataFrame())
    if not isinstance(source, pd.DataFrame) or source.empty:
        return pd.DataFrame()
    frame = source.copy()
    requested_lane = _normalize_state(lane)
    if requested_lane:
        frame = frame[
            frame.get(
                "multibagger_lane",
                pd.Series("", index=frame.index),
            ).map(_normalize_state).eq(requested_lane)
        ].copy()
    if frame.empty:
        return pd.DataFrame()

    if "research_eligible" in frame:
        research_mask = frame["research_eligible"].fillna(False).astype(bool)
    else:
        status = frame.get(
            "multibagger_status", pd.Series("", index=frame.index),
        ).map(_normalize_state)
        turnaround = frame.get(
            "turnaround_research_state", pd.Series("", index=frame.index),
        ).map(_normalize_state)
        research_mask = status.isin({
            "MULTIBAGGER_A_CANDIDATE", "MULTIBAGGER_B_CANDIDATE",
        }) | turnaround.isin({
            "TURNAROUND_CONFIRMED", "TURNAROUND_EARLY",
        })
    frame = frame.loc[research_mask].copy()
    if frame.empty:
        return pd.DataFrame()

    # Research eligibility and allocation eligibility are separate decisions.
    # Neutralise only allocation-only labels before applying the unchanged P0
    # freshness, syariah, critical, and missing-data checks.
    for column in ("allocation_action", "review_action"):
        if column in frame:
            values = frame[column].fillna("").astype(str)
            allocation_only = values.str.upper().str.contains(
                "NO_ALLOCATION|NO_COMPOUNDING_ALLOCATION",
                regex=True,
            )
            frame.loc[allocation_only, column] = "RESEARCH_AND_WAIT"
    isolated = dict(result)
    isolated_focus = {
        "multibagger": frame,
        "core_swing": pd.DataFrame(),
        "profit_order_builder": pd.DataFrame(),
    }
    isolated["focus_screens"] = isolated_focus
    isolated["signals"] = pd.DataFrame()
    ranking = build_top20_ranking(isolated, limit=max(1, int(limit)))
    if not ranking.empty:
        ranking["radar"] = requested_lane or "MULTIBAGGER_RESEARCH"
        ranking["ranking_priority"] = (
            "LANE_SCORE → CONFIDENCE-ADJUSTED_SILENT_ACCUMULATION"
        )
        audit = dict(ranking.attrs.get("candidate_pool_audit", {}))
        audit.update({
            "radar": requested_lane or "MULTIBAGGER_RESEARCH",
            "allocation_status_changes_research_rank": False,
            "cross_category_score_comparison": False,
        })
        ranking.attrs["candidate_pool_audit"] = audit
    return ranking


def build_multibagger_provisional_ranking(
    result: Mapping[str, Any],
    limit: int = 12,
    lane: str | None = None,
) -> pd.DataFrame:
    """Return scored near-misses without pretending they passed research gates.

    The qualified ranking must remain empty when no issuer passes its evidence
    gates.  This separate queue keeps the best *research* follow-ups visible,
    forces allocation to zero, and exposes the exact evidence still required.
    """
    focus = result.get("focus_screens", {}) or {}
    requested_lane = _normalize_state(lane)
    key = (
        "multibagger_turnaround_research_queue"
        if requested_lane == "TURNAROUND_CYCLICAL"
        else "multibagger_growth_research_queue"
    )
    source = focus.get(key, pd.DataFrame())
    if not isinstance(source, pd.DataFrame) or source.empty:
        source = focus.get(
            "multibagger_turnaround_near_miss"
            if requested_lane == "TURNAROUND_CYCLICAL"
            else "multibagger_growth_near_miss",
            pd.DataFrame(),
        )

    if isinstance(source, pd.DataFrame) and not source.empty:
        frame = source.copy()
    else:
        raw = focus.get("multibagger", pd.DataFrame())
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return pd.DataFrame()
        frame = raw.copy()
        if requested_lane:
            frame = frame[
                frame.get(
                    "multibagger_lane",
                    pd.Series("", index=frame.index),
                ).map(_normalize_state).eq(requested_lane)
            ].copy()
        eligible = frame.get(
            "research_eligible", pd.Series(False, index=frame.index),
        ).fillna(False).astype(bool)
        coverage = pd.to_numeric(
            frame.get(
                "fundamental_coverage",
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        )
        score = pd.to_numeric(
            frame.get(
                "fundamental_score",
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        )
        frame = frame.loc[~eligible & coverage.gt(0.0) & score.notna()].copy()
        if frame.empty:
            return pd.DataFrame()
        history_sources = pd.to_numeric(
            frame.get(
                "fundamental_history_source_count",
                pd.Series(0.0, index=frame.index),
            ),
            errors="coerce",
        ).fillna(0.0)
        grade = frame.get(
            "fundamental_data_grade",
            pd.Series("D", index=frame.index),
        ).fillna("D").astype(str).str.upper()
        frame["candidate_state"] = "PROVISIONAL_RESEARCH"
        frame["near_miss_state"] = np.where(
            history_sources.le(0.0),
            "STATEMENT_HISTORY_PENDING",
            np.where(
                grade.isin({"A", "B", "C"}),
                "QUALITY_GATE_PENDING",
                "DATA_PROVENANCE_PENDING",
            ),
        )
        frame["next_required_evidence"] = np.where(
            history_sources.le(0.0),
            "BACKFILL_AND_VALIDATE_STATEMENT_HISTORY",
            np.where(
                grade.isin({"A", "B", "C"}),
                "PASS_QUALITY_OR_RECOVERY_GATE",
                "IMPROVE_DATA_PROVENANCE_TO_GRADE_C_OR_BETTER",
            ),
        )
        frame["near_miss_reason"] = frame.get(
            "research_eligibility_reason",
            pd.Series(
                "Quality/recovery gate belum lolos", index=frame.index,
            ),
        )
        frame["capital_state"] = "RESEARCH_ONLY_NO_ALLOCATION"

    frame["portfolio_allocation_eligible"] = False
    frame["capital_state"] = "RESEARCH_ONLY_NO_ALLOCATION"
    score_column = (
        "turnaround_provisional_priority_score"
        if requested_lane == "TURNAROUND_CYCLICAL"
        else "growth_provisional_priority_score"
    )
    if score_column not in frame:
        score_column = (
            "turnaround_selection_score"
            if requested_lane == "TURNAROUND_CYCLICAL"
            else "growth_compounder_selection_score"
        )
    if score_column not in frame:
        score_column = "multibagger_selection_score"
    frame["provisional_score"] = pd.to_numeric(
        frame.get(score_column, pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    state_rank = frame.get(
        "near_miss_state", pd.Series("", index=frame.index),
    ).map({
        "QUALITY_GATE_PENDING": 0,
        "STATEMENT_HISTORY_PENDING": 1,
        "DATA_PROVENANCE_PENDING": 2,
        "CRITICAL_RISK_BLOCKED": 9,
    }).fillna(5)
    frame["_state_rank"] = state_rank
    frame["_silent_rank"] = pd.to_numeric(
        frame.get(
            "effective_silent_accumulation_score",
            pd.Series(np.nan, index=frame.index),
        ),
        errors="coerce",
    )
    frame = (
        frame.sort_values(
            ["_state_rank", "provisional_score", "_silent_rank", "ticker"],
            ascending=[True, False, False, True],
            kind="stable",
            na_position="last",
        )
        .head(max(1, int(limit)))
        .drop(columns=["_state_rank", "_silent_rank"], errors="ignore")
        .reset_index(drop=True)
    )
    if "research_queue_rank" in frame:
        frame["research_queue_rank"] = np.arange(1, len(frame) + 1)
    else:
        frame.insert(0, "research_queue_rank", np.arange(1, len(frame) + 1))
    frame.attrs["candidate_pool_audit"] = {
        "radar": requested_lane or "MULTIBAGGER_PROVISIONAL",
        "qualified": False,
        "allocation_forced_zero": True,
        "policy": "SCORED_NEAR_MISS_ONLY_NO_GATE_RELAXATION",
    }
    return frame


def build_swing_ranking(
    result: Mapping[str, Any],
    limit: int = 20,
) -> pd.DataFrame:
    """Rank Swing/Core only, using its technical and excess-return horizon."""
    focus = result.get("focus_screens", {}) or {}
    isolated = dict(result)
    isolated["focus_screens"] = {
        "core_swing": focus.get("core_swing", pd.DataFrame()),
        "profit_order_builder": focus.get(
            "profit_order_builder", pd.DataFrame(),
        ),
        "multibagger": pd.DataFrame(),
    }
    ranking = build_top20_ranking(isolated, limit=max(1, int(limit)))
    if not ranking.empty:
        ranking = ranking[
            ranking["category"].astype(str).str.contains("SWING/CORE", na=False)
        ].reset_index(drop=True)
        ranking["rank"] = np.arange(1, len(ranking) + 1)
        ranking["radar"] = "SWING_CORE"
        ranking["ranking_priority"] = (
            "FINAL_SCORE → SILENT_ACCUMULATION → EXECUTION_READINESS"
        )
        audit = dict(ranking.attrs.get("candidate_pool_audit", {}))
        audit.update({
            "radar": "SWING_CORE",
            "cross_category_score_comparison": False,
        })
        ranking.attrs["candidate_pool_audit"] = audit
    return ranking


def _safe_display_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a PyArrow-safe view without fully empty presentation columns."""
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    source = frame.loc[:, ~frame.columns.duplicated(keep="first")]
    selected = list(dict.fromkeys(
        column
        for column in columns
        if column in source.columns
        and _display_column_has_evidence(source[column])
    ))
    display = source.loc[:, selected].copy()
    display.attrs = {}
    return display


def _display_column_has_evidence(series: pd.Series) -> bool:
    """Treat nulls and blank strings as absent, while keeping real zero/False."""
    if series is None or len(series) == 0:
        return False
    observed = series[series.notna()]
    if observed.empty:
        return False
    if (
        pd.api.types.is_object_dtype(observed.dtype)
        or pd.api.types.is_string_dtype(observed.dtype)
    ):
        return bool(observed.map(
            lambda value: (
                bool(value)
                if isinstance(value, (Mapping, list, tuple, set))
                else bool(str(value).strip())
            )
        ).any())
    return True


def streamlit_safe_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return a unique-column, metadata-free, Arrow-compatible display copy."""
    if frame is None:
        return pd.DataFrame()
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame(frame)
    out = frame.loc[:, ~frame.columns.duplicated(keep="first")].copy()
    out = out.loc[:, [
        column
        for column in out.columns
        if _display_column_has_evidence(out[column])
    ]]
    out.attrs = {}
    for column in out.columns:
        series = out[column]
        if not pd.api.types.is_object_dtype(series.dtype):
            continue
        non_null = series[series.notna()]
        if non_null.empty:
            continue
        inferred = pd.api.types.infer_dtype(non_null, skipna=True)
        nested = non_null.map(
            lambda value: isinstance(value, (Mapping, list, tuple, set)),
        ).any()
        if nested or inferred.startswith("mixed"):
            def display_value(value: Any) -> Any:
                if value is None or (
                    isinstance(value, float) and not np.isfinite(value)
                ):
                    return None
                if isinstance(value, Mapping):
                    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                if isinstance(value, (list, tuple, set)):
                    return json.dumps(list(value), ensure_ascii=False, default=str)
                return str(value)
            out[column] = series.map(display_value).astype("string")
        elif inferred in {"string", "unicode", "bytes"}:
            out[column] = series.astype("string")
    return out


def _ranking_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["buy_window"] = out.apply(
        lambda row: " – ".join([value for value in (_text(row.get("buy_window_start")), _text(row.get("buy_window_end"))) if value]),
        axis=1,
    )
    out["entry_zone"] = out.apply(_format_entry_zone, axis=1)
    columns = [
        "rank", "execution_rank", "ticker", "category", "strategy",
        "ranking_priority", "final_score", "execution_priority_score",
        "silent_accumulation_score", "decision_precedence_state", "decision",
        "best_buy_date", "best_buy_date_state", "buy_window", "entry_zone",
        "execution_price", "trigger", "confirmation_level", "stop_loss",
        "tp1", "tp2", "rr1", "rr2", "execution_plan_source",
        "execution_plan_mode", "execution_plan_state",
        "time_cycle_score",
        "time_cycle_confidence", "phase", "eoff_strength", "ihsg_regime",
        "ihsg_consensus", "ihsg_risk_budget_pct", "status",
        "selected_reason", "not_entry_reason", "trigger_waiting",
        "invalidation_reason", "primary_risk",
    ]
    # Ranking attrs include a diagnostic DataFrame.  Streamlit/PyArrow attempts
    # to JSON-serialise attrs, so keep diagnostics on the source ranking only.
    return _safe_display_columns(out, columns)


def _selected_rows(event: Any) -> list[int]:
    try:
        selection = event.selection
        if isinstance(selection, Mapping):
            return list(selection.get("rows", []))
        return list(getattr(selection, "rows", []) or [])
    except Exception:
        return []


def render_ranked_detail(result: Mapping[str, Any], ranking: pd.DataFrame, candidate_id: str) -> None:
    if ranking.empty or not candidate_id:
        return
    selected = ranking[ranking["candidate_id"].eq(candidate_id)] if "candidate_id" in ranking else pd.DataFrame()
    if selected.empty:
        return
    row = selected.iloc[0].to_dict()
    ticker = _text(row.get("ticker"))
    st.markdown(f"### Detail {ticker} — {_text(row.get('strategy'))}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Ranking", f"#{int(_num(row.get('rank'), 0))}")
    c2.metric("Final Score", f"{_num(row.get('final_score', row.get('combined_score')), 0):.1f}")
    c3.metric("Keputusan", _text(row.get("decision")) or "REVIEW")
    c4.metric("Tanggal terbaik", _text(row.get("best_buy_date")) or "Belum valid")
    c5.metric("Time-cycle", f"{_num(row.get('time_cycle_score'), 0):.1f}")
    c6.metric("EOFF", _text(row.get("eoff_strength")) or "LOW")

    st.info(
        f"**Buy window:** {_text(row.get('buy_window_start')) or '—'} sampai {_text(row.get('buy_window_end')) or '—'}  |  "
        f"**Entry:** {_format_entry_zone(row)}  |  "
        f"**Execution:** {_fmt_price(row.get('execution_price'))}  |  "
        f"**Order trigger:** {_fmt_price(row.get('trigger'))}  |  "
        f"**Confirmation:** {_fmt_price(row.get('confirmation_level'))}  |  **SL:** {_fmt_price(row.get('stop_loss'))}  |  "
        f"**TP1/TP2:** {_fmt_price(row.get('tp1'))} / {_fmt_price(row.get('tp2'))}"
    )
    if _text(row.get("best_buy_target_basis")):
        st.caption(f"Target basis: {_text(row.get('best_buy_target_basis'))}")
    if _text(row.get("reason")):
        st.write("**Alasan:**", _text(row.get("reason")))
    if _text(row.get("no_trade")):
        st.warning("**Pembatalan/risiko:** " + _text(row.get("no_trade")))

    all_rows = row.get("all_rows") if isinstance(row.get("all_rows"), list) else []
    if all_rows:
        detail_table = pd.DataFrame(all_rows)
        detail_columns = [
            "category", "strategy", "final_score", "silent_accumulation_score", "base_conviction",
            "multibagger_quality_score", "confidence_adjusted_multibagger_score", "overall_research_confidence",
            "multibagger_lane", "growth_compounder_score",
            "growth_compounder_selection_score", "turnaround_recovery_score",
            "confidence_adjusted_turnaround_score", "turnaround_selection_score",
            "turnaround_research_state", "turnaround_recovery_signals",
            "research_eligible", "research_eligibility_reason",
            "portfolio_allocation_eligible",
            "overall_research_confidence_grade", "data_confidence_score", "fundamental_confidence_score",
            "future_fundamental_confidence_score", "technical_confidence_score", "eoff_confidence_score",
            "execution_readiness_score", "multibagger_candidate_type", "research_recommendation_status",
            "top_positive_drivers", "top_negative_drivers", "scoring_reason_codes",
            "economic_earnings_score", "economic_earnings_state", "minority_leakage_pct",
            "silent_accumulation_state", "accumulation_persistence_score",
            "accumulation_positive_windows_pct", "accumulation_longest_run", "accumulation_regime",
            "absorption_confirmed_days20", "failed_absorption_days20", "persistent_bid_score", "project_stage",
            "quick_buy_score", "time_cycle_score", "time_cycle_confidence", "decision", "status", "reason", "no_trade",
        ]
        detail_display = _safe_display_columns(detail_table, detail_columns)
        st.dataframe(detail_display, hide_index=True, width="stretch")

    prepared = result.get("prepared", {}) or {}
    price_frame = prepared.get(ticker)
    if price_frame is None:
        price_frame = (result.get("all_histories", {}) or {}).get(ticker)
    signals = result.get("signals", pd.DataFrame())
    if isinstance(signals, pd.DataFrame) and not signals.empty and ticker in set(signals.get("ticker", pd.Series(dtype=str)).astype(str)):
        signal_rows = signals[signals["ticker"].eq(ticker)].copy()
        signal_rows = signal_rows.sort_values(
            [column for column in ("analyst_fusion_score", "composite_score", "quality_score") if column in signal_rows.columns],
            ascending=False,
            na_position="last",
        ) if any(column in signal_rows.columns for column in ("analyst_fusion_score", "composite_score", "quality_score")) else signal_rows
        signal = signal_rows.iloc[0].to_dict()
        if price_frame is not None and not price_frame.empty:
            try:
                st.plotly_chart(make_signal_chart(price_frame, signal), width="stretch", key=f"top20_signal_{ticker}")
            except Exception as exc:
                st.caption(f"Chart setup tidak dapat dirender: {exc}")
    else:
        source = row.get("source_row") if isinstance(row.get("source_row"), Mapping) else row
        if price_frame is not None and not price_frame.empty:
            try:
                chart = make_time_cycle_chart(price_frame, source, ticker)
                if chart is not None:
                    st.plotly_chart(chart, width="stretch", key=f"top20_cycle_{ticker}")
            except Exception as exc:
                st.caption(f"Chart time-cycle tidak dapat dirender: {exc}")


def _render_separated_ranking_panel(
    result: Mapping[str, Any],
    ranking: pd.DataFrame,
    *,
    key_prefix: str,
    empty_message: str,
    download_filename: str,
) -> None:
    if ranking.empty:
        st.info(empty_message)
        return
    audit = ranking.attrs.get("candidate_pool_audit", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kandidat", len(ranking))
    c2.metric(
        "Buy/prepare",
        int(ranking["decision"].isin([
            "BUY_ON_CONFIRMED_TRIGGER", "PREPARE_BUY_WAIT_TRIGGER",
        ]).sum()),
    )
    c3.metric(
        "Silent Accum tertinggi",
        f"{_num(ranking['silent_accumulation_score'].max(), 0.0):.1f}",
    )
    c4.metric("Excluded gate", int(audit.get("excluded_total", 0)))

    display = _ranking_display(ranking)
    selected_index: int | None = None
    try:
        event = st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            height=min(640, 38 * (len(display) + 2)),
            on_select="rerun",
            selection_mode="single-row",
            key=f"{key_prefix}_clickable",
            column_config={
                "final_score": st.column_config.NumberColumn(
                    "Final", format="%.1f",
                ),
                "silent_accumulation_score": st.column_config.NumberColumn(
                    "Silent Accum (effective)", format="%.1f",
                ),
                "trigger": st.column_config.NumberColumn(
                    "Trigger", format="Rp %.0f",
                ),
                "stop_loss": st.column_config.NumberColumn(
                    "SL", format="Rp %.0f",
                ),
                "tp1": st.column_config.NumberColumn(
                    "TP1", format="Rp %.0f",
                ),
                "tp2": st.column_config.NumberColumn(
                    "TP2", format="Rp %.0f",
                ),
                "ihsg_risk_budget_pct": st.column_config.NumberColumn(
                    "IHSG risk cap", format="%.0f%%",
                ),
            },
        )
        rows = _selected_rows(event)
        if rows:
            selected_index = int(rows[0])
    except TypeError:
        st.dataframe(display, hide_index=True, width="stretch")

    state_key = f"{key_prefix}_selected_candidate"
    if selected_index is not None and 0 <= selected_index < len(ranking):
        st.session_state[state_key] = ranking.iloc[selected_index]["candidate_id"]
    labels = {
        (
            f"#{int(row['rank'])} {row['ticker']} | "
            f"{row['category']} | {row['strategy']}"
        ): row["candidate_id"]
        for _, row in ranking.iterrows()
    }
    prior_id = st.session_state.get(state_key)
    if prior_id not in set(labels.values()):
        prior_id = next(iter(labels.values()))
    prior_label = next(
        label for label, value in labels.items() if value == prior_id
    )
    selected_label = st.selectbox(
        "Klik baris tabel atau pilih kandidat untuk membuka detail",
        list(labels),
        index=list(labels).index(prior_label),
        key=f"{key_prefix}_selector",
    )
    candidate_id = labels[selected_label]
    st.session_state[state_key] = candidate_id
    render_ranked_detail(result, ranking, candidate_id)
    st.download_button(
        "Download ranking",
        _ranking_display(ranking).to_csv(index=False).encode("utf-8"),
        download_filename,
        "text/csv",
        width="stretch",
        key=f"{key_prefix}_download",
    )


def _render_provisional_multibagger_panel(
    result: Mapping[str, Any],
    queue: pd.DataFrame,
    *,
    key_prefix: str,
    download_filename: str,
) -> None:
    """Render a transparent fallback when the qualified radar has zero rows."""
    if queue.empty:
        focus = result.get("focus_screens", {}) or {}
        pending = focus.get("multibagger_data_pending", pd.DataFrame())
        if isinstance(pending, pd.DataFrame) and not pending.empty:
            st.warning(
                "Belum ada near-miss yang dapat dinilai. Berikut adalah prioritas "
                "backfill data—ini belum merupakan kandidat Multibagger."
            )
            pending_columns = [
                "research_queue_rank", "ticker",
                "data_refresh_priority_score",
                "effective_silent_accumulation_score",
                "selector_trend_score", "selector_relative_strength_score",
                "adtv20_idr", "next_required_evidence",
                "near_miss_reason", "capital_state",
            ]
            st.dataframe(
                pending.loc[:, [
                    column for column in pending_columns
                    if column in pending.columns
                ]].head(12),
                hide_index=True,
                width="stretch",
            )
        return

    st.warning(
        "Qualified = 0. Scanner menampilkan Provisional Research Queue agar "
        "radar tidak tampak kosong. Baris ini belum lolos gate, bukan sinyal "
        "beli, dan alokasi modal dipaksa 0."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Provisional", len(queue))
    c2.metric(
        "History pending",
        int(queue.get(
            "near_miss_state", pd.Series("", index=queue.index),
        ).eq("STATEMENT_HISTORY_PENDING").sum()),
    )
    c3.metric(
        "Silent Accum tertinggi",
        f"{pd.to_numeric(queue.get('effective_silent_accumulation_score'), errors='coerce').max():.1f}"
        if "effective_silent_accumulation_score" in queue
        and pd.to_numeric(
            queue["effective_silent_accumulation_score"],
            errors="coerce",
        ).notna().any()
        else "—",
    )
    columns = [
        "research_queue_rank", "ticker", "candidate_state",
        "provisional_score", "growth_provisional_priority_score",
        "turnaround_provisional_priority_score",
        "growth_compounder_selection_score",
        "turnaround_selection_score", "growth_compounder_score",
        "turnaround_recovery_score",
        "effective_silent_accumulation_score",
        "selector_trend_score", "selector_relative_strength_score",
        "fundamental_score", "fundamental_coverage",
        "fundamental_data_grade", "fundamental_history_source_count",
        "overall_research_confidence", "near_miss_state",
        "next_required_evidence", "selected_reason",
        "not_entry_reason", "trigger_waiting", "invalidation_reason",
        "primary_risk", "near_miss_reason", "capital_state",
    ]
    display = queue.loc[:, [
        column for column in columns if column in queue.columns
    ]].copy()
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=min(640, 38 * (len(display) + 2)),
        column_config={
            "provisional_score": st.column_config.NumberColumn(
                "Provisional", format="%.1f",
            ),
            "effective_silent_accumulation_score": (
                st.column_config.NumberColumn(
                    "Silent Accum (effective)", format="%.1f",
                )
            ),
            "portfolio_allocation_eligible": (
                st.column_config.CheckboxColumn("Capital eligible")
            ),
        },
    )
    st.download_button(
        "Download provisional research queue",
        display.to_csv(index=False).encode("utf-8"),
        download_filename,
        "text/csv",
        width="stretch",
        key=f"{key_prefix}_download",
    )


def render_top20_dashboard(
    result: Mapping[str, Any],
    single_ticker_runner: Callable[[str, str], Mapping[str, Any]],
) -> None:
    st.subheader("Radar Terpisah — Multibagger dan Swing/Core")
    st.caption(
        "Skor Multibagger 12–36 bulan tidak lagi dibandingkan langsung dengan skor Swing/Core 2–40 hari. "
        "Multibagger dibagi menjadi Growth Compounder dan Turnaround/Cyclical. Status alokasi modal tidak "
        "menghapus kandidat riset; data kedaluwarsa, risiko kritis, dan non-syariah tetap menjadi hard gate. "
        "Setup, Best Buy Date/EOFF, dan IHSG tetap berfungsi sebagai timing/risk cap."
    )
    multibagger_tab, swing_tab, detail_tab = st.tabs([
        "Top Multibagger", "Top Swing/Core", "Bedah Ticker Tanpa CSV",
    ])
    with multibagger_tab:
        growth_tab, turnaround_tab = st.tabs([
            "Growth Compounder", "Turnaround / Cyclical",
        ])
        with growth_tab:
            st.caption(
                "Prioritas: kualitas compounder dan reinvestment runway, "
                "lalu Silent Accumulation efektif; setup dicari setelah seleksi."
            )
            growth_qualified = build_multibagger_ranking(
                result, limit=12, lane="GROWTH_COMPOUNDER",
            )
            _render_separated_ranking_panel(
                result,
                growth_qualified,
                key_prefix="v753_growth_compounder",
                empty_message=(
                    "Belum ada Growth Compounder yang lolos gate riset."
                ),
                download_filename="top_multibagger_growth_compounder.csv",
            )
            if growth_qualified.empty:
                _render_provisional_multibagger_panel(
                    result,
                    build_multibagger_provisional_ranking(
                        result, limit=12, lane="GROWTH_COMPOUNDER",
                    ),
                    key_prefix="v753_growth_provisional",
                    download_filename=(
                        "multibagger_growth_provisional_research.csv"
                    ),
                )
        with turnaround_tab:
            st.caption(
                "Radar riset recovery point-in-time. Kandidat dini dapat masuk "
                "radar, tetapi tidak mendapat alokasi sebelum gate modal A/B lolos."
            )
            turnaround_qualified = build_multibagger_ranking(
                result, limit=12, lane="TURNAROUND_CYCLICAL",
            )
            _render_separated_ranking_panel(
                result,
                turnaround_qualified,
                key_prefix="v753_turnaround",
                empty_message=(
                    "Belum ada Turnaround/Cyclical dengan minimal dua sinyal "
                    "recovery dan tanpa risiko kritis."
                ),
                download_filename="top_multibagger_turnaround_cyclical.csv",
            )
            if turnaround_qualified.empty:
                _render_provisional_multibagger_panel(
                    result,
                    build_multibagger_provisional_ranking(
                        result, limit=12, lane="TURNAROUND_CYCLICAL",
                    ),
                    key_prefix="v753_turnaround_provisional",
                    download_filename=(
                        "multibagger_turnaround_provisional_research.csv"
                    ),
                )
    with swing_tab:
        st.caption(
            "Prioritas: excess return terhadap IHSG, trend/relative strength, "
            "Silent Accumulation, lalu setup dan execution plan."
        )
        _render_separated_ranking_panel(
            result,
            build_swing_ranking(result, limit=20),
            key_prefix="v751_swing_core",
            empty_message=(
                "Belum ada kandidat Swing/Core yang lolos candidate pool."
            ),
            download_filename="top_swing_core.csv",
        )

    with detail_tab:
        st.caption("Ketik satu ticker IDX. Mini scanner menjalankan OHLCV, core setup, fundamental, forward review, Multibagger, dan Time-Cycle tanpa upload CSV.")
        with st.form("single_ticker_deep_dive_form_v660"):
            ticker_text = st.text_input("Ticker IDX", value="ANTM", placeholder="ANTM")
            lookback = st.selectbox("Lookback", ["5y", "3y", "2y"], index=1, key="single_ticker_lookback_v660")
            run = st.form_submit_button("Bedah Ticker", type="primary")
        if run:
            ticker = normalize_idx_ticker(ticker_text)
            if not ticker:
                st.warning("Masukkan ticker yang valid.")
            else:
                with st.spinner(f"Membedah {ticker} tanpa CSV…"):
                    try:
                        st.session_state["single_ticker_detail_v660"] = dict(single_ticker_runner(ticker, lookback))
                    except Exception as exc:
                        st.session_state["single_ticker_detail_v660"] = {"error": str(exc), "ticker": ticker}
        detail_result = st.session_state.get("single_ticker_detail_v660", {})
        if isinstance(detail_result, Mapping) and detail_result:
            if detail_result.get("error"):
                st.error(f"Bedah ticker gagal: {detail_result['error']}")
            else:
                ticker = _text(detail_result.get("ticker"))
                st.markdown(f"### Hasil bedah {ticker}")
                summary = detail_result.get("summary", {}) or {}
                d1, d2, d3, d4, d5 = st.columns(5)
                d1.metric("Keputusan", _text(summary.get("decision")) or "REVIEW")
                d2.metric("Skor", f"{_num(summary.get('score'), 0):.1f}")
                d3.metric("Tanggal terbaik", _text(summary.get("best_buy_date")) or "Belum valid")
                d4.metric("Time-cycle", f"{_num(summary.get('time_cycle_score'), 0):.1f}")
                d5.metric("Multibagger", f"{_num(summary.get('multibagger_score')):.1f}" if np.isfinite(_num(summary.get('multibagger_score'))) else "N/A")
                if _text(summary.get("multibagger_scoring_state")).startswith("DATA_NOT_SCORED"):
                    st.caption("Multibagger N/A: " + (_text(summary.get("multibagger_score_reason")) or "fundamental belum cukup; bukan skor nol."))
                st.info(
                    f"Entry {_format_entry_zone(summary)} | "
                    f"Execution {_fmt_price(summary.get('execution_price'))} | "
                    f"Trigger {_fmt_price(summary.get('trigger'))} | "
                    f"Confirmation {_fmt_price(summary.get('confirmation_level'))} | "
                    f"SL {_fmt_price(summary.get('stop_loss'))} | "
                    f"TP1 {_fmt_price(summary.get('tp1'))} | TP2 {_fmt_price(summary.get('tp2'))}"
                )
                if _text(summary.get("reason")):
                    st.write("**Kesimpulan:**", _text(summary.get("reason")))
                audit_warnings = detail_result.get("audit_warnings", [])
                if isinstance(audit_warnings, list) and audit_warnings:
                    st.warning("Audit data: " + " • ".join(str(value) for value in audit_warnings))
                signals = detail_result.get("signals", pd.DataFrame())
                multibagger = detail_result.get("multibagger", pd.DataFrame())
                tc = detail_result.get("time_cycle", {}) or {}
                sub1, sub2, sub3 = st.tabs(["Core/Swing", "Multibagger", "Time-Cycle"])
                with sub1:
                    if isinstance(signals, pd.DataFrame) and not signals.empty:
                        st.dataframe(signals, hide_index=True, width="stretch")
                    else:
                        st.info("Tidak ada core setup terdeteksi.")
                with sub2:
                    if isinstance(multibagger, pd.DataFrame) and not multibagger.empty:
                        st.dataframe(multibagger, hide_index=True, width="stretch")
                    else:
                        st.info("Data Multibagger belum cukup.")
                with sub3:
                    st.json(tc)
                frame = detail_result.get("history")
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    chart = make_time_cycle_chart(frame, tc, ticker)
                    if chart is not None:
                        st.plotly_chart(chart, width="stretch", key=f"manual_detail_cycle_{ticker}")


def _time_cycle_config_from_scan(cfg: Any) -> TimeCycleConfig:
    return TimeCycleConfig(
        min_bars=int(getattr(cfg, "time_cycle_min_history_bars", 260)),
        lunar_enabled=bool(getattr(cfg, "time_cycle_lunar_enabled", True)),
        eoff_enabled=bool(getattr(cfg, "eoff_enabled", True)),
        eoff_ephemeris_enabled=bool(getattr(cfg, "eoff_ephemeris_enabled", True)),
        eoff_min_fib_cluster=int(getattr(cfg, "eoff_min_fib_cluster", 4)),
        eoff_aspect_orb_deg=float(getattr(cfg, "eoff_aspect_orb_deg", 3.0)),
        eoff_require_astro_fib_confluence=bool(getattr(cfg, "eoff_require_astro_fib_confluence", True)),
    )


def render_top15_dashboard(
    result: Mapping[str, Any],
    single_ticker_runner: Callable[[str, str], Mapping[str, Any]],
) -> None:
    """Backward-compatible alias for the Top-20 dashboard."""
    render_top20_dashboard(result, single_ticker_runner)


def render_time_cycle_main_tab(
    cfg: Any,
    downloader: Callable[[tuple[str, ...], str], tuple[Mapping[str, pd.DataFrame], Any]],
) -> None:
    st.subheader("Time-Cycle Intelligence — Kesimpulan Pembelian")
    st.caption(
        "Ketik satu atau beberapa ticker. Bagian teratas langsung menampilkan tanggal terbaik, buy window, entry, trigger, SL, dan target. "
        "Detail lunar/ephemeris tetap tersedia di bagian bawah untuk audit."
    )
    with st.form("time_cycle_main_form_v660"):
        ticker_text = st.text_input("Ticker", value="ANTM", placeholder="ANTM atau ANTM, MDKA, NCKL", key="time_cycle_main_tickers_v660")
        lookback = st.selectbox("Lookback", ["5y", "3y", "2y"], index=1, key="time_cycle_main_lookback_v660")
        run = st.form_submit_button("Scan Time-Cycle", type="primary")
    if run:
        raw = [part.strip() for part in ticker_text.replace(";", ",").split(",") if part.strip()]
        tickers = list(dict.fromkeys(normalize_idx_ticker(name) for name in raw if normalize_idx_ticker(name)))[:10]
        if not tickers:
            st.warning("Masukkan minimal satu ticker IDX.")
        else:
            with st.spinner(f"Menghitung Time-Cycle dan EOFF untuk {len(tickers)} ticker…"):
                histories, report = downloader(tuple(tickers), lookback)
                rows = []
                config = _time_cycle_config_from_scan(cfg)
                for ticker in tickers:
                    rows.append({"ticker": ticker, **analyze_time_cycle(histories.get(ticker), config=config)})
                st.session_state["time_cycle_main_result_v660"] = {
                    "rows": pd.DataFrame(rows), "histories": dict(histories), "report": report,
                }
    payload = st.session_state.get("time_cycle_main_result_v660", {})
    rows = payload.get("rows", pd.DataFrame()) if isinstance(payload, Mapping) else pd.DataFrame()
    if not isinstance(rows, pd.DataFrame) or rows.empty:
        st.info("Masukkan ticker dan tekan Scan Time-Cycle.")
        return

    quick_columns = [
        "ticker", "quick_buy_action", "best_buy_date", "best_buy_window_start", "best_buy_window_end",
        "best_buy_score", "best_buy_confidence", "best_buy_entry_low", "best_buy_entry_high",
        "best_buy_trigger", "best_buy_stop_loss", "best_buy_tp1", "best_buy_tp2", "best_buy_rr1", "best_buy_rr2",
        "time_cycle_phase", "time_cycle_direction_bias", "eoff_strength_label",
    ]
    quick = rows[[column for column in quick_columns if column in rows.columns]].copy()
    quick = quick.sort_values([column for column in ("best_buy_score", "best_buy_confidence") if column in quick.columns], ascending=False, na_position="last")
    st.markdown("### Kesimpulan cepat")
    st.dataframe(
        quick,
        hide_index=True,
        width="stretch",
        column_config={
            "best_buy_score": st.column_config.NumberColumn("Buy score", format="%.1f"),
            "best_buy_confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
            "best_buy_entry_low": st.column_config.NumberColumn("Entry low", format="Rp %.0f"),
            "best_buy_entry_high": st.column_config.NumberColumn("Entry high", format="Rp %.0f"),
            "best_buy_trigger": st.column_config.NumberColumn("Trigger", format="Rp %.0f"),
            "best_buy_stop_loss": st.column_config.NumberColumn("SL", format="Rp %.0f"),
            "best_buy_tp1": st.column_config.NumberColumn("TP1", format="Rp %.0f"),
            "best_buy_tp2": st.column_config.NumberColumn("TP2", format="Rp %.0f"),
        },
    )
    choices = rows["ticker"].astype(str).tolist()
    ticker = st.selectbox("Ticker detail", choices, key="time_cycle_main_detail_v660")
    detail = rows[rows["ticker"].eq(ticker)].iloc[0].to_dict()
    q1, q2, q3, q4, q5, q6 = st.columns(6)
    q1.metric("Keputusan", _text(detail.get("quick_buy_action")) or "WAIT")
    q2.metric("Tanggal terbaik", _text(detail.get("best_buy_date")) or "Belum valid")
    q3.metric("Buy score", f"{_num(detail.get('best_buy_score'), 0):.1f}")
    q4.metric("Confidence", f"{_num(detail.get('best_buy_confidence'), 0):.1f}%")
    q5.metric("Phase", _text(detail.get("time_cycle_phase")) or "UNKNOWN")
    q6.metric("EOFF", _text(detail.get("eoff_strength_label")) or "LOW")
    st.info(
        f"**Window:** {_text(detail.get('best_buy_window_start')) or '—'} sampai {_text(detail.get('best_buy_window_end')) or '—'} | "
        f"**Entry:** {_fmt_price(detail.get('best_buy_entry_low'))} – {_fmt_price(detail.get('best_buy_entry_high'))} | "
        f"**Trigger:** {_fmt_price(detail.get('best_buy_trigger'))} | **SL:** {_fmt_price(detail.get('best_buy_stop_loss'))} | "
        f"**TP1/TP2:** {_fmt_price(detail.get('best_buy_tp1'))} / {_fmt_price(detail.get('best_buy_tp2'))}"
    )
    if _text(detail.get("best_buy_reason")):
        st.write("**Alasan:**", _text(detail.get("best_buy_reason")))
    if _text(detail.get("best_buy_no_trade_condition")):
        st.warning(_text(detail.get("best_buy_no_trade_condition")))

    histories = payload.get("histories", {}) or {}
    chart = make_time_cycle_chart(histories.get(ticker), detail, ticker)
    if chart is not None:
        st.plotly_chart(chart, width="stretch", key=f"time_cycle_main_chart_{ticker}")

    with st.expander("Audit komponen Time-Cycle, lunar, dan EOFF", expanded=False):
        technical_columns = [
            "time_cycle_state", "time_cycle_score", "time_cycle_confidence", "time_cycle_direction_bias",
            "time_cycle_phase", "dominant_cycle_bars", "pivot_cycle_bars", "autocorr_cycle_bars",
            "spectral_cycle_bars", "cycle_agreement_score", "cycle_historical_hit_rate",
            "cycle_validation_samples", "cycle_median_error_pct", "price_time_confluence_score",
            "fibonacci_time_score", "lunar_phase", "lunar_days_to_major_marker",
            "lunar_historical_hit_rate", "lunar_historical_lift", "eoff_state",
            "eoff_reconstruction_score", "eoff_strength_label", "eoff_signal_active",
            "eoff_validation_path", "eoff_astro_weight_policy", "eoff_core_astro_score", "eoff_adaptive_astro_score",
            "eoff_adaptive_total_weight_pct", "eoff_secondary_prior_share_pct", "eoff_adaptive_active_factors",
            "eoff_phase_base_weight_pct", "eoff_aspect_base_weight_pct", "eoff_declination_base_weight_pct",
            "eoff_ingress_base_weight_pct", "eoff_retrograde_base_weight_pct", "eoff_sun_base_weight_pct",
            "eoff_phase_weight_pct", "eoff_aspect_weight_pct",
            "eoff_direction_bias", "eoff_fib_cluster_count", "eoff_fib_unique_anchor_count",
            "eoff_historical_hit_rate", "eoff_historical_baseline_rate", "eoff_historical_lift",
            "eoff_confluence_historical_hit_rate", "eoff_confluence_historical_events",
            "eoff_public_validation_state", "eoff_public_validation_method",
            "eoff_public_directional_events", "eoff_public_reversal_hit_rate",
            "eoff_public_baseline_rate", "eoff_public_lift",
            "eoff_public_forward_hit_rate", "eoff_public_median_directional_return_pct",
            "eoff_reversal_date", "eoff_ephemeris_date", "eoff_moon_declination_deg",
            "eoff_moon_phase", "eoff_sun_sign", "eoff_sun_annual_cycle_bias",
            "eoff_retrograde_planets", "eoff_stationary_planets", "eoff_retrograde_transition_events", "eoff_ingress_events",
            "eoff_declination_validation_state", "eoff_declination_oos_events", "eoff_declination_oos_lift", "eoff_declination_oos_forward_hit_rate", "eoff_declination_weight_pct",
            "eoff_ingress_validation_state", "eoff_ingress_oos_events", "eoff_ingress_oos_lift", "eoff_ingress_oos_forward_hit_rate", "eoff_ingress_weight_pct",
            "eoff_retrograde_validation_state", "eoff_retrograde_oos_events", "eoff_retrograde_oos_lift", "eoff_retrograde_oos_forward_hit_rate", "eoff_retrograde_weight_pct",
            "eoff_sun_validation_state", "eoff_sun_oos_events", "eoff_sun_oos_lift", "eoff_sun_oos_forward_hit_rate", "eoff_sun_weight_pct",
            "eoff_active_aspects", "eoff_astro_events",
        ]
        audit = pd.DataFrame([{"metric": column, "value": detail.get(column)} for column in technical_columns if column in detail])
        st.dataframe(audit, hide_index=True, width="stretch")
        st.caption(
            "Bobot public-prior EOFF selalu tersedia: Moon phase 25%, planetary aspect 25%, Moon declination 15%, "
            "ingress 10%, retrograde/station 10%, dan Sun annual cycle 15%. Walk-forward hanya memodulasi bobot "
            "sekunder dalam rentang 75%–125% dari prior lalu menormalisasi kembali ke 100%; faktor tidak lagi hilang menjadi 0%. "
            "Ini adalah rekonstruksi clean-room, bukan formula proprietary. Astro tidak dapat membuat order sendiri: Fibonacci time cluster, struktur harga, pattern, momentum, entry, dan invalidation tetap wajib."
        )


__all__ = [
    "DASHBOARD_VERSION", "streamlit_safe_frame",
    "build_top20_ranking",
    "build_top15_ranking",
    "build_multibagger_ranking",
    "build_multibagger_provisional_ranking",
    "build_swing_ranking",
    "render_top20_dashboard",
    "render_top15_dashboard",
    "render_time_cycle_main_tab",
]

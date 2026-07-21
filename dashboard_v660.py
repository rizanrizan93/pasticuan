from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from scanner import make_signal_chart, normalize_idx_ticker
from time_cycle import TimeCycleConfig, analyze_time_cycle, make_time_cycle_chart

DASHBOARD_VERSION = "6.6.5-best-buy-eoff-top20"
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

EOFF_STRONG_LEVELS = {"STRONG", "VERY_STRONG"}
INVALID_BEST_BUY_DATE_TOKENS = {
    "", "N/A", "NA", "NONE", "NAN", "NULL", "UNKNOWN", "UNAVAILABLE",
    "BELUM VALID", "BELUM TERSEDIA", "NO_VALID_DATE", "NO DATE", "—", "-",
}
UNSAFE_RANK_ACTIONS = {"AVOID_NEW_BUY", "NO_ALLOCATION"}

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


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).upper() in {"1", "TRUE", "YES", "Y", "ON"}


def _clip(value: Any, default: float = 50.0) -> float:
    return float(max(0.0, min(100.0, _num(value, default))))


def _fmt_price(value: Any) -> str:
    number = _num(value)
    if not np.isfinite(number):
        return "—"
    return f"Rp{number:,.0f}".replace(",", ".")


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


def _valid_trade_plan(
    entry_low: float,
    entry_high: float,
    trigger: float,
    stop: float,
    tp1: float,
    rr1: float,
    *,
    minimum_rr: float = 1.50,
) -> bool:
    return bool(
        all(np.isfinite(value) and value > 0 for value in (entry_low, entry_high, trigger, stop, tp1, rr1))
        and entry_low <= entry_high
        and trigger > stop
        and tp1 > trigger
        and rr1 >= minimum_rr - 1e-9
    )


def _guard_actionable_plan(
    action: str,
    entry_low: float,
    entry_high: float,
    trigger: float,
    stop: float,
    tp1: float,
    rr1: float,
) -> tuple[str, str]:
    if action not in ACTIONABLE_TRADE_ACTIONS:
        return action, ""
    if _valid_trade_plan(entry_low, entry_high, trigger, stop, tp1, rr1):
        return action, ""
    return "WAIT_FOR_EVIDENCE", "Trade plan belum lengkap/valid (entry, trigger, SL, TP1, atau RR1 minimum 1,50)."


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
    A builder containing only specialty/intraday rows therefore suppressed valid
    daily Swing/Core rows still present in ``result['signals']``.  This function
    always unions both sources, then prefers the richer builder record when the
    same ticker/strategy appears twice.
    """
    specialty = result.get("specialty_screens", {}) or {}
    frames: list[pd.DataFrame] = []
    builder = specialty.get("profit_order_builder", pd.DataFrame())
    if isinstance(builder, pd.DataFrame) and not builder.empty:
        part = builder.copy()
        part["_candidate_source"] = "PROFIT_ORDER_BUILDER"
        part["_source_priority"] = 0
        frames.append(part)
    signals = result.get("signals", pd.DataFrame())
    if isinstance(signals, pd.DataFrame) and not signals.empty:
        part = signals.copy()
        part["_candidate_source"] = "SIGNALS_FALLBACK"
        part["_source_priority"] = 1
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    source = pd.concat(frames, ignore_index=True, sort=False)
    source["_normalized_core_strategy"] = source.apply(
        lambda row: _normalize_core_strategy(_text(row.get("strategy")) or _text(row.get("setup"))), axis=1
    )
    source = source[source["_normalized_core_strategy"].isin(CORE_STRATEGIES)].copy()
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
        if strategy not in CORE_STRATEGIES:
            continue
        conviction = _num(row.get("hybrid_conviction_score"), np.nan)
        if not np.isfinite(conviction):
            conviction = _num(row.get("profit_conviction_score"), np.nan)
        if not np.isfinite(conviction):
            conviction = _num(row.get("analyst_fusion_score"), np.nan)
        if not np.isfinite(conviction):
            conviction = _num(row.get("composite_score", row.get("quality_score")), 0.0)
        quick = _quick_score(row)
        cycle = _clip(row.get("time_cycle_alignment_score", row.get("time_cycle_score")), 50.0)
        data = _clip(row.get("data_quality_score", row.get("data_completeness_score")), 50.0)
        validation = _clip(row.get("validation_score", row.get("validation_gate_score")), 50.0)
        rr2 = _num(row.get("rr2"), np.nan)
        rr_score = 50.0 if not np.isfinite(rr2) else max(0.0, min(100.0, 30.0 + 25.0 * rr2))
        base_score = 0.80 * _clip(conviction) + 0.08 * data + 0.07 * validation + 0.05 * rr_score
        cycle_state = _text(row.get("time_cycle_state")).upper()
        cycle_weight = min(0.10, max(0.0, _num(row.get("time_cycle_effective_weight_pct"), 0.0) / 100.0)) if cycle_state == "VALIDATED" else 0.0
        cycle_signal = 0.60 * quick + 0.40 * cycle
        final_score = (1.0 - cycle_weight) * base_score + cycle_weight * cycle_signal
        action = _quick_action(row)
        if action in {"AVOID_NEW_BUY", "RECALCULATE_WINDOW"}:
            final_score -= 7.0
        low, high = _entry_zone(row)
        trigger = _first_finite(row, "best_buy_trigger", "trigger_price", "stockbit_trigger_price", "entry")
        stop = _first_finite(row, "best_buy_stop_loss", "stop_loss")
        tp1 = _first_finite(row, "best_buy_tp1", "tp1")
        tp2 = _first_finite(row, "best_buy_tp2", "tp2")
        rr1 = _first_finite(row, "best_buy_rr1", "rr1")
        action, plan_warning = _guard_actionable_plan(action, low, high, trigger, stop, tp1, rr1)
        if plan_warning:
            final_score -= 5.0
        reason = _text(row.get("best_buy_reason")) or _text(row.get("conviction_basis")) or _text(row.get("reason"))
        if plan_warning:
            reason = " • ".join(value for value in (reason, plan_warning) if value)
        rows.append({
            "ticker": _text(row.get("ticker")),
            "category": "SWING/CORE",
            "strategy": strategy,
            "combined_score": round(max(0.0, min(100.0, final_score)), 1),
            "base_conviction": round(_clip(conviction), 1),
            "quick_buy_score": round(quick, 1),
            "time_cycle_score": round(_clip(row.get("time_cycle_score"), cycle), 1),
            "time_cycle_confidence": round(_clip(row.get("time_cycle_confidence"), 0.0), 1),
            "time_cycle_state": cycle_state or "UNAVAILABLE",
            "time_cycle_effective_weight_pct": round(100.0 * cycle_weight, 2),
            "decision": action,
            "best_buy_date": _text(row.get("best_buy_date")),
            "buy_window_start": _text(row.get("best_buy_window_start", row.get("next_reversal_window_start"))),
            "buy_window_end": _text(row.get("best_buy_window_end", row.get("next_reversal_window_end"))),
            "entry_low": low,
            "entry_high": high,
            "trigger": trigger,
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "rr1": rr1,
            "rr2": _num(row.get("best_buy_rr2", row.get("rr2"))),
            "phase": _text(row.get("time_cycle_phase")) or "UNKNOWN",
            "eoff_strength": _text(row.get("eoff_strength_label")) or "LOW",
            "status": _text(row.get("decision_state")) or _text(row.get("setup_status")) or _text(row.get("status")),
            "reason": reason,
            "no_trade": _text(row.get("best_buy_no_trade_condition")) or _text(row.get("warnings")),
            "best_buy_target_basis": _text(row.get("best_buy_target_basis")),
            "candidate_source": _text(row.get("_candidate_source")) or "UNKNOWN",
            "source_row": row.to_dict(),
        })
    return rows


def _build_multibagger_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    specialty = result.get("specialty_screens", {}) or {}
    frame = specialty.get("multibagger", pd.DataFrame())
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        conviction = _num(row.get("capital_conviction_score"), np.nan)
        if not np.isfinite(conviction):
            conviction = _num(row.get("multibagger_score"), 0.0)
        quick = _quick_score(row)
        cycle = _clip(row.get("multibagger_time_cycle_score", row.get("time_cycle_score")), 50.0)
        future = _clip(row.get("future_fundamental_impact_score"), 50.0)
        project = _clip(row.get("project_pipeline_score"), 50.0)
        management = _clip(row.get("management_quality_score"), 50.0)
        coverage = _clip(row.get("forward_quality_coverage", row.get("fundamental_coverage")), 50.0)
        base_score = 0.76 * _clip(conviction) + 0.08 * future + 0.06 * project + 0.05 * management + 0.05 * coverage
        cycle_state = _text(row.get("time_cycle_state")).upper()
        cycle_weight = min(0.05, max(0.0, _num(row.get("time_cycle_capital_weight_pct", row.get("time_cycle_effective_weight_pct")), 0.0) / 100.0)) if cycle_state == "VALIDATED" else 0.0
        cycle_signal = 0.55 * quick + 0.45 * cycle
        final_score = (1.0 - cycle_weight) * base_score + cycle_weight * cycle_signal
        action = _quick_action(row, "RESEARCH_AND_WAIT")
        if action in {"AVOID_NEW_BUY", "NO_ALLOCATION"}:
            final_score -= 5.0
        low, high = _entry_zone(row)
        trigger = _first_finite(row, "best_buy_trigger", "entry")
        stop = _first_finite(row, "best_buy_stop_loss", "stop_loss")
        tp1 = _first_finite(row, "best_buy_tp1", "tp1")
        tp2 = _first_finite(row, "best_buy_tp2", "tp2")
        rr1 = _first_finite(row, "best_buy_rr1", "rr1")
        action, plan_warning = _guard_actionable_plan(action, low, high, trigger, stop, tp1, rr1)
        if plan_warning:
            final_score -= 5.0
        reason = _text(row.get("best_buy_reason")) or _text(row.get("allocation_reason")) or _text(row.get("note"))
        if plan_warning:
            reason = " • ".join(value for value in (reason, plan_warning) if value)
        rows.append({
            "ticker": _text(row.get("ticker")),
            "category": "MULTIBAGGER",
            "strategy": _text(row.get("active_setup")) or "MULTIBAGGER",
            "combined_score": round(max(0.0, min(100.0, final_score)), 1),
            "base_conviction": round(_clip(conviction), 1),
            "quick_buy_score": round(quick, 1),
            "time_cycle_score": round(cycle, 1),
            "time_cycle_confidence": round(_clip(row.get("time_cycle_confidence"), 0.0), 1),
            "time_cycle_state": cycle_state or "UNAVAILABLE",
            "time_cycle_effective_weight_pct": round(100.0 * cycle_weight, 2),
            "decision": action,
            "best_buy_date": _text(row.get("best_buy_date")),
            "buy_window_start": _text(row.get("best_buy_window_start", row.get("next_reversal_window_start"))),
            "buy_window_end": _text(row.get("best_buy_window_end", row.get("next_reversal_window_end"))),
            "entry_low": low,
            "entry_high": high,
            "trigger": trigger,
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "rr1": rr1,
            "rr2": _num(row.get("best_buy_rr2")),
            "phase": _text(row.get("time_cycle_phase")) or "UNKNOWN",
            "eoff_strength": _text(row.get("eoff_strength_label")) or "LOW",
            "status": _text(row.get("capital_tier")) or _text(row.get("multibagger_status")),
            "reason": reason,
            "no_trade": _text(row.get("best_buy_no_trade_condition")) or _text(row.get("red_flags")),
            "best_buy_target_basis": _text(row.get("best_buy_target_basis")),
            "source_row": row.to_dict(),
        })
    return rows


def build_top20_ranking(result: Mapping[str, Any], limit: int = 20) -> pd.DataFrame:
    """Rank unique tickers with timing evidence first.

    Priority order is deliberately explicit:
    1) valid best-buy date AND EOFF STRONG/VERY_STRONG;
    2) valid best-buy date;
    3) EOFF STRONG/VERY_STRONG;
    4) remaining candidates by combined quality.

    Unsafe actions remain below valid candidates.  Within each tier, quality
    score—not strategy label—determines rank.  One ticker occupies one Top-20
    slot; all Multibagger and Swing/Core rows remain available in ``all_rows``.
    """
    raw = _build_swing_rows(result) + _build_multibagger_rows(result)
    if not raw:
        return pd.DataFrame()
    frame = pd.DataFrame(raw)
    frame = frame[frame["ticker"].astype(str).str.len().gt(0)].copy()
    frame = frame.sort_values("combined_score", ascending=False, na_position="last")
    frame = frame.drop_duplicates(["ticker", "category", "strategy"], keep="first").copy()
    frame["has_best_buy_date"] = frame.apply(
        lambda row: _has_valid_best_buy_date(row.get("best_buy_date"))
        and _text(row.get("decision")).upper() != "RECALCULATE_WINDOW",
        axis=1,
    )
    frame["eoff_strong"] = frame["eoff_strength"].map(_is_eoff_strong)
    timing = frame.apply(
        lambda row: _timing_priority_label(
            row.get("best_buy_date"), row.get("eoff_strength"), row.get("decision")
        ),
        axis=1,
    )
    frame["timing_priority"] = [value[0] for value in timing]
    frame["ranking_priority"] = [value[1] for value in timing]
    frame["_safety_rank"] = frame["decision"].isin(UNSAFE_RANK_ACTIONS).astype(int)
    frame["_action_rank"] = frame["decision"].map(ACTION_PRIORITY).fillna(8)
    frame = frame.sort_values(
        ["_safety_rank", "timing_priority", "combined_score", "base_conviction", "_action_rank"],
        ascending=[True, True, False, False, True],
        na_position="last",
        kind="stable",
    )

    ticker_details = {
        ticker: group.sort_values(
            ["_safety_rank", "timing_priority", "combined_score", "base_conviction", "_action_rank"],
            ascending=[True, True, False, False, True],
            na_position="last",
            kind="stable",
        ).to_dict("records")
        for ticker, group in frame.groupby("ticker", sort=False)
    }

    # One stock = one ranking slot.  The first row is the highest-priority
    # representation, while category/strategy labels summarize every valid row.
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
        pool.loc[pool["category"].str.contains("MULTIBAGGER", na=False), "combined_score"].max(), np.nan
    )
    best_swing = _num(
        pool.loc[pool["category"].str.contains("SWING/CORE", na=False), "combined_score"].max(), np.nan
    )
    top_limit = max(1, int(limit))
    out = out.head(top_limit).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    final = out.drop(columns=["_action_rank", "_safety_rank"])
    cutoff = _num(final["combined_score"].min(), np.nan) if not final.empty else np.nan
    final.attrs["candidate_pool_audit"] = {
        "eligible_total": int(len(pool)),
        "eligible_multibagger": eligible_multibagger,
        "eligible_swing_core": eligible_swing,
        "top20_multibagger": int(final["category"].str.contains("MULTIBAGGER", na=False).sum()),
        "top20_swing_core": int(final["category"].str.contains("SWING/CORE", na=False).sum()),
        "top20_date_eoff_strong": int((final["timing_priority"] == 0).sum()),
        "top20_with_best_buy_date": int(final["has_best_buy_date"].sum()),
        "top20_eoff_strong": int(final["eoff_strong"].sum()),
        "best_multibagger_score": best_multibagger,
        "best_swing_core_score": best_swing,
        "top20_cutoff_score": cutoff,
        "forced_category_quota": False,
        "unique_ticker_ranking": True,
        "priority_policy": "BEST_BUY_DATE_AND_EOFF_STRONG_FIRST",
    }
    return final


def build_top15_ranking(result: Mapping[str, Any], limit: int = 20) -> pd.DataFrame:
    """Backward-compatible alias.  The dashboard now defaults to Top 20."""
    return build_top20_ranking(result, limit=limit)


def _ranking_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["buy_window"] = out.apply(
        lambda row: " – ".join([value for value in (_text(row.get("buy_window_start")), _text(row.get("buy_window_end"))) if value]),
        axis=1,
    )
    out["entry_zone"] = out.apply(
        lambda row: f"{_fmt_price(row.get('entry_low'))} – {_fmt_price(row.get('entry_high'))}", axis=1,
    )
    columns = [
        "rank", "ticker", "category", "strategy", "ranking_priority", "combined_score", "decision",
        "best_buy_date", "buy_window", "entry_zone", "trigger", "stop_loss", "tp1", "tp2",
        "time_cycle_score", "time_cycle_confidence", "phase", "eoff_strength", "status",
    ]
    return out[[column for column in columns if column in out.columns]]


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
    c2.metric("Skor gabungan", f"{_num(row.get('combined_score'), 0):.1f}")
    c3.metric("Keputusan", _text(row.get("decision")) or "REVIEW")
    c4.metric("Tanggal terbaik", _text(row.get("best_buy_date")) or "Belum valid")
    c5.metric("Time-cycle", f"{_num(row.get('time_cycle_score'), 0):.1f}")
    c6.metric("EOFF", _text(row.get("eoff_strength")) or "LOW")

    st.info(
        f"**Buy window:** {_text(row.get('buy_window_start')) or '—'} sampai {_text(row.get('buy_window_end')) or '—'}  |  "
        f"**Entry zone:** {_fmt_price(row.get('entry_low'))} – {_fmt_price(row.get('entry_high'))}  |  "
        f"**Trigger:** {_fmt_price(row.get('trigger'))}  |  **SL:** {_fmt_price(row.get('stop_loss'))}  |  "
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
            "category", "strategy", "combined_score", "base_conviction", "quick_buy_score",
            "time_cycle_score", "time_cycle_confidence", "decision", "status", "reason", "no_trade",
        ]
        st.dataframe(detail_table[[c for c in detail_columns if c in detail_table.columns]], hide_index=True, width="stretch")

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


def render_top20_dashboard(
    result: Mapping[str, Any],
    single_ticker_runner: Callable[[str, str], Mapping[str, Any]],
) -> None:
    st.subheader("Top 20 Saham Terbaik — Best Buy Date & EOFF Priority")
    st.caption(
        "Urutan pertama diberikan kepada ticker yang memiliki Best Buy Date valid sekaligus EOFF STRONG/VERY_STRONG; "
        "berikutnya Best Buy Date saja, EOFF strong saja, lalu kandidat lain berdasarkan combined quality. "
        "Satu ticker hanya memakai satu slot. BPJS, BSJP, ARA, dan intraday tidak masuk."
    )
    top_tab, detail_tab = st.tabs(["Top 20 Ranking", "Bedah Ticker Tanpa CSV"])
    with top_tab:
        ranking = build_top20_ranking(result, limit=20)
        if ranking.empty:
            st.info("Belum ada kandidat Multibagger atau swing/core yang dapat diranking dari scan saat ini.")
        else:
            audit = ranking.attrs.get("candidate_pool_audit", {})
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            m1.metric("Top 20", len(ranking))
            m2.metric("Buy/prepare", int(ranking["decision"].isin(["BUY_ON_CONFIRMED_TRIGGER", "PREPARE_BUY_WAIT_TRIGGER"]).sum()))
            m3.metric("Date + EOFF strong", int(audit.get("top20_date_eoff_strong", 0)))
            m4.metric("Top w/ Multibagger", int(audit.get("top20_multibagger", 0)))
            m5.metric("Top w/ Swing/Core", int(audit.get("top20_swing_core", 0)))
            m6.metric("Pool Multibagger", int(audit.get("eligible_multibagger", 0)))
            m7.metric("Pool Swing/Core", int(audit.get("eligible_swing_core", 0)))
            swing_pool = int(audit.get("eligible_swing_core", 0))
            swing_top = int(audit.get("top20_swing_core", 0))
            if swing_pool == 0:
                st.warning(
                    "Tidak ada kandidat Swing/Core yang masuk candidate pool. Periksa Core Setups/Signals; "
                    "ranking tidak lagi bergantung eksklusif pada Profit Order Builder."
                )
            elif swing_top == 0:
                st.info(
                    f"Ada {swing_pool} kandidat Swing/Core di pool, tetapi tidak masuk Top 20 murni. "
                    f"Skor Swing/Core terbaik {_num(audit.get('best_swing_core_score'), 0.0):.1f}; "
                    f"cutoff Top 20 {_num(audit.get('top20_cutoff_score'), 0.0):.1f}. Tidak ada kuota kategori paksa."
                )
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
                    key="top20_clickable_ranking_v665",
                    column_config={
                        "combined_score": st.column_config.NumberColumn("Final", format="%.1f"),
                        "trigger": st.column_config.NumberColumn("Trigger", format="Rp %.0f"),
                        "stop_loss": st.column_config.NumberColumn("SL", format="Rp %.0f"),
                        "tp1": st.column_config.NumberColumn("TP1", format="Rp %.0f"),
                        "tp2": st.column_config.NumberColumn("TP2", format="Rp %.0f"),
                        "time_cycle_score": st.column_config.NumberColumn("Cycle", format="%.1f"),
                        "time_cycle_confidence": st.column_config.NumberColumn("Cycle conf", format="%.1f%%"),
                    },
                )
                rows = _selected_rows(event)
                if rows:
                    selected_index = int(rows[0])
            except TypeError:
                st.dataframe(display, hide_index=True, width="stretch")
            if selected_index is not None and 0 <= selected_index < len(ranking):
                st.session_state["top20_selected_candidate_v665"] = ranking.iloc[selected_index]["candidate_id"]
            labels = {
                f"#{int(row['rank'])} {row['ticker']} | {row['category']} | {row['strategy']}": row["candidate_id"]
                for _, row in ranking.iterrows()
            }
            prior_id = st.session_state.get("top20_selected_candidate_v665")
            if prior_id not in set(labels.values()):
                prior_id = next(iter(labels.values()))
            prior_label = next(label for label, value in labels.items() if value == prior_id)
            selected_label = st.selectbox(
                "Klik baris tabel atau pilih setup untuk membuka detail",
                list(labels),
                index=list(labels).index(prior_label),
                key="top20_detail_selector_v665",
            )
            candidate_id = labels[selected_label]
            st.session_state["top20_selected_candidate_v665"] = candidate_id
            render_ranked_detail(result, ranking, candidate_id)
            st.download_button(
                "Download Top 20 ranking",
                _ranking_display(ranking).to_csv(index=False).encode("utf-8"),
                "top20_best_buy_eoff_multibagger_swing_core.csv",
                "text/csv",
                width="stretch",
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
                d5.metric("Multibagger", f"{_num(summary.get('multibagger_score'), 0):.1f}")
                st.info(
                    f"Entry {_fmt_price(summary.get('entry_low'))} – {_fmt_price(summary.get('entry_high'))} | "
                    f"Trigger {_fmt_price(summary.get('trigger'))} | SL {_fmt_price(summary.get('stop_loss'))} | "
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
    "DASHBOARD_VERSION",
    "build_top20_ranking",
    "build_top15_ranking",
    "render_top20_dashboard",
    "render_top15_dashboard",
    "render_time_cycle_main_tab",
]

from __future__ import annotations

"""Two-stage routing helpers for the v9 macro-first scanner.

The module is deliberately decision-neutral.  It only decides which tickers
receive expensive external enrichment after the full universe has completed a
local/cache-first ranking pass.  Routing scores have zero production weight. Final decisions are owned by
``simple_focus.py``.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TWO_STAGE_PIPELINE_VERSION = "9.0.0-two-stage-routing-only-v1"


def _ticker(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _unique_tickers(values: Iterable[object]) -> list[str]:
    return [value for value in dict.fromkeys(_ticker(item) for item in values) if value]


def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().upper() in {"1", "TRUE", "YES", "Y", "OK", "ELIGIBLE"}


def _numeric(frame: pd.DataFrame, names: Sequence[str], default: float = np.nan) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _ranked_tickers(
    frame: pd.DataFrame | None,
    *,
    score_columns: Sequence[str],
    eligibility_columns: Sequence[str] = (),
) -> list[str]:
    if frame is None or frame.empty or "ticker" not in frame.columns:
        return []
    out = frame.copy()
    out["ticker"] = out["ticker"].map(_ticker)
    out = out[out["ticker"].ne("")]
    if out.empty:
        return []
    eligible = pd.Series(True, index=out.index)
    for column in eligibility_columns:
        if column not in out.columns:
            continue
        values = out[column]
        if pd.api.types.is_bool_dtype(values):
            eligible &= values.fillna(False)
        else:
            eligible &= values.fillna("").astype(str).str.upper().isin(
                {"TRUE", "1", "YES", "Y", "ELIGIBLE", "WATCHLIST", "ENTRY_PLAN_READY", "EXECUTION_READY"}
            )
    score = _numeric(out, score_columns)
    out["_eligible"] = eligible.astype(int)
    out["_score"] = score
    out = out.sort_values(
        ["_eligible", "_score", "ticker"],
        ascending=[False, False, True],
        kind="stable",
        na_position="last",
    )
    return out.drop_duplicates("ticker", keep="first")["ticker"].tolist()



def _percentile_score(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna()
    out = pd.Series(50.0, index=series.index, dtype=float)
    if valid.sum() >= 2:
        ranked = numeric.loc[valid].rank(method="average", pct=True) * 100.0
        out.loc[valid] = ranked if higher_is_better else 100.0 - ranked
    elif valid.sum() == 1:
        out.loc[valid] = 50.0
    return out.clip(0.0, 100.0)


def build_lightweight_preliminary_focus(
    prepared: Mapping[str, pd.DataFrame],
    *,
    fundamentals: pd.DataFrame | None = None,
    signals: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build a cheap cache-first pre-ranking for the full universe.

    This is only a routing score for Stage-B enrichment.  It has zero production
    weight and never replaces the final decision score in ``simple_focus``.
    Narrative is intentionally omitted because it is the expensive evidence that
    Stage B is designed to refresh.
    """
    rows: list[dict[str, object]] = []
    for ticker, frame in prepared.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        last = frame.iloc[-1]
        rows.append({
            "ticker": _ticker(ticker),
            "close": pd.to_numeric(pd.Series([last.get("Close")]), errors="coerce").iloc[0],
            "ema20": pd.to_numeric(pd.Series([last.get("EMA20")]), errors="coerce").iloc[0],
            "ema50": pd.to_numeric(pd.Series([last.get("EMA50")]), errors="coerce").iloc[0],
            "ema200": pd.to_numeric(pd.Series([last.get("EMA200")]), errors="coerce").iloc[0],
            "vwap20": pd.to_numeric(pd.Series([last.get("VWAP20")]), errors="coerce").iloc[0],
            "roc20": pd.to_numeric(pd.Series([last.get("ROC20")]), errors="coerce").iloc[0],
            "roc60": pd.to_numeric(pd.Series([last.get("ROC60")]), errors="coerce").iloc[0],
            "roc120": pd.to_numeric(pd.Series([last.get("ROC120")]), errors="coerce").iloc[0],
            "relative_strength60": pd.to_numeric(pd.Series([last.get("REL_STRENGTH60")]), errors="coerce").iloc[0],
            "distance_52w_high": pd.to_numeric(pd.Series([last.get("DIST_52W_HIGH")]), errors="coerce").iloc[0],
            "cmf20": pd.to_numeric(pd.Series([last.get("CMF20")]), errors="coerce").iloc[0],
            "cmf60": pd.to_numeric(pd.Series([last.get("CMF60")]), errors="coerce").iloc[0],
            "obv_slope20": pd.to_numeric(pd.Series([last.get("OBV_SLOPE20")]), errors="coerce").iloc[0],
            "adl_slope20": pd.to_numeric(pd.Series([last.get("ADL_SLOPE20")]), errors="coerce").iloc[0],
            "adtv20": pd.to_numeric(pd.Series([last.get("ADTV20")]), errors="coerce").iloc[0],
            "atr_pct": pd.to_numeric(pd.Series([last.get("ATR_PCT")]), errors="coerce").iloc[0],
        })
    base = pd.DataFrame(rows)
    if base.empty:
        empty = pd.DataFrame(columns=["ticker"])
        return {"multibagger": empty.copy(), "profit_order_builder": empty.copy(), "preliminary_audit": empty.copy()}

    close = pd.to_numeric(base["close"], errors="coerce")
    trend_conditions = pd.DataFrame({
        "close_above_ema20": close > pd.to_numeric(base["ema20"], errors="coerce"),
        "close_above_ema50": close > pd.to_numeric(base["ema50"], errors="coerce"),
        "ema20_above_ema50": pd.to_numeric(base["ema20"], errors="coerce") >= pd.to_numeric(base["ema50"], errors="coerce"),
        "ema50_near_above_ema200": pd.to_numeric(base["ema50"], errors="coerce") >= 0.97 * pd.to_numeric(base["ema200"], errors="coerce"),
        "close_above_vwap20": close >= pd.to_numeric(base["vwap20"], errors="coerce"),
    }).fillna(False)
    base["preliminary_trend_score"] = trend_conditions.mean(axis=1) * 100.0
    base["preliminary_momentum_score"] = pd.concat([
        _percentile_score(base["roc20"]),
        _percentile_score(base["roc60"]),
        _percentile_score(base["roc120"]),
        _percentile_score(base["relative_strength60"]),
        _percentile_score(base["distance_52w_high"]),
    ], axis=1).mean(axis=1)
    base["preliminary_flow_score"] = pd.concat([
        _percentile_score(base["cmf20"]),
        _percentile_score(base["cmf60"]),
        _percentile_score(base["obv_slope20"]),
        _percentile_score(base["adl_slope20"]),
    ], axis=1).mean(axis=1)
    base["preliminary_liquidity_score"] = _percentile_score(np.log1p(pd.to_numeric(base["adtv20"], errors="coerce").clip(lower=0)))
    atr = pd.to_numeric(base["atr_pct"], errors="coerce")
    base["preliminary_risk_score"] = (100.0 - 100.0 * ((atr - 0.025).abs() / 0.075)).clip(0.0, 100.0).fillna(50.0)

    fund_map: dict[str, dict[str, object]] = {}
    if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty and "ticker" in fundamentals:
        fund = fundamentals.copy()
        fund["ticker"] = fund["ticker"].map(_ticker)
        fund_map = fund.drop_duplicates("ticker", keep="last").set_index("ticker").to_dict("index")
    fund_scores = []
    fund_coverages = []
    for ticker in base["ticker"]:
        row = fund_map.get(ticker, {})
        score = np.nan
        for column in ("fundamental_score", "fundamental_overall_score", "future_fundamental_score"):
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if pd.notna(value):
                score = float(value)
                break
        coverage = 0.0
        for column in ("fundamental_coverage", "fundamental_overall_coverage", "fundamental_metric_coverage_pct"):
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if pd.notna(value):
                coverage = float(value)
                break
        fund_scores.append(score)
        fund_coverages.append(max(0.0, min(100.0, coverage)))
    base["cached_fundamental_score"] = pd.Series(fund_scores, index=base.index, dtype=float)
    base["cached_fundamental_coverage_pct"] = pd.Series(fund_coverages, index=base.index, dtype=float)
    raw_fund = base["cached_fundamental_score"].fillna(50.0).clip(0.0, 100.0)
    base["preliminary_fundamental_score"] = (
        50.0 + (raw_fund - 50.0) * (base["cached_fundamental_coverage_pct"] / 100.0)
    ).clip(0.0, 100.0)

    signal_map: dict[str, dict[str, object]] = {}
    if isinstance(signals, pd.DataFrame) and not signals.empty and "ticker" in signals:
        sig = signals.copy()
        sig["ticker"] = sig["ticker"].map(_ticker)
        quality = _numeric(sig, ("quality_score", "score", "technical_score", "confidence_score"), default=np.nan)
        sig["_quality"] = quality
        status = sig.get("status", sig.get("setup_status", pd.Series("", index=sig.index))).fillna("").astype(str).str.upper()
        sig["_status_bonus"] = status.map({
            "EXECUTION_READY": 100.0,
            "READY_FOR_PRICE_VERIFY": 90.0,
            "READY_FOR_STOCKBIT_VERIFY": 88.0,
            "ENTRY_PLAN_READY": 82.0,
            "PENDING_DATA": 65.0,
            "WATCHLIST": 60.0,
            "BLOCKED_CONTEXT": 35.0,
            "REJECT": 0.0,
        }).fillna(50.0)
        sig["_signal_route_score"] = sig["_quality"].fillna(sig["_status_bonus"])
        sig = sig.sort_values(["ticker", "_signal_route_score"], ascending=[True, False], kind="stable")
        signal_map = sig.drop_duplicates("ticker", keep="first").set_index("ticker").to_dict("index")
    base["preliminary_signal_score"] = [
        float(signal_map.get(ticker, {}).get("_signal_route_score", np.nan))
        if pd.notna(signal_map.get(ticker, {}).get("_signal_route_score", np.nan))
        else 50.0
        for ticker in base["ticker"]
    ]
    base["preliminary_signal_score"] = pd.to_numeric(base["preliminary_signal_score"], errors="coerce").fillna(50.0).clip(0.0, 100.0)

    base["preliminary_multibagger_score"] = (
        0.45 * base["preliminary_fundamental_score"]
        + 0.20 * base["preliminary_momentum_score"]
        + 0.20 * base["preliminary_flow_score"]
        + 0.10 * base["preliminary_trend_score"]
        + 0.05 * base["preliminary_liquidity_score"]
    ).round(1)
    base["preliminary_core_score"] = (
        0.35 * base["preliminary_signal_score"]
        + 0.20 * base["preliminary_trend_score"]
        + 0.20 * base["preliminary_momentum_score"]
        + 0.15 * base["preliminary_flow_score"]
        + 0.05 * base["preliminary_liquidity_score"]
        + 0.05 * base["preliminary_risk_score"]
    ).round(1)
    base["preliminary_only"] = True
    base["production_weight_pct"] = 0.0
    base["two_stage_pipeline_version"] = TWO_STAGE_PIPELINE_VERSION

    multibagger = base.sort_values(
        ["preliminary_multibagger_score", "preliminary_flow_score", "ticker"],
        ascending=[False, False, True], kind="stable",
    ).reset_index(drop=True)
    core = base.sort_values(
        ["preliminary_core_score", "preliminary_signal_score", "ticker"],
        ascending=[False, False, True], kind="stable",
    ).reset_index(drop=True)
    return {
        "multibagger": multibagger,
        "profit_order_builder": core,
        "preliminary_audit": base.reset_index(drop=True),
    }


@dataclass(frozen=True)
class ShortlistConfig:
    max_tickers: int = 60
    multibagger_quota: int = 30
    core_quota: int = 30
    technical_rescue_quota: int = 15


def build_enrichment_shortlist(
    universe: Iterable[object],
    *,
    preliminary_focus: Mapping[str, pd.DataFrame] | None = None,
    signals: pd.DataFrame | None = None,
    portfolio_tickers: Iterable[object] = (),
    config: ShortlistConfig | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Return a bounded, auditable enrichment shortlist.

    Portfolio holdings always receive priority.  The remaining slots are filled
    from preliminary Multibagger, Core Swing, and a raw technical rescue lane.
    The rescue lane prevents incomplete cached fundamentals/news from removing a
    technically strong new candidate before live enrichment can inspect it.
    """

    cfg = config or ShortlistConfig()
    universe_order = _unique_tickers(universe)
    universe_set = set(universe_order)
    focus = preliminary_focus or {}

    sources: dict[str, list[str]] = {
        "PORTFOLIO": [t for t in _unique_tickers(portfolio_tickers) if t in universe_set],
        "MULTIBAGGER_PRELIMINARY": _ranked_tickers(
            focus.get("multibagger") if isinstance(focus, Mapping) else None,
            score_columns=("preliminary_multibagger_score", "v8_strategic_score", "final_score", "growth_compounder_selection_score", "turnaround_selection_score"),
        )[: max(0, int(cfg.multibagger_quota))],
        "CORE_PRELIMINARY": _ranked_tickers(
            focus.get("profit_order_builder") if isinstance(focus, Mapping) else None,
            score_columns=("preliminary_core_score", "core_priority_score", "final_score", "profit_conviction_score", "quality_score"),
        )[: max(0, int(cfg.core_quota))],
        "TECHNICAL_RESCUE": _ranked_tickers(
            signals,
            score_columns=("quality_score", "score", "technical_score", "confidence_score"),
        )[: max(0, int(cfg.technical_rescue_quota))],
    }

    reasons: dict[str, list[str]] = {}
    source_rank: dict[tuple[str, str], int] = {}
    ordered: list[str] = []
    max_tickers = max(1, int(cfg.max_tickers))

    for source_name, names in sources.items():
        for rank, ticker in enumerate(names, start=1):
            if ticker not in universe_set:
                continue
            reasons.setdefault(ticker, []).append(source_name)
            source_rank[(ticker, source_name)] = rank
            if ticker not in ordered and len(ordered) < max_tickers:
                ordered.append(ticker)

    # Fill any unused capacity in original universe order.  This makes small
    # universes complete and keeps the selection deterministic.
    for ticker in universe_order:
        if len(ordered) >= max_tickers:
            break
        if ticker not in ordered:
            ordered.append(ticker)
            reasons.setdefault(ticker, []).append("UNIVERSE_CAPACITY_FILL")

    mb = focus.get("multibagger", pd.DataFrame()) if isinstance(focus, Mapping) else pd.DataFrame()
    core = focus.get("profit_order_builder", pd.DataFrame()) if isinstance(focus, Mapping) else pd.DataFrame()
    mb_map = {}
    core_map = {}
    if isinstance(mb, pd.DataFrame) and not mb.empty and "ticker" in mb:
        tmp = mb.copy()
        tmp["ticker"] = tmp["ticker"].map(_ticker)
        mb_map = tmp.drop_duplicates("ticker", keep="first").set_index("ticker").to_dict("index")
    if isinstance(core, pd.DataFrame) and not core.empty and "ticker" in core:
        tmp = core.copy()
        tmp["ticker"] = tmp["ticker"].map(_ticker)
        core_map = tmp.drop_duplicates("ticker", keep="first").set_index("ticker").to_dict("index")

    rows: list[dict[str, object]] = []
    for final_rank, ticker in enumerate(ordered, start=1):
        mb_row = mb_map.get(ticker, {})
        core_row = core_map.get(ticker, {})
        rows.append({
            "ticker": ticker,
            "enrichment_rank": final_rank,
            "enrichment_reasons": " | ".join(reasons.get(ticker, [])),
            "portfolio_priority": "PORTFOLIO" in reasons.get(ticker, []),
            "preliminary_multibagger_rank": source_rank.get((ticker, "MULTIBAGGER_PRELIMINARY"), np.nan),
            "preliminary_core_rank": source_rank.get((ticker, "CORE_PRELIMINARY"), np.nan),
            "technical_rescue_rank": source_rank.get((ticker, "TECHNICAL_RESCUE"), np.nan),
            "preliminary_multibagger_score": mb_row.get("preliminary_multibagger_score", mb_row.get("v8_strategic_score", mb_row.get("final_score", np.nan))),
            "preliminary_core_score": core_row.get("preliminary_core_score", core_row.get("core_priority_score", core_row.get("final_score", np.nan))),
            "two_stage_pipeline_version": TWO_STAGE_PIPELINE_VERSION,
        })
    return ordered, pd.DataFrame(rows)


def default_refresh_state_path() -> Path:
    override = os.getenv("SCANNER_TWO_STAGE_REFRESH_STATE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / ".scanner_cache" / "two_stage_refresh_state.json"


def _load_state(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return {}


def _write_state(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass


def plan_round_robin_refresh(
    universe: Iterable[object],
    *,
    priority_tickers: Iterable[object] = (),
    max_tickers: int = 40,
    priority_quota: int = 24,
    state_key: str = "official_fundamental",
    state_path: str | Path | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Plan bounded refresh work and persist a cursor for the next scan.

    The first slots are reserved for the current shortlist; remaining slots
    rotate through the full universe.  Therefore a 400-ticker universe receives
    broad official refresh coverage across scans without making every daily scan
    perform 400 external filing requests.
    """

    universe_order = _unique_tickers(universe)
    universe_set = set(universe_order)
    cap = max(0, min(int(max_tickers), len(universe_order)))
    if cap == 0:
        return [], pd.DataFrame(columns=["ticker", "refresh_rank", "refresh_reason"])

    priorities = [t for t in _unique_tickers(priority_tickers) if t in universe_set]
    priority_cap = min(cap, max(0, int(priority_quota)))
    selected = priorities[:priority_cap]
    reasons = {ticker: "SHORTLIST_PRIORITY" for ticker in selected}

    path = Path(state_path).expanduser() if state_path is not None else default_refresh_state_path()
    state = _load_state(path)
    cursors = state.get("cursors", {}) if isinstance(state.get("cursors", {}), dict) else {}
    cursor = int(cursors.get(state_key, 0) or 0) % max(len(universe_order), 1)

    visited = 0
    position = cursor
    while len(selected) < cap and visited < len(universe_order):
        ticker = universe_order[position]
        if ticker not in selected:
            selected.append(ticker)
            reasons[ticker] = "ROUND_ROBIN_BACKFILL"
        position = (position + 1) % len(universe_order)
        visited += 1

    cursors[state_key] = position
    state.update({
        "version": TWO_STAGE_PIPELINE_VERSION,
        "cursors": cursors,
        "last_universe_size": len(universe_order),
        "last_plan_size": len(selected),
    })
    _write_state(path, state)

    audit = pd.DataFrame([
        {
            "ticker": ticker,
            "refresh_rank": rank,
            "refresh_reason": reasons.get(ticker, "ROUND_ROBIN_BACKFILL"),
            "state_key": state_key,
            "next_cursor": position,
            "two_stage_pipeline_version": TWO_STAGE_PIPELINE_VERSION,
        }
        for rank, ticker in enumerate(selected, start=1)
    ])
    return selected, audit


def build_two_stage_coverage_audit(
    universe: Iterable[object],
    *,
    shortlist: Iterable[object],
    fundamentals: pd.DataFrame | None,
    news_review: pd.DataFrame | None,
    market_status: pd.DataFrame | None,
) -> pd.DataFrame:
    """Produce a compact per-ticker coverage map for the UI and exports."""

    names = _unique_tickers(universe)
    shortlist_set = set(_unique_tickers(shortlist))

    def lookup(frame: pd.DataFrame | None) -> dict[str, dict[str, object]]:
        if frame is None or frame.empty or "ticker" not in frame.columns:
            return {}
        out = frame.copy()
        out["ticker"] = out["ticker"].map(_ticker)
        return out.drop_duplicates("ticker", keep="last").set_index("ticker").to_dict("index")

    fund_map = lookup(fundamentals)
    news_map = lookup(news_review)
    market_map = lookup(market_status)
    rows = []
    for ticker in names:
        fund = fund_map.get(ticker, {})
        news = news_map.get(ticker, {})
        market = market_map.get(ticker, {})
        fund_coverage = pd.to_numeric(pd.Series([fund.get("fundamental_coverage", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
        fund_ok = _truthy(fund.get("fundamental_score_eligible", False)) or float(fund_coverage) >= 45.0
        news_ok = _truthy(news.get("news_score_eligible", False))
        market_ok = _truthy(market.get("market_status_score_eligible", False))
        rows.append({
            "ticker": ticker,
            "stage_b_shortlisted": ticker in shortlist_set,
            "fundamental_coverage_pct": round(float(fund_coverage), 1),
            "fundamental_ready": fund_ok,
            "fundamental_route_state": fund.get("fundamental_route_state", "UNAVAILABLE_NOT_SCORED"),
            "news_ready": news_ok,
            "news_route_state": news.get("news_route_state", "UNAVAILABLE_NOT_SCORED"),
            "market_status_ready": market_ok,
            "market_status_route_state": market.get("market_status_route_state", "UNAVAILABLE_FAIL_CLOSED"),
            "expensive_enrichment_complete": bool(ticker in shortlist_set and fund_ok and news_ok and market_ok),
            "two_stage_pipeline_version": TWO_STAGE_PIPELINE_VERSION,
        })
    return pd.DataFrame(rows)


__all__ = [
    "TWO_STAGE_PIPELINE_VERSION",
    "ShortlistConfig",
    "build_lightweight_preliminary_focus",
    "build_enrichment_shortlist",
    "plan_round_robin_refresh",
    "build_two_stage_coverage_audit",
    "default_refresh_state_path",
]

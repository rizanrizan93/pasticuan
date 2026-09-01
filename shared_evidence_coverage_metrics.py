from __future__ import annotations

"""Scanner-neutral Phase 5.6 factual coverage and consumption reporting."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd


CONTRACT_VERSION = "1.0.0-phase5.6-task36"
SCANNERS = ("EMIR", "PASTICUAN")
CACHE_COUNTERS = (
    "provider_calls",
    "cache_hits",
    "requests_avoided",
    "refresh_lock_collisions",
    "refresh_errors",
)


@dataclass(frozen=True)
class MetricSpec:
    family: str
    definition: str
    date_field: str = ""


METRIC_SPECS: Mapping[str, MetricSpec] = OrderedDict({
    "market_ohlcv": MetricSpec("MARKET", "ticker has valid completed-session OHLCV", "market_latest_session"),
    "market_stock_summary": MetricSpec("MARKET", "ticker has valid shared IDX stock-summary facts", "market_latest_session"),
    "fundamental_report_discovery": MetricSpec("FUNDAMENTALS", "official financial report was discovered", "fundamental_latest_publication"),
    "fundamental_production_valid": MetricSpec("FUNDAMENTALS", "report identity, period, source, and production fact checks pass", "fundamental_latest_publication"),
    "fundamental_revenue": MetricSpec("FUNDAMENTALS", "production-valid revenue fact exists", "fundamental_latest_publication"),
    "fundamental_net_income": MetricSpec("FUNDAMENTALS", "production-valid net-income fact exists", "fundamental_latest_publication"),
    "fundamental_assets": MetricSpec("FUNDAMENTALS", "production-valid assets fact exists", "fundamental_latest_publication"),
    "fundamental_liabilities": MetricSpec("FUNDAMENTALS", "production-valid liabilities fact exists", "fundamental_latest_publication"),
    "fundamental_equity": MetricSpec("FUNDAMENTALS", "production-valid equity fact exists", "fundamental_latest_publication"),
    "fundamental_cash": MetricSpec("FUNDAMENTALS", "production-valid cash fact exists", "fundamental_latest_publication"),
    "fundamental_debt": MetricSpec("FUNDAMENTALS", "production-valid debt fact exists", "fundamental_latest_publication"),
    "fundamental_ocf": MetricSpec("FUNDAMENTALS", "production-valid operating-cash-flow fact exists", "fundamental_latest_publication"),
    "fundamental_capex": MetricSpec("FUNDAMENTALS", "production-valid compatible-period capex fact exists", "fundamental_latest_publication"),
    "fundamental_fcf": MetricSpec("FUNDAMENTALS", "FCF is derivable only from compatible-period OCF and capex", "fundamental_latest_publication"),
    "foreign_breadth": MetricSpec("FOREIGN", "ticker has at least one valid foreign-flow session", "foreign_latest_session"),
    "foreign_latest_session_covered": MetricSpec("FOREIGN", "ticker includes the latest completed IDX session", "foreign_latest_session"),
    "foreign_5_session_sufficient": MetricSpec("FOREIGN", "ticker has all five expected completed sessions", "foreign_latest_session"),
    "foreign_10_session_sufficient": MetricSpec("FOREIGN", "ticker has all ten expected completed sessions", "foreign_latest_session"),
    "foreign_20_session_sufficient": MetricSpec("FOREIGN", "ticker has the configured sufficient 20-session history", "foreign_latest_session"),
    "ownership_1pct_file": MetricSpec("OWNERSHIP", "ticker appears in a valid official 1% publication", "ownership_latest_publication"),
    "ownership_5pct_file": MetricSpec("OWNERSHIP", "ticker appears in a valid official 5% publication", "ownership_latest_publication"),
    "ownership_classification": MetricSpec("OWNERSHIP", "reported-holder classification is preserved", "ownership_latest_publication"),
    "ownership_type": MetricSpec("OWNERSHIP", "reported-holder type is preserved", "ownership_latest_publication"),
    "ownership_parsed_ticker": MetricSpec("OWNERSHIP", "ticker has at least one validated parsed ownership row", "ownership_latest_publication"),
    "ownership_historical_delta": MetricSpec("OWNERSHIP", "ticker has comparable historical ownership snapshots", "ownership_latest_publication"),
    "forward_material_event": MetricSpec("FORWARD", "ticker has document-supported material-event evidence", "forward_latest_evidence_date"),
    "forward_official_verified": MetricSpec("FORWARD", "ticker has official source-verified forward evidence", "forward_latest_evidence_date"),
    "capital_issued_shares": MetricSpec("CAPITAL_STRUCTURE", "issued-share state is explicitly available", "capital_latest_evidence_date"),
    "capital_rights_or_action": MetricSpec("CAPITAL_STRUCTURE", "rights or capital-action evidence state is available", "capital_latest_evidence_date"),
    "capital_recent_dilution_state": MetricSpec("CAPITAL_STRUCTURE", "recent dilution state is known from explicit compatible share deltas", "capital_latest_evidence_date"),
    "risk_uma_state": MetricSpec("RISK", "UMA evidence state is known, including an explicit no-event state", "risk_latest_evidence_date"),
    "risk_suspension_state": MetricSpec("RISK", "suspension evidence state is known, including an explicit no-event state", "risk_latest_evidence_date"),
    "risk_margin_state": MetricSpec("RISK", "margin eligibility state is known for the provider period", "risk_latest_evidence_date"),
    "risk_lendable_state": MetricSpec("RISK", "lendable observation state is known", "risk_latest_evidence_date"),
    "participant_ticker": MetricSpec("PARTICIPANT", "ticker appears in validated official Trade Detail", "participant_latest_session"),
    "participant_history_20_session_sufficient": MetricSpec("PARTICIPANT", "ticker has at least 20 validated Trade Detail sessions", "participant_latest_session"),
})

# This deliberately strict measurement is not a scanner gate.  Presence metrics mean
# the evidence state is known; they do not mean that an event (UMA, suspension, etc.) occurred.
FULLY_EVIDENCE_READY_COMPONENTS = tuple(METRIC_SPECS)


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _flag(value: Any) -> bool | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().upper()
    if text in {"1", "TRUE", "YES", "VALID", "AVAILABLE", "KNOWN", "PRESENT", "PASS", "VERIFIED"}:
        return True
    if text in {"0", "FALSE", "NO", "INVALID", "UNAVAILABLE", "ABSENT", "FAIL"}:
        return False
    return None


def _as_day(value: Any) -> pd.Timestamp | None:
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(stamp):
        return None
    return pd.Timestamp(stamp).tz_convert("Asia/Jakarta").tz_localize(None).normalize()


def _percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def build_shared_coverage_report(
    universe: Iterable[Any],
    facts: pd.DataFrame,
    *,
    scanner_consumption: pd.DataFrame | None = None,
    cache_counters: Mapping[str, Any] | None = None,
    as_of: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure shared facts first, then scanner consumption against those facts."""

    tickers = list(dict.fromkeys(_ticker(value) for value in universe if _ticker(value)))
    source = facts.copy() if isinstance(facts, pd.DataFrame) else pd.DataFrame()
    if not source.empty and "ticker" not in source:
        raise ValueError("COVERAGE_TICKER_COLUMN_MISSING")
    if "ticker" in source:
        source["ticker"] = source["ticker"].map(_ticker)
        source = source[source["ticker"] != ""]
        if source["ticker"].duplicated().any():
            raise ValueError("COVERAGE_TICKER_DUPLICATE")
        source = source.set_index("ticker", drop=False)
    as_of_day = _as_day(as_of or pd.Timestamp.now(tz="Asia/Jakarta"))
    if as_of_day is None:
        raise ValueError("COVERAGE_AS_OF_INVALID")

    detail = pd.DataFrame({"ticker": tickers})
    records = source.to_dict(orient="index") if not source.empty else {}
    date_fields = {spec.date_field for spec in METRIC_SPECS.values() if spec.date_field}
    date_cache: dict[tuple[str, str], pd.Timestamp | None] = {}
    parsed_dates: dict[str, pd.Timestamp | None] = {}
    for ticker in tickers:
        row = records.get(ticker, {})
        for field in date_fields:
            raw = row.get(field)
            key = repr(raw)
            if key not in parsed_dates:
                parsed_dates[key] = _as_day(raw)
            date_cache[(ticker, field)] = parsed_dates[key]
    factual: dict[str, dict[str, Any]] = {}
    families: dict[str, dict[str, Any]] = OrderedDict()
    for metric, spec in METRIC_SPECS.items():
        values: list[bool | None] = []
        dates: list[pd.Timestamp | None] = []
        for ticker in tickers:
            row = records.get(ticker, {})
            value = _flag(row.get(metric))
            observed = date_cache.get((ticker, spec.date_field)) if spec.date_field else None
            if value is True and spec.date_field and (observed is None or observed > as_of_day):
                value = False
            values.append(value)
            if observed is not None and observed <= as_of_day:
                dates.append(observed)
        detail[metric] = pd.array(values, dtype="boolean")
        known = int(detail[metric].notna().sum())
        count = int(detail[metric].fillna(False).sum())
        latest = max(dates).date().isoformat() if dates else ""
        entry = {
            "definition": spec.definition,
            "numerator": count,
            "denominator": len(tickers),
            "percentage": _percentage(count, len(tickers)),
            "known": known,
            "unknown": len(tickers) - known,
            "latest_source_date": latest,
        }
        factual[metric] = entry
        families.setdefault(spec.family, OrderedDict())[metric] = entry

    ready = detail[list(FULLY_EVIDENCE_READY_COMPONENTS)].fillna(False).all(axis=1)
    detail["fully_evidence_ready"] = ready.astype(bool)
    ready_count = int(ready.sum())
    detail_index = detail.set_index("ticker", drop=False)

    consumption = scanner_consumption.copy() if isinstance(scanner_consumption, pd.DataFrame) else pd.DataFrame()
    required_columns = {"scanner", "ticker", "metric", "consumed"}
    if not consumption.empty and not required_columns.issubset(consumption.columns):
        raise ValueError("CONSUMPTION_COLUMNS_MISSING")
    scanner_report: dict[str, dict[str, Any]] = OrderedDict()
    if not consumption.empty:
        consumption["scanner"] = consumption["scanner"].astype(str).str.strip().str.upper()
        consumption["ticker"] = consumption["ticker"].map(_ticker)
        consumption["metric"] = consumption["metric"].astype(str).str.strip()
        if not set(consumption["scanner"]).issubset(SCANNERS):
            raise ValueError("CONSUMPTION_SCANNER_INVALID")
        if not set(consumption["metric"]).issubset(METRIC_SPECS):
            raise ValueError("CONSUMPTION_METRIC_INVALID")
        if consumption.duplicated(["scanner", "ticker", "metric"]).any():
            raise ValueError("CONSUMPTION_IDENTITY_DUPLICATE")
        if not set(consumption["ticker"]).issubset(tickers):
            raise ValueError("CONSUMPTION_TICKER_OUTSIDE_UNIVERSE")
        consumption["consumed"] = consumption["consumed"].map(_flag)
        if consumption["consumed"].isna().any():
            raise ValueError("CONSUMPTION_STATE_INVALID")
        for row in consumption.itertuples(index=False):
            if bool(row.consumed):
                fact_state = detail_index.loc[row.ticker, row.metric]
                if pd.isna(fact_state) or not bool(fact_state):
                    raise ValueError("CONSUMPTION_WITHOUT_SHARED_FACT")

    for scanner in SCANNERS:
        scanner_metrics: dict[str, Any] = OrderedDict()
        subset = consumption[consumption["scanner"] == scanner] if not consumption.empty else pd.DataFrame()
        for metric in METRIC_SPECS:
            consumed = int(subset.loc[subset["metric"] == metric, "consumed"].fillna(False).astype(bool).sum()) if not subset.empty else 0
            available = factual[metric]["numerator"]
            scanner_metrics[metric] = {
                "consumed_count": consumed,
                "available_fact_count": available,
                "universe_count": len(tickers),
                "percentage_of_available": _percentage(consumed, available),
                "percentage_of_universe": _percentage(consumed, len(tickers)),
            }
        scanner_report[scanner] = scanner_metrics

    counters: dict[str, int] = {}
    supplied_counters = dict(cache_counters or {})
    for name in CACHE_COUNTERS:
        raw = supplied_counters.get(name, 0)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"CACHE_COUNTER_INVALID:{name}")
        counters[name] = raw

    def _latest(column: str) -> str:
        dates = [_as_day(value) for value in source.get(column, pd.Series(dtype=object))]
        visible = [value for value in dates if value is not None and value <= as_of_day]
        return max(visible).date().isoformat() if visible else ""

    forward_sources = source.get("forward_source", pd.Series(dtype=object)).dropna().astype(str).str.strip()
    forward_sources = forward_sources[forward_sources != ""]
    participant_ids: set[str] = set()
    for value in source.get("participant_ids", pd.Series(dtype=object)).dropna():
        participant_ids.update(item.strip().upper() for item in str(value).replace(",", "|").split("|") if item.strip())
    depths = pd.to_numeric(source.get("participant_history_depth", pd.Series(dtype=float)), errors="coerce").dropna()
    observations = {
        "market_latest_session": _latest("market_latest_session"),
        "foreign_latest_session": _latest("foreign_latest_session"),
        "ownership_latest_publication": _latest("ownership_latest_publication"),
        "forward_latest_evidence_date": _latest("forward_latest_evidence_date"),
        "forward_source_distribution": dict(sorted(forward_sources.value_counts().to_dict().items())),
        "participant_latest_trade_detail_session": _latest("participant_latest_session"),
        "participant_ticker_breadth": factual["participant_ticker"],
        "participant_breadth_distinct": len(participant_ids),
        "participant_historical_session_depth_min": int(depths.min()) if len(depths) else 0,
        "participant_historical_session_depth_max": int(depths.max()) if len(depths) else 0,
    }
    summary = {
        "contract_version": CONTRACT_VERSION,
        "as_of": as_of_day.date().isoformat(),
        "universe_count": len(tickers),
        "shared_factual_coverage": families,
        "scanner_consumption": scanner_report,
        "observations": observations,
        "shared_cache": counters,
        "fully_evidence_ready": {
            "definition": "all listed Phase 5.6 factual coverage components are explicitly available as of the cutoff; this is measurement only and is not a scanner gate",
            "required_metrics": list(FULLY_EVIDENCE_READY_COMPONENTS),
            "count": ready_count,
            "denominator": len(tickers),
            "percentage": _percentage(ready_count, len(tickers)),
        },
        "policy_effect": "MEASUREMENT_ONLY_NO_SCORING_OR_GATE_CHANGE",
    }
    return detail, summary


__all__ = [
    "CACHE_COUNTERS", "CONTRACT_VERSION", "FULLY_EVIDENCE_READY_COMPONENTS",
    "METRIC_SPECS", "SCANNERS", "MetricSpec", "build_shared_coverage_report",
]

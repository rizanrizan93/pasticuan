from __future__ import annotations

from pathlib import Path

from shared_fundamental_runtime import (
    canonicalize_metric_rows,
    normalize_operational_snapshot_rows,
    normalize_pluang_payloads,
    normalize_yahoo_payloads,
)


ROOT = Path(__file__).resolve().parents[1]


def test_operational_percent_ratios_are_promoted_as_percentage_points() -> None:
    rows = normalize_operational_snapshot_rows([{
        "ticker": "TLKM.JK",
        "revenue_growth": 0.0639,
        "earnings_growth": 0.2485,
        "roe": 0.1921,
        "roa": 0.0911,
        "net_margin": 0.1827,
        "debt_equity": 0.5998,
        "fundamental_coverage": 55.0,
        "fundamental_source_families": "IDX_OFFICIAL_XBRL • YAHOO",
        "fundamental_fetched_at": "2026-09-03T00:00:00+00:00",
        "content_hash": "abc",
    }])
    values = {row["metric_name"]: row["metric_value"] for row in rows}
    assert round(values["revenue_growth_pct"], 4) == 6.39
    assert round(values["earnings_growth_pct"], 4) == 24.85
    assert round(values["roe_pct"], 4) == 19.21
    assert values["debt_equity"] == 0.5998
    assert values["fundamental_coverage_pct"] == 55.0
    assert all(row["official_verified"] is False for row in rows)


def test_canonicalizer_corrects_legacy_aggregate_rows_but_not_v2_rows() -> None:
    base = {
        "provider": "OPERATIONAL_FUNDAMENTAL_BRIDGE", "ticker": "TLKM",
        "period_end": None, "metric_name": "roe_pct", "metric_unit": "PERCENT",
        "source_families": "YAHOO", "official_verified": False,
        "source_record_hash": "x", "observed_at": "2026-09-03T00:00:00+00:00",
        "validation_state": "VALID", "fetched_at": "2026-09-03T00:00:00+00:00",
    }
    legacy = {**base, "metric_value": 0.1921, "lineage_state": "BRIDGED_AGGREGATED_OPERATIONAL_FACTS"}
    corrected = {**base, "metric_value": 19.21, "source_record_hash": "y", "lineage_state": "BRIDGED_AGGREGATED_OPERATIONAL_METRIC_PERCENT_CANONICAL_V2", "observed_at": "2026-09-04T00:00:00+00:00"}
    assert round(canonicalize_metric_rows([legacy])["TLKM"]["proxy_metrics"]["roe_pct"], 4) == 19.21
    assert round(canonicalize_metric_rows([corrected])["TLKM"]["proxy_metrics"]["roe_pct"], 4) == 19.21


def test_exact_official_bundle_uses_latest_period_and_same_quarter_yoy() -> None:
    def row(period: str, metric: str, value: float) -> dict[str, object]:
        return {
            "provider":"OPERATIONAL_FINANCIAL_FACT_BRIDGE", "ticker":"ABMM", "period_end":period,
            "metric_name":metric, "metric_value":value, "metric_unit":"NORMALIZED",
            "source_families":"IDX_OFFICIAL_XBRL", "official_verified":True,
            "source_record_hash":f"{period}-{metric}", "lineage_state":"OPERATIONAL_FINANCIAL_FACT_EXACT_LINEAGE",
            "observed_at":"2026-08-14T00:00:00+00:00", "validation_state":"VALID", "fetched_at":"2026-09-03T00:00:00+00:00",
        }
    rows = [
        row("2025-06-30", "revenue", 100), row("2025-06-30", "net_income", 10),
        row("2026-03-31", "revenue", 90), row("2026-06-30", "revenue", 120),
        row("2026-06-30", "net_income", 15), row("2026-06-30", "equity", 100),
        row("2026-06-30", "total_debt", 40), row("2026-06-30", "cash", 20),
    ]
    item = canonicalize_metric_rows(rows)["ABMM"]
    assert item["official_period_end"] == "2026-06-30"
    assert round(item["official_metrics"]["revenue_growth_yoy_pct"], 6) == 20.0
    assert round(item["official_metrics"]["earnings_growth_yoy_pct"], 6) == 50.0
    assert item["official_metrics"]["interest_bearing_debt_to_equity"] == 0.4
    assert item["official_metrics"]["cash_to_debt_ratio"] == 0.5


def test_pluang_resolved_payload_normalizes_profitability_and_cashflow() -> None:
    fundamentals = {
        "code":"BBCA", "source":"pluang",
        "ratios":{"profitability":{"roe":"20.44%","roa":"3.63%","npm":"45.22%"},"solvency":{"cr":"1.2x","de":"4.6%"}},
        "overview":{"revenue":"Rp127.23T","net_income":"Rp57.54T"},
        "earnings":[{"quarter":"Q2 '26","actualEps":484}],
    }
    financials = {"code":"BBCA","source":"pluang","quarterly":{"cashFlow":{"chart":[{"timeframe":"Q2 ’26","operating":80,"netCF":30}]}}}
    rows = normalize_pluang_payloads("BBCA", fundamentals, financials, observed_at="2026-09-04T00:00:00+00:00")
    values = {row["metric_name"]: row["metric_value"] for row in rows}
    assert values["roe_pct"] == 20.44
    assert values["debt_equity"] == 0.046
    assert values["operating_cash_flow"] == 80
    assert all(row["official_verified"] is False for row in rows)


def test_yahoo_structured_payload_normalizes_statement_and_ratios() -> None:
    summary = {"symbol":"BBCA.JK","provider":"yahoo","revenueGrowthPercent":2.5,"returnOnEquityPercent":21.8,"profitMarginPercent":53.1,"totalCash":50,"totalDebt":20,"revenue":120}
    statements = {
        "income":{"items":[{"date":"2026-06-30","revenue":120,"netIncome":60},{"date":"2025-06-30","revenue":100,"netIncome":50}]},
        "balance":{"items":[{"date":"2026-06-30","totalAssets":500,"totalLiabilities":300,"stockholdersEquity":200,"currentAssets":100,"currentLiabilities":50,"totalDebt":20,"cash":50}]},
        "cashflow":{"items":[{"date":"2026-06-30","operatingCashFlow":70,"capitalExpenditure":-10}]},
    }
    rows = normalize_yahoo_payloads("BBCA.JK", summary, statements, observed_at="2026-09-04T00:00:00+00:00")
    values = {row["metric_name"]: row["metric_value"] for row in rows}
    assert values["revenue_growth_pct"] == 2.5
    assert round(values["earnings_growth_pct"], 6) == 20.0
    assert values["current_ratio"] == 2.0
    assert values["debt_equity"] == 0.1
    assert round(values["ocf_conversion_ratio"], 8) == round(70 / 60, 8)


def test_shared_runtime_contract_contains_no_scanner_decision_fields() -> None:
    text = (ROOT / "shared_fundamental_runtime.py").read_text(encoding="utf-8").lower()
    for forbidden in ("emir_score", "pasticuan_score", "entry_price", "stop_loss", "take_profit"):
        assert forbidden not in text

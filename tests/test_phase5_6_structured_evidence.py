from datetime import date, datetime, timezone
from pathlib import Path

import shared_structured_fundamental_evidence as fundamentals
from shared_structured_ownership_evidence import (
    normalize_idx_company_profile_row,
    normalize_pluang_profile,
    validate_shareholder_rows,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "database" / "migration_v29_structured_fundamental_ownership.sql"


def test_structured_migration_is_additive_backend_only_and_scanner_neutral() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()
    assert "create table if not exists public.evidence_fundamental_metrics" in sql
    assert "create table if not exists public.evidence_shareholder_profiles" in sql
    assert sql.count("enable row level security") >= 1
    assert "from public, anon, authenticated" in sql
    assert "grant select, insert, update" in sql
    assert "grant delete" not in sql
    for forbidden in ("emir_score", "pasticuan_score", "recommendation", "entry_price", "stop_loss", "take_profit"):
        assert forbidden not in sql


def test_operational_bridge_keeps_source_backed_metrics_and_rejects_source_less_rows() -> None:
    now = datetime(2026, 9, 3, tzinfo=timezone.utc).isoformat()
    rows = fundamentals.normalize_operational_snapshots([
        {
            "ticker": "BBCA.JK",
            "period_end": "2026-06-30",
            "statement_date": "2026-06-30",
            "roe": 20.4,
            "operating_cash_flow": 123.0,
            "fundamental_source_families": "IDX_OFFICIAL_XBRL • YAHOO",
            "fundamental_official_verified": True,
            "fundamental_fetched_at": now,
            "content_hash": "abc",
        },
        {
            "ticker": "EMPTY.JK",
            "roe": 10.0,
            "fundamental_source_families": "",
            "fundamental_fetched_at": now,
        },
    ])
    assert {row["ticker"] for row in rows} == {"BBCA"}
    assert {row["metric_name"] for row in rows} == {"roe_pct", "operating_cash_flow"}
    assert all(row["official_verified"] is False for row in rows)
    assert all(row["lineage_state"] == "BRIDGED_AGGREGATED_OPERATIONAL_METRIC_NOT_FIELD_OFFICIAL" for row in rows)
    assert fundamentals.validate_structured_metrics(rows) == (True, "VALID")


    assert fundamentals.OPERATIONAL_METRICS["fundamental_coverage"] == ("fundamental_coverage_pct", "PERCENT")
    source = (ROOT / "shared_structured_fundamental_evidence.py").read_text(encoding="utf-8")
    select_block = source.split("select=(", 1)[-1] if "select=(" in source else source
    assert "fundamental_official_source_coverage_pct,fundamental_cashflow_statement_coverage_pct" not in select_block


def test_exact_operational_financial_facts_keep_field_level_official_lineage() -> None:
    periods = [{
        "financial_period_id": "p1",
        "ticker": "BBCA.JK",
        "period_end": "2026-06-30",
        "filing_date": None,
        "source_family": "IDX_OFFICIAL_XBRL",
        "document_id": "IDX-XBRL-BBCA-Q2",
        "document_hash": None,
        "is_current": True,
        "created_at": "2026-07-30T00:00:00+00:00",
        "updated_at": "2026-07-30T00:00:00+00:00",
    }]
    facts = [{
        "financial_fact_id": "f1",
        "financial_period_id": "p1",
        "ticker": "BBCA.JK",
        "metric_code": "REVENUE",
        "reported_value": 100.0,
        "normalized_value": 100.0,
        "currency": "IDR",
        "source_lineage": {"source_family": "IDX_OFFICIAL_XBRL", "source_verified": True},
        "created_at": "2026-07-30T00:00:00+00:00",
    }]
    rows = fundamentals.normalize_operational_financial_facts(periods, facts)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "BBCA"
    assert rows[0]["metric_name"] == "revenue"
    assert rows[0]["official_verified"] is True
    assert rows[0]["source_families"] == "IDX_OFFICIAL_XBRL"
    assert rows[0]["lineage_state"] == "OPERATIONAL_FINANCIAL_FACT_EXACT_LINEAGE"
    assert fundamentals.validate_structured_metrics(rows) == (True, "VALID")


def test_pluang_financials_normalize_cashflow_balance_and_income_without_scanner_scores() -> None:
    f = {
        "code": "BBCA",
        "source": "pluang",
        "ratios": {
            "profitability": {"roe": "20.44%", "roa": "3.63%", "npm": "45.22%"},
            "solvency": {"cr": "1.2x", "de": "4.6%"},
        },
        "overview": {"revenue": "Rp127.23T", "net_income": "Rp57.54T"},
        "earnings": [{"quarter": "Q2 '26", "actualEps": 484}],
    }
    s = {
        "code": "BBCA",
        "source": "pluang",
        "quarterly": {
            "incomeStatement": {"chart": [{"timeframe": "Q2\n’26", "revenue": 100, "netProfitLoss": 40, "profitMargin": 0.4}]},
            "balanceSheet": {"chart": [{"timeframe": "Q2\n’26", "assets": 500, "liabilities": 300, "debtToAsset": 0.6}]},
            "cashFlow": {"chart": [{"timeframe": "Q2\n’26", "operating": 80, "investing": -30, "finance": -20, "netCF": 30}]},
        },
    }
    rows = fundamentals._pluang_metric_rows(
        "BBCA", f, s, observed_at=datetime(2026, 9, 3, tzinfo=timezone.utc)
    )
    names = {row["metric_name"] for row in rows}
    assert {"roe_pct", "revenue", "net_income", "assets", "liabilities", "operating_cash_flow"} <= names
    assert all("score" not in row["metric_name"] for row in rows)
    assert all(row["official_verified"] is False for row in rows)


def test_idx_and_pluang_shareholders_remain_separate_from_ksei_semantics() -> None:
    idx_rows = normalize_idx_company_profile_row({
        "ticker": "BBCA",
        "source_period": "2026-09-01",
        "observed_on": "2026-09-03",
        "payload_hash": "profilehash",
        "source_url": "https://api.zpi.web.id/v1/finance:idx/company-profile",
        "source_verified": True,
        "fetched_at": "2026-09-03T00:00:00+00:00",
        "profile": {
            "relationships": {
                "shareholders": [
                    {"name": "PT Dwimuria", "shares": 10, "sharePct": 54.94, "category": "CONTROL"}
                ]
            }
        },
    })
    pluang_rows = normalize_pluang_profile({
        "code": "BBCA",
        "source": "pluang",
        "shareholders": [{"name": "PT Dwimuria", "share": 54.942}],
    }, observed_on=date(2026, 9, 3))
    assert idx_rows[0]["provider"] == "IDX_COMPANY_PROFILE_VIA_ZAPI"
    assert pluang_rows[0]["provider"] == "PLUANG_COMPANY_PROFILE_VIA_ZAPI"
    assert validate_shareholder_rows(idx_rows) == (True, "VALID")
    assert validate_shareholder_rows(pluang_rows) == (True, "VALID")
    assert all("ksei" not in row["provider"].lower() for row in idx_rows + pluang_rows)

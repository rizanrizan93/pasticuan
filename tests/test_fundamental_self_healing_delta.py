from __future__ import annotations

import pandas as pd

import scanner


def _official_history() -> pd.DataFrame:
    return scanner.normalize_fundamental_history(pd.DataFrame([{
        "ticker": "TEST.JK", "period_end": "2026-06-30", "period_type": "Q2",
        "statement_basis": "YTD_CUMULATIVE", "source_family": "IDX_OFFICIAL_XBRL",
        "source_name": "IDX official", "source_url": "https://www.idx.co.id/TEST/instance.zip",
        "currency": "IDR", "revenue": 100.0, "net_income": 10.0,
        "total_assets": 200.0, "total_liabilities": 80.0, "equity": 120.0,
        "operating_cash_flow": 12.0, "capex": 2.0,
        "source_verified": True, "issuer_match": True,
        "provider": "IDX_OFFICIAL_XBRL", "evidence_type": "FUNDAMENTAL_STATEMENT",
        "available_at": "2026-08-01T10:00:00+07:00",
    }]))


def test_valid_official_period_is_reused_without_attachment_download(monkeypatch) -> None:
    existing = _official_history()
    manifest = pd.DataFrame([{
        "ticker": "TEST.JK", "year": 2026, "period_code": "tw2", "period_type": "Q2",
        "period_end": pd.Timestamp("2026-06-30"), "filename": "instance.zip",
        "attachment_url": "https://www.idx.co.id/TEST/instance.zip", "attachment_rank": 0,
    }])
    monkeypatch.setattr(scanner, "_load_cache", lambda _name: pd.DataFrame())
    monkeypatch.setattr(scanner, "_idx_manifest_rows", lambda *args, **kwargs: (manifest.copy(), pd.DataFrame()))

    def no_download(*args, **kwargs):
        raise AssertionError("valid cached period must not be downloaded")

    history, report = scanner.fetch_idx_fundamental_history(
        ["TEST.JK"], request_get=no_download, existing_history=existing,
        missing_periods_only=True,
    )

    assert not history.empty
    row = report.loc[report["ticker"].eq("TEST.JK")].iloc[0]
    assert row["status"] == "VALID_CACHE_REUSED"
    assert int(row["documents_requested"]) == 0
    assert int(row["cached_valid_periods"]) == 1


def test_provider_outage_preserves_durable_valid_history(monkeypatch) -> None:
    existing = _official_history()
    outage = pd.DataFrame([{
        "ticker": "ALL_REQUESTED", "provider": "IDX_OFFICIAL_XBRL_MANIFEST",
        "status": "FAILED", "error": "HTTP 403",
    }])
    monkeypatch.setattr(scanner, "_load_cache", lambda _name: pd.DataFrame())
    monkeypatch.setattr(scanner, "_idx_manifest_rows", lambda *args, **kwargs: (pd.DataFrame(), outage.copy()))

    history, report = scanner.fetch_idx_fundamental_history(
        ["TEST.JK"], existing_history=existing, missing_periods_only=True,
    )

    assert len(history) == len(existing)
    row = report.loc[report["ticker"].eq("TEST.JK")].iloc[0]
    assert row["status"] == "CACHE_FALLBACK"
    assert int(row["rows"]) == len(existing)

import numpy as np
import pandas as pd
import pytest

from idx_trading_calendar import (
    CalendarCoverageError, CalendarState, calendar_state, is_idx_session, latest_expected_completed_session,
    n_idx_sessions_ago, previous_idx_session, trading_session_age,
)
from provider_semantics import (
    EvidenceProvenance, ProviderResult, ProviderStatus, aggregate_provenance,
    normalize_provider_result, normalize_provenance,
)
from scanner import parse_broker_summary_csv


def test_typed_provider_result_matrix_is_fail_closed_and_zero_safe():
    cases = [
        ({"status": "SUCCESS", "value": 7}, ProviderStatus.SUCCESS),
        (0, ProviderStatus.SUCCESS),
        ({"status": "PARTIAL", "value": [1]}, ProviderStatus.PARTIAL),
        ({"status": "MISSING"}, ProviderStatus.MISSING),
        ({"status": "STALE", "value": 3}, ProviderStatus.STALE),
        ({"status": "INVALID", "value": "bad"}, ProviderStatus.INVALID),
        ({"status": "ERROR", "value": 0, "error": "provider timeout"}, ProviderStatus.PROVIDER_ERROR),
        ({"status": "mystery", "value": 1}, ProviderStatus.INVALID),
        ({"status": "NOT_APPLICABLE"}, ProviderStatus.NOT_APPLICABLE),
        ("unrecognized legacy payload", ProviderStatus.INVALID),
        (None, ProviderStatus.MISSING),
        (float("nan"), ProviderStatus.MISSING),
        (pd.NA, ProviderStatus.MISSING),
        (True, ProviderStatus.INVALID),
        ({"status": "SUCCESS", "value": 1, "freshness": "STALE"}, ProviderStatus.STALE),
    ]
    results = [normalize_provider_result(value, provider="FIXTURE") for value, _ in cases]
    assert [result.status for result in results] == [expected for _, expected in cases]
    assert results[1].value == 0
    assert results[6].value == 0
    assert results[6].status is not ProviderStatus.SUCCESS
    assert results[6].error_category.value == "TIMEOUT"
    assert all(results[index].value is None for index in (3, 10, 11, 12))
    assert all(result.provider == "FIXTURE" for result in results)

    for malformed in (
        {"status": "SUCCESS"},
        {"status": "SUCCESS", "value": None},
        {"status": "SUCCESS", "value": np.nan},
        {"status": "SUCCESS", "value": pd.NA},
    ):
        assert normalize_provider_result(malformed, provider="FIXTURE").status is ProviderStatus.INVALID
    zero = normalize_provider_result({"status": "SUCCESS", "value": 0}, provider="FIXTURE")
    assert zero.status is ProviderStatus.SUCCESS and zero.value == 0
    boolean = normalize_provider_result({"status": "SUCCESS", "value": False}, provider="FIXTURE")
    assert boolean.status is ProviderStatus.SUCCESS and boolean.value is False

    canonical = normalize_provider_result(
        {"status": "SUCCESS", "value": 3}, provider="FIXTURE", provenance="VERIFIED"
    )
    assert normalize_provider_result(canonical) is canonical
    assert normalize_provider_result(canonical, provider="FIXTURE") is canonical
    conflict = normalize_provider_result(canonical, provider="OTHER")
    assert conflict.status is ProviderStatus.INVALID
    assert conflict.provider == "FIXTURE"
    provider_error = normalize_provider_result(
        {"status": "ERROR", "value": 0, "error": "timeout"}, provider="FIXTURE"
    )
    assert normalize_provider_result(provider_error) is provider_error
    for provider_status in ProviderStatus:
        canonical_status = ProviderResult(status=provider_status, provider="FIXTURE", value=0)
        assert normalize_provider_result(canonical_status) is canonical_status


def test_idx_calendar_weekend_holiday_timezone_and_unknown_coverage():
    assert is_idx_session("2026-08-07")
    assert not is_idx_session("2026-08-08")
    assert calendar_state("2026-08-17") is CalendarState.CLOSED
    assert calendar_state("2025-08-18") is CalendarState.CLOSED
    assert previous_idx_session("2026-08-10").date().isoformat() == "2026-08-10"
    assert n_idx_sessions_ago("2026-08-10", 1).date().isoformat() == "2026-08-07"
    assert trading_session_age("2026-08-07", "2026-08-10") == 1
    assert trading_session_age("2026-03-17", "2026-03-25") == 1
    assert trading_session_age("2026-08-08", "2026-08-10") is None
    assert calendar_state("2027-01-04") is CalendarState.UNKNOWN
    assert calendar_state("2027-01-02") is CalendarState.UNKNOWN
    assert calendar_state("2099-01-01") is CalendarState.UNKNOWN
    assert trading_session_age("2027-01-04", "2027-01-05") is None
    assert trading_session_age("2025-12-30", "2026-01-02") == 1
    assert trading_session_age("2026-12-30", "2027-01-01") is None
    with pytest.raises(CalendarCoverageError):
        latest_expected_completed_session("2027-01-01")
    with pytest.raises(CalendarCoverageError):
        previous_idx_session("2024-12-31")
    with pytest.raises(CalendarCoverageError):
        n_idx_sessions_ago("2027-01-01", 1)
    assert latest_expected_completed_session("2026-08-10T08:30:00Z").date().isoformat() == "2026-08-07"
    assert latest_expected_completed_session("2026-08-10T09:30:00Z").date().isoformat() == "2026-08-10"


def test_provenance_vocabulary_never_upgrades_and_provider_is_independent():
    mapping = {
        "IDX_OFFICIAL_XBRL": EvidenceProvenance.DIRECT_OR_OFFICIAL,
        "VERIFIED_VENDOR_API": EvidenceProvenance.VERIFIED,
        "GOOGLE_NEWS_PUBLIC_RESEARCH": EvidenceProvenance.PUBLIC_RESEARCH,
        "MODEL_INFERRED": EvidenceProvenance.INFERRED,
        "OHLCV_PROXY": EvidenceProvenance.PROXY,
        "YFINANCE_PROXY_NOT_OFFICIAL_FILING": EvidenceProvenance.PROXY,
        None: EvidenceProvenance.MISSING,
        "unknown legacy authority": EvidenceProvenance.MISSING,
    }
    assert {value: normalize_provenance(value) for value in mapping} == mapping
    assert aggregate_provenance("IDX_OFFICIAL_XBRL", "OHLCV_PROXY") is EvidenceProvenance.PROXY
    assert aggregate_provenance("VERIFIED", None) is EvidenceProvenance.MISSING
    result = normalize_provider_result(1, provider="ZAPI", provenance="PUBLIC_RESEARCH")
    assert result.provider == "ZAPI"
    assert result.provenance is EvidenceProvenance.PUBLIC_RESEARCH


def test_pasticuan_broker_freshness_uses_expected_completed_session():
    frame = pd.DataFrame([
        {"ticker": "AAA", "date": day, "broker_code": "AA", "buy_value": 100, "sell_value": 90}
        for day in ("2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07")
    ])
    before_close = parse_broker_summary_csv(
        frame, source_type="PROVIDER_API", source_verified=True,
        as_of="2026-08-10T08:30:00Z", max_age_days=0,
    ).iloc[0]
    after_close = parse_broker_summary_csv(
        frame, source_type="PROVIDER_API", source_verified=True,
        as_of="2026-08-10T09:30:00Z", max_age_days=0,
    ).iloc[0]
    assert before_close["broksum_age_sessions"] == 0
    assert bool(before_close["broksum_current"])
    assert after_close["broksum_age_sessions"] == 1
    assert not bool(after_close["broksum_current"])

    future_only = pd.DataFrame([
        {"ticker": "AAA", "date": day, "broker_code": "AA", "buy_value": 100, "sell_value": 90}
        for day in ("2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14")
    ])
    assert parse_broker_summary_csv(
        future_only, source_type="PROVIDER_API", source_verified=True,
        as_of="2026-08-07T09:30:00Z", max_age_days=1,
    ).empty

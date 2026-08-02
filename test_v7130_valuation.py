from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import free_data_providers as providers
import scanner


class _TextResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _TextSession:
    def __init__(self, html: str) -> None:
        self.html = html
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _TextResponse:
        self.calls.append({"url": url, **kwargs})
        return _TextResponse(self.html)


class _JsonResponse:
    status_code = 200

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class _JsonSession:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def get(self, _url: str, **_kwargs: object) -> _JsonResponse:
        return _JsonResponse(self.payload)


def _price_frame(price: float = 1_000.0) -> pd.DataFrame:
    index = pd.bdate_range("2026-07-20", "2026-07-31")
    frame = pd.DataFrame({
        "Open": np.full(len(index), price),
        "High": np.full(len(index), price * 1.01),
        "Low": np.full(len(index), price * 0.99),
        "Close": np.full(len(index), price),
        "Volume": np.full(len(index), 1_000_000.0),
    }, index=index)
    return frame


def _fundamental_row(currency: str = "IDR") -> pd.DataFrame:
    scale = 1.0 if currency == "IDR" else 1.0 / 18_058.0
    return pd.DataFrame([{
        "ticker": "TEST.JK",
        "fundamental_primary_currency": currency,
        "fundamental_model": "NON_FINANCIAL",
        "latest_statement_date": "2026-03-31",
        "history_shares_outstanding_latest": 1_000_000_000.0,
        "history_net_income_ttm": 100_000_000_000.0 * scale,
        "history_fcf_ttm": 80_000_000_000.0 * scale,
        "history_equity_latest": 500_000_000_000.0 * scale,
        "history_total_debt_latest": 100_000_000_000.0 * scale,
        "history_cash_latest": 50_000_000_000.0 * scale,
        "history_ebitda_ttm": 150_000_000_000.0 * scale,
        "history_earnings_growth": 0.20,
        "history_positive_earnings_ratio": 1.0,
    }])


def _official_usd_fx() -> pd.DataFrame:
    return pd.DataFrame([{
        "currency": "USD",
        "idr_per_unit": 18_058.0,
        "as_of": "2026-07-31",
        "source_family": "BANK_INDONESIA_JISDOR",
        "source_verified": True,
        "source_url": "https://www.bi.go.id/",
    }])


@pytest.mark.parametrize(
    ("date_text", "rate_text"),
    [
        ("31 July 2026", "Rp18,058.00"),
        ("31 Juli 2026", "Rp18.058,00"),
    ],
)
def test_jisdor_parser_supports_english_and_indonesian_locales(
    date_text: str, rate_text: str,
) -> None:
    html = f"<table><tr><th>Date</th><th>Rate</th></tr><tr><td>{date_text}</td><td>{rate_text}</td></tr></table>"
    session = _TextSession(html)

    frame, report = providers.bank_indonesia_jisdor_reference(
        as_of="2026-08-02", session=session, retry_backoff=0,
    )

    assert len(session.calls) == 1
    assert report["status"] == "OK"
    assert report["usd_idr"] == pytest.approx(18_058.0)
    assert bool(frame.loc[0, "source_verified"])
    assert pd.Timestamp(frame.loc[0, "as_of"]).date().isoformat() == "2026-07-31"


def test_yahoo_chart_retains_split_ratio_not_only_the_date() -> None:
    timestamp = int(pd.Timestamp("2026-07-17", tz="UTC").timestamp())
    payload = {"chart": {"error": None, "result": [{
        "timestamp": [timestamp],
        "indicators": {
            "quote": [{
                "open": [100.0], "high": [101.0], "low": [99.0],
                "close": [100.0], "volume": [1_000.0],
            }],
            "adjclose": [{"adjclose": [100.0]}],
        },
        "events": {"splits": {str(timestamp): {
            "date": timestamp, "numerator": 5.0, "denominator": 1.0,
            "splitRatio": "5:1",
        }}},
        "meta": {"currency": "IDR"},
    }]}}

    frame, _ = providers.yahoo_chart_direct(
        "TEST.JK", session=_JsonSession(payload), retry_backoff=0,
    )

    assert frame.attrs["corporate_action_split_dates"] == ["2026-07-17"]
    assert frame.attrs["corporate_action_splits"][0]["ratio"] == pytest.approx(5.0)


def test_split_adjustment_applies_known_post_statement_ratio() -> None:
    frame = _price_frame()
    frame.attrs.update({
        "corporate_action_split_dates": ["2026-07-16"],
        "corporate_action_splits": [{
            "date": "2026-07-16", "numerator": 5,
            "denominator": 1, "ratio": 5.0,
        }],
    })

    factor, state, events = scanner._valuation_split_adjustment(
        frame, "2026-03-31", "2026-07-31",
    )

    assert factor == pytest.approx(5.0)
    assert state == "POST_STATEMENT_SPLIT_ADJUSTED"
    assert "2026-07-16x5" in events


def test_split_adjustment_fails_closed_when_ratio_is_missing() -> None:
    frame = _price_frame()
    frame.attrs["corporate_action_split_dates"] = ["2026-07-16"]

    factor, state, _ = scanner._valuation_split_adjustment(
        frame, "2026-03-31", "2026-07-31",
    )

    assert np.isnan(factor)
    assert state == "POST_STATEMENT_SPLIT_RATIO_MISSING"


def test_idr_valuation_derives_auditable_metrics() -> None:
    result = scanner.enrich_fundamentals_with_valuation(
        _fundamental_row("IDR"), {"TEST.JK": _price_frame()},
        now="2026-08-02 12:00:00+07:00",
    ).iloc[0]

    assert result["market_cap"] == pytest.approx(1_000_000_000_000.0)
    assert result["trailing_pe"] == pytest.approx(10.0)
    assert result["earnings_yield"] == pytest.approx(0.10)
    assert result["price_to_book"] == pytest.approx(2.0)
    assert result["fcf_yield"] == pytest.approx(0.08)
    assert result["ev_ebitda"] == pytest.approx(7.0)
    assert result["peg_ratio"] == pytest.approx(0.5)
    assert bool(result["valuation_score_eligible"])
    assert bool(result["valuation_production_eligible"])
    assert result["valuation_state"] == "FULL_AUDITABLE_DERIVED_VALUATION"
    assert result["valuation_source_tier"] == "CROSSCHECKED_DERIVED_RESEARCH_DATA"


def test_usd_valuation_uses_verified_current_fx() -> None:
    result = scanner.enrich_fundamentals_with_valuation(
        _fundamental_row("USD"), {"TEST.JK": _price_frame()},
        reference_fx=_official_usd_fx(),
        now="2026-08-02 12:00:00+07:00",
    ).iloc[0]

    assert result["trailing_pe"] == pytest.approx(10.0)
    assert result["fcf_yield"] == pytest.approx(0.08)
    assert result["valuation_fx_rate_to_idr"] == pytest.approx(18_058.0)
    assert result["valuation_fx_source"] == "BANK_INDONESIA_JISDOR"
    assert bool(result["valuation_fx_verified"])
    assert bool(result["valuation_score_eligible"])


def test_non_idr_valuation_without_current_fx_is_not_scored() -> None:
    result = scanner.enrich_fundamentals_with_valuation(
        _fundamental_row("USD"), {"TEST.JK": _price_frame()},
        now="2026-08-02 12:00:00+07:00",
    ).iloc[0]

    assert result["market_cap"] == pytest.approx(1_000_000_000_000.0)
    assert np.isnan(result["trailing_pe"])
    assert not bool(result["valuation_score_eligible"])
    assert "CURRENT_FX_RATE_MISSING" in result["valuation_flags"]


def test_peg_caps_base_effect_growth_at_one_hundred_percent() -> None:
    fundamentals = _fundamental_row("IDR")
    fundamentals.loc[0, "history_earnings_growth"] = 2.0

    result = scanner.enrich_fundamentals_with_valuation(
        fundamentals, {"TEST.JK": _price_frame()},
        now="2026-08-02 12:00:00+07:00",
    ).iloc[0]

    assert result["valuation_peg_growth_pct_used"] == pytest.approx(100.0)
    assert result["peg_ratio"] == pytest.approx(0.1)
    assert "PEG_GROWTH_CAPPED_AT_100PCT" in result["valuation_flags"]


def test_current_market_cap_reference_reconciles_unexplained_share_change() -> None:
    fundamentals = _fundamental_row("IDR")
    fundamentals.loc[0, "market_cap"] = 2_000_000_000_000.0
    fundamentals.loc[0, "market_cap_asof"] = "2026-07-31"
    fundamentals.loc[0, "market_cap_source"] = "DATED_REFERENCE"
    fundamentals.loc[0, "market_cap_source_verified"] = False

    result = scanner.enrich_fundamentals_with_valuation(
        fundamentals, {"TEST.JK": _price_frame(100.0)},
        now="2026-08-02 12:00:00+07:00",
    ).iloc[0]

    assert result["market_cap_derived_idr"] == pytest.approx(100_000_000_000.0)
    assert result["market_cap"] == pytest.approx(2_000_000_000_000.0)
    assert result["valuation_market_cap_mode"] == "CURRENT_REFERENCE_RECONCILES_SHARE_CHANGE"
    assert result["valuation_market_cap_divergence_pct"] == pytest.approx(95.0)
    assert "SHARE_COUNT_RECONCILED_TO_CURRENT_MARKET_CAP" in result["valuation_flags"]
    assert not bool(result["valuation_production_eligible"])


def test_market_cap_observation_date_precedes_later_ingestion_timestamp() -> None:
    fundamentals = _fundamental_row("IDR")
    fundamentals.loc[0, "market_cap"] = 1_010_000_000_000.0
    fundamentals.loc[0, "market_cap_asof"] = "2026-07-31"
    fundamentals.loc[0, "fundamental_fetched_at"] = (
        "2026-08-02T06:43:43.329020+07:00"
    )
    fundamentals.loc[0, "market_cap_source_verified"] = False

    result = scanner.enrich_fundamentals_with_valuation(
        fundamentals, {"TEST.JK": _price_frame()},
        now="2026-08-02 12:00:00+07:00",
    ).iloc[0]

    assert result["valuation_market_cap_mode"] == "CURRENT_REFERENCE_CROSSCHECK_PASS"
    assert pd.Timestamp(
        result["valuation_market_cap_reference_asof"]
    ).date().isoformat() == "2026-07-31"
    assert bool(result["valuation_production_eligible"])


def test_company_profile_selects_financial_valuation_model() -> None:
    fundamentals = _fundamental_row("IDR")
    fundamentals.loc[0, "fundamental_model"] = "GENERAL"
    fundamentals.loc[0, "company_name"] = "PT Bank Contoh Indonesia Tbk"

    result = scanner.enrich_fundamentals_with_valuation(
        fundamentals, {"TEST.JK": _price_frame()},
        now="2026-08-02 12:00:00+07:00",
    ).iloc[0]

    assert result["fundamental_model"] == "FINANCIAL"
    assert result["valuation_sector_model_source"] == "SECTOR_OR_COMPANY_PROFILE"
    assert result["valuation_data_coverage_pct"] == pytest.approx(100.0)


def test_daily_cache_preserves_split_ratio_lineage(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IDX_SCANNER_CACHE_DIR", str(tmp_path))
    frame = _price_frame()
    frame.attrs.update({
        "corporate_action_split_dates": ["2026-07-16"],
        "corporate_action_dividend_dates": ["2026-06-20"],
        "corporate_action_splits": [{
            "date": "2026-07-16", "numerator": 5.0,
            "denominator": 1.0, "ratio": 5.0,
            "source": "UNIT_TEST",
        }],
    })

    scanner._write_daily_ohlcv_cache(
        "TEST.JK", frame, "UNIT_TEST",
        now="2026-08-02 12:00:00+07:00",
    )
    restored = scanner._load_daily_ohlcv_cache("TEST.JK")

    assert restored.attrs["schema_version"] == 6
    assert restored.attrs["corporate_action_split_dates"] == ["2026-07-16"]
    assert restored.attrs["corporate_action_splits"][0]["ratio"] == pytest.approx(5.0)
    assert restored.attrs["corporate_action_splits"][0]["source"] == "UNIT_TEST"

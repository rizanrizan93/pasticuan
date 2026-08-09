from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import pathlib
import py_compile

import numpy as np
import pandas as pd

import resumable_app_engine as eng
from scanner import attach_fundamentals, parse_universe_csv
from scanner_database import _json_safe, _normalise_record


ROOT = pathlib.Path(__file__).resolve().parent


def _history(end: pd.Timestamp, bars: int = 320) -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=bars)
    close = np.linspace(100.0, 140.0, bars)
    return pd.DataFrame({
        "Open": close - 1.0,
        "High": close + 2.0,
        "Low": close - 2.0,
        "Close": close,
        "Volume": np.full(bars, 1_000_000.0),
    }, index=index)


def check_compile() -> None:
    for path in ROOT.glob("*.py"):
        py_compile.compile(str(path), doraise=True)


def check_metadata_only_fundamental_is_fail_soft() -> None:
    signals = pd.DataFrame([{"ticker": "TEST.JK", "quality_score": 72.0}])
    placeholder = pd.DataFrame([{"ticker": "TEST.JK", "idx_sector": "Energy"}])
    result = attach_fundamentals(signals, placeholder)
    assert len(result) == 1
    assert result.iloc[0]["fundamental_coverage"] == 0.0
    assert pd.isna(result.iloc[0]["fundamental_score"])
    assert result.iloc[0]["composite_score"] == 72.0


def check_uploaded_idx_sector_survives() -> None:
    source = pd.DataFrame([
        {
            "rank_universe": 1,
            "ticker": "AADI",
            "yahoo_ticker": "AADI.JK",
            "sector_idx_ic": "Energy",
            "universe_role": "Sector Leader / Core",
            "priority": "A",
            "active_scan": 1,
        },
        {
            "rank_universe": 2,
            "ticker": "MARK.JK",
            "yahoo_ticker": "MARK.JK",
            "sector_idx_ic": "Basic Materials",
            "universe_role": "Multibagger",
            "priority": "A",
            "active_scan": 1,
        },
    ])
    parsed = parse_universe_csv(source, max_tickers=400, strict_limit=True)
    assert parsed["ticker"].tolist() == ["AADI.JK", "MARK.JK"]
    assert parsed["idx_sector"].tolist() == ["Energy", "Basic Materials"]
    metadata = eng._universe_metadata(parsed.to_dict("records"), parsed["ticker"].tolist())
    by_ticker = metadata.set_index("ticker")
    assert by_ticker.at["AADI.JK", "sector"] == "ENERGY"
    assert by_ticker.at["MARK.JK", "sector"] == "BASIC MATERIALS"
    assert set(metadata["universe_metadata_source"]) == {"UPLOADED_UNIVERSE"}


def check_missing_metadata_is_not_stringified() -> None:
    source = pd.DataFrame([{
        "ticker": "MARK",
        "sector_idx_ic": pd.NA,
        "rank_universe": pd.NA,
    }])
    parsed = parse_universe_csv(source, max_tickers=400, strict_limit=True)
    assert parsed.iloc[0]["idx_sector"] == ""
    assert pd.isna(parsed.iloc[0]["rank_universe"])


def check_ohlcv_audit_statuses_do_not_overwrite_acquisition() -> None:
    tickers = ["AAA.JK", "BBB.JK"]
    expected = eng._expected_completed_session()
    histories = {ticker: _history(expected) for ticker in tickers}

    class Bridge:
        def read_ohlcv_cache(self, names, min_bars=60):
            return {}, pd.DataFrame([
                {"ticker": ticker, "provider": "SUPABASE_OHLCV", "status": "DATABASE_MISS", "payload_format": "NONE"}
                for ticker in names
            ])

        def write_ohlcv_cache(self, values, **kwargs):
            return pd.DataFrame([
                {"ticker": ticker, "provider": "SUPABASE_OHLCV", "status": "WRITE_FAILED", "rows_written": 0, "error": "db offline"}
                for ticker in values
            ])

    report = SimpleNamespace(
        source_tiers={ticker: "LIVE_YAHOO" for ticker in tickers},
        failed={},
        warnings={},
    )
    with patch.object(eng, "download_ohlcv", return_value=(histories, report)):
        prepared, _, audit = eng._database_first_ohlcv(
            Bridge(), tickers, period="5y", itick_api_token="", min_bars=260,
        )
    assert set(prepared) == set(tickers)
    ticker_audit = audit.dropna(subset=["ticker"]).set_index("ticker")
    assert len(ticker_audit) == 2
    assert set(ticker_audit["status"]) == {"CURRENT"}
    assert set(ticker_audit["acquisition_status"]) == {"CURRENT"}
    assert set(ticker_audit["source_tier"]) == {"LIVE_YAHOO"}
    assert set(ticker_audit["database_cache_status"]) == {"DATABASE_MISS"}
    assert set(ticker_audit["database_write_status"]) == {"WRITE_FAILED"}
    assert set(ticker_audit["session_lag"].astype(int)) == {0}


def check_nat_never_crosses_database_contract() -> None:
    assert _json_safe(pd.NaT) is None
    assert _json_safe({"latest_statement_date": "NaT"})["latest_statement_date"] is None
    record = _normalise_record("fundamental_cache", {
        "ticker": "TEST.JK",
        "statement_date": "NaT",
        "payload": {"latest_statement_date": pd.NaT},
        "source_checked_at": "2026-08-09T00:00:00Z",
        "model_version": "test",
        "schema_version": "test",
    })
    assert record is not None
    assert record["statement_date"] is None
    assert record["payload"]["latest_statement_date"] is None


def main() -> None:
    checks = [
        check_compile,
        check_metadata_only_fundamental_is_fail_soft,
        check_uploaded_idx_sector_survives,
        check_missing_metadata_is_not_stringified,
        check_ohlcv_audit_statuses_do_not_overwrite_acquisition,
        check_nat_never_crosses_database_contract,
    ]
    for check in checks:
        check()
        print("PASS", check.__name__)
    print("VALIDATION_V9_8_2_HOTFIX6=PASS")


if __name__ == "__main__":
    main()

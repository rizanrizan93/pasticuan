from __future__ import annotations

import pathlib
import py_compile

import pandas as pd

from resumable_app_engine import _coalesce_primary_evidence, _universe_metadata
from scanner import parse_universe_csv


ROOT = pathlib.Path(__file__).resolve().parent


def check_compile() -> None:
    for path in ROOT.glob("*.py"):
        py_compile.compile(str(path), doraise=True)


def check_explicit_upload_replaces_unknown_cached_sector() -> None:
    uploaded = parse_universe_csv(pd.DataFrame([{
        "rank_universe": 1,
        "ticker": "AADI",
        "yahoo_ticker": "AADI.JK",
        "sector_idx_ic": "Energy",
        "universe_role": "Sector Leader / Core",
        "priority": "A",
        "active_scan": 1,
    }]), max_tickers=400, strict_limit=True)
    metadata = _universe_metadata(uploaded.to_dict("records"), ["AADI.JK"])
    cached = pd.DataFrame([{
        "ticker": "AADI.JK",
        "sector": "UNKNOWN",
        "sector_raw": "",
        "sector_source": "MISSING",
        "sector_confidence_pct": 0.0,
        "fundamental_score": 68.5,
    }])
    merged = _coalesce_primary_evidence(cached, metadata)
    row = merged.iloc[0]
    assert row["sector"] == "ENERGY"
    assert row["sector_source"] == "EXPLICIT_PROVIDER"
    assert row["sector_confidence_pct"] == 100.0
    assert row["idx_sector"] == "Energy"
    assert row["sector_idx_ic"] == "Energy"
    assert row["fundamental_score"] == 68.5
    assert row["universe_role"] == "Sector Leader / Core"


def check_valid_primary_sector_still_wins() -> None:
    primary = pd.DataFrame([{
        "ticker": "AADI.JK", "sector": "ENERGY", "sector_source": "OFFICIAL",
        "sector_confidence_pct": 100.0,
    }])
    fallback = pd.DataFrame([{
        "ticker": "AADI.JK", "sector": "INDUSTRIALS", "sector_source": "EXPLICIT_PROVIDER",
        "sector_confidence_pct": 100.0,
    }])
    merged = _coalesce_primary_evidence(primary, fallback)
    assert merged.iloc[0]["sector"] == "ENERGY"
    assert merged.iloc[0]["sector_source"] == "OFFICIAL"


def main() -> None:
    checks = (
        check_compile,
        check_explicit_upload_replaces_unknown_cached_sector,
        check_valid_primary_sector_still_wins,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")


if __name__ == "__main__":
    main()

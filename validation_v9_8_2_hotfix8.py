from __future__ import annotations

import numpy as np
import pandas as pd

from scanner import (
    _median_statement_periods,
    build_fundamental_history_features,
    normalize_fundamental_history,
)
from validation_v9_8_2_hotfix6 import (
    check_compile,
    check_metadata_only_fundamental_is_fail_soft,
    check_missing_metadata_is_not_stringified,
    check_nat_never_crosses_database_contract,
    check_ohlcv_audit_statuses_do_not_overwrite_acquisition,
    check_uploaded_idx_sector_survives,
)


def _statement(
    period_end: str,
    period_type: str,
    source_family: str,
    source_verified: bool,
    revenue: float | None,
    net_income: float | None,
) -> dict[str, object]:
    return {
        "ticker": "DMAS.JK",
        "period_end": period_end,
        "period_type": period_type,
        "source_family": source_family,
        "source_verified": source_verified,
        "currency": "IDR",
        "revenue": revenue,
        "net_income": net_income,
    }


def check_period_preference_is_fact_specific() -> None:
    history = normalize_fundamental_history(pd.DataFrame([
        _statement("2026-06-30", "Q", "YAHOO", False, 750.0, 374.0),
        _statement("2026-06-30", "Q2", "IDX_OFFICIAL_XBRL", True, None, 375.0),
    ]))
    period = _median_statement_periods(history, annual=False).iloc[-1]
    assert period["revenue"] == 750.0
    assert period["net_income"] == 375.0


def check_non_contiguous_history_uses_same_quarter() -> None:
    # This mirrors the production DMAS pattern: official Q2/Q3 rows can be
    # present while standalone-quarter facts are absent, and the prior-year
    # sequence is not a contiguous four-row window.
    rows = [
        _statement("2024-12-31", "Q", "YAHOO", False, 342.925, 210.490),
        _statement("2025-03-31", "Q", "YAHOO", False, 507.885, 355.453),
        _statement("2025-03-31", "Q1", "IDX_OFFICIAL_XBRL", True, 507.885, 355.791),
        _statement("2025-06-30", "Q", "YAHOO", False, 105.473, 77.563),
        _statement("2025-06-30", "Q2", "IDX_OFFICIAL_XBRL", True, None, None),
        _statement("2025-09-30", "Q3", "IDX_OFFICIAL_XBRL", True, None, None),
        _statement("2025-12-31", "Q", "YAHOO", False, 529.508, 275.167),
        _statement("2026-03-31", "Q", "YAHOO", False, 1053.230, 818.267),
        _statement("2026-03-31", "Q1", "IDX_OFFICIAL_XBRL", True, 1053.230, 819.032),
        _statement("2026-06-30", "Q", "YAHOO", False, 750.216, 373.986),
        _statement("2026-06-30", "Q2", "IDX_OFFICIAL_XBRL", True, None, None),
    ]
    features = build_fundamental_history_features(
        pd.DataFrame(rows), now="2026-08-09",
    ).iloc[0]
    assert np.isclose(
        features["history_revenue_growth"], 750.216 / 105.473 - 1.0,
    )
    assert np.isclose(
        features["history_earnings_growth"], 373.986 / 77.563 - 1.0,
    )
    assert features["fundamental_reconciliation_state"] == "OFFICIAL_PARTIAL_PROXY_FILL"
    assert pd.Timestamp(features["fundamental_history_latest_period"]).date().isoformat() == "2026-06-30"


def main() -> None:
    checks = [
        check_compile,
        check_metadata_only_fundamental_is_fail_soft,
        check_uploaded_idx_sector_survives,
        check_missing_metadata_is_not_stringified,
        check_ohlcv_audit_statuses_do_not_overwrite_acquisition,
        check_nat_never_crosses_database_contract,
        check_period_preference_is_fact_specific,
        check_non_contiguous_history_uses_same_quarter,
    ]
    for check in checks:
        check()
        print("PASS", check.__name__)
    print("VALIDATION_V9_8_2_HOTFIX8=PASS")


if __name__ == "__main__":
    main()

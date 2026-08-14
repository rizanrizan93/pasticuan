from __future__ import annotations

"""Official-first canonicalisation for Super Scanner fundamental evidence.

Yahoo/public snapshots may enrich valuation or issuer identity, but they must not
erase point-in-time IDX XBRL provenance already present in the merged row. This
module derives evidence flags only from concrete embedded ``idx_official_*``
fields; it never invents financial values or coverage.
"""

from typing import Any

import numpy as np
import pandas as pd

OFFICIAL_EVIDENCE_GUARD_VERSION = "1.0.0-official-first"


def _present(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().fillna("").ne("")


def canonicalize_official_fundamental_evidence(frame: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    if "ticker" not in out.columns:
        return out

    def col(name: str) -> pd.Series:
        return out[name] if name in out.columns else pd.Series(pd.NA, index=out.index)

    source_url = col("idx_official_source_url")
    current_verified = col("fundamental_current_period_official_verified").astype("string").str.lower().isin(
        ["true", "1", "yes", "verified"]
    )
    provenance = col("idx_official_provenance_state").astype("string").str.upper()
    official_anchor = _present(source_url) & (current_verified | provenance.str.contains("IDX_OFFICIAL", na=False))

    income = _present(col("idx_official_revenue")) | _present(col("idx_official_net_income"))
    balance = (
        _present(col("idx_official_assets"))
        | _present(col("idx_official_equity"))
        | _present(col("idx_official_liabilities"))
        | _present(col("idx_official_cash"))
    )
    ocf = _present(col("idx_official_ocf"))
    fcf = _present(col("idx_official_fcf_proxy"))
    cashflow = ocf | fcf
    statement_families = income.astype(int) + balance.astype(int) + cashflow.astype(int)
    official_cov = (100.0 * statement_families / 3.0).round(1)
    cashflow_cov = np.where(ocf & fcf, 100.0, np.where(ocf | fcf, 50.0, 0.0))
    valid = official_anchor & statement_families.ge(1)

    if not valid.any():
        return out

    out.loc[valid, "fundamental_official_verified"] = True
    out.loc[valid, "fundamental_official_reference"] = True
    out.loc[valid, "valuation_statement_official_verified"] = True
    out.loc[valid, "fundamental_official_source_coverage_pct"] = official_cov.loc[valid]
    out.loc[valid, "fundamental_statement_family_coverage_pct"] = official_cov.loc[valid]
    out.loc[valid, "fundamental_cashflow_statement_coverage_pct"] = pd.Series(cashflow_cov, index=out.index).loc[valid]
    out.loc[valid, "fundamental_official_state"] = "IDX_OFFICIAL_XBRL_VERIFIED"
    out.loc[valid, "fundamental_official_source_urls"] = source_url.loc[valid].astype(str)
    if "idx_official_period_end" in out.columns:
        out.loc[valid, "fundamental_official_latest_period"] = out.loc[valid, "idx_official_period_end"]

    current_families = col("fundamental_source_families").astype("string").fillna("")
    has_yahoo = current_families.str.upper().str.contains("YAHOO", na=False)
    out.loc[valid & has_yahoo, "fundamental_source_families"] = "IDX_OFFICIAL_XBRL • YAHOO"
    out.loc[valid & ~has_yahoo, "fundamental_source_families"] = "IDX_OFFICIAL_XBRL"
    out.loc[valid & has_yahoo, "fundamental_reconciliation_state"] = "OFFICIAL_PLUS_PUBLIC_CROSSCHECK"
    out.loc[valid & ~has_yahoo, "fundamental_reconciliation_state"] = "OFFICIAL_PRIMARY"

    for field in ("fundamental_source_count", "fundamental_all_source_count"):
        existing = pd.to_numeric(col(field), errors="coerce").fillna(0.0)
        out.loc[valid, field] = np.maximum(existing.loc[valid], np.where(has_yahoo.loc[valid], 2.0, 1.0))
    out.loc[valid, "fundamental_snapshot_source_count"] = 1
    out.loc[valid, "official_evidence_guard_version"] = OFFICIAL_EVIDENCE_GUARD_VERSION
    return out


__all__ = ["OFFICIAL_EVIDENCE_GUARD_VERSION", "canonicalize_official_fundamental_evidence"]

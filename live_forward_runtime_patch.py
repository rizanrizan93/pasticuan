from __future__ import annotations

"""Company-identity enrichment for scan-time forward evidence acquisition."""

from functools import wraps
from typing import Any, Iterable

import pandas as pd

from live_forward_evidence import collect_live_forward_evidence

PATCH_VERSION = "1.0.0"


def _ticker(value: Any) -> str:
    text = str(value or "").upper().strip()
    return text if text.endswith(".JK") else f"{text}.JK" if text else ""


def _company_names(bridge: Any, tickers: list[str]) -> dict[str, str]:
    if not tickers or getattr(getattr(bridge, "settings", None), "mode", "") != "SUPABASE_REST":
        return {}
    try:
        rows = bridge._get_rows("issuer_master", {"select": "ticker,company_name", "limit": str(max(1000, len(tickers) * 2))})
    except Exception:
        return {}
    universe = set(tickers)
    return {
        _ticker(row.get("ticker")): str(row.get("company_name") or "").strip()
        for row in rows
        if _ticker(row.get("ticker")) in universe and str(row.get("company_name") or "").strip()
    }


def _fresh_checked(frame: pd.DataFrame, max_age_hours: float = 24.0) -> set[str]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return set()
    local = frame.copy()
    local["ticker"] = local["ticker"].map(_ticker)
    state = local.get("forward_collection_state", pd.Series("", index=local.index)).fillna("").astype(str)
    stamp = pd.Series(pd.NaT, index=local.index, dtype="datetime64[ns, UTC]")
    for column in ("source_checked_at", "database_source_checked_at", "last_verified_at"):
        if column in local.columns:
            parsed = pd.to_datetime(local[column], errors="coerce", utc=True)
            stamp = stamp.where(stamp.notna(), parsed)
    age = (pd.Timestamp.now(tz="UTC") - stamp).dt.total_seconds() / 3600.0
    return set(local.loc[state.str.len().gt(0) & age.between(0.0, float(max_age_hours)), "ticker"])


def install(database_cls: Any) -> None:
    original = getattr(database_cls, "read_forward_quality_cache", None)
    if not callable(original) or getattr(original, "__company_identity_forward_v1__", False):
        return

    @wraps(original)
    def wrapped(self: Any, tickers: Iterable[Any]):
        names = list(dict.fromkeys(_ticker(value) for value in tickers if _ticker(value)))
        if names and getattr(getattr(self, "settings", None), "mode", "") == "SUPABASE_REST":
            try:
                current, _ = self._read_cache_table("forward_quality_cache", names, self.settings.forward_max_age_days, "FORWARD_QUALITY")
            except Exception:
                current = pd.DataFrame()
            missing = [ticker for ticker in names if ticker not in _fresh_checked(current)]
            if missing:
                company_names = _company_names(self, missing)
                live = collect_live_forward_evidence(
                    missing,
                    company_names=company_names,
                    lookback_days=180,
                    max_workers=12,
                    timeout=5.0,
                )
                if isinstance(live, pd.DataFrame) and not live.empty:
                    try:
                        self.persist_scan_result(
                            {
                                "mode": "live_forward_company_entity_refresh",
                                "scanner_version": getattr(self, "scanner_version", ""),
                                "as_of": pd.Timestamp.now(tz="UTC"),
                                "project_management_review": live,
                            },
                            tables=("forward_quality_cache",),
                        )
                    except Exception:
                        pass
        return original(self, names)

    wrapped.__company_identity_forward_v1__ = True
    setattr(database_cls, "read_forward_quality_cache", wrapped)


def install_runtime() -> None:
    try:
        import scanner_database
    except Exception:
        return
    install(scanner_database.ScannerDatabaseBridge)


__all__ = ["PATCH_VERSION", "install", "install_runtime"]

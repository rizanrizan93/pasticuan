from __future__ import annotations

"""Transport-only Phase 5.6 coverage integrity fixes.

This patch closes two persistence/binding gaps without changing scoring, ranking,
setup detection, execution geometry, or authorization:
- public ownership context fills missing canonical ``ownership_public_*`` fields
  instead of being stranded in merge suffix columns when the canonical columns
  already exist but are null;
- current technical aliases fill null/blank legacy audit fields as well as
  entirely absent columns, while preserving any explicit legacy value.

The public ownership context remains non-KSEI/non-regulatory context.  Missing
RR/SL/TP, direct forward quorum, regulatory free float, and authorization are
never inferred here.
"""

from typing import Any, Iterable, Mapping

import pandas as pd

PATCH_VERSION = "1.0.0-phase5.6-runtime-coverage-coalescing"


def _meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return not bool(pd.isna(value))
    except Exception:
        return True


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _coalescing_ownership_merge(
    frame: pd.DataFrame,
    context: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Fill only missing canonical public-ownership context fields.

    Existing meaningful canonical values always win.  This deliberately avoids
    ``DataFrame.merge(..., suffixes=...)`` because an already-present-but-null
    canonical column otherwise hides the valid context in ``*_phase56``.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return frame
    normalized_context = {
        _ticker(ticker): dict(values or {})
        for ticker, values in dict(context or {}).items()
        if _ticker(ticker)
    }
    if not normalized_context:
        return frame

    local = frame.copy()
    local["ticker"] = local["ticker"].map(_ticker)
    fields = sorted({
        field
        for values in normalized_context.values()
        for field in values
        if str(field).startswith("ownership_public_")
    })
    for field in fields:
        incoming = local["ticker"].map(
            lambda ticker: normalized_context.get(str(ticker), {}).get(field)
        )
        if field not in local.columns:
            local[field] = incoming
            continue
        missing = ~local[field].map(_meaningful)
        local.loc[missing, field] = incoming.loc[missing]
    return local


def _fill_alias(
    frame: pd.DataFrame,
    target: str,
    sources: Iterable[str],
) -> None:
    if target not in frame.columns:
        frame[target] = pd.NA
    missing = ~frame[target].map(_meaningful)
    if not bool(missing.any()):
        return
    for source in sources:
        if source not in frame.columns:
            continue
        candidate = frame[source]
        usable = candidate.map(_meaningful)
        take = missing & usable
        if bool(take.any()):
            frame.loc[take, target] = candidate.loc[take]
            missing = ~frame[target].map(_meaningful)
        if not bool(missing.any()):
            break


def _null_aware_technical_aliases(
    frame: pd.DataFrame | None,
    columns: Iterable[str],
) -> pd.DataFrame | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    requested = set(columns)
    if "active_setup" not in requested and "technical_entry_state" not in requested:
        return frame
    local = frame.copy()
    if "active_setup" in requested:
        _fill_alias(local, "active_setup", ("strategy", "setup"))
    if "technical_entry_state" in requested:
        _fill_alias(local, "technical_entry_state", ("setup_status", "status", "decision_state"))
    return local


def install() -> dict[str, str]:
    import phase56_coverage_runtime_patch as coverage
    import technical_persistence_semantics_patch as technical

    if getattr(coverage, "_phase56_runtime_coverage_integrity_patch", "") == PATCH_VERSION:
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}

    coverage._merge_context = _coalescing_ownership_merge
    technical._with_current_aliases = _null_aware_technical_aliases
    coverage._phase56_runtime_coverage_integrity_patch = PATCH_VERSION
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "ownership": "CANONICAL_NULLS_FILLED_PUBLIC_CONTEXT_ONLY",
        "technical": "NULL_AWARE_PERSISTENCE_ALIASES_ONLY",
    }


__all__ = [
    "PATCH_VERSION",
    "install",
    "_coalescing_ownership_merge",
    "_null_aware_technical_aliases",
]

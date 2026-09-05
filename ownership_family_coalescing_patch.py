from __future__ import annotations

"""State-aware coalescing for canonical ownership context.

Production frames pre-initialize ownership telemetry with numeric zeroes.  A
plain null-aware merge therefore mistakes an unpopulated family for observed
zero evidence and blocks valid Shared Hub facts.  This patch replaces the whole
family only when its family-level coverage/provenance says it is still missing;
once a family is valid, explicit observed zeroes remain authoritative.

This is transport/persistence only.  It never derives regulatory free float,
beneficial ownership, score, rank, recommendation, or execution authorization.
"""

from typing import Any, Mapping

import numpy as np
import pandas as pd

PATCH_VERSION = "1.0.0-ownership-family-state-coalescing"


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _family_context(context: Mapping[str, Mapping[str, Any]], prefix: str) -> dict[str, dict[str, Any]]:
    return {
        _ticker(ticker): {str(k): v for k, v in dict(values or {}).items() if str(k).startswith(prefix)}
        for ticker, values in dict(context or {}).items()
        if _ticker(ticker)
    }


def _public_family_valid(row: Mapping[str, Any]) -> bool:
    coverage = _number(row.get("ownership_public_context_coverage_pct"))
    return bool(np.isfinite(coverage) and coverage > 0.0 and str(row.get("ownership_public_source_authority") or "").strip())


def _ksei_family_valid(row: Mapping[str, Any]) -> bool:
    coverage = _number(row.get("ownership_ksei_context_coverage_pct"))
    authority = str(row.get("ownership_ksei_source_authority") or "").strip().upper()
    verified = bool(row.get("ownership_ksei_official_verified"))
    return bool(np.isfinite(coverage) and coverage > 0.0 and verified and authority == "OFFICIAL_KSEI")


def _coalesce_family(
    frame: pd.DataFrame,
    context: Mapping[str, Mapping[str, Any]],
    *,
    prefix: str,
    valid_family,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return frame
    normalized = _family_context(context, prefix)
    if not normalized:
        return frame

    out = frame.copy()
    out["ticker"] = out["ticker"].map(_ticker)
    fields = sorted({field for values in normalized.values() for field in values})
    for field in fields:
        if field not in out.columns:
            out[field] = pd.NA

    for idx in out.index:
        ticker = str(out.at[idx, "ticker"] or "")
        incoming = normalized.get(ticker)
        if not incoming:
            continue
        existing = out.loc[idx].to_dict()
        family_is_valid = bool(valid_family(existing))
        for field, value in incoming.items():
            # An invalid family consists of placeholders and is replaced as one
            # unit.  A valid family preserves explicit values, including 0/100.
            if not family_is_valid or _missing(out.at[idx, field]):
                out.at[idx, field] = value
    return out


def merge_public(frame: pd.DataFrame, context: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    return _coalesce_family(
        frame,
        context,
        prefix="ownership_public_",
        valid_family=_public_family_valid,
    )


def merge_ksei(frame: pd.DataFrame, context: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    return _coalesce_family(
        frame,
        context,
        prefix="ownership_ksei_",
        valid_family=_ksei_family_valid,
    )


def install() -> dict[str, str]:
    import phase56_coverage_runtime_patch as public_context
    import pasticuan_ksei_runtime_patch as ksei_context

    public_context._merge_context = merge_public
    ksei_context._merge_context = merge_ksei
    public_context._ownership_family_coalescing_patch = PATCH_VERSION
    ksei_context._ownership_family_coalescing_patch = PATCH_VERSION
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "public": "PLACEHOLDER_FAMILY_REPLACED_BY_CANONICAL_CONTEXT",
        "ksei": "PLACEHOLDER_FAMILY_REPLACED_BY_OFFICIAL_KSEI_CONTEXT",
        "free_float": "NOT_INFERRED",
        "authorization": "UNCHANGED",
    }


__all__ = ["PATCH_VERSION", "install", "merge_public", "merge_ksei"]

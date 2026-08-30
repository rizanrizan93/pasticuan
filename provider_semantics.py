"""Canonical, fail-closed provider and provenance semantics.

Provider identity and evidence authority are deliberately separate.  Adapters
may retain repository-specific columns while using :class:`ProviderResult` as
the internal semantic representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Number
from typing import Any, Mapping

import pandas as pd


class ProviderStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    STALE = "STALE"
    INVALID = "INVALID"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CONFLICT = "CONFLICT"
    AMBIGUOUS = "AMBIGUOUS"


class EvidenceProvenance(str, Enum):
    DIRECT_OR_OFFICIAL = "DIRECT_OR_OFFICIAL"
    VERIFIED = "VERIFIED"
    PUBLIC_RESEARCH = "PUBLIC_RESEARCH"
    INFERRED = "INFERRED"
    PROXY = "PROXY"
    MISSING = "MISSING"


class Freshness(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ProviderErrorCategory(str, Enum):
    NONE = "NONE"
    TIMEOUT = "TIMEOUT"
    AUTHENTICATION = "AUTHENTICATION"
    RATE_LIMIT = "RATE_LIMIT"
    TRANSPORT = "TRANSPORT"
    PARSE = "PARSE"
    SCHEMA = "SCHEMA"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProviderResult:
    status: ProviderStatus
    provider: str
    value: Any = None
    observed_at: Any = None
    requested_window: Any = None
    observed_window: Any = None
    freshness: Freshness = Freshness.UNKNOWN
    completeness: float | None = None
    provenance: EvidenceProvenance = EvidenceProvenance.MISSING
    error_category: ProviderErrorCategory = ProviderErrorCategory.NONE
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status is ProviderStatus.SUCCESS


_STATUS_ALIASES = {
    "OK": ProviderStatus.SUCCESS,
    "VALID": ProviderStatus.SUCCESS,
    "CURRENT": ProviderStatus.SUCCESS,
    "CACHE_HIT": ProviderStatus.SUCCESS,
    "COLD_REFRESH": ProviderStatus.SUCCESS,
    "INCREMENTAL_REFRESH": ProviderStatus.SUCCESS,
    "NO_ITEMS": ProviderStatus.SUCCESS,
    "FULL": ProviderStatus.SUCCESS,
    "PARTIAL_ONLY": ProviderStatus.PARTIAL,
    "INSUFFICIENT": ProviderStatus.PARTIAL,
    "STALE_CACHE": ProviderStatus.STALE,
    "STALE_CACHE_FALLBACK": ProviderStatus.STALE,
    "UNAVAILABLE": ProviderStatus.MISSING,
    "NO_DATA": ProviderStatus.MISSING,
    "NONE": ProviderStatus.MISSING,
    "N/A": ProviderStatus.NOT_APPLICABLE,
    "NA": ProviderStatus.NOT_APPLICABLE,
    "ERROR": ProviderStatus.PROVIDER_ERROR,
    "FAILED": ProviderStatus.PROVIDER_ERROR,
    "FAIL": ProviderStatus.PROVIDER_ERROR,
    "PROVIDER_FAILED": ProviderStatus.PROVIDER_ERROR,
    "TIMEOUT": ProviderStatus.PROVIDER_ERROR,
    "MALFORMED": ProviderStatus.INVALID,
    "UNKNOWN": ProviderStatus.AMBIGUOUS,
    "FORWARD_CHECK_COMPLETED_NO_MATERIAL_EVENT": ProviderStatus.SUCCESS,
    "MATERIAL_FORWARD_RESEARCH_EVIDENCE_FOUND": ProviderStatus.SUCCESS,
    "FORWARD_CHECK_FAILED_RETRYABLE": ProviderStatus.PROVIDER_ERROR,
    "CURRENT_RELATIVE_TO_UNIVERSE": ProviderStatus.SUCCESS,
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if not pd.api.types.is_scalar(value):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _status(value: Any) -> ProviderStatus | None:
    if isinstance(value, ProviderStatus):
        return value
    text = str(value or "").strip().upper()
    if not text:
        return None
    try:
        return ProviderStatus(text)
    except ValueError:
        if text in _STATUS_ALIASES:
            return _STATUS_ALIASES[text]
        if "STALE" in text:
            return ProviderStatus.STALE
        if any(token in text for token in ("ERROR", "FAIL", "TIMEOUT", "RATE_LIMIT")):
            return ProviderStatus.PROVIDER_ERROR
        if any(token in text for token in ("MISSING", "NOT_AVAILABLE", "NO_VALID")):
            return ProviderStatus.MISSING
        if any(token in text for token in ("PARTIAL", "INSUFFICIENT")):
            return ProviderStatus.PARTIAL
        if "INVALID" in text:
            return ProviderStatus.INVALID
        if "CONFLICT" in text or "UNRESOLVED" in text:
            return ProviderStatus.CONFLICT
        return None


def normalize_provenance(value: Any) -> EvidenceProvenance:
    """Map legacy authority labels without consulting provider identity."""
    if isinstance(value, EvidenceProvenance):
        return value
    if _is_missing(value):
        return EvidenceProvenance.MISSING
    text = str(value).strip().upper()
    if not text or text in {"MISSING", "NONE", "UNAVAILABLE", "NOT_AVAILABLE", "PROVIDER_FAILED"}:
        return EvidenceProvenance.MISSING
    try:
        return EvidenceProvenance(text)
    except ValueError:
        pass
    if any(token in text for token in ("PUBLIC_RESEARCH", "PUBLIC_NEWS", "GOOGLE_NEWS", "RESEARCH_ONLY", "EXPLICIT_PUBLIC", "PUBLIC_SYNTHESIS")):
        return EvidenceProvenance.PUBLIC_RESEARCH
    if any(token in text for token in ("PROXY", "BEHAVIOURAL", "BEHAVIORAL")):
        return EvidenceProvenance.PROXY
    if any(token in text for token in ("INFERRED", "DERIVED", "SYNTHETIC")):
        return EvidenceProvenance.INFERRED
    if any(token in text for token in ("OFFICIAL", "IDX_PUBLIC", "DIRECT_SOURCE", "DIRECT_OR_OFFICIAL")):
        return EvidenceProvenance.DIRECT_OR_OFFICIAL
    if any(token in text for token in ("VERIFIED", "AUTHENTICATED")):
        return EvidenceProvenance.VERIFIED
    return EvidenceProvenance.MISSING


_PROVENANCE_TRUST = {
    EvidenceProvenance.MISSING: 0,
    EvidenceProvenance.PROXY: 1,
    EvidenceProvenance.INFERRED: 2,
    EvidenceProvenance.PUBLIC_RESEARCH: 3,
    EvidenceProvenance.VERIFIED: 4,
    EvidenceProvenance.DIRECT_OR_OFFICIAL: 5,
}


def aggregate_provenance(*values: Any) -> EvidenceProvenance:
    """Return the weakest input authority, so aggregation cannot upgrade it."""
    normalized = [normalize_provenance(value) for value in values]
    if not normalized:
        return EvidenceProvenance.MISSING
    return min(normalized, key=_PROVENANCE_TRUST.__getitem__)


def _error_category(value: Any) -> ProviderErrorCategory:
    text = str(value or "").strip().upper()
    if not text:
        return ProviderErrorCategory.NONE
    if "TIMEOUT" in text:
        return ProviderErrorCategory.TIMEOUT
    if any(token in text for token in ("AUTH", "UNAUTHORIZED", "FORBIDDEN")):
        return ProviderErrorCategory.AUTHENTICATION
    if any(token in text for token in ("RATE", "429", "QUOTA")):
        return ProviderErrorCategory.RATE_LIMIT
    if any(token in text for token in ("PARSE", "JSON", "XML")):
        return ProviderErrorCategory.PARSE
    if any(token in text for token in ("SCHEMA", "FIELD", "COLUMN")):
        return ProviderErrorCategory.SCHEMA
    if any(token in text for token in ("HTTP", "NETWORK", "CONNECTION", "DNS")):
        return ProviderErrorCategory.TRANSPORT
    return ProviderErrorCategory.UNKNOWN


def normalize_provider_result(
    value: Any,
    *,
    provider: str = "UNKNOWN",
    status: Any = None,
    provenance: Any = None,
    observed_at: Any = None,
    requested_window: Any = None,
    observed_window: Any = None,
    freshness: Any = None,
    completeness: Any = None,
    error: Any = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> ProviderResult:
    """Normalize legacy evidence conservatively; unknown never means success."""
    if isinstance(value, ProviderResult):
        requested_provider = str(provider or "").strip()
        if requested_provider and requested_provider.upper() != "UNKNOWN" and requested_provider != value.provider:
            return replace(
                value,
                status=ProviderStatus.INVALID,
                error_category=ProviderErrorCategory.SCHEMA,
                diagnostics={
                    **dict(value.diagnostics),
                    "provider_identity_conflict": requested_provider,
                },
            )
        return value

    record = value if isinstance(value, Mapping) else {}
    status_envelope = bool(record) and ("status" in record or "state" in record)
    explicit_payload = not record or "value" in record or "payload" in record
    raw_status = status if status is not None else record.get("status", record.get("state"))
    if "value" in record:
        payload = record["value"]
    elif "payload" in record:
        payload = record["payload"]
    elif status_envelope:
        payload = None
    else:
        payload = record if record else value
    raw_error = error if error is not None else record.get("error", record.get("detail") if _status(raw_status) is ProviderStatus.PROVIDER_ERROR else None)
    resolved = _status(raw_status)
    if resolved is None:
        if record or (not _is_missing(raw_status) and str(raw_status).strip()):
            resolved = ProviderStatus.INVALID
        elif _is_missing(payload):
            resolved = ProviderStatus.MISSING
        elif isinstance(payload, bool):
            resolved = ProviderStatus.INVALID
        elif isinstance(payload, Number) and not _is_missing(payload):
            resolved = ProviderStatus.SUCCESS
        else:
            resolved = ProviderStatus.INVALID
    if not _is_missing(raw_error) and str(raw_error).strip() and resolved not in {ProviderStatus.MISSING, ProviderStatus.NOT_APPLICABLE}:
        resolved = ProviderStatus.PROVIDER_ERROR
    if resolved is ProviderStatus.SUCCESS and (not explicit_payload or _is_missing(payload)):
        resolved = ProviderStatus.INVALID
    if not explicit_payload and resolved in {
        ProviderStatus.MISSING, ProviderStatus.INVALID, ProviderStatus.PROVIDER_ERROR,
        ProviderStatus.NOT_APPLICABLE, ProviderStatus.AMBIGUOUS,
    }:
        payload = None
    if resolved is ProviderStatus.MISSING:
        payload = None

    raw_freshness = freshness if freshness is not None else record.get("freshness")
    fresh_text = str(raw_freshness or "").strip().upper()
    resolved_freshness = (
        Freshness.FRESH if fresh_text in {"FRESH", "CURRENT", "CURRENT_COMPLETED_SESSION"}
        else Freshness.STALE if "STALE" in fresh_text or resolved is ProviderStatus.STALE
        else Freshness.NOT_APPLICABLE if resolved is ProviderStatus.NOT_APPLICABLE
        else Freshness.UNKNOWN
    )
    if resolved is ProviderStatus.SUCCESS and resolved_freshness is Freshness.STALE:
        resolved = ProviderStatus.STALE
    raw_completeness = completeness if completeness is not None else record.get("completeness", record.get("coverage_ratio"))
    try:
        normalized_completeness = float(raw_completeness) if not _is_missing(raw_completeness) else None
    except (TypeError, ValueError):
        normalized_completeness = None
        if resolved is ProviderStatus.SUCCESS:
            resolved = ProviderStatus.INVALID
    if normalized_completeness is not None and not 0.0 <= normalized_completeness <= 1.0:
        if 0.0 <= normalized_completeness <= 100.0:
            normalized_completeness /= 100.0
        else:
            normalized_completeness = None
            resolved = ProviderStatus.INVALID

    raw_provenance = provenance if provenance is not None else record.get(
        "canonical_provenance", record.get("provenance", record.get("provenance_state"))
    )
    return ProviderResult(
        status=resolved,
        provider=str(provider or record.get("provider") or "UNKNOWN").strip() or "UNKNOWN",
        value=payload,
        observed_at=observed_at if observed_at is not None else record.get("observed_at", record.get("last_date")),
        requested_window=requested_window if requested_window is not None else record.get("requested_window", record.get("requested_observations")),
        observed_window=observed_window if observed_window is not None else record.get("observed_window", record.get("observed_observations")),
        freshness=resolved_freshness,
        completeness=normalized_completeness,
        provenance=normalize_provenance(raw_provenance),
        error_category=(
            _error_category(raw_error)
            if resolved is ProviderStatus.PROVIDER_ERROR and not _is_missing(raw_error)
            else ProviderErrorCategory.UNKNOWN
            if resolved is ProviderStatus.PROVIDER_ERROR
            else ProviderErrorCategory.NONE
        ),
        diagnostics=dict(diagnostics or record.get("diagnostics") or {}),
    )


def canonicalize_provider_audit(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Add backward-compatible canonical semantic columns to provider audits."""
    out = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if out.empty:
        for name in ("provider_result_status", "provider_result_freshness", "canonical_provenance", "provider_error_category"):
            out[name] = pd.Series(dtype="string")
        return out
    results: list[ProviderResult] = []
    for row in out.to_dict("records"):
        audit_value = next(
            (row.get(name) for name in ("value", "payload", "bars", "rows") if name in row),
            None,
        )
        provenance = next((row.get(name) for name in (
            "canonical_provenance", "provenance", "provenance_state", "review_origin",
        ) if not _is_missing(row.get(name))), None)
        results.append(normalize_provider_result(
            audit_value,
            provider=row.get("provider", "UNKNOWN"),
            status=row.get("status", row.get("state", row.get("quality_state"))),
            provenance=provenance,
            observed_at=row.get("observed_at", row.get("last_date", row.get("checked_at"))),
            freshness=row.get("freshness", row.get("completed_session_state", row.get("quality_state"))),
            completeness=row.get("completeness", row.get("coverage_ratio", row.get("coverage_pct"))),
            error=row.get("error"),
        ))
    out["provider_result_status"] = [result.status.value for result in results]
    out["provider_result_freshness"] = [result.freshness.value for result in results]
    out["canonical_provenance"] = [result.provenance.value for result in results]
    out["provider_error_category"] = [result.error_category.value for result in results]
    return out


__all__ = [
    "EvidenceProvenance", "Freshness", "ProviderErrorCategory", "ProviderResult",
    "ProviderStatus", "aggregate_provenance", "canonicalize_provider_audit",
    "normalize_provider_result", "normalize_provenance",
]

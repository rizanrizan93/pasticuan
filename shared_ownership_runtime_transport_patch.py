from __future__ import annotations

"""Route PASTICUAN ownership context through the canonical Shared Hub client.

The scan-time public-ownership and KSEI adapters historically opened a second
REST path from ``DatabaseSettings`` even though the same facts already live in
the Shared Evidence Hub.  Canonical forward evidence uses ``HubConfig`` and was
healthy in production, while both ownership adapters returned empty context.

This transport-only patch makes the two ownership loaders use the same Shared
Hub configuration/backend as the canonical forward layer.  It changes no score,
rank, regulatory-free-float inference, or execution authorization.
"""

import time
from typing import Any, Iterable, Mapping

from shared_evidence_hub import HubConfig, SupabaseEvidenceBackend
from shared_ksei_holding_composition import CATEGORY, canonical_context


PATCH_VERSION = "1.0.0-shared-ownership-runtime-transport"
PUBLIC_TABLE = "phase56_public_ownership_snapshots"
KSEI_TABLE = "evidence_ownership_snapshots"
PUBLIC_CACHE_TTL_SECONDS = 6 * 60 * 60
KSEI_CACHE_TTL_SECONDS = 12 * 60 * 60

_PUBLIC_CACHE: dict[str, dict[str, Any]] = {}
_PUBLIC_CACHE_AT = 0.0
_KSEI_CACHE: dict[str, dict[str, Any]] = {}
_KSEI_CACHE_AT = 0.0


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _bare(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _requested(tickers: Iterable[str] | None) -> set[str]:
    return {_ticker(value) for value in (tickers or []) if _ticker(value)}


def _backend() -> SupabaseEvidenceBackend | None:
    config = HubConfig.from_environment(client_id="PASTICUAN")
    if not config.ready:
        return None
    return SupabaseEvidenceBackend(config)


def _shared_public_ownership(tickers: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    global _PUBLIC_CACHE, _PUBLIC_CACHE_AT
    now = time.monotonic()
    requested = _requested(tickers)
    if _PUBLIC_CACHE and now - _PUBLIC_CACHE_AT < PUBLIC_CACHE_TTL_SECONDS:
        return {
            key: dict(value) for key, value in _PUBLIC_CACHE.items()
            if not requested or key in requested
        }

    backend = _backend()
    if backend is None:
        return {
            key: dict(value) for key, value in _PUBLIC_CACHE.items()
            if not requested or key in requested
        }
    try:
        rows = backend.read_rows(PUBLIC_TABLE, {}, limit=50000)
    except Exception:
        return {
            key: dict(value) for key, value in _PUBLIC_CACHE.items()
            if not requested or key in requested
        }

    fresh: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        ticker = _ticker(raw.get("ticker"))
        if not ticker:
            continue
        fresh[ticker] = {
            "ownership_public_insiders_held_pct": raw.get("insiders_held_pct"),
            "ownership_public_institutions_held_pct": raw.get("institutions_held_pct"),
            "ownership_public_institutions_float_held_pct": raw.get("institutions_float_held_pct"),
            "ownership_public_institutions_count": raw.get("institutions_count"),
            "ownership_public_context_coverage_pct": raw.get("coverage_pct") or 0.0,
            "ownership_public_source_period": raw.get("source_period"),
            "ownership_public_observed_on": raw.get("observed_on"),
            "ownership_public_source_authority": str(raw.get("source_authority") or "PUBLIC_PROVIDER"),
            "ownership_public_official_verified": bool(raw.get("official_verified", False)),
            "ownership_public_context_provenance_state": str(
                raw.get("provenance_state") or "PUBLIC_PROVIDER_NOT_IDX_KSEI"
            ),
            "ownership_public_context_state": "CONTEXT_ONLY_NOT_REGULATORY_FREE_FLOAT",
        }
    if fresh:
        _PUBLIC_CACHE = fresh
        _PUBLIC_CACHE_AT = now
    return {
        key: dict(value) for key, value in fresh.items()
        if not requested or key in requested
    }


def _shared_ksei_ownership(tickers: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    global _KSEI_CACHE, _KSEI_CACHE_AT
    now = time.monotonic()
    requested = _requested(tickers)
    if _KSEI_CACHE and now - _KSEI_CACHE_AT < KSEI_CACHE_TTL_SECONDS:
        return {
            key: dict(value) for key, value in _KSEI_CACHE.items()
            if not requested or key in requested
        }

    backend = _backend()
    if backend is None:
        return {
            key: dict(value) for key, value in _KSEI_CACHE.items()
            if not requested or key in requested
        }
    try:
        rows = backend.read_rows(KSEI_TABLE, {}, limit=50000)
    except Exception:
        return {
            key: dict(value) for key, value in _KSEI_CACHE.items()
            if not requested or key in requested
        }

    valid_rows: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        ticker = _ticker(raw.get("ticker"))
        if not ticker:
            continue
        if str(raw.get("category") or "") != CATEGORY:
            continue
        if not bool(raw.get("source_verified")):
            continue
        if str(raw.get("validation_state") or "").upper() != "VALID":
            continue
        item = dict(raw)
        item["ticker"] = _bare(ticker)
        valid_rows.append(item)

    facts = canonical_context(valid_rows)
    fresh: dict[str, dict[str, Any]] = {}
    for ticker, values in facts.items():
        normalized = _ticker(ticker)
        if not normalized:
            continue
        observed = str(values.get("ownership_ksei_observed_on") or "")
        source_url = next((
            str(row.get("source_url") or "")
            for row in valid_rows
            if _ticker(row.get("ticker")) == normalized
            and str(row.get("report_date") or "") == observed
            and row.get("source_url")
        ), "")
        fresh[normalized] = {**dict(values), "ownership_ksei_source_url": source_url}

    if fresh:
        _KSEI_CACHE = fresh
        _KSEI_CACHE_AT = now
    return {
        key: dict(value) for key, value in fresh.items()
        if not requested or key in requested
    }


def install() -> dict[str, str]:
    import phase56_coverage_runtime_patch as public_context
    import pasticuan_ksei_runtime_patch as ksei_context

    public_context._load_public_ownership = _shared_public_ownership
    ksei_context._load_ksei_ownership = _shared_ksei_ownership
    public_context._shared_ownership_runtime_transport_patch = PATCH_VERSION
    ksei_context._shared_ownership_runtime_transport_patch = PATCH_VERSION
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "transport": "SHARED_HUB_BACKEND_FOR_PUBLIC_AND_KSEI_CONTEXT",
        "public_ownership": "PUBLIC_PROVIDER_CONTEXT_NOT_OFFICIAL",
        "ksei": "OFFICIAL_REGISTRATION_COMPOSITION_NOT_REGULATORY_FREE_FLOAT",
        "authorization": "UNCHANGED",
    }


__all__ = [
    "PATCH_VERSION",
    "_shared_public_ownership",
    "_shared_ksei_ownership",
    "install",
]

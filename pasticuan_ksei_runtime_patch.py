from __future__ import annotations

"""Expose persisted official KSEI holding-composition facts to PASTICUAN runtime.

Observability/context only.  No KSEI scripless composition value is converted
into regulatory free float, beneficial ownership, score, rank, or execution
authorization.
"""

import time
from typing import Any, Iterable, Mapping

import pandas as pd
import requests

from scanner_database import DatabaseSettings
from shared_ksei_holding_composition import CATEGORY, canonical_context

PATCH_VERSION = "1.0.0-phase5.6-ksei-runtime-context"
CACHE_TTL_SECONDS = 12 * 60 * 60
_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_AT = 0.0


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _bare(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _load_ksei_ownership(tickers: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    global _CACHE, _CACHE_AT
    now = time.monotonic()
    if _CACHE and now - _CACHE_AT < CACHE_TTL_SECONDS:
        return {key: dict(value) for key, value in _CACHE.items()}
    settings = DatabaseSettings.from_env()
    if settings.mode != "SUPABASE_REST" or not settings.supabase_url or not settings.supabase_key:
        return {key: dict(value) for key, value in _CACHE.items()}

    symbols = list(dict.fromkeys(_bare(value) for value in (tickers or []) if _bare(value)))
    chunks: list[list[str] | None] = [None] if not symbols else [symbols[start:start + 70] for start in range(0, len(symbols), 70)]
    headers = {
        "apikey": settings.supabase_key,
        "Accept": "application/json",
        "Accept-Profile": settings.schema,
    }
    if settings.supabase_key_type == "SERVICE_ROLE":
        headers["Authorization"] = f"Bearer {settings.supabase_key}"
    rows: list[dict[str, Any]] = []
    try:
        with requests.Session() as http:
            for chunk in chunks:
                params: dict[str, Any] = {
                    "select": "category,ticker,holder_classification,shares_held,ownership_percentage,report_date,source_url,source_verified,validation_state,fetched_at",
                    "category": f"eq.{CATEGORY}",
                    "source_verified": "eq.true",
                    "validation_state": "eq.VALID",
                    "order": "report_date.desc,fetched_at.desc",
                    "limit": 2000,
                }
                if chunk:
                    quoted = ",".join(f'"{symbol}"' for symbol in chunk)
                    params["ticker"] = f"in.({quoted})"
                response = http.get(
                    f"{settings.supabase_url}/rest/v1/evidence_ownership_snapshots",
                    params=params,
                    headers=headers,
                    timeout=min(10.0, float(settings.timeout_seconds)),
                )
                if response.status_code != 200:
                    return {key: dict(value) for key, value in _CACHE.items()}
                payload = response.json()
                if not isinstance(payload, list):
                    return {key: dict(value) for key, value in _CACHE.items()}
                rows.extend(dict(item) for item in payload if isinstance(item, Mapping))
    except Exception:
        return {key: dict(value) for key, value in _CACHE.items()}

    fresh_raw = canonical_context(rows)
    fresh: dict[str, dict[str, Any]] = {}
    for ticker, values in fresh_raw.items():
        normalized = _ticker(ticker)
        if not normalized:
            continue
        source_url = next(
            (str(row.get("source_url") or "") for row in rows if _bare(row.get("ticker")) == _bare(ticker) and str(row.get("report_date") or "") == str(values.get("ownership_ksei_observed_on") or "") and row.get("source_url")),
            "",
        )
        fresh[normalized] = {**dict(values), "ownership_ksei_source_url": source_url}
    if fresh:
        _CACHE = fresh
        _CACHE_AT = now
    return {key: dict(value) for key, value in _CACHE.items()}


def _merge_context(frame: pd.DataFrame, context: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns or not context:
        return frame
    local = frame.copy()
    local["ticker"] = local["ticker"].map(_ticker)
    fields = sorted({field for values in context.values() for field in values if str(field).startswith("ownership_ksei_")})
    for field in fields:
        incoming = local["ticker"].map(lambda ticker: context.get(str(ticker), {}).get(field))
        if field not in local.columns:
            local[field] = incoming
            continue
        missing = local[field].isna() | local[field].astype(str).str.strip().eq("")
        local.loc[missing, field] = incoming.loc[missing]
    return local


def install() -> dict[str, str]:
    import simple_focus as focus

    marker = "_pasticuan_ksei_runtime_patch"
    if getattr(focus, marker, "") == PATCH_VERSION:
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}
    original_build = focus.build_simple_focus

    def build_with_ksei_context(*args: Any, **kwargs: Any):
        result = original_build(*args, **kwargs)
        context = _load_ksei_ownership(kwargs.get("universe_tickers") or [])
        if isinstance(result, Mapping) and context:
            result = dict(result)
            for key, value in list(result.items()):
                if isinstance(value, pd.DataFrame) and not value.empty and "ticker" in value.columns:
                    result[key] = _merge_context(value, context)
        return result

    focus.build_simple_focus = build_with_ksei_context
    setattr(focus, marker, PATCH_VERSION)
    try:
        import resumable_app_engine as app_engine
        app_engine.build_simple_focus = build_with_ksei_context
    except Exception:
        pass
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "ownership": "OFFICIAL_KSEI_COMPOSITION_CONTEXT",
        "free_float": "NOT_INFERRED",
        "authorization": "UNCHANGED",
    }


__all__ = ["PATCH_VERSION", "install", "_load_ksei_ownership", "_merge_context"]

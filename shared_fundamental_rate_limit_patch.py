from __future__ import annotations

"""Rate-limit guard for Phase 5.6 structured ZAPI fundamental collection."""

from datetime import datetime, timezone
import os
import threading
import time
from typing import Any, Mapping

import requests

from shared_evidence_hub import MissingReason, normalize_failure_reason


PATCH_VERSION = "1.0.1-phase5.6-zapi-rate-limit"
_DEFAULT_MIN_INTERVAL_SECONDS = 0.90
_DEFAULT_MINUTE_COOLDOWN_SECONDS = 61.0
_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0
_COOLDOWN_UNTIL = 0.0
_MONTH_BLOCKED_UNTIL = 0.0


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _next_month_epoch() -> float:
    now = datetime.now(timezone.utc)
    if now.month == 12:
        next_month = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_month = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
    return next_month.timestamp()


def _header_int(response: Any, name: str) -> int | None:
    try:
        raw = str(response.headers.get(name, "")).strip()
        return int(raw) if raw else None
    except (TypeError, ValueError, AttributeError):
        return None


def _header_float(response: Any, name: str) -> float | None:
    try:
        raw = str(response.headers.get(name, "")).strip()
        return float(raw) if raw else None
    except (TypeError, ValueError, AttributeError):
        return None


def _rate_limit_detail(response: Any) -> dict[str, Any]:
    payload: Mapping[str, Any] = {}
    try:
        candidate = response.json()
        if isinstance(candidate, Mapping):
            payload = candidate
    except Exception:
        payload = {}
    remaining_minute = _header_int(response, "X-RateLimit-Remaining-Minute")
    remaining_month = _header_int(response, "X-RateLimit-Remaining-Month")
    window = str(payload.get("window") or "").strip().lower()
    if window not in {"minute", "month"}:
        if remaining_month == 0:
            window = "month"
        elif remaining_minute == 0:
            window = "minute"
        else:
            window = "minute"
    return {
        "window": window,
        "remaining_minute": remaining_minute,
        "remaining_month": remaining_month,
        "retry_after": _header_float(response, "Retry-After"),
        "limit": _header_int(response, "X-RateLimit-Limit"),
        "plan_expired": str(getattr(response, "headers", {}).get("X-Plan-Expired", "") or "").strip(),
    }


def _rate_state(exc: Exception) -> str:
    text = str(exc or "").strip()
    if text.startswith("ZAPI_RATE_LIMIT_"):
        return text.split(":", 1)[0]
    return normalize_failure_reason(exc)


def _patched_zapi_get(self: Any, url: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
    global _LAST_REQUEST_AT, _COOLDOWN_UNTIL, _MONTH_BLOCKED_UNTIL
    if not getattr(self, "api_key", ""):
        raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)

    with _LOCK:
        now_mono = time.monotonic()
        now_epoch = time.time()
        if _MONTH_BLOCKED_UNTIL > now_epoch:
            setattr(self, "_last_zapi_rate_limit", {
                "window": "month",
                "blocked_until_utc": datetime.fromtimestamp(_MONTH_BLOCKED_UNTIL, tz=timezone.utc).isoformat(),
            })
            raise RuntimeError("ZAPI_RATE_LIMIT_MONTH")
        if _MONTH_BLOCKED_UNTIL and _MONTH_BLOCKED_UNTIL <= now_epoch:
            _MONTH_BLOCKED_UNTIL = 0.0
        if _COOLDOWN_UNTIL > now_mono:
            setattr(self, "_last_zapi_rate_limit", {
                "window": "minute",
                "cooldown_remaining_seconds": round(_COOLDOWN_UNTIL - now_mono, 3),
            })
            raise RuntimeError("ZAPI_RATE_LIMIT_MINUTE_COOLDOWN")

        interval = _float_env("ZAPI_MIN_REQUEST_INTERVAL_SECONDS", _DEFAULT_MIN_INTERVAL_SECONDS, 0.25, 10.0)
        delay = interval - (now_mono - _LAST_REQUEST_AT)
        if delay > 0:
            time.sleep(delay)

        try:
            response = self.session.get(
                url,
                params=dict(params),
                headers={"x-api-key": self.api_key, "Accept": "application/json"},
                timeout=18,
            )
        except requests.Timeout as exc:
            _LAST_REQUEST_AT = time.monotonic()
            raise RuntimeError(MissingReason.TIMEOUT.value) from exc
        except requests.ConnectionError as exc:
            _LAST_REQUEST_AT = time.monotonic()
            raise RuntimeError(MissingReason.CONNECTION_ERROR.value) from exc

        _LAST_REQUEST_AT = time.monotonic()
        if response.status_code == 429:
            detail = _rate_limit_detail(response)
            setattr(self, "_last_zapi_rate_limit", detail)
            if detail["window"] == "month":
                _MONTH_BLOCKED_UNTIL = _next_month_epoch()
                detail["blocked_until_utc"] = datetime.fromtimestamp(_MONTH_BLOCKED_UNTIL, tz=timezone.utc).isoformat()
                setattr(self, "_last_zapi_rate_limit", detail)
                raise RuntimeError("ZAPI_RATE_LIMIT_MONTH")
            cooldown = detail.get("retry_after")
            if not isinstance(cooldown, (int, float)) or cooldown <= 0:
                cooldown = _float_env("ZAPI_MINUTE_COOLDOWN_SECONDS", _DEFAULT_MINUTE_COOLDOWN_SECONDS, 5.0, 120.0)
            _COOLDOWN_UNTIL = time.monotonic() + float(cooldown)
            raise RuntimeError("ZAPI_RATE_LIMIT_MINUTE")
        if response.status_code in {401, 403, 404}:
            raise RuntimeError(f"HTTP_{response.status_code}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        return payload


def _patched_refresh_structured(self: Any, ticker: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from shared_fundamental_runtime import bare_ticker

    attempts: list[dict[str, Any]] = []
    for provider, function in (("PLUANG", self.refresh_pluang), ("YAHOO_ZAPI", self.refresh_yahoo)):
        try:
            rows, meta = function(ticker)
            attempts.append({"provider": provider, "state": meta.get("state"), "rows": len(rows)})
            if rows:
                return rows, {**meta, "attempts": attempts}
        except Exception as exc:
            state = _rate_state(exc)
            attempts.append({"provider": provider, "state": state, "detail": str(exc)[:160]})
            if state.startswith("ZAPI_RATE_LIMIT_"):
                return [], {
                    "state": state,
                    "ticker": bare_ticker(ticker),
                    "rows": 0,
                    "attempts": attempts,
                    "rate_limit": dict(getattr(self, "_last_zapi_rate_limit", {}) or {}),
                }
    return [], {"state": "STRUCTURED_PROVIDERS_EXHAUSTED", "ticker": bare_ticker(ticker), "rows": 0, "attempts": attempts}


def install() -> None:
    from shared_fundamental_runtime import SharedFundamentalRuntime
    if getattr(SharedFundamentalRuntime, "_phase56_rate_limit_patch", "") == PATCH_VERSION:
        return
    SharedFundamentalRuntime._zapi_get = _patched_zapi_get
    SharedFundamentalRuntime.refresh_structured = _patched_refresh_structured
    SharedFundamentalRuntime._phase56_rate_limit_patch = PATCH_VERSION


def _reset_for_tests() -> None:
    global _LAST_REQUEST_AT, _COOLDOWN_UNTIL, _MONTH_BLOCKED_UNTIL
    with _LOCK:
        _LAST_REQUEST_AT = 0.0
        _COOLDOWN_UNTIL = 0.0
        _MONTH_BLOCKED_UNTIL = 0.0


__all__ = ["PATCH_VERSION", "install", "_rate_limit_detail", "_reset_for_tests"]

from __future__ import annotations

"""Truthful discovery diagnostics for official IDX Trade Detail Publik files."""

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import urljoin

import requests


PUBLIC_INDEX_URL = (
    "https://www.idxdata3.co.id/INET_Specification/Market_Summary/Market_Indices/"
    "IX200720.TXT?directory=.%2FIDX+Reporting+PSPP%2FRevitalisasi%2FPUBLIK%2F"
)
ALLOWED_CONTENT_TYPES = ("text/csv", "text/plain", "octet-stream", "application/vnd.ms-excel")


@dataclass(frozen=True)
class DiscoveryAttempt:
    requested_trade_date: str
    url: str
    discovery_method: str
    result_state: str
    http_status: int | None = None
    content_type: str = ""
    content_length: int | None = None
    error_type: str = ""

    def safe_dict(self) -> dict[str, Any]:
        return asdict(self)


class TradeDetailDiscoveryError(RuntimeError):
    def __init__(self, filename: str, attempts: list[DiscoveryAttempt]):
        self.filename = filename
        self.attempts = tuple(attempts)
        state = attempts[-1].result_state if attempts else "NO_FILE"
        super().__init__(f"IDX_TRADE_DETAIL_{state}:{filename}")


def _headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; Shared-IDX-Evidence-Hub/1.0)",
        "Accept": "text/csv,text/plain,application/octet-stream,*/*",
    }


def _exception_state(exc: Exception) -> str:
    if isinstance(exc, requests.Timeout):
        return "TIMEOUT"
    if isinstance(exc, requests.ConnectionError):
        return "CONNECTION_ERROR"
    return type(exc).__name__.upper()


def _response_state(response: Any) -> str:
    status = int(getattr(response, "status_code", 0) or 0)
    if status in {401, 403, 404, 429}:
        return f"HTTP_{status}"
    if not 200 <= status < 300:
        return f"HTTP_{status}" if status else "CONNECTION_ERROR"
    content_type = str(getattr(response, "headers", {}).get("content-type", "")).lower()
    if not any(token in content_type for token in ALLOWED_CONTENT_TYPES):
        return "INVALID_CONTENT_TYPE"
    raw_length = getattr(response, "headers", {}).get("content-length")
    try:
        content_length = int(raw_length) if raw_length is not None else None
    except (TypeError, ValueError):
        content_length = None
    if content_length == 0:
        return "EMPTY_RESPONSE"
    return "FOUND"


def _attempt(trade_date: date, url: str, method: str, response: Any) -> DiscoveryAttempt:
    raw_length = getattr(response, "headers", {}).get("content-length")
    try:
        length = int(raw_length) if raw_length is not None else None
    except (TypeError, ValueError):
        length = None
    return DiscoveryAttempt(
        requested_trade_date=trade_date.isoformat(),
        url=url,
        discovery_method=method,
        result_state=_response_state(response),
        http_status=int(getattr(response, "status_code", 0) or 0) or None,
        content_type=str(getattr(response, "headers", {}).get("content-type", ""))[:160],
        content_length=length,
    )


def discover_trade_detail_url(
    trade_date: date,
    *,
    timeout: int = 20,
    diagnostics: list[DiscoveryAttempt] | None = None,
    request_get: Any = requests.get,
) -> str:
    attempts = diagnostics if diagnostics is not None else []
    filename = f"Trade-Detail-Publik_{trade_date:%Y%m%d}.csv"
    try:
        response = request_get(PUBLIC_INDEX_URL, headers=_headers(), timeout=timeout)
        index_attempt = _attempt(trade_date, PUBLIC_INDEX_URL, "PUBLIC_DIRECTORY_INDEX", response)
        attempts.append(index_attempt)
        if 200 <= int(getattr(response, "status_code", 0) or 0) < 300:
            matches = re.findall(
                r"href=[\"']([^\"']*" + re.escape(filename) + r")[\"']",
                str(getattr(response, "text", "")),
                flags=re.I,
            )
            for href in matches:
                candidate = urljoin(str(getattr(response, "url", PUBLIC_INDEX_URL)), href)
                if filename.lower() in candidate.lower():
                    attempts.append(DiscoveryAttempt(
                        trade_date.isoformat(), candidate, "PUBLIC_DIRECTORY_LINK", "FOUND",
                    ))
                    return candidate
    except Exception as exc:
        attempts.append(DiscoveryAttempt(
            trade_date.isoformat(), PUBLIC_INDEX_URL, "PUBLIC_DIRECTORY_INDEX",
            _exception_state(exc), error_type=type(exc).__name__,
        ))

    candidates = (
        f"https://www.idxdata3.co.id/IDX%20Reporting%20PSPP/Revitalisasi/PUBLIK/{filename}",
        f"https://idxdata3.co.id/IDX%20Reporting%20PSPP/Revitalisasi/PUBLIK/{filename}",
        f"https://www.idxdata3.co.id/Market_Summary/Market_Summary/{filename}",
    )
    for candidate in candidates:
        response = None
        try:
            response = request_get(candidate, headers=_headers(), timeout=timeout, stream=True)
            candidate_attempt = _attempt(trade_date, candidate, "DOCUMENTED_PATH_PROBE", response)
            attempts.append(candidate_attempt)
            if candidate_attempt.result_state == "FOUND":
                return candidate
        except Exception as exc:
            attempts.append(DiscoveryAttempt(
                trade_date.isoformat(), candidate, "DOCUMENTED_PATH_PROBE",
                _exception_state(exc), error_type=type(exc).__name__,
            ))
        finally:
            if response is not None:
                response.close()
    raise TradeDetailDiscoveryError(filename, attempts)


def download_trade_detail(
    trade_date: date,
    *,
    timeout: int = 45,
    diagnostics: list[DiscoveryAttempt] | None = None,
    request_get: Any = requests.get,
) -> tuple[Path, str]:
    attempts = diagnostics if diagnostics is not None else []
    url = discover_trade_detail_url(
        trade_date,
        timeout=timeout,
        diagnostics=attempts,
        request_get=request_get,
    )
    response = request_get(url, headers=_headers(), timeout=timeout, stream=True)
    download_attempt = _attempt(trade_date, url, "DOWNLOAD", response)
    attempts.append(download_attempt)
    if download_attempt.result_state != "FOUND":
        response.close()
        raise TradeDetailDiscoveryError(Path(url).name, attempts)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    path = Path(handle.name)
    handle.close()
    size = 0
    try:
        with path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
                    size += len(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        response.close()
    if size <= 0:
        path.unlink(missing_ok=True)
        attempts.append(DiscoveryAttempt(
            trade_date.isoformat(), url, "DOWNLOAD_BODY", "EMPTY_RESPONSE",
            http_status=download_attempt.http_status,
            content_type=download_attempt.content_type,
            content_length=0,
        ))
        raise TradeDetailDiscoveryError(Path(url).name, attempts)
    return path, url


__all__ = [
    "DiscoveryAttempt",
    "PUBLIC_INDEX_URL",
    "TradeDetailDiscoveryError",
    "discover_trade_detail_url",
    "download_trade_detail",
]

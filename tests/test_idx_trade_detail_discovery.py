from __future__ import annotations

from datetime import date

import pytest
import requests

import idx_trade_detail_discovery as discovery


DAY = date(2026, 8, 31)


class Response:
    def __init__(self, status: int, *, content_type: str = "text/csv", length: int | None = 10, text: str = "", url: str = "https://idx.example/index"):
        self.status_code = status
        self.headers = {"content-type": content_type}
        if length is not None:
            self.headers["content-length"] = str(length)
        self.text = text
        self.url = url
        self.content = b"x" if length else b""

    def close(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        if self.content:
            yield self.content


@pytest.mark.parametrize("status,reason", [(401, "HTTP_401"), (403, "HTTP_403"), (404, "HTTP_404"), (429, "HTTP_429")])
def test_http_reason_is_preserved_for_every_candidate(status: int, reason: str) -> None:
    attempts: list[discovery.DiscoveryAttempt] = []

    def get(*args, **kwargs):
        return Response(status)

    with pytest.raises(discovery.TradeDetailDiscoveryError) as caught:
        discovery.discover_trade_detail_url(DAY, diagnostics=attempts, request_get=get)
    assert attempts
    assert all(attempt.result_state == reason for attempt in attempts)
    assert caught.value.attempts == tuple(attempts)


@pytest.mark.parametrize(
    "exc,reason",
    [(requests.Timeout(), "TIMEOUT"), (requests.ConnectionError(), "CONNECTION_ERROR")],
)
def test_transport_reason_is_preserved(exc: Exception, reason: str) -> None:
    attempts: list[discovery.DiscoveryAttempt] = []

    def get(*args, **kwargs):
        raise exc

    with pytest.raises(discovery.TradeDetailDiscoveryError):
        discovery.discover_trade_detail_url(DAY, diagnostics=attempts, request_get=get)
    assert attempts and all(attempt.result_state == reason for attempt in attempts)


def test_invalid_content_type_is_not_accepted_as_csv() -> None:
    attempts: list[discovery.DiscoveryAttempt] = []

    with pytest.raises(discovery.TradeDetailDiscoveryError):
        discovery.discover_trade_detail_url(
            DAY,
            diagnostics=attempts,
            request_get=lambda *args, **kwargs: Response(200, content_type="text/html"),
        )
    assert any(attempt.result_state == "INVALID_CONTENT_TYPE" for attempt in attempts)


def test_zero_content_length_is_empty_response() -> None:
    attempts: list[discovery.DiscoveryAttempt] = []

    with pytest.raises(discovery.TradeDetailDiscoveryError):
        discovery.discover_trade_detail_url(
            DAY,
            diagnostics=attempts,
            request_get=lambda *args, **kwargs: Response(200, length=0),
        )
    assert any(attempt.result_state == "EMPTY_RESPONSE" for attempt in attempts)


def test_directory_discovery_returns_exact_official_link() -> None:
    filename = "Trade-Detail-Publik_20260831.csv"
    html = f'<a href="/official/{filename}">download</a>'
    attempts: list[discovery.DiscoveryAttempt] = []
    found = discovery.discover_trade_detail_url(
        DAY,
        diagnostics=attempts,
        request_get=lambda *args, **kwargs: Response(200, text=html, url="https://www.idxdata3.co.id/index/"),
    )
    assert found == f"https://www.idxdata3.co.id/official/{filename}"
    assert attempts[-1].discovery_method == "PUBLIC_DIRECTORY_LINK"
    assert attempts[-1].result_state == "FOUND"


def test_diagnostics_never_contain_request_headers_or_credentials() -> None:
    attempt = discovery.DiscoveryAttempt(DAY.isoformat(), "https://idx.example/file.csv", "TEST", "HTTP_403")
    payload = attempt.safe_dict()
    assert "headers" not in payload
    assert "authorization" not in str(payload).lower()
    assert "api-key" not in str(payload).lower()

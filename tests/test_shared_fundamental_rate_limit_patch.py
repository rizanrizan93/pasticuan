from __future__ import annotations

from types import SimpleNamespace

from shared_fundamental_rate_limit_patch import _rate_limit_detail, _reset_for_tests, install
from shared_fundamental_runtime import SharedFundamentalRuntime


class FakeResponse:
    def __init__(self, *, status_code: int, payload: dict, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> dict:
        return dict(self._payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP_{self.status_code}")


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def get(self, *_args, **_kwargs) -> FakeResponse:
        self.calls += 1
        if not self.responses:
            raise AssertionError("unexpected extra ZAPI request")
        return self.responses.pop(0)


def _runtime(session: FakeSession) -> SharedFundamentalRuntime:
    return SharedFundamentalRuntime(
        "TEST",
        config=SimpleNamespace(ready=True),
        backend=object(),
        api_key="zpi_test",
        session=session,
    )


def test_month_rate_limit_stops_cross_provider_fallback() -> None:
    _reset_for_tests()
    install()
    session = FakeSession([
        FakeResponse(
            status_code=429,
            payload={"window": "month"},
            headers={"X-RateLimit-Remaining-Month": "0", "X-RateLimit-Remaining-Minute": "99"},
        )
    ])
    rows, meta = _runtime(session).refresh_structured("BBCA")
    assert rows == []
    assert meta["state"] == "ZAPI_RATE_LIMIT_MONTH"
    assert len(meta["attempts"]) == 1
    assert meta["attempts"][0]["provider"] == "PLUANG"
    assert session.calls == 1


def test_minute_rate_limit_stops_cross_provider_fallback() -> None:
    _reset_for_tests()
    install()
    session = FakeSession([
        FakeResponse(
            status_code=429,
            payload={"window": "minute"},
            headers={"X-RateLimit-Remaining-Minute": "0", "X-RateLimit-Remaining-Month": "321"},
        )
    ])
    rows, meta = _runtime(session).refresh_structured("AALI")
    assert rows == []
    assert meta["state"] == "ZAPI_RATE_LIMIT_MINUTE"
    assert meta["rate_limit"]["remaining_month"] == 321
    assert session.calls == 1


def test_rate_limit_detail_uses_headers_when_window_missing() -> None:
    response = FakeResponse(
        status_code=429,
        payload={"error": "too many requests"},
        headers={"X-RateLimit-Remaining-Month": "0", "X-RateLimit-Remaining-Minute": "50"},
    )
    detail = _rate_limit_detail(response)
    assert detail["window"] == "month"
    assert detail["remaining_month"] == 0

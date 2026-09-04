from __future__ import annotations

from typing import Any

from shared_public_fundamental_projection import refresh_public_fundamental_projection


class FakeBackend:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.writes: list[dict[str, Any]] = []

    def read_rows(self, table: str, filters: dict[str, Any], *, select: str = "*", limit: int = 10000) -> list[dict[str, Any]]:
        assert table == "evidence_fundamental_metrics"
        assert filters == {}
        assert "metric_name" in select
        return list(self.rows)

    def upsert_rows(self, table: str, rows: list[dict[str, Any]], *, conflict: tuple[str, ...]) -> list[dict[str, Any]]:
        assert table == "phase56_public_fundamental_snapshots"
        assert conflict == ("ticker",)
        self.writes.extend(dict(row) for row in rows)
        return [dict(row) for row in rows]


def row(*, provider: str, ticker: str, metric: str, value: float, period: str | None, observed: str, official: bool) -> dict[str, Any]:
    return {
        "provider": provider,
        "ticker": ticker,
        "period_end": period,
        "statement_date": period,
        "metric_name": metric,
        "metric_value": value,
        "metric_unit": "PERCENT" if metric.endswith("_pct") else "CURRENCY_NATIVE",
        "source_families": "IDX_OFFICIAL_XBRL" if official else "YAHOO_DIRECT_YFINANCE",
        "official_verified": official,
        "source_record_hash": f"{provider}:{ticker}:{metric}:{period}:{observed}",
        "lineage_state": "TEST",
        "observed_at": observed,
        "validation_state": "VALID",
        "fetched_at": observed,
    }


def test_projection_is_one_row_per_ticker_and_preserves_official_separation() -> None:
    rows = [
        row(provider="YAHOO", ticker="BBCA", metric="roe_pct", value=18.2, period=None, observed="2026-09-03T10:00:00+00:00", official=False),
        row(provider="IDX", ticker="BBCA", metric="revenue", value=1000, period="2026-06-30", observed="2026-08-01T10:00:00+00:00", official=True),
        row(provider="IDX", ticker="BBCA", metric="net_income", value=250, period="2026-06-30", observed="2026-08-01T10:00:00+00:00", official=True),
        row(provider="IDX", ticker="BBCA", metric="revenue", value=850, period="2025-06-30", observed="2025-08-01T10:00:00+00:00", official=True),
        row(provider="YAHOO", ticker="AALI", metric="revenue_growth_pct", value=4.5, period="2026-03-31", observed="2026-09-03T11:00:00+00:00", official=False),
    ]
    backend = FakeBackend(rows)

    meta = refresh_public_fundamental_projection(backend, batch_size=1)

    assert meta["state"] == "REFRESHED"
    assert meta["projection_rows"] == 2
    assert meta["persisted_rows"] == 2
    assert len(backend.writes) == 2

    bbca = next(item for item in backend.writes if item["ticker"] == "BBCA")
    assert bbca["official_period_end"] == "2026-06-30"
    # Canonicalizer parity: proxy period falls back to the official period when
    # a non-official metric has no usable period anchor.
    assert bbca["proxy_period_end"] == "2026-06-30"
    assert bbca["proxy_metrics"]["roe_pct"] == 18.2
    assert bbca["official_metrics"]["revenue"] == 1000.0
    assert bbca["official_metrics"]["net_income"] == 250.0
    assert bbca["official_metrics"]["net_margin_pct"] == 25.0
    assert bbca["proxy_metrics"]["revenue"] == 1000.0
    assert bbca["source_state"] == "PHASE5_6_PUBLIC_FACTUAL_PROJECTION"


def test_projection_contains_no_scanner_decision_fields() -> None:
    backend = FakeBackend([
        row(provider="YAHOO", ticker="ANTM", metric="roe_pct", value=26.8, period="2026-03-31", observed="2026-09-03T12:00:00+00:00", official=False),
    ])

    meta = refresh_public_fundamental_projection(backend)
    assert meta["state"] == "REFRESHED"
    assert len(backend.writes) == 1
    payload = backend.writes[0]
    forbidden = {
        "score", "rank", "recommendation", "gate", "entry", "stop", "stop_loss",
        "target", "tp1", "tp2", "future_fundamental", "future_fundamental_score",
    }
    assert forbidden.isdisjoint(payload)
    assert forbidden.isdisjoint(payload["proxy_metrics"])
    assert forbidden.isdisjoint(payload["official_metrics"])

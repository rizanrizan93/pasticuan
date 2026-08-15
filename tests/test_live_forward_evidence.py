from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import live_forward_evidence as lfe
import runtime_integrity_patch as rip


class _Response:
    def __init__(self, xml: str):
        self.content = xml.encode("utf-8")
    def raise_for_status(self) -> None:
        return None


def _rss(title: str) -> str:
    return f"""<?xml version='1.0' encoding='UTF-8'?>
    <rss><channel><item><title>{title}</title><link>https://example.com/a</link>
    <pubDate>Fri, 14 Aug 2026 08:00:00 GMT</pubDate><source>Example Publisher</source></item></channel></rss>"""


def test_live_forward_material_event_is_research_only(monkeypatch):
    monkeypatch.setattr(lfe.requests, "get", lambda *a, **k: _Response(_rss("ABCD raih kontrak baru dan backlog bertambah")))
    frame = lfe.collect_live_forward_evidence(["ABCD.JK"], max_workers=1)
    row = frame.iloc[0]
    assert row["forward_collection_state"] == "MATERIAL_FORWARD_RESEARCH_EVIDENCE_FOUND"
    assert row["project_pipeline_score_observed"] > 0
    assert bool(row["forward_research_only"]) is True
    assert bool(row["project_source_quorum_verified"]) is False
    assert float(row["forward_collection_coverage_pct"]) == 100.0


def test_live_forward_completed_check_without_event_does_not_invent_score(monkeypatch):
    monkeypatch.setattr(lfe.requests, "get", lambda *a, **k: _Response(_rss("ABCD menggelar kegiatan sosial")))
    frame = lfe.collect_live_forward_evidence(["ABCD.JK"], max_workers=1)
    row = frame.iloc[0]
    assert row["forward_collection_state"] == "FORWARD_CHECK_COMPLETED_NO_MATERIAL_EVENT"
    assert float(row["forward_collection_coverage_pct"]) == 100.0
    assert "project_pipeline_score_observed" not in frame.columns or pd.isna(row.get("project_pipeline_score_observed"))


def test_live_forward_reader_collects_missing_and_persists(monkeypatch):
    calls = {"persist": 0}

    class FakeBridge:
        def __init__(self):
            self.settings = SimpleNamespace(mode="SUPABASE_REST")
        def read_forward_quality_cache(self, tickers):
            return pd.DataFrame(), pd.DataFrame()
        def _get_rows(self, table, params):
            return []
        def persist_scan_result(self, result, tables=None):
            calls["persist"] += 1
            assert tables == ("forward_quality_cache",)
            assert len(result["project_management_review"]) == 2
            return pd.DataFrame([{"state": "WRITE_ALL"}])

    monkeypatch.setattr(rip, "collect_live_forward_evidence", lambda tickers, **kwargs: pd.DataFrame([
        {"ticker": "AAA.JK", "project_name": "FORWARD_EVIDENCE_CHECK", "project_stage": "NO_MATERIAL_FORWARD_EVENT_FOUND", "forward_collection_state": "FORWARD_CHECK_COMPLETED_NO_MATERIAL_EVENT", "forward_collection_coverage_pct": 100.0, "project_data_coverage": 0.0, "last_verified_at": pd.Timestamp.now(tz="UTC").isoformat()},
        {"ticker": "BBB.JK", "project_name": "Contract", "project_stage": "PROJECT_OR_CONTRACT", "forward_collection_state": "MATERIAL_FORWARD_RESEARCH_EVIDENCE_FOUND", "forward_collection_coverage_pct": 100.0, "project_data_coverage": 48.0, "project_pipeline_score_observed": 68.0, "last_verified_at": pd.Timestamp.now(tz="UTC").isoformat()},
    ]))
    rip._GOVERNED_EVIDENCE_CACHE.clear()
    rip._wrap_forward_quality_reader(FakeBridge)
    frame, audit = FakeBridge().read_forward_quality_cache(["AAA.JK", "BBB.JK"])
    assert calls["persist"] == 1
    assert set(frame["ticker"]) == {"AAA.JK", "BBB.JK"}
    summary = audit.loc[audit.get("provider", pd.Series(dtype=str)).eq("LIVE_FORWARD_EVIDENCE")].iloc[-1]
    assert float(summary["collection_coverage_pct"]) == 100.0


def test_smart_money_cost_blocks_are_one_per_card():
    import v9_dashboard

    lfe.install_dashboard_cost_integrity()
    top = pd.DataFrame([
        {"dashboard_rank": 1, "ticker": "AAA.JK", "last_price": 100.0, "research_accumulation_zone_low": 90.0, "research_accumulation_zone_high": 94.0, "silent_accumulation_score": 70.0},
        {"dashboard_rank": 2, "ticker": "BBB.JK", "last_price": 200.0, "research_accumulation_zone_low": 180.0, "research_accumulation_zone_high": 188.0, "silent_accumulation_score": 65.0},
        {"dashboard_rank": 3, "ticker": "CCC.JK", "last_price": 300.0, "research_accumulation_zone_low": 270.0, "research_accumulation_zone_high": 282.0, "silent_accumulation_score": 60.0},
    ])
    html = v9_dashboard.render_dashboard_html(top, model="NEXT_LEADER")
    assert html.count('class="v9-cost-basis"') == 3
    first_ticker = html.find("AAA.JK")
    first_cost = html.find('class="v9-cost-basis"')
    second_ticker = html.find("BBB.JK", first_ticker + 1)
    second_cost = html.find('class="v9-cost-basis"', first_cost + 1)
    third_ticker = html.find("CCC.JK", second_ticker + 1)
    third_cost = html.find('class="v9-cost-basis"', second_cost + 1)
    assert -1 not in (first_ticker, first_cost, second_ticker, second_cost, third_ticker, third_cost)
    assert first_ticker < first_cost < second_ticker < second_cost < third_ticker < third_cost

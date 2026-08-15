from __future__ import annotations

import live_forward_evidence as lfe


class _Response:
    def __init__(self, xml: str):
        self.content = xml.encode('utf-8')
    def raise_for_status(self):
        return None


def test_company_name_can_resolve_event_without_ticker_in_title(monkeypatch):
    xml = """<rss><channel><item>
    <title>Jayamas Medica ekspansi kapasitas dan luncurkan produk baru</title>
    <link>https://example.com/onemed</link>
    <pubDate>Fri, 14 Aug 2026 08:00:00 GMT</pubDate><source>Example</source>
    </item></channel></rss>"""
    monkeypatch.setattr(lfe.requests, 'get', lambda *a, **k: _Response(xml))
    frame = lfe.collect_live_forward_evidence(
        ['OMED.JK'], company_names={'OMED.JK': 'JAYAMAS MEDICA INDUSTRI Tbk, PT'}, max_workers=1,
    )
    assert frame.iloc[0]['forward_collection_state'] == 'MATERIAL_FORWARD_RESEARCH_EVIDENCE_FOUND'
    assert frame.iloc[0]['entity_match_method'].startswith('COMPANY_TOKEN_MATCH_')
    assert bool(frame.iloc[0]['project_source_quorum_verified']) is False

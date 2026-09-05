from __future__ import annotations

from types import SimpleNamespace

import shared_ownership_runtime_transport_patch as patch


class FakeBackend:
    def __init__(self, public_rows=None, ksei_rows=None):
        self.public_rows = list(public_rows or [])
        self.ksei_rows = list(ksei_rows or [])
        self.calls = []

    def read_rows(self, table, filters, limit=50000):
        self.calls.append((table, dict(filters), limit))
        if table == patch.PUBLIC_TABLE:
            return list(self.public_rows)
        if table == patch.KSEI_TABLE:
            return list(self.ksei_rows)
        raise AssertionError(table)


def _reset_cache() -> None:
    patch._PUBLIC_CACHE = {}
    patch._PUBLIC_CACHE_AT = 0.0
    patch._KSEI_CACHE = {}
    patch._KSEI_CACHE_AT = 0.0


def test_public_ownership_loader_uses_shared_projection_without_promoting_official(monkeypatch) -> None:
    _reset_cache()
    backend = FakeBackend(public_rows=[{
        "ticker": "AALI",
        "insiders_held_pct": 79.7,
        "institutions_held_pct": 12.3,
        "institutions_float_held_pct": 60.1,
        "institutions_count": 44,
        "coverage_pct": 100.0,
        "source_period": "2026-08-31",
        "observed_on": "2026-09-01",
        "source_authority": "PUBLIC_PROVIDER",
        "official_verified": False,
        "provenance_state": "PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI",
    }])
    monkeypatch.setattr(patch, "_backend", lambda: backend)

    result = patch._shared_public_ownership(["AALI.JK"])
    row = result["AALI.JK"]

    assert row["ownership_public_context_coverage_pct"] == 100.0
    assert row["ownership_public_official_verified"] is False
    assert row["ownership_public_source_authority"] == "PUBLIC_PROVIDER"
    assert "NOT_IDX_KSEI" in row["ownership_public_context_provenance_state"]
    assert backend.calls[0][0] == patch.PUBLIC_TABLE


def test_ksei_loader_uses_shared_official_registration_composition(monkeypatch) -> None:
    _reset_cache()
    common = {
        "source_file_hash": "hash",
        "category": patch.CATEGORY,
        "ticker": "AALI",
        "report_date": "2026-08-31",
        "publication_date": "2026-08-31",
        "source_url": "https://web.ksei.co.id/Download/BalanceposEfek20260831.zip",
        "source_verified": True,
        "validation_state": "VALID",
    }
    backend = FakeBackend(ksei_rows=[
        {**common, "holder_classification": "KSEI_SECURITY_NUMBER", "shares_held": 1000, "ownership_percentage": None},
        {**common, "holder_classification": "KSEI_SCRIPLESS_TOTAL", "shares_held": 800, "ownership_percentage": 80.0},
        {**common, "holder_classification": "KSEI_LOCAL_TOTAL", "shares_held": 600, "ownership_percentage": 75.0},
        {**common, "holder_classification": "KSEI_FOREIGN_TOTAL", "shares_held": 200, "ownership_percentage": 25.0},
    ])
    monkeypatch.setattr(patch, "_backend", lambda: backend)

    result = patch._shared_ksei_ownership(["AALI.JK"])
    row = result["AALI.JK"]

    assert row["ownership_ksei_context_coverage_pct"] == 100.0
    assert row["ownership_ksei_scripless_pct"] == 80.0
    assert row["ownership_ksei_local_pct"] == 75.0
    assert row["ownership_ksei_foreign_pct"] == 25.0
    assert row["ownership_ksei_official_verified"] is True
    assert row["ownership_ksei_source_authority"] == "OFFICIAL_KSEI"
    assert "NOT_REGULATORY_FREE_FLOAT" in row["ownership_ksei_provenance_state"]
    assert row["ownership_ksei_source_url"].endswith("BalanceposEfek20260831.zip")
    assert backend.calls[0][0] == patch.KSEI_TABLE


def test_install_rebinds_existing_runtime_loader_globals(monkeypatch) -> None:
    import phase56_coverage_runtime_patch as public_context
    import pasticuan_ksei_runtime_patch as ksei_context

    monkeypatch.setattr(public_context, "_load_public_ownership", lambda *_: {"old": {}})
    monkeypatch.setattr(ksei_context, "_load_ksei_ownership", lambda *_: {"old": {}})

    state = patch.install()

    assert state["state"] == "INSTALLED"
    assert public_context._load_public_ownership is patch._shared_public_ownership
    assert ksei_context._load_ksei_ownership is patch._shared_ksei_ownership
    assert "NOT_OFFICIAL" in state["public_ownership"]
    assert "NOT_REGULATORY_FREE_FLOAT" in state["ksei"]


def test_runtime_release_installs_transport_before_ownership_telemetry() -> None:
    source = open("runtime_release.py", encoding="utf-8").read()
    transport = source.index('_try_optional_patch("shared_ownership_runtime_transport_patch", "install")')
    public_telemetry = source.index('_try_optional_patch("pasticuan_ownership_calibration_telemetry_patch", "install")')
    ksei_telemetry = source.index('_try_optional_patch("pasticuan_ksei_calibration_telemetry_patch", "install")')
    assert transport < public_telemetry < ksei_telemetry

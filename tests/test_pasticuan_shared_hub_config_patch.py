from __future__ import annotations

from dataclasses import dataclass

from pasticuan_shared_hub_config_patch import apply_settings


@dataclass
class Settings:
    supabase_url: str
    supabase_key: str = "backend-key"
    supabase_key_type: str = "SECRET"
    mode: str = "SUPABASE_REST"


def _clear(monkeypatch) -> None:
    for name in (
        "SHARED_EVIDENCE_SUPABASE_URL",
        "SHARED_EVIDENCE_SUPABASE_SECRET_KEY",
        "SHARED_EVIDENCE_SUPABASE_SERVICE_ROLE_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_aliases_only_expected_pasticuan_project(monkeypatch) -> None:
    _clear(monkeypatch)
    state = apply_settings(Settings("https://mbtsvflwszcgdtijdgas.supabase.co"))
    assert state == "PASTICUAN_OPERATIONAL_CONFIG_ALIASED"
    assert __import__("os").environ["SHARED_EVIDENCE_SUPABASE_URL"].endswith("mbtsvflwszcgdtijdgas.supabase.co")
    assert __import__("os").environ["SHARED_EVIDENCE_SUPABASE_SECRET_KEY"] == "backend-key"


def test_rejects_cross_project_and_unsafe_keys(monkeypatch) -> None:
    _clear(monkeypatch)
    assert apply_settings(Settings("https://vbtpwpmkfxzqeuvztcmz.supabase.co")) == "PROJECT_REF_REJECTED"
    assert "SHARED_EVIDENCE_SUPABASE_URL" not in __import__("os").environ
    assert apply_settings(Settings("https://mbtsvflwszcgdtijdgas.supabase.co", supabase_key_type="ANON")) == "BACKEND_CREDENTIALS_UNAVAILABLE"


def test_explicit_shared_config_is_never_overwritten(monkeypatch) -> None:
    monkeypatch.setenv("SHARED_EVIDENCE_SUPABASE_URL", "https://explicit.example")
    state = apply_settings(Settings("https://mbtsvflwszcgdtijdgas.supabase.co"))
    assert state == "EXPLICIT_SHARED_CONFIG_PRESERVED"
    assert __import__("os").environ["SHARED_EVIDENCE_SUPABASE_URL"] == "https://explicit.example"

from pathlib import Path

import pandas as pd

from pasticuan_ownership_calibration_telemetry_patch import ALL_FIELDS, _extract_context


ROOT = Path(__file__).resolve().parents[1]


def test_extract_context_keeps_only_explicit_public_runtime_fields():
    frame = pd.DataFrame(
        [
            {
                "ticker": "bbca",
                "ownership_public_context_coverage_pct": 100.0,
                "ownership_public_institutions_count": 52,
                "ownership_public_source_authority": "PUBLIC_PROVIDER",
                "ownership_public_official_verified": False,
                "ownership_public_context_provenance_state": "PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI",
                "ownership_public_context_state": "CONTEXT_ONLY_NOT_REGULATORY_FREE_FLOAT",
                "regulatory_free_float_pct": 42.0,
                "ownership_score": 99.0,
                "real_money_authorization_pass": True,
            }
        ]
    )

    context = _extract_context(frame)

    assert set(context) == {"BBCA.JK"}
    assert set(context["BBCA.JK"]) <= ALL_FIELDS
    assert context["BBCA.JK"]["ownership_public_context_coverage_pct"] == 100.0
    assert context["BBCA.JK"]["ownership_public_official_verified"] is False
    assert "regulatory_free_float_pct" not in context["BBCA.JK"]
    assert "ownership_score" not in context["BBCA.JK"]
    assert "real_money_authorization_pass" not in context["BBCA.JK"]


def test_v34_migration_is_additive_and_does_not_backfill_history():
    sql = (ROOT / "database" / "migration_v34_ownership_calibration_telemetry.sql").read_text()
    lowered = sql.lower()

    assert "alter table public.multibagger_snapshots" in lowered
    assert "alter table public.pasticuan_calibration_snapshots" in lowered
    assert "ownership_public_context_coverage_pct" in lowered
    assert "security definer" in lowered
    assert "set search_path = ''" in lowered
    assert "from public, anon, authenticated, service_role" in lowered
    assert "update public.pasticuan_calibration_snapshots set" not in lowered
    assert "regulatory_free_float_pct" not in lowered


def test_runtime_release_installs_telemetry_after_phase56_context_patch():
    runtime = (ROOT / "runtime_release.py").read_text()
    phase56 = runtime.index('"phase56_coverage_runtime_patch", "install"')
    telemetry = runtime.index('"pasticuan_ownership_calibration_telemetry_patch", "install"')
    assert telemetry > phase56

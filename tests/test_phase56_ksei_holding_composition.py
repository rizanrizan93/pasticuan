from __future__ import annotations

from datetime import date
from io import BytesIO
from zipfile import ZipFile

import pandas as pd

import pasticuan_ksei_calibration_telemetry_patch as telemetry
import pasticuan_ksei_runtime_patch as runtime
import shared_ksei_holding_composition as ksei


def _fixture_zip() -> bytes:
    header = [
        "Date", "Code", "Type", "Sec. Num", "Price",
        "Local IS", "Local CP", "Local PF", "Local IB", "Local ID", "Local MF", "Local SC", "Local FD", "Local OT", "Total",
        "Foreign IS", "Foreign CP", "Foreign PF", "Foreign IB", "Foreign ID", "Foreign MF", "Foreign SC", "Foreign FD", "Foreign OT", "Total",
    ]
    row = [
        "31-AUG-2026", "TEST", "EQUITY", "1000", "100",
        "50", "300", "20", "20", "100", "50", "30", "10", "20", "600",
        "20", "50", "10", "20", "50", "20", "10", "5", "15", "200",
    ]
    text = "|".join(header) + "\n" + "|".join(row) + "\n"
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("BalanceposEfek20260831.txt", text)
    return buffer.getvalue()


def test_parse_archive_keeps_four_compact_official_facts() -> None:
    file_row, rows = ksei.parse_archive(
        _fixture_zip(),
        source_url="https://web.ksei.co.id/Download/BalanceposEfek20260831.zip",
        observed_on=date(2026, 8, 31),
    )
    assert file_row["category"] == "ksei-komposisi"
    assert file_row["source_verified"] is True
    assert len(rows) == 4
    facts = {row["holder_classification"]: row for row in rows}
    assert float(facts["KSEI_SECURITY_NUMBER"]["shares_held"]) == 1000.0
    assert round(float(facts["KSEI_SCRIPLESS_TOTAL"]["ownership_percentage"]), 2) == 80.0
    assert round(float(facts["KSEI_LOCAL_TOTAL"]["ownership_percentage"]), 2) == 75.0
    assert round(float(facts["KSEI_FOREIGN_TOTAL"]["ownership_percentage"]), 2) == 25.0


def test_canonical_context_is_official_but_not_free_float() -> None:
    _, rows = ksei.parse_archive(
        _fixture_zip(),
        source_url="https://web.ksei.co.id/Download/BalanceposEfek20260831.zip",
        observed_on=date(2026, 8, 31),
    )
    context = ksei.canonical_context(rows)["TEST"]
    assert context["ownership_ksei_context_coverage_pct"] == 100.0
    assert context["ownership_ksei_official_verified"] is True
    assert context["ownership_ksei_source_authority"] == "OFFICIAL_KSEI"
    assert context["ownership_ksei_provenance_state"] == "KSEI_REGISTRATION_COMPOSITION_NOT_REGULATORY_FREE_FLOAT"
    assert "free_float" not in context


def test_runtime_merge_fills_missing_ksei_fields_without_overwriting_explicit() -> None:
    frame = pd.DataFrame([{
        "ticker": "TEST.JK",
        "ownership_ksei_local_pct": None,
        "ownership_ksei_context_state": "EXPLICIT_STATE",
        "final_score": 77.0,
    }])
    context = {"TEST.JK": {
        "ownership_ksei_local_pct": 75.0,
        "ownership_ksei_context_coverage_pct": 100.0,
        "ownership_ksei_context_state": "CONTEXT_ONLY_NOT_BENEFICIAL_OWNERSHIP_OR_FREE_FLOAT",
    }}
    out = runtime._merge_context(frame, context)
    assert float(out.loc[0, "ownership_ksei_local_pct"]) == 75.0
    assert float(out.loc[0, "ownership_ksei_context_coverage_pct"]) == 100.0
    assert out.loc[0, "ownership_ksei_context_state"] == "EXPLICIT_STATE"
    assert float(out.loc[0, "final_score"]) == 77.0


def test_telemetry_extracts_only_explicit_ksei_context() -> None:
    frame = pd.DataFrame([{
        "ticker": "TEST.JK",
        "ownership_ksei_local_pct": 75.0,
        "ownership_ksei_official_verified": True,
        "ownership_ksei_provenance_state": "KSEI_REGISTRATION_COMPOSITION_NOT_REGULATORY_FREE_FLOAT",
        "real_money_authorization_pass": True,
    }])
    context = telemetry._extract_context(frame)["TEST.JK"]
    assert context["ownership_ksei_local_pct"] == 75.0
    assert context["ownership_ksei_official_verified"] is True
    assert "real_money_authorization_pass" not in context


def test_v35_migration_is_additive_no_backfill_and_security_definer_hardened() -> None:
    source = open("database/migration_v35_ksei_ownership_calibration_telemetry.sql", encoding="utf-8").read().lower()
    assert "add column if not exists ownership_ksei_" in source
    assert "security definer" in source
    assert "set search_path = ''" in source
    assert "revoke all on function" in source
    assert "update public.pasticuan_calibration_snapshots set" not in source
    assert "regulatory free float" in source


def test_runtime_release_install_order_keeps_context_before_telemetry() -> None:
    source = open("runtime_release.py", encoding="utf-8").read()
    integrity = source.index('"phase56_runtime_coverage_integrity_patch", "install"')
    ksei_runtime = source.index('"pasticuan_ksei_runtime_patch", "install"')
    public_telemetry = source.index('"pasticuan_ownership_calibration_telemetry_patch", "install"')
    ksei_telemetry = source.index('"pasticuan_ksei_calibration_telemetry_patch", "install"')
    assert integrity < ksei_runtime < public_telemetry < ksei_telemetry

from __future__ import annotations

from pathlib import Path

from phase5_6_adversarial_contract import ADVERSARIAL_WITNESSES, audit_adversarial_witnesses
from shared_ownership_evidence import derive_ownership_changes


def _snapshot(holder: str, report: str, shares: int, pct: float, source_hash: str) -> dict:
    return {
        "source_file_hash": source_hash,
        "category": "lima-persen",
        "ticker": "BBCA",
        "holder_identity_hash": holder,
        "report_date": report,
        "shares_held": shares,
        "ownership_percentage": pct,
        "source_verified": True,
    }


def test_every_master_adversarial_case_has_an_existing_behavioral_witness() -> None:
    result = audit_adversarial_witnesses(Path(__file__).resolve().parent)
    assert result["families"] == 12
    assert result["cases"] == 71
    assert result["fixture_only"] is True and result["status"] == "COMPLETE"


def test_ownership_reduction_threshold_and_no_change_states() -> None:
    previous = [
        _snapshot("reduced", "2026-06-30", 100, 6.0, "old"),
        _snapshot("stable", "2026-06-30", 100, 5.5, "old"),
    ]
    current = [
        _snapshot("reduced", "2026-07-31", 90, 5.2, "new"),
        _snapshot("stable", "2026-07-31", 100, 5.5, "new"),
        _snapshot("threshold", "2026-07-31", 120, 5.1, "new"),
    ]
    states = {row["holder_identity_hash"]: row["change_state"] for row in derive_ownership_changes(previous, current)}
    assert states == {
        "reduced": "REDUCED_REPORTED_HOLDING",
        "threshold": "NEW_5PCT_HOLDER",
    }
    assert "stable" not in states


def test_manifest_has_exact_required_families_and_no_live_witness_file() -> None:
    assert set(ADVERSARIAL_WITNESSES) == {
        "WORKFLOW", "SHARED_CACHE", "ZAPI", "STOCK_SUMMARY", "OWNERSHIP", "FINANCIAL",
        "ANNOUNCEMENTS", "CAPITAL_ACTIONS", "PARTICIPANT", "TEMPORAL", "DB", "SECURITY",
    }
    assert all(
        not witness.test_file.startswith("test_live_")
        for cases in ADVERSARIAL_WITNESSES.values()
        for witness in cases.values()
    )

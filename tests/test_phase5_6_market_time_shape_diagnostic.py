from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_market_time_diagnostic_is_shape_only() -> None:
    source = (ROOT / "tools" / "phase5_6_market_time_shape_diagnostic.py").read_text(encoding="utf-8")
    assert '{"set": "market-time"}' in source
    assert '"keys"' in source
    assert '"item_types"' in source
    assert '"mapping_keys"' in source
    assert "payload.get(\"items\")" in source
    for forbidden in (
        "api_key=",
        "headers=",
        "response.text",
        "repr(payload)",
        "print(payload)",
        "print(response)",
    ):
        assert forbidden not in source


def test_market_time_diagnostic_is_auto_read_only_and_unscheduled() -> None:
    workflow = (ROOT / ".github" / "workflows" / "market-time-shape-diagnostic.yml").read_text(encoding="utf-8").lower()
    assert "push:" in workflow and "branches: [main]" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "schedule:" not in workflow
    assert "shared_evidence_supabase" not in workflow

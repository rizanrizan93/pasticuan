from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_company_reference_gate_is_bulk_bounded_and_factual_only() -> None:
    source = (ROOT / "tools" / "phase5_6_company_reference_live_validation.py").read_text(encoding="utf-8")
    assert "get_directory(observed_on, max_pages=3)" in source
    assert "get_reference(set_name, observed_on)" in source
    assert "get_profile(" not in source
    for set_name in ("sectors", "boards", "market-time"):
        assert set_name in (ROOT / "shared_company_evidence.py").read_text(encoding="utf-8")
    assert "FORBIDDEN_SHARED_SEMANTICS" in source
    assert "CONSUMER_TOTAL_NETWORK_BUDGET_VIOLATION" in source


def test_company_reference_workflow_is_auto_read_only_and_unscheduled() -> None:
    workflow = (ROOT / ".github" / "workflows" / "company-reference-live-validation.yml").read_text(encoding="utf-8").lower()
    assert "push:" in workflow and "branches: [main]" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "--observed-on 2026-09-03" in workflow
    assert '"shared_evidence_hub.py"' in workflow
    assert "schedule:" not in workflow

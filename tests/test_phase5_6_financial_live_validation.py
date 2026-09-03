from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "financial-live-validation.yml"
SCRIPT = ROOT / "tools" / "phase5_6_financial_live_validation.py"


def test_financial_live_producer_is_auto_bounded_and_read_only() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "push:" in workflow
    assert "branches: [main]" in workflow
    assert "schedule:" not in workflow
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "persist-credentials: false" in workflow
    assert "ref: ${{ inputs.selected_ref || github.sha }}" in workflow
    assert "ZAPI_KEY: ${{ secrets.ZAPI_KEY }}" in workflow
    assert "SHARED_EVIDENCE_SUPABASE_SERVICE_ROLE_KEY" in workflow
    assert "--mode producer" in workflow
    assert "--client-id PASTICUAN" in workflow

    assert "UPSTREAM_ACCESS_BLOCKED" in script
    assert 'stage == "OFFICIAL_ATTACHMENT"' in script
    assert 'state == "HTTP_403"' in script
    assert '"authorization": "NO_SHARED_FINANCIAL_FACTS"' in script
    assert '"scoring_eligible": False' in script
    assert "PRODUCER_REQUEST_BUDGET_VIOLATION" in script
    assert "FINANCIAL_CURRENCY_AMBIGUOUS" in script
    assert "FINANCIAL_DOCUMENT_IDENTITY_AMBIGUOUS" in script
    assert "FORBIDDEN_SHARED_SEMANTICS" in script

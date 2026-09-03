from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ownership-live-validation.yml"
SCRIPT = ROOT / "tools" / "phase5_6_ownership_live_validation.py"


def test_ownership_live_producer_is_auto_bounded_and_read_only() -> None:
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
    assert "${{ inputs.category || 'lima-persen' }}" in workflow
    assert "${{ inputs.publication_date || '2026-07-30' }}" in workflow
    assert "ZAPI_KEY: ${{ secrets.ZAPI_KEY }}" in workflow
    assert "SHARED_EVIDENCE_SUPABASE_SERVICE_ROLE_KEY" in workflow
    assert "SHARED_EVIDENCE_SUPABASE_SECRET_KEY" not in workflow
    assert "--mode producer" in workflow
    assert "--client-id PASTICUAN" in workflow

    assert "MAX_INDEX_PAGES" in script
    assert "MAX_FILES_PER_PUBLICATION" in script
    assert 'state != "REFRESHED"' in script
    assert "PRODUCER_API_CALL_BUDGET_VIOLATION" in script
    assert "PRODUCER_FILE_CALL_BUDGET_VIOLATION" in script
    assert "FORBIDDEN_SHARED_SEMANTICS" in script

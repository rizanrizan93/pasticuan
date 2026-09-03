from pathlib import Path


def test_ownership_producer_auto_trigger_is_bounded_and_read_only() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "ownership-live-validation.yml").read_text(encoding="utf-8")
    lowered = workflow.lower()

    assert "push:" in workflow
    assert "branches: [main]" in workflow
    assert "workflow_dispatch:" in workflow
    assert "${{ inputs.selected_ref || github.sha }}" in workflow
    assert "${{ inputs.category || 'lima-persen' }}" in workflow
    assert "${{ inputs.publication_date || '2026-07-30' }}" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "2>&1 | tee ownership_live_output/summary.log" in workflow
    assert '"tests/test_phase5_6_ownership_auto_trigger.py"' in workflow
    assert '"tests/test_phase5_6_ownership_live_validation.py"' in workflow
    assert "schedule:" not in workflow
    assert "git push" not in lowered
    assert "contents: write" not in lowered

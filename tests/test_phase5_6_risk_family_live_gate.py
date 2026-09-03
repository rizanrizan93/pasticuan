from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_risk_family_live_gate_is_bounded_and_fixed_to_factual_feeds() -> None:
    source = (ROOT / "tools" / "phase5_6_risk_family_live_validation.py").read_text(encoding="utf-8")
    for call in ("get_notice_window(", "get_margin(", "get_lendable("):
        assert call in source
    for feed in ("uma", "suspension", "margin-summary", "lendable-stock"):
        assert feed in source
    assert "max_pages=10" in source
    assert "max_pages=3" in source
    assert "ATTACHMENT_CALL_NOT_ALLOWED" in source
    assert "FORBIDDEN_SHARED_SEMANTICS" in source


def test_risk_family_workflow_is_auto_bounded_and_read_only() -> None:
    workflow = (ROOT / ".github" / "workflows" / "risk-family-live-validation.yml").read_text(encoding="utf-8").lower()
    assert "push:" in workflow and "branches: [main]" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "--observed-on 2026-09-03" in workflow
    assert "--notice-from 2026-08-01" in workflow
    assert "--notice-to 2026-08-31" in workflow
    assert "--margin-date 2026-07-14" in workflow
    assert "schedule:" not in workflow

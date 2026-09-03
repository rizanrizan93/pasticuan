from pathlib import Path

from tools.phase5_6_issued_history_live_validation import ALLOWED_FEED


def test_issued_history_live_gate_is_locked_to_single_family() -> None:
    assert ALLOWED_FEED == "issued-history"
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "capital-action-issued-history-live-validation.yml"
    ).read_text(encoding="utf-8").lower()
    assert "--feed issued-history" in workflow
    assert "rights-offerings" not in workflow
    assert "stock-splits" not in workflow
    assert "additional-listings" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "--year" not in workflow
    assert "--month" not in workflow


def test_issued_history_live_gate_is_bounded_and_global() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "phase5_6_issued_history_live_validation.py"
    ).read_text(encoding="utf-8")
    assert "get_issued_history(" in source
    assert "max_pages=MAX_PAGES_PER_RUN" in source
    assert "PRODUCER_FEED_NOT_BOUNDED_COMPLETE" in source
    assert "CONSUMER_NETWORK_BUDGET_VIOLATION" in source
    assert "CONSUMER_REQUEST_NOT_AVOIDED" in source

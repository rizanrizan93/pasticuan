from pathlib import Path

from tools.phase5_6_stock_split_live_validation import ALLOWED_FEED


def test_stock_split_live_gate_is_locked_to_single_family() -> None:
    assert ALLOWED_FEED == "stock-splits"
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "capital-action-stock-split-live-validation.yml"
    ).read_text(encoding="utf-8").lower()
    assert "--feed stock-splits" in workflow
    assert "rights-offerings" not in workflow
    assert "additional-listings" not in workflow
    assert "issued-history" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow

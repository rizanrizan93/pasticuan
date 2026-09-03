from pathlib import Path
from tools.phase5_6_additional_listing_live_validation import ALLOWED_FEED

def test_additional_listing_live_gate_is_locked_to_single_family() -> None:
    assert ALLOWED_FEED == "additional-listings"
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "capital-action-additional-listing-live-validation.yml"
    ).read_text(encoding="utf-8").lower()
    assert "--feed additional-listings" in workflow
    assert "rights-offerings" not in workflow
    assert "stock-splits" not in workflow
    assert "issued-history" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow

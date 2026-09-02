from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"


def _read(name: str) -> str:
    return (WF / name).read_text(encoding="utf-8")


def test_phase5_6_scheduler_ownership_and_repository_write_safety() -> None:
    stock = _read("shared-stock-summary.yml")
    foreign = _read("zapi-foreign-flow.yml")
    participant = _read("public-broker-flow.yml")
    verify = _read("verify-canonical-participant-flow.yml")

    assert "schedule:" in stock
    assert 'cron: "50 10 * * 1-5"' in stock
    for text in (foreign, participant, verify):
        assert "schedule:" not in text
        assert "\npush:" not in text

    for text in (stock, foreign, participant, verify):
        assert "contents: write" not in text
        assert "git push" not in text
        assert "git commit" not in text
        assert "persist-credentials: false" in text

    for text in (stock, foreign, participant):
        assert "SHARED_EVIDENCE_SUPABASE_SERVICE_ROLE_KEY" in text
        assert "SHARED_EVIDENCE_SUPABASE_SECRET_KEY" not in text

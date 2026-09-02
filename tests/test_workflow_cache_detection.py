from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = {
    ROOT / ".github/workflows/zapi-foreign-flow.yml": (
        "data/zapi_foreign_flow_60d.csv.gz",
        "data/zapi_foreign_flow_state.json",
    ),
    ROOT / ".github/workflows/public-broker-flow.yml": (
        "data/public_broker_flow_30d.csv.gz",
    ),
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout


def _changed(repo: Path, paths: tuple[str, ...]) -> bool:
    return bool(_git(repo, "status", "--porcelain", "--", *paths).strip())


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Workflow Test")
    _git(repo, "config", "user.email", "workflow-test@example.invalid")
    (repo / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "baseline")
    return repo


def test_workflows_do_not_persist_runtime_cache_files() -> None:
    for workflow in WORKFLOWS:
        source = workflow.read_text(encoding="utf-8")
        assert "contents: write" not in source
        assert "git status --porcelain --" not in source
        assert "git add " not in source
        assert "git commit" not in source
        assert "git push" not in source


def test_first_untracked_cache_is_detected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    path = repo / "data/cache.csv.gz"
    path.parent.mkdir()
    path.write_bytes(b"first")
    assert _changed(repo, ("data/cache.csv.gz",))


def test_tracked_cache_modification_is_detected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    path = repo / "data/cache.csv.gz"
    path.parent.mkdir()
    path.write_bytes(b"first")
    _git(repo, "add", "data/cache.csv.gz")
    _git(repo, "commit", "-qm", "cache")
    path.write_bytes(b"second")
    assert _changed(repo, ("data/cache.csv.gz",))


def test_unchanged_cache_is_not_detected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    path = repo / "data/cache.csv.gz"
    path.parent.mkdir()
    path.write_bytes(b"first")
    _git(repo, "add", "data/cache.csv.gz")
    _git(repo, "commit", "-qm", "cache")
    assert not _changed(repo, ("data/cache.csv.gz",))


def test_unrelated_untracked_and_runtime_artifacts_are_ignored(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    path = repo / "data/cache.csv.gz"
    path.parent.mkdir()
    path.write_bytes(b"first")
    _git(repo, "add", "data/cache.csv.gz")
    _git(repo, "commit", "-qm", "cache")
    (repo / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
    runtime = repo / ".scanner_cache/runtime.json"
    runtime.parent.mkdir()
    runtime.write_text("{}\n", encoding="utf-8")
    assert not _changed(repo, ("data/cache.csv.gz",))

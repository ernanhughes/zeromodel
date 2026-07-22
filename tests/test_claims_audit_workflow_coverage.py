from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
CLAIMS_AUDIT_WORKFLOW = WORKFLOWS_DIR / "claims-audit.yml"


def test_claims_audit_reacts_to_the_six_package_workspace() -> None:
    source = CLAIMS_AUDIT_WORKFLOW.read_text(encoding="utf-8")
    for required in ("packages/", "tests/", "integration_tests/", "scripts/", "README.md", "package-boundaries.toml"):
        assert required in source, f"claims-audit.yml does not react to {required}"


def test_claims_audit_no_longer_watches_the_deleted_root_source_tree() -> None:
    source = CLAIMS_AUDIT_WORKFLOW.read_text(encoding="utf-8")
    assert 'startswith("zeromodel/")' not in source


def test_no_active_workflow_declares_a_trigger_path_under_the_deleted_root_tree() -> None:
    offenders: list[str] = []
    for workflow in sorted(WORKFLOWS_DIR.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip().strip("-").strip().strip('"').strip("'")
            if stripped == "zeromodel/**":
                offenders.append(workflow.name)
    assert offenders == [], f"workflows still trigger on the deleted root tree: {offenders}"

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES_PATH = REPO_ROOT / "package-boundaries.toml"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

def _publishable_packages(boundaries: dict[str, object]) -> set[str]:
    packages = boundaries["packages"]
    assert isinstance(packages, dict)
    return {
        key
        for key, package in packages.items()
        if isinstance(package, dict) and package.get("publishable") is True
    }


def _assert_workflow_coverage(
    *,
    boundaries: dict[str, object],
    workflow_dir: Path = WORKFLOW_DIR,
) -> None:
    packages = _publishable_packages(boundaries)
    assert packages, "package-boundaries.toml must declare publishable packages"

    python_workflow = (workflow_dir / "python.yml").read_text(encoding="utf-8")
    assert "packages/**" in python_workflow
    assert "package-boundaries.toml" in python_workflow
    assert "python scripts/validate_packaging_contract.py" in python_workflow


def test_every_publishable_package_is_covered_by_manifest_driven_ci() -> None:
    with BOUNDARIES_PATH.open("rb") as handle:
        boundaries = tomllib.load(handle)

    _assert_workflow_coverage(boundaries=boundaries)


def test_active_workflows_do_not_reintroduce_per_package_ci() -> None:
    package_workflows = sorted(WORKFLOW_DIR.glob("*-package.yml"))
    assert package_workflows == []

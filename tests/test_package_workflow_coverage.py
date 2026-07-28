from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES_PATH = REPO_ROOT / "package-boundaries.toml"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

PACKAGE_WORKFLOWS = {
    "analysis": "analysis-package.yml",
    "artifacts": "artifacts-package.yml",
    "core": "core-package.yml",
    "navigation": "navigation-package.yml",
    "observation": "observation-package.yml",
    "observer": "observer-package.yml",
    "perception": "perception-package.yml",
    "sqlalchemy": "sqlalchemy-package.yml",
    "trust": "trust-package.yml",
    "video": "video-package.yml",
    "vision": "vision-package.yml",
}


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
    missing_mapping = packages - set(PACKAGE_WORKFLOWS)
    assert not missing_mapping, (
        "publishable packages lack declared workflow coverage mapping: "
        f"{sorted(missing_mapping)}"
    )
    missing_files = {
        package
        for package in packages
        if not (workflow_dir / PACKAGE_WORKFLOWS[package]).is_file()
    }
    assert not missing_files, (
        f"publishable packages lack workflow files: {sorted(missing_files)}"
    )


def test_every_publishable_package_has_workflow_coverage() -> None:
    with BOUNDARIES_PATH.open("rb") as handle:
        boundaries = tomllib.load(handle)

    _assert_workflow_coverage(boundaries=boundaries)


def test_synthetic_publishable_package_without_workflow_fails() -> None:
    with BOUNDARIES_PATH.open("rb") as handle:
        boundaries = tomllib.load(handle)
    synthetic = dict(boundaries)
    packages = dict(synthetic["packages"])
    packages["synthetic"] = {
        "distribution": "zeromodel-synthetic",
        "namespace": "zeromodel.synthetic",
        "source_root": "packages/synthetic/src",
        "depends_on": [],
        "publishable": True,
        "owned_prefixes": ["zeromodel.synthetic"],
    }
    synthetic["packages"] = packages

    try:
        _assert_workflow_coverage(boundaries=synthetic)
    except AssertionError as exc:
        assert "synthetic" in str(exc)
    else:
        raise AssertionError("synthetic package without workflow coverage passed")

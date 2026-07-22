from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.13"
PACKAGES = {
    "core": {
        "path": Path("packages/core"),
        "distribution": "zeromodel",
        "wheel_stem": "zeromodel",
        "namespace": "zeromodel.core",
        "requires": {"numpy>=1.23"},
    },
    "analysis": {
        "path": Path("packages/analysis"),
        "distribution": "zeromodel-analysis",
        "wheel_stem": "zeromodel_analysis",
        "namespace": "zeromodel.analysis",
        "requires": {"numpy>=1.23", f"zeromodel=={VERSION}"},
    },
    "observation": {
        "path": Path("packages/observation"),
        "distribution": "zeromodel-observation",
        "wheel_stem": "zeromodel_observation",
        "namespace": "zeromodel.observation",
        "requires": {"numpy>=1.23", f"zeromodel=={VERSION}"},
    },
    "vision": {
        "path": Path("packages/vision"),
        "distribution": "zeromodel-vision",
        "wheel_stem": "zeromodel_vision",
        "namespace": "zeromodel.vision",
        "requires": {
            "numpy>=1.23",
            f"zeromodel=={VERSION}",
            f"zeromodel-observation=={VERSION}",
        },
    },
    "video": {
        "path": Path("packages/video"),
        "distribution": "zeromodel-video",
        "wheel_stem": "zeromodel_video",
        "namespace": "zeromodel.video",
        "requires": {
            "numpy>=1.23",
            f"zeromodel=={VERSION}",
            f"zeromodel-observation=={VERSION}",
        },
    },
    "sqlalchemy": {
        "path": Path("packages/sqlalchemy"),
        "distribution": "zeromodel-sqlalchemy",
        "wheel_stem": "zeromodel_sqlalchemy",
        "namespace": "zeromodel.persistence.sqlalchemy",
        "requires": {
            "numpy>=1.23",
            "SQLAlchemy>=2.0,<3",
            f"zeromodel=={VERSION}",
            f"zeromodel-video=={VERSION}",
        },
    },
}
FORBIDDEN_WHEEL_PREFIXES = ("tests/", "research/", "examples/", "docs/", "scripts/")
FORBIDDEN_WHEEL_FILES = {"zeromodel/__init__.py", "zeromodel/persistence/__init__.py"}


@dataclass(frozen=True)
class Artifact:
    package_key: str
    distribution: str
    path: Path
    kind: str


def run(command: list[str], *, timeout: int = 120) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True, timeout=timeout)


def package_pyproject(package_key: str) -> dict[str, Any]:
    path = REPO_ROOT / PACKAGES[package_key]["path"] / "pyproject.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def validate_versions() -> None:
    for key, expected in PACKAGES.items():
        project = package_pyproject(key)["project"]
        if project["name"] != expected["distribution"]:
            raise SystemExit(f"{key}: unexpected distribution name {project['name']}")
        if project["version"] != VERSION:
            raise SystemExit(f"{key}: unexpected version {project['version']}")
        actual = set(project.get("dependencies", ()))
        missing = expected["requires"] - actual
        extra_internal = {
            dep
            for dep in actual
            if dep.startswith("zeromodel") and dep not in expected["requires"]
        }
        if missing or extra_internal:
            raise SystemExit(
                f"{key}: dependency mismatch missing={sorted(missing)} "
                f"extra_internal={sorted(extra_internal)}"
            )


def clean_artifacts() -> None:
    for expected in PACKAGES.values():
        for name in ("build", "dist"):
            shutil.rmtree(REPO_ROOT / expected["path"] / name, ignore_errors=True)


def build_packages() -> None:
    for expected in PACKAGES.values():
        run([sys.executable, "-m", "build", str(expected["path"])], timeout=180)
        dist_files = sorted((REPO_ROOT / expected["path"] / "dist").iterdir())
        run([sys.executable, "-m", "twine", "check", *map(str, dist_files)])


def artifacts() -> list[Artifact]:
    found: list[Artifact] = []
    for key, expected in PACKAGES.items():
        dist_dir = REPO_ROOT / expected["path"] / "dist"
        wheels = sorted(dist_dir.glob("*.whl"))
        sdists = sorted(dist_dir.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise SystemExit(f"{key}: expected one wheel and one sdist")
        found.append(Artifact(key, expected["distribution"], wheels[0], "wheel"))
        found.append(Artifact(key, expected["distribution"], sdists[0], "sdist"))
    return found


def wheel_names() -> list[Path]:
    return [item.path for item in artifacts() if item.kind == "wheel"]


def parse_wheel_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith("/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
    return {
        "name": metadata["Name"],
        "version": metadata["Version"],
        "requires_python": metadata.get("Requires-Python", ""),
        "requires_dist": metadata.get_all("Requires-Dist") or [],
    }


def validate_wheels() -> dict[str, str]:
    owners: dict[str, str] = {}
    for wheel in wheel_names():
        metadata = parse_wheel_metadata(wheel)
        if metadata["version"] != VERSION:
            raise SystemExit(f"{wheel.name}: unexpected wheel version")
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                if member in FORBIDDEN_WHEEL_FILES or member.startswith(
                    FORBIDDEN_WHEEL_PREFIXES
                ):
                    raise SystemExit(f"{wheel.name}: forbidden member {member}")
                if ".dist-info/" not in member:
                    previous = owners.setdefault(member, wheel.name)
                    if previous != wheel.name:
                        raise SystemExit(
                            f"wheel member overlap: {member} in {previous} and {wheel.name}"
                        )
    return owners


def validate_sdists() -> None:
    for artifact in artifacts():
        if artifact.kind != "sdist":
            continue
        with tarfile.open(artifact.path, "r:gz") as archive:
            names = archive.getnames()
        bad = [
            name
            for name in names
            if "/research/" in name
            or "/examples/" in name
            or name.endswith(".sqlite")
            or name.endswith(".sqlite3")
        ]
        if bad:
            raise SystemExit(f"{artifact.path.name}: forbidden sdist members {bad[:5]}")


def install_and_probe() -> None:
    venv = REPO_ROOT / "build" / "full-integration-venv"
    shutil.rmtree(venv, ignore_errors=True)
    run([sys.executable, "-m", "venv", str(venv)], timeout=120)
    python = venv / "Scripts" / "python.exe"
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"], timeout=120)
    run([str(python), "-m", "pip", "install", *map(str, wheel_names())], timeout=180)
    run([str(python), "-m", "pip", "check"], timeout=120)
    probe = """
import importlib, inspect, json, sys
modules = [
    'zeromodel.core',
    'zeromodel.analysis',
    'zeromodel.observation',
    'zeromodel.vision',
    'zeromodel.video',
    'zeromodel.persistence.sqlalchemy',
]
locations = {name: inspect.getfile(importlib.import_module(name)) for name in modules}
try:
    from zeromodel import ScoreTable
except ImportError:
    root_import = 'blocked'
else:
    root_import = 'available'
print(json.dumps({'locations': locations, 'root_import': root_import}, sort_keys=True))
if root_import != 'blocked':
    raise SystemExit('root import unexpectedly available')
"""
    result = subprocess.run(
        [str(python), "-c", probe],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    site_packages = str(venv / "Lib" / "site-packages")
    for name, location in payload["locations"].items():
        if not str(location).startswith(site_packages):
            raise SystemExit(f"{name} imported from checkout: {location}")


def manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts():
        data = artifact.path.read_bytes()
        top_level: set[str] = set()
        direct_dependencies: list[str]
        requires_python = ""
        if artifact.kind == "wheel":
            metadata = parse_wheel_metadata(artifact.path)
            direct_dependencies = list(metadata["requires_dist"])
            requires_python = str(metadata["requires_python"])
            with zipfile.ZipFile(artifact.path) as archive:
                top_level = {name.split("/", 1)[0] for name in archive.namelist()}
        else:
            direct_dependencies = sorted(
                package_pyproject(artifact.package_key)["project"].get(
                    "dependencies", ()
                )
            )
            requires_python = package_pyproject(artifact.package_key)["project"][
                "requires-python"
            ]
            with tarfile.open(artifact.path, "r:gz") as archive:
                top_level = {name.split("/", 1)[0] for name in archive.getnames()}
        rows.append(
            {
                "filename": artifact.path.name,
                "distribution": artifact.distribution,
                "version": VERSION,
                "kind": artifact.kind,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
                "contained_top_level_paths": sorted(top_level),
                "direct_dependencies": direct_dependencies,
                "python_requirement": requires_python,
            }
        )
    return rows


def write_manifest() -> None:
    path = REPO_ROOT / "docs" / "architecture" / "package-release-artifacts-1.0.13.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version": VERSION,
                "artifacts": manifest_rows(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(path.relative_to(REPO_ROOT).as_posix())


def write_public_exports() -> None:
    path = REPO_ROOT / "docs" / "architecture" / "package-public-api-1.0.13.csv"
    rows = []
    for key, expected in PACKAGES.items():
        module = expected["namespace"]
        rows.append(
            {
                "distribution": expected["distribution"],
                "namespace": module,
                "exported_symbol": "__all__",
                "owning_module": module,
                "reason_public": "explicit package public API",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            [
                "distribution",
                "namespace",
                "exported_symbol",
                "owning_module",
                "reason_public",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    validate_versions()
    if not args.skip_build:
        clean_artifacts()
        build_packages()
    validate_wheels()
    validate_sdists()
    install_and_probe()
    write_manifest()
    write_public_exports()
    print("Release candidate validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

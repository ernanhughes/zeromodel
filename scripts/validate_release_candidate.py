from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
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
        "depends_on": (),
    },
    "analysis": {
        "path": Path("packages/analysis"),
        "distribution": "zeromodel-analysis",
        "wheel_stem": "zeromodel_analysis",
        "namespace": "zeromodel.analysis",
        "requires": {"numpy>=1.23", f"zeromodel=={VERSION}"},
        "depends_on": ("core",),
    },
    "observation": {
        "path": Path("packages/observation"),
        "distribution": "zeromodel-observation",
        "wheel_stem": "zeromodel_observation",
        "namespace": "zeromodel.observation",
        "requires": {"numpy>=1.23", f"zeromodel=={VERSION}"},
        "depends_on": ("core",),
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
        "depends_on": ("core", "observation"),
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
        "depends_on": ("core", "observation"),
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
        "depends_on": ("core", "video", "observation"),
    },
    "artifacts": {
        "path": Path("packages/artifacts"),
        "distribution": "zeromodel-artifacts",
        "wheel_stem": "zeromodel_artifacts",
        "namespace": "zeromodel.artifacts",
        "requires": {"numpy>=1.23", f"zeromodel=={VERSION}"},
        "depends_on": ("core",),
    },
    "trust": {
        "path": Path("packages/trust"),
        "distribution": "zeromodel-trust",
        "wheel_stem": "zeromodel_trust",
        "namespace": "zeromodel.trust",
        "requires": {
            "numpy>=1.23",
            "cryptography>=41",
            f"zeromodel=={VERSION}",
            f"zeromodel-artifacts=={VERSION}",
        },
        "depends_on": ("core", "artifacts"),
    },
    "navigation": {
        "path": Path("packages/navigation"),
        "distribution": "zeromodel-navigation",
        "wheel_stem": "zeromodel_navigation",
        "namespace": "zeromodel.navigation",
        "requires": {
            "numpy>=1.23",
            f"zeromodel=={VERSION}",
            f"zeromodel-artifacts=={VERSION}",
        },
        "depends_on": ("core", "artifacts"),
    },
}
PUBLIC_API_CSV_COLUMNS = [
    "distribution",
    "namespace",
    "exported_symbol",
    "owning_module",
    "object_kind",
    "source_module",
    "is_reexport",
]
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


def venv_python(venv: Path, *, is_windows: bool | None = None) -> Path:
    """Return the interpreter path for a venv created at ``venv``, on either OS.

    ``is_windows`` is an explicit override so this is unit-testable on any host;
    it defaults to the actual running platform via ``os.name``.
    """
    if is_windows is None:
        is_windows = os.name == "nt"
    if is_windows:
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def is_beneath(path: Path, root: Path) -> bool:
    """True if ``path`` resolves to a location inside ``root``."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def install_and_probe() -> dict[str, dict[str, Any]]:
    """Build a clean venv, install all six wheels, and smoke-probe each namespace.

    Returns a per-package wheel-smoke result dict (used by
    release_test_layer_report() below): each package's namespace either
    imports cleanly from the installed wheel (inside the clean venv, never
    the checkout) or the whole validation fails outright, since this
    function raises SystemExit on any violation before returning.
    """
    venv = REPO_ROOT / "build" / "full-integration-venv"
    shutil.rmtree(venv, ignore_errors=True)
    run([sys.executable, "-m", "venv", str(venv)], timeout=120)
    python = venv_python(venv)
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
    namespace_to_key = {
        expected["namespace"]: key for key, expected in PACKAGES.items()
    }
    wheel_smoke: dict[str, dict[str, Any]] = {}
    for name, location in payload["locations"].items():
        beneath = is_beneath(Path(location), venv)
        wheel_smoke[namespace_to_key[name]] = {
            "namespace": name,
            "location": location,
            "imported_from_installed_wheel": beneath,
        }
        if not beneath:
            raise SystemExit(
                f"{name} imported from outside the clean virtual environment: {location}"
            )
    return wheel_smoke


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


_PUBLIC_API_PROBE = """
import importlib
import inspect
import json

namespaces = {namespaces!r}
report = {{}}
for key, namespace in namespaces.items():
    module = importlib.import_module(namespace)
    all_symbols = getattr(module, "__all__", None)
    if all_symbols is None:
        raise SystemExit(f"{{namespace}}: package has no __all__")
    symbols = {{}}
    for name in all_symbols:
        if name not in symbols:
            symbols[name] = []
        symbols[name].append(True)
        obj = getattr(module, name)
        if inspect.isclass(obj):
            kind = "Class"
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj):
            kind = "Function"
        else:
            kind = "Constant"
        source_module = getattr(obj, "__module__", None) or namespace
        symbols[name] = {{
            "kind": kind,
            "source_module": source_module,
            "count": len(symbols[name]),
        }}
    report[key] = {{"namespace": namespace, "all": list(all_symbols), "symbols": symbols}}
print(json.dumps(report))
"""


def _probe_public_api(python: Path) -> dict[str, Any]:
    namespaces = {key: expected["namespace"] for key, expected in PACKAGES.items()}
    script = _PUBLIC_API_PROBE.format(namespaces=namespaces)
    result = subprocess.run(
        [str(python), "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def write_public_exports() -> None:
    """Generate a real, per-symbol public API manifest.

    Introspects each package's actual `__all__` in the same clean venv built
    by install_and_probe() (an isolated validation environment with all six
    wheels installed, nothing importable from the checkout) - never a
    placeholder row per distribution. `__all__` remains the sole authority:
    this never walks `dir(module)` or exports anything not explicitly listed.
    """
    venv = REPO_ROOT / "build" / "full-integration-venv"
    python = venv_python(venv)
    report = _probe_public_api(python)

    allowed_namespace_prefixes: dict[str, tuple[str, ...]] = {}
    for key, expected in PACKAGES.items():
        own = (expected["namespace"],)
        deps = tuple(PACKAGES[dep]["namespace"] for dep in expected["depends_on"])
        allowed_namespace_prefixes[key] = own + deps

    rows: list[dict[str, Any]] = []
    for key, expected in PACKAGES.items():
        payload = report[key]
        namespace = payload["namespace"]
        symbols = payload["symbols"]
        all_list = payload["all"]

        duplicates = sorted({name for name in all_list if all_list.count(name) > 1})
        if duplicates:
            raise SystemExit(f"{namespace}: duplicate symbols in __all__: {duplicates}")

        private = sorted(name for name in all_list if name.startswith("_"))
        if private:
            raise SystemExit(
                f"{namespace}: __all__ exports private-looking names: {private}"
            )

        for name in all_list:
            info = symbols[name]
            source_module = info["source_module"]
            allowed_prefixes = allowed_namespace_prefixes[key]
            if not any(
                source_module == prefix or source_module.startswith(f"{prefix}.")
                for prefix in allowed_prefixes
            ):
                raise SystemExit(
                    f"{namespace}.{name}: implementation module {source_module!r} does not "
                    f"belong to {expected['distribution']} or one of its declared "
                    f"dependencies {expected['depends_on']}"
                )
            rows.append(
                {
                    "distribution": expected["distribution"],
                    "namespace": namespace,
                    "exported_symbol": name,
                    "owning_module": namespace,
                    "object_kind": info["kind"],
                    "source_module": source_module,
                    "is_reexport": str(source_module != namespace).lower(),
                }
            )

    if len(rows) <= len(PACKAGES):
        raise SystemExit(
            "Public API manifest looks like a placeholder (one row per "
            "distribution) rather than a real per-symbol manifest."
        )

    rows.sort(key=lambda row: (row["distribution"], row["exported_symbol"]))

    path = REPO_ROOT / "docs" / "architecture" / "package-public-api-1.0.13.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, PUBLIC_API_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{path.relative_to(REPO_ROOT).as_posix()}: {len(rows)} public symbols across {len(PACKAGES)} distributions"
    )


def _pytest_count(args: list[str], *, timeout: int = 180) -> dict[str, Any]:
    """Run pytest with -q and pull collected/passed/failed counts from its summary line.

    Used only for source-tree counts here (fast suite, package-local
    source), where full structured reporting already exists via
    scripts/fast_suite_reporter.py for the canonical fast-suite command
    itself; this is a lighter-weight counter for the additional per-package
    breakdowns this report adds on top of that.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    summary_line = ""
    for line in reversed(result.stdout.splitlines()):
        if " in " in line and (
            "passed" in line
            or "failed" in line
            or "error" in line
            or "no tests ran" in line
        ):
            summary_line = line.strip()
            break
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for key in counts:
        match = re.search(rf"(\d+) {key}", summary_line)
        if match:
            counts[key] = int(match.group(1))
    counts["returncode"] = result.returncode
    counts["summary_line"] = summary_line
    return counts


def release_test_layer_report() -> dict[str, Any]:
    """Report exactly which test layers back a release candidate, and which don't.

    Distinguishes source-tree fast production tests, package-local source
    tests (per package), cross-package integration tests, installed-wheel
    smoke results (per package, from install_and_probe()'s clean-venv
    import probe - never editable source), and explicitly states that
    research is out of the production release contract by policy rather
    than leaving it silently absent from the report.
    """
    fast_suite = _pytest_count(
        [
            "--maxfail=1",
            "-m",
            "not slow and not external and not research",
            "tests",
            "integration_tests",
        ]
    )
    package_local: dict[str, Any] = {}
    for key in PACKAGES:
        package_local[key] = _pytest_count([f"packages/{key}/tests"])
    integration = _pytest_count(
        [
            "--run-integration",
            "-m",
            "integration",
            "tests",
            "integration_tests",
            *(f"packages/{key}/tests" for key in PACKAGES),
        ]
    )

    report = {
        "source_tree_fast_production_tests": fast_suite,
        "package_local_source_tests_by_package": package_local,
        "cross_package_integration_tests": integration,
        "installed_wheel_smoke_result_by_package": WHEEL_SMOKE_RESULTS,
        "research": {
            "status": "excluded_by_policy",
            "note": (
                "research/ is not part of the production release contract and is "
                "not executed by this validator. See "
                "docs/reviews/post-split-research-health.md for its own, "
                "separately-tracked collection/health status."
            ),
        },
    }
    path = (
        REPO_ROOT / "docs" / "architecture" / "package-release-test-layers-1.0.13.json"
    )
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{path.relative_to(REPO_ROOT).as_posix()}: test-layer report written")
    return report


WHEEL_SMOKE_RESULTS: dict[str, dict[str, Any]] = {}


def main() -> int:
    global WHEEL_SMOKE_RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    validate_versions()
    if not args.skip_build:
        clean_artifacts()
        build_packages()
    validate_wheels()
    validate_sdists()
    WHEEL_SMOKE_RESULTS = install_and_probe()
    write_manifest()
    write_public_exports()
    release_test_layer_report()
    print("Release candidate validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

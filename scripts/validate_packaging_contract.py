from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES_FILE = REPO_ROOT / "package-boundaries.toml"
VERSION_FILE = REPO_ROOT / "VERSION"
WHEELHOUSE = REPO_ROOT / "build" / "wheelhouse"
REPORT_PATH = REPO_ROOT / "build" / "reports" / "packaging-contract.json"


def load_manifest() -> dict[str, Any]:
    return tomllib.loads(BOUNDARIES_FILE.read_text(encoding="utf-8"))


def read_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def package_root(package_key: str, manifest: dict[str, Any]) -> Path:
    source_root = Path(manifest["packages"][package_key]["source_root"])
    if source_root.name == "src":
        return REPO_ROOT / source_root.parent
    return REPO_ROOT / source_root


def wheel_stem(distribution: str) -> str:
    return re.sub(r"[-.]+", "_", distribution)


def publishable_package_keys(manifest: dict[str, Any]) -> list[str]:
    return [
        key
        for key, config in manifest["packages"].items()
        if config.get("publishable") is True
    ]


def runtime_package_keys(manifest: dict[str, Any]) -> list[str]:
    return [
        key
        for key, config in manifest["packages"].items()
        if config.get("kind", "runtime") == "runtime"
    ]


def meta_package_key(manifest: dict[str, Any]) -> str:
    matches = [
        key
        for key, config in manifest["packages"].items()
        if config.get("kind", "runtime") == "meta"
    ]
    if matches != ["meta"]:
        raise SystemExit(f"Expected exactly one meta package named 'meta': {matches}")
    return matches[0]


def run(command: list[str], *, timeout: int = 240) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        timeout=timeout,
        text=True,
        capture_output=True,
    )


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def parse_wheel_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith("/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        payload_members = [
            name
            for name in archive.namelist()
            if ".dist-info/" not in name and not name.endswith("/")
        ]
    return {
        "name": metadata["Name"],
        "version": metadata["Version"],
        "requires_dist": metadata.get_all("Requires-Dist") or [],
        "payload_members": payload_members,
    }


def built_wheel(package_key: str, manifest: dict[str, Any]) -> Path:
    config = manifest["packages"][package_key]
    dist_dir = package_root(package_key, manifest) / "dist"
    pattern = f"{wheel_stem(config['distribution'])}-{read_version()}-*.whl"
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"{package_key}: expected one wheel matching {pattern}")
    return matches[0]


def build_wheelhouse(manifest: dict[str, Any]) -> list[Path]:
    shutil.rmtree(WHEELHOUSE, ignore_errors=True)
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    wheels: list[Path] = []
    for key in publishable_package_keys(manifest):
        root = package_root(key, manifest)
        shutil.rmtree(root / "dist", ignore_errors=True)
        shutil.rmtree(root / "build", ignore_errors=True)
        run([sys.executable, "-m", "build", str(root)], timeout=240)
        run([sys.executable, "-m", "twine", "check", *map(str, (root / "dist").iterdir())])
        wheel = built_wheel(key, manifest)
        target = WHEELHOUSE / wheel.name
        shutil.copy2(wheel, target)
        wheels.append(target)
    return wheels


def expected_internal_requirement_set(
    package_key: str, manifest: dict[str, Any]
) -> set[str]:
    version = read_version()
    packages = manifest["packages"]
    return {
        f"{packages[dependency]['distribution']}=={version}"
        for dependency in packages[package_key].get("depends_on", [])
    }


def validate_metadata_invariants(manifest: dict[str, Any], wheels: list[Path]) -> None:
    version = read_version()
    by_name = {parse_wheel_metadata(path)["name"]: path for path in wheels}
    packages = manifest["packages"]
    meta_key = meta_package_key(manifest)
    meta_distribution = packages[meta_key]["distribution"]
    if meta_distribution != "zeromodel":
        raise SystemExit("Meta package must publish the zeromodel distribution")
    if packages["core"]["distribution"] != "zeromodel-core":
        raise SystemExit("Core package must publish zeromodel-core")

    for key in publishable_package_keys(manifest):
        config = packages[key]
        metadata = parse_wheel_metadata(by_name[config["distribution"]])
        if metadata["version"] != version:
            raise SystemExit(f"{key}: wheel version is not {version}")
        internal = {
            requirement.split(";", 1)[0].strip()
            for requirement in metadata["requires_dist"]
            if requirement.startswith("zeromodel")
        }
        expected = expected_internal_requirement_set(key, manifest)
        if internal != expected:
            raise SystemExit(
                f"{key}: internal wheel requirements {sorted(internal)} "
                f"!= {sorted(expected)}"
            )
        if key != meta_key and "zeromodel==" in " ".join(internal):
            raise SystemExit(f"{key}: implementation package depends on umbrella")
        if key == meta_key and metadata["payload_members"]:
            raise SystemExit(f"{key}: umbrella wheel owns payload files")

    expected_umbrella = {
        f"{packages[key]['distribution']}=={version}"
        for key in runtime_package_keys(manifest)
    }
    actual_umbrella = expected_internal_requirement_set(meta_key, manifest)
    if actual_umbrella != expected_umbrella:
        raise SystemExit(
            "umbrella dependency set does not match runtime package set: "
            f"{sorted(actual_umbrella)} != {sorted(expected_umbrella)}"
        )
    if f"{meta_distribution}=={version}" in actual_umbrella:
        raise SystemExit("umbrella depends on itself")


def external_runtime_dependencies(manifest: dict[str, Any]) -> list[str]:
    dependencies: set[str] = set()
    for key in publishable_package_keys(manifest):
        pyproject = tomllib.loads(
            (package_root(key, manifest) / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        dependencies.update(
            dependency
            for dependency in pyproject["project"].get("dependencies", [])
            if not dependency.startswith("zeromodel")
        )
    return sorted(dependencies)


def install_external_runtime_dependencies(venv: Path, manifest: dict[str, Any]) -> None:
    dependencies = external_runtime_dependencies(manifest)
    if dependencies:
        run([str(venv_python(venv)), "-m", "pip", "install", *dependencies], timeout=240)


def install_from_wheelhouse(
    venv: Path, requirement: str, manifest: dict[str, Any]
) -> tuple[str, str]:
    shutil.rmtree(venv, ignore_errors=True)
    run([sys.executable, "-m", "venv", str(venv)])
    python = venv_python(venv)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    install_external_runtime_dependencies(venv, manifest)
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(WHEELHOUSE),
            requirement,
        ],
        timeout=240,
    )
    check = run([str(python), "-m", "pip", "check"])
    listing = run([str(python), "-m", "pip", "list", "--format=json"])
    return check.stdout, listing.stdout


def public_import_probe(venv: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    modules = [
        config["namespace"]
        for config in manifest["packages"].values()
        if config.get("kind", "runtime") == "runtime"
    ]
    probe = """
import importlib, inspect, json
modules = {modules!r}
locations = {{name: inspect.getfile(importlib.import_module(name)) for name in modules}}
from zeromodel.core import ScoreTable, LayoutRecipe, VPMPolicyLookup
print(json.dumps({{"locations": locations, "symbols": ["ScoreTable", "LayoutRecipe", "VPMPolicyLookup"]}}, sort_keys=True))
""".format(modules=modules)
    result = run([str(venv_python(venv)), "-c", probe])
    return json.loads(result.stdout)


def validate_core_only(venv: Path) -> list[dict[str, Any]]:
    _, listing = install_from_wheelhouse(
        venv, f"zeromodel-core=={read_version()}", load_manifest()
    )
    installed = json.loads(listing)
    names = {item["name"] for item in installed}
    unexpected = sorted(name for name in names if name.startswith("zeromodel-") and name != "zeromodel-core")
    if "zeromodel" in names or unexpected:
        raise SystemExit(f"core-only install pulled unrelated packages: {unexpected}")
    run([str(venv_python(venv)), "-c", "from zeromodel.core import ScoreTable"])
    return installed


def validate_component_installs(manifest: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for key in runtime_package_keys(manifest):
        distribution = manifest["packages"][key]["distribution"]
        venv = REPO_ROOT / "build" / "component-install-venvs" / key
        _, listing = install_from_wheelhouse(
            venv, f"{distribution}=={read_version()}", manifest
        )
        namespace = manifest["packages"][key]["namespace"]
        run([str(venv_python(venv)), "-c", f"import {namespace}"])
        installed = json.loads(listing)
        results[key] = {
            "distribution": distribution,
            "namespace": namespace,
            "installed_zeromodel_distributions": sorted(
                item["name"]
                for item in installed
                if item["name"] == "zeromodel" or item["name"].startswith("zeromodel-")
            ),
        }
    return results


def validate_upgrade_path(venv: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    shutil.rmtree(venv, ignore_errors=True)
    run([sys.executable, "-m", "venv", str(venv)])
    python = venv_python(venv)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python), "-m", "pip", "install", "zeromodel==1.2.0"], timeout=240)
    install_external_runtime_dependencies(venv, manifest)
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-index",
            "--find-links",
            str(WHEELHOUSE),
            f"zeromodel=={read_version()}",
        ],
        timeout=240,
    )
    check = run([str(python), "-m", "pip", "check"])
    imports = public_import_probe(venv, manifest)
    listing = json.loads(run([str(python), "-m", "pip", "list", "--format=json"]).stdout)
    show_umbrella = run([str(python), "-m", "pip", "show", "zeromodel"]).stdout
    show_core = run([str(python), "-m", "pip", "show", "zeromodel-core"]).stdout
    return {
        "pip_check": check.stdout.strip(),
        "imports": imports,
        "pip_list": listing,
        "pip_show_zeromodel": show_umbrella,
        "pip_show_zeromodel_core": show_core,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-upgrade", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest()
    wheels = sorted(WHEELHOUSE.glob("*.whl")) if args.skip_build else build_wheelhouse(manifest)
    validate_metadata_invariants(manifest, wheels)

    umbrella_venv = REPO_ROOT / "build" / "packaging-umbrella-venv"
    check, listing = install_from_wheelhouse(
        umbrella_venv, f"zeromodel=={read_version()}", manifest
    )
    imports = public_import_probe(umbrella_venv, manifest)
    core_only = validate_core_only(REPO_ROOT / "build" / "packaging-core-only-venv")
    components = validate_component_installs(manifest)
    upgrade = None if args.skip_upgrade else validate_upgrade_path(
        REPO_ROOT / "build" / "packaging-upgrade-venv", manifest
    )

    report = {
        "version": read_version(),
        "wheelhouse": str(WHEELHOUSE),
        "umbrella_install": {
            "requirement": f"zeromodel=={read_version()}",
            "pip_check": check.strip(),
            "pip_list": json.loads(listing),
            "imports": imports,
        },
        "core_only_install": core_only,
        "component_installs": components,
        "upgrade_1_2_0_to_current": upgrade,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{REPORT_PATH.relative_to(REPO_ROOT).as_posix()}: packaging contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

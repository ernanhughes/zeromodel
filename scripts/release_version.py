from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = REPO_ROOT / "VERSION"
BOUNDARIES_FILE = REPO_ROOT / "package-boundaries.toml"
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)?$")
INTERNAL_REQUIREMENT_PATTERN = re.compile(
    r"(?P<name>zeromodel(?:-[A-Za-z0-9-]+)?)==(?P<version>[^\s;\"']+)"
)
PACKAGE_VERSION_CONSTANT_PATTERN = re.compile(
    r'^(?P<name>[A-Z][A-Z0-9_]*PACKAGE_VERSION)\s*=\s*"(?P<version>[^"]+)"$',
    re.MULTILINE,
)
HARDCODED_WORKFLOW_WHEEL_PATTERN = re.compile(
    r"zeromodel(?:[_-][A-Za-z0-9]+)*-[0-9]+\.[0-9]+\.[0-9]+[^\s\"']*\.whl",
    re.IGNORECASE,
)


def read_version() -> str:
    version = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not VERSION_PATTERN.fullmatch(version):
        raise SystemExit(f"Invalid release version in {VERSION_FILE}: {version!r}")
    return version


def load_manifest() -> dict[str, Any]:
    return tomllib.loads(BOUNDARIES_FILE.read_text(encoding="utf-8"))


def package_root(package_key: str, manifest: dict[str, Any] | None = None) -> Path:
    manifest = load_manifest() if manifest is None else manifest
    try:
        source_root = Path(manifest["packages"][package_key]["source_root"])
    except KeyError as exc:
        known = ", ".join(sorted(manifest["packages"]))
        raise SystemExit(
            f"Unknown package {package_key!r}; expected one of: {known}"
        ) from exc
    return REPO_ROOT / source_root.parent


def version_constant_files() -> list[Path]:
    # Only package entry points carry the coordinated public package version.
    # Historical staged API modules may retain the version of the stage they record.
    return sorted((REPO_ROOT / "packages").glob("*/src/**/__init__.py"))


def wheel_stem(distribution: str) -> str:
    return re.sub(r"[-.]+", "_", distribution)


def built_wheel_path(package_key: str) -> Path:
    manifest = load_manifest()
    config = manifest["packages"].get(package_key)
    if config is None:
        known = ", ".join(sorted(manifest["packages"]))
        raise SystemExit(f"Unknown package {package_key!r}; expected one of: {known}")
    version = read_version()
    dist_dir = package_root(package_key, manifest) / "dist"
    pattern = f"{wheel_stem(config['distribution'])}-{version}-*.whl"
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        found = (
            ", ".join(path.name for path in sorted(dist_dir.glob("*.whl"))) or "none"
        )
        raise SystemExit(
            f"Expected exactly one {package_key} wheel matching {pattern!r} in "
            f"{dist_dir}; found {len(matches)} matching wheel(s), all wheels: {found}"
        )
    return matches[0].resolve()


def install_built_wheels(python_executable: str, package_keys: list[str]) -> None:
    wheels = [str(built_wheel_path(package_key)) for package_key in package_keys]
    if not wheels:
        raise SystemExit("At least one package key is required")
    subprocess.run(
        [python_executable, "-m", "pip", "install", *wheels],
        cwd=REPO_ROOT,
        check=True,
    )


def expected_internal_requirements(
    package_key: str, manifest: dict[str, Any], version: str
) -> set[str]:
    distributions = {
        key: config["distribution"] for key, config in manifest["packages"].items()
    }
    return {
        f"{distributions[dependency]}=={version}"
        for dependency in manifest["packages"][package_key].get("depends_on", [])
    }


def version_sync_errors() -> list[str]:
    version = read_version()
    manifest = load_manifest()
    errors: list[str] = []

    if str(manifest.get("release_version", "")) != version:
        errors.append(
            "package-boundaries.toml: release_version is not synchronized with VERSION"
        )

    for package_key, config in manifest["packages"].items():
        pyproject = package_root(package_key, manifest) / "pyproject.toml"
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
        if project["name"] != config["distribution"]:
            errors.append(
                f"{pyproject.relative_to(REPO_ROOT)}: project.name is "
                f"{project['name']!r}, expected {config['distribution']!r}"
            )
        if project["version"] != version:
            errors.append(
                f"{pyproject.relative_to(REPO_ROOT)}: project.version is "
                f"{project['version']!r}, expected VERSION {version!r}"
            )
        actual_internal = {
            requirement
            for requirement in project.get("dependencies", [])
            if requirement.startswith("zeromodel")
        }
        expected_internal = expected_internal_requirements(
            package_key, manifest, version
        )
        if actual_internal != expected_internal:
            errors.append(
                f"{pyproject.relative_to(REPO_ROOT)}: internal dependencies are "
                f"{sorted(actual_internal)!r}, expected {sorted(expected_internal)!r}"
            )

    for source_file in version_constant_files():
        text = source_file.read_text(encoding="utf-8")
        for match in PACKAGE_VERSION_CONSTANT_PATTERN.finditer(text):
            if match.group("version") != version:
                errors.append(
                    f"{source_file.relative_to(REPO_ROOT)}: {match.group('name')} is "
                    f"{match.group('version')!r}, expected VERSION {version!r}"
                )

    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        for match in HARDCODED_WORKFLOW_WHEEL_PATTERN.finditer(text):
            errors.append(
                f"{workflow.relative_to(REPO_ROOT)}: hard-coded wheel path "
                f"{match.group(0)!r}; use scripts/release_version.py"
            )
    return errors


def sync_version_mirrors() -> None:
    version = read_version()
    boundaries_text = BOUNDARIES_FILE.read_text(encoding="utf-8")
    boundaries_text, count = re.subn(
        r'(?m)^release_version = "[^"]+"$',
        f'release_version = "{version}"',
        boundaries_text,
        count=1,
    )
    if count != 1:
        raise SystemExit(f"Could not update release_version in {BOUNDARIES_FILE}")
    BOUNDARIES_FILE.write_text(boundaries_text, encoding="utf-8")

    manifest = load_manifest()
    for package_key in manifest["packages"]:
        pyproject = package_root(package_key, manifest) / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        text, count = re.subn(
            r'(?m)^version = "[^"]+"$',
            f'version = "{version}"',
            text,
            count=1,
        )
        if count != 1:
            raise SystemExit(f"Could not update one project version in {pyproject}")
        text = INTERNAL_REQUIREMENT_PATTERN.sub(
            lambda match: f"{match.group('name')}=={version}", text
        )
        pyproject.write_text(text, encoding="utf-8")

    for source_file in version_constant_files():
        text = source_file.read_text(encoding="utf-8")
        updated = PACKAGE_VERSION_CONSTANT_PATTERN.sub(
            lambda match: f'{match.group("name")} = "{version}"', text
        )
        if updated != text:
            source_file.write_text(updated, encoding="utf-8")


def command_check() -> int:
    errors = version_sync_errors()
    if errors:
        print("Release version check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Release version check passed: {read_version()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the repository-wide ZeroModel release version."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show", help="Print the canonical release version")
    commands.add_parser("check", help="Verify all generated version mirrors")
    commands.add_parser("sync", help="Update package metadata from VERSION")
    wheel = commands.add_parser("wheel-path", help="Print an exact built wheel path")
    wheel.add_argument("package")
    install = commands.add_parser("install", help="Install exact built wheels")
    install.add_argument("--python", required=True, dest="python_executable")
    install.add_argument("packages", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "show":
        print(read_version())
        return 0
    if args.command == "check":
        return command_check()
    if args.command == "sync":
        sync_version_mirrors()
        return command_check()
    if args.command == "wheel-path":
        print(built_wheel_path(args.package))
        return 0
    if args.command == "install":
        install_built_wheels(args.python_executable, args.packages)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

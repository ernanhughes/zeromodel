"""Bootstrap a local ZeroModel development environment from requirements-dev.txt."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_DEV = REPO_ROOT / "requirements-dev.txt"
BOUNDARIES = REPO_ROOT / "package-boundaries.toml"
CRITICAL_IMPORTS = (
    "pytest",
    "numpy",
    "PIL",
    "matplotlib",
    "cryptography",
    "sqlalchemy",
)
def _manifest_packages() -> dict[str, object]:
    return tomllib.loads(BOUNDARIES.read_text(encoding="utf-8"))["packages"]


def _zeromodel_imports() -> tuple[str, ...]:
    return tuple(
        config["namespace"]
        for config in _manifest_packages().values()
        if config.get("kind", "runtime") == "runtime"
    )


def _version_distributions() -> tuple[str, ...]:
    return tuple(
        config["distribution"]
        for config in _manifest_packages().values()
        if config.get("publishable") is True
    )


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def install_requirements() -> None:
    if not REQUIREMENTS_DEV.exists():
        raise SystemExit(f"Missing development requirements: {REQUIREMENTS_DEV}")
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS_DEV),
        ]
    )


def verify_imports() -> dict[str, object]:
    modules = {}
    for module_name in CRITICAL_IMPORTS + _zeromodel_imports():
        module = importlib.import_module(module_name)
        modules[module_name] = getattr(module, "__file__", None)
    versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in _version_distributions()
    }
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "modules": modules,
        "versions": versions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-fast-tests",
        action="store_true",
        help="run scripts/run_fast_tests.py after installation and import verification",
    )
    args = parser.parse_args(argv)

    install_requirements()
    payload = verify_imports()
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.run_fast_tests:
        _run([sys.executable, str(REPO_ROOT / "scripts" / "run_fast_tests.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

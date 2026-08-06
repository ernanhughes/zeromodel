"""Bootstrap a local ZeroModel development environment from requirements-dev.txt."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_DEV = REPO_ROOT / "requirements-dev.txt"
CRITICAL_IMPORTS = (
    "pytest",
    "numpy",
    "PIL",
    "matplotlib",
    "cryptography",
    "sqlalchemy",
)
ZEROMODEL_IMPORTS = (
    "zeromodel.core",
    "zeromodel.analysis",
    "zeromodel.observation",
    "zeromodel.vision",
    "zeromodel.perception",
    "zeromodel.observer",
    "zeromodel.video",
    "zeromodel.persistence.sqlalchemy",
    "zeromodel.artifacts",
    "zeromodel.trust",
    "zeromodel.navigation",
    "zeromodel.search",
)
VERSION_DISTRIBUTIONS = (
    "zeromodel",
    "zeromodel-analysis",
    "zeromodel-observation",
    "zeromodel-vision",
    "zeromodel-perception",
    "zeromodel-observer",
    "zeromodel-video",
    "zeromodel-sqlalchemy",
    "zeromodel-artifacts",
    "zeromodel-trust",
    "zeromodel-navigation",
    "zeromodel-search",
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
    for module_name in CRITICAL_IMPORTS + ZEROMODEL_IMPORTS:
        module = importlib.import_module(module_name)
        modules[module_name] = getattr(module, "__file__", None)
    versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in VERSION_DISTRIBUTIONS
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

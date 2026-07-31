from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "bootstrap_dev_environment.py"

SPEC = importlib.util.spec_from_file_location("bootstrap_dev_environment", SCRIPT)
assert SPEC is not None
bootstrap = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = bootstrap
SPEC.loader.exec_module(bootstrap)


def test_bootstrap_installs_requirements_dev_not_duplicate_package_list(
    monkeypatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(bootstrap, "_run", lambda command: commands.append(command))

    bootstrap.install_requirements()

    assert commands[0][:4] == [sys.executable, "-m", "pip", "install"]
    assert commands[0][-1] == "pip"
    assert commands[1] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(REPO_ROOT / "requirements-dev.txt"),
    ]


def test_bootstrap_run_fast_tests_waits_until_after_verify(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        bootstrap, "install_requirements", lambda: events.append("install")
    )
    monkeypatch.setattr(
        bootstrap,
        "verify_imports",
        lambda: events.append("verify") or {"ok": True},
    )
    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda command: events.append(" ".join(command)),
    )

    assert bootstrap.main(["--run-fast-tests"]) == 0

    assert events[0:2] == ["install", "verify"]
    assert events[2].endswith("scripts\\run_fast_tests.py") or events[2].endswith(
        "scripts/run_fast_tests.py"
    )

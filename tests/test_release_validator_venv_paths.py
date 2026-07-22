from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path("scripts/validate_release_candidate.py")
SPEC = importlib.util.spec_from_file_location("validate_release_candidate", SCRIPT)
assert SPEC is not None
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def test_venv_python_resolves_windows_interpreter_path(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    assert (
        validator.venv_python(venv, is_windows=True) == venv / "Scripts" / "python.exe"
    )


def test_venv_python_resolves_posix_interpreter_path(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    assert validator.venv_python(venv, is_windows=False) == venv / "bin" / "python"


def test_venv_python_defaults_to_the_running_platform(tmp_path: Path) -> None:
    import os

    venv = tmp_path / "venv"
    resolved = validator.venv_python(venv)
    if os.name == "nt":
        assert resolved == venv / "Scripts" / "python.exe"
    else:
        assert resolved == venv / "bin" / "python"


def test_is_beneath_rejects_a_checkout_path(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    venv.mkdir()
    checkout_module = (
        tmp_path / "packages" / "core" / "src" / "zeromodel" / "core" / "__init__.py"
    )
    checkout_module.parent.mkdir(parents=True)
    checkout_module.write_text("", encoding="utf-8")

    assert validator.is_beneath(checkout_module, venv) is False


def test_is_beneath_accepts_a_virtual_environment_import(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    site_packages_module = (
        venv / "Lib" / "site-packages" / "zeromodel" / "core" / "__init__.py"
    )
    site_packages_module.parent.mkdir(parents=True)
    site_packages_module.write_text("", encoding="utf-8")

    assert validator.is_beneath(site_packages_module, venv) is True


def test_is_beneath_accepts_a_posix_virtual_environment_import(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    site_packages_module = (
        venv
        / "lib"
        / "python3.11"
        / "site-packages"
        / "zeromodel"
        / "core"
        / "__init__.py"
    )
    site_packages_module.parent.mkdir(parents=True)
    site_packages_module.write_text("", encoding="utf-8")

    assert validator.is_beneath(site_packages_module, venv) is True

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = REPO_ROOT / ".vscode" / "settings.json"

SPEC = importlib.util.spec_from_file_location(
    "run_fast_tests", REPO_ROOT / "scripts" / "run_fast_tests.py"
)
assert SPEC is not None
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_vscode_settings_is_valid_json() -> None:
    json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def test_vscode_settings_ends_with_a_newline() -> None:
    assert SETTINGS_PATH.read_text(encoding="utf-8").endswith("\n")


def test_vscode_settings_uses_workspace_folder_variable() -> None:
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    for path in settings["python.analysis.extraPaths"]:
        assert path.startswith("${workspaceFolder}/")


def test_vscode_analysis_paths_point_at_all_six_package_src_roots() -> None:
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    extra_paths = {path.replace("${workspaceFolder}/", "") for path in settings["python.analysis.extraPaths"]}
    for package in ("core", "analysis", "observation", "vision", "video", "sqlalchemy"):
        assert f"packages/{package}/src" in extra_paths


def test_vscode_test_args_match_the_canonical_fast_suite_test_roots() -> None:
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    assert settings["python.testing.pytestArgs"] == runner.TEST_ROOTS

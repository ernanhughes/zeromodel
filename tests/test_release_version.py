from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path("scripts/release_version.py")
SPEC = importlib.util.spec_from_file_location("release_version", SCRIPT)
assert SPEC is not None
release_version = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = release_version
SPEC.loader.exec_module(release_version)


def test_version_file_is_the_canonical_release_authority() -> None:
    assert release_version.read_version() == "1.2.0"


def test_all_generated_version_mirrors_are_synchronized() -> None:
    assert release_version.version_sync_errors() == []


def test_workflows_do_not_embed_versioned_wheel_filenames() -> None:
    workflows = Path(".github/workflows").glob("*.yml")
    offenders = {
        workflow.as_posix(): release_version.HARDCODED_WORKFLOW_WHEEL_PATTERN.findall(
            workflow.read_text(encoding="utf-8")
        )
        for workflow in workflows
    }
    assert {path: matches for path, matches in offenders.items() if matches} == {}

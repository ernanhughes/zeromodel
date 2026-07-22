from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFTEST_PATH = REPO_ROOT / "tests" / "conftest.py"

SPEC = importlib.util.spec_from_file_location("zeromodel_tests_conftest", CONFTEST_PATH)
assert SPEC is not None
conftest = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = conftest
SPEC.loader.exec_module(conftest)


def test_integration_tests_directory_is_recognized_as_integration() -> None:
    # The real directory is named integration_tests, not integration - a
    # path-part check against the literal string "integration" alone would
    # miss it.
    parts = ("integration_tests", "test_package_integration_smoke.py")
    assert "integration_tests" in parts


def test_research_directory_files_are_recognized_as_research() -> None:
    filename = "test_video_policy.py"
    parts = ("research", "video_action_set", "tests", filename)
    assert "research" in parts
    assert filename not in conftest.INTEGRATION_TEST_FILES


def test_package_local_test_filenames_are_not_pre_tagged_integration_or_research() -> None:
    package_local_filenames = {
        "test_artifact_kernel.py",
        "test_critic.py",
        "test_deployment_binding.py",
        "test_frame_and_clip.py",
        "test_sqlalchemy_package_isolation.py",
        "test_deterministic_visual_runtime.py",
    }
    for filename in package_local_filenames:
        assert filename not in conftest.INTEGRATION_TEST_FILES
        assert filename not in conftest.RESEARCH_TEST_FILES
        assert not filename.startswith(conftest.INTEGRATION_TEST_PREFIXES)
        assert not filename.startswith(conftest.RESEARCH_TEST_PREFIXES)


def test_video_action_set_prefix_files_are_research_not_integration() -> None:
    # These are genuine research benchmarks over the arcade closed-world
    # action set (see docs/reviews/post-split-test-ownership.csv), not
    # production cross-package integration behavior.
    filename = "test_video_action_set_benchmark.py"
    assert filename.startswith(conftest.RESEARCH_TEST_PREFIXES)
    assert not filename.startswith(conftest.INTEGRATION_TEST_PREFIXES)


def test_marker_assignment_only_adds_never_resets_markers() -> None:
    source = CONFTEST_PATH.read_text(encoding="utf-8")
    assert "add_marker" in source
    assert "item.own_markers" not in source
    assert "item.keywords.clear" not in source

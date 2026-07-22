from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_MARKERS = {"fast", "slow", "integration", "external", "research"}

# Pytest/plugin builtins are not part of this repository's own taxonomy and
# are always registered by pytest itself.
BUILTIN_MARKERS = {
    "parametrize",
    "skip",
    "skipif",
    "xfail",
    "usefixtures",
    "filterwarnings",
    "no_cover",
    "tryfirst",
    "trylast",
}

MARKER_USAGE_PATTERN = re.compile(r"pytest\.mark\.([a-zA-Z_][a-zA-Z0-9_]*)")

SEARCH_ROOTS = (
    "tests",
    "integration_tests",
    "packages",
    "research",
)


def _registered_markers() -> set[str]:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"markers\s*=\s*\[(.*?)\]", pyproject, re.DOTALL)
    assert match is not None, "pyproject.toml has no [tool.pytest.ini_options] markers list"
    names = re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*):', match.group(1))
    return set(names)


def _markers_used_in_repository() -> set[str]:
    used: set[str] = set()
    for root_name in SEARCH_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            used.update(MARKER_USAGE_PATTERN.findall(text))
    return used - BUILTIN_MARKERS


def test_registered_markers_are_exactly_the_canonical_five_with_no_duplicates() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"markers\s*=\s*\[(.*?)\]", pyproject, re.DOTALL)
    assert match is not None
    names = re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*):', match.group(1))
    assert names == sorted(set(names), key=names.index), "duplicate marker registration found"
    assert set(names) == CANONICAL_MARKERS


def test_research_marker_is_registered() -> None:
    assert "research" in _registered_markers()


def test_every_marker_used_anywhere_in_the_repository_is_registered() -> None:
    used = _markers_used_in_repository()
    registered = _registered_markers()
    unregistered = used - registered
    assert unregistered == set(), f"unregistered markers in use: {sorted(unregistered)}"


def test_conftest_does_not_duplicate_marker_registration() -> None:
    # Markers are registered once, declaratively, in pyproject.toml.
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "addinivalue_line" not in conftest

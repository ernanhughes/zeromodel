from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

SPEC = importlib.util.spec_from_file_location(
    "check_architecture", SCRIPTS_ROOT / "check_architecture.py"
)
assert SPEC is not None
checker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def test_discover_modules_inspects_more_than_zero_production_modules() -> None:
    modules = checker.discover_modules()
    assert len(modules) > 100


def test_discover_modules_covers_all_six_source_roots() -> None:
    modules = checker.discover_modules()
    expected_prefixes = {
        "zeromodel.core",
        "zeromodel.analysis",
        "zeromodel.observation",
        "zeromodel.vision",
        "zeromodel.video",
        "zeromodel.persistence.sqlalchemy",
    }
    for prefix in expected_prefixes:
        assert any(
            module == prefix or module.startswith(f"{prefix}.") for module in modules
        ), f"no discovered module belongs to {prefix}"


def test_main_fails_rather_than_passing_vacuously_when_no_source_root_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_boundaries = tmp_path / "package-boundaries.toml"
    empty_boundaries.write_text(
        'schema_version = 1\nrelease_version = "0.0.0"\n\n[packages]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(checker, "BOUNDARIES", empty_boundaries)

    assert checker.discover_modules() == {}
    with pytest.raises(SystemExit):
        checker.main()


def test_cycle_violations_detects_a_deliberate_local_import_cycle() -> None:
    graph = {
        "zeromodel.core.a": {"zeromodel.core.b"},
        "zeromodel.core.b": {"zeromodel.core.a"},
    }

    violations = checker.cycle_violations(graph)

    assert violations
    assert any(v.rule == "local import cycle" for v in violations)


def test_cycle_violations_is_clean_for_an_acyclic_graph() -> None:
    graph = {"zeromodel.core.a": {"zeromodel.core.b"}, "zeromodel.core.b": set()}

    assert checker.cycle_violations(graph) == []


def test_forbidden_edge_violations_still_accepts_the_three_field_import_edge() -> None:
    # Regression guard: research/video_action_set/tests/test_video_benchmark_facade.py
    # and test_video_verification_closure_kernel.py construct checker.ImportEdge with
    # exactly (importer, imported, line) and call checker.forbidden_edge_violations()
    # directly. Their usage must keep working even though this module's discovery was
    # repointed at the workspace.
    edge = checker.ImportEdge(
        importer="zeromodel.core.a", imported="zeromodel.core.b", line=1
    )
    assert checker.forbidden_edge_violations([edge]) == []

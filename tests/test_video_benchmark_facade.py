import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

import research.benchmarks.video_action_set_benchmark as benchmark
from zeromodel.video.domains.video_action_set import (
    artifact_io,
    build_orchestration,
    mutation_filesystem,
    mutation_orchestration,
    verification_orchestration,
)
from research.video_action_set.video_action_set_cli import main


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_is_direct_alias_compatibility_surface() -> None:
    assert benchmark._write_json is artifact_io._write_json
    assert benchmark.freeze_benchmark is build_orchestration.freeze_benchmark
    assert (
        benchmark.verify_reference_instrument
        is verification_orchestration.verify_reference_instrument
    )
    assert (
        benchmark._apply_reference_mutation
        is mutation_filesystem._apply_reference_mutation
    )
    assert (
        benchmark.run_reference_mutation_audit
        is mutation_orchestration.run_reference_mutation_audit
    )
    assert benchmark.main is main
    assert (
        str(inspect.signature(benchmark.profile_runtime))
        == "(output_dir: 'Path', repo_root: 'Path', *, provider: 'str' = 'all', frame_count: 'int' = 8) -> 'dict[str, Any]'"
    )


def test_benchmark_facade_contains_no_function_implementations() -> None:
    path = REPO_ROOT / "zeromodel" / "video_action_set_benchmark.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert [node.name for node in tree.body if isinstance(node, ast.FunctionDef)] == [
        "_profiling_records",
        "build_split",
    ]


def test_build_split_adapter_preserves_benchmark_monkeypatch_seams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    materialize = object()
    prototypes = object()
    measure = object()
    observer = object()
    monkeypatch.setattr(benchmark, "_materialize_records", materialize)
    monkeypatch.setattr(benchmark, "canonical_prototypes", prototypes)
    monkeypatch.setattr(benchmark, "measure_record_collection", measure)

    def fake_build(
        split: str,
        output_dir: Path,
        repo_root: Path,
        *,
        progress_observer=None,
    ):
        assert progress_observer is observer
        assert build_orchestration._materialize_records is materialize
        assert build_orchestration.canonical_prototypes is prototypes
        assert build_orchestration.measure_record_collection is measure
        return {"split": split, "output": output_dir, "repo": repo_root}

    monkeypatch.setattr(build_orchestration, "build_split", fake_build)
    assert (
        benchmark.build_split(
            "selection",
            tmp_path,
            REPO_ROOT,
            progress_observer=observer,
        )["split"]
        == "selection"
    )


def _load_architecture_checker():
    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "stage7c_check_architecture", scripts / "check_architecture.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("importer", "imported"),
    [
        (
            "zeromodel.domains.video_action_set.reference_verification",
            "zeromodel.domains.video_action_set.artifact_io",
        ),
        (
            "zeromodel.domains.video_action_set.provider_measurement",
            "zeromodel.domains.video_action_set.build_orchestration",
        ),
        ("zeromodel.domains.video_action_set.report_rendering", "pathlib"),
        (
            "zeromodel.domains.video_action_set.build_orchestration",
            "zeromodel.domains.video_action_set.mutation_orchestration",
        ),
        ("zeromodel.video_action_set_cli", "zeromodel.video_action_set_benchmark"),
    ],
)
def test_stage7c_architecture_rejects_representative_edges(
    importer: str, imported: str
) -> None:
    checker = _load_architecture_checker()
    edge = checker.ImportEdge(importer=importer, imported=imported, line=7)
    assert checker.forbidden_edge_violations([edge])


def test_stage7c_architecture_allows_expected_direction() -> None:
    checker = _load_architecture_checker()
    edge = checker.ImportEdge(
        importer="zeromodel.domains.video_action_set.mutation_orchestration",
        imported="zeromodel.domains.video_action_set.verification_orchestration",
        line=7,
    )
    assert checker.forbidden_edge_violations([edge]) == []

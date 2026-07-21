"""Dependency rules for the Stage 7C video instrument shell."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class ImportEdgeLike(Protocol):
    importer: str
    imported: str
    line: int


ViolationFactory = Callable[[str, ImportEdgeLike], Any]

DOMAIN_PREFIX = "zeromodel.domains.video_action_set"
ARTIFACT_LAYOUT_MODULE = f"{DOMAIN_PREFIX}.artifact_layout"
ARTIFACT_IO_MODULE = f"{DOMAIN_PREFIX}.artifact_io"
REPORT_RENDERING_MODULE = f"{DOMAIN_PREFIX}.report_rendering"
MUTATION_FILESYSTEM_MODULE = f"{DOMAIN_PREFIX}.mutation_filesystem"
BUILD_ORCHESTRATION_MODULE = f"{DOMAIN_PREFIX}.build_orchestration"
VERIFICATION_GATE_ORCHESTRATION_MODULE = f"{DOMAIN_PREFIX}.verification_gates"
VERIFICATION_ORCHESTRATION_MODULE = f"{DOMAIN_PREFIX}.verification_orchestration"
MUTATION_ORCHESTRATION_MODULE = f"{DOMAIN_PREFIX}.mutation_orchestration"
CLI_MODULE = "zeromodel.video_action_set_cli"
BENCHMARK_MODULE = "zeromodel.video_action_set_benchmark"

STAGE7C_MODULES = {
    ARTIFACT_LAYOUT_MODULE,
    ARTIFACT_IO_MODULE,
    REPORT_RENDERING_MODULE,
    MUTATION_FILESYSTEM_MODULE,
    BUILD_ORCHESTRATION_MODULE,
    VERIFICATION_GATE_ORCHESTRATION_MODULE,
    VERIFICATION_ORCHESTRATION_MODULE,
    MUTATION_ORCHESTRATION_MODULE,
    CLI_MODULE,
}

_VERIFICATION_SHELL_MODULES = {
    VERIFICATION_GATE_ORCHESTRATION_MODULE,
    VERIFICATION_ORCHESTRATION_MODULE,
}
_PLANNING_EXECUTION_MODULES = {
    f"{DOMAIN_PREFIX}.control_histories",
    f"{DOMAIN_PREFIX}.family_intervention_planning",
    f"{DOMAIN_PREFIX}.episode_planning",
}
_MATERIALIZATION_EXECUTION_MODULES = {
    f"{DOMAIN_PREFIX}.episode_materialization",
    f"{DOMAIN_PREFIX}.materialization_kernels",
    f"{DOMAIN_PREFIX}.materialization_validation",
}
_MEASUREMENT_MODULES = {
    f"{DOMAIN_PREFIX}.provider_measurement",
    f"{DOMAIN_PREFIX}.runtime_profiling",
}


def video_instrument_shell_forbidden_edge_violations(
    edge: ImportEdgeLike,
    violation_factory: ViolationFactory,
) -> list[Any]:
    importer = edge.importer
    imported = edge.imported
    violations: list[Any] = []

    def reject(rule: str) -> None:
        violations.append(violation_factory(rule, edge))

    if (
        imported in STAGE7C_MODULES
        and importer.startswith("zeromodel")
        and importer not in STAGE7C_MODULES | {BENCHMARK_MODULE}
    ):
        reject("scientific modules must not import the Stage 7C instrument shell")

    if importer == ARTIFACT_LAYOUT_MODULE and imported.startswith("zeromodel"):
        reject("artifact layout must not import scientific execution modules")
    if (
        importer == ARTIFACT_IO_MODULE
        and imported.startswith("zeromodel")
        and imported != f"{DOMAIN_PREFIX}.canonical_json"
    ):
        reject(
            "artifact I/O may import only canonical JSON and artifact layout authorities"
        )
    if importer == REPORT_RENDERING_MODULE and imported in {
        "json",
        "os",
        "pathlib",
        "shutil",
        "sqlite3",
        "sqlalchemy",
        "subprocess",
        "tempfile",
        BENCHMARK_MODULE,
        CLI_MODULE,
        "zeromodel.runtime",
    }:
        reject("report rendering must remain pure and filesystem-free")
    if importer == MUTATION_FILESYSTEM_MODULE and imported in (
        _PLANNING_EXECUTION_MODULES
        | _MATERIALIZATION_EXECUTION_MODULES
        | _MEASUREMENT_MODULES
        | {
            f"{DOMAIN_PREFIX}.verification",
            MUTATION_ORCHESTRATION_MODULE,
            CLI_MODULE,
            BENCHMARK_MODULE,
        }
    ):
        reject("mutation filesystem must not own execution, closure, or dispatch")
    if importer == BUILD_ORCHESTRATION_MODULE and imported in (
        _VERIFICATION_SHELL_MODULES
        | {MUTATION_ORCHESTRATION_MODULE, CLI_MODULE, BENCHMARK_MODULE}
    ):
        reject(
            "build orchestration must not depend on verification, mutation, CLI, or benchmark"
        )
    if importer in _VERIFICATION_SHELL_MODULES and imported in {
        MUTATION_ORCHESTRATION_MODULE,
        CLI_MODULE,
        BENCHMARK_MODULE,
    }:
        reject(
            "verification orchestration must not depend on mutation, CLI, or benchmark"
        )
    if importer == MUTATION_ORCHESTRATION_MODULE and imported in {
        CLI_MODULE,
        BENCHMARK_MODULE,
    }:
        reject("mutation orchestration must not depend on CLI or benchmark")
    if importer == CLI_MODULE and imported == BENCHMARK_MODULE:
        reject("video action-set CLI must dispatch directly to orchestration modules")

    return violations


__all__ = [
    "ARTIFACT_IO_MODULE",
    "ARTIFACT_LAYOUT_MODULE",
    "BENCHMARK_MODULE",
    "BUILD_ORCHESTRATION_MODULE",
    "CLI_MODULE",
    "MUTATION_FILESYSTEM_MODULE",
    "MUTATION_ORCHESTRATION_MODULE",
    "REPORT_RENDERING_MODULE",
    "STAGE7C_MODULES",
    "VERIFICATION_GATE_ORCHESTRATION_MODULE",
    "VERIFICATION_ORCHESTRATION_MODULE",
    "video_instrument_shell_forbidden_edge_violations",
]

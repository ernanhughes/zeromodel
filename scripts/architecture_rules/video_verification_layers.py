from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from .video_instrument_shell import STAGE7C_MODULES


class ImportEdgeLike(Protocol):
    importer: str
    imported: str
    line: int


ViolationFactory = Callable[[str, ImportEdgeLike], Any]

DOMAIN_PREFIX = "zeromodel.domains.video_action_set"
BENCHMARK_MODULE = "zeromodel.video_action_set_benchmark"
REFERENCE_VERIFICATION_MODULE = f"{DOMAIN_PREFIX}.reference_verification"
EVIDENCE_AUDIT_MODULE = f"{DOMAIN_PREFIX}.evidence_audit"
MUTATION_AUDIT_MODULE = f"{DOMAIN_PREFIX}.mutation_audit"
MUTATION_MATRIX_MODULE = f"{DOMAIN_PREFIX}.mutation_matrix"
VERIFICATION_MODULE = f"{DOMAIN_PREFIX}.verification"

STAGE7B_MODULES = {
    REFERENCE_VERIFICATION_MODULE,
    EVIDENCE_AUDIT_MODULE,
    MUTATION_AUDIT_MODULE,
    MUTATION_MATRIX_MODULE,
    VERIFICATION_MODULE,
}

_STAGE7B_ORDER = {
    REFERENCE_VERIFICATION_MODULE: 0,
    EVIDENCE_AUDIT_MODULE: 1,
    MUTATION_AUDIT_MODULE: 2,
    MUTATION_MATRIX_MODULE: 3,
    VERIFICATION_MODULE: 4,
}

_EXTERNAL_INFRASTRUCTURE = {
    "csv",
    "json",
    "os",
    "pathlib",
    "shutil",
    "sqlite3",
    "sqlalchemy",
    "subprocess",
    "tempfile",
}
_FORBIDDEN_PREFIXES = {
    "zeromodel.db",
    "zeromodel.stores",
    "zeromodel.runtime",
    f"{DOMAIN_PREFIX}.engine",
    f"{DOMAIN_PREFIX}.facade",
    f"{DOMAIN_PREFIX}.identity_service",
    f"{DOMAIN_PREFIX}.episode_plan_service",
    f"{DOMAIN_PREFIX}.observation_service",
    f"{DOMAIN_PREFIX}.store",
    "zeromodel.cli",
    "zeromodel.reporting",
}
_EXECUTION_MODULES = {
    f"{DOMAIN_PREFIX}.episode_materialization",
    f"{DOMAIN_PREFIX}.materialization_kernels",
    f"{DOMAIN_PREFIX}.control_histories",
    f"{DOMAIN_PREFIX}.family_intervention_planning",
    f"{DOMAIN_PREFIX}.episode_planning",
}


def _is_under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _is_stage7b_infrastructure(module: str) -> bool:
    return module in _EXTERNAL_INFRASTRUCTURE or any(
        _is_under(module, prefix) for prefix in _FORBIDDEN_PREFIXES
    )


def stage7b_forbidden_edge_violations(
    edge: ImportEdgeLike,
    violation_factory: ViolationFactory,
) -> list[Any]:
    importer = edge.importer
    imported = edge.imported
    violations: list[Any] = []

    def reject(rule: str) -> None:
        violations.append(violation_factory(rule, edge))

    if (
        imported in STAGE7B_MODULES
        and importer not in STAGE7B_MODULES | STAGE7C_MODULES | {BENCHMARK_MODULE}
        and _is_under(importer, "zeromodel")
    ):
        reject("Stage 6 and lower scientific modules must not import Stage 7B")

    if importer not in STAGE7B_MODULES:
        return violations

    if imported == BENCHMARK_MODULE:
        reject("Stage 7B modules must not import the legacy benchmark")
    if _is_stage7b_infrastructure(imported):
        reject(
            "Stage 7B modules must not import filesystem, persistence, runtime, "
            "RMDTO service, reporting, or CLI layers"
        )
    if imported in _EXECUTION_MODULES:
        reject("Stage 7B modules must not import planning or materialization executors")
    if (
        imported in STAGE7B_MODULES
        and _STAGE7B_ORDER[imported] > _STAGE7B_ORDER[importer]
    ):
        reject("Stage 7B modules must follow reference-to-closure dependency order")
    return violations


__all__ = [
    "EVIDENCE_AUDIT_MODULE",
    "MUTATION_AUDIT_MODULE",
    "MUTATION_MATRIX_MODULE",
    "REFERENCE_VERIFICATION_MODULE",
    "STAGE7B_MODULES",
    "VERIFICATION_MODULE",
    "stage7b_forbidden_edge_violations",
]

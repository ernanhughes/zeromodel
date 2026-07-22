from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class ImportEdgeLike(Protocol):
    importer: str
    imported: str
    line: int


ViolationFactory = Callable[[str, ImportEdgeLike], Any]

DOMAIN_PREFIX = "zeromodel.domains.video_action_set"
BENCHMARK_MODULE = "zeromodel.video_action_set_benchmark"
DB_ORM_PREFIX = "zeromodel.db.orm"
DB_STORES_PREFIX = "zeromodel.db.stores"

PROVIDER_OBSERVATION_BOUNDARY_MODULE = f"{DOMAIN_PREFIX}.provider_observation_boundary"
MATERIALIZATION_REACHABILITY_MODULE = f"{DOMAIN_PREFIX}.materialization_reachability"
REACHABILITY_COMPOSITION_MODULE = f"{DOMAIN_PREFIX}.reachability_composition"
PROVIDER_MEASUREMENT_MODULE = f"{DOMAIN_PREFIX}.provider_measurement"
RUNTIME_PROFILING_MODULE = f"{DOMAIN_PREFIX}.runtime_profiling"

STAGE7A_MODULES = {
    REACHABILITY_COMPOSITION_MODULE,
    PROVIDER_MEASUREMENT_MODULE,
    RUNTIME_PROFILING_MODULE,
}

LOWER_LAYER_MODULES = {
    f"{DOMAIN_PREFIX}.canonical_json",
    f"{DOMAIN_PREFIX}.contracts",
    f"{DOMAIN_PREFIX}.pixel_digest",
    f"{DOMAIN_PREFIX}.transformations",
    f"{DOMAIN_PREFIX}.arcade_observation",
    f"{DOMAIN_PREFIX}.observation_universe",
    f"{DOMAIN_PREFIX}.observation_provenance_dto",
    f"{DOMAIN_PREFIX}.observation_legacy_adapters",
    f"{DOMAIN_PREFIX}.observation_provenance",
    f"{DOMAIN_PREFIX}.observation_replay",
    f"{DOMAIN_PREFIX}.episode_families",
    f"{DOMAIN_PREFIX}.frame_family_kernels",
    f"{DOMAIN_PREFIX}.family_provenance",
    f"{DOMAIN_PREFIX}.family_validation",
    f"{DOMAIN_PREFIX}.control_histories",
    f"{DOMAIN_PREFIX}.family_intervention_planning",
    f"{DOMAIN_PREFIX}.episode_planning",
    f"{DOMAIN_PREFIX}.materialization_reachability",
    f"{DOMAIN_PREFIX}.materialization_kernels",
    f"{DOMAIN_PREFIX}.episode_materialization",
    f"{DOMAIN_PREFIX}.materialization_validation",
    f"{DOMAIN_PREFIX}.dto",
    f"{DOMAIN_PREFIX}.observation_common",
    f"{DOMAIN_PREFIX}.observation_dto",
    f"{DOMAIN_PREFIX}.observation_materialization",
    f"{DOMAIN_PREFIX}.provider_observation_dto",
    "zeromodel.video_complete_row_evidence",
    "zeromodel.video_prospective_providers",
    "zeromodel.video_discriminative_evidence",
    "zeromodel.video_discriminative_joint_evidence",
    "zeromodel.video_local_correlation",
}

PLANNING_MODULES = {
    f"{DOMAIN_PREFIX}.control_histories",
    f"{DOMAIN_PREFIX}.family_intervention_planning",
    f"{DOMAIN_PREFIX}.episode_planning",
}

MATERIALIZATION_MODULES = {
    f"{DOMAIN_PREFIX}.materialization_kernels",
    f"{DOMAIN_PREFIX}.episode_materialization",
    f"{DOMAIN_PREFIX}.materialization_validation",
}

PROVIDER_SCORING_MODULES = {
    "zeromodel.video_complete_row_evidence",
    "zeromodel.video_prospective_providers",
    "zeromodel.video_discriminative_evidence",
    "zeromodel.video_discriminative_joint_evidence",
    "zeromodel.video_local_correlation",
}

VERIFICATION_OR_AUDIT_PREFIXES = {
    f"{DOMAIN_PREFIX}.evidence_audit",
    f"{DOMAIN_PREFIX}.mutation_audit",
    f"{DOMAIN_PREFIX}.mutation_matrix",
    f"{DOMAIN_PREFIX}.reference_verification",
    f"{DOMAIN_PREFIX}.verification",
    "zeromodel.video_action_equivalence",
    "zeromodel.video_action_set.mutation_audit",
    "zeromodel.video_action_set.verification",
}

FILESYSTEM_OR_DATABASE_MODULES = {
    "json",
    "pathlib",
    "sqlite3",
}


def _is_under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _is_sqlalchemy_import(module: str) -> bool:
    return module == "sqlalchemy" or module.startswith("sqlalchemy.")


def _is_forbidden_infrastructure_import(module: str) -> bool:
    return (
        _is_under(module, DB_ORM_PREFIX)
        or _is_under(module, DB_STORES_PREFIX)
        or _is_under(module, "zeromodel.db.session")
        or _is_under(module, "zeromodel.stores")
        or module == "zeromodel.runtime"
        or _is_sqlalchemy_import(module)
        or module in FILESYSTEM_OR_DATABASE_MODULES
    )


def _is_verification_or_audit_import(module: str) -> bool:
    return any(_is_under(module, prefix) for prefix in VERIFICATION_OR_AUDIT_PREFIXES)


def stage7a_forbidden_edge_violations(
    edge: ImportEdgeLike,
    violation_factory: ViolationFactory,
) -> list[Any]:
    violations: list[Any] = []
    importer = edge.importer
    imported = edge.imported

    def reject(rule: str) -> None:
        violations.append(violation_factory(rule, edge))

    if importer in LOWER_LAYER_MODULES and imported in STAGE7A_MODULES:
        reject("lower video action-set modules must not import Stage 7A modules")

    if importer == REACHABILITY_COMPOSITION_MODULE and (
        imported == BENCHMARK_MODULE
        or imported in {PROVIDER_MEASUREMENT_MODULE, RUNTIME_PROFILING_MODULE}
        or imported in PROVIDER_SCORING_MODULES
        or imported in PLANNING_MODULES
        or imported in MATERIALIZATION_MODULES
        or _is_forbidden_infrastructure_import(imported)
        or _is_verification_or_audit_import(imported)
    ):
        reject(
            "reachability_composition must not import benchmark, provider measurement, profiling, scoring, planning, materialization execution, verification, mutation audit, persistence, runtime, or filesystem layers"
        )

    if importer == RUNTIME_PROFILING_MODULE and (
        imported == BENCHMARK_MODULE
        or imported == REACHABILITY_COMPOSITION_MODULE
        or imported in PLANNING_MODULES
        or imported in MATERIALIZATION_MODULES
        or _is_forbidden_infrastructure_import(imported)
        or _is_verification_or_audit_import(imported)
    ):
        reject(
            "runtime_profiling must not import benchmark, reachability composition, planning, materialization, verification, mutation audit, persistence, runtime, or filesystem/report layers"
        )

    if importer == PROVIDER_OBSERVATION_BOUNDARY_MODULE and imported in {
        PROVIDER_MEASUREMENT_MODULE,
        RUNTIME_PROFILING_MODULE,
    }:
        reject("provider_observation_boundary must remain below Stage 7A modules")

    if importer == MATERIALIZATION_REACHABILITY_MODULE and imported in STAGE7A_MODULES:
        reject("materialization_reachability must remain below Stage 7A modules")

    return violations


__all__ = ["stage7a_forbidden_edge_violations"]

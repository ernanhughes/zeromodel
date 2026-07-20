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
MATERIALIZATION_KERNELS_MODULE = f"{DOMAIN_PREFIX}.materialization_kernels"
EPISODE_MATERIALIZATION_MODULE = f"{DOMAIN_PREFIX}.episode_materialization"
MATERIALIZATION_VALIDATION_MODULE = f"{DOMAIN_PREFIX}.materialization_validation"

STAGE6_MATERIALIZATION_MODULES = {
    PROVIDER_OBSERVATION_BOUNDARY_MODULE,
    MATERIALIZATION_REACHABILITY_MODULE,
    MATERIALIZATION_KERNELS_MODULE,
    EPISODE_MATERIALIZATION_MODULE,
    MATERIALIZATION_VALIDATION_MODULE,
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
    f"{DOMAIN_PREFIX}.dto",
}

PLAN_PRODUCER_MODULES = {
    f"{DOMAIN_PREFIX}.family_intervention_planning",
    f"{DOMAIN_PREFIX}.episode_planning",
}

SCORING_OR_REPORT_MODULES = {
    "zeromodel.video_complete_row_evidence",
    "zeromodel.video_prospective_providers",
}

FILESYSTEM_OR_DATABASE_MODULES = {
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


def stage6_forbidden_edge_violations(
    edge: ImportEdgeLike,
    violation_factory: ViolationFactory,
) -> list[Any]:
    violations: list[Any] = []
    importer = edge.importer
    imported = edge.imported

    def reject(rule: str) -> None:
        violations.append(violation_factory(rule, edge))

    if importer in STAGE6_MATERIALIZATION_MODULES and (
        imported == BENCHMARK_MODULE
        or _is_forbidden_infrastructure_import(imported)
        or imported in SCORING_OR_REPORT_MODULES
    ):
        reject(
            "Stage 6 materialization modules must not import benchmark, persistence, runtime, filesystem, or scoring/report layers"
        )
    if importer in LOWER_LAYER_MODULES and imported in STAGE6_MATERIALIZATION_MODULES:
        reject(
            "lower video action-set modules must not import Stage 6 materialization modules"
        )
    if importer == PROVIDER_OBSERVATION_BOUNDARY_MODULE and (
        imported
        in STAGE6_MATERIALIZATION_MODULES - {PROVIDER_OBSERVATION_BOUNDARY_MODULE}
        or imported in PLAN_PRODUCER_MODULES
        or imported.startswith("zeromodel.video_")
    ):
        reject(
            "provider_observation_boundary must not import planning, materialization, or scoring modules"
        )
    if importer == MATERIALIZATION_REACHABILITY_MODULE and (
        imported in STAGE6_MATERIALIZATION_MODULES
        or imported in SCORING_OR_REPORT_MODULES
        or imported in PLAN_PRODUCER_MODULES
    ):
        reject(
            "materialization_reachability must not import planning, materialization peers, or scoring modules"
        )
    if importer == MATERIALIZATION_KERNELS_MODULE and imported in {
        MATERIALIZATION_REACHABILITY_MODULE,
        EPISODE_MATERIALIZATION_MODULE,
        MATERIALIZATION_VALIDATION_MODULE,
        *PLAN_PRODUCER_MODULES,
        *SCORING_OR_REPORT_MODULES,
    }:
        reject(
            "materialization_kernels must not import reachability, execution, validation, planning, or scoring modules"
        )
    if importer == EPISODE_MATERIALIZATION_MODULE and imported in PLAN_PRODUCER_MODULES:
        reject(
            "episode_materialization must consume sealed plans and must not import planning modules"
        )
    if (
        importer == EPISODE_MATERIALIZATION_MODULE
        and imported == MATERIALIZATION_VALIDATION_MODULE
    ):
        reject("episode_materialization must not import materialization_validation")
    if (
        importer == MATERIALIZATION_VALIDATION_MODULE
        and imported in PLAN_PRODUCER_MODULES
    ):
        reject(
            "materialization_validation must consume plans and must not import planning modules"
        )
    return violations


__all__ = [
    "stage6_forbidden_edge_violations",
]

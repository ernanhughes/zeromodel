from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class ImportEdgeLike(Protocol):
    importer: str
    imported: str
    line: int


ViolationFactory = Callable[[str, ImportEdgeLike], Any]


VIDEO_ACTION_SET_PREFIX = "zeromodel.video_action_set"
VIDEO_ACTION_SET_BENCHMARK_MODULE = "zeromodel.video_action_set_benchmark"
DOMAIN_VIDEO_ACTION_SET_PREFIX = "zeromodel.domains.video_action_set"
DB_ORM_PREFIX = "zeromodel.db.orm"
DB_STORES_PREFIX = "zeromodel.db.stores"
ARCADE_OBSERVATION_MODULE = "zeromodel.domains.video_action_set.arcade_observation"
CANONICAL_JSON_MODULE = "zeromodel.domains.video_action_set.canonical_json"
CONTRACTS_MODULE = "zeromodel.domains.video_action_set.contracts"
OBSERVATION_LEGACY_ADAPTERS_MODULE = (
    "zeromodel.domains.video_action_set.observation_legacy_adapters"
)
OBSERVATION_PROVENANCE_DTO_MODULE = (
    "zeromodel.domains.video_action_set.observation_provenance_dto"
)
OBSERVATION_PROVENANCE_MODULE = (
    "zeromodel.domains.video_action_set.observation_provenance"
)
OBSERVATION_REPLAY_MODULE = "zeromodel.domains.video_action_set.observation_replay"
OBSERVATION_UNIVERSE_MODULE = "zeromodel.domains.video_action_set.observation_universe"
PIXEL_DIGEST_MODULE = "zeromodel.domains.video_action_set.pixel_digest"
TRANSFORMATIONS_MODULE = "zeromodel.domains.video_action_set.transformations"
EPISODE_FAMILIES_MODULE = "zeromodel.domains.video_action_set.episode_families"
FRAME_FAMILY_KERNELS_MODULE = "zeromodel.domains.video_action_set.frame_family_kernels"
FAMILY_PROVENANCE_MODULE = "zeromodel.domains.video_action_set.family_provenance"
FAMILY_VALIDATION_MODULE = "zeromodel.domains.video_action_set.family_validation"
CONTROL_HISTORIES_MODULE = "zeromodel.domains.video_action_set.control_histories"
FAMILY_INTERVENTION_PLANNING_MODULE = (
    "zeromodel.domains.video_action_set.family_intervention_planning"
)
EPISODE_PLANNING_MODULE = "zeromodel.domains.video_action_set.episode_planning"
FAMILY_SCIENCE_MODULES = {
    EPISODE_FAMILIES_MODULE,
    FRAME_FAMILY_KERNELS_MODULE,
    FAMILY_PROVENANCE_MODULE,
    FAMILY_VALIDATION_MODULE,
}
EPISODE_PLANNING_MODULES = {
    CONTROL_HISTORIES_MODULE,
    FAMILY_INTERVENTION_PLANNING_MODULE,
    EPISODE_PLANNING_MODULE,
}
VIDEO_OBSERVATION_KERNEL_MODULES = {
    ARCADE_OBSERVATION_MODULE,
    OBSERVATION_PROVENANCE_MODULE,
    OBSERVATION_REPLAY_MODULE,
    OBSERVATION_UNIVERSE_MODULE,
    PIXEL_DIGEST_MODULE,
    TRANSFORMATIONS_MODULE,
}
LOWER_OBSERVATION_MODULES = {
    CANONICAL_JSON_MODULE,
    CONTRACTS_MODULE,
    PIXEL_DIGEST_MODULE,
    TRANSFORMATIONS_MODULE,
    ARCADE_OBSERVATION_MODULE,
    OBSERVATION_UNIVERSE_MODULE,
    OBSERVATION_PROVENANCE_DTO_MODULE,
    OBSERVATION_LEGACY_ADAPTERS_MODULE,
}
VIDEO_ACTION_SET_ORCHESTRATION_MODULES = {
    "zeromodel.domains.video_action_set.episode_plan_service",
    "zeromodel.domains.video_action_set.identity_service",
    "zeromodel.domains.video_action_set.observation_service",
    "zeromodel.domains.video_action_set.engine",
    "zeromodel.domains.video_action_set.facade",
    "zeromodel.runtime",
}
VIDEO_ACTION_SET_PURE_DOMAIN_MODULES = {
    ARCADE_OBSERVATION_MODULE,
    CANONICAL_JSON_MODULE,
    CONTROL_HISTORIES_MODULE,
    CONTRACTS_MODULE,
    EPISODE_FAMILIES_MODULE,
    EPISODE_PLANNING_MODULE,
    FAMILY_INTERVENTION_PLANNING_MODULE,
    FAMILY_PROVENANCE_MODULE,
    FAMILY_VALIDATION_MODULE,
    FRAME_FAMILY_KERNELS_MODULE,
    "zeromodel.domains.video_action_set.dto",
    "zeromodel.domains.video_action_set.episode_plan_service",
    "zeromodel.domains.video_action_set.observation_common",
    "zeromodel.domains.video_action_set.observation_dto",
    OBSERVATION_LEGACY_ADAPTERS_MODULE,
    "zeromodel.domains.video_action_set.observation_materialization",
    OBSERVATION_PROVENANCE_DTO_MODULE,
    OBSERVATION_PROVENANCE_MODULE,
    OBSERVATION_REPLAY_MODULE,
    "zeromodel.domains.video_action_set.observation_service",
    OBSERVATION_UNIVERSE_MODULE,
    PIXEL_DIGEST_MODULE,
    "zeromodel.domains.video_action_set.provider_observation_dto",
    TRANSFORMATIONS_MODULE,
}
VIDEO_ACTION_SET_POLICY_MODULES = {
    f"{VIDEO_ACTION_SET_PREFIX}.contracts",
    f"{VIDEO_ACTION_SET_PREFIX}.mutation_audit",
    f"{VIDEO_ACTION_SET_PREFIX}.verification",
}


def is_video_action_set_runtime_module(module: str) -> bool:
    prefix = f"{VIDEO_ACTION_SET_PREFIX}."
    if not module.startswith(prefix):
        return False
    if module in VIDEO_ACTION_SET_POLICY_MODULES:
        return False
    suffix = module[len(prefix) :]
    return not (
        suffix.startswith("contracts.")
        or suffix.startswith("mutation_audit.")
        or suffix.startswith("verification.")
    )


def is_module_under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def is_sqlalchemy_import(module: str) -> bool:
    return module == "sqlalchemy" or module.startswith("sqlalchemy.")


def legacy_forbidden_edge_violations(
    edge: ImportEdgeLike,
    violation_factory: ViolationFactory,
) -> list[Any]:
    violations: list[Any] = []
    if edge.imported == "tests" or edge.imported.startswith("tests."):
        violations.append(violation_factory("zeromodel/ must not import tests.*", edge))
    if edge.importer == "zeromodel.artifact" and (
        edge.imported.startswith("zeromodel.video_")
        or edge.imported == VIDEO_ACTION_SET_PREFIX
        or edge.imported.startswith(f"{VIDEO_ACTION_SET_PREFIX}.")
    ):
        violations.append(
            violation_factory(
                "zeromodel.artifact must not import video research modules",
                edge,
            )
        )
    if edge.importer == f"{VIDEO_ACTION_SET_PREFIX}.contracts" and (
        edge.imported == VIDEO_ACTION_SET_PREFIX
        or edge.imported.startswith(f"{VIDEO_ACTION_SET_PREFIX}.")
    ):
        violations.append(
            violation_factory(
                "video_action_set/contracts.py must not import implementation modules",
                edge,
            )
        )
    if is_video_action_set_runtime_module(edge.importer) and (
        edge.imported == f"{VIDEO_ACTION_SET_PREFIX}.verification"
        or edge.imported.startswith(f"{VIDEO_ACTION_SET_PREFIX}.verification.")
        or edge.imported == f"{VIDEO_ACTION_SET_PREFIX}.mutation_audit"
        or edge.imported.startswith(f"{VIDEO_ACTION_SET_PREFIX}.mutation_audit.")
    ):
        violations.append(
            violation_factory(
                "runtime modules must not depend on verification or mutation_audit",
                edge,
            )
        )
    return violations


def transformation_kernel_forbidden_edge_violations(
    edge: ImportEdgeLike,
    violation_factory: ViolationFactory,
) -> list[Any]:
    violations: list[Any] = []
    if edge.importer == PIXEL_DIGEST_MODULE and edge.imported == TRANSFORMATIONS_MODULE:
        violations.append(
            violation_factory(
                "pixel_digest must not import transformations",
                edge,
            )
        )
    if edge.importer == PIXEL_DIGEST_MODULE and edge.imported in {
        ARCADE_OBSERVATION_MODULE,
        OBSERVATION_UNIVERSE_MODULE,
    }:
        violations.append(
            violation_factory(
                "pixel_digest must not import arcade_observation or observation_universe",
                edge,
            )
        )
    if edge.importer == TRANSFORMATIONS_MODULE and edge.imported in {
        ARCADE_OBSERVATION_MODULE,
        OBSERVATION_UNIVERSE_MODULE,
    }:
        violations.append(
            violation_factory(
                "transformations must not import arcade_observation or observation_universe",
                edge,
            )
        )
    if (
        edge.importer == ARCADE_OBSERVATION_MODULE
        and edge.imported == OBSERVATION_UNIVERSE_MODULE
    ):
        violations.append(
            violation_factory(
                "arcade_observation must not import observation_universe",
                edge,
            )
        )
    if edge.importer in VIDEO_OBSERVATION_KERNEL_MODULES and (
        is_module_under(edge.imported, DB_ORM_PREFIX)
        or is_module_under(edge.imported, DB_STORES_PREFIX)
        or is_module_under(edge.imported, "zeromodel.db.session")
        or is_module_under(edge.imported, "zeromodel.stores")
        or edge.imported == "zeromodel.runtime"
        or is_sqlalchemy_import(edge.imported)
    ):
        violations.append(
            violation_factory(
                "video observation kernel modules must not import persistence or runtime layers",
                edge,
            )
        )
    if edge.importer in VIDEO_OBSERVATION_KERNEL_MODULES and edge.imported == (
        VIDEO_ACTION_SET_BENCHMARK_MODULE
    ):
        violations.append(
            violation_factory(
                "video observation kernel modules must not import legacy benchmark facade",
                edge,
            )
        )
    return violations


def observation_provenance_replay_forbidden_edge_violations(
    edge: ImportEdgeLike,
    violation_factory: ViolationFactory,
) -> list[Any]:
    violations: list[Any] = []
    if (
        edge.importer == OBSERVATION_PROVENANCE_MODULE
        and edge.imported == OBSERVATION_REPLAY_MODULE
    ):
        violations.append(
            violation_factory(
                "observation_provenance must not import observation_replay",
                edge,
            )
        )
    if (
        edge.importer == OBSERVATION_REPLAY_MODULE
        and edge.imported == OBSERVATION_PROVENANCE_MODULE
    ):
        violations.append(
            violation_factory(
                "observation_replay must not import observation_provenance",
                edge,
            )
        )
    if (
        edge.importer
        in {
            OBSERVATION_PROVENANCE_MODULE,
            OBSERVATION_REPLAY_MODULE,
        }
        and edge.imported == VIDEO_ACTION_SET_BENCHMARK_MODULE
    ):
        violations.append(
            violation_factory(
                "observation provenance/replay modules must not import legacy benchmark",
                edge,
            )
        )
    if edge.importer in {
        OBSERVATION_PROVENANCE_MODULE,
        OBSERVATION_REPLAY_MODULE,
    } and (
        is_module_under(edge.imported, DB_ORM_PREFIX)
        or is_module_under(edge.imported, DB_STORES_PREFIX)
        or is_module_under(edge.imported, "zeromodel.db.session")
        or is_module_under(edge.imported, "zeromodel.stores")
        or edge.imported == "zeromodel.runtime"
        or is_sqlalchemy_import(edge.imported)
    ):
        violations.append(
            violation_factory(
                "observation provenance/replay modules must not import persistence or runtime layers",
                edge,
            )
        )
    if edge.importer in LOWER_OBSERVATION_MODULES and edge.imported in {
        OBSERVATION_PROVENANCE_MODULE,
        OBSERVATION_REPLAY_MODULE,
    }:
        violations.append(
            violation_factory(
                "lower observation modules must not import observation provenance/replay",
                edge,
            )
        )
    return violations


def family_science_forbidden_edge_violations(
    edge: ImportEdgeLike,
    violation_factory: ViolationFactory,
) -> list[Any]:
    violations: list[Any] = []
    if edge.importer in FAMILY_SCIENCE_MODULES and (
        edge.imported == VIDEO_ACTION_SET_BENCHMARK_MODULE
        or is_module_under(edge.imported, DB_ORM_PREFIX)
        or is_module_under(edge.imported, DB_STORES_PREFIX)
        or is_module_under(edge.imported, "zeromodel.db.session")
        or is_module_under(edge.imported, "zeromodel.stores")
        or edge.imported == "zeromodel.runtime"
        or is_sqlalchemy_import(edge.imported)
    ):
        violations.append(
            violation_factory(
                "family science modules must not import legacy benchmark, persistence, or runtime",
                edge,
            )
        )
    if (
        edge.importer in FAMILY_SCIENCE_MODULES
        and edge.imported in EPISODE_PLANNING_MODULES
    ):
        violations.append(
            violation_factory("family science must not import planning", edge)
        )
    if edge.importer == EPISODE_FAMILIES_MODULE and edge.imported in {
        ARCADE_OBSERVATION_MODULE,
        FAMILY_PROVENANCE_MODULE,
        FAMILY_VALIDATION_MODULE,
        FRAME_FAMILY_KERNELS_MODULE,
        OBSERVATION_PROVENANCE_MODULE,
        OBSERVATION_REPLAY_MODULE,
        OBSERVATION_UNIVERSE_MODULE,
        TRANSFORMATIONS_MODULE,
    }:
        violations.append(
            violation_factory(
                "episode_families must not import higher family or observation kernels",
                edge,
            )
        )
    if edge.importer == FRAME_FAMILY_KERNELS_MODULE and edge.imported in {
        FAMILY_PROVENANCE_MODULE,
        FAMILY_VALIDATION_MODULE,
        OBSERVATION_PROVENANCE_MODULE,
        OBSERVATION_REPLAY_MODULE,
        "zeromodel.domains.video_action_set.dto",
        "zeromodel.domains.video_action_set.episode_plan_service",
    }:
        violations.append(
            violation_factory(
                "frame_family_kernels must not import provenance, replay, validation, or planning",
                edge,
            )
        )
    if edge.importer == FAMILY_PROVENANCE_MODULE and edge.imported in {
        FAMILY_VALIDATION_MODULE,
        OBSERVATION_REPLAY_MODULE,
    }:
        violations.append(
            violation_factory(
                "family_provenance must not import replay or validation",
                edge,
            )
        )
    if edge.importer == FAMILY_VALIDATION_MODULE and edge.imported in {
        FAMILY_PROVENANCE_MODULE,
        FRAME_FAMILY_KERNELS_MODULE,
    }:
        violations.append(
            violation_factory(
                "family_validation must not import family provenance or frame kernels",
                edge,
            )
        )
    if (
        edge.importer == OBSERVATION_REPLAY_MODULE
        and edge.imported in FAMILY_SCIENCE_MODULES
    ):
        violations.append(
            violation_factory(
                "observation_replay must not import family implementation modules",
                edge,
            )
        )
    if (
        edge.importer in LOWER_OBSERVATION_MODULES
        and edge.imported in FAMILY_SCIENCE_MODULES
    ):
        violations.append(
            violation_factory(
                "lower observation modules must not import family implementation modules",
                edge,
            )
        )
    return violations


def episode_planning_forbidden_edge_violations(
    edge: ImportEdgeLike,
    violation_factory: ViolationFactory,
) -> list[Any]:
    violations: list[Any] = []
    importer = edge.importer
    imported = edge.imported

    def reject(rule: str) -> None:
        violations.append(violation_factory(rule, edge))

    if (
        importer in VIDEO_OBSERVATION_KERNEL_MODULES
        and imported in EPISODE_PLANNING_MODULES
    ):
        reject("video observation kernels must not import episode planning modules")
    if importer in EPISODE_PLANNING_MODULES and (
        imported == VIDEO_ACTION_SET_BENCHMARK_MODULE
        or is_module_under(imported, DB_ORM_PREFIX)
        or is_module_under(imported, DB_STORES_PREFIX)
        or is_module_under(imported, "zeromodel.db.session")
        or is_module_under(imported, "zeromodel.stores")
        or imported == "zeromodel.runtime"
        or is_sqlalchemy_import(imported)
    ):
        reject("planning must not import benchmark/persistence/runtime")
    if importer in EPISODE_PLANNING_MODULES and imported in {
        ARCADE_OBSERVATION_MODULE,
        FAMILY_PROVENANCE_MODULE,
        FAMILY_VALIDATION_MODULE,
        OBSERVATION_PROVENANCE_MODULE,
        OBSERVATION_REPLAY_MODULE,
        PIXEL_DIGEST_MODULE,
    }:
        reject("planning must not import rendering/replay/provenance/pixels")
    if importer == CONTROL_HISTORIES_MODULE and imported in {
        FAMILY_INTERVENTION_PLANNING_MODULE,
        EPISODE_PLANNING_MODULE,
    }:
        reject("control_histories must not import higher episode planning modules")
    if (
        importer == FAMILY_INTERVENTION_PLANNING_MODULE
        and imported == EPISODE_PLANNING_MODULE
    ):
        reject("family_intervention_planning must not import episode_planning")
    return violations


def rmdto_forbidden_edge_violations(
    edge: ImportEdgeLike,
    violation_factory: ViolationFactory,
) -> list[Any]:
    violations: list[Any] = []

    def reject(rule: str) -> None:
        violations.append(violation_factory(rule, edge))

    if is_module_under(edge.importer, DOMAIN_VIDEO_ACTION_SET_PREFIX) and (
        is_module_under(edge.imported, DB_ORM_PREFIX)
        or is_module_under(edge.imported, DB_STORES_PREFIX)
        or is_module_under(edge.imported, "zeromodel.db.session")
        or is_sqlalchemy_import(edge.imported)
    ):
        reject("video_action_set domain modules must not import persistence layers")
    if is_module_under(edge.importer, DOMAIN_VIDEO_ACTION_SET_PREFIX) and (
        edge.imported == VIDEO_ACTION_SET_BENCHMARK_MODULE
    ):
        reject("video_action_set domain modules must not import legacy benchmark")
    if edge.importer in VIDEO_ACTION_SET_PURE_DOMAIN_MODULES and (
        is_module_under(edge.imported, "zeromodel.stores")
        or edge.imported == "zeromodel.runtime"
    ):
        reject(
            "video_action_set DTO, contracts, and Services must not import runtime layers"
        )
    if edge.importer == "zeromodel.runtime" and (
        is_module_under(edge.imported, "zeromodel.db")
        or is_sqlalchemy_import(edge.imported)
    ):
        violations.append(
            violation_factory(
                "zeromodel.runtime must not import database or SQLAlchemy modules",
                edge,
            )
        )
    if edge.importer == "zeromodel.matrix_blob" and (
        is_module_under(edge.imported, DOMAIN_VIDEO_ACTION_SET_PREFIX)
        or is_module_under(edge.imported, "zeromodel.db")
        or is_module_under(edge.imported, "zeromodel.stores")
        or edge.imported == "zeromodel.runtime"
    ):
        violations.append(
            violation_factory(
                "zeromodel.matrix_blob must remain independent of video_action_set, stores, database, and runtime",
                edge,
            )
        )
    if is_module_under(edge.importer, DB_ORM_PREFIX) and (
        edge.imported in VIDEO_ACTION_SET_ORCHESTRATION_MODULES
        or edge.imported == VIDEO_ACTION_SET_BENCHMARK_MODULE
    ):
        violations.append(
            violation_factory(
                "ORM modules must not import Services, Engines, Facades, Runtime, or legacy benchmark",
                edge,
            )
        )
    if is_module_under(edge.importer, DB_STORES_PREFIX) and (
        edge.imported in VIDEO_ACTION_SET_ORCHESTRATION_MODULES
        or edge.imported == VIDEO_ACTION_SET_BENCHMARK_MODULE
    ):
        violations.append(
            violation_factory(
                "database Store implementations must not import orchestration or legacy benchmark layers",
                edge,
            )
        )
    return violations


def video_science_forbidden_edge_violations(
    edge: ImportEdgeLike,
    violation_factory: ViolationFactory,
) -> list[Any]:
    violations: list[Any] = []
    violations.extend(legacy_forbidden_edge_violations(edge, violation_factory))
    violations.extend(
        transformation_kernel_forbidden_edge_violations(edge, violation_factory)
    )
    violations.extend(
        observation_provenance_replay_forbidden_edge_violations(edge, violation_factory)
    )
    violations.extend(family_science_forbidden_edge_violations(edge, violation_factory))
    violations.extend(
        episode_planning_forbidden_edge_violations(edge, violation_factory)
    )
    violations.extend(rmdto_forbidden_edge_violations(edge, violation_factory))
    return violations


__all__ = ["video_science_forbidden_edge_violations"]

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    BENCHMARK_VERSION,
    EPISODE_PLAN_VERSION,
    GENERATOR_VERSION,
    SEED_DERIVATION_VERSION,
)
from zeromodel.video.domains.video_action_set.control_histories import (
    control_source_rows,
)
from zeromodel.video.domains.video_action_set.dto import (
    BenchmarkIdentityDTO,
    EpisodePlanDTO,
)
from zeromodel.video.domains.video_action_set.episode_families import (
    denominator_class,
    episode_disposition,
    family_contract,
    family_schedule,
)
from zeromodel.video.domains.video_action_set.family_intervention_planning import (
    conflicting_splice_source_rows,
    derived_seed,
    family_intervention_plan,
    frame_count_for_plan,
    secondary_row_for_splice,
)
from zeromodel.video.domains.video_action_set.transformations import (
    _transformation_parameters,
)


def final_observation_provenance(split: str) -> dict[str, Any]:
    if split == "final":
        return {
            "materialization_status": "prospective_materialization_prohibited",
            "observation_payload_included": False,
            "provenance": "sealed_plan_only",
        }
    return {
        "materialization_status": "materialized",
        "observation_payload_included": True,
        "provenance": "in_memory_generation",
    }


def frame_plans(
    identity: BenchmarkIdentityDTO,
    *,
    split: str,
    ordinal: int,
    family_label: str,
    mutation_kind: str | None,
    frame_count: int,
    concrete_episode_seed_digest: str,
) -> list[dict[str, Any]]:
    schedule = family_schedule()
    schedule_digest = canonical_sha256({"family_schedule": list(schedule)})
    frames = []
    for frame_index in range(frame_count):
        frame_seed = derived_seed(
            identity,
            split=split,
            ordinal=ordinal,
            namespace="frame_identity",
            parent_identities=(
                ("concrete_episode_seed", concrete_episode_seed_digest),
                ("frame_index", str(frame_index)),
            ),
        )
        transform_family_seed = derived_seed(
            identity,
            split=split,
            ordinal=ordinal,
            namespace="transformation_family",
            parent_identities=(
                ("frame_identity", frame_seed["seed_digest"]),
                ("family_schedule_digest", schedule_digest),
                ("episode_family", family_label),
            ),
        )
        if family_label == "frame_invalid":
            transformation_family = "exact"
        else:
            transformation_family = schedule[
                transform_family_seed["seed_int64"] % len(schedule)
            ]
        transform_parameter_seed = derived_seed(
            identity,
            split=split,
            ordinal=ordinal,
            namespace="transformation_parameters",
            parent_identities=(
                ("transformation_family_seed", transform_family_seed["seed_digest"]),
                ("transformation_family", transformation_family),
                ("frame_index", str(frame_index)),
            ),
        )
        transition_choice_seed = derived_seed(
            identity,
            split=split,
            ordinal=ordinal,
            namespace="transition_choice",
            parent_identities=(
                ("frame_identity", frame_seed["seed_digest"]),
                ("frame_index", str(frame_index)),
            ),
        )
        temporal_mutation_seed = derived_seed(
            identity,
            split=split,
            ordinal=ordinal,
            namespace="temporal_mutation_choice",
            parent_identities=(
                ("frame_identity", frame_seed["seed_digest"]),
                ("temporal_mutation", mutation_kind or "none"),
            ),
        )
        parameters = _transformation_parameters(
            transformation_family, transform_parameter_seed["seed_int64"]
        )
        frames.append(
            {
                "frame_index": frame_index,
                "frame_seed_identity": frame_seed["seed_digest"],
                "transformation_family_seed_identity": transform_family_seed[
                    "seed_digest"
                ],
                "transformation_family": transformation_family,
                "transformation_seed_identity": transform_parameter_seed["seed_digest"],
                "transformation_seed": transform_parameter_seed["seed_int64"],
                "transformation_parameter_digest": parameters["parameter_digest"],
                "transformation_parameters": parameters,
                "transition_choice_seed_identity": transition_choice_seed[
                    "seed_digest"
                ],
                "transition_choice_seed": transition_choice_seed["seed_int64"],
                "temporal_mutation_seed_identity": temporal_mutation_seed[
                    "seed_digest"
                ],
                "temporal_mutation_kind": mutation_kind
                if family_label == "temporal_negative"
                else None,
            }
        )
    return frames


def _validate_episode_plan_inputs(
    *,
    source_row_id: str,
    row_actions: Mapping[str, str],
    mutation_kind: str | None,
    secondary_row_id: str | None,
) -> None:
    if source_row_id not in row_actions:
        raise VPMValidationError(
            "episode source row is absent from the policy universe"
        )
    if secondary_row_id is not None and secondary_row_id not in row_actions:
        raise VPMValidationError(
            "episode secondary row is absent from the policy universe"
        )
    if mutation_kind != "conflicting_action_splice" and secondary_row_id is not None:
        raise VPMValidationError(
            "secondary row is only admissible for conflicting-action splices"
        )
    if mutation_kind == "conflicting_action_splice":
        if secondary_row_id is None:
            raise VPMValidationError(
                "conflicting-action splice requires a secondary row"
            )
        if row_actions[secondary_row_id] == row_actions[source_row_id]:
            raise VPMValidationError(
                "conflicting-action splice secondary row must govern a different action"
            )


def _seed_node(
    identity: BenchmarkIdentityDTO,
    *,
    split: str,
    ordinal: int,
    namespace: str,
    parents: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    return derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace=namespace,
        parent_identities=parents,
    )


def _source_seed_lineage(
    identity: BenchmarkIdentityDTO,
    *,
    split: str,
    ordinal: int,
    family_label: str,
    family_ordinal: int,
    source_row_id: str,
    mutation_kind: str | None = None,
    secondary_row_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    split_seed = _seed_node(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="split_identity",
        parents=(
            ("benchmark_version", BENCHMARK_VERSION),
            ("generator_version", GENERATOR_VERSION),
        ),
    )
    ordinal_seed = _seed_node(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="episode_ordinal",
        parents=(
            ("split_identity", split_seed["seed_digest"]),
            ("family_ordinal", str(family_ordinal)),
        ),
    )
    family_seed = _seed_node(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="episode_family",
        parents=(
            ("episode_ordinal", ordinal_seed["seed_digest"]),
            ("family_label", family_label),
            ("mutation_kind", mutation_kind or "none"),
        ),
    )
    source_seed = _seed_node(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="source_row_choice",
        parents=(
            ("episode_family", family_seed["seed_digest"]),
            ("source_row_id", source_row_id),
        ),
    )
    secondary_seed = _seed_node(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="secondary_splice_row_choice",
        parents=(
            ("source_row_choice", source_seed["seed_digest"]),
            ("secondary_row_id", secondary_row_id or "none"),
        ),
    )
    return {
        "split_identity": split_seed,
        "episode_ordinal": ordinal_seed,
        "episode_family": family_seed,
        "source_row_choice": source_seed,
        "secondary_splice_row_choice": secondary_seed,
    }


def _transformation_seed_lineage(
    identity: BenchmarkIdentityDTO,
    *,
    split: str,
    ordinal: int,
    family_label: str,
    mutation_kind: str | None,
    source_lineage: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    split_seed = source_lineage["split_identity"]
    ordinal_seed = source_lineage["episode_ordinal"]
    family_seed = source_lineage["episode_family"]
    source_seed = source_lineage["source_row_choice"]
    secondary_seed = source_lineage["secondary_splice_row_choice"]
    transformation_family_seed = _seed_node(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="transformation_family",
        parents=(
            ("secondary_splice_row_choice", secondary_seed["seed_digest"]),
            (
                "family_schedule_digest",
                canonical_sha256({"family_schedule": list(family_schedule())}),
            ),
        ),
    )
    transformation_parameter_seed = _seed_node(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="transformation_parameters",
        parents=(
            ("transformation_family", transformation_family_seed["seed_digest"]),
            ("frame_count", str(frame_count_for_plan(split, family_label))),
        ),
    )
    temporal_mutation_seed = _seed_node(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="temporal_mutation_choice",
        parents=(
            ("transformation_parameters", transformation_parameter_seed["seed_digest"]),
            ("temporal_mutation", mutation_kind or "none"),
        ),
    )
    episode_seed = _seed_node(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="concrete_episode_seed",
        parents=(
            ("split_identity", split_seed["seed_digest"]),
            ("episode_ordinal", ordinal_seed["seed_digest"]),
            ("episode_family", family_seed["seed_digest"]),
            ("source_row_choice", source_seed["seed_digest"]),
            ("secondary_splice_row_choice", secondary_seed["seed_digest"]),
            ("transformation_family", transformation_family_seed["seed_digest"]),
            ("transformation_parameters", transformation_parameter_seed["seed_digest"]),
            ("temporal_mutation_choice", temporal_mutation_seed["seed_digest"]),
        ),
    )
    return {
        "transformation_family": transformation_family_seed,
        "transformation_parameters": transformation_parameter_seed,
        "temporal_mutation_choice": temporal_mutation_seed,
        "concrete_episode_seed": episode_seed,
    }


def _episode_seed_lineage(
    identity: BenchmarkIdentityDTO,
    *,
    split: str,
    ordinal: int,
    family_label: str,
    family_ordinal: int,
    source_row_id: str,
    mutation_kind: str | None = None,
    secondary_row_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    source_lineage = _source_seed_lineage(
        identity,
        split=split,
        ordinal=ordinal,
        family_label=family_label,
        family_ordinal=family_ordinal,
        source_row_id=source_row_id,
        mutation_kind=mutation_kind,
        secondary_row_id=secondary_row_id,
    )
    transformation_lineage = _transformation_seed_lineage(
        identity,
        split=split,
        ordinal=ordinal,
        family_label=family_label,
        mutation_kind=mutation_kind,
        source_lineage=source_lineage,
    )
    return source_lineage | transformation_lineage


def _episode_id_seed(
    identity: BenchmarkIdentityDTO,
    *,
    split: str,
    ordinal: int,
    episode_seed: Mapping[str, Any],
    contract: Mapping[str, Any],
    intervention: Mapping[str, Any],
    disposition: str,
    denominator: str,
    source_row_id: str,
    secondary_row_id: str | None,
) -> dict[str, Any]:
    return _seed_node(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="concrete_episode_id",
        parents=(
            ("concrete_episode_seed", episode_seed["seed_digest"]),
            ("family_contract", contract["family_version"]),
            ("family_intervention", intervention["intervention_digest"]),
            ("episode_disposition", disposition),
            ("denominator_class", denominator),
            ("source_row_id", source_row_id),
            ("secondary_row_id", secondary_row_id or "none"),
        ),
    )


def _episode_plan_payload(fields: Mapping[str, Any]) -> dict[str, Any]:
    split = str(fields["split"])
    family_label = str(fields["family_label"])
    episode_id_seed = fields["episode_id_seed"]
    episode_id = f"{split}:{family_label}:{episode_id_seed['seed_digest'].removeprefix('sha256:')[:16]}"
    return {
        "version": EPISODE_PLAN_VERSION,
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "episode_id": episode_id,
        "split": split,
        "ordinal": int(fields["ordinal"]),
        "family_label": family_label,
        "family_ordinal": int(fields["family_ordinal"]),
        "episode_family": family_label,
        "episode_disposition": fields["disposition"],
        "denominator_class": fields["denominator"],
        "final_observation_provenance": fields["provenance"],
        "mutation_kind": fields["mutation_kind"],
        "source_row_id": fields["source_row_id"],
        "secondary_row_id": fields["secondary_row_id"],
        "family_contract": fields["contract"],
        "family_intervention": fields["intervention"],
        "derived_seed_identity": fields["episode_seed"]["seed_digest"],
        "episode_seed": fields["episode_seed"]["seed_int64"],
        "frame_count": fields["frame_count"],
        "seed_lineage": fields["lineage"] | {"concrete_episode_id": episode_id_seed},
        "frame_plans": fields["frames"],
    }


def make_episode_plan(
    identity: BenchmarkIdentityDTO,
    *,
    split: str,
    ordinal: int,
    family_label: str,
    family_ordinal: int,
    source_row_id: str,
    row_actions: Mapping[str, str],
    mutation_kind: str | None = None,
    secondary_row_id: str | None = None,
) -> dict[str, Any]:
    _validate_episode_plan_inputs(
        source_row_id=source_row_id,
        row_actions=row_actions,
        mutation_kind=mutation_kind,
        secondary_row_id=secondary_row_id,
    )
    lineage = _episode_seed_lineage(
        identity,
        split=split,
        ordinal=ordinal,
        family_label=family_label,
        family_ordinal=family_ordinal,
        source_row_id=source_row_id,
        mutation_kind=mutation_kind,
        secondary_row_id=secondary_row_id,
    )
    episode_seed = lineage["concrete_episode_seed"]
    frame_count = frame_count_for_plan(split, family_label)
    frames = frame_plans(
        identity,
        split=split,
        ordinal=ordinal,
        family_label=family_label,
        mutation_kind=mutation_kind,
        frame_count=frame_count,
        concrete_episode_seed_digest=episode_seed["seed_digest"],
    )
    contract = family_contract(family_label, mutation_kind)
    intervention = family_intervention_plan(
        identity=identity,
        split=split,
        ordinal=ordinal,
        family_label=family_label,
        mutation_kind=mutation_kind,
        source_row_id=source_row_id,
        secondary_row_id=secondary_row_id,
        row_ids=tuple(row_actions.keys()),
        row_actions=row_actions,
    )
    disposition = episode_disposition(family_label, mutation_kind)
    denominator = denominator_class(family_label, mutation_kind)
    episode_id_seed = _episode_id_seed(
        identity,
        split=split,
        ordinal=ordinal,
        episode_seed=episode_seed,
        contract=contract,
        intervention=intervention,
        disposition=disposition,
        denominator=denominator,
        source_row_id=source_row_id,
        secondary_row_id=secondary_row_id,
    )
    plan = _episode_plan_payload(
        {
            "split": split,
            "ordinal": ordinal,
            "family_label": family_label,
            "family_ordinal": family_ordinal,
            "disposition": disposition,
            "denominator": denominator,
            "provenance": final_observation_provenance(split),
            "mutation_kind": mutation_kind,
            "source_row_id": source_row_id,
            "secondary_row_id": secondary_row_id,
            "contract": contract,
            "intervention": intervention,
            "episode_seed": episode_seed,
            "episode_id_seed": episode_id_seed,
            "frame_count": frame_count,
            "frames": frames,
            "lineage": lineage,
        }
    )
    dto = EpisodePlanDTO.from_dict(plan | {"plan_digest": canonical_sha256(plan)})
    return dto.to_dict()


def episode_plans_for_split(
    identity: BenchmarkIdentityDTO,
    split: str,
    row_ids: list[str],
    row_actions: Mapping[str, str],
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []

    def add(
        family_label: str,
        family_ordinal: int,
        row_id: str,
        *,
        mutation_kind: str | None = None,
    ) -> None:
        secondary = None
        if mutation_kind == "conflicting_action_splice":
            secondary = secondary_row_for_splice(row_ids, row_actions, row_id)
        plans.append(
            make_episode_plan(
                identity,
                split=split,
                ordinal=len(plans),
                family_label=family_label,
                family_ordinal=family_ordinal,
                source_row_id=row_id,
                row_actions=row_actions,
                mutation_kind=mutation_kind,
                secondary_row_id=secondary,
            )
        )

    if split == "development":
        for index, row_id in enumerate(row_ids):
            add("valid", index, row_id)
        return plans
    if split == "calibration":
        for index, row_id in enumerate(row_ids):
            add("valid", index, row_id)
        return plans
    if split in {"selection", "final"}:
        for index, row_id in enumerate(row_ids):
            add("valid", index, row_id)
        splice_rows = conflicting_splice_source_rows(row_ids, row_actions, 28)
        for index, row_id in enumerate(splice_rows):
            add(
                "frame_invalid",
                index,
                row_id,
                mutation_kind="conflicting_action_splice",
            )
        for index, row_id in enumerate(row_ids[28:56]):
            add(
                "frame_invalid",
                28 + index,
                row_id,
                mutation_kind="critical_evidence_corruption",
            )
        temporal_rows = row_ids[:56]
        kinds = (
            "reordered_frames",
            "stale_repeated_frame",
            "impossible_transition",
            "declared_gap_or_unknown_action",
        )
        for group_index, kind in enumerate(kinds):
            for index, row_id in enumerate(
                temporal_rows[group_index * 14 : (group_index + 1) * 14]
            ):
                add(
                    "temporal_negative",
                    group_index * 14 + index,
                    row_id,
                    mutation_kind=kind,
                )
        control_rows = control_source_rows(
            row_ids,
            count=28,
            frame_count=frame_count_for_plan(split, "information_control"),
        )
        for index, row_id in enumerate(control_rows):
            add("information_control", index, row_id)
        return plans
    raise VPMValidationError("unsupported split")


def episode_ids_by_family(plans: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "valid": [],
        "frame_invalid": [],
        "temporal_negative": [],
        "information_control": [],
    }
    for plan in plans:
        grouped[str(plan["family_label"])].append(str(plan["episode_id"]))
    return grouped


def validate_episode_plan(
    identity: BenchmarkIdentityDTO,
    plan: Mapping[str, Any],
    row_actions: Mapping[str, str],
) -> None:
    try:
        EpisodePlanDTO.from_dict(plan)
    except VPMValidationError as exc:
        raise VPMValidationError(
            "episode plan is inconsistent with declared seed lineage or identity"
        ) from exc
    expected = make_episode_plan(
        identity,
        split=str(plan["split"]),
        ordinal=int(plan["ordinal"]),
        family_label=str(plan["family_label"]),
        family_ordinal=int(plan["family_ordinal"]),
        source_row_id=str(plan["source_row_id"]),
        row_actions=row_actions,
        mutation_kind=plan.get("mutation_kind"),
        secondary_row_id=plan.get("secondary_row_id"),
    )
    if dict(plan) != expected:
        raise VPMValidationError(
            "episode plan is inconsistent with declared seed lineage or identity"
        )


def validate_episode_plan_collection(
    identity: BenchmarkIdentityDTO,
    plans_by_split: Mapping[str, list[dict[str, Any]]],
    row_actions: Mapping[str, str],
) -> None:
    seen: dict[str, str] = {}
    for split, plans in plans_by_split.items():
        for plan in plans:
            episode_id = str(plan["episode_id"])
            if plan["split"] != split:
                raise VPMValidationError(
                    "episode plan split does not match containing split"
                )
            if episode_id in seen:
                if seen[episode_id] != split:
                    raise VPMValidationError(
                        "episode identity reassigned to another split"
                    )
                raise VPMValidationError("duplicate concrete episode identity")
            seen[episode_id] = split
            validate_episode_plan(identity, plan, row_actions)


__all__ = [
    "derived_seed",
    "episode_ids_by_family",
    "episode_plans_for_split",
    "final_observation_provenance",
    "frame_count_for_plan",
    "frame_plans",
    "make_episode_plan",
    "validate_episode_plan",
    "validate_episode_plan_collection",
]

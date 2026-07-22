from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from zeromodel.video.arcade_policy import (
    ACTIONS,
    ShooterConfig,
    compile_policy_artifact,
    parse_state_row_id,
    render_state_frame,
)
from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.video.domains.video_action_set.arcade_observation import render_row_frame
from zeromodel.video.domains.video_action_set.dto import BenchmarkIdentityDTO
from zeromodel.video.domains.video_action_set.episode_families import expected_frame_disposition
from zeromodel.video.domains.video_action_set.family_provenance import (
    conflicting_splice_operation_chain,
    critical_corruption_operation_chain,
    impossible_transition_operation_chain,
    information_control_operation_chain,
    stale_repeat_operation_chain,
)
from zeromodel.video.domains.video_action_set.frame_family_kernels import apply_conflicting_splice, apply_critical_corruption
from zeromodel.video.domains.video_action_set.materialization_kernels import apply_frame_plan, frame_descriptor
from zeromodel.video.domains.video_action_set.materialization_reachability import next_materialized_row, tile_edge
from zeromodel.video.domains.video_action_set.observation_provenance import (
    gap_event_operation_chain,
    valid_frame_operation_chain,
)
from zeromodel.video.domains.video_action_set.pixel_digest import array_digest, pixel_digest
from zeromodel.video.domains.video_action_set.provider_observation_boundary import refresh_provider_observation_metadata


def valid_episode(
    *,
    plan: Mapping[str, Any],
    identity: BenchmarkIdentityDTO,
    reachability_tile: Mapping[str, Any],
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    split = str(plan["split"])
    episode_id = str(plan["episode_id"])
    episode_seed = int(plan["episode_seed"])
    current = str(plan["source_row_id"])
    frames = []
    for idx, frame_plan in enumerate(plan["frame_plans"]):
        tank, target, cooldown = parse_state_row_id(str(current))
        base = render_state_frame(tank, target, cooldown, width=config.width)
        family = str(frame_plan["transformation_family"])
        pixels, transformation_trace = apply_frame_plan(base, frame_plan)
        next_row, action, choice_index, reachability_trace = next_materialized_row(
            lookup,
            current,
            choice_seed=int(frame_plan["transition_choice_seed"]),
            config=config,
            reachability_tile=reachability_tile,
        )
        descriptor = frame_descriptor(
            split=split,
            episode_id=episode_id,
            frame_index=idx,
            row_id=current,
            expected_action=action,
            actual_action=action,
            family=family,
            pixels=pixels,
            expected_disposition="valid",
            episode_family=str(plan["episode_family"]),
            episode_disposition=str(plan["episode_disposition"]),
            frame_disposition=expected_frame_disposition(
                str(plan["family_label"]),
                plan.get("mutation_kind"),
                idx,
                plan.get("family_intervention"),
            ),
            denominator_class=str(plan["denominator_class"]),
            metadata={
                "episode_seed": episode_seed,
                "seed_digest": identity.seed_digest,
                "derived_seed_identity": plan["derived_seed_identity"],
                "episode_plan_digest": plan["plan_digest"],
                "frame_seed_identity": frame_plan["frame_seed_identity"],
                "frame_transform_seed": frame_plan["transformation_seed"],
                "frame_transform_seed_identity": frame_plan[
                    "transformation_seed_identity"
                ],
                "transition_choice_seed": frame_plan["transition_choice_seed"],
                "transition_choice_seed_identity": frame_plan[
                    "transition_choice_seed_identity"
                ],
                "transformation_family": family,
                "transformation_parameters": dict(
                    frame_plan["transformation_parameters"]
                ),
                "source_observation_digest": transformation_trace[
                    "source_observation_digest"
                ],
                "transformed_observation_digest": transformation_trace[
                    "transformed_observation_digest"
                ],
                "transformation_parameter_digest": transformation_trace[
                    "transformation_parameter_digest"
                ],
                "transformation_changed_pixel_count": transformation_trace[
                    "changed_pixel_count"
                ],
                "observation_operation_chain": valid_frame_operation_chain(
                    current, frame_plan["transformation_parameters"]
                ),
                "transition_choice_index": choice_index,
                "next_row": next_row,
                "reachability_trace": reachability_trace,
            },
        )
        frames.append(descriptor | {"pixels": pixels})
        current = next_row
    return frames


def _invalid_episode_metadata(
    *,
    plan: Mapping[str, Any],
    identity: BenchmarkIdentityDTO,
    frame_plan: Mapping[str, Any],
    transformation_trace: Mapping[str, Any],
    intervention_trace: Mapping[str, Any],
    row_id: str,
    base_action: str,
    other_row: Any,
    lookup: VPMPolicyLookup,
) -> dict[str, Any]:
    kind = str(plan["mutation_kind"])
    intervention = plan["family_intervention"]
    competitor_action = None if other_row is None else lookup.choose(str(other_row))
    operation_chain = (
        conflicting_splice_operation_chain(
            primary_row_id=row_id,
            secondary_row_id=str(other_row),
            primary_action_id=base_action,
            secondary_action_id=lookup.choose(str(other_row)),
            primary_transformation_parameters=frame_plan["transformation_parameters"],
            mask_manifest=intervention["splice_mask"],
        )
        if kind == "conflicting_action_splice"
        else critical_corruption_operation_chain(
            row_id,
            frame_plan["transformation_parameters"],
            intervention["critical_coordinates"],
        )
    )
    return {
        "episode_seed": int(plan["episode_seed"]),
        "seed_digest": identity.seed_digest,
        "derived_seed_identity": plan["derived_seed_identity"],
        "episode_plan_digest": plan["plan_digest"],
        "frame_seed_identity": frame_plan["frame_seed_identity"],
        "frame_transform_seed": frame_plan["transformation_seed"],
        "frame_transform_seed_identity": frame_plan["transformation_seed_identity"],
        "transformation_parameters": dict(frame_plan["transformation_parameters"]),
        "source_observation_digest": transformation_trace["source_observation_digest"],
        "transformed_observation_digest": transformation_trace[
            "transformed_observation_digest"
        ],
        "transformation_parameter_digest": transformation_trace[
            "transformation_parameter_digest"
        ],
        "transformation_changed_pixel_count": transformation_trace[
            "changed_pixel_count"
        ],
        "family_contract": plan["family_contract"],
        "family_intervention": intervention,
        "family_intervention_trace": intervention_trace,
        "observation_operation_chain": operation_chain,
        "source_row_id": row_id,
        "source_action_id": base_action,
        "competitor_row_id": None if other_row is None else str(other_row),
        "competitor_action_id": competitor_action,
        "collision_audit": "distinguishable_invalid",
    }


def invalid_episode(
    *,
    plan: Mapping[str, Any],
    identity: BenchmarkIdentityDTO,
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    split = str(plan["split"])
    kind = str(plan["mutation_kind"])
    row_id = str(plan["source_row_id"])
    episode_id = str(plan["episode_id"])
    tank, target, cooldown = parse_state_row_id(str(row_id))
    base = render_state_frame(tank, target, cooldown, width=config.width)
    base_action = lookup.choose(str(row_id))
    other_row = plan.get("secondary_row_id")
    other = None
    if other_row is not None:
        other_tank, other_target, other_cooldown = parse_state_row_id(str(other_row))
        other = render_state_frame(
            other_tank, other_target, other_cooldown, width=config.width
        )
    frames = []
    intervention = plan["family_intervention"]
    for idx, frame_plan in enumerate(plan["frame_plans"]):
        source_pixels, transformation_trace = apply_frame_plan(base, frame_plan)
        intervention_trace: dict[str, Any]
        if kind == "conflicting_action_splice":
            if other is None:
                raise VPMValidationError(
                    "conflicting-action splice plan missing secondary frame"
                )
            pixels, intervention_trace = apply_conflicting_splice(
                primary_pixels=source_pixels,
                secondary_pixels=other,
                primary_row_id=row_id,
                secondary_row_id=str(other_row),
                primary_action_id=base_action,
                secondary_action_id=lookup.choose(str(other_row)),
                mask_manifest=intervention["splice_mask"],
            )
        else:
            pixels, intervention_trace = apply_critical_corruption(
                source_pixels, intervention["critical_coordinates"]
            )
        descriptor = frame_descriptor(
            split=split,
            episode_id=episode_id,
            frame_index=idx,
            row_id=None,
            expected_action=None,
            actual_action=base_action,
            family=kind,
            pixels=pixels,
            expected_disposition="distinguishable_invalid_input",
            episode_family=str(plan["episode_family"]),
            episode_disposition=str(plan["episode_disposition"]),
            frame_disposition=expected_frame_disposition(
                str(plan["family_label"]),
                plan.get("mutation_kind"),
                idx,
                plan.get("family_intervention"),
            ),
            denominator_class=str(plan["denominator_class"]),
            metadata=_invalid_episode_metadata(
                plan=plan,
                identity=identity,
                frame_plan=frame_plan,
                transformation_trace=transformation_trace,
                intervention_trace=intervention_trace,
                row_id=row_id,
                base_action=base_action,
                other_row=other_row,
                lookup=lookup,
            ),
        )
        frames.append(descriptor | {"pixels": pixels})
    return frames


def _materialize_reordered_frames(
    valid: list[dict[str, Any]], intervention: Mapping[str, Any]
) -> list[dict[str, Any]]:
    order = list(intervention["materialized_order"])
    if order == list(intervention["original_order"]):
        raise VPMValidationError("reordered family requires non-identity order")
    if sorted(order) != list(intervention["original_order"]):
        raise VPMValidationError("reordered family must be a complete permutation")
    frames = [valid[i] for i in order]
    for idx, item in enumerate(frames):
        item["sequence_number"] = idx
        item["frame_disposition"] = "temporally_reordered_frame_payload"
        item["metadata"]["original_frame_index"] = order[idx]
        item["metadata"]["materialized_order"] = order
        item["metadata"]["sequence_rule"] = intervention["sequence_rule"]
    return frames


def _materialize_stale_repeated_frame(
    valid: list[dict[str, Any]], intervention: Mapping[str, Any]
) -> list[dict[str, Any]]:
    repeat = intervention["stale_repeat"]
    source_index = int(repeat["source_frame_index"])
    destination_index = int(repeat["destination_frame_index"])
    original_destination_digest = valid[destination_index]["observation_pixel_digest"]
    replacement_digest = valid[source_index]["observation_pixel_digest"]
    if original_destination_digest == replacement_digest:
        raise VPMValidationError("stale repeat requires an actual payload replacement")
    destination_before = dict(
        valid[destination_index], metadata=dict(valid[destination_index]["metadata"])
    )
    replacement_source = dict(
        valid[source_index], metadata=dict(valid[source_index]["metadata"])
    )
    repeat_metadata = {
        **repeat,
        "original_destination_digest": original_destination_digest,
        "replacement_digest": replacement_digest,
    }
    valid[destination_index]["pixels"] = np.array(
        valid[source_index]["pixels"], copy=True
    )
    valid[destination_index]["observation_pixel_digest"] = replacement_digest
    valid[destination_index]["frame_disposition"] = "stale_repeated_frame_payload"
    valid[destination_index]["metadata"]["stale_repeat"] = repeat_metadata
    valid[destination_index]["metadata"]["observation_operation_chain"] = (
        stale_repeat_operation_chain(
            destination_before, replacement_source, repeat_metadata
        )
    )
    refresh_provider_observation_metadata(valid[destination_index])
    return valid


def _materialize_impossible_transition(
    valid: list[dict[str, Any]],
    intervention: Mapping[str, Any],
    reachability_tile: Mapping[str, Any],
    *,
    config: ShooterConfig,
) -> list[dict[str, Any]]:
    transition_plan = intervention["impossible_transition"]
    source_index = int(transition_plan["source_frame_index"])
    destination_index = int(transition_plan["destination_frame_index"])
    source_row = str(transition_plan["source_row_id"])
    destination_row = str(transition_plan["destination_row_id"])
    edge = tile_edge(
        reachability_tile, source_row, str(transition_plan["source_action_id"])
    )
    if destination_row in set(edge["reachable_row_ids"]):
        raise VPMValidationError("impossible transition destination is reachable")
    destination_before = dict(
        valid[destination_index], metadata=dict(valid[destination_index]["metadata"])
    )
    destination_pixels = render_row_frame(destination_row, config=config)
    valid[destination_index]["pixels"] = destination_pixels
    valid[destination_index]["observation_pixel_digest"] = pixel_digest(
        destination_pixels
    )
    valid[destination_index]["frame_disposition"] = (
        "unreachable_destination_frame_payload"
    )
    valid[destination_index]["expected_row"] = destination_row
    valid[destination_index]["expected_action"] = transition_plan[
        "destination_action_id"
    ]
    transition_metadata = {
        **transition_plan,
        "source_observation_digest": valid[source_index]["observation_pixel_digest"],
        "destination_observation_digest": valid[destination_index][
            "observation_pixel_digest"
        ],
        "reachability_tile_digest": reachability_tile["tile_digest"],
        "consulted_edge": {
            "source_row_id": edge["source_row_id"],
            "action_id": edge["action_id"],
            "reachable_row_ids": list(edge["reachable_row_ids"]),
        },
        "pairwise_reachability_status": "impossible",
    }
    valid[destination_index]["metadata"]["impossible_transition"] = transition_metadata
    valid[destination_index]["metadata"]["observation_operation_chain"] = (
        impossible_transition_operation_chain(destination_before, transition_metadata)
    )
    valid[destination_index]["metadata"]["next_row"] = "impossible_transition_marker"
    refresh_provider_observation_metadata(valid[destination_index])
    return valid


def _materialize_declared_gap(
    valid: list[dict[str, Any]], intervention: Mapping[str, Any]
) -> list[dict[str, Any]]:
    gap = intervention["gap_event"]
    position = int(gap["position"])
    valid[position]["pixels"] = None
    valid[position]["event_type"] = "gap_unknown"
    valid[position]["frame_disposition"] = "declared_gap_or_unknown_action"
    valid[position]["expected_row"] = None
    valid[position]["expected_action"] = None
    valid[position]["actual_executed_action"] = None
    valid[position]["action_known"] = False
    valid[position]["gap_declaration"] = "declared_gap"
    valid[position]["observation_pixel_digest"] = None
    valid[position]["metadata"]["gap_event"] = gap
    valid[position]["metadata"]["event_identity"] = gap["event_id"]
    valid[position]["metadata"]["observation_operation_chain"] = (
        gap_event_operation_chain(gap)
    )
    refresh_provider_observation_metadata(valid[position])
    return valid


def temporal_negative_episode(
    *,
    plan: Mapping[str, Any],
    identity: BenchmarkIdentityDTO,
    reachability_tile: Mapping[str, Any],
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    valid = valid_episode(
        plan=plan,
        identity=identity,
        reachability_tile=reachability_tile,
        config=config,
    )
    kind = str(plan["mutation_kind"])
    intervention = plan["family_intervention"]
    for item in valid:
        item["episode_family"] = "temporal_negative"
        item["episode_disposition"] = str(plan["episode_disposition"])
        item["frame_disposition"] = "valid_frame_payload"
        item["denominator_class"] = str(plan["denominator_class"])
        item["metadata"]["family_contract"] = plan["family_contract"]
        item["metadata"]["family_intervention"] = intervention
        item["metadata"]["sequence_digest"] = intervention["sequence_digest"]
    if kind == "reordered_frames":
        return _materialize_reordered_frames(valid, intervention)
    if kind == "stale_repeated_frame":
        return _materialize_stale_repeated_frame(valid, intervention)
    if kind == "impossible_transition":
        return _materialize_impossible_transition(
            valid, intervention, reachability_tile, config=config
        )
    if kind == "declared_gap_or_unknown_action":
        return _materialize_declared_gap(valid, intervention)
    raise VPMValidationError("unsupported temporal-negative kind")


def control_episode(
    *,
    plan: Mapping[str, Any],
    identity: BenchmarkIdentityDTO,
    reachability_tile: Mapping[str, Any],
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    split = str(plan["split"])
    episode_id = str(plan["episode_id"])
    current_row = str(plan["source_row_id"])
    current_pixels = render_row_frame(current_row, config=config)
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    current_action = lookup.choose(current_row)
    intervention = plan["family_intervention"]
    control_group = intervention["control_group"]
    histories = list(control_group["hidden_source_histories"])
    if len({history["normalized_causal_tuple_digest"] for history in histories}) < 2:
        raise VPMValidationError(
            "information controls require grounded causal ambiguity"
        )
    frames = []
    for idx, frame_plan in enumerate(plan["frame_plans"]):
        pixels = np.array(current_pixels, copy=True)
        hidden_history = histories[idx % len(histories)]
        descriptor = frame_descriptor(
            split=split,
            episode_id=episode_id,
            frame_index=idx,
            row_id=None,
            expected_action=None,
            actual_action=None,
            family="information_control",
            pixels=pixels,
            expected_disposition="information_theoretic_control",
            episode_family=str(plan["episode_family"]),
            episode_disposition=str(plan["episode_disposition"]),
            frame_disposition=expected_frame_disposition(
                str(plan["family_label"]), plan.get("mutation_kind"), idx, intervention
            ),
            denominator_class=str(plan["denominator_class"]),
            metadata={
                "episode_seed": int(plan["episode_seed"]),
                "seed_digest": identity.seed_digest,
                "derived_seed_identity": plan["derived_seed_identity"],
                "episode_plan_digest": plan["plan_digest"],
                "frame_seed_identity": frame_plan["frame_seed_identity"],
                "source_row_id": current_row,
                "source_action_id": current_action,
                "family_contract": plan["family_contract"],
                "family_intervention": intervention,
                "control_group_id": control_group["control_group_id"],
                "control_current_row_id": current_row,
                "control_observation_digest": array_digest(current_pixels),
                "observation_operation_chain": information_control_operation_chain(
                    current_row
                ),
                "grounded_causal_history": hidden_history,
                "grounded_causal_history_id": hidden_history["history_id"],
                "hidden_source_history": hidden_history,
                "hidden_source_history_id": hidden_history["hidden_history_id"],
                "hidden_source_history_index": hidden_history["hidden_history_index"],
                "hidden_source_label_digest": hidden_history[
                    "hidden_source_label_digest"
                ],
                "hidden_source_label_digest_group": control_group[
                    "hidden_source_label_digest"
                ],
                "provider_observation_source_id": (
                    f"control:{control_group['control_group_id']}"
                ),
                "provider_observation_timestamp": None,
                "provider_observation_metadata": {},
                "provider_visible_fields": control_group["provider_visible_fields"],
                "provider_hidden_fields": control_group["provider_hidden_fields"],
                "denominator_eligible": False,
                "control_reason": "byte_identical_observation_with_multiple_grounded_causal_histories",
            },
        )
        descriptor["metadata"]["control_visible_digest"] = descriptor["metadata"][
            "provider_observation_digest"
        ]
        frames.append(descriptor | {"pixels": pixels})
    digests = {item["observation_pixel_digest"] for item in frames}
    provider_digests = {
        item["metadata"]["provider_observation_digest"] for item in frames
    }
    if len(digests) != 1:
        raise VPMValidationError(
            "information controls require byte-identical observations"
        )
    if len(provider_digests) != 1:
        raise VPMValidationError(
            "information controls require identical provider-visible observations"
        )
    return frames


def materialize_plan(
    plan: Mapping[str, Any],
    identity: BenchmarkIdentityDTO,
    reachability_tile: Mapping[str, Any],
    *,
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    family = plan["family_label"]
    if family == "valid":
        return valid_episode(
            plan=plan,
            identity=identity,
            reachability_tile=reachability_tile,
            config=config,
        )
    if family == "frame_invalid":
        return invalid_episode(plan=plan, identity=identity, config=config)
    if family == "temporal_negative":
        return temporal_negative_episode(
            plan=plan,
            identity=identity,
            reachability_tile=reachability_tile,
            config=config,
        )
    if family == "information_control":
        return control_episode(
            plan=plan,
            identity=identity,
            reachability_tile=reachability_tile,
            config=config,
        )
    raise VPMValidationError("unsupported episode family")


def materialize_plan_collection(
    plans: Sequence[Mapping[str, Any]],
    identity: BenchmarkIdentityDTO,
    reachability_tile: Mapping[str, Any],
    *,
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for plan in plans:
        records.extend(
            materialize_plan(
                plan,
                identity,
                reachability_tile,
                config=config,
            )
        )
    return records


__all__ = [
    "control_episode",
    "invalid_episode",
    "materialize_plan",
    "materialize_plan_collection",
    "temporal_negative_episode",
    "valid_episode",
]

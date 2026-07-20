from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .arcade_observation import render_row_frame, shooter_config_payload
from .canonical_json import canonical_sha256
from .contracts import (
    ARCADE_RENDERER_CONTRACT_VERSION,
    CONFLICTING_ACTION_SPLICE_VERSION,
    CRITICAL_EVIDENCE_CORRUPTION_VERSION,
    IMPOSSIBLE_TRANSITION_VERSION,
    INFORMATION_CONTROL_VERSION,
    OBSERVATION_OPERATION_CHAIN_VERSION,
    PROVIDER_OBSERVATION_BOUNDARY_VERSION,
    STALE_REPEATED_FRAME_VERSION,
    TRANSFORMATION_FAMILY_VERSION,
)
from .frame_family_kernels import apply_conflicting_splice, apply_critical_corruption
from .observation_legacy_adapters import operation_chain, operation_record
from .observation_provenance import valid_frame_operation_chain
from .pixel_digest import array_digest
from .transformations import _apply_transformation


def conflicting_splice_operation_chain(
    *,
    primary_row_id: str,
    secondary_row_id: str,
    primary_action_id: str,
    secondary_action_id: str,
    primary_transformation_parameters: Mapping[str, Any],
    mask_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    primary_base = render_row_frame(primary_row_id)
    primary_transformed, primary_trace = _apply_transformation(
        primary_base, primary_transformation_parameters
    )
    secondary = render_row_frame(secondary_row_id)
    output, splice_trace = apply_conflicting_splice(
        primary_pixels=primary_transformed,
        secondary_pixels=secondary,
        primary_row_id=primary_row_id,
        secondary_row_id=secondary_row_id,
        primary_action_id=primary_action_id,
        secondary_action_id=secondary_action_id,
        mask_manifest=mask_manifest,
    )
    primary_base_digest = primary_trace["source_observation_digest"]
    primary_transformed_digest = primary_trace["transformed_observation_digest"]
    secondary_digest = array_digest(secondary)
    output_digest = array_digest(output)
    operations = [
        operation_record(
            index=0,
            operation="render_canonical_row",
            operation_version=ARCADE_RENDERER_CONTRACT_VERSION,
            input_digests=[],
            parameters={
                "row_id": primary_row_id,
                "renderer": "zeromodel.arcade_policy.render_state_frame",
                "config": shooter_config_payload(),
            },
            output_digest=primary_base_digest,
        ),
        operation_record(
            index=1,
            operation="apply_bounded_transformation",
            operation_version=TRANSFORMATION_FAMILY_VERSION,
            input_digests=[primary_base_digest],
            parameters={
                "transformation_parameters": dict(primary_transformation_parameters)
            },
            output_digest=primary_transformed_digest,
        ),
        operation_record(
            index=2,
            operation="render_canonical_row",
            operation_version=ARCADE_RENDERER_CONTRACT_VERSION,
            input_digests=[],
            parameters={
                "row_id": secondary_row_id,
                "renderer": "zeromodel.arcade_policy.render_state_frame",
                "config": shooter_config_payload(),
            },
            output_digest=secondary_digest,
        ),
        operation_record(
            index=3,
            operation="compose_simultaneous_target_evidence",
            operation_version=CONFLICTING_ACTION_SPLICE_VERSION,
            input_digests=[primary_transformed_digest, secondary_digest],
            parameters={
                "primary_row_id": primary_row_id,
                "secondary_row_id": secondary_row_id,
                "primary_action_id": primary_action_id,
                "secondary_action_id": secondary_action_id,
                "mask_manifest": dict(mask_manifest),
                "splice_trace_digest": splice_trace["splice_trace_digest"],
            },
            output_digest=output_digest,
        ),
        operation_record(
            index=4,
            operation="emit_observation",
            operation_version=OBSERVATION_OPERATION_CHAIN_VERSION,
            input_digests=[output_digest],
            parameters={"event_type": "frame"},
            output_digest=output_digest,
        ),
    ]
    return operation_chain(operations, output_digest)


def critical_corruption_operation_chain(
    row_id: str,
    transformation_parameters: Mapping[str, Any],
    coordinate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source = render_row_frame(row_id)
    transformed, transform_trace = _apply_transformation(
        source, transformation_parameters
    )
    corrupted, corruption_trace = apply_critical_corruption(
        transformed, coordinate_manifest
    )
    source_digest = transform_trace["source_observation_digest"]
    transformed_digest = transform_trace["transformed_observation_digest"]
    output_digest = array_digest(corrupted)
    operations = [
        operation_record(
            index=0,
            operation="render_canonical_row",
            operation_version=ARCADE_RENDERER_CONTRACT_VERSION,
            input_digests=[],
            parameters={
                "row_id": row_id,
                "renderer": "zeromodel.arcade_policy.render_state_frame",
                "config": shooter_config_payload(),
            },
            output_digest=source_digest,
        ),
        operation_record(
            index=1,
            operation="apply_bounded_transformation",
            operation_version=TRANSFORMATION_FAMILY_VERSION,
            input_digests=[source_digest],
            parameters={"transformation_parameters": dict(transformation_parameters)},
            output_digest=transformed_digest,
        ),
        operation_record(
            index=2,
            operation="apply_critical_coordinate_corruption",
            operation_version=CRITICAL_EVIDENCE_CORRUPTION_VERSION,
            input_digests=[transformed_digest],
            parameters={
                "critical_coordinates": dict(coordinate_manifest),
                "critical_corruption_digest": corruption_trace[
                    "critical_corruption_digest"
                ],
            },
            output_digest=output_digest,
        ),
        operation_record(
            index=3,
            operation="emit_observation",
            operation_version=OBSERVATION_OPERATION_CHAIN_VERSION,
            input_digests=[output_digest],
            parameters={"event_type": "frame"},
            output_digest=output_digest,
        ),
    ]
    return operation_chain(operations, output_digest)


def stale_repeat_operation_chain(
    destination_before: Mapping[str, Any],
    replacement_source: Mapping[str, Any],
    repeat: Mapping[str, Any],
) -> dict[str, Any]:
    dest_params = destination_before.get("metadata", {}).get(
        "transformation_parameters", {}
    )
    src_params = replacement_source.get("metadata", {}).get(
        "transformation_parameters", {}
    )
    dest_row = str(destination_before.get("expected_row"))
    src_row = str(replacement_source.get("expected_row"))
    dest_chain = valid_frame_operation_chain(dest_row, dest_params)
    src_chain = valid_frame_operation_chain(src_row, src_params)
    original_digest = str(destination_before["observation_pixel_digest"])
    replacement_digest = str(replacement_source["observation_pixel_digest"])
    operations = list(dest_chain["operations"][:-1])
    offset = len(operations)
    for op in src_chain["operations"][:-1]:
        copied = dict(op)
        copied["index"] = int(op["index"]) + offset
        copied["operation_digest"] = canonical_sha256(
            {key: value for key, value in copied.items() if key != "operation_digest"}
        )
        operations.append(copied)
    operations.append(
        operation_record(
            index=len(operations),
            operation="stale_repeat_replace",
            operation_version=STALE_REPEATED_FRAME_VERSION,
            input_digests=[original_digest, replacement_digest],
            parameters=dict(repeat),
            output_digest=replacement_digest,
        )
    )
    operations.append(
        operation_record(
            index=len(operations),
            operation="emit_observation",
            operation_version=OBSERVATION_OPERATION_CHAIN_VERSION,
            input_digests=[replacement_digest],
            parameters={"event_type": "frame"},
            output_digest=replacement_digest,
        )
    )
    return operation_chain(operations, replacement_digest)


def impossible_transition_operation_chain(
    destination_before: Mapping[str, Any], transition_plan: Mapping[str, Any]
) -> dict[str, Any]:
    dest_params = destination_before.get("metadata", {}).get(
        "transformation_parameters", {}
    )
    ordinary_row = str(destination_before.get("expected_row"))
    ordinary_chain = valid_frame_operation_chain(ordinary_row, dest_params)
    ordinary_digest = str(destination_before["observation_pixel_digest"])
    unreachable_row = str(transition_plan["destination_row_id"])
    unreachable_pixels = render_row_frame(unreachable_row)
    unreachable_digest = array_digest(unreachable_pixels)
    operations = list(ordinary_chain["operations"][:-1])
    operations.append(
        operation_record(
            index=len(operations),
            operation="render_canonical_row",
            operation_version=ARCADE_RENDERER_CONTRACT_VERSION,
            input_digests=[],
            parameters={
                "row_id": unreachable_row,
                "renderer": "zeromodel.arcade_policy.render_state_frame",
                "config": shooter_config_payload(),
            },
            output_digest=unreachable_digest,
        )
    )
    operations.append(
        operation_record(
            index=len(operations),
            operation="impossible_transition_replace",
            operation_version=IMPOSSIBLE_TRANSITION_VERSION,
            input_digests=[ordinary_digest, unreachable_digest],
            parameters=dict(transition_plan),
            output_digest=unreachable_digest,
        )
    )
    operations.append(
        operation_record(
            index=len(operations),
            operation="emit_observation",
            operation_version=OBSERVATION_OPERATION_CHAIN_VERSION,
            input_digests=[unreachable_digest],
            parameters={"event_type": "frame"},
            output_digest=unreachable_digest,
        )
    )
    return operation_chain(operations, unreachable_digest)


def information_control_operation_chain(current_row_id: str) -> dict[str, Any]:
    current = render_row_frame(current_row_id)
    current_digest = array_digest(current)
    operations = [
        operation_record(
            index=0,
            operation="render_canonical_row",
            operation_version=ARCADE_RENDERER_CONTRACT_VERSION,
            input_digests=[],
            parameters={
                "row_id": str(current_row_id),
                "renderer": "zeromodel.arcade_policy.render_state_frame",
                "config": shooter_config_payload(),
            },
            output_digest=current_digest,
        ),
        operation_record(
            index=1,
            operation="emit_provider_identical_control_observation",
            operation_version=INFORMATION_CONTROL_VERSION,
            input_digests=[current_digest],
            parameters={
                "current_row_id": str(current_row_id),
                "provider_observation_boundary_version": PROVIDER_OBSERVATION_BOUNDARY_VERSION,
            },
            output_digest=current_digest,
        ),
        operation_record(
            index=2,
            operation="emit_observation",
            operation_version=OBSERVATION_OPERATION_CHAIN_VERSION,
            input_digests=[current_digest],
            parameters={"event_type": "frame"},
            output_digest=current_digest,
        ),
    ]
    return operation_chain(operations, current_digest)


__all__ = [
    "conflicting_splice_operation_chain",
    "critical_corruption_operation_chain",
    "impossible_transition_operation_chain",
    "information_control_operation_chain",
    "stale_repeat_operation_chain",
]

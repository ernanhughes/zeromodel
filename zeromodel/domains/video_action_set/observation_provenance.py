from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .arcade_observation import render_row_frame, shooter_config_payload
from .contracts import (
    ARCADE_RENDERER_CONTRACT_VERSION,
    GAP_EVENT_VERSION,
    OBSERVATION_OPERATION_CHAIN_VERSION,
    TRANSFORMATION_FAMILY_VERSION,
)
from .observation_legacy_adapters import operation_chain, operation_record
from .transformations import _apply_transformation


def valid_frame_operation_chain(
    row_id: str,
    transformation_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    source = render_row_frame(row_id)
    _transformed, trace = _apply_transformation(source, transformation_parameters)
    source_digest = trace["source_observation_digest"]
    output_digest = trace["transformed_observation_digest"]
    operations = [
        operation_record(
            index=0,
            operation="render_canonical_row",
            operation_version=ARCADE_RENDERER_CONTRACT_VERSION,
            input_digests=[],
            parameters={
                "row_id": str(row_id),
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
            parameters={
                "transformation_parameters": dict(transformation_parameters),
            },
            output_digest=output_digest,
        ),
        operation_record(
            index=2,
            operation="emit_observation",
            operation_version=OBSERVATION_OPERATION_CHAIN_VERSION,
            input_digests=[output_digest],
            parameters={"event_type": "frame"},
            output_digest=output_digest,
        ),
    ]
    return operation_chain(operations, output_digest)


def gap_event_operation_chain(gap_event: Mapping[str, Any]) -> dict[str, Any]:
    operations = [
        operation_record(
            index=0,
            operation="emit_typed_gap_event",
            operation_version=GAP_EVENT_VERSION,
            input_digests=[],
            parameters=dict(gap_event),
            output_digest=None,
        ),
        operation_record(
            index=1,
            operation="emit_observation",
            operation_version=OBSERVATION_OPERATION_CHAIN_VERSION,
            input_digests=[None],
            parameters={"event_type": "gap_unknown"},
            output_digest=None,
        ),
    ]
    return operation_chain(operations, None)


__all__ = ["gap_event_operation_chain", "valid_frame_operation_chain"]

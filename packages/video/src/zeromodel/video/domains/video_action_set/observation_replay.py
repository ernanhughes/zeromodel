from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.arcade_observation import render_row_frame
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    OBSERVATION_OPERATION_CHAIN_VERSION,
)
from zeromodel.video.domains.video_action_set.observation_provenance_dto import (
    ObservationOperationChainDTO,
)
from zeromodel.video.domains.video_action_set.pixel_digest import array_digest
from zeromodel.video.domains.video_action_set.transformations import (
    _apply_transformation,
)


class ConflictingSpliceExecutor(Protocol):
    def __call__(
        self,
        *,
        primary_pixels: np.ndarray,
        secondary_pixels: np.ndarray,
        primary_row_id: str,
        secondary_row_id: str,
        primary_action_id: str,
        secondary_action_id: str,
        mask_manifest: Mapping[str, Any],
    ) -> tuple[np.ndarray, Mapping[str, Any]]: ...


class CriticalCorruptionExecutor(Protocol):
    def __call__(
        self,
        source: np.ndarray,
        coordinate_manifest: Mapping[str, Any],
    ) -> tuple[np.ndarray, Mapping[str, Any]]: ...


def _operation_parts(
    op: Mapping[str, Any],
) -> tuple[list[str | None], str, Mapping[str, Any], str | None]:
    expected_param_digest = canonical_sha256(op.get("parameters", {}))
    if op.get("parameter_digest") != expected_param_digest:
        raise VPMValidationError("operation parameter digest mismatch")
    if op.get("operation_digest") != canonical_sha256(
        {key: value for key, value in op.items() if key != "operation_digest"}
    ):
        raise VPMValidationError("operation digest mismatch")
    input_digests = [
        None if item is None else str(item) for item in op.get("input_digests", [])
    ]
    output_digest = op.get("output_digest")
    return (
        input_digests,
        str(op.get("operation")),
        cast(Mapping[str, Any], op.get("parameters", {})),
        None if output_digest is None else str(output_digest),
    )


def _replay_pixel_operation(
    operation: str,
    input_digests: list[str | None],
    parameters: Mapping[str, Any],
    output_digest: str | None,
    pixels_by_digest: dict[str, np.ndarray],
    conflicting_splice_executor: ConflictingSpliceExecutor,
    critical_corruption_executor: CriticalCorruptionExecutor,
) -> tuple[str, np.ndarray] | None:
    if operation == "render_canonical_row":
        pixels = render_row_frame(str(parameters["row_id"]))
        digest = array_digest(pixels)
        mismatch = "render operation output digest mismatch"
    elif operation == "apply_bounded_transformation":
        if len(input_digests) != 1 or input_digests[0] not in pixels_by_digest:
            raise VPMValidationError("transformation input digest missing")
        pixels, trace = _apply_transformation(
            pixels_by_digest[str(input_digests[0])],
            parameters["transformation_parameters"],
        )
        digest = trace["transformed_observation_digest"]
        mismatch = "transformation output digest mismatch"
    elif operation == "compose_simultaneous_target_evidence":
        if (
            len(input_digests) != 2
            or input_digests[0] not in pixels_by_digest
            or input_digests[1] not in pixels_by_digest
        ):
            raise VPMValidationError("splice input digest missing")
        pixels, _trace = conflicting_splice_executor(
            primary_pixels=pixels_by_digest[str(input_digests[0])],
            secondary_pixels=pixels_by_digest[str(input_digests[1])],
            primary_row_id=str(parameters["primary_row_id"]),
            secondary_row_id=str(parameters["secondary_row_id"]),
            primary_action_id=str(parameters["primary_action_id"]),
            secondary_action_id=str(parameters["secondary_action_id"]),
            mask_manifest=parameters["mask_manifest"],
        )
        digest = array_digest(pixels)
        mismatch = "splice output digest mismatch"
    elif operation == "apply_critical_coordinate_corruption":
        if len(input_digests) != 1 or input_digests[0] not in pixels_by_digest:
            raise VPMValidationError("critical corruption input digest missing")
        pixels, _trace = critical_corruption_executor(
            pixels_by_digest[str(input_digests[0])],
            parameters["critical_coordinates"],
        )
        digest = array_digest(pixels)
        mismatch = "critical corruption output digest mismatch"
    elif operation in {"stale_repeat_replace", "impossible_transition_replace"}:
        if len(input_digests) != 2 or input_digests[1] not in pixels_by_digest:
            if operation == "stale_repeat_replace":
                raise VPMValidationError("stale repeat input digest missing")
            raise VPMValidationError("impossible transition input digest missing")
        pixels = np.array(pixels_by_digest[str(input_digests[1])], copy=True)
        digest = array_digest(pixels)
        mismatch = (
            "stale repeat output digest mismatch"
            if operation == "stale_repeat_replace"
            else "impossible transition output digest mismatch"
        )
    else:
        return None
    if digest != output_digest:
        raise VPMValidationError(mismatch)
    pixels_by_digest[digest] = pixels
    return digest, pixels


def _replay_emit(
    input_digests: list[str | None],
    output_digest: str | None,
    pixels_by_digest: dict[str, np.ndarray],
) -> tuple[str | None, np.ndarray | None]:
    if output_digest is None:
        if input_digests != [None]:
            raise VPMValidationError("no-pixel emit input mismatch")
        return None, None
    if (
        len(input_digests) != 1
        or input_digests[0] != output_digest
        or str(output_digest) not in pixels_by_digest
    ):
        raise VPMValidationError("emit operation input/output mismatch")
    return str(output_digest), pixels_by_digest[str(output_digest)]


def _checked_chain_operations(chain: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if chain.get("version") != OBSERVATION_OPERATION_CHAIN_VERSION:
        raise VPMValidationError("unsupported observation operation chain version")
    operations = cast(list[Mapping[str, Any]], list(chain.get("operations", [])))
    if [int(op.get("index", -1)) for op in operations] != list(range(len(operations))):
        raise VPMValidationError("operation chain indices are not ordered")
    stored_chain_digest = chain.get("operation_chain_digest")
    if stored_chain_digest != canonical_sha256(
        {key: value for key, value in chain.items() if key != "operation_chain_digest"}
    ):
        raise VPMValidationError("operation chain digest mismatch")
    return operations


def replay_observation_operation_chain(
    chain: Mapping[str, Any],
    *,
    conflicting_splice_executor: ConflictingSpliceExecutor,
    critical_corruption_executor: CriticalCorruptionExecutor,
) -> dict[str, Any]:
    chain = ObservationOperationChainDTO.from_dict(chain).to_dict()
    chain = cast(dict[str, Any], chain)
    operations = _checked_chain_operations(chain)
    pixels_by_digest: dict[str, np.ndarray] = {}
    current_digest: str | None = None
    current_pixels: np.ndarray | None = None
    typed_gap = False
    for op in operations:
        input_digests, operation, parameters, output_digest = _operation_parts(op)
        replay_pixels = _replay_pixel_operation(
            operation,
            input_digests,
            parameters,
            output_digest,
            pixels_by_digest,
            conflicting_splice_executor,
            critical_corruption_executor,
        )
        if replay_pixels is not None:
            current_digest, current_pixels = replay_pixels
            continue
        if operation == "emit_typed_gap_event":
            if output_digest is not None:
                raise VPMValidationError("typed gap operation must not emit pixels")
            typed_gap = True
            current_digest = None
            current_pixels = None
            continue
        if operation in {
            "emit_provider_identical_control_observation",
            "emit_observation",
        }:
            current_digest, current_pixels = _replay_emit(
                input_digests,
                output_digest,
                pixels_by_digest,
            )
            continue
        raise VPMValidationError("unsupported operation chain operation")
    if chain.get("final_emitted_digest") != current_digest:
        raise VPMValidationError("operation chain final digest mismatch")
    return {
        "pixels": current_pixels,
        "final_emitted_digest": current_digest,
        "typed_gap": typed_gap,
        "operation_count": len(operations),
    }


def validate_observation_operation_chain(
    record: Mapping[str, Any],
    *,
    conflicting_splice_executor: ConflictingSpliceExecutor,
    critical_corruption_executor: CriticalCorruptionExecutor,
) -> str:
    chain = record.get("metadata", {}).get("observation_operation_chain")
    if not isinstance(chain, Mapping):
        return "final_observation_provenance_mismatch"
    try:
        replay = replay_observation_operation_chain(
            chain,
            conflicting_splice_executor=conflicting_splice_executor,
            critical_corruption_executor=critical_corruption_executor,
        )
    except (KeyError, TypeError, ValueError, VPMValidationError):
        return "final_observation_provenance_mismatch"
    if replay["final_emitted_digest"] != record.get("observation_pixel_digest"):
        return "final_observation_provenance_mismatch"
    if record.get("event_type") == "gap_unknown":
        if (
            replay["final_emitted_digest"] is not None
            or replay["typed_gap"] is not True
        ):
            return "final_observation_provenance_mismatch"
    elif replay["final_emitted_digest"] is None:
        return "final_observation_provenance_mismatch"
    pixels = record.get("pixels")
    if (
        pixels is not None
        and replay["pixels"] is not None
        and array_digest(np.ascontiguousarray(pixels, dtype=np.uint8))
        != replay["final_emitted_digest"]
    ):
        return "final_observation_provenance_mismatch"
    return "ok"


__all__ = [
    "ConflictingSpliceExecutor",
    "CriticalCorruptionExecutor",
    "replay_observation_operation_chain",
    "validate_observation_operation_chain",
]

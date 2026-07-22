from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import OBSERVATION_OPERATION_CHAIN_VERSION
from zeromodel.video.domains.video_action_set.observation_provenance_dto import (
    ObservationOperationChainDTO,
    ObservationOperationDTO,
)


def operation_record(
    *,
    index: int,
    operation: str,
    operation_version: str,
    input_digests: Sequence[str | None],
    parameters: Mapping[str, Any],
    output_digest: str | None,
) -> dict[str, Any]:
    payload = {
        "index": int(index),
        "operation": str(operation),
        "operation_version": str(operation_version),
        "input_digests": [
            None if item is None else str(item) for item in input_digests
        ],
        "parameters": dict(parameters),
        "parameter_digest": canonical_sha256(parameters),
        "output_digest": output_digest,
    }
    payload["operation_digest"] = canonical_sha256(payload)
    return ObservationOperationDTO.from_dict(payload).to_dict()


def operation_chain(
    operations: Sequence[Mapping[str, Any]],
    final_emitted_digest: str | None,
) -> dict[str, Any]:
    payload = {
        "version": OBSERVATION_OPERATION_CHAIN_VERSION,
        "operations": [dict(item) for item in operations],
        "final_emitted_digest": final_emitted_digest,
    }
    payload["operation_chain_digest"] = canonical_sha256(payload)
    return ObservationOperationChainDTO.from_dict(payload).to_dict()


__all__ = ["operation_chain", "operation_record"]

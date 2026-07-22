from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.observation.visual_address import IMAGE_OBSERVATION_VERSION, ImageObservation
from zeromodel.video.domains.video_action_set.canonical_json import canonical_json_value, canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import PROVIDER_OBSERVATION_BOUNDARY_VERSION


def provider_observation_digest(descriptor: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {"version": PROVIDER_OBSERVATION_BOUNDARY_VERSION, "descriptor": descriptor}
    )


def control_provider_source_id(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata", {})
    group_id = metadata.get("control_group_id") or metadata.get(
        "family_intervention", {}
    ).get("control_group", {}).get("control_group_id")
    if not group_id:
        raise VPMValidationError("information control record lacks a control group id")
    return str(metadata.get("provider_observation_source_id") or f"control:{group_id}")


def provider_observation_for_record(record: Mapping[str, Any]) -> ImageObservation:
    if record.get("pixels") is None:
        raise VPMValidationError("provider observation requires materialized pixels")
    pixels = np.ascontiguousarray(record["pixels"], dtype=np.uint8)
    metadata = record.get("metadata", {})
    if record.get("expected_disposition") == "information_theoretic_control":
        source_id = control_provider_source_id(record)
        timestamp = metadata.get("provider_observation_timestamp")
        visible_metadata = metadata.get("provider_observation_metadata", {})
    else:
        source_id = str(record.get("frame_id"))
        timestamp = (
            metadata.get("provider_observation_timestamp")
            if "provider_observation_timestamp" in metadata
            else None
        )
        visible_metadata = (
            metadata.get("provider_observation_metadata", {})
            if "provider_observation_metadata" in metadata
            else {}
        )
    return ImageObservation(
        pixels,
        timestamp=None if timestamp is None else str(timestamp),
        source_id=source_id,
        metadata=visible_metadata,
    )


def provider_observation_descriptor_for_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if record.get("pixels") is not None:
        return provider_observation_for_record(record).to_descriptor()
    metadata = record.get("metadata", {})
    stored = metadata.get("provider_observation_descriptor")
    if isinstance(stored, Mapping):
        raw_digest = stored.get("raw_digest")
        shape = stored.get("shape")
        version = stored.get("version", IMAGE_OBSERVATION_VERSION)
    else:
        raw_digest = metadata.get("provider_observation_raw_digest")
        shape = metadata.get("provider_observation_shape")
        version = IMAGE_OBSERVATION_VERSION
    if raw_digest is None or shape is None:
        raise VPMValidationError(
            "stored provider observation descriptor lacks raw digest or shape"
        )
    if record.get("expected_disposition") == "information_theoretic_control":
        source_id = control_provider_source_id(record)
        timestamp = metadata.get("provider_observation_timestamp")
        visible_metadata = metadata.get("provider_observation_metadata", {})
    else:
        source_id = str(record.get("frame_id"))
        timestamp = (
            metadata.get("provider_observation_timestamp")
            if "provider_observation_timestamp" in metadata
            else None
        )
        visible_metadata = (
            metadata.get("provider_observation_metadata", {})
            if "provider_observation_metadata" in metadata
            else {}
        )
    return {
        "version": str(version),
        "raw_digest": str(raw_digest),
        "shape": [int(item) for item in shape],
        "timestamp": None if timestamp is None else str(timestamp),
        "source_id": source_id,
        "metadata": canonical_json_value(visible_metadata),
    }


def refresh_provider_observation_metadata(record: dict[str, Any]) -> None:
    metadata = record.setdefault("metadata", {})
    for key in (
        "provider_observation_boundary_version",
        "provider_observation_descriptor",
        "provider_observation_digest",
    ):
        metadata.pop(key, None)
    if record.get("pixels") is None:
        return
    descriptor = provider_observation_descriptor_for_record(record)
    metadata["provider_observation_boundary_version"] = (
        PROVIDER_OBSERVATION_BOUNDARY_VERSION
    )
    metadata["provider_observation_descriptor"] = descriptor
    metadata["provider_observation_digest"] = provider_observation_digest(descriptor)


__all__ = [
    "control_provider_source_id",
    "provider_observation_descriptor_for_record",
    "provider_observation_digest",
    "provider_observation_for_record",
    "refresh_provider_observation_metadata",
]

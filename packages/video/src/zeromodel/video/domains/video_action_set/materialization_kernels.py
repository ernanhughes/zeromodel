from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.contracts import (
    BENCHMARK_VERSION,
    GENERATOR_VERSION,
    PROVIDER_OBSERVATION_BOUNDARY_VERSION,
)
from zeromodel.video.domains.video_action_set.pixel_digest import pixel_digest
from zeromodel.video.domains.video_action_set.provider_observation_boundary import (
    provider_observation_descriptor_for_record,
    provider_observation_digest,
)
from zeromodel.video.domains.video_action_set.transformations import (
    _apply_transformation,
    _transformation_parameters,
)


def apply_family(frame: np.ndarray, family: str, *, seed: int) -> np.ndarray:
    result, _metadata = _apply_transformation(
        frame, _transformation_parameters(family, seed)
    )
    return result


def apply_frame_plan(
    frame: np.ndarray, frame_plan: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    params = frame_plan["transformation_parameters"]
    if str(params["parameter_digest"]) != str(
        frame_plan["transformation_parameter_digest"]
    ):
        raise VPMValidationError("frame transformation parameter digest mismatch")
    return _apply_transformation(frame, params)


def frame_descriptor(
    *,
    split: str,
    episode_id: str,
    frame_index: int,
    row_id: str | None,
    expected_action: str | None,
    actual_action: str | None,
    family: str,
    pixels: np.ndarray | None,
    expected_disposition: str,
    episode_family: str,
    episode_disposition: str,
    frame_disposition: str,
    denominator_class: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    frame_id = f"{split}:{episode_id}:frame-{frame_index:02d}"
    clip_id = f"{split}:{episode_id}:clip"
    record = {
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "split": split,
        "episode_id": episode_id,
        "clip_id": clip_id,
        "frame_id": frame_id,
        "sequence_number": frame_index,
        "event_type": metadata.get("event_type", "frame"),
        "family": family,
        "expected_disposition": expected_disposition,
        "episode_family": episode_family,
        "episode_disposition": episode_disposition,
        "frame_disposition": frame_disposition,
        "denominator_class": denominator_class,
        "expected_row": row_id,
        "expected_action": expected_action,
        "actual_executed_action": actual_action,
        "action_known": actual_action is not None,
        "gap_declaration": metadata.get("gap_declaration"),
        "observation_pixel_digest": pixel_digest(pixels),
        "metadata": dict(metadata),
    }
    if pixels is not None:
        descriptor = provider_observation_descriptor_for_record(
            record | {"pixels": pixels}
        )
        record["metadata"]["provider_observation_boundary_version"] = (
            PROVIDER_OBSERVATION_BOUNDARY_VERSION
        )
        record["metadata"]["provider_observation_descriptor"] = descriptor
        record["metadata"]["provider_observation_digest"] = provider_observation_digest(
            descriptor
        )
    return record


__all__ = [
    "apply_family",
    "apply_frame_plan",
    "frame_descriptor",
]

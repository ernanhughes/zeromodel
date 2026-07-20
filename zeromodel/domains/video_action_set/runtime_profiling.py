from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from ...video_prospective_providers import (
    score_all_rows_optimized,
    score_all_rows_reference,
)
from ...visual_address import ImageObservation
from .provider_measurement import SOURCE_SCOPE
from .provider_observation_boundary import provider_observation_for_record


def select_profiling_records(
    records: Sequence[dict[str, Any]],
    frame_count: int,
) -> list[dict[str, Any]]:
    candidates = []
    valid_records = [
        record for record in records if record["expected_disposition"] == "valid"
    ]
    invalid_records = [
        record
        for record in records
        if record["expected_disposition"] == "distinguishable_invalid_input"
    ]
    control_records = [
        record
        for record in records
        if record["expected_disposition"] == "information_theoretic_control"
    ]
    candidates.extend(valid_records[:4])
    candidates.extend(valid_records[4:6])
    candidates.extend(invalid_records[:2])
    candidates.extend(control_records[:2])
    return candidates[: max(1, frame_count)]


def profile_provider(
    *,
    provider_id: str,
    records: Sequence[dict[str, Any]],
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
    implementation: str,
) -> dict[str, Any]:
    scorer = (
        score_all_rows_reference
        if implementation == "reference"
        else score_all_rows_optimized
    )
    durations = []
    for record in records:
        observation = provider_observation_for_record(record)
        start = time.perf_counter()
        scorer(
            provider_id=provider_id,
            observation=observation,
            prototypes=prototypes,
            policy_artifact_id=policy_artifact_id,
            source_scope=SOURCE_SCOPE,
        )
        durations.append(time.perf_counter() - start)
    total = float(sum(durations))
    mean_frame = total / float(len(durations) or 1)
    return {
        "provider_id": provider_id,
        "implementation": implementation,
        "frame_count": len(durations),
        "total_seconds": total,
        "mean_seconds_per_frame": mean_frame,
        "mean_seconds_per_candidate": mean_frame / 112.0,
        "provider_scoring_call_count": len(durations),
        "candidate_comparison_count": len(durations) * 112,
    }


def runtime_profile_payload(
    *,
    provider_scope: str,
    provider_ids: Sequence[str],
    profile_frame_count: int,
    reference: Sequence[Mapping[str, Any]],
    optimized: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_map = {item["provider_id"]: item for item in reference}
    optimized_map = {item["provider_id"]: item for item in optimized}
    comparison = {
        provider_id: {
            "reference_mean_seconds_per_frame": reference_map[provider_id][
                "mean_seconds_per_frame"
            ],
            "optimized_mean_seconds_per_frame": optimized_map[provider_id][
                "mean_seconds_per_frame"
            ],
            "speedup": (
                reference_map[provider_id]["mean_seconds_per_frame"]
                / optimized_map[provider_id]["mean_seconds_per_frame"]
                if optimized_map[provider_id]["mean_seconds_per_frame"] > 0.0
                else None
            ),
        }
        for provider_id in provider_ids
    }
    projected_observations = {"development": 112, "calibration": 448, "selection": 1008}
    projected_runtime = {
        split: sum(
            optimized_map[provider_id]["mean_seconds_per_frame"] * frame_count
            for provider_id, frame_count in (
                (provider_id, count) for provider_id in provider_ids
            )
        )
        for split, count in projected_observations.items()
    }
    return {
        "profile_frame_count": profile_frame_count,
        "provider_scope": provider_scope,
        "reference": list(reference),
        "optimized": list(optimized),
        "comparison": comparison,
        "projected_runtime_seconds": projected_runtime,
    }


__all__ = [
    "profile_provider",
    "runtime_profile_payload",
    "select_profiling_records",
]

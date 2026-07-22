from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Sequence

from zeromodel.core.artifact import VPMValidationError
from research.video.video_prospective_providers import (
    PROSPECTIVE_PROVIDER_IDS,
    PROSPECTIVE_PROVIDER_VERSIONS,
    score_b3_joint_fit,
    score_normalized_pixel,
    score_registered_local_correlation,
)
from zeromodel.observation.visual_address import ImageObservation
from zeromodel.video.domains.video_action_set.contracts import GENERATOR_VERSION, REACHABILITY_TILE_DIGEST
from zeromodel.video.domains.video_action_set.provider_observation_boundary import (
    provider_observation_digest,
    provider_observation_for_record,
)
from zeromodel.video.domains.video_action_set.reachability_composition import (
    compose_reachability_trace,
    gap_reachability_state,
    state_from_trace,
)


SOURCE_SCOPE = "zeromodel-video-action-set-reachability-benchmark-v1"
PROVIDER_SCORE_CALLS_PER_SCOREABLE_FRAME = len(PROSPECTIVE_PROVIDER_IDS)


@dataclass(frozen=True)
class SplitBuildProgress:
    split: str
    processed_frame_count: int
    total_frame_count: int
    scoreable_frame_count_processed: int
    typed_gap_count_processed: int
    provider_scoring_calls_completed: int
    elapsed_seconds: float
    percentage_complete: float


SplitBuildProgressObserver = Callable[[SplitBuildProgress], None]


def provider_version(provider_id: str) -> str:
    return PROSPECTIVE_PROVIDER_VERSIONS[provider_id]


def score_vector_to_payload(vector: Any) -> dict[str, Any]:
    return {
        "provider_id": vector.provider_id,
        "provider_version": vector.provider_version,
        "row_ids": list(vector.row_ids),
        "raw_scores": list(vector.raw_scores),
        "quantized_scores": list(vector.quantized_scores),
        "ranking": list(vector.evidence.ranking.ranked_row_ids),
        "tie_groups": [group.to_dict() for group in vector.evidence.ranking.tie_groups],
        "score_vector_digest": vector.evidence.score_vector_digest,
        "ranking_digest": vector.evidence.ranking.to_dict()["ranking_digest"],
    }


def _score_all_providers(
    *,
    observation: ImageObservation,
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
) -> tuple[Any, Any, Any]:
    results = []
    for provider_id in PROSPECTIVE_PROVIDER_IDS:
        if provider_id == "P1":
            results.append(
                score_normalized_pixel(
                    observation=observation,
                    prototypes=prototypes,
                    policy_artifact_id=policy_artifact_id,
                )
            )
        elif provider_id == "P2":
            results.append(
                score_registered_local_correlation(
                    observation=observation,
                    prototypes=prototypes,
                    policy_artifact_id=policy_artifact_id,
                    source_scope=SOURCE_SCOPE,
                )
            )
        elif provider_id == "P3":
            results.append(
                score_b3_joint_fit(
                    observation=observation,
                    prototypes=prototypes,
                    policy_artifact_id=policy_artifact_id,
                    source_scope=SOURCE_SCOPE,
                )
            )
        else:
            raise VPMValidationError(f"unsupported provider_id: {provider_id}")
    return tuple(results)


def _reachability_trace_for_result(
    *,
    record: dict[str, Any],
    result: Any,
    reachability_tile: Mapping[str, Any] | None,
    reachability_state: dict[str, Mapping[str, Any] | None] | None,
    row_actions: Mapping[str, str] | None,
) -> dict[str, Any] | None:
    if reachability_tile is None or reachability_state is None or row_actions is None:
        return None
    previous = reachability_state.get(result.provider_id)
    operational_trace = compose_reachability_trace(
        frame_id=record["frame_id"],
        semantic_outcome=result.semantic_top_set_outcome.to_dict(),
        previous_state=previous,
        reachability_tile=reachability_tile,
        row_actions=row_actions,
    )
    reachability_state[result.provider_id] = state_from_trace(operational_trace)
    return operational_trace


def _provider_evidence_row(
    *,
    record: dict[str, Any],
    result: Any,
    policy_artifact_id: str,
    observation_descriptor: dict[str, Any],
    observation_descriptor_digest: str,
    operational_trace: dict[str, Any] | None,
) -> dict[str, Any]:
    outcome = result.semantic_top_set_outcome
    winner_quantized_score = (
        outcome.top_quantized_score if outcome.resolved_row_id is not None else None
    )
    row_scores = result.evidence.to_dict()["row_scores"]
    runner_up_row = result.evidence.ranking.ranked_row_ids[1]
    runner_up_index = [item.row_id for item in result.evidence.row_scores].index(
        runner_up_row
    )
    return {
        **{key: value for key, value in record.items() if key != "pixels"},
        "provider_id": result.provider_id,
        "provider_version": result.provider_version,
        "policy_artifact_id": policy_artifact_id,
        "reachability_tile_digest": REACHABILITY_TILE_DIGEST,
        "all_112_row_ids": [item["row_id"] for item in row_scores],
        "all_112_raw_scores": [item["raw_score"] for item in row_scores],
        "all_112_quantized_scores": [item["quantized_score"] for item in row_scores],
        "complete_ordered_ranking": list(result.evidence.ranking.ranked_row_ids),
        "tie_groups": [group.to_dict() for group in result.evidence.ranking.tie_groups],
        "semantic_top_set_outcome": outcome.to_dict(),
        "semantic_status": outcome.status,
        "resolved_row": outcome.resolved_row_id,
        "resolved_action": outcome.resolved_action_id,
        "top_quantized_score": outcome.top_quantized_score,
        "top_row_ids": list(outcome.top_row_ids),
        "top_action_ids": list(outcome.top_action_ids),
        "semantic_outcome_digest": outcome.semantic_outcome_digest,
        "reachability_composition_trace": operational_trace,
        "winner_row": result.winner_row_id,
        "winner_action": result.winner_action_id,
        "winner_quantized_score": winner_quantized_score,
        "runner_up_row": runner_up_row,
        "runner_up_quantized_score": result.evidence.row_scores[
            runner_up_index
        ].quantized_score,
        "policy_row_universe_digest": result.evidence.policy_row_universe_digest,
        "quantized_score_vector_digest": result.evidence.quantized_score_vector_digest,
        "raw_score_diagnostic_digest": result.evidence.raw_score_diagnostic_digest,
        "score_vector_digest": result.evidence.score_vector_digest,
        "ranking_digest": result.evidence.ranking.to_dict()["ranking_digest"],
        "observation_digest": record["observation_pixel_digest"],
        "provider_observation_descriptor": observation_descriptor,
        "provider_observation_digest": observation_descriptor_digest,
        "episode_seed": record["metadata"]["episode_seed"],
        "generator_identity": {
            "generator_version": GENERATOR_VERSION,
            "seed_digest": record["metadata"]["seed_digest"],
        },
        "provider_diagnostics": dict(result.diagnostics),
    }


def score_record(
    record: dict[str, Any],
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
    *,
    reachability_tile: Mapping[str, Any] | None = None,
    reachability_state: dict[str, Mapping[str, Any] | None] | None = None,
    row_actions: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    if "pixels" not in record:
        raise VPMValidationError("materialized record missing pixels")
    if record["pixels"] is None:
        raise VPMValidationError(
            "typed gap events cannot be provider-scored as ordinary frames"
        )
    observation = provider_observation_for_record(record)
    observation_descriptor = observation.to_descriptor()
    observation_descriptor_digest = provider_observation_digest(observation_descriptor)
    provider_results = _score_all_providers(
        observation=observation,
        prototypes=prototypes,
        policy_artifact_id=policy_artifact_id,
    )
    outputs = []
    for result in provider_results:
        operational_trace = _reachability_trace_for_result(
            record=record,
            result=result,
            reachability_tile=reachability_tile,
            reachability_state=reachability_state,
            row_actions=row_actions,
        )
        outputs.append(
            _provider_evidence_row(
                record=record,
                result=result,
                policy_artifact_id=policy_artifact_id,
                observation_descriptor=observation_descriptor,
                observation_descriptor_digest=observation_descriptor_digest,
                operational_trace=operational_trace,
            )
        )
    return outputs


def measure_record_collection(
    records: Sequence[dict[str, Any]],
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
    *,
    reachability_tile: Mapping[str, Any],
    row_actions: Mapping[str, str],
    split: str | None = None,
    progress_observer: SplitBuildProgressObserver | None = None,
) -> list[dict[str, Any]]:
    reachability_state: dict[str, Mapping[str, Any] | None] = {
        provider_id: None for provider_id in PROSPECTIVE_PROVIDER_IDS
    }
    scored_rows: list[dict[str, Any]] = []
    split_name = (
        str(split)
        if split is not None
        else str(records[0].get("split", ""))
        if records
        else ""
    )
    total_frame_count = len(records)
    processed_frame_count = 0
    scoreable_frame_count_processed = 0
    typed_gap_count_processed = 0
    provider_scoring_calls_completed = 0
    started = time.monotonic()

    def emit_progress() -> None:
        if progress_observer is None:
            return
        percentage_complete = (
            100.0
            if total_frame_count == 0
            else 100.0 * processed_frame_count / total_frame_count
        )
        # Progress observers are operator control-plane callbacks. Let their
        # exceptions propagate so broken monitoring cannot silently disappear.
        progress_observer(
            SplitBuildProgress(
                split=split_name,
                processed_frame_count=processed_frame_count,
                total_frame_count=total_frame_count,
                scoreable_frame_count_processed=scoreable_frame_count_processed,
                typed_gap_count_processed=typed_gap_count_processed,
                provider_scoring_calls_completed=provider_scoring_calls_completed,
                elapsed_seconds=time.monotonic() - started,
                percentage_complete=percentage_complete,
            )
        )

    if total_frame_count == 0:
        emit_progress()
        return scored_rows

    for record in records:
        if record.get("event_type") == "gap_unknown" or record.get("pixels") is None:
            for provider_id in reachability_state:
                reachability_state[provider_id] = gap_reachability_state(record)
            processed_frame_count += 1
            typed_gap_count_processed += 1
            emit_progress()
            continue
        scored_rows.extend(
            score_record(
                record,
                prototypes,
                policy_artifact_id,
                reachability_tile=reachability_tile,
                reachability_state=reachability_state,
                row_actions=row_actions,
            )
        )
        processed_frame_count += 1
        scoreable_frame_count_processed += 1
        provider_scoring_calls_completed += PROVIDER_SCORE_CALLS_PER_SCOREABLE_FRAME
        emit_progress()
    return scored_rows


__all__ = [
    "SOURCE_SCOPE",
    "PROSPECTIVE_PROVIDER_IDS",
    "PROVIDER_SCORE_CALLS_PER_SCOREABLE_FRAME",
    "SplitBuildProgress",
    "SplitBuildProgressObserver",
    "measure_record_collection",
    "provider_version",
    "score_record",
    "score_vector_to_payload",
]

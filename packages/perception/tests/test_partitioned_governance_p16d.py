from __future__ import annotations

import pytest

from zeromodel.perception import (
    DATASET_PARTITION_SEMANTICS,
    DATASET_PARTITION_VERSION,
    DatasetPartitionDTO,
    PerceptionPartitionGovernanceError,
    TemporalComparisonExampleDTO,
    TemporalInferenceComparisonReportDTO,
    bind_comparison_report_to_partition,
    calibrate_partition_owned_candidates,
    promote_partition_owned_model,
)


def _partition(split: str, interaction_ids: tuple[str, ...]) -> DatasetPartitionDTO:
    return DatasetPartitionDTO(
        partition_id=f"sha256:{split}-partition",
        dataset_id="sha256:dataset",
        split=split,
        action_schema_id="sha256:actions",
        interaction_ids=tuple(sorted(interaction_ids)),
        sequence_ids=(f"{split}-sequence",),
        source_pixel_digests=(f"sha256:{split}-pixels",),
        semantics=DATASET_PARTITION_SEMANTICS,
        version=DATASET_PARTITION_VERSION,
    )


def _report(split: str = "validation") -> TemporalInferenceComparisonReportDTO:
    examples = (
        TemporalComparisonExampleDTO(
            interaction_id="interaction-1",
            expected_action="LEFT",
            single_selected_action="RIGHT",
            temporal_selected_action="LEFT",
            single_margin=0.05,
            temporal_margin=0.8,
            single_status="accepted",
            temporal_status="accepted",
            single_correct=False,
            temporal_correct=True,
            conflict_group=True,
        ),
        TemporalComparisonExampleDTO(
            interaction_id="interaction-2",
            expected_action="RIGHT",
            single_selected_action="RIGHT",
            temporal_selected_action="RIGHT",
            single_margin=0.7,
            temporal_margin=0.9,
            single_status="accepted",
            temporal_status="accepted",
            single_correct=True,
            temporal_correct=True,
            conflict_group=True,
        ),
    )
    return TemporalInferenceComparisonReportDTO(
        report_id=f"sha256:{split}-comparison",
        split=split,
        single_translator_id="sha256:single",
        temporal_translator_id="sha256:temporal",
        temporal_window_spec_id="sha256:window",
        example_count=2,
        single_accuracy=0.5,
        temporal_accuracy=1.0,
        accuracy_improvement=0.5,
        single_accepted_accuracy=0.5,
        temporal_accepted_accuracy=1.0,
        single_coverage=1.0,
        temporal_coverage=1.0,
        mean_single_margin=0.375,
        mean_temporal_margin=0.85,
        conflict_example_count=2,
        conflict_single_accuracy=0.5,
        conflict_temporal_accuracy=1.0,
        conflict_resolution_improvement=0.5,
        rejection_threshold=0.0,
        examples=examples,
    )


def test_validation_partition_owns_calibration_and_promotion() -> None:
    partition = _partition("validation", ("interaction-1", "interaction-2"))
    owned = bind_comparison_report_to_partition(_report(), partition)

    first = bind_comparison_report_to_partition(_report(), partition)
    calibrations = calibrate_partition_owned_candidates(owned)
    decision, promoted = promote_partition_owned_model(owned)

    assert owned == first
    assert owned.partition_id == partition.partition_id
    assert calibrations[1].model_kind == "temporal"
    assert decision.selected_model_kind == "temporal"
    assert promoted.validation_comparison_report_id == _report().report_id


def test_binding_rejects_report_interaction_outside_partition() -> None:
    partition = _partition("validation", ("interaction-1",))

    with pytest.raises(
        PerceptionPartitionGovernanceError,
        match="outside partition",
    ):
        bind_comparison_report_to_partition(_report(), partition)


def test_calibration_and_promotion_reject_test_partition() -> None:
    partition = _partition("test", ("interaction-1", "interaction-2"))
    owned = bind_comparison_report_to_partition(_report(split="test"), partition)

    with pytest.raises(
        PerceptionPartitionGovernanceError,
        match="validation partition ownership",
    ):
        calibrate_partition_owned_candidates(owned)

    with pytest.raises(
        PerceptionPartitionGovernanceError,
        match="validation partition ownership",
    ):
        promote_partition_owned_model(owned)


def test_binding_rejects_declared_split_mismatch() -> None:
    partition = _partition("test", ("interaction-1", "interaction-2"))

    with pytest.raises(
        PerceptionPartitionGovernanceError,
        match="does not match partition",
    ):
        bind_comparison_report_to_partition(_report(), partition)

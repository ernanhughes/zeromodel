from __future__ import annotations

import pytest

from zeromodel.perception import (
    PerceptionPromotionError,
    PromotionPolicyDTO,
    TemporalComparisonExampleDTO,
    TemporalInferenceComparisonReportDTO,
    calibrate_comparison_candidates,
    promote_perception_model,
)


def _report(*, split: str = "validation") -> TemporalInferenceComparisonReportDTO:
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
        report_id="sha256:validation-report",
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


def test_calibration_is_validation_owned_and_deterministic() -> None:
    report = _report()
    first = calibrate_comparison_candidates(report)
    second = calibrate_comparison_candidates(report)

    assert first == second
    single, temporal = first
    assert single.rejection_threshold == 0.7
    assert single.accepted_accuracy == 1.0
    assert single.coverage == 0.5
    assert temporal.accepted_accuracy == 1.0
    assert temporal.coverage == 1.0


def test_promotion_selects_temporal_when_it_improves_validation_operating_point() -> None:
    decision, promoted = promote_perception_model(_report())

    assert decision.selected_model_kind == "temporal"
    assert promoted.model_kind == "temporal"
    assert promoted.model_id == "sha256:temporal"
    assert promoted.temporal_window_spec_id == "sha256:window"
    assert promoted.validation_comparison_report_id == "sha256:validation-report"


def test_exact_tie_prefers_simpler_single_frame_candidate() -> None:
    report = _report()
    tied_examples = tuple(
        TemporalComparisonExampleDTO(
            interaction_id=item.interaction_id,
            expected_action=item.expected_action,
            single_selected_action=item.temporal_selected_action,
            temporal_selected_action=item.temporal_selected_action,
            single_margin=item.temporal_margin,
            temporal_margin=item.temporal_margin,
            single_status=item.single_status,
            temporal_status=item.temporal_status,
            single_correct=item.temporal_correct,
            temporal_correct=item.temporal_correct,
            conflict_group=item.conflict_group,
        )
        for item in report.examples
    )
    tied = TemporalInferenceComparisonReportDTO(
        report_id="sha256:tied",
        split="validation",
        single_translator_id=report.single_translator_id,
        temporal_translator_id=report.temporal_translator_id,
        temporal_window_spec_id=report.temporal_window_spec_id,
        example_count=2,
        single_accuracy=1.0,
        temporal_accuracy=1.0,
        accuracy_improvement=0.0,
        single_accepted_accuracy=1.0,
        temporal_accepted_accuracy=1.0,
        single_coverage=1.0,
        temporal_coverage=1.0,
        mean_single_margin=0.85,
        mean_temporal_margin=0.85,
        conflict_example_count=2,
        conflict_single_accuracy=1.0,
        conflict_temporal_accuracy=1.0,
        conflict_resolution_improvement=0.0,
        rejection_threshold=0.0,
        examples=tied_examples,
    )

    decision, promoted = promote_perception_model(tied)

    assert decision.selected_model_kind == "single_frame"
    assert promoted.model_kind == "single_frame"
    assert promoted.temporal_window_spec_id is None


def test_calibration_rejects_test_report() -> None:
    with pytest.raises(PerceptionPromotionError, match="validation"):
        calibrate_comparison_candidates(_report(split="test"))


def test_policy_validates_coverage() -> None:
    with pytest.raises(PerceptionPromotionError, match="minimum_coverage"):
        PromotionPolicyDTO(minimum_coverage=1.1)

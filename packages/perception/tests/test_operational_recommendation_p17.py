from __future__ import annotations

from zeromodel.perception.compatibility import build_model_compatibility_contract
from zeromodel.perception.health import (
    ActionFrequencyDTO,
    OperationalHealthFindingDTO,
    OperationalHealthReportDTO,
)
from zeromodel.perception.lifecycle import (
    InMemoryPerceptionModelLifecycleStore,
    activate_promoted_model,
    build_model_lifecycle_snapshot,
    register_promoted_model,
    supersede_active_model,
)
from zeromodel.perception.promotion import PromotedPerceptionModelDTO
from zeromodel.perception.recommendation import recommend_operational_response


def _model(name: str) -> PromotedPerceptionModelDTO:
    return PromotedPerceptionModelDTO(
        promoted_model_id=f"promoted-{name}",
        model_kind="single_frame",
        model_id=f"translator-{name}",
        rejection_threshold=0.25,
        calibration_id=f"calibration-{name}",
        promotion_decision_id=f"decision-{name}",
        validation_comparison_report_id=f"validation-{name}",
        training_split="train",
        evaluation_split="validation",
    )


def _contract(model: PromotedPerceptionModelDTO, *, action_schema_id: str = "actions-v1"):
    return build_model_compatibility_contract(
        model,
        action_schema_id=action_schema_id,
        source_encoder_spec_id="encoder-v1",
        field_schema_id="fields-v1",
        inference_semantics_version="runtime-v1",
        deployment_slot="primary",
    )


def _health(model_id: str, status: str) -> OperationalHealthReportDTO:
    finding_status = status
    findings = tuple(
        OperationalHealthFindingDTO(
            finding_id=f"finding-{metric}-{status}",
            metric=metric,
            status=finding_status,
            reference_value=0.9,
            observed_value=0.9 if status == "healthy" else 0.5,
            delta=0.0 if status == "healthy" else -0.4,
            threshold=0.1,
            evidence_count=50,
            rationale=f"{metric} is {status}",
        )
        for metric in (
            "coverage",
            "mean_margin",
            "action_distribution",
            "raw_accuracy",
            "accepted_accuracy",
        )
    )
    return OperationalHealthReportDTO(
        report_id=f"health-{status}",
        reference_profile_id="reference-1",
        promoted_model_id=model_id,
        start_sequence_number=1,
        end_sequence_number=50,
        production_metrics_report_id="metrics-1",
        production_action_distribution=(
            ActionFrequencyDTO(action_label="LEFT", count=50, frequency=1.0),
        ),
        action_distribution_distance=0.0 if status == "healthy" else 0.4,
        overall_status=status,
        findings=findings,
        inference_record_ids=tuple(f"inference-{index}" for index in range(50)),
        outcome_ids=tuple(f"outcome-{index}" for index in range(50)),
    )


def _lifecycle():
    store = InMemoryPerceptionModelLifecycleStore()
    earlier = _model("earlier")
    current = _model("current")
    register_promoted_model(store, earlier, registered_by="test", registration_reason="earlier")
    register_promoted_model(store, current, registered_by="test", registration_reason="current")
    activate_promoted_model(store, earlier.promoted_model_id, actor="operator", reason="activate")
    supersede_active_model(store, current.promoted_model_id, actor="operator", reason="supersede")
    return earlier, current, build_model_lifecycle_snapshot(store)


def test_insufficient_evidence_withholds_operational_action() -> None:
    earlier, current, snapshot = _lifecycle()

    recommendation = recommend_operational_response(
        _health(current.promoted_model_id, "insufficient_evidence"),
        snapshot,
        current_contract=_contract(current),
        candidate_contracts={earlier.promoted_model_id: _contract(earlier)},
    )

    assert recommendation.status == "insufficient_evidence"
    assert recommendation.selected_target_promoted_model_id is None
    assert recommendation.assessed_candidates == ()


def test_healthy_evidence_recommends_no_action() -> None:
    earlier, current, snapshot = _lifecycle()

    recommendation = recommend_operational_response(
        _health(current.promoted_model_id, "healthy"),
        snapshot,
        current_contract=_contract(current),
        candidate_contracts={earlier.promoted_model_id: _contract(earlier)},
    )

    assert recommendation.status == "no_action"
    assert recommendation.selected_target_promoted_model_id is None


def test_supported_drift_identifies_compatible_historical_candidate() -> None:
    earlier, current, snapshot = _lifecycle()

    first = recommend_operational_response(
        _health(current.promoted_model_id, "drifted"),
        snapshot,
        current_contract=_contract(current),
        candidate_contracts={earlier.promoted_model_id: _contract(earlier)},
    )
    second = recommend_operational_response(
        _health(current.promoted_model_id, "drifted"),
        snapshot,
        current_contract=_contract(current),
        candidate_contracts={earlier.promoted_model_id: _contract(earlier)},
    )

    assert first == second
    assert first.status == "rollback_candidate"
    assert first.selected_target_promoted_model_id == earlier.promoted_model_id
    assert first.assessed_candidates[0].status == "compatible"


def test_supported_drift_without_compatible_candidate_requires_investigation() -> None:
    earlier, current, snapshot = _lifecycle()

    recommendation = recommend_operational_response(
        _health(current.promoted_model_id, "drifted"),
        snapshot,
        current_contract=_contract(current),
        candidate_contracts={
            earlier.promoted_model_id: _contract(earlier, action_schema_id="actions-v0")
        },
    )

    assert recommendation.status == "investigate"
    assert recommendation.selected_target_promoted_model_id is None
    assert recommendation.assessed_candidates[0].status == "incompatible"
    assert recommendation.assessed_candidates[0].mismatched_fields == ("action_schema_id",)

from __future__ import annotations

from zeromodel.perception.compatibility import (
    assess_rollback_compatibility,
    build_model_compatibility_contract,
)
from zeromodel.perception.disposition import disposition_operational_recommendation
from zeromodel.perception.execution_journal import (
    SqliteGovernedExecutionAttemptStore,
    _event,
    build_governed_execution_attempt,
    execute_journaled_approved_rollback,
)
from zeromodel.perception.governance_audit import audit_governance_integrity
from zeromodel.perception.lifecycle import (
    InMemoryPerceptionModelLifecycleStore,
    activate_promoted_model,
    build_model_lifecycle_snapshot,
    register_promoted_model,
    supersede_active_model,
)
from zeromodel.perception.promotion import PromotedPerceptionModelDTO
from zeromodel.perception.recommendation import OperationalRecommendationDTO
from zeromodel.perception.sql_governance import (
    GovernanceExecutionReceiptDTO,
    SqlitePerceptionGovernanceLedgerStore,
)


def _promoted(name: str) -> PromotedPerceptionModelDTO:
    return PromotedPerceptionModelDTO(
        promoted_model_id=f"promoted-{name}",
        model_kind="single_frame",
        model_id=f"model-{name}",
        rejection_threshold=0.25,
        calibration_id=f"calibration-{name}",
        promotion_decision_id=f"decision-{name}",
        validation_comparison_report_id=f"validation-{name}",
        training_split="train",
        evaluation_split="validation",
    )


def _contract(model: PromotedPerceptionModelDTO):
    return build_model_compatibility_contract(
        model,
        action_schema_id="actions-v1",
        source_encoder_spec_id="encoder-v1",
        field_schema_id="fields-v1",
        inference_semantics_version="runtime-v1",
        deployment_slot="primary",
    )


def _fixture(governance_database):
    lifecycle = InMemoryPerceptionModelLifecycleStore()
    earlier = _promoted("earlier")
    current = _promoted("current")
    for model in (earlier, current):
        register_promoted_model(
            lifecycle,
            model,
            registered_by="test",
            registration_reason="candidate",
        )
    activate_promoted_model(lifecycle, earlier.promoted_model_id, actor="test", reason="activate")
    supersede_active_model(lifecycle, current.promoted_model_id, actor="test", reason="supersede")
    snapshot = build_model_lifecycle_snapshot(lifecycle)
    current_contract = _contract(current)
    target_contract = _contract(earlier)
    assessment = assess_rollback_compatibility(current_contract, target_contract)
    recommendation = OperationalRecommendationDTO(
        recommendation_id="sha256:recommendation",
        health_report_id="sha256:health",
        lifecycle_snapshot_id=snapshot.snapshot_id,
        active_pointer_id=snapshot.active_pointer.pointer_id,
        active_pointer_revision=snapshot.active_pointer.revision,
        active_promoted_model_id=current.promoted_model_id,
        current_contract_id=current_contract.contract_id,
        status="rollback_candidate",
        selected_target_promoted_model_id=earlier.promoted_model_id,
        selected_assessment_id=assessment.assessment_id,
        assessed_candidates=(assessment,),
        rationale="supported drift and compatible historical target",
    )
    disposition = disposition_operational_recommendation(
        recommendation,
        status="approved",
        reviewed_by="operator",
        reason="restore prior compatible model",
    )
    governance = SqlitePerceptionGovernanceLedgerStore(governance_database)
    governance.append_recommendation(recommendation)
    governance.append_disposition(disposition)
    return (
        lifecycle,
        governance,
        earlier,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    )


def test_complete_governance_chain_is_valid_and_deterministic(tmp_path) -> None:
    (
        lifecycle,
        governance,
        _,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    with governance, SqliteGovernedExecutionAttemptStore(
        tmp_path / "attempts.sqlite3"
    ) as attempts:
        execute_journaled_approved_rollback(
            lifecycle,
            governance,
            attempts,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        first = audit_governance_integrity(lifecycle, governance, attempts)
        second = audit_governance_integrity(lifecycle, governance, attempts)

    assert first == second
    assert first.status == "valid"
    assert first.findings == ()
    assert first.recommendation_count == 1
    assert first.disposition_count == 1
    assert first.receipt_count == 1
    assert first.attempt_count == 1


def test_prepared_only_attempt_requires_attention_not_corruption(tmp_path) -> None:
    (
        lifecycle,
        governance,
        _,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    with governance, SqliteGovernedExecutionAttemptStore(
        tmp_path / "attempts.sqlite3"
    ) as attempts:
        attempt = build_governed_execution_attempt(
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        attempts.append_attempt(attempt)
        attempts.append_event(_event(attempt, sequence_number=1, event_kind="prepared"))
        report = audit_governance_integrity(lifecycle, governance, attempts)

    assert report.status == "attention_required"
    assert tuple(item.code for item in report.findings) == ("attempt_prepared_incomplete",)
    assert report.findings[0].severity == "warning"


def test_receipt_with_missing_lifecycle_transition_is_invalid(tmp_path) -> None:
    (
        lifecycle,
        governance,
        earlier,
        _,
        _,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    receipt = GovernanceExecutionReceiptDTO(
        receipt_id="sha256:receipt",
        disposition_id=disposition.disposition_id,
        recommendation_id=recommendation.recommendation_id,
        assessment_id=disposition.selected_assessment_id or "",
        transition_id="sha256:missing-transition",
        pointer_id="sha256:pointer-after",
        pointer_revision=recommendation.active_pointer_revision + 1,
        resulting_promoted_model_id=earlier.promoted_model_id,
    )
    governance.append_execution_receipt(receipt)
    with governance, SqliteGovernedExecutionAttemptStore(
        tmp_path / "attempts.sqlite3"
    ) as attempts:
        report = audit_governance_integrity(lifecycle, governance, attempts)

    assert report.status == "invalid"
    assert "receipt_missing_transition" in tuple(item.code for item in report.findings)
    assert "legacy_unjournaled_receipt" in tuple(item.code for item in report.findings)


def test_undisposed_recommendation_is_informational_and_valid(tmp_path) -> None:
    lifecycle = InMemoryPerceptionModelLifecycleStore()
    current = _promoted("current-only")
    register_promoted_model(
        lifecycle,
        current,
        registered_by="test",
        registration_reason="candidate",
    )
    activate_promoted_model(
        lifecycle,
        current.promoted_model_id,
        actor="test",
        reason="activate",
    )
    pointer = lifecycle.get_active_pointer()
    governance = SqlitePerceptionGovernanceLedgerStore(tmp_path / "governance.sqlite3")
    recommendation = OperationalRecommendationDTO(
        recommendation_id="sha256:recommendation-only",
        health_report_id="sha256:health",
        lifecycle_snapshot_id="sha256:snapshot",
        active_pointer_id=pointer.pointer_id,
        active_pointer_revision=pointer.revision,
        active_promoted_model_id=current.promoted_model_id,
        current_contract_id="sha256:contract",
        status="investigate",
        selected_target_promoted_model_id=None,
        selected_assessment_id=None,
        assessed_candidates=(),
        rationale="investigation required",
    )
    governance.append_recommendation(recommendation)
    with governance, SqliteGovernedExecutionAttemptStore(
        tmp_path / "attempts.sqlite3"
    ) as attempts:
        report = audit_governance_integrity(lifecycle, governance, attempts)

    assert report.status == "valid"
    assert tuple(item.code for item in report.findings) == (
        "recommendation_awaiting_disposition",
    )
    assert report.findings[0].severity == "info"

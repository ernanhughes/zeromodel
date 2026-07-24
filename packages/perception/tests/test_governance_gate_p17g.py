from __future__ import annotations

import pytest

from zeromodel.perception.compatibility import (
    assess_rollback_compatibility,
    build_model_compatibility_contract,
)
from zeromodel.perception.disposition import disposition_operational_recommendation
from zeromodel.perception.execution_journal import (
    GovernedExecutionAttemptEventDTO,
    SqliteGovernedExecutionAttemptStore,
    build_governed_execution_attempt,
)
from zeromodel.perception.governance_audit import (
    GOVERNANCE_AUDIT_FINDING_VERSION,
    GOVERNANCE_AUDIT_SEMANTICS,
    GOVERNANCE_AUDIT_VERSION,
    GovernanceIntegrityAuditReportDTO,
    GovernanceIntegrityFindingDTO,
    audit_governance_integrity,
)
from zeromodel.perception.governance_gate import (
    PerceptionGovernanceExecutionGateError,
    authorize_governance_execution,
    execute_audit_gated_approved_rollback,
)
from zeromodel.perception.governed_execution import execute_or_reconcile_approved_rollback
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
    activate_promoted_model(
        lifecycle,
        earlier.promoted_model_id,
        actor="test",
        reason="activate",
    )
    supersede_active_model(
        lifecycle,
        current.promoted_model_id,
        actor="test",
        reason="supersede",
    )
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


def test_clean_chain_executes_and_finishes_with_valid_audit(tmp_path) -> None:
    (
        lifecycle,
        governance,
        earlier,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    with governance, SqliteGovernedExecutionAttemptStore(
        tmp_path / "attempts.sqlite3"
    ) as attempts:
        gate, attempt, receipt, post_audit = execute_audit_gated_approved_rollback(
            lifecycle,
            governance,
            attempts,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )

    assert gate.authorization_mode == "clean"
    assert gate.attempt_id == attempt.attempt_id
    assert gate.receipt_id == receipt.receipt_id
    assert gate.post_audit_report_id == post_audit.report_id
    assert receipt.resulting_promoted_model_id == earlier.promoted_model_id
    assert post_audit.status == "valid"
    assert post_audit.findings == ()


def test_exact_prepared_attempt_can_resume_through_gate(tmp_path) -> None:
    (
        lifecycle,
        governance,
        _,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    attempt_database = tmp_path / "attempts.sqlite3"
    attempt = build_governed_execution_attempt(
        recommendation,
        disposition,
        current_contract=current_contract,
        target_contract=target_contract,
    )
    prepared = GovernedExecutionAttemptEventDTO(
        event_id="sha256:prepared",
        attempt_id=attempt.attempt_id,
        sequence_number=1,
        event_kind="prepared",
        receipt_id=None,
        pointer_revision=None,
        failure_type=None,
        failure_message=None,
    )
    with SqliteGovernedExecutionAttemptStore(attempt_database) as attempts:
        attempts.append_attempt(attempt)
        attempts.append_event(prepared)

    execute_or_reconcile_approved_rollback(
        lifecycle,
        governance,
        recommendation,
        disposition,
        current_contract=current_contract,
        target_contract=target_contract,
    )
    governance.close()

    with SqlitePerceptionGovernanceLedgerStore(
        tmp_path / "governance.sqlite3"
    ) as reopened_governance, SqliteGovernedExecutionAttemptStore(
        attempt_database
    ) as attempts:
        pre_audit = audit_governance_integrity(
            lifecycle,
            reopened_governance,
            attempts,
        )
        gate, resumed, receipt, post_audit = execute_audit_gated_approved_rollback(
            lifecycle,
            reopened_governance,
            attempts,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        events = attempts.list_events(resumed.attempt_id)

    assert pre_audit.status == "attention_required"
    assert tuple(item.code for item in pre_audit.findings) == (
        "attempt_prepared_incomplete",
    )
    assert gate.authorization_mode == "recover_prepared_attempt"
    assert receipt.pointer_revision == recommendation.active_pointer_revision + 1
    assert tuple(item.event_kind for item in events) == ("prepared", "idempotent")
    assert post_audit.status == "valid"


def test_invalid_integrity_report_blocks_execution(tmp_path) -> None:
    (
        lifecycle,
        governance,
        earlier,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    governance.append_execution_receipt(
        GovernanceExecutionReceiptDTO(
            receipt_id="sha256:receipt",
            disposition_id=disposition.disposition_id,
            recommendation_id=recommendation.recommendation_id,
            assessment_id=disposition.selected_assessment_id or "",
            transition_id="sha256:missing-transition",
            pointer_id="sha256:pointer-after",
            pointer_revision=recommendation.active_pointer_revision + 1,
            resulting_promoted_model_id=earlier.promoted_model_id,
        )
    )
    with governance, SqliteGovernedExecutionAttemptStore(
        tmp_path / "attempts.sqlite3"
    ) as attempts:
        report = audit_governance_integrity(lifecycle, governance, attempts)
        attempt = build_governed_execution_attempt(
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        with pytest.raises(
            PerceptionGovernanceExecutionGateError,
            match="integrity audit is invalid",
        ):
            authorize_governance_execution(report, attempt)

    assert report.status == "invalid"
    assert lifecycle.get_active_pointer().revision == recommendation.active_pointer_revision


def test_attention_for_another_subject_does_not_authorize_recovery() -> None:
    attempt = build_governed_execution_attempt.__annotations__
    del attempt
    finding = GovernanceIntegrityFindingDTO(
        finding_id="sha256:finding",
        severity="warning",
        code="legacy_unjournaled_receipt",
        subject_kind="receipt",
        subject_id="sha256:receipt",
        related_ids=(),
        message="historical receipt has no attempt journal",
        version=GOVERNANCE_AUDIT_FINDING_VERSION,
    )
    report = GovernanceIntegrityAuditReportDTO(
        report_id="sha256:report",
        status="attention_required",
        active_pointer_id="sha256:pointer",
        active_pointer_revision=2,
        recommendation_count=1,
        disposition_count=1,
        receipt_count=1,
        attempt_count=0,
        finding_count=1,
        findings=(finding,),
        semantics=GOVERNANCE_AUDIT_SEMANTICS,
        version=GOVERNANCE_AUDIT_VERSION,
    )
    unrelated_attempt = type(
        "Attempt",
        (),
        {"attempt_id": "sha256:attempt"},
    )()

    with pytest.raises(
        PerceptionGovernanceExecutionGateError,
        match="attention unrelated",
    ):
        authorize_governance_execution(report, unrelated_attempt)  # type: ignore[arg-type]

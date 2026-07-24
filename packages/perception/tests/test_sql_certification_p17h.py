from __future__ import annotations

from dataclasses import replace

import pytest

from zeromodel.perception.compatibility import (
    assess_rollback_compatibility,
    build_model_compatibility_contract,
)
from zeromodel.perception.disposition import disposition_operational_recommendation
from zeromodel.perception.execution_journal import SqliteGovernedExecutionAttemptStore
from zeromodel.perception.lifecycle import (
    InMemoryPerceptionModelLifecycleStore,
    activate_promoted_model,
    build_model_lifecycle_snapshot,
    register_promoted_model,
    supersede_active_model,
)
from zeromodel.perception.promotion import PromotedPerceptionModelDTO
from zeromodel.perception.recommendation import OperationalRecommendationDTO
from zeromodel.perception.sql_certification import (
    PerceptionSqlCertificationError,
    SqliteGovernanceCertificationStore,
    execute_and_certify_audit_gated_rollback,
)
from zeromodel.perception.sql_governance import SqlitePerceptionGovernanceLedgerStore


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
        lifecycle, earlier.promoted_model_id, actor="test", reason="activate"
    )
    supersede_active_model(
        lifecycle, current.promoted_model_id, actor="test", reason="supersede"
    )
    snapshot = build_model_lifecycle_snapshot(lifecycle)
    current_contract = build_model_compatibility_contract(
        current,
        action_schema_id="actions-v1",
        source_encoder_spec_id="encoder-v1",
        field_schema_id="fields-v1",
        inference_semantics_version="runtime-v1",
        deployment_slot="primary",
    )
    target_contract = build_model_compatibility_contract(
        earlier,
        action_schema_id="actions-v1",
        source_encoder_spec_id="encoder-v1",
        field_schema_id="fields-v1",
        inference_semantics_version="runtime-v1",
        deployment_slot="primary",
    )
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
        current_contract,
        target_contract,
        recommendation,
        disposition,
    )


def test_certification_survives_restart_with_exact_audits(tmp_path) -> None:
    (
        lifecycle,
        governance,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    certification_database = tmp_path / "certifications.sqlite3"
    with governance, SqliteGovernedExecutionAttemptStore(
        tmp_path / "attempts.sqlite3"
    ) as attempts, SqliteGovernanceCertificationStore(certification_database) as certifications:
        bundle, attempt, receipt = execute_and_certify_audit_gated_rollback(
            lifecycle,
            governance,
            attempts,
            certifications,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        certifications.append_certification(bundle)

    with SqliteGovernanceCertificationStore(certification_database) as reopened:
        restored = reopened.get_certification(bundle.certification.certification_id)
        listed = reopened.list_certifications()

    assert restored == bundle
    assert listed == (bundle,)
    assert bundle.certification.attempt_id == attempt.attempt_id
    assert bundle.certification.receipt_id == receipt.receipt_id
    assert bundle.pre_audit.report_id == bundle.gate.pre_audit_report_id
    assert bundle.post_audit.status == "valid"
    assert bundle.post_audit.active_pointer_revision == receipt.pointer_revision


def test_conflicting_certification_for_same_attempt_is_rejected(tmp_path) -> None:
    (
        lifecycle,
        governance,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    with governance, SqliteGovernedExecutionAttemptStore(
        tmp_path / "attempts.sqlite3"
    ) as attempts, SqliteGovernanceCertificationStore(
        tmp_path / "certifications.sqlite3"
    ) as certifications:
        bundle, _, _ = execute_and_certify_audit_gated_rollback(
            lifecycle,
            governance,
            attempts,
            certifications,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        conflicting = replace(
            bundle,
            certification=replace(
                bundle.certification,
                certification_id="sha256:conflicting-certification",
            ),
        )
        with pytest.raises(
            PerceptionSqlCertificationError,
            match="conflicting certification",
        ):
            certifications.append_certification(conflicting)


def test_bundle_rejects_non_valid_post_audit(tmp_path) -> None:
    (
        lifecycle,
        governance,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    with governance, SqliteGovernedExecutionAttemptStore(
        tmp_path / "attempts.sqlite3"
    ) as attempts, SqliteGovernanceCertificationStore(
        tmp_path / "certifications.sqlite3"
    ) as certifications:
        bundle, _, _ = execute_and_certify_audit_gated_rollback(
            lifecycle,
            governance,
            attempts,
            certifications,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )

    with pytest.raises(PerceptionSqlCertificationError, match="valid post-audit"):
        replace(bundle, post_audit=replace(bundle.post_audit, status="attention_required"))

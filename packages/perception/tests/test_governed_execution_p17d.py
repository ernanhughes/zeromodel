from __future__ import annotations

import pytest

from zeromodel.perception.compatibility import (
    assess_rollback_compatibility,
    build_model_compatibility_contract,
)
from zeromodel.perception.disposition import (
    disposition_operational_recommendation,
    execute_approved_rollback,
)
from zeromodel.perception.governed_execution import (
    PerceptionGovernedExecutionError,
    execute_or_reconcile_approved_rollback,
)
from zeromodel.perception.lifecycle import (
    InMemoryPerceptionModelLifecycleStore,
    activate_promoted_model,
    build_model_lifecycle_snapshot,
    register_promoted_model,
    supersede_active_model,
)
from zeromodel.perception.promotion import PromotedPerceptionModelDTO
from zeromodel.perception.recommendation import OperationalRecommendationDTO
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


def _contract(model: PromotedPerceptionModelDTO):
    return build_model_compatibility_contract(
        model,
        action_schema_id="actions-v1",
        source_encoder_spec_id="encoder-v1",
        field_schema_id="fields-v1",
        inference_semantics_version="runtime-v1",
        deployment_slot="primary",
    )


def _fixture(tmp_path):
    lifecycle = InMemoryPerceptionModelLifecycleStore()
    earlier = _promoted("earlier")
    current = _promoted("current")
    other = _promoted("other")
    for model in (earlier, current, other):
        register_promoted_model(
            lifecycle,
            model,
            registered_by="test",
            registration_reason="candidate",
        )
    activate_promoted_model(lifecycle, earlier.promoted_model_id, actor="test", reason="activate")
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
        rationale="supported drift with compatible historical candidate",
    )
    disposition = disposition_operational_recommendation(
        recommendation,
        status="approved",
        reviewed_by="operator",
        reason="restore prior compatible model",
    )
    governance = SqlitePerceptionGovernanceLedgerStore(tmp_path / "governance.sqlite3")
    governance.append_recommendation(recommendation)
    governance.append_disposition(disposition)
    return (
        lifecycle,
        governance,
        earlier,
        current,
        other,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    )


def test_executes_and_persists_one_receipt(tmp_path) -> None:
    (
        lifecycle,
        governance,
        earlier,
        _,
        _,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path)
    try:
        receipt = execute_or_reconcile_approved_rollback(
            lifecycle,
            governance,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        repeated = execute_or_reconcile_approved_rollback(
            lifecycle,
            governance,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        assert repeated == receipt
        assert lifecycle.get_active_pointer().active_promoted_model_id == earlier.promoted_model_id
        assert governance.list_execution_receipts() == (receipt,)
    finally:
        governance.close()


def test_recovers_receipt_after_crash_between_mutation_and_persistence(tmp_path) -> None:
    (
        lifecycle,
        governance,
        earlier,
        _,
        _,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path)
    database = tmp_path / "governance.sqlite3"
    try:
        _, transition, pointer = execute_approved_rollback(
            lifecycle,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        assert pointer.active_promoted_model_id == earlier.promoted_model_id
        assert governance.list_execution_receipts() == ()
        governance.close()

        with SqlitePerceptionGovernanceLedgerStore(database) as restarted:
            receipt = execute_or_reconcile_approved_rollback(
                lifecycle,
                restarted,
                recommendation,
                disposition,
                current_contract=current_contract,
                target_contract=target_contract,
            )
            assert receipt.transition_id == transition.transition_id
            assert receipt.pointer_revision == pointer.revision
            assert restarted.list_execution_receipts() == (receipt,)
    finally:
        try:
            governance.close()
        except Exception:
            pass


def test_unrelated_pointer_movement_is_not_misread_as_recovery(tmp_path) -> None:
    (
        lifecycle,
        governance,
        _,
        _,
        other,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path)
    try:
        supersede_active_model(
            lifecycle,
            other.promoted_model_id,
            actor="different-operator",
            reason="unrelated deployment",
        )
        with pytest.raises(
            PerceptionGovernedExecutionError,
            match="neither reviewed pre-state nor exact rollback post-state",
        ):
            execute_or_reconcile_approved_rollback(
                lifecycle,
                governance,
                recommendation,
                disposition,
                current_contract=current_contract,
                target_contract=target_contract,
            )
        assert governance.list_execution_receipts() == ()
    finally:
        governance.close()


def test_execution_requires_exact_persisted_chain(tmp_path) -> None:
    (
        lifecycle,
        governance,
        _,
        _,
        _,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path)
    try:
        unpersisted = OperationalRecommendationDTO(
            **{
                **recommendation.__dict__,
                "recommendation_id": "sha256:different-recommendation",
            }
        )
        with pytest.raises(PerceptionGovernedExecutionError, match="exact persisted recommendation"):
            execute_or_reconcile_approved_rollback(
                lifecycle,
                governance,
                unpersisted,
                disposition,
                current_contract=current_contract,
                target_contract=target_contract,
            )
    finally:
        governance.close()

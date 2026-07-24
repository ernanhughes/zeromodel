from __future__ import annotations

import pytest

from zeromodel.perception.compatibility import (
    build_model_compatibility_contract,
)
from zeromodel.perception.disposition import disposition_operational_recommendation
from zeromodel.perception.execution_journal import (
    PerceptionExecutionJournalError,
    SqliteGovernedExecutionAttemptStore,
    build_governed_execution_attempt,
    execute_journaled_approved_rollback,
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
    from zeromodel.perception.compatibility import assess_rollback_compatibility

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


def test_normal_execution_records_prepared_and_completed(tmp_path) -> None:
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
        attempt, receipt = execute_journaled_approved_rollback(
            lifecycle,
            governance,
            attempts,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        events = attempts.list_events(attempt.attempt_id)

    assert tuple(item.event_kind for item in events) == ("prepared", "completed")
    assert receipt.resulting_promoted_model_id == earlier.promoted_model_id
    assert lifecycle.get_active_pointer().revision == recommendation.active_pointer_revision + 1


def test_prepared_only_attempt_is_reconciled_after_crash(tmp_path) -> None:
    (
        lifecycle,
        governance,
        _,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    attempts_database = tmp_path / "attempts.sqlite3"
    attempt = build_governed_execution_attempt(
        recommendation,
        disposition,
        current_contract=current_contract,
        target_contract=target_contract,
    )
    from zeromodel.perception.execution_journal import _event

    with SqliteGovernedExecutionAttemptStore(attempts_database) as attempts:
        attempts.append_attempt(attempt)
        attempts.append_event(_event(attempt, sequence_number=1, event_kind="prepared"))
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
    ) as reopened_governance, SqliteGovernedExecutionAttemptStore(attempts_database) as attempts:
        resumed, receipt = execute_journaled_approved_rollback(
            lifecycle,
            reopened_governance,
            attempts,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        events = attempts.list_events(resumed.attempt_id)

    assert tuple(item.event_kind for item in events) == ("prepared", "reconciled")
    assert receipt.pointer_revision == recommendation.active_pointer_revision + 1


def test_existing_receipt_with_new_attempt_is_idempotently_linked(tmp_path) -> None:
    (
        lifecycle,
        governance,
        _,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    expected = execute_or_reconcile_approved_rollback(
        lifecycle,
        governance,
        recommendation,
        disposition,
        current_contract=current_contract,
        target_contract=target_contract,
    )
    with governance, SqliteGovernedExecutionAttemptStore(
        tmp_path / "attempts.sqlite3"
    ) as attempts:
        attempt, receipt = execute_journaled_approved_rollback(
            lifecycle,
            governance,
            attempts,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        events = attempts.list_events(attempt.attempt_id)

    assert receipt == expected
    assert tuple(item.event_kind for item in events) == ("prepared", "idempotent")


def test_failure_is_terminal_and_does_not_retry(tmp_path) -> None:
    (
        lifecycle,
        governance,
        _,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    supersede_target = _promoted("third")
    register_promoted_model(
        lifecycle,
        supersede_target,
        registered_by="test",
        registration_reason="unrelated",
    )
    supersede_active_model(
        lifecycle,
        supersede_target.promoted_model_id,
        actor="other",
        reason="unrelated change",
    )
    with governance, SqliteGovernedExecutionAttemptStore(
        tmp_path / "attempts.sqlite3"
    ) as attempts:
        with pytest.raises(PerceptionExecutionJournalError):
            execute_journaled_approved_rollback(
                lifecycle,
                governance,
                attempts,
                recommendation,
                disposition,
                current_contract=current_contract,
                target_contract=target_contract,
            )
        attempt = build_governed_execution_attempt(
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
        events = attempts.list_events(attempt.attempt_id)
        with pytest.raises(PerceptionExecutionJournalError, match="terminally failed"):
            execute_journaled_approved_rollback(
                lifecycle,
                governance,
                attempts,
                recommendation,
                disposition,
                current_contract=current_contract,
                target_contract=target_contract,
            )

    assert tuple(item.event_kind for item in events) == ("prepared", "failed")


def test_attempt_and_events_survive_restart(tmp_path) -> None:
    (
        lifecycle,
        governance,
        _,
        current_contract,
        target_contract,
        recommendation,
        disposition,
    ) = _fixture(tmp_path / "governance.sqlite3")
    database = tmp_path / "attempts.sqlite3"
    with governance, SqliteGovernedExecutionAttemptStore(database) as attempts:
        attempt, _ = execute_journaled_approved_rollback(
            lifecycle,
            governance,
            attempts,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
    with SqliteGovernedExecutionAttemptStore(database) as reopened:
        assert reopened.get_attempt(attempt.attempt_id) == attempt
        assert tuple(item.event_kind for item in reopened.list_events(attempt.attempt_id)) == (
            "prepared",
            "completed",
        )

from __future__ import annotations

import dataclasses
import sqlite3

import pytest

from zeromodel.perception.compatibility import RollbackCompatibilityAssessmentDTO
from zeromodel.perception.disposition import disposition_operational_recommendation
from zeromodel.perception.recommendation import OperationalRecommendationDTO
from zeromodel.perception.sql_governance import (
    GovernanceExecutionReceiptDTO,
    PerceptionSqlGovernanceError,
    SQL_GOVERNANCE_SCHEMA_VERSION,
    SqlitePerceptionGovernanceLedgerStore,
)


def _artifacts():
    assessment = RollbackCompatibilityAssessmentDTO(
        assessment_id="sha256:assessment",
        current_contract_id="sha256:current-contract",
        target_contract_id="sha256:target-contract",
        current_promoted_model_id="promoted-current",
        target_promoted_model_id="promoted-earlier",
        status="compatible",
        mismatched_fields=(),
    )
    recommendation = OperationalRecommendationDTO(
        recommendation_id="sha256:recommendation",
        health_report_id="sha256:health",
        lifecycle_snapshot_id="sha256:snapshot",
        active_pointer_id="sha256:pointer",
        active_pointer_revision=2,
        active_promoted_model_id="promoted-current",
        current_contract_id="sha256:current-contract",
        status="rollback_candidate",
        selected_target_promoted_model_id="promoted-earlier",
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
    receipt = GovernanceExecutionReceiptDTO(
        receipt_id="sha256:receipt",
        disposition_id=disposition.disposition_id,
        recommendation_id=recommendation.recommendation_id,
        assessment_id=assessment.assessment_id,
        transition_id="sha256:transition",
        pointer_id="sha256:pointer-after",
        pointer_revision=3,
        resulting_promoted_model_id="promoted-earlier",
    )
    return recommendation, disposition, receipt


def test_governance_chain_survives_restart(tmp_path) -> None:
    database = tmp_path / "governance.sqlite3"
    recommendation, disposition, receipt = _artifacts()

    with SqlitePerceptionGovernanceLedgerStore(database) as store:
        store.append_recommendation(recommendation)
        store.append_disposition(disposition)
        store.append_execution_receipt(receipt)

    with SqlitePerceptionGovernanceLedgerStore(database) as reopened:
        assert reopened.get_recommendation(recommendation.recommendation_id) == recommendation
        assert reopened.get_disposition(disposition.disposition_id) == disposition
        assert reopened.get_execution_receipt(receipt.receipt_id) == receipt
        assert reopened.list_recommendations() == (recommendation,)
        assert reopened.list_dispositions() == (disposition,)
        assert reopened.list_execution_receipts() == (receipt,)


def test_disposition_requires_persisted_recommendation(tmp_path) -> None:
    _, disposition, _ = _artifacts()
    with SqlitePerceptionGovernanceLedgerStore(tmp_path / "governance.sqlite3") as store:
        with pytest.raises(PerceptionSqlGovernanceError, match="persisted recommendation"):
            store.append_disposition(disposition)


def test_one_final_disposition_per_recommendation(tmp_path) -> None:
    recommendation, disposition, _ = _artifacts()
    conflicting = dataclasses.replace(
        disposition,
        disposition_id="sha256:rejected-disposition",
        status="rejected",
        reason="continue investigation",
    )
    with SqlitePerceptionGovernanceLedgerStore(tmp_path / "governance.sqlite3") as store:
        store.append_recommendation(recommendation)
        store.append_disposition(disposition)
        with pytest.raises(PerceptionSqlGovernanceError, match="final disposition"):
            store.append_disposition(conflicting)


def test_execution_receipt_is_idempotent_but_not_replayable(tmp_path) -> None:
    recommendation, disposition, receipt = _artifacts()
    conflicting = dataclasses.replace(
        receipt,
        receipt_id="sha256:second-receipt",
        transition_id="sha256:second-transition",
    )
    with SqlitePerceptionGovernanceLedgerStore(tmp_path / "governance.sqlite3") as store:
        store.append_recommendation(recommendation)
        store.append_disposition(disposition)
        store.append_execution_receipt(receipt)
        store.append_execution_receipt(receipt)
        with pytest.raises(PerceptionSqlGovernanceError, match="already been executed"):
            store.append_execution_receipt(conflicting)


def test_unknown_schema_version_is_rejected(tmp_path) -> None:
    database = tmp_path / "governance.sqlite3"
    with SqlitePerceptionGovernanceLedgerStore(database):
        pass
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE perception_governance_metadata SET value='future' WHERE key='schema_version'"
    )
    connection.commit()
    connection.close()

    with pytest.raises(PerceptionSqlGovernanceError, match="schema version"):
        SqlitePerceptionGovernanceLedgerStore(database)


def test_schema_version_is_explicit() -> None:
    assert SQL_GOVERNANCE_SCHEMA_VERSION == "perception-sql-governance-schema/1"

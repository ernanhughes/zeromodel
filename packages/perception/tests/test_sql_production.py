from __future__ import annotations

import sqlite3

import pytest

from zeromodel.perception import (
    PerceptionSqlProductionError,
    ProductionInferenceRecordDTO,
    ProductionOutcomeRecordDTO,
    SQL_PRODUCTION_SCHEMA_VERSION,
    SqlitePerceptionProductionLedgerStore,
)


def _inference(sequence: int, *, model: str = "promoted-a") -> ProductionInferenceRecordDTO:
    return ProductionInferenceRecordDTO(
        record_id=f"record-{sequence}",
        sequence_number=sequence,
        pointer_id=f"pointer-{sequence}",
        pointer_revision=sequence,
        promoted_model_id=model,
        model_id="translator-a",
        model_kind="single_frame",
        input_id=f"input-{sequence}",
        interaction_id=f"interaction-{sequence}",
        inference_result_id=f"result-{sequence}",
        selected_action="left",
        margin=0.75,
        status="accepted",
        rejection_threshold=0.25,
    )


def _outcome(sequence: int, record_id: str) -> ProductionOutcomeRecordDTO:
    return ProductionOutcomeRecordDTO(
        outcome_id=f"outcome-{sequence}",
        inference_record_id=record_id,
        outcome_sequence_number=sequence,
        observed_action="left",
        source="environment",
        correct=True,
    )


def test_sql_store_restores_inferences_and_outcomes_after_restart(tmp_path) -> None:
    database = tmp_path / "production.sqlite"
    inference = _inference(1)
    outcome = _outcome(1, inference.record_id)

    with SqlitePerceptionProductionLedgerStore(database) as store:
        store.append_inference(inference)
        store.append_outcome(outcome)

    with SqlitePerceptionProductionLedgerStore(database) as reopened:
        assert reopened.get_inference(inference.record_id) == inference
        assert reopened.get_outcome_for_inference(inference.record_id) == outcome
        assert reopened.list_inferences() == (inference,)
        assert reopened.list_outcomes() == (outcome,)


def test_sql_store_enforces_global_contiguous_sequences(tmp_path) -> None:
    with SqlitePerceptionProductionLedgerStore(tmp_path / "production.sqlite") as store:
        with pytest.raises(PerceptionSqlProductionError):
            store.append_inference(_inference(2))

        store.append_inference(_inference(1))
        with pytest.raises(PerceptionSqlProductionError):
            store.append_outcome(_outcome(2, "record-1"))


def test_sql_store_rejects_conflicting_immutable_records(tmp_path) -> None:
    with SqlitePerceptionProductionLedgerStore(tmp_path / "production.sqlite") as store:
        original = _inference(1)
        store.append_inference(original)
        conflicting = ProductionInferenceRecordDTO(
            **{**original.__dict__, "selected_action": "right"}
        )
        with pytest.raises(PerceptionSqlProductionError):
            store.append_inference(conflicting)

        outcome = _outcome(1, original.record_id)
        store.append_outcome(outcome)
        conflicting_outcome = ProductionOutcomeRecordDTO(
            **{**outcome.__dict__, "observed_action": "right", "correct": False}
        )
        with pytest.raises(PerceptionSqlProductionError):
            store.append_outcome(conflicting_outcome)


def test_sql_store_supports_indexed_model_sequence_windows(tmp_path) -> None:
    with SqlitePerceptionProductionLedgerStore(tmp_path / "production.sqlite") as store:
        store.append_inference(_inference(1, model="promoted-a"))
        store.append_inference(_inference(2, model="promoted-b"))
        store.append_inference(_inference(3, model="promoted-a"))

        window = store.list_inferences_in_window(
            start_sequence_number=2,
            end_sequence_number=3,
            promoted_model_id="promoted-a",
        )
        assert tuple(item.sequence_number for item in window) == (3,)


def test_sql_store_rejects_unknown_schema_version(tmp_path) -> None:
    database = tmp_path / "production.sqlite"
    with SqlitePerceptionProductionLedgerStore(database):
        pass
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE perception_production_metadata SET value = ? WHERE key = 'schema_version'",
        ("future-schema",),
    )
    connection.commit()
    connection.close()

    with pytest.raises(PerceptionSqlProductionError):
        SqlitePerceptionProductionLedgerStore(database)


def test_sql_production_schema_version_is_explicit() -> None:
    assert SQL_PRODUCTION_SCHEMA_VERSION == "perception-sql-production-schema/1"

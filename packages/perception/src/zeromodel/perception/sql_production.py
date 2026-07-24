"""Durable SQLite persistence for the Stage P14 production ledger.

P15 implements the existing DTO-only production-ledger store boundary with SQLite.
Inference and outcome records remain immutable and globally ordered, while indexed
sequence and promoted-model columns support durable operational metric windows.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final

from .production import (
    PRODUCTION_INFERENCE_RECORD_VERSION,
    PRODUCTION_INFERENCE_SEMANTICS,
    PRODUCTION_OUTCOME_RECORD_VERSION,
    PRODUCTION_OUTCOME_SEMANTICS,
    PerceptionProductionLedgerError,
    ProductionInferenceRecordDTO,
    ProductionOutcomeRecordDTO,
)

SQL_PRODUCTION_SCHEMA_VERSION: Final = "perception-sql-production-schema/1"
SQL_PRODUCTION_STORE_VERSION: Final = "perception-sql-production-store/1"
SQL_PRODUCTION_SEMANTICS: Final = (
    "durable_append_only_production_inference_and_outcome_ledger"
)


class PerceptionSqlProductionError(PerceptionProductionLedgerError):
    """Raised when durable production-ledger persistence contracts are violated."""


class SqlitePerceptionProductionLedgerStore:
    """SQLite implementation of the P14 production-ledger store protocol."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        self._connection = sqlite3.connect(self.database_path)
        self._connection.row_factory = sqlite3.Row
        self._closed = False
        self._configure()
        self._initialize_schema()

    def _configure(self) -> None:
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS perception_production_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS perception_production_inferences (
                    record_id TEXT PRIMARY KEY,
                    sequence_number INTEGER NOT NULL UNIQUE CHECK (sequence_number > 0),
                    pointer_id TEXT NOT NULL,
                    pointer_revision INTEGER NOT NULL CHECK (pointer_revision > 0),
                    promoted_model_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_kind TEXT NOT NULL CHECK (model_kind IN ('single_frame', 'temporal')),
                    input_id TEXT NOT NULL,
                    interaction_id TEXT,
                    inference_result_id TEXT NOT NULL UNIQUE,
                    selected_action TEXT NOT NULL,
                    margin REAL NOT NULL CHECK (margin >= 0.0 AND margin <= 1.0),
                    status TEXT NOT NULL CHECK (status IN ('accepted', 'rejected_ambiguous')),
                    rejection_threshold REAL NOT NULL CHECK (
                        rejection_threshold >= 0.0 AND rejection_threshold <= 1.0
                    ),
                    semantics TEXT NOT NULL,
                    version TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS perception_production_outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    inference_record_id TEXT NOT NULL UNIQUE,
                    outcome_sequence_number INTEGER NOT NULL UNIQUE CHECK (
                        outcome_sequence_number > 0
                    ),
                    observed_action TEXT NOT NULL,
                    source TEXT NOT NULL,
                    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
                    semantics TEXT NOT NULL,
                    version TEXT NOT NULL,
                    FOREIGN KEY (inference_record_id)
                        REFERENCES perception_production_inferences(record_id)
                );

                CREATE INDEX IF NOT EXISTS idx_production_inference_model_sequence
                    ON perception_production_inferences(
                        promoted_model_id,
                        sequence_number
                    );

                CREATE INDEX IF NOT EXISTS idx_production_inference_pointer_revision
                    ON perception_production_inferences(pointer_revision, sequence_number);

                CREATE INDEX IF NOT EXISTS idx_production_outcome_inference
                    ON perception_production_outcomes(inference_record_id);
                """
            )
            row = self._connection.execute(
                "SELECT value FROM perception_production_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO perception_production_metadata(key, value) VALUES (?, ?)",
                    ("schema_version", SQL_PRODUCTION_SCHEMA_VERSION),
                )
            elif row["value"] != SQL_PRODUCTION_SCHEMA_VERSION:
                raise PerceptionSqlProductionError(
                    f"unsupported production database schema: {row['value']}"
                )

    def _ensure_open(self) -> None:
        if self._closed:
            raise PerceptionSqlProductionError("production ledger store is closed")

    def append_inference(self, record: ProductionInferenceRecordDTO) -> None:
        self._ensure_open()
        existing = self._connection.execute(
            "SELECT * FROM perception_production_inferences WHERE record_id = ?",
            (record.record_id,),
        ).fetchone()
        if existing is not None:
            if self._inference_from_row(existing) == record:
                return
            raise PerceptionSqlProductionError("production inference identity conflict")
        expected = self._connection.execute(
            "SELECT COALESCE(MAX(sequence_number), 0) + 1 AS expected "
            "FROM perception_production_inferences"
        ).fetchone()["expected"]
        if record.sequence_number != expected:
            raise PerceptionSqlProductionError(
                "production inference sequence is not contiguous"
            )
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO perception_production_inferences(
                        record_id, sequence_number, pointer_id, pointer_revision,
                        promoted_model_id, model_id, model_kind, input_id,
                        interaction_id, inference_result_id, selected_action,
                        margin, status, rejection_threshold, semantics, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.sequence_number,
                        record.pointer_id,
                        record.pointer_revision,
                        record.promoted_model_id,
                        record.model_id,
                        record.model_kind,
                        record.input_id,
                        record.interaction_id,
                        record.inference_result_id,
                        record.selected_action,
                        record.margin,
                        record.status,
                        record.rejection_threshold,
                        record.semantics,
                        record.version,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise PerceptionSqlProductionError(
                "production inference violates durable ledger constraints"
            ) from exc

    def get_inference(self, record_id: str) -> ProductionInferenceRecordDTO:
        self._ensure_open()
        row = self._connection.execute(
            "SELECT * FROM perception_production_inferences WHERE record_id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            raise PerceptionSqlProductionError(
                f"unknown production inference: {record_id}"
            )
        return self._inference_from_row(row)

    def list_inferences(self) -> tuple[ProductionInferenceRecordDTO, ...]:
        self._ensure_open()
        rows = self._connection.execute(
            "SELECT * FROM perception_production_inferences ORDER BY sequence_number"
        ).fetchall()
        return tuple(self._inference_from_row(row) for row in rows)

    def append_outcome(self, outcome: ProductionOutcomeRecordDTO) -> None:
        self._ensure_open()
        self.get_inference(outcome.inference_record_id)
        existing = self._connection.execute(
            "SELECT * FROM perception_production_outcomes "
            "WHERE inference_record_id = ?",
            (outcome.inference_record_id,),
        ).fetchone()
        if existing is not None:
            if self._outcome_from_row(existing) == outcome:
                return
            raise PerceptionSqlProductionError(
                "inference already has a different outcome"
            )
        expected = self._connection.execute(
            "SELECT COALESCE(MAX(outcome_sequence_number), 0) + 1 AS expected "
            "FROM perception_production_outcomes"
        ).fetchone()["expected"]
        if outcome.outcome_sequence_number != expected:
            raise PerceptionSqlProductionError(
                "production outcome sequence is not contiguous"
            )
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO perception_production_outcomes(
                        outcome_id, inference_record_id, outcome_sequence_number,
                        observed_action, source, correct, semantics, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        outcome.outcome_id,
                        outcome.inference_record_id,
                        outcome.outcome_sequence_number,
                        outcome.observed_action,
                        outcome.source,
                        int(outcome.correct),
                        outcome.semantics,
                        outcome.version,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise PerceptionSqlProductionError(
                "production outcome violates durable ledger constraints"
            ) from exc

    def get_outcome_for_inference(
        self, record_id: str
    ) -> ProductionOutcomeRecordDTO | None:
        self._ensure_open()
        row = self._connection.execute(
            "SELECT * FROM perception_production_outcomes "
            "WHERE inference_record_id = ?",
            (record_id,),
        ).fetchone()
        return None if row is None else self._outcome_from_row(row)

    def list_outcomes(self) -> tuple[ProductionOutcomeRecordDTO, ...]:
        self._ensure_open()
        rows = self._connection.execute(
            "SELECT * FROM perception_production_outcomes "
            "ORDER BY outcome_sequence_number"
        ).fetchall()
        return tuple(self._outcome_from_row(row) for row in rows)

    def list_inferences_in_window(
        self,
        *,
        start_sequence_number: int,
        end_sequence_number: int,
        promoted_model_id: str | None = None,
    ) -> tuple[ProductionInferenceRecordDTO, ...]:
        """Return one indexed inclusive sequence window without changing DTO semantics."""

        self._ensure_open()
        if start_sequence_number <= 0 or end_sequence_number < start_sequence_number:
            raise PerceptionSqlProductionError("invalid production inference window")
        parameters: list[object] = [start_sequence_number, end_sequence_number]
        predicate = "sequence_number BETWEEN ? AND ?"
        if promoted_model_id is not None:
            predicate += " AND promoted_model_id = ?"
            parameters.append(promoted_model_id)
        rows = self._connection.execute(
            f"SELECT * FROM perception_production_inferences WHERE {predicate} "
            "ORDER BY sequence_number",
            tuple(parameters),
        ).fetchall()
        return tuple(self._inference_from_row(row) for row in rows)

    @staticmethod
    def _inference_from_row(row: sqlite3.Row) -> ProductionInferenceRecordDTO:
        if row["version"] != PRODUCTION_INFERENCE_RECORD_VERSION:
            raise PerceptionSqlProductionError(
                "unsupported persisted production inference version"
            )
        if row["semantics"] != PRODUCTION_INFERENCE_SEMANTICS:
            raise PerceptionSqlProductionError(
                "unsupported persisted production inference semantics"
            )
        return ProductionInferenceRecordDTO(
            record_id=row["record_id"],
            sequence_number=row["sequence_number"],
            pointer_id=row["pointer_id"],
            pointer_revision=row["pointer_revision"],
            promoted_model_id=row["promoted_model_id"],
            model_id=row["model_id"],
            model_kind=row["model_kind"],
            input_id=row["input_id"],
            interaction_id=row["interaction_id"],
            inference_result_id=row["inference_result_id"],
            selected_action=row["selected_action"],
            margin=row["margin"],
            status=row["status"],
            rejection_threshold=row["rejection_threshold"],
            semantics=row["semantics"],
            version=row["version"],
        )

    @staticmethod
    def _outcome_from_row(row: sqlite3.Row) -> ProductionOutcomeRecordDTO:
        if row["version"] != PRODUCTION_OUTCOME_RECORD_VERSION:
            raise PerceptionSqlProductionError(
                "unsupported persisted production outcome version"
            )
        if row["semantics"] != PRODUCTION_OUTCOME_SEMANTICS:
            raise PerceptionSqlProductionError(
                "unsupported persisted production outcome semantics"
            )
        return ProductionOutcomeRecordDTO(
            outcome_id=row["outcome_id"],
            inference_record_id=row["inference_record_id"],
            outcome_sequence_number=row["outcome_sequence_number"],
            observed_action=row["observed_action"],
            source=row["source"],
            correct=bool(row["correct"]),
            semantics=row["semantics"],
            version=row["version"],
        )

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "SqlitePerceptionProductionLedgerStore":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

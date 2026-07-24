"""Durable SQLite promoted-model lifecycle persistence for Stage P13.

The store implements the P12 DTO-only lifecycle boundary using Python's standard
``sqlite3`` module. Ledger entries and transitions are immutable rows. The active
pointer is a singleton revisioned row. ``append_transition`` begins a transaction and
``replace_active_pointer`` commits it, making the transition and pointer change atomic
for the existing P12 lifecycle service functions.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Final, Mapping

from .lifecycle import (
    ACTIVE_MODEL_POINTER_VERSION,
    ACTIVE_POINTER_SEMANTICS,
    MODEL_LEDGER_ENTRY_VERSION,
    MODEL_LEDGER_SEMANTICS,
    MODEL_TRANSITION_SEMANTICS,
    MODEL_TRANSITION_VERSION,
    ActiveModelPointerDTO,
    ModelLifecycleTransitionDTO,
    PerceptionModelLifecycleError,
    PromotedModelLedgerEntryDTO,
    _empty_pointer,
)
from .promotion import MODEL_PROMOTION_VERSION, PromotedPerceptionModelDTO

SQL_LIFECYCLE_SCHEMA_VERSION: Final = "perception-sql-lifecycle-schema/1"
SQL_LIFECYCLE_STORE_VERSION: Final = "perception-sql-lifecycle-store/1"
SQL_LIFECYCLE_SEMANTICS: Final = (
    "sqlite_append_only_model_ledger_with_atomic_transition_pointer_commit"
)


class PerceptionSqlLifecycleError(PerceptionModelLifecycleError):
    """Raised when the durable lifecycle store cannot preserve its contract."""


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _promoted_model_payload(model: PromotedPerceptionModelDTO) -> Mapping[str, object]:
    return {
        "calibration_id": model.calibration_id,
        "evaluation_split": model.evaluation_split,
        "model_id": model.model_id,
        "model_kind": model.model_kind,
        "promoted_model_id": model.promoted_model_id,
        "promotion_decision_id": model.promotion_decision_id,
        "rejection_threshold": model.rejection_threshold,
        "temporal_window_spec_id": model.temporal_window_spec_id,
        "training_split": model.training_split,
        "validation_comparison_report_id": model.validation_comparison_report_id,
        "version": model.version,
    }


def _promoted_model_from_payload(payload: Mapping[str, object]) -> PromotedPerceptionModelDTO:
    return PromotedPerceptionModelDTO(
        promoted_model_id=str(payload["promoted_model_id"]),
        model_kind=str(payload["model_kind"]),
        model_id=str(payload["model_id"]),
        rejection_threshold=float(payload["rejection_threshold"]),
        calibration_id=str(payload["calibration_id"]),
        promotion_decision_id=str(payload["promotion_decision_id"]),
        validation_comparison_report_id=str(payload["validation_comparison_report_id"]),
        training_split=str(payload["training_split"]),
        evaluation_split=str(payload["evaluation_split"]),
        temporal_window_spec_id=(
            str(payload["temporal_window_spec_id"])
            if payload.get("temporal_window_spec_id") is not None
            else None
        ),
        version=str(payload.get("version", MODEL_PROMOTION_VERSION)),
    )


def _ledger_payload(entry: PromotedModelLedgerEntryDTO) -> Mapping[str, object]:
    return {
        "ledger_entry_id": entry.ledger_entry_id,
        "promoted_model": _promoted_model_payload(entry.promoted_model),
        "registered_by": entry.registered_by,
        "registration_reason": entry.registration_reason,
        "semantics": entry.semantics,
        "test_evaluation_report_id": entry.test_evaluation_report_id,
        "version": entry.version,
    }


def _ledger_from_payload(payload: Mapping[str, object]) -> PromotedModelLedgerEntryDTO:
    promoted_payload = payload["promoted_model"]
    if not isinstance(promoted_payload, dict):
        raise PerceptionSqlLifecycleError("stored promoted model payload is invalid")
    return PromotedModelLedgerEntryDTO(
        ledger_entry_id=str(payload["ledger_entry_id"]),
        promoted_model=_promoted_model_from_payload(promoted_payload),
        test_evaluation_report_id=(
            str(payload["test_evaluation_report_id"])
            if payload.get("test_evaluation_report_id") is not None
            else None
        ),
        registered_by=str(payload["registered_by"]),
        registration_reason=str(payload["registration_reason"]),
        semantics=str(payload.get("semantics", MODEL_LEDGER_SEMANTICS)),
        version=str(payload.get("version", MODEL_LEDGER_ENTRY_VERSION)),
    )


def _transition_payload(item: ModelLifecycleTransitionDTO) -> Mapping[str, object]:
    return {
        "actor": item.actor,
        "next_promoted_model_id": item.next_promoted_model_id,
        "previous_promoted_model_id": item.previous_promoted_model_id,
        "reason": item.reason,
        "related_transition_id": item.related_transition_id,
        "semantics": item.semantics,
        "sequence_number": item.sequence_number,
        "transition_id": item.transition_id,
        "transition_kind": item.transition_kind,
        "version": item.version,
    }


def _transition_from_payload(payload: Mapping[str, object]) -> ModelLifecycleTransitionDTO:
    return ModelLifecycleTransitionDTO(
        transition_id=str(payload["transition_id"]),
        sequence_number=int(payload["sequence_number"]),
        transition_kind=str(payload["transition_kind"]),
        previous_promoted_model_id=(
            str(payload["previous_promoted_model_id"])
            if payload.get("previous_promoted_model_id") is not None
            else None
        ),
        next_promoted_model_id=(
            str(payload["next_promoted_model_id"])
            if payload.get("next_promoted_model_id") is not None
            else None
        ),
        actor=str(payload["actor"]),
        reason=str(payload["reason"]),
        related_transition_id=(
            str(payload["related_transition_id"])
            if payload.get("related_transition_id") is not None
            else None
        ),
        semantics=str(payload.get("semantics", MODEL_TRANSITION_SEMANTICS)),
        version=str(payload.get("version", MODEL_TRANSITION_VERSION)),
    )


class SqlitePerceptionModelLifecycleStore:
    """Restart-safe SQLite implementation of the P12 lifecycle store protocol."""

    version: Final = SQL_LIFECYCLE_STORE_VERSION
    semantics: Final = SQL_LIFECYCLE_SEMANTICS

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        self._connection = sqlite3.connect(
            self.database_path,
            isolation_level=None,
            timeout=30.0,
        )
        self._connection.row_factory = sqlite3.Row
        self._pending_transition = False
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._initialize_schema()
        except Exception:
            self._connection.close()
            raise

    def __enter__(self) -> "SqlitePerceptionModelLifecycleStore":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc is not None and self._pending_transition:
            self._connection.execute("ROLLBACK")
            self._pending_transition = False
        self.close()

    def close(self) -> None:
        if self._pending_transition:
            self._connection.execute("ROLLBACK")
            self._pending_transition = False
        self._connection.close()

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS perception_lifecycle_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS perception_promoted_model_ledger (
                promoted_model_id TEXT PRIMARY KEY,
                ledger_entry_id TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS perception_model_transitions (
                sequence_number INTEGER PRIMARY KEY,
                transition_id TEXT NOT NULL UNIQUE,
                previous_promoted_model_id TEXT,
                next_promoted_model_id TEXT,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(previous_promoted_model_id)
                    REFERENCES perception_promoted_model_ledger(promoted_model_id),
                FOREIGN KEY(next_promoted_model_id)
                    REFERENCES perception_promoted_model_ledger(promoted_model_id)
            );

            CREATE TABLE IF NOT EXISTS perception_active_model_pointer (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                pointer_id TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision >= 0),
                active_promoted_model_id TEXT,
                last_transition_id TEXT,
                semantics TEXT NOT NULL,
                version TEXT NOT NULL,
                FOREIGN KEY(active_promoted_model_id)
                    REFERENCES perception_promoted_model_ledger(promoted_model_id),
                FOREIGN KEY(last_transition_id)
                    REFERENCES perception_model_transitions(transition_id)
            );
            """
        )
        row = self._connection.execute(
            "SELECT value FROM perception_lifecycle_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO perception_lifecycle_metadata(key, value) VALUES('schema_version', ?)",
                (SQL_LIFECYCLE_SCHEMA_VERSION,),
            )
        elif str(row["value"]) != SQL_LIFECYCLE_SCHEMA_VERSION:
            raise PerceptionSqlLifecycleError("unsupported lifecycle database schema version")

        pointer = self._connection.execute(
            "SELECT singleton FROM perception_active_model_pointer WHERE singleton = 1"
        ).fetchone()
        if pointer is None:
            empty = _empty_pointer()
            self._connection.execute(
                """
                INSERT INTO perception_active_model_pointer(
                    singleton, pointer_id, revision, active_promoted_model_id,
                    last_transition_id, semantics, version
                ) VALUES(1, ?, ?, ?, ?, ?, ?)
                """,
                (
                    empty.pointer_id,
                    empty.revision,
                    empty.active_promoted_model_id,
                    empty.last_transition_id,
                    empty.semantics,
                    empty.version,
                ),
            )

    def put_ledger_entry(self, entry: PromotedModelLedgerEntryDTO) -> None:
        payload_json = _canonical_json(_ledger_payload(entry))
        existing = self._connection.execute(
            "SELECT payload_json FROM perception_promoted_model_ledger WHERE promoted_model_id = ?",
            (entry.promoted_model.promoted_model_id,),
        ).fetchone()
        if existing is not None:
            if str(existing["payload_json"]) != payload_json:
                raise PerceptionSqlLifecycleError(
                    "promoted model identity already has different ledger entry"
                )
            return
        self._connection.execute(
            """
            INSERT INTO perception_promoted_model_ledger(
                promoted_model_id, ledger_entry_id, payload_json
            ) VALUES(?, ?, ?)
            """,
            (
                entry.promoted_model.promoted_model_id,
                entry.ledger_entry_id,
                payload_json,
            ),
        )

    def get_ledger_entry(self, promoted_model_id: str) -> PromotedModelLedgerEntryDTO:
        row = self._connection.execute(
            "SELECT payload_json FROM perception_promoted_model_ledger WHERE promoted_model_id = ?",
            (promoted_model_id,),
        ).fetchone()
        if row is None:
            raise PerceptionSqlLifecycleError(
                f"unknown promoted model identity: {promoted_model_id}"
            )
        payload = json.loads(str(row["payload_json"]))
        if not isinstance(payload, dict):
            raise PerceptionSqlLifecycleError("stored ledger payload is invalid")
        return _ledger_from_payload(payload)

    def list_ledger_entries(self) -> tuple[PromotedModelLedgerEntryDTO, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM perception_promoted_model_ledger ORDER BY promoted_model_id"
        ).fetchall()
        return tuple(
            _ledger_from_payload(payload)
            for payload in (json.loads(str(row["payload_json"])) for row in rows)
            if isinstance(payload, dict)
        )

    def append_transition(self, transition: ModelLifecycleTransitionDTO) -> None:
        if self._pending_transition:
            raise PerceptionSqlLifecycleError("another lifecycle transition is already pending")
        self._connection.execute("BEGIN IMMEDIATE")
        self._pending_transition = True
        try:
            pointer = self.get_active_pointer()
            if transition.sequence_number != pointer.revision + 1:
                raise PerceptionSqlLifecycleError(
                    "transition sequence does not follow active pointer revision"
                )
            self._connection.execute(
                """
                INSERT INTO perception_model_transitions(
                    sequence_number, transition_id, previous_promoted_model_id,
                    next_promoted_model_id, payload_json
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    transition.sequence_number,
                    transition.transition_id,
                    transition.previous_promoted_model_id,
                    transition.next_promoted_model_id,
                    _canonical_json(_transition_payload(transition)),
                ),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            self._pending_transition = False
            raise

    def list_transitions(self) -> tuple[ModelLifecycleTransitionDTO, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM perception_model_transitions ORDER BY sequence_number"
        ).fetchall()
        result: list[ModelLifecycleTransitionDTO] = []
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            if not isinstance(payload, dict):
                raise PerceptionSqlLifecycleError("stored transition payload is invalid")
            result.append(_transition_from_payload(payload))
        return tuple(result)

    def get_active_pointer(self) -> ActiveModelPointerDTO:
        row = self._connection.execute(
            """
            SELECT pointer_id, revision, active_promoted_model_id,
                   last_transition_id, semantics, version
            FROM perception_active_model_pointer WHERE singleton = 1
            """
        ).fetchone()
        if row is None:
            raise PerceptionSqlLifecycleError("active model pointer row is missing")
        return ActiveModelPointerDTO(
            pointer_id=str(row["pointer_id"]),
            revision=int(row["revision"]),
            active_promoted_model_id=(
                str(row["active_promoted_model_id"])
                if row["active_promoted_model_id"] is not None
                else None
            ),
            last_transition_id=(
                str(row["last_transition_id"])
                if row["last_transition_id"] is not None
                else None
            ),
            semantics=str(row["semantics"]),
            version=str(row["version"]),
        )

    def replace_active_pointer(
        self,
        pointer: ActiveModelPointerDTO,
        *,
        expected_revision: int,
    ) -> None:
        if not self._pending_transition:
            raise PerceptionSqlLifecycleError(
                "pointer replacement requires a pending transition transaction"
            )
        if pointer.revision != expected_revision + 1:
            self._connection.execute("ROLLBACK")
            self._pending_transition = False
            raise PerceptionSqlLifecycleError("replacement pointer must advance revision by one")
        try:
            cursor = self._connection.execute(
                """
                UPDATE perception_active_model_pointer
                SET pointer_id = ?, revision = ?, active_promoted_model_id = ?,
                    last_transition_id = ?, semantics = ?, version = ?
                WHERE singleton = 1 AND revision = ?
                """,
                (
                    pointer.pointer_id,
                    pointer.revision,
                    pointer.active_promoted_model_id,
                    pointer.last_transition_id,
                    pointer.semantics,
                    pointer.version,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise PerceptionSqlLifecycleError(
                    "active model pointer revision changed during lifecycle update"
                )
            self._connection.execute("COMMIT")
            self._pending_transition = False
        except Exception:
            self._connection.execute("ROLLBACK")
            self._pending_transition = False
            raise

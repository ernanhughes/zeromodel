"""Append-only governed execution attempt journal for Stage P17E.

P17D makes rollback execution restart-recoverable. P17E makes the attempt itself durable and
observable: an approved disposition receives one deterministic attempt identity, a prepared
event before execution, and exactly one terminal event after completion, reconciliation,
idempotent replay, or governed failure.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Mapping

from .compatibility import ModelCompatibilityContractDTO
from .disposition import OperationalRecommendationDispositionDTO
from .governed_execution import (
    PerceptionGovernedExecutionError,
    execute_or_reconcile_approved_rollback,
)
from .lifecycle import PerceptionModelLifecycleStore
from .recommendation import OperationalRecommendationDTO
from .sql_governance import (
    GovernanceExecutionReceiptDTO,
    SqlitePerceptionGovernanceLedgerStore,
)

EXECUTION_ATTEMPT_VERSION: Final = "perception-governed-execution-attempt/1"
EXECUTION_ATTEMPT_EVENT_VERSION: Final = "perception-governed-execution-attempt-event/1"
SQL_EXECUTION_JOURNAL_SCHEMA_VERSION: Final = "perception-sql-execution-journal-schema/1"
EXECUTION_ATTEMPT_EVENT_KINDS: Final = {
    "prepared",
    "completed",
    "reconciled",
    "idempotent",
    "failed",
}
EXECUTION_ATTEMPT_TERMINAL_KINDS: Final = EXECUTION_ATTEMPT_EVENT_KINDS - {"prepared"}


class PerceptionExecutionJournalError(ValueError):
    """Raised when execution-attempt journal contracts are violated."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, object]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


@dataclass(frozen=True)
class GovernedExecutionAttemptDTO:
    attempt_id: str
    recommendation_id: str
    disposition_id: str
    reviewed_pointer_id: str
    reviewed_pointer_revision: int
    reviewed_active_promoted_model_id: str
    target_promoted_model_id: str
    current_contract_id: str
    target_contract_id: str
    version: str = EXECUTION_ATTEMPT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.attempt_id,
                self.recommendation_id,
                self.disposition_id,
                self.reviewed_pointer_id,
                self.reviewed_active_promoted_model_id,
                self.target_promoted_model_id,
                self.current_contract_id,
                self.target_contract_id,
            )
        ):
            raise PerceptionExecutionJournalError("execution attempt identities must be non-empty")
        if self.reviewed_pointer_revision <= 0:
            raise PerceptionExecutionJournalError(
                "execution attempt requires a positive reviewed pointer revision"
            )
        if self.version != EXECUTION_ATTEMPT_VERSION:
            raise PerceptionExecutionJournalError("unsupported execution attempt version")


@dataclass(frozen=True)
class GovernedExecutionAttemptEventDTO:
    event_id: str
    attempt_id: str
    sequence_number: int
    event_kind: str
    receipt_id: str | None
    pointer_revision: int | None
    failure_type: str | None
    failure_message: str | None
    version: str = EXECUTION_ATTEMPT_EVENT_VERSION

    def __post_init__(self) -> None:
        if not self.event_id or not self.attempt_id:
            raise PerceptionExecutionJournalError("attempt event identities must be non-empty")
        if self.sequence_number not in {1, 2}:
            raise PerceptionExecutionJournalError("attempt event sequence must be one or two")
        if self.event_kind not in EXECUTION_ATTEMPT_EVENT_KINDS:
            raise PerceptionExecutionJournalError("unsupported attempt event kind")
        if self.event_kind == "prepared":
            if self.sequence_number != 1 or any(
                value is not None
                for value in (
                    self.receipt_id,
                    self.pointer_revision,
                    self.failure_type,
                    self.failure_message,
                )
            ):
                raise PerceptionExecutionJournalError("prepared event cannot contain terminal data")
        elif self.event_kind == "failed":
            if self.sequence_number != 2 or not self.failure_type or not self.failure_message:
                raise PerceptionExecutionJournalError("failed event requires failure details")
            if self.receipt_id is not None or self.pointer_revision is not None:
                raise PerceptionExecutionJournalError("failed event cannot contain receipt data")
        else:
            if self.sequence_number != 2 or not self.receipt_id or self.pointer_revision is None:
                raise PerceptionExecutionJournalError("successful terminal event requires receipt data")
            if self.pointer_revision <= 0:
                raise PerceptionExecutionJournalError("terminal pointer revision must be positive")
            if self.failure_type is not None or self.failure_message is not None:
                raise PerceptionExecutionJournalError("successful terminal event cannot contain failure data")
        if self.version != EXECUTION_ATTEMPT_EVENT_VERSION:
            raise PerceptionExecutionJournalError("unsupported attempt event version")


def build_governed_execution_attempt(
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
    *,
    current_contract: ModelCompatibilityContractDTO,
    target_contract: ModelCompatibilityContractDTO,
) -> GovernedExecutionAttemptDTO:
    target_id = disposition.selected_target_promoted_model_id
    if disposition.status != "approved" or target_id is None:
        raise PerceptionExecutionJournalError("execution attempt requires an approved rollback target")
    if disposition.recommendation_id != recommendation.recommendation_id:
        raise PerceptionExecutionJournalError("disposition does not belong to recommendation")
    if current_contract.contract_id != recommendation.current_contract_id:
        raise PerceptionExecutionJournalError("current contract does not match recommendation")
    if target_contract.promoted_model_id != target_id:
        raise PerceptionExecutionJournalError("target contract does not describe approved target")
    payload: Mapping[str, object] = {
        "current_contract_id": current_contract.contract_id,
        "disposition_id": disposition.disposition_id,
        "recommendation_id": recommendation.recommendation_id,
        "reviewed_active_promoted_model_id": recommendation.active_promoted_model_id,
        "reviewed_pointer_id": recommendation.active_pointer_id,
        "reviewed_pointer_revision": recommendation.active_pointer_revision,
        "target_contract_id": target_contract.contract_id,
        "target_promoted_model_id": target_id,
        "version": EXECUTION_ATTEMPT_VERSION,
    }
    return GovernedExecutionAttemptDTO(attempt_id=_digest(payload), **payload)


def _event(
    attempt: GovernedExecutionAttemptDTO,
    *,
    sequence_number: int,
    event_kind: str,
    receipt_id: str | None = None,
    pointer_revision: int | None = None,
    failure_type: str | None = None,
    failure_message: str | None = None,
) -> GovernedExecutionAttemptEventDTO:
    payload: Mapping[str, object] = {
        "attempt_id": attempt.attempt_id,
        "event_kind": event_kind,
        "failure_message": failure_message,
        "failure_type": failure_type,
        "pointer_revision": pointer_revision,
        "receipt_id": receipt_id,
        "sequence_number": sequence_number,
        "version": EXECUTION_ATTEMPT_EVENT_VERSION,
    }
    return GovernedExecutionAttemptEventDTO(event_id=_digest(payload), **payload)


class SqliteGovernedExecutionAttemptStore:
    """Append-only SQLite journal for governed execution attempts and events."""

    def __init__(self, database: str | Path) -> None:
        self._connection = sqlite3.connect(str(database))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def __enter__(self) -> "SqliteGovernedExecutionAttemptStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS perception_execution_journal_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS perception_governed_execution_attempts (
                attempt_id TEXT PRIMARY KEY,
                disposition_id TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS perception_governed_execution_attempt_events (
                event_id TEXT PRIMARY KEY,
                attempt_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                event_kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(attempt_id, sequence_number),
                FOREIGN KEY(attempt_id)
                    REFERENCES perception_governed_execution_attempts(attempt_id)
            );
            """
        )
        row = self._connection.execute(
            "SELECT value FROM perception_execution_journal_metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO perception_execution_journal_metadata(key, value) "
                "VALUES('schema_version', ?)",
                (SQL_EXECUTION_JOURNAL_SCHEMA_VERSION,),
            )
            self._connection.commit()
        elif row["value"] != SQL_EXECUTION_JOURNAL_SCHEMA_VERSION:
            raise PerceptionExecutionJournalError("unsupported execution journal schema version")

    def append_attempt(self, attempt: GovernedExecutionAttemptDTO) -> None:
        encoded = json.dumps(asdict(attempt), sort_keys=True, separators=(",", ":"))
        row = self._connection.execute(
            "SELECT attempt_id, payload_json FROM perception_governed_execution_attempts "
            "WHERE disposition_id = ?",
            (attempt.disposition_id,),
        ).fetchone()
        if row is not None:
            if row["attempt_id"] != attempt.attempt_id or row["payload_json"] != encoded:
                raise PerceptionExecutionJournalError(
                    "disposition already has a conflicting execution attempt"
                )
            return
        self._connection.execute(
            "INSERT INTO perception_governed_execution_attempts"
            "(attempt_id, disposition_id, payload_json) VALUES(?, ?, ?)",
            (attempt.attempt_id, attempt.disposition_id, encoded),
        )
        self._connection.commit()

    def append_event(self, event: GovernedExecutionAttemptEventDTO) -> None:
        if self.get_attempt(event.attempt_id) is None:
            raise PerceptionExecutionJournalError("attempt event requires persisted attempt")
        existing = self._connection.execute(
            "SELECT event_id, payload_json FROM perception_governed_execution_attempt_events "
            "WHERE attempt_id = ? AND sequence_number = ?",
            (event.attempt_id, event.sequence_number),
        ).fetchone()
        encoded = json.dumps(asdict(event), sort_keys=True, separators=(",", ":"))
        if existing is not None:
            if existing["event_id"] != event.event_id or existing["payload_json"] != encoded:
                raise PerceptionExecutionJournalError("attempt sequence already has another event")
            return
        events = self.list_events(event.attempt_id)
        expected_sequence = len(events) + 1
        if event.sequence_number != expected_sequence:
            raise PerceptionExecutionJournalError("attempt event sequence is not contiguous")
        if events and events[-1].event_kind in EXECUTION_ATTEMPT_TERMINAL_KINDS:
            raise PerceptionExecutionJournalError("terminal execution attempt cannot be extended")
        self._connection.execute(
            "INSERT INTO perception_governed_execution_attempt_events"
            "(event_id, attempt_id, sequence_number, event_kind, payload_json) "
            "VALUES(?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.attempt_id,
                event.sequence_number,
                event.event_kind,
                encoded,
            ),
        )
        self._connection.commit()

    def get_attempt(self, attempt_id: str) -> GovernedExecutionAttemptDTO | None:
        row = self._connection.execute(
            "SELECT payload_json FROM perception_governed_execution_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        return None if row is None else GovernedExecutionAttemptDTO(**json.loads(row["payload_json"]))

    def list_events(self, attempt_id: str) -> tuple[GovernedExecutionAttemptEventDTO, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM perception_governed_execution_attempt_events "
            "WHERE attempt_id = ? ORDER BY sequence_number",
            (attempt_id,),
        ).fetchall()
        return tuple(
            GovernedExecutionAttemptEventDTO(**json.loads(row["payload_json"])) for row in rows
        )


def execute_journaled_approved_rollback(
    lifecycle_store: PerceptionModelLifecycleStore,
    governance_store: SqlitePerceptionGovernanceLedgerStore,
    attempt_store: SqliteGovernedExecutionAttemptStore,
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
    *,
    current_contract: ModelCompatibilityContractDTO,
    target_contract: ModelCompatibilityContractDTO,
) -> tuple[GovernedExecutionAttemptDTO, GovernanceExecutionReceiptDTO]:
    """Journal preparation, execute or reconcile once, and append one terminal event."""

    attempt = build_governed_execution_attempt(
        recommendation,
        disposition,
        current_contract=current_contract,
        target_contract=target_contract,
    )
    attempt_store.append_attempt(attempt)
    events = attempt_store.list_events(attempt.attempt_id)
    if events and events[-1].event_kind in EXECUTION_ATTEMPT_TERMINAL_KINDS:
        terminal = events[-1]
        if terminal.event_kind == "failed":
            raise PerceptionExecutionJournalError(
                f"execution attempt is terminally failed: {terminal.failure_message}"
            )
        receipts = tuple(
            item
            for item in governance_store.list_execution_receipts()
            if item.receipt_id == terminal.receipt_id
        )
        if len(receipts) != 1:
            raise PerceptionExecutionJournalError(
                "terminal attempt references missing or ambiguous execution receipt"
            )
        return attempt, receipts[0]
    if not events:
        attempt_store.append_event(_event(attempt, sequence_number=1, event_kind="prepared"))

    existing_receipt = next(
        (
            item
            for item in governance_store.list_execution_receipts()
            if item.disposition_id == disposition.disposition_id
        ),
        None,
    )
    pointer_before = lifecycle_store.get_active_pointer()
    reviewed_pre_state = all(
        (
            pointer_before.pointer_id == recommendation.active_pointer_id,
            pointer_before.revision == recommendation.active_pointer_revision,
            pointer_before.active_promoted_model_id == recommendation.active_promoted_model_id,
        )
    )
    try:
        receipt = execute_or_reconcile_approved_rollback(
            lifecycle_store,
            governance_store,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
    except (PerceptionGovernedExecutionError, ValueError) as error:
        attempt_store.append_event(
            _event(
                attempt,
                sequence_number=2,
                event_kind="failed",
                failure_type=type(error).__name__,
                failure_message=str(error),
            )
        )
        raise PerceptionExecutionJournalError(str(error)) from error

    if existing_receipt is not None:
        event_kind = "idempotent"
    elif reviewed_pre_state:
        event_kind = "completed"
    else:
        event_kind = "reconciled"
    attempt_store.append_event(
        _event(
            attempt,
            sequence_number=2,
            event_kind=event_kind,
            receipt_id=receipt.receipt_id,
            pointer_revision=receipt.pointer_revision,
        )
    )
    return attempt, receipt

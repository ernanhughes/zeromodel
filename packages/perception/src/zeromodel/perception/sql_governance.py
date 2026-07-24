"""Durable recommendation, disposition, and execution ledger for Stage P17C."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Iterable

from .compatibility import RollbackCompatibilityAssessmentDTO
from .disposition import OperationalRecommendationDispositionDTO
from .lifecycle import ActiveModelPointerDTO, ModelLifecycleTransitionDTO
from .recommendation import OperationalRecommendationDTO

SQL_GOVERNANCE_SCHEMA_VERSION: Final = "perception-sql-governance-schema/1"
SQL_GOVERNANCE_STORE_VERSION: Final = "perception-sql-governance-store/1"
GOVERNANCE_EXECUTION_RECEIPT_VERSION: Final = "perception-governance-execution-receipt/1"


class PerceptionSqlGovernanceError(ValueError):
    """Raised when durable P17 governance contracts are violated."""


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _assessment(payload: dict[str, object]) -> RollbackCompatibilityAssessmentDTO:
    return RollbackCompatibilityAssessmentDTO(**payload)  # type: ignore[arg-type]


def _recommendation(payload: dict[str, object]) -> OperationalRecommendationDTO:
    payload = dict(payload)
    payload["assessed_candidates"] = tuple(
        _assessment(item) for item in payload["assessed_candidates"]  # type: ignore[union-attr]
    )
    return OperationalRecommendationDTO(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True)
class GovernanceExecutionReceiptDTO:
    receipt_id: str
    disposition_id: str
    recommendation_id: str
    assessment_id: str
    transition_id: str
    pointer_id: str
    pointer_revision: int
    resulting_promoted_model_id: str
    version: str = GOVERNANCE_EXECUTION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.receipt_id,
                self.disposition_id,
                self.recommendation_id,
                self.assessment_id,
                self.transition_id,
                self.pointer_id,
                self.resulting_promoted_model_id,
            )
        ):
            raise PerceptionSqlGovernanceError("execution receipt identities must be non-empty")
        if self.pointer_revision <= 0:
            raise PerceptionSqlGovernanceError("execution receipt pointer revision must be positive")
        if self.version != GOVERNANCE_EXECUTION_RECEIPT_VERSION:
            raise PerceptionSqlGovernanceError("unsupported execution receipt version")

    @classmethod
    def from_execution(
        cls,
        disposition: OperationalRecommendationDispositionDTO,
        assessment: RollbackCompatibilityAssessmentDTO,
        transition: ModelLifecycleTransitionDTO,
        pointer: ActiveModelPointerDTO,
    ) -> "GovernanceExecutionReceiptDTO":
        if disposition.status != "approved":
            raise PerceptionSqlGovernanceError("execution receipt requires approved disposition")
        if disposition.selected_assessment_id != assessment.assessment_id:
            raise PerceptionSqlGovernanceError("assessment does not match disposition")
        if pointer.active_promoted_model_id is None:
            raise PerceptionSqlGovernanceError("execution receipt requires resulting active model")
        payload = {
            "assessment_id": assessment.assessment_id,
            "disposition_id": disposition.disposition_id,
            "pointer_id": pointer.pointer_id,
            "pointer_revision": pointer.revision,
            "recommendation_id": disposition.recommendation_id,
            "resulting_promoted_model_id": pointer.active_promoted_model_id,
            "transition_id": transition.transition_id,
            "version": GOVERNANCE_EXECUTION_RECEIPT_VERSION,
        }
        import hashlib

        receipt_id = f"sha256:{hashlib.sha256(_json(payload).encode()).hexdigest()}"
        return cls(receipt_id=receipt_id, **payload)


class SqlitePerceptionGovernanceLedgerStore:
    """Append-only SQLite store for P17 governance artifacts."""

    def __init__(self, database: str | Path) -> None:
        self._connection = sqlite3.connect(str(database))
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def __enter__(self) -> "SqlitePerceptionGovernanceLedgerStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS perception_governance_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS perception_operational_recommendations (
                recommendation_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS perception_operational_dispositions (
                disposition_id TEXT PRIMARY KEY,
                recommendation_id TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(recommendation_id)
                    REFERENCES perception_operational_recommendations(recommendation_id)
            );
            CREATE TABLE IF NOT EXISTS perception_governance_execution_receipts (
                receipt_id TEXT PRIMARY KEY,
                disposition_id TEXT NOT NULL UNIQUE,
                recommendation_id TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(disposition_id)
                    REFERENCES perception_operational_dispositions(disposition_id)
            );
            """
        )
        row = self._connection.execute(
            "SELECT value FROM perception_governance_metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO perception_governance_metadata(key, value) VALUES('schema_version', ?)",
                (SQL_GOVERNANCE_SCHEMA_VERSION,),
            )
            self._connection.commit()
        elif row["value"] != SQL_GOVERNANCE_SCHEMA_VERSION:
            raise PerceptionSqlGovernanceError("unsupported governance schema version")
        self._connection.execute("PRAGMA foreign_keys = ON")

    def _append(self, table: str, identity_column: str, identity: str, payload: object) -> None:
        encoded = _json(asdict(payload))
        row = self._connection.execute(
            f"SELECT payload_json FROM {table} WHERE {identity_column} = ?", (identity,)
        ).fetchone()
        if row is not None:
            if row["payload_json"] != encoded:
                raise PerceptionSqlGovernanceError("conflicting immutable governance artifact")
            return
        try:
            self._connection.execute(
                f"INSERT INTO {table}({identity_column}, payload_json) VALUES(?, ?)",
                (identity, encoded),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as error:
            self._connection.rollback()
            raise PerceptionSqlGovernanceError(str(error)) from error

    def append_recommendation(self, item: OperationalRecommendationDTO) -> None:
        self._append(
            "perception_operational_recommendations",
            "recommendation_id",
            item.recommendation_id,
            item,
        )

    def append_disposition(self, item: OperationalRecommendationDispositionDTO) -> None:
        if self.get_recommendation(item.recommendation_id) is None:
            raise PerceptionSqlGovernanceError("disposition requires persisted recommendation")
        encoded = _json(asdict(item))
        existing = self._connection.execute(
            "SELECT disposition_id, payload_json FROM perception_operational_dispositions "
            "WHERE recommendation_id = ?",
            (item.recommendation_id,),
        ).fetchone()
        if existing is not None:
            if existing["disposition_id"] != item.disposition_id or existing["payload_json"] != encoded:
                raise PerceptionSqlGovernanceError("recommendation already has a final disposition")
            return
        try:
            self._connection.execute(
                "INSERT INTO perception_operational_dispositions"
                "(disposition_id, recommendation_id, payload_json) VALUES(?, ?, ?)",
                (item.disposition_id, item.recommendation_id, encoded),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as error:
            self._connection.rollback()
            raise PerceptionSqlGovernanceError(str(error)) from error

    def append_execution_receipt(self, item: GovernanceExecutionReceiptDTO) -> None:
        disposition = self.get_disposition(item.disposition_id)
        if disposition is None or disposition.status != "approved":
            raise PerceptionSqlGovernanceError("execution receipt requires persisted approval")
        if disposition.recommendation_id != item.recommendation_id:
            raise PerceptionSqlGovernanceError("receipt recommendation does not match disposition")
        encoded = _json(asdict(item))
        existing = self._connection.execute(
            "SELECT receipt_id, payload_json FROM perception_governance_execution_receipts "
            "WHERE disposition_id = ? OR recommendation_id = ?",
            (item.disposition_id, item.recommendation_id),
        ).fetchone()
        if existing is not None:
            if existing["receipt_id"] != item.receipt_id or existing["payload_json"] != encoded:
                raise PerceptionSqlGovernanceError("approved disposition has already been executed")
            return
        self._connection.execute(
            "INSERT INTO perception_governance_execution_receipts"
            "(receipt_id, disposition_id, recommendation_id, payload_json) VALUES(?, ?, ?, ?)",
            (item.receipt_id, item.disposition_id, item.recommendation_id, encoded),
        )
        self._connection.commit()

    def get_recommendation(self, recommendation_id: str) -> OperationalRecommendationDTO | None:
        row = self._connection.execute(
            "SELECT payload_json FROM perception_operational_recommendations WHERE recommendation_id = ?",
            (recommendation_id,),
        ).fetchone()
        return None if row is None else _recommendation(json.loads(row["payload_json"]))

    def get_disposition(self, disposition_id: str) -> OperationalRecommendationDispositionDTO | None:
        row = self._connection.execute(
            "SELECT payload_json FROM perception_operational_dispositions WHERE disposition_id = ?",
            (disposition_id,),
        ).fetchone()
        return None if row is None else OperationalRecommendationDispositionDTO(**json.loads(row["payload_json"]))

    def get_execution_receipt(self, receipt_id: str) -> GovernanceExecutionReceiptDTO | None:
        row = self._connection.execute(
            "SELECT payload_json FROM perception_governance_execution_receipts WHERE receipt_id = ?",
            (receipt_id,),
        ).fetchone()
        return None if row is None else GovernanceExecutionReceiptDTO(**json.loads(row["payload_json"]))

    def _list_payloads(self, table: str, order_column: str) -> Iterable[dict[str, object]]:
        rows = self._connection.execute(
            f"SELECT payload_json FROM {table} ORDER BY {order_column}"
        ).fetchall()
        return tuple(json.loads(row["payload_json"]) for row in rows)

    def list_recommendations(self) -> tuple[OperationalRecommendationDTO, ...]:
        return tuple(
            _recommendation(item)
            for item in self._list_payloads(
                "perception_operational_recommendations", "recommendation_id"
            )
        )

    def list_dispositions(self) -> tuple[OperationalRecommendationDispositionDTO, ...]:
        return tuple(
            OperationalRecommendationDispositionDTO(**item)  # type: ignore[arg-type]
            for item in self._list_payloads(
                "perception_operational_dispositions", "disposition_id"
            )
        )

    def list_execution_receipts(self) -> tuple[GovernanceExecutionReceiptDTO, ...]:
        return tuple(
            GovernanceExecutionReceiptDTO(**item)  # type: ignore[arg-type]
            for item in self._list_payloads(
                "perception_governance_execution_receipts", "receipt_id"
            )
        )

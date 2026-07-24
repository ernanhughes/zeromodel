"""Durable certification-execution admission ledger for Stage P17K."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from .certification_audit import (
    CertificationIntegrityAuditReportDTO,
    CertificationIntegrityFindingDTO,
)
from .certification_gate import (
    CertificationExecutionGateDTO,
    execute_certification_gated_approved_rollback,
)
from .compatibility import ModelCompatibilityContractDTO
from .disposition import OperationalRecommendationDispositionDTO
from .execution_journal import GovernedExecutionAttemptDTO, SqliteGovernedExecutionAttemptStore
from .lifecycle import PerceptionModelLifecycleStore
from .recommendation import OperationalRecommendationDTO
from .sql_certification import (
    GovernanceExecutionCertificationBundleDTO,
    SqliteGovernanceCertificationStore,
)
from .sql_governance import GovernanceExecutionReceiptDTO, SqlitePerceptionGovernanceLedgerStore

SQL_ADMISSION_SCHEMA_VERSION: Final = "perception-sql-admission-schema/1"
SQL_ADMISSION_STORE_VERSION: Final = "perception-sql-admission-store/1"


class PerceptionSqlAdmissionError(ValueError):
    """Raised when durable execution-admission contracts are violated."""


@dataclass(frozen=True)
class CertificationExecutionAdmissionBundleDTO:
    gate: CertificationExecutionGateDTO
    preflight: CertificationIntegrityAuditReportDTO
    postflight: CertificationIntegrityAuditReportDTO

    def __post_init__(self) -> None:
        if self.gate.preflight_report_id != self.preflight.report_id:
            raise PerceptionSqlAdmissionError("gate does not reference supplied preflight report")
        if self.gate.postflight_report_id != self.postflight.report_id:
            raise PerceptionSqlAdmissionError("gate does not reference supplied postflight report")
        if self.preflight.status != "valid" or self.postflight.status != "valid":
            raise PerceptionSqlAdmissionError("admission bundle requires valid preflight and postflight")


def _decode_report(payload: dict[str, object]) -> CertificationIntegrityAuditReportDTO:
    data = dict(payload)
    data["findings"] = tuple(
        CertificationIntegrityFindingDTO(
            **{**item, "related_ids": tuple(item["related_ids"])}
        )
        for item in data["findings"]
    )
    return CertificationIntegrityAuditReportDTO(**data)


class SqliteCertificationExecutionAdmissionStore:
    """Append-only SQLite store for P17J admission and postcondition proof."""

    def __init__(self, database: str | Path) -> None:
        self._connection = sqlite3.connect(str(database))
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def __enter__(self) -> "SqliteCertificationExecutionAdmissionStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS perception_admission_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS perception_execution_admissions (
                gate_id TEXT PRIMARY KEY,
                certification_id TEXT NOT NULL UNIQUE,
                attempt_id TEXT NOT NULL UNIQUE,
                gate_json TEXT NOT NULL,
                preflight_json TEXT NOT NULL,
                postflight_json TEXT NOT NULL
            );
            """
        )
        row = self._connection.execute(
            "SELECT value FROM perception_admission_metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO perception_admission_metadata(key, value) VALUES('schema_version', ?)",
                (SQL_ADMISSION_SCHEMA_VERSION,),
            )
            self._connection.commit()
        elif row["value"] != SQL_ADMISSION_SCHEMA_VERSION:
            raise PerceptionSqlAdmissionError("unsupported execution-admission schema version")

    def append_admission(self, bundle: CertificationExecutionAdmissionBundleDTO) -> None:
        gate = bundle.gate
        encoded = (
            json.dumps(asdict(gate), sort_keys=True, separators=(",", ":")),
            json.dumps(asdict(bundle.preflight), sort_keys=True, separators=(",", ":")),
            json.dumps(asdict(bundle.postflight), sort_keys=True, separators=(",", ":")),
        )
        row = self._connection.execute(
            "SELECT gate_id, gate_json, preflight_json, postflight_json "
            "FROM perception_execution_admissions WHERE certification_id=? OR attempt_id=?",
            (gate.certification_id, gate.attempt_id),
        ).fetchone()
        if row is not None:
            existing = (row["gate_json"], row["preflight_json"], row["postflight_json"])
            if row["gate_id"] != gate.gate_id or existing != encoded:
                raise PerceptionSqlAdmissionError(
                    "certification or attempt already has a conflicting admission"
                )
            return
        self._connection.execute(
            "INSERT INTO perception_execution_admissions"
            "(gate_id, certification_id, attempt_id, gate_json, preflight_json, postflight_json) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (gate.gate_id, gate.certification_id, gate.attempt_id, *encoded),
        )
        self._connection.commit()

    def get_admission(self, gate_id: str) -> CertificationExecutionAdmissionBundleDTO | None:
        row = self._connection.execute(
            "SELECT gate_json, preflight_json, postflight_json "
            "FROM perception_execution_admissions WHERE gate_id=?",
            (gate_id,),
        ).fetchone()
        if row is None:
            return None
        return CertificationExecutionAdmissionBundleDTO(
            gate=CertificationExecutionGateDTO(**json.loads(row["gate_json"])),
            preflight=_decode_report(json.loads(row["preflight_json"])),
            postflight=_decode_report(json.loads(row["postflight_json"])),
        )

    def list_admissions(self) -> tuple[CertificationExecutionAdmissionBundleDTO, ...]:
        rows = self._connection.execute(
            "SELECT gate_id FROM perception_execution_admissions ORDER BY gate_id"
        ).fetchall()
        return tuple(
            item
            for row in rows
            if (item := self.get_admission(row["gate_id"])) is not None
        )


def execute_and_persist_certification_admission(
    lifecycle_store: PerceptionModelLifecycleStore,
    governance_store: SqlitePerceptionGovernanceLedgerStore,
    attempt_store: SqliteGovernedExecutionAttemptStore,
    certification_store: SqliteGovernanceCertificationStore,
    admission_store: SqliteCertificationExecutionAdmissionStore,
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
    *,
    current_contract: ModelCompatibilityContractDTO,
    target_contract: ModelCompatibilityContractDTO,
) -> tuple[
    CertificationExecutionAdmissionBundleDTO,
    GovernanceExecutionCertificationBundleDTO,
    GovernedExecutionAttemptDTO,
    GovernanceExecutionReceiptDTO,
]:
    gate, certification, attempt, receipt, postflight = (
        execute_certification_gated_approved_rollback(
            lifecycle_store,
            governance_store,
            attempt_store,
            certification_store,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )
    )
    preflight = CertificationIntegrityAuditReportDTO(
        report_id=gate.preflight_report_id,
        status="valid",
        governance_audit_report_id=certification.pre_audit.report_id,
        governance_audit_status=certification.pre_audit.status,
        certification_count=max(postflight.certification_count - 1, 0),
        successful_attempt_count=max(postflight.successful_attempt_count - 1, 0),
        finding_count=0,
        findings=(),
    )
    bundle = CertificationExecutionAdmissionBundleDTO(
        gate=gate,
        preflight=preflight,
        postflight=postflight,
    )
    admission_store.append_admission(bundle)
    restored = admission_store.get_admission(gate.gate_id)
    if restored != bundle:
        raise PerceptionSqlAdmissionError("persisted admission differs from completed gate bundle")
    return bundle, certification, attempt, receipt

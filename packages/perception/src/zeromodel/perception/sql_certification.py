"""Durable audit-gate certification ledger for Stage P17H.

P17G returns a content-addressed gate after a governed rollback, but the gate and the audit
reports it references are otherwise process-local. P17H persists the complete certification
bundle without becoming another lifecycle authority.
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
from .execution_journal import GovernedExecutionAttemptDTO, SqliteGovernedExecutionAttemptStore
from .governance_audit import (
    GovernanceIntegrityAuditReportDTO,
    GovernanceIntegrityFindingDTO,
    audit_governance_integrity,
)
from .governance_gate import GovernanceExecutionGateDTO, execute_audit_gated_approved_rollback
from .lifecycle import PerceptionModelLifecycleStore
from .recommendation import OperationalRecommendationDTO
from .sql_governance import GovernanceExecutionReceiptDTO, SqlitePerceptionGovernanceLedgerStore

GOVERNANCE_CERTIFICATION_VERSION: Final = "perception-governance-certification/1"
SQL_CERTIFICATION_SCHEMA_VERSION: Final = "perception-sql-certification-schema/1"
SQL_CERTIFICATION_STORE_VERSION: Final = "perception-sql-certification-store/1"
GOVERNANCE_CERTIFICATION_SEMANTICS: Final = (
    "durable_bundle_of_execution_gate_and_exact_pre_and_post_integrity_audits"
)


class PerceptionSqlCertificationError(ValueError):
    """Raised when durable governance certification contracts are violated."""


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
class GovernanceExecutionCertificationDTO:
    certification_id: str
    gate_id: str
    recommendation_id: str
    disposition_id: str
    attempt_id: str
    receipt_id: str
    pre_audit_report_id: str
    post_audit_report_id: str
    resulting_pointer_revision: int
    semantics: str = GOVERNANCE_CERTIFICATION_SEMANTICS
    version: str = GOVERNANCE_CERTIFICATION_VERSION

    def __post_init__(self) -> None:
        identities = (
            self.certification_id,
            self.gate_id,
            self.recommendation_id,
            self.disposition_id,
            self.attempt_id,
            self.receipt_id,
            self.pre_audit_report_id,
            self.post_audit_report_id,
        )
        if not all(identities):
            raise PerceptionSqlCertificationError("certification identities must be non-empty")
        if self.resulting_pointer_revision <= 0:
            raise PerceptionSqlCertificationError(
                "certification requires a positive resulting pointer revision"
            )
        if self.semantics != GOVERNANCE_CERTIFICATION_SEMANTICS:
            raise PerceptionSqlCertificationError("unsupported certification semantics")
        if self.version != GOVERNANCE_CERTIFICATION_VERSION:
            raise PerceptionSqlCertificationError("unsupported certification version")


@dataclass(frozen=True)
class GovernanceExecutionCertificationBundleDTO:
    certification: GovernanceExecutionCertificationDTO
    gate: GovernanceExecutionGateDTO
    pre_audit: GovernanceIntegrityAuditReportDTO
    post_audit: GovernanceIntegrityAuditReportDTO

    def __post_init__(self) -> None:
        certification = self.certification
        gate = self.gate
        if certification.gate_id != gate.gate_id:
            raise PerceptionSqlCertificationError("certification does not reference supplied gate")
        for field in (
            "recommendation_id",
            "disposition_id",
            "attempt_id",
            "receipt_id",
            "pre_audit_report_id",
            "post_audit_report_id",
            "resulting_pointer_revision",
        ):
            if getattr(certification, field) != getattr(gate, field):
                raise PerceptionSqlCertificationError(
                    f"certification and gate disagree on {field}"
                )
        if gate.pre_audit_report_id != self.pre_audit.report_id:
            raise PerceptionSqlCertificationError("gate does not reference supplied pre-audit")
        if gate.post_audit_report_id != self.post_audit.report_id:
            raise PerceptionSqlCertificationError("gate does not reference supplied post-audit")
        if self.post_audit.status != "valid":
            raise PerceptionSqlCertificationError("certification requires a valid post-audit")
        if self.post_audit.active_pointer_revision != gate.resulting_pointer_revision:
            raise PerceptionSqlCertificationError(
                "post-audit pointer revision does not match certified result"
            )


def build_governance_execution_certification(
    gate: GovernanceExecutionGateDTO,
    pre_audit: GovernanceIntegrityAuditReportDTO,
    post_audit: GovernanceIntegrityAuditReportDTO,
) -> GovernanceExecutionCertificationBundleDTO:
    payload: Mapping[str, object] = {
        "attempt_id": gate.attempt_id,
        "disposition_id": gate.disposition_id,
        "gate_id": gate.gate_id,
        "post_audit_report_id": post_audit.report_id,
        "pre_audit_report_id": pre_audit.report_id,
        "receipt_id": gate.receipt_id,
        "recommendation_id": gate.recommendation_id,
        "resulting_pointer_revision": gate.resulting_pointer_revision,
        "semantics": GOVERNANCE_CERTIFICATION_SEMANTICS,
        "version": GOVERNANCE_CERTIFICATION_VERSION,
    }
    certification = GovernanceExecutionCertificationDTO(
        certification_id=_digest(payload), **payload
    )
    return GovernanceExecutionCertificationBundleDTO(
        certification=certification,
        gate=gate,
        pre_audit=pre_audit,
        post_audit=post_audit,
    )


def _decode_audit(payload: Mapping[str, object]) -> GovernanceIntegrityAuditReportDTO:
    data = dict(payload)
    data["findings"] = tuple(
        GovernanceIntegrityFindingDTO(
            **{**item, "related_ids": tuple(item["related_ids"])}
        )
        for item in data["findings"]
    )
    return GovernanceIntegrityAuditReportDTO(**data)


class SqliteGovernanceCertificationStore:
    """Append-only SQLite store for complete execution certification bundles."""

    def __init__(self, database: str | Path) -> None:
        self._connection = sqlite3.connect(str(database))
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def __enter__(self) -> "SqliteGovernanceCertificationStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS perception_certification_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS perception_governance_certifications (
                certification_id TEXT PRIMARY KEY,
                gate_id TEXT NOT NULL UNIQUE,
                attempt_id TEXT NOT NULL UNIQUE,
                disposition_id TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                gate_json TEXT NOT NULL,
                pre_audit_json TEXT NOT NULL,
                post_audit_json TEXT NOT NULL
            );
            """
        )
        row = self._connection.execute(
            "SELECT value FROM perception_certification_metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO perception_certification_metadata(key, value) VALUES('schema_version', ?)",
                (SQL_CERTIFICATION_SCHEMA_VERSION,),
            )
            self._connection.commit()
        elif row["value"] != SQL_CERTIFICATION_SCHEMA_VERSION:
            raise PerceptionSqlCertificationError("unsupported certification schema version")

    def append_certification(
        self, bundle: GovernanceExecutionCertificationBundleDTO
    ) -> None:
        certification = bundle.certification
        encoded = (
            json.dumps(asdict(certification), sort_keys=True, separators=(",", ":")),
            json.dumps(asdict(bundle.gate), sort_keys=True, separators=(",", ":")),
            json.dumps(asdict(bundle.pre_audit), sort_keys=True, separators=(",", ":")),
            json.dumps(asdict(bundle.post_audit), sort_keys=True, separators=(",", ":")),
        )
        row = self._connection.execute(
            "SELECT certification_id, payload_json, gate_json, pre_audit_json, post_audit_json "
            "FROM perception_governance_certifications WHERE attempt_id = ? OR disposition_id = ?",
            (certification.attempt_id, certification.disposition_id),
        ).fetchone()
        if row is not None:
            existing = (
                row["payload_json"],
                row["gate_json"],
                row["pre_audit_json"],
                row["post_audit_json"],
            )
            if row["certification_id"] != certification.certification_id or existing != encoded:
                raise PerceptionSqlCertificationError(
                    "attempt or disposition already has a conflicting certification"
                )
            return
        self._connection.execute(
            "INSERT INTO perception_governance_certifications"
            "(certification_id, gate_id, attempt_id, disposition_id, payload_json, gate_json, "
            "pre_audit_json, post_audit_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                certification.certification_id,
                certification.gate_id,
                certification.attempt_id,
                certification.disposition_id,
                *encoded,
            ),
        )
        self._connection.commit()

    def get_certification(
        self, certification_id: str
    ) -> GovernanceExecutionCertificationBundleDTO | None:
        row = self._connection.execute(
            "SELECT payload_json, gate_json, pre_audit_json, post_audit_json "
            "FROM perception_governance_certifications WHERE certification_id = ?",
            (certification_id,),
        ).fetchone()
        if row is None:
            return None
        return GovernanceExecutionCertificationBundleDTO(
            certification=GovernanceExecutionCertificationDTO(
                **json.loads(row["payload_json"])
            ),
            gate=GovernanceExecutionGateDTO(**json.loads(row["gate_json"])),
            pre_audit=_decode_audit(json.loads(row["pre_audit_json"])),
            post_audit=_decode_audit(json.loads(row["post_audit_json"])),
        )

    def list_certifications(self) -> tuple[GovernanceExecutionCertificationBundleDTO, ...]:
        rows = self._connection.execute(
            "SELECT certification_id FROM perception_governance_certifications "
            "ORDER BY certification_id"
        ).fetchall()
        return tuple(
            bundle
            for row in rows
            if (bundle := self.get_certification(row["certification_id"])) is not None
        )


def execute_and_certify_audit_gated_rollback(
    lifecycle_store: PerceptionModelLifecycleStore,
    governance_store: SqlitePerceptionGovernanceLedgerStore,
    attempt_store: SqliteGovernedExecutionAttemptStore,
    certification_store: SqliteGovernanceCertificationStore,
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
    *,
    current_contract: ModelCompatibilityContractDTO,
    target_contract: ModelCompatibilityContractDTO,
) -> tuple[
    GovernanceExecutionCertificationBundleDTO,
    GovernedExecutionAttemptDTO,
    GovernanceExecutionReceiptDTO,
]:
    """Execute through P17G and durably persist the exact surrounding audit evidence."""

    pre_audit = audit_governance_integrity(
        lifecycle_store, governance_store, attempt_store
    )
    gate, attempt, receipt, post_audit = execute_audit_gated_approved_rollback(
        lifecycle_store,
        governance_store,
        attempt_store,
        recommendation,
        disposition,
        current_contract=current_contract,
        target_contract=target_contract,
    )
    if gate.pre_audit_report_id != pre_audit.report_id:
        raise PerceptionSqlCertificationError(
            "execution gate pre-audit differs from certification pre-audit"
        )
    bundle = build_governance_execution_certification(gate, pre_audit, post_audit)
    certification_store.append_certification(bundle)
    return bundle, attempt, receipt

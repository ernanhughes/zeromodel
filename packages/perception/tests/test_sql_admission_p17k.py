from __future__ import annotations

from dataclasses import replace

import pytest

from zeromodel.perception.certification_audit import CertificationIntegrityAuditReportDTO
from zeromodel.perception.certification_gate import CertificationExecutionGateDTO
from zeromodel.perception.sql_admission import (
    CertificationExecutionAdmissionBundleDTO,
    PerceptionSqlAdmissionError,
    SqliteCertificationExecutionAdmissionStore,
)


def _report(identity: str, *, certifications: int) -> CertificationIntegrityAuditReportDTO:
    return CertificationIntegrityAuditReportDTO(
        report_id=identity,
        status="valid",
        governance_audit_report_id=f"{identity}:governance",
        governance_audit_status="valid",
        certification_count=certifications,
        successful_attempt_count=certifications,
        finding_count=0,
        findings=(),
    )


def _bundle() -> CertificationExecutionAdmissionBundleDTO:
    preflight = _report("sha256:preflight", certifications=0)
    postflight = _report("sha256:postflight", certifications=1)
    gate = CertificationExecutionGateDTO(
        gate_id="sha256:gate",
        certification_id="sha256:certification",
        attempt_id="sha256:attempt",
        receipt_id="sha256:receipt",
        preflight_report_id=preflight.report_id,
        postflight_report_id=postflight.report_id,
        resulting_pointer_revision=3,
    )
    return CertificationExecutionAdmissionBundleDTO(
        gate=gate,
        preflight=preflight,
        postflight=postflight,
    )


def test_admission_survives_restart_and_exact_reappend_is_idempotent(tmp_path) -> None:
    database = tmp_path / "admission.sqlite3"
    bundle = _bundle()
    with SqliteCertificationExecutionAdmissionStore(database) as store:
        store.append_admission(bundle)
        store.append_admission(bundle)

    with SqliteCertificationExecutionAdmissionStore(database) as reopened:
        assert reopened.get_admission(bundle.gate.gate_id) == bundle
        assert reopened.list_admissions() == (bundle,)


def test_conflicting_admission_for_same_attempt_is_rejected(tmp_path) -> None:
    bundle = _bundle()
    conflicting = replace(
        bundle,
        gate=replace(bundle.gate, gate_id="sha256:other-gate"),
    )
    with SqliteCertificationExecutionAdmissionStore(
        tmp_path / "admission.sqlite3"
    ) as store:
        store.append_admission(bundle)
        with pytest.raises(PerceptionSqlAdmissionError, match="conflicting admission"):
            store.append_admission(conflicting)


def test_bundle_requires_exact_valid_preflight_and_postflight() -> None:
    bundle = _bundle()
    with pytest.raises(PerceptionSqlAdmissionError, match="preflight report"):
        replace(bundle, preflight=replace(bundle.preflight, report_id="sha256:wrong"))
    with pytest.raises(PerceptionSqlAdmissionError, match="valid preflight"):
        replace(bundle, postflight=replace(bundle.postflight, status="attention_required"))

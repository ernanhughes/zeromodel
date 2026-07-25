"""ZeroModel perception public API through Stage P18A."""

from __future__ import annotations

from ._public_api_p17f import *  # noqa: F401,F403
from .certification_audit import (  # noqa: F401
    CERTIFICATION_AUDIT_FINDING_VERSION,
    CERTIFICATION_AUDIT_SEMANTICS,
    CERTIFICATION_AUDIT_SEVERITIES,
    CERTIFICATION_AUDIT_STATUSES,
    CERTIFICATION_AUDIT_VERSION,
    CertificationIntegrityAuditReportDTO,
    CertificationIntegrityFindingDTO,
    PerceptionCertificationAuditError,
    audit_certification_integrity,
)
from .certification_gate import (  # noqa: F401
    CERTIFICATION_EXECUTION_GATE_SEMANTICS,
    CERTIFICATION_EXECUTION_GATE_VERSION,
    CertificationExecutionGateDTO,
    PerceptionCertificationExecutionGateError,
    authorize_certification_execution,
    execute_certification_gated_approved_rollback,
)
from .governance_gate import (  # noqa: F401
    GOVERNANCE_EXECUTION_GATE_MODES,
    GOVERNANCE_EXECUTION_GATE_SEMANTICS,
    GOVERNANCE_EXECUTION_GATE_VERSION,
    GovernanceExecutionGateDTO,
    PerceptionGovernanceExecutionGateError,
    authorize_governance_execution,
    execute_audit_gated_approved_rollback,
)
from .sql_admission import (  # noqa: F401
    SQL_ADMISSION_SCHEMA_VERSION,
    SQL_ADMISSION_STORE_VERSION,
    CertificationExecutionAdmissionBundleDTO,
    PerceptionSqlAdmissionError,
    SqliteCertificationExecutionAdmissionStore,
    execute_and_persist_certification_admission,
)
from .sql_certification import (  # noqa: F401
    GOVERNANCE_CERTIFICATION_SEMANTICS,
    GOVERNANCE_CERTIFICATION_VERSION,
    SQL_CERTIFICATION_SCHEMA_VERSION,
    SQL_CERTIFICATION_STORE_VERSION,
    GovernanceExecutionCertificationBundleDTO,
    GovernanceExecutionCertificationDTO,
    PerceptionSqlCertificationError,
    SqliteGovernanceCertificationStore,
    build_governance_execution_certification,
    execute_and_certify_audit_gated_rollback,
)
from .transition_evidence import (  # noqa: F401
    TRANSITION_CHANGED_FRACTION_SEMANTICS,
    TRANSITION_CHANGE_SEMANTICS,
    TRANSITION_EVIDENCE_VPM_VERSION,
    TRANSITION_FIELD_EVIDENCE_VERSION,
    TRANSITION_RENDER_SEMANTICS,
    TRANSITION_SIGNED_CHANGE_SEMANTICS,
    PerceptionTransitionEvidenceError,
    TransitionEvidenceVPMDTO,
    TransitionFieldEvidenceDTO,
    build_transition_evidence_vpm,
)

PERCEPTION_PACKAGE_VERSION = "1.0.13"
PERCEPTION_STAGE = "P18A"

__all__ = [
    name
    for name in globals()
    if not name.startswith("_") and name not in {"annotations"}
]

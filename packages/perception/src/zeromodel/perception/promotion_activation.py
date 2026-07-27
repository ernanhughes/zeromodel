"""Baseline-gated atomic activation for P18F change sets (Stage P18G).

P18G audits an inactive P18F change set against the exact active baseline, creates
an admission bound to the observed state, and applies every forward operation by
compare-and-swap. The reference in-memory store uses copy-on-write: a failure at
any operation leaves the active state and activation ledger unchanged. The exact
P18F inverse plan is stored for a later, separately governed rollback stage.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from typing import Final, Mapping, Protocol

from .expectations import PerceptionRegionAnnotationDTO, RelationAnnotationDTO
from .promotion_materialization import (
    PROMOTION_MATERIALIZATION_OBJECT_KINDS,
    PROMOTION_MATERIALIZATION_TARGET_KINDS,
    PromotionMaterializationBaselineDTO,
    PromotionMaterializationChangeSetDTO,
    PromotionMaterializationOperationDTO,
)
from .transition_conformance import TransitionExpectationDTO

PROMOTION_ACTIVE_STATE_VERSION: Final = "perception-promotion-active-state/1"
PROMOTION_ACTIVATION_POLICY_VERSION: Final = "perception-promotion-activation-policy/1"
PROMOTION_ACTIVATION_AUDIT_FINDING_VERSION: Final = (
    "perception-promotion-activation-audit-finding/1"
)
PROMOTION_ACTIVATION_AUDIT_REPORT_VERSION: Final = (
    "perception-promotion-activation-audit-report/1"
)
PROMOTION_ACTIVATION_ADMISSION_VERSION: Final = (
    "perception-promotion-activation-admission/1"
)
PROMOTION_ROLLBACK_PLAN_VERSION: Final = "perception-promotion-rollback-plan/1"
PROMOTION_ACTIVATION_RECEIPT_VERSION: Final = (
    "perception-promotion-activation-receipt/1"
)
PROMOTION_ACTIVATION_BUNDLE_VERSION: Final = "perception-promotion-activation-bundle/1"
PROMOTION_ACTIVATION_STORE_VERSION: Final = "perception-promotion-activation-store/1"
PROMOTION_ACTIVATION_SEMANTICS: Final = (
    "exact_baseline_compare_and_swap_atomic_activation_with_inactive_inverse_plan"
)
PROMOTION_ACTIVATION_AUDIT_STATUSES: Final = {
    "admissible",
    "blocked",
    "not_applicable",
}
PROMOTION_ACTIVATION_FINDING_SEVERITIES: Final = {"info", "error"}
PROMOTION_ACTIVATION_ADMISSION_STATUS: Final = "admitted"
PROMOTION_ACTIVATION_RECEIPT_STATUS: Final = "activated"
PROMOTION_ROLLBACK_PLAN_STATUS: Final = "stored_inactive"
PROMOTION_ROLLBACK_PLAN_STATUSES: Final = {
    "stored_inactive",
    "admitted",
    "executed",
    "blocked",
}
PROMOTION_ROLLBACK_REQUEST_VERSION: Final = "perception-promotion-rollback-request/1"
PROMOTION_ROLLBACK_POLICY_VERSION: Final = "perception-promotion-rollback-policy/1"
PROMOTION_ROLLBACK_AUDIT_REPORT_VERSION: Final = (
    "perception-promotion-rollback-audit-report/1"
)
PROMOTION_ROLLBACK_ADMISSION_VERSION: Final = (
    "perception-promotion-rollback-admission/1"
)
PROMOTION_ROLLBACK_RECEIPT_VERSION: Final = "perception-promotion-rollback-receipt/1"
PROMOTION_ROLLBACK_BUNDLE_VERSION: Final = "perception-promotion-rollback-bundle/1"
PROMOTION_ROLLBACK_RECEIPT_STATUS: Final = "rolled_back"
PROMOTION_SQLITE_ACTIVATION_SCHEMA_VERSION: Final = 1


class PerceptionPromotionActivationError(ValueError):
    """Raised when P18G admission or activation contracts are invalid."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, object]) -> str:
    encoded = _canonical_json(payload)
    hasher = hashlib.sha256()
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)
    return f"sha256:{hasher.hexdigest()}"


def _payload(value: object, identity_field: str) -> dict[str, object]:
    payload = asdict(value)  # type: ignore[arg-type]
    payload.pop(identity_field)
    return payload


def _ordered_unique(
    name: str,
    values: tuple[str, ...],
    *,
    allow_empty: bool = True,
) -> None:
    if not allow_empty and not values:
        raise PerceptionPromotionActivationError(f"{name} must be non-empty")
    if values != tuple(sorted(set(values))):
        raise PerceptionPromotionActivationError(
            f"{name} must be unique and sorted"
        )


def _non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PerceptionPromotionActivationError(
            f"{name} must be a non-negative integer"
        )


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PerceptionPromotionActivationError(
            f"{name} must be a positive integer"
        )


def _object_digest(kind: str, value: object) -> str:
    return _digest({"object_kind": kind, "payload": asdict(value)})  # type: ignore[arg-type]


def _annotation_identity(annotation: PerceptionRegionAnnotationDTO) -> str:
    return _digest(
        {
            "field_schema_id": annotation.field_schema_id,
            "field_ids": list(annotation.field_ids),
            "label": annotation.label,
            "properties": [list(item) for item in annotation.properties],
            "provenance_ref": annotation.provenance_ref,
            "role": annotation.role,
            "version": annotation.version,
        }
    )


def _relation_identity(relation: RelationAnnotationDTO) -> str:
    return _digest(
        {
            "relation_type": relation.relation_type,
            "member_annotation_ids": relation.member_annotation_ids,
            "derived_field_ids": relation.derived_field_ids,
            "value": relation.value,
            "version": relation.version,
        }
    )


@dataclass(frozen=True)
class ActivePromotionStateDTO:
    """Complete active annotation, relation, and transition-expectation state."""

    state_id: str
    revision: int
    baseline_version_id: str
    field_schema_id: str
    annotations: tuple[PerceptionRegionAnnotationDTO, ...]
    relations: tuple[RelationAnnotationDTO, ...]
    transition_expectations: tuple[TransitionExpectationDTO, ...]
    last_change_set_id: str | None = None
    version: str = PROMOTION_ACTIVE_STATE_VERSION

    def __post_init__(self) -> None:
        if not self.state_id or not self.baseline_version_id or not self.field_schema_id:
            raise PerceptionPromotionActivationError(
                "active promotion state identities must be non-empty"
            )
        _non_negative_int("active promotion state revision", self.revision)
        annotation_ids = tuple(item.annotation_id for item in self.annotations)
        relation_ids = tuple(item.relation_id for item in self.relations)
        expectation_ids = tuple(
            item.expectation_id for item in self.transition_expectations
        )
        _ordered_unique("active annotation identities", annotation_ids)
        _ordered_unique("active relation identities", relation_ids)
        _ordered_unique("active transition expectation identities", expectation_ids)
        if any(
            item.field_schema_id != self.field_schema_id for item in self.annotations
        ):
            raise PerceptionPromotionActivationError(
                "active annotation field schema mismatch"
            )
        for annotation in self.annotations:
            if annotation.annotation_id != _annotation_identity(annotation):
                raise PerceptionPromotionActivationError(
                    "active annotation identity disagrees with payload"
                )
        annotation_id_set = set(annotation_ids)
        for relation in self.relations:
            if relation.relation_id != _relation_identity(relation):
                raise PerceptionPromotionActivationError(
                    "active relation identity disagrees with payload"
                )
            missing_members = set(relation.member_annotation_ids) - annotation_id_set
            if missing_members:
                raise PerceptionPromotionActivationError(
                    "active relation references unknown annotations: "
                    f"{sorted(missing_members)}"
                )
        relation_id_set = set(relation_ids)
        for expectation in self.transition_expectations:
            if expectation.field_schema_id != self.field_schema_id:
                raise PerceptionPromotionActivationError(
                    "active transition expectation field schema mismatch"
                )
            unknown_annotations = set(expectation.annotation_ids) - annotation_id_set
            unknown_relations = set(expectation.relation_ids) - relation_id_set
            if unknown_annotations or unknown_relations:
                raise PerceptionPromotionActivationError(
                    "active transition expectation references unknown targets"
                )
        if self.version != PROMOTION_ACTIVE_STATE_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported active promotion state version"
            )
        if self.state_id != _digest(_payload(self, "state_id")):
            raise PerceptionPromotionActivationError(
                "active promotion state identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        *,
        revision: int,
        baseline_version_id: str,
        field_schema_id: str,
        annotations: tuple[PerceptionRegionAnnotationDTO, ...] = (),
        relations: tuple[RelationAnnotationDTO, ...] = (),
        transition_expectations: tuple[TransitionExpectationDTO, ...] = (),
        last_change_set_id: str | None = None,
    ) -> "ActivePromotionStateDTO":
        ordered_annotations = tuple(
            sorted(annotations, key=lambda item: item.annotation_id)
        )
        ordered_relations = tuple(sorted(relations, key=lambda item: item.relation_id))
        ordered_expectations = tuple(
            sorted(
                transition_expectations,
                key=lambda item: item.expectation_id,
            )
        )
        values: dict[str, object] = {
            "revision": revision,
            "baseline_version_id": baseline_version_id,
            "field_schema_id": field_schema_id,
            "annotations": ordered_annotations,
            "relations": ordered_relations,
            "transition_expectations": ordered_expectations,
            "last_change_set_id": last_change_set_id,
            "version": PROMOTION_ACTIVE_STATE_VERSION,
        }
        identity_values = dict(values)
        identity_values["annotations"] = tuple(
            asdict(item) for item in ordered_annotations
        )
        identity_values["relations"] = tuple(asdict(item) for item in ordered_relations)
        identity_values["transition_expectations"] = tuple(
            asdict(item) for item in ordered_expectations
        )
        return cls(
            state_id=_digest(identity_values),
            **values,  # type: ignore[arg-type]
        )

    def baseline(self) -> PromotionMaterializationBaselineDTO:
        return PromotionMaterializationBaselineDTO.create(
            baseline_version_id=self.baseline_version_id,
            field_schema_id=self.field_schema_id,
            existing_annotation_ids=tuple(
                item.annotation_id for item in self.annotations
            ),
            existing_relation_ids=tuple(item.relation_id for item in self.relations),
            existing_transition_expectation_ids=tuple(
                item.expectation_id for item in self.transition_expectations
            ),
        )


@dataclass(frozen=True)
class PromotionActivationPolicyDTO:
    policy_id: str
    maximum_change_count: int = 100
    allowed_target_kinds: tuple[str, ...] = tuple(
        sorted(PROMOTION_MATERIALIZATION_TARGET_KINDS)
    )
    version: str = PROMOTION_ACTIVATION_POLICY_VERSION

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise PerceptionPromotionActivationError(
                "promotion activation policy identity must be non-empty"
            )
        _positive_int("maximum_change_count", self.maximum_change_count)
        _ordered_unique(
            "allowed activation target kinds",
            self.allowed_target_kinds,
            allow_empty=False,
        )
        unknown = set(self.allowed_target_kinds) - PROMOTION_MATERIALIZATION_TARGET_KINDS
        if unknown:
            raise PerceptionPromotionActivationError(
                f"unsupported activation target kinds: {sorted(unknown)}"
            )
        if self.version != PROMOTION_ACTIVATION_POLICY_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation policy version"
            )
        if self.policy_id != _digest(_payload(self, "policy_id")):
            raise PerceptionPromotionActivationError(
                "promotion activation policy identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        *,
        maximum_change_count: int = 100,
        allowed_target_kinds: tuple[str, ...] = tuple(
            sorted(PROMOTION_MATERIALIZATION_TARGET_KINDS)
        ),
    ) -> "PromotionActivationPolicyDTO":
        values: dict[str, object] = {
            "maximum_change_count": maximum_change_count,
            "allowed_target_kinds": tuple(sorted(set(allowed_target_kinds))),
            "version": PROMOTION_ACTIVATION_POLICY_VERSION,
        }
        return cls(policy_id=_digest(values), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PromotionActivationAuditFindingDTO:
    finding_id: str
    severity: str
    code: str
    subject_id: str
    detail: str
    version: str = PROMOTION_ACTIVATION_AUDIT_FINDING_VERSION

    def __post_init__(self) -> None:
        if not all((self.finding_id, self.code, self.subject_id, self.detail)):
            raise PerceptionPromotionActivationError(
                "activation audit finding identities and detail must be non-empty"
            )
        if self.severity not in PROMOTION_ACTIVATION_FINDING_SEVERITIES:
            raise PerceptionPromotionActivationError(
                f"unsupported activation finding severity: {self.severity}"
            )
        if self.version != PROMOTION_ACTIVATION_AUDIT_FINDING_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation audit finding version"
            )
        if self.finding_id != _digest(_payload(self, "finding_id")):
            raise PerceptionPromotionActivationError(
                "activation audit finding identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class PromotionActivationAuditReportDTO:
    report_id: str
    status: str
    change_set_id: str
    policy_id: str
    observed_state_id: str
    observed_baseline_id: str
    observed_baseline_version_id: str
    expected_baseline_id: str
    expected_baseline_version_id: str
    resulting_state_id: str | None
    resulting_baseline_id: str | None
    resulting_baseline_version_id: str | None
    finding_ids: tuple[str, ...]
    findings: tuple[PromotionActivationAuditFindingDTO, ...]
    semantics: str = PROMOTION_ACTIVATION_SEMANTICS
    version: str = PROMOTION_ACTIVATION_AUDIT_REPORT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.report_id,
                self.change_set_id,
                self.policy_id,
                self.observed_state_id,
                self.observed_baseline_id,
                self.observed_baseline_version_id,
                self.expected_baseline_id,
                self.expected_baseline_version_id,
            )
        ):
            raise PerceptionPromotionActivationError(
                "activation audit report identities must be non-empty"
            )
        if self.status not in PROMOTION_ACTIVATION_AUDIT_STATUSES:
            raise PerceptionPromotionActivationError(
                f"unsupported promotion activation audit status: {self.status}"
            )
        actual_ids = tuple(sorted(item.finding_id for item in self.findings))
        if actual_ids != self.finding_ids:
            raise PerceptionPromotionActivationError(
                "activation audit finding identities disagree with findings"
            )
        _ordered_unique("activation audit finding identities", self.finding_ids)
        has_error = any(item.severity == "error" for item in self.findings)
        if self.status == "admissible":
            if has_error or not all(
                (
                    self.resulting_state_id,
                    self.resulting_baseline_id,
                    self.resulting_baseline_version_id,
                )
            ):
                raise PerceptionPromotionActivationError(
                    "admissible activation audits require a resulting state and no errors"
                )
        elif any(
            value is not None
            for value in (
                self.resulting_state_id,
                self.resulting_baseline_id,
                self.resulting_baseline_version_id,
            )
        ):
            raise PerceptionPromotionActivationError(
                "non-admissible activation audits cannot declare a resulting state"
            )
        if self.status == "blocked" and not has_error:
            raise PerceptionPromotionActivationError(
                "blocked activation audits require an error finding"
            )
        if self.status == "not_applicable" and has_error:
            raise PerceptionPromotionActivationError(
                "not-applicable activation audits cannot contain errors"
            )
        if self.semantics != PROMOTION_ACTIVATION_SEMANTICS:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation semantics"
            )
        if self.version != PROMOTION_ACTIVATION_AUDIT_REPORT_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation audit report version"
            )
        if self.report_id != _digest(_payload(self, "report_id")):
            raise PerceptionPromotionActivationError(
                "activation audit report identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class PromotionActivationAdmissionDTO:
    admission_id: str
    status: str
    change_set_id: str
    policy_id: str
    audit_report_id: str
    expected_state_id: str
    expected_baseline_id: str
    expected_baseline_version_id: str
    resulting_state_id: str
    resulting_baseline_id: str
    resulting_baseline_version_id: str
    forward_operation_ids: tuple[str, ...]
    inverse_operation_ids: tuple[str, ...]
    semantics: str = PROMOTION_ACTIVATION_SEMANTICS
    version: str = PROMOTION_ACTIVATION_ADMISSION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.admission_id,
                self.change_set_id,
                self.policy_id,
                self.audit_report_id,
                self.expected_state_id,
                self.expected_baseline_id,
                self.expected_baseline_version_id,
                self.resulting_state_id,
                self.resulting_baseline_id,
                self.resulting_baseline_version_id,
            )
        ):
            raise PerceptionPromotionActivationError(
                "promotion activation admission identities must be non-empty"
            )
        if self.status != PROMOTION_ACTIVATION_ADMISSION_STATUS:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation admission status"
            )
        if not self.forward_operation_ids or not self.inverse_operation_ids:
            raise PerceptionPromotionActivationError(
                "promotion activation admission requires forward and inverse operations"
            )
        if len(self.forward_operation_ids) != len(set(self.forward_operation_ids)):
            raise PerceptionPromotionActivationError(
                "admitted forward operation identities must be unique"
            )
        if len(self.inverse_operation_ids) != len(set(self.inverse_operation_ids)):
            raise PerceptionPromotionActivationError(
                "admitted inverse operation identities must be unique"
            )
        if self.semantics != PROMOTION_ACTIVATION_SEMANTICS:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation semantics"
            )
        if self.version != PROMOTION_ACTIVATION_ADMISSION_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation admission version"
            )
        if self.admission_id != _digest(_payload(self, "admission_id")):
            raise PerceptionPromotionActivationError(
                "promotion activation admission identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class PromotionRollbackPlanDTO:
    rollback_plan_id: str
    status: str
    change_set_id: str
    admission_id: str
    activated_state_id: str
    activated_baseline_id: str
    activated_baseline_version_id: str
    restore_state: ActivePromotionStateDTO
    restore_baseline_id: str
    restore_baseline_version_id: str
    inverse_operation_ids: tuple[str, ...]
    inverse_operations: tuple[PromotionMaterializationOperationDTO, ...]
    semantics: str = PROMOTION_ACTIVATION_SEMANTICS
    version: str = PROMOTION_ROLLBACK_PLAN_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.rollback_plan_id,
                self.change_set_id,
                self.admission_id,
                self.activated_state_id,
                self.activated_baseline_id,
                self.activated_baseline_version_id,
                self.restore_baseline_id,
                self.restore_baseline_version_id,
            )
        ):
            raise PerceptionPromotionActivationError(
                "promotion rollback plan identities must be non-empty"
            )
        if self.status not in PROMOTION_ROLLBACK_PLAN_STATUSES:
            raise PerceptionPromotionActivationError(
                "unsupported promotion rollback plan status"
            )
        actual_ids = tuple(item.operation_id for item in self.inverse_operations)
        if actual_ids != self.inverse_operation_ids:
            raise PerceptionPromotionActivationError(
                "rollback inverse operation identities disagree with operations"
            )
        if not self.inverse_operations or any(
            item.direction != "inverse" for item in self.inverse_operations
        ):
            raise PerceptionPromotionActivationError(
                "rollback plan requires inverse materialization operations"
            )
        if tuple(item.sequence for item in self.inverse_operations) != tuple(
            range(1, len(self.inverse_operations) + 1)
        ):
            raise PerceptionPromotionActivationError(
                "rollback inverse operation sequence must be contiguous"
            )
        restore_baseline = self.restore_state.baseline()
        if (
            restore_baseline.baseline_id != self.restore_baseline_id
            or self.restore_state.baseline_version_id
            != self.restore_baseline_version_id
        ):
            raise PerceptionPromotionActivationError(
                "rollback restore baseline disagrees with restore state"
            )
        if self.semantics != PROMOTION_ACTIVATION_SEMANTICS:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation semantics"
            )
        if self.version != PROMOTION_ROLLBACK_PLAN_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion rollback plan version"
            )
        if self.rollback_plan_id != _digest(_payload(self, "rollback_plan_id")):
            raise PerceptionPromotionActivationError(
                "promotion rollback plan identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class PromotionActivationReceiptDTO:
    receipt_id: str
    status: str
    change_set_id: str
    admission_id: str
    audit_report_id: str
    previous_state_id: str
    resulting_state_id: str
    previous_baseline_id: str
    resulting_baseline_id: str
    previous_baseline_version_id: str
    resulting_baseline_version_id: str
    resulting_revision: int
    forward_operation_ids: tuple[str, ...]
    rollback_plan_id: str
    semantics: str = PROMOTION_ACTIVATION_SEMANTICS
    version: str = PROMOTION_ACTIVATION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.receipt_id,
                self.change_set_id,
                self.admission_id,
                self.audit_report_id,
                self.previous_state_id,
                self.resulting_state_id,
                self.previous_baseline_id,
                self.resulting_baseline_id,
                self.previous_baseline_version_id,
                self.resulting_baseline_version_id,
                self.rollback_plan_id,
            )
        ):
            raise PerceptionPromotionActivationError(
                "promotion activation receipt identities must be non-empty"
            )
        if self.status != PROMOTION_ACTIVATION_RECEIPT_STATUS:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation receipt status"
            )
        _positive_int("resulting activation revision", self.resulting_revision)
        if not self.forward_operation_ids or len(self.forward_operation_ids) != len(
            set(self.forward_operation_ids)
        ):
            raise PerceptionPromotionActivationError(
                "activation receipt requires unique forward operation identities"
            )
        if self.semantics != PROMOTION_ACTIVATION_SEMANTICS:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation semantics"
            )
        if self.version != PROMOTION_ACTIVATION_RECEIPT_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation receipt version"
            )
        if self.receipt_id != _digest(_payload(self, "receipt_id")):
            raise PerceptionPromotionActivationError(
                "promotion activation receipt identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class PromotionActivationBundleDTO:
    bundle_id: str
    change_set_id: str
    audit_report: PromotionActivationAuditReportDTO
    admission: PromotionActivationAdmissionDTO
    rollback_plan: PromotionRollbackPlanDTO
    receipt: PromotionActivationReceiptDTO
    resulting_state: ActivePromotionStateDTO
    semantics: str = PROMOTION_ACTIVATION_SEMANTICS
    version: str = PROMOTION_ACTIVATION_BUNDLE_VERSION

    def __post_init__(self) -> None:
        if not self.bundle_id or not self.change_set_id:
            raise PerceptionPromotionActivationError(
                "promotion activation bundle identities must be non-empty"
            )
        if self.audit_report.status != "admissible":
            raise PerceptionPromotionActivationError(
                "activation bundles require an admissible audit"
            )
        if not (
            self.audit_report.change_set_id
            == self.admission.change_set_id
            == self.rollback_plan.change_set_id
            == self.receipt.change_set_id
            == self.change_set_id
        ):
            raise PerceptionPromotionActivationError(
                "activation bundle change-set lineage disagrees"
            )
        if (
            self.admission.audit_report_id != self.audit_report.report_id
            or self.receipt.audit_report_id != self.audit_report.report_id
            or self.receipt.admission_id != self.admission.admission_id
            or self.rollback_plan.admission_id != self.admission.admission_id
            or self.receipt.rollback_plan_id != self.rollback_plan.rollback_plan_id
        ):
            raise PerceptionPromotionActivationError(
                "activation bundle artifact lineage disagrees"
            )
        resulting_baseline = self.resulting_state.baseline()
        if not (
            self.resulting_state.state_id
            == self.admission.resulting_state_id
            == self.receipt.resulting_state_id
            == self.rollback_plan.activated_state_id
            and resulting_baseline.baseline_id
            == self.admission.resulting_baseline_id
            == self.receipt.resulting_baseline_id
            == self.rollback_plan.activated_baseline_id
            and self.resulting_state.baseline_version_id
            == self.admission.resulting_baseline_version_id
            == self.receipt.resulting_baseline_version_id
            == self.rollback_plan.activated_baseline_version_id
        ):
            raise PerceptionPromotionActivationError(
                "activation bundle resulting state lineage disagrees"
            )
        if not (
            self.rollback_plan.restore_state.state_id
            == self.admission.expected_state_id
            == self.receipt.previous_state_id
            and self.rollback_plan.restore_baseline_id
            == self.admission.expected_baseline_id
            == self.receipt.previous_baseline_id
            and self.rollback_plan.restore_baseline_version_id
            == self.admission.expected_baseline_version_id
            == self.receipt.previous_baseline_version_id
        ):
            raise PerceptionPromotionActivationError(
                "activation bundle restore-state lineage disagrees"
            )
        if self.semantics != PROMOTION_ACTIVATION_SEMANTICS:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation semantics"
            )
        if self.version != PROMOTION_ACTIVATION_BUNDLE_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion activation bundle version"
            )
        if self.bundle_id != _digest(_payload(self, "bundle_id")):
            raise PerceptionPromotionActivationError(
                "promotion activation bundle identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class PromotionRollbackRequestDTO:
    request_id: str
    rollback_plan_id: str
    expected_active_state_id: str
    requested_by: str
    reason: str
    version: str = PROMOTION_ROLLBACK_REQUEST_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.request_id,
                self.rollback_plan_id,
                self.expected_active_state_id,
                self.requested_by,
                self.reason,
            )
        ):
            raise PerceptionPromotionActivationError(
                "rollback request identities, requester, and reason must be non-empty"
            )
        if self.version != PROMOTION_ROLLBACK_REQUEST_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion rollback request version"
            )
        if self.request_id != _digest(_payload(self, "request_id")):
            raise PerceptionPromotionActivationError(
                "promotion rollback request identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        *,
        rollback_plan_id: str,
        expected_active_state_id: str,
        requested_by: str,
        reason: str,
    ) -> "PromotionRollbackRequestDTO":
        values: dict[str, object] = {
            "rollback_plan_id": rollback_plan_id,
            "expected_active_state_id": expected_active_state_id,
            "requested_by": requested_by,
            "reason": reason,
            "version": PROMOTION_ROLLBACK_REQUEST_VERSION,
        }
        return cls(request_id=_digest(values), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PromotionRollbackPolicyDTO:
    policy_id: str
    require_latest_activation: bool = True
    maximum_operations: int = 100
    permitted_operation_kinds: tuple[str, ...] = tuple(
        sorted(PROMOTION_MATERIALIZATION_OBJECT_KINDS)
    )
    require_reason: bool = True
    version: str = PROMOTION_ROLLBACK_POLICY_VERSION

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise PerceptionPromotionActivationError(
                "rollback policy identity must be non-empty"
            )
        if not self.require_latest_activation:
            raise PerceptionPromotionActivationError(
                "P18H supports rollback of the exact latest activation only"
            )
        _positive_int("maximum rollback operations", self.maximum_operations)
        _ordered_unique(
            "permitted rollback operation kinds",
            self.permitted_operation_kinds,
            allow_empty=False,
        )
        unknown = set(self.permitted_operation_kinds) - PROMOTION_MATERIALIZATION_OBJECT_KINDS
        if unknown:
            raise PerceptionPromotionActivationError(
                f"unsupported rollback operation kinds: {sorted(unknown)}"
            )
        if self.version != PROMOTION_ROLLBACK_POLICY_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion rollback policy version"
            )
        if self.policy_id != _digest(_payload(self, "policy_id")):
            raise PerceptionPromotionActivationError(
                "promotion rollback policy identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        *,
        require_latest_activation: bool = True,
        maximum_operations: int = 100,
        permitted_operation_kinds: tuple[str, ...] = tuple(
            sorted(PROMOTION_MATERIALIZATION_OBJECT_KINDS)
        ),
        require_reason: bool = True,
    ) -> "PromotionRollbackPolicyDTO":
        values: dict[str, object] = {
            "require_latest_activation": require_latest_activation,
            "maximum_operations": maximum_operations,
            "permitted_operation_kinds": tuple(sorted(set(permitted_operation_kinds))),
            "require_reason": require_reason,
            "version": PROMOTION_ROLLBACK_POLICY_VERSION,
        }
        return cls(policy_id=_digest(values), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PromotionRollbackAuditDTO:
    report_id: str
    status: str
    request_id: str
    policy_id: str
    rollback_plan_id: str
    current_state_id: str | None
    predicted_restore_state_id: str | None
    finding_ids: tuple[str, ...]
    findings: tuple[PromotionActivationAuditFindingDTO, ...]
    semantics: str = PROMOTION_ACTIVATION_SEMANTICS
    version: str = PROMOTION_ROLLBACK_AUDIT_REPORT_VERSION

    def __post_init__(self) -> None:
        if not all((self.report_id, self.request_id, self.policy_id, self.rollback_plan_id)):
            raise PerceptionPromotionActivationError(
                "rollback audit identities must be non-empty"
            )
        if self.status not in {"admissible", "blocked", "not_applicable", "already_executed"}:
            raise PerceptionPromotionActivationError("unsupported rollback audit status")
        actual_ids = tuple(sorted(item.finding_id for item in self.findings))
        if actual_ids != self.finding_ids:
            raise PerceptionPromotionActivationError(
                "rollback audit finding identities disagree with findings"
            )
        if self.version != PROMOTION_ROLLBACK_AUDIT_REPORT_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion rollback audit version"
            )
        if self.report_id != _digest(_payload(self, "report_id")):
            raise PerceptionPromotionActivationError(
                "promotion rollback audit identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class PromotionRollbackAdmissionDTO:
    admission_id: str
    status: str
    request_id: str
    policy_id: str
    audit_report_id: str
    rollback_plan_id: str
    expected_state_id: str
    expected_revision: int
    expected_baseline_id: str
    expected_baseline_version_id: str
    predicted_restore_state_id: str
    predicted_restore_baseline_id: str
    predicted_restore_baseline_version_id: str
    inverse_operation_ids: tuple[str, ...]
    semantics: str = PROMOTION_ACTIVATION_SEMANTICS
    version: str = PROMOTION_ROLLBACK_ADMISSION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.admission_id,
                self.request_id,
                self.policy_id,
                self.audit_report_id,
                self.rollback_plan_id,
                self.expected_state_id,
                self.expected_baseline_id,
                self.expected_baseline_version_id,
                self.predicted_restore_state_id,
                self.predicted_restore_baseline_id,
                self.predicted_restore_baseline_version_id,
            )
        ):
            raise PerceptionPromotionActivationError(
                "rollback admission identities must be non-empty"
            )
        if self.status != "admitted":
            raise PerceptionPromotionActivationError("unsupported rollback admission status")
        _non_negative_int("rollback admission expected revision", self.expected_revision)
        if not self.inverse_operation_ids or len(self.inverse_operation_ids) != len(
            set(self.inverse_operation_ids)
        ):
            raise PerceptionPromotionActivationError(
                "rollback admission requires unique inverse operation identities"
            )
        if self.version != PROMOTION_ROLLBACK_ADMISSION_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion rollback admission version"
            )
        if self.admission_id != _digest(_payload(self, "admission_id")):
            raise PerceptionPromotionActivationError(
                "promotion rollback admission identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class PromotionRollbackReceiptDTO:
    receipt_id: str
    status: str
    admission_id: str
    rollback_plan_id: str
    activation_receipt_id: str
    prior_state_id: str
    restored_state_id: str
    execution_revision: int
    inverse_operation_ids: tuple[str, ...]
    semantics: str = PROMOTION_ACTIVATION_SEMANTICS
    version: str = PROMOTION_ROLLBACK_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.receipt_id,
                self.admission_id,
                self.rollback_plan_id,
                self.activation_receipt_id,
                self.prior_state_id,
                self.restored_state_id,
            )
        ):
            raise PerceptionPromotionActivationError(
                "rollback receipt identities must be non-empty"
            )
        if self.status != PROMOTION_ROLLBACK_RECEIPT_STATUS:
            raise PerceptionPromotionActivationError("unsupported rollback receipt status")
        _positive_int("rollback execution revision", self.execution_revision)
        if self.version != PROMOTION_ROLLBACK_RECEIPT_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion rollback receipt version"
            )
        if self.receipt_id != _digest(_payload(self, "receipt_id")):
            raise PerceptionPromotionActivationError(
                "promotion rollback receipt identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class PromotionRollbackBundleDTO:
    bundle_id: str
    receipt: PromotionRollbackReceiptDTO
    restored_state: ActivePromotionStateDTO
    rollback_plan: PromotionRollbackPlanDTO
    activation_receipt: PromotionActivationReceiptDTO
    semantics: str = PROMOTION_ACTIVATION_SEMANTICS
    version: str = PROMOTION_ROLLBACK_BUNDLE_VERSION

    def __post_init__(self) -> None:
        if self.rollback_plan.status != PROMOTION_ROLLBACK_PLAN_STATUS:
            raise PerceptionPromotionActivationError(
                "rollback bundles preserve the original stored inactive rollback plan"
            )
        if (
            self.receipt.rollback_plan_id != self.rollback_plan.rollback_plan_id
            or self.receipt.rollback_plan_id != self.activation_receipt.rollback_plan_id
            or self.receipt.activation_receipt_id != self.activation_receipt.receipt_id
            or self.receipt.restored_state_id != self.restored_state.state_id
        ):
            raise PerceptionPromotionActivationError("rollback bundle lineage disagrees")
        if self.version != PROMOTION_ROLLBACK_BUNDLE_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported promotion rollback bundle version"
            )
        if self.bundle_id != _digest(_payload(self, "bundle_id")):
            raise PerceptionPromotionActivationError(
                "promotion rollback bundle identity disagrees with canonical payload"
            )


class PromotionActivationStore(Protocol):
    """DTO-only store boundary for one active policy state and activation ledger."""

    version: str

    def get_active_state(self) -> ActivePromotionStateDTO:
        ...

    def commit_activation(
        self,
        expected_state: ActivePromotionStateDTO,
        change_set: PromotionMaterializationChangeSetDTO,
        bundle: PromotionActivationBundleDTO,
    ) -> None:
        ...

    def get_activation_bundle(self, change_set_id: str) -> PromotionActivationBundleDTO:
        ...

    def list_activation_bundles(self) -> tuple[PromotionActivationBundleDTO, ...]:
        ...

    def get_rollback_plan(self, rollback_plan_id: str) -> PromotionRollbackPlanDTO:
        ...

    def admit_rollback(
        self,
        request: PromotionRollbackRequestDTO,
        policy: PromotionRollbackPolicyDTO | None = None,
    ) -> PromotionRollbackAdmissionDTO:
        ...

    def commit_rollback(
        self,
        admission: PromotionRollbackAdmissionDTO,
    ) -> PromotionRollbackBundleDTO:
        ...


def _finding(
    *,
    severity: str,
    code: str,
    subject_id: str,
    detail: str,
) -> PromotionActivationAuditFindingDTO:
    values: dict[str, object] = {
        "severity": severity,
        "code": code,
        "subject_id": subject_id,
        "detail": detail,
        "version": PROMOTION_ACTIVATION_AUDIT_FINDING_VERSION,
    }
    return PromotionActivationAuditFindingDTO(
        finding_id=_digest(values),
        **values,  # type: ignore[arg-type]
    )


def _resulting_baseline_version_id(
    current: ActivePromotionStateDTO,
    change_set: PromotionMaterializationChangeSetDTO,
) -> str:
    return _digest(
        {
            "previous_state_id": current.state_id,
            "previous_baseline_id": current.baseline().baseline_id,
            "previous_baseline_version_id": current.baseline_version_id,
            "change_set_id": change_set.change_set_id,
            "resulting_revision": current.revision + 1,
            "version": PROMOTION_ACTIVE_STATE_VERSION,
        }
    )


def _change_set_objects(
    change_set: PromotionMaterializationChangeSetDTO,
) -> dict[tuple[str, str], object]:
    objects: dict[tuple[str, str], object] = {}
    for change in change_set.changes:
        if change.annotation is not None:
            key = ("annotation", change.annotation.annotation_id)
            value: object = change.annotation
        else:
            if change.relation is None:
                raise PerceptionPromotionActivationError(
                    "materialized promotion change has no target object"
                )
            key = ("relation", change.relation.relation_id)
            value = change.relation
        if key in objects:
            raise PerceptionPromotionActivationError(
                "change set repeats a materialized target object"
            )
        objects[key] = value
        expectation_key = (
            "transition_expectation",
            change.transition_expectation.expectation_id,
        )
        if expectation_key in objects:
            raise PerceptionPromotionActivationError(
                "change set repeats a transition expectation"
            )
        objects[expectation_key] = change.transition_expectation
    return objects


def _apply_forward_operations(
    current: ActivePromotionStateDTO,
    change_set: PromotionMaterializationChangeSetDTO,
    *,
    operation_hook: object | None = None,
) -> ActivePromotionStateDTO:
    if change_set.status != "staged_inactive" or not change_set.changes:
        raise PerceptionPromotionActivationError(
            "atomic activation requires a non-empty staged_inactive change set"
        )
    if change_set.field_schema_id != current.field_schema_id:
        raise PerceptionPromotionActivationError(
            "change-set field schema differs from active state"
        )
    objects = _change_set_objects(change_set)
    operations = change_set.operations("forward")
    if {(
        item.object_kind,
        item.object_id,
    ) for item in operations} != set(objects):
        raise PerceptionPromotionActivationError(
            "forward operations do not exactly cover materialized objects"
        )
    annotations = {item.annotation_id: item for item in current.annotations}
    relations = {item.relation_id: item for item in current.relations}
    expectations = {
        item.expectation_id: item for item in current.transition_expectations
    }
    for operation in operations:
        key = (operation.object_kind, operation.object_id)
        try:
            value = objects[key]
        except KeyError as exc:
            raise PerceptionPromotionActivationError(
                "forward operation references an unknown materialized object"
            ) from exc
        if operation.payload_digest != _object_digest(operation.object_kind, value):
            raise PerceptionPromotionActivationError(
                "forward operation payload digest disagrees with materialized object"
            )
        if operation.object_kind == "annotation":
            if not isinstance(value, PerceptionRegionAnnotationDTO):
                raise PerceptionPromotionActivationError(
                    "annotation operation payload has the wrong DTO type"
                )
            if operation.object_id in annotations:
                raise PerceptionPromotionActivationError(
                    "activation would overwrite an active annotation"
                )
            if value.field_schema_id != current.field_schema_id:
                raise PerceptionPromotionActivationError(
                    "materialized annotation field schema mismatch"
                )
            annotations[value.annotation_id] = value
        elif operation.object_kind == "relation":
            if not isinstance(value, RelationAnnotationDTO):
                raise PerceptionPromotionActivationError(
                    "relation operation payload has the wrong DTO type"
                )
            if operation.object_id in relations:
                raise PerceptionPromotionActivationError(
                    "activation would overwrite an active relation"
                )
            missing_members = set(value.member_annotation_ids) - set(annotations)
            if missing_members:
                raise PerceptionPromotionActivationError(
                    "materialized relation references inactive annotations: "
                    f"{sorted(missing_members)}"
                )
            relations[value.relation_id] = value
        else:
            if not isinstance(value, TransitionExpectationDTO):
                raise PerceptionPromotionActivationError(
                    "transition expectation operation payload has the wrong DTO type"
                )
            if operation.object_id in expectations:
                raise PerceptionPromotionActivationError(
                    "activation would overwrite an active transition expectation"
                )
            if value.field_schema_id != current.field_schema_id:
                raise PerceptionPromotionActivationError(
                    "materialized transition expectation field schema mismatch"
                )
            if set(value.annotation_ids) - set(annotations) or set(
                value.relation_ids
            ) - set(relations):
                raise PerceptionPromotionActivationError(
                    "materialized transition expectation references inactive targets"
                )
            expectations[value.expectation_id] = value
        if operation_hook is not None:
            hook = getattr(operation_hook, "_after_operation_applied", None)
            if hook is not None:
                hook(operation)
    return ActivePromotionStateDTO.create(
        revision=current.revision + 1,
        baseline_version_id=_resulting_baseline_version_id(current, change_set),
        field_schema_id=current.field_schema_id,
        annotations=tuple(annotations.values()),
        relations=tuple(relations.values()),
        transition_expectations=tuple(expectations.values()),
        last_change_set_id=change_set.change_set_id,
    )


def _apply_inverse_operations(
    current: ActivePromotionStateDTO,
    plan: PromotionRollbackPlanDTO,
    *,
    operation_hook: object | None = None,
) -> ActivePromotionStateDTO:
    if current.state_id != plan.activated_state_id:
        raise PerceptionPromotionActivationError(
            "rollback plan does not match the current activated state"
        )
    annotations = {item.annotation_id: item for item in current.annotations}
    relations = {item.relation_id: item for item in current.relations}
    expectations = {
        item.expectation_id: item for item in current.transition_expectations
    }
    for operation in plan.inverse_operations:
        if operation.object_kind == "transition_expectation":
            value = expectations.get(operation.object_id)
            if value is None:
                raise PerceptionPromotionActivationError(
                    "inverse operation references a missing transition expectation"
                )
            if operation.payload_digest != _object_digest(operation.object_kind, value):
                raise PerceptionPromotionActivationError(
                    "inverse transition expectation digest disagrees with active state"
                )
            del expectations[operation.object_id]
        elif operation.object_kind == "relation":
            value = relations.get(operation.object_id)
            if value is None:
                raise PerceptionPromotionActivationError(
                    "inverse operation references a missing relation"
                )
            if operation.payload_digest != _object_digest(operation.object_kind, value):
                raise PerceptionPromotionActivationError(
                    "inverse relation digest disagrees with active state"
                )
            if any(operation.object_id in item.relation_ids for item in expectations.values()):
                raise PerceptionPromotionActivationError(
                    "cannot remove relation before dependent transition expectations"
                )
            del relations[operation.object_id]
        else:
            value = annotations.get(operation.object_id)
            if value is None:
                raise PerceptionPromotionActivationError(
                    "inverse operation references a missing annotation"
                )
            if operation.payload_digest != _object_digest(operation.object_kind, value):
                raise PerceptionPromotionActivationError(
                    "inverse annotation digest disagrees with active state"
                )
            if any(operation.object_id in item.member_annotation_ids for item in relations.values()):
                raise PerceptionPromotionActivationError(
                    "cannot remove annotation before dependent relations"
                )
            if any(operation.object_id in item.annotation_ids for item in expectations.values()):
                raise PerceptionPromotionActivationError(
                    "cannot remove annotation before dependent transition expectations"
                )
            del annotations[operation.object_id]
        if operation_hook is not None:
            hook = getattr(operation_hook, "_after_rollback_operation_applied", None)
            if hook is not None:
                hook(operation)
    restored = ActivePromotionStateDTO.create(
        revision=plan.restore_state.revision,
        baseline_version_id=plan.restore_baseline_version_id,
        field_schema_id=current.field_schema_id,
        annotations=tuple(annotations.values()),
        relations=tuple(relations.values()),
        transition_expectations=tuple(expectations.values()),
        last_change_set_id=plan.restore_state.last_change_set_id,
    )
    if restored != plan.restore_state:
        raise PerceptionPromotionActivationError(
            "inverse plan result disagrees with exact stored restore state"
        )
    return restored


def _rollback_finding(
    *,
    severity: str,
    code: str,
    subject_id: str,
    detail: str,
) -> PromotionActivationAuditFindingDTO:
    return _finding(
        severity=severity,
        code=code,
        subject_id=subject_id,
        detail=detail,
    )


def audit_promotion_rollback(
    current: ActivePromotionStateDTO | None,
    plan: PromotionRollbackPlanDTO | None,
    request: PromotionRollbackRequestDTO,
    policy: PromotionRollbackPolicyDTO | None = None,
) -> PromotionRollbackAuditDTO:
    resolved = policy or PromotionRollbackPolicyDTO.create()
    findings: list[PromotionActivationAuditFindingDTO] = []
    predicted: ActivePromotionStateDTO | None = None
    status = "admissible"
    if plan is None:
        findings.append(
            _rollback_finding(
                severity="error",
                code="rollback_plan_missing",
                subject_id=request.rollback_plan_id,
                detail="The requested rollback plan is not stored.",
            )
        )
    elif plan.status == "executed":
        findings.append(
            _rollback_finding(
                severity="info",
                code="rollback_plan_already_executed",
                subject_id=plan.rollback_plan_id,
                detail="The rollback plan has already been executed.",
            )
        )
        status = "already_executed"
    elif plan.status != PROMOTION_ROLLBACK_PLAN_STATUS:
        findings.append(
            _rollback_finding(
                severity="error",
                code="rollback_plan_status_not_executable",
                subject_id=plan.rollback_plan_id,
                detail="Rollback execution requires a stored inactive plan.",
            )
        )
    if current is None:
        findings.append(
            _rollback_finding(
                severity="error",
                code="active_state_missing",
                subject_id=request.expected_active_state_id,
                detail="Rollback admission requires a current active state.",
            )
        )
    elif current.state_id != request.expected_active_state_id:
        findings.append(
            _rollback_finding(
                severity="error",
                code="request_active_state_mismatch",
                subject_id=current.state_id,
                detail="The rollback request does not name the exact current state.",
            )
        )
    if plan is not None and current is not None and status != "already_executed":
        if current.state_id != plan.activated_state_id:
            findings.append(
                _rollback_finding(
                    severity="error",
                    code="active_state_mismatch",
                    subject_id=current.state_id,
                    detail="Current state does not equal the plan's activated state.",
                )
            )
        if current.baseline().baseline_id != plan.activated_baseline_id:
            findings.append(
                _rollback_finding(
                    severity="error",
                    code="baseline_identity_mismatch",
                    subject_id=current.state_id,
                    detail="Current baseline identity differs from the activated baseline.",
                )
            )
        if current.baseline_version_id != plan.activated_baseline_version_id:
            findings.append(
                _rollback_finding(
                    severity="error",
                    code="baseline_version_mismatch",
                    subject_id=current.state_id,
                    detail="Current baseline version differs from the activated baseline.",
                )
            )
        if resolved.require_reason and not request.reason.strip():
            findings.append(
                _rollback_finding(
                    severity="error",
                    code="rollback_reason_required",
                    subject_id=request.request_id,
                    detail="Rollback policy requires a reason.",
                )
            )
        if len(plan.inverse_operations) > resolved.maximum_operations:
            findings.append(
                _rollback_finding(
                    severity="error",
                    code="operation_count_exceeds_policy",
                    subject_id=plan.rollback_plan_id,
                    detail="The inverse operation count exceeds the rollback policy limit.",
                )
            )
        disallowed = {
            item.object_kind for item in plan.inverse_operations
        } - set(resolved.permitted_operation_kinds)
        if disallowed:
            findings.append(
                _rollback_finding(
                    severity="error",
                    code="operation_kind_disallowed",
                    subject_id=plan.rollback_plan_id,
                    detail=f"Rollback policy disallows operation kinds: {sorted(disallowed)}",
                )
            )
        if not any(item.severity == "error" for item in findings):
            try:
                predicted = _apply_inverse_operations(current, plan)
            except Exception as exc:
                findings.append(
                    _rollback_finding(
                        severity="error",
                        code="inverse_plan_invalid",
                        subject_id=plan.rollback_plan_id,
                        detail=str(exc),
                    )
                )
    if status != "already_executed":
        if any(item.severity == "error" for item in findings):
            status = "blocked"
        else:
            findings.append(
                _rollback_finding(
                    severity="info",
                    code="rollback_admissible",
                    subject_id=request.rollback_plan_id,
                    detail="Exact active state and stored inverse plan authorize rollback.",
                )
            )
    ordered_findings = tuple(sorted(findings, key=lambda item: item.finding_id))
    values: dict[str, object] = {
        "status": status,
        "request_id": request.request_id,
        "policy_id": resolved.policy_id,
        "rollback_plan_id": request.rollback_plan_id,
        "current_state_id": None if current is None else current.state_id,
        "predicted_restore_state_id": None if predicted is None else predicted.state_id,
        "finding_ids": tuple(item.finding_id for item in ordered_findings),
        "findings": ordered_findings,
        "semantics": PROMOTION_ACTIVATION_SEMANTICS,
        "version": PROMOTION_ROLLBACK_AUDIT_REPORT_VERSION,
    }
    identity_values = dict(values)
    identity_values["findings"] = tuple(asdict(item) for item in ordered_findings)
    return PromotionRollbackAuditDTO(
        report_id=_digest(identity_values),
        **values,  # type: ignore[arg-type]
    )


def authorize_promotion_rollback(
    report: PromotionRollbackAuditDTO,
    current: ActivePromotionStateDTO,
    plan: PromotionRollbackPlanDTO,
    request: PromotionRollbackRequestDTO,
    policy: PromotionRollbackPolicyDTO,
) -> tuple[PromotionRollbackAdmissionDTO, ActivePromotionStateDTO]:
    if report.status != "admissible":
        raise PerceptionPromotionActivationError("promotion rollback audit is not admissible")
    if (
        report.request_id != request.request_id
        or report.policy_id != policy.policy_id
        or report.rollback_plan_id != plan.rollback_plan_id
        or report.current_state_id != current.state_id
    ):
        raise PerceptionPromotionActivationError(
            "promotion rollback audit no longer matches the supplied state"
        )
    restored = _apply_inverse_operations(current, plan)
    restored_baseline = restored.baseline()
    current_baseline = current.baseline()
    values: dict[str, object] = {
        "status": "admitted",
        "request_id": request.request_id,
        "policy_id": policy.policy_id,
        "audit_report_id": report.report_id,
        "rollback_plan_id": plan.rollback_plan_id,
        "expected_state_id": current.state_id,
        "expected_revision": current.revision,
        "expected_baseline_id": current_baseline.baseline_id,
        "expected_baseline_version_id": current.baseline_version_id,
        "predicted_restore_state_id": restored.state_id,
        "predicted_restore_baseline_id": restored_baseline.baseline_id,
        "predicted_restore_baseline_version_id": restored.baseline_version_id,
        "inverse_operation_ids": plan.inverse_operation_ids,
        "semantics": PROMOTION_ACTIVATION_SEMANTICS,
        "version": PROMOTION_ROLLBACK_ADMISSION_VERSION,
    }
    admission = PromotionRollbackAdmissionDTO(
        admission_id=_digest(values),
        **values,  # type: ignore[arg-type]
    )
    return admission, restored


def _annotation_from_dict(data: Mapping[str, object]) -> PerceptionRegionAnnotationDTO:
    return PerceptionRegionAnnotationDTO(
        annotation_id=str(data["annotation_id"]),
        field_schema_id=str(data["field_schema_id"]),
        field_ids=tuple(str(item) for item in data["field_ids"]),  # type: ignore[index]
        label=data["label"] if data.get("label") is not None else None,  # type: ignore[arg-type]
        role=data["role"] if data.get("role") is not None else None,  # type: ignore[arg-type]
        properties=tuple(tuple(str(part) for part in item) for item in data["properties"]),  # type: ignore[index]
        provenance_ref=(
            data["provenance_ref"] if data.get("provenance_ref") is not None else None
        ),  # type: ignore[arg-type]
        version=str(data["version"]),
    )


def _relation_from_dict(data: Mapping[str, object]) -> RelationAnnotationDTO:
    return RelationAnnotationDTO(
        relation_id=str(data["relation_id"]),
        relation_type=str(data["relation_type"]),
        member_annotation_ids=tuple(str(item) for item in data["member_annotation_ids"]),  # type: ignore[index]
        derived_field_ids=tuple(str(item) for item in data["derived_field_ids"]),  # type: ignore[index]
        value=data.get("value"),  # type: ignore[arg-type]
        version=str(data["version"]),
    )


def _expectation_from_dict(data: Mapping[str, object]) -> TransitionExpectationDTO:
    return TransitionExpectationDTO(
        expectation_id=str(data["expectation_id"]),
        field_schema_id=str(data["field_schema_id"]),
        annotation_ids=tuple(str(item) for item in data["annotation_ids"]),  # type: ignore[index]
        relation_ids=tuple(str(item) for item in data["relation_ids"]),  # type: ignore[index]
        expected_change=str(data["expected_change"]),
        minimum_mean_absolute_change=float(data["minimum_mean_absolute_change"]),
        maximum_mean_absolute_change=float(data["maximum_mean_absolute_change"]),
        minimum_changed_fraction=float(data["minimum_changed_fraction"]),
        maximum_changed_fraction=float(data["maximum_changed_fraction"]),
        minimum_signed_change_magnitude=float(data["minimum_signed_change_magnitude"]),
        version=str(data["version"]),
    )


def _state_from_dict(data: Mapping[str, object]) -> ActivePromotionStateDTO:
    return ActivePromotionStateDTO(
        state_id=str(data["state_id"]),
        revision=int(data["revision"]),
        baseline_version_id=str(data["baseline_version_id"]),
        field_schema_id=str(data["field_schema_id"]),
        annotations=tuple(_annotation_from_dict(item) for item in data["annotations"]),  # type: ignore[arg-type,index]
        relations=tuple(_relation_from_dict(item) for item in data["relations"]),  # type: ignore[arg-type,index]
        transition_expectations=tuple(
            _expectation_from_dict(item) for item in data["transition_expectations"]  # type: ignore[arg-type,index]
        ),
        last_change_set_id=(
            str(data["last_change_set_id"])
            if data.get("last_change_set_id") is not None
            else None
        ),
        version=str(data["version"]),
    )


def _operation_from_dict(data: Mapping[str, object]) -> PromotionMaterializationOperationDTO:
    try:
        return PromotionMaterializationOperationDTO(
            operation_id=str(data["operation_id"]),
            pair_id=str(data["pair_id"]),
            direction=str(data["direction"]),
            action=str(data["action"]),
            object_kind=str(data["object_kind"]),
            object_id=str(data["object_id"]),
            payload_digest=str(data["payload_digest"]),
            proposal_id=str(data["proposal_id"]),
            decision_id=str(data["decision_id"]),
            sequence=int(data["sequence"]),
            version=str(data["version"]),
        )
    except Exception as exc:
        raise PerceptionPromotionActivationError(
            "stored rollback operation ordinals are malformed"
        ) from exc


def _finding_from_dict(data: Mapping[str, object]) -> PromotionActivationAuditFindingDTO:
    return PromotionActivationAuditFindingDTO(
        finding_id=str(data["finding_id"]),
        severity=str(data["severity"]),
        code=str(data["code"]),
        subject_id=str(data["subject_id"]),
        detail=str(data["detail"]),
        version=str(data["version"]),
    )


def _activation_report_from_dict(data: Mapping[str, object]) -> PromotionActivationAuditReportDTO:
    return PromotionActivationAuditReportDTO(
        report_id=str(data["report_id"]),
        status=str(data["status"]),
        change_set_id=str(data["change_set_id"]),
        policy_id=str(data["policy_id"]),
        observed_state_id=str(data["observed_state_id"]),
        observed_baseline_id=str(data["observed_baseline_id"]),
        observed_baseline_version_id=str(data["observed_baseline_version_id"]),
        expected_baseline_id=str(data["expected_baseline_id"]),
        expected_baseline_version_id=str(data["expected_baseline_version_id"]),
        resulting_state_id=data.get("resulting_state_id"),  # type: ignore[arg-type]
        resulting_baseline_id=data.get("resulting_baseline_id"),  # type: ignore[arg-type]
        resulting_baseline_version_id=data.get("resulting_baseline_version_id"),  # type: ignore[arg-type]
        finding_ids=tuple(str(item) for item in data["finding_ids"]),  # type: ignore[index]
        findings=tuple(_finding_from_dict(item) for item in data["findings"]),  # type: ignore[arg-type,index]
        semantics=str(data["semantics"]),
        version=str(data["version"]),
    )


def _activation_admission_from_dict(data: Mapping[str, object]) -> PromotionActivationAdmissionDTO:
    return PromotionActivationAdmissionDTO(
        admission_id=str(data["admission_id"]),
        status=str(data["status"]),
        change_set_id=str(data["change_set_id"]),
        policy_id=str(data["policy_id"]),
        audit_report_id=str(data["audit_report_id"]),
        expected_state_id=str(data["expected_state_id"]),
        expected_baseline_id=str(data["expected_baseline_id"]),
        expected_baseline_version_id=str(data["expected_baseline_version_id"]),
        resulting_state_id=str(data["resulting_state_id"]),
        resulting_baseline_id=str(data["resulting_baseline_id"]),
        resulting_baseline_version_id=str(data["resulting_baseline_version_id"]),
        forward_operation_ids=tuple(str(item) for item in data["forward_operation_ids"]),  # type: ignore[index]
        inverse_operation_ids=tuple(str(item) for item in data["inverse_operation_ids"]),  # type: ignore[index]
        semantics=str(data["semantics"]),
        version=str(data["version"]),
    )


def _rollback_plan_from_dict(data: Mapping[str, object]) -> PromotionRollbackPlanDTO:
    operations = tuple(_operation_from_dict(item) for item in data["inverse_operations"])  # type: ignore[arg-type,index]
    if tuple(item.sequence for item in operations) != tuple(range(1, len(operations) + 1)):
        raise PerceptionPromotionActivationError(
            "stored rollback operation ordinals are malformed"
        )
    return PromotionRollbackPlanDTO(
        rollback_plan_id=str(data["rollback_plan_id"]),
        status=str(data["status"]),
        change_set_id=str(data["change_set_id"]),
        admission_id=str(data["admission_id"]),
        activated_state_id=str(data["activated_state_id"]),
        activated_baseline_id=str(data["activated_baseline_id"]),
        activated_baseline_version_id=str(data["activated_baseline_version_id"]),
        restore_state=_state_from_dict(data["restore_state"]),  # type: ignore[arg-type]
        restore_baseline_id=str(data["restore_baseline_id"]),
        restore_baseline_version_id=str(data["restore_baseline_version_id"]),
        inverse_operation_ids=tuple(str(item) for item in data["inverse_operation_ids"]),  # type: ignore[index]
        inverse_operations=operations,
        semantics=str(data["semantics"]),
        version=str(data["version"]),
    )


def _activation_receipt_from_dict(data: Mapping[str, object]) -> PromotionActivationReceiptDTO:
    return PromotionActivationReceiptDTO(
        receipt_id=str(data["receipt_id"]),
        status=str(data["status"]),
        change_set_id=str(data["change_set_id"]),
        admission_id=str(data["admission_id"]),
        audit_report_id=str(data["audit_report_id"]),
        previous_state_id=str(data["previous_state_id"]),
        resulting_state_id=str(data["resulting_state_id"]),
        previous_baseline_id=str(data["previous_baseline_id"]),
        resulting_baseline_id=str(data["resulting_baseline_id"]),
        previous_baseline_version_id=str(data["previous_baseline_version_id"]),
        resulting_baseline_version_id=str(data["resulting_baseline_version_id"]),
        resulting_revision=int(data["resulting_revision"]),
        forward_operation_ids=tuple(str(item) for item in data["forward_operation_ids"]),  # type: ignore[index]
        rollback_plan_id=str(data["rollback_plan_id"]),
        semantics=str(data["semantics"]),
        version=str(data["version"]),
    )


def _activation_bundle_from_dict(data: Mapping[str, object]) -> PromotionActivationBundleDTO:
    return PromotionActivationBundleDTO(
        bundle_id=str(data["bundle_id"]),
        change_set_id=str(data["change_set_id"]),
        audit_report=_activation_report_from_dict(data["audit_report"]),  # type: ignore[arg-type]
        admission=_activation_admission_from_dict(data["admission"]),  # type: ignore[arg-type]
        rollback_plan=_rollback_plan_from_dict(data["rollback_plan"]),  # type: ignore[arg-type]
        receipt=_activation_receipt_from_dict(data["receipt"]),  # type: ignore[arg-type]
        resulting_state=_state_from_dict(data["resulting_state"]),  # type: ignore[arg-type]
        semantics=str(data["semantics"]),
        version=str(data["version"]),
    )


def _rollback_admission_from_dict(data: Mapping[str, object]) -> PromotionRollbackAdmissionDTO:
    try:
        return PromotionRollbackAdmissionDTO(
            admission_id=str(data["admission_id"]),
            status=str(data["status"]),
            request_id=str(data["request_id"]),
            policy_id=str(data["policy_id"]),
            audit_report_id=str(data["audit_report_id"]),
            rollback_plan_id=str(data["rollback_plan_id"]),
            expected_state_id=str(data["expected_state_id"]),
            expected_revision=int(data["expected_revision"]),
            expected_baseline_id=str(data["expected_baseline_id"]),
            expected_baseline_version_id=str(data["expected_baseline_version_id"]),
            predicted_restore_state_id=str(data["predicted_restore_state_id"]),
            predicted_restore_baseline_id=str(data["predicted_restore_baseline_id"]),
            predicted_restore_baseline_version_id=str(data["predicted_restore_baseline_version_id"]),
            inverse_operation_ids=tuple(str(item) for item in data["inverse_operation_ids"]),  # type: ignore[index]
            semantics=str(data["semantics"]),
            version=str(data["version"]),
        )
    except Exception as exc:
        raise PerceptionPromotionActivationError(
            "stored rollback admission is malformed"
        ) from exc


def _rollback_receipt_from_dict(data: Mapping[str, object]) -> PromotionRollbackReceiptDTO:
    return PromotionRollbackReceiptDTO(
        receipt_id=str(data["receipt_id"]),
        status=str(data["status"]),
        admission_id=str(data["admission_id"]),
        rollback_plan_id=str(data["rollback_plan_id"]),
        activation_receipt_id=str(data["activation_receipt_id"]),
        prior_state_id=str(data["prior_state_id"]),
        restored_state_id=str(data["restored_state_id"]),
        execution_revision=int(data["execution_revision"]),
        inverse_operation_ids=tuple(str(item) for item in data["inverse_operation_ids"]),  # type: ignore[index]
        semantics=str(data["semantics"]),
        version=str(data["version"]),
    )


def _rollback_bundle_from_dict(data: Mapping[str, object]) -> PromotionRollbackBundleDTO:
    return PromotionRollbackBundleDTO(
        bundle_id=str(data["bundle_id"]),
        receipt=_rollback_receipt_from_dict(data["receipt"]),  # type: ignore[arg-type]
        restored_state=_state_from_dict(data["restored_state"]),  # type: ignore[arg-type]
        rollback_plan=_rollback_plan_from_dict(data["rollback_plan"]),  # type: ignore[arg-type]
        activation_receipt=_activation_receipt_from_dict(data["activation_receipt"]),  # type: ignore[arg-type]
        semantics=str(data["semantics"]),
        version=str(data["version"]),
    )


def _loads(raw: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PerceptionPromotionActivationError("malformed persisted JSON") from exc
    if not isinstance(value, dict):
        raise PerceptionPromotionActivationError("persisted DTO payload must be an object")
    return value


def _dumps(value: object) -> str:
    return _canonical_json(asdict(value)).decode("utf-8")  # type: ignore[arg-type]


class SqlitePromotionActivationStore:
    """SQLite-backed reference implementation of the P18G/P18H activation store."""

    version: Final = PROMOTION_ACTIVATION_STORE_VERSION

    def __init__(self, path: str, initial_state: ActivePromotionStateDTO | None = None) -> None:
        self._path = path
        with self._connect() as conn:
            self._create_schema(conn)
            row = conn.execute("SELECT payload FROM active_promotion_state WHERE scope = 'default'").fetchone()
            if row is None:
                if initial_state is None:
                    raise PerceptionPromotionActivationError(
                        "new SQLite activation store requires an initial state"
                    )
                conn.execute(
                    "INSERT INTO active_promotion_state(scope, state_id, revision, baseline_version_id, payload) VALUES('default', ?, ?, ?, ?)",
                    (
                        initial_state.state_id,
                        initial_state.revision,
                        initial_state.baseline_version_id,
                        _dumps(initial_state),
                    ),
                )
            elif initial_state is not None and _state_from_dict(_loads(row[0])) != initial_state:
                raise PerceptionPromotionActivationError(
                    "initial state disagrees with persisted active state"
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version)
                SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
            CREATE TABLE IF NOT EXISTS active_promotion_state(
                scope TEXT PRIMARY KEY,
                state_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                baseline_version_id TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS activation_bundles(
                change_set_id TEXT PRIMARY KEY,
                receipt_id TEXT NOT NULL UNIQUE,
                rollback_plan_id TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS activation_operations(
                change_set_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                operation_id TEXT NOT NULL,
                PRIMARY KEY(change_set_id, direction, ordinal),
                UNIQUE(change_set_id, direction, operation_id)
            );
            CREATE TABLE IF NOT EXISTS rollback_admissions(
                admission_id TEXT PRIMARY KEY,
                rollback_plan_id TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rollback_bundles(
                rollback_plan_id TEXT PRIMARY KEY,
                admission_id TEXT NOT NULL UNIQUE,
                receipt_id TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL
            );
            """
        )
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        if version != PROMOTION_SQLITE_ACTIVATION_SCHEMA_VERSION:
            raise PerceptionPromotionActivationError(
                "unsupported SQLite promotion activation schema version"
            )

    def get_active_state(self) -> ActivePromotionStateDTO:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM active_promotion_state WHERE scope = 'default'").fetchone()
            if row is None:
                raise PerceptionPromotionActivationError("active state is missing")
            return _state_from_dict(_loads(row[0]))

    def commit_activation(
        self,
        expected_state: ActivePromotionStateDTO,
        change_set: PromotionMaterializationChangeSetDTO,
        bundle: PromotionActivationBundleDTO,
    ) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT payload FROM activation_bundles WHERE change_set_id = ?", (change_set.change_set_id,)).fetchone()
            if row is not None:
                if _activation_bundle_from_dict(_loads(row[0])) == bundle:
                    conn.commit()
                    return
                raise PerceptionPromotionActivationError(
                    "promotion change set was already activated differently"
                )
            current = _state_from_dict(_loads(conn.execute("SELECT payload FROM active_promotion_state WHERE scope = 'default'").fetchone()[0]))
            if current != expected_state:
                raise PerceptionPromotionActivationError(
                    "active state changed before atomic activation commit"
                )
            actual = _apply_forward_operations(current, change_set, operation_hook=self)
            if actual != bundle.resulting_state:
                raise PerceptionPromotionActivationError(
                    "atomic store result differs from admitted resulting state"
                )
            self._before_atomic_swap(bundle)
            conn.execute(
                "INSERT INTO activation_bundles(change_set_id, receipt_id, rollback_plan_id, payload) VALUES(?, ?, ?, ?)",
                (
                    change_set.change_set_id,
                    bundle.receipt.receipt_id,
                    bundle.rollback_plan.rollback_plan_id,
                    _dumps(bundle),
                ),
            )
            for operation in change_set.operations("forward"):
                conn.execute(
                    "INSERT INTO activation_operations(change_set_id, direction, ordinal, operation_id) VALUES(?, ?, ?, ?)",
                    (change_set.change_set_id, "forward", operation.sequence, operation.operation_id),
                )
            for operation in bundle.rollback_plan.inverse_operations:
                conn.execute(
                    "INSERT INTO activation_operations(change_set_id, direction, ordinal, operation_id) VALUES(?, ?, ?, ?)",
                    (change_set.change_set_id, "inverse", operation.sequence, operation.operation_id),
                )
            conn.execute(
                "UPDATE active_promotion_state SET state_id = ?, revision = ?, baseline_version_id = ?, payload = ? WHERE scope = 'default'",
                (actual.state_id, actual.revision, actual.baseline_version_id, _dumps(actual)),
            )
            conn.commit()

    def _after_operation_applied(self, operation: PromotionMaterializationOperationDTO) -> None:
        """Extension hook used by fault-injection tests before transaction commit."""

    def _before_atomic_swap(self, bundle: PromotionActivationBundleDTO) -> None:
        """Extension hook used by fault-injection tests before transaction commit."""

    def get_activation_bundle(self, change_set_id: str) -> PromotionActivationBundleDTO:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM activation_bundles WHERE change_set_id = ?", (change_set_id,)).fetchone()
            if row is None:
                raise PerceptionPromotionActivationError(
                    f"unknown activated change set: {change_set_id}"
                )
            bundle = _activation_bundle_from_dict(_loads(row[0]))
            self._validate_operation_rows(conn, bundle)
            return bundle

    def list_activation_bundles(self) -> tuple[PromotionActivationBundleDTO, ...]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM activation_bundles ORDER BY change_set_id").fetchall()
            bundles = tuple(_activation_bundle_from_dict(_loads(row[0])) for row in rows)
            for bundle in bundles:
                self._validate_operation_rows(conn, bundle)
            return bundles

    def get_rollback_plan(self, rollback_plan_id: str) -> PromotionRollbackPlanDTO:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM rollback_bundles WHERE rollback_plan_id = ?", (rollback_plan_id,)).fetchone()
            if row is not None:
                bundle = _rollback_bundle_from_dict(_loads(row[0]))
                activation_row = conn.execute(
                    "SELECT payload FROM activation_bundles WHERE change_set_id = ?",
                    (bundle.activation_receipt.change_set_id,),
                ).fetchone()
                if activation_row is None:
                    raise PerceptionPromotionActivationError(
                        "rollback bundle references a missing activation owner"
                    )
                activation = _activation_bundle_from_dict(_loads(activation_row[0]))
                self._validate_operation_rows(conn, activation)
                return bundle.rollback_plan
            row = conn.execute("SELECT payload FROM activation_bundles WHERE rollback_plan_id = ?", (rollback_plan_id,)).fetchone()
            if row is None:
                raise PerceptionPromotionActivationError(
                    f"unknown rollback plan: {rollback_plan_id}"
                )
            bundle = _activation_bundle_from_dict(_loads(row[0]))
            self._validate_operation_rows(conn, bundle)
            return bundle.rollback_plan

    def _validate_operation_rows(
        self,
        conn: sqlite3.Connection,
        bundle: PromotionActivationBundleDTO,
    ) -> None:
        rows = conn.execute(
            "SELECT direction, ordinal, operation_id FROM activation_operations WHERE change_set_id = ? ORDER BY direction, ordinal",
            (bundle.change_set_id,),
        ).fetchall()
        observed = {
            direction: tuple((ordinal, operation_id) for _, ordinal, operation_id in rows if direction == _)
            for direction in ("forward", "inverse")
        }
        expected = {
            "forward": tuple(
                (index, operation_id)
                for index, operation_id in enumerate(
                    bundle.receipt.forward_operation_ids,
                    start=1,
                )
            ),
            "inverse": tuple(
                (operation.sequence, operation.operation_id)
                for operation in bundle.rollback_plan.inverse_operations
            ),
        }
        if observed != expected:
            raise PerceptionPromotionActivationError(
                "persisted activation operation ordinals disagree with bundle payload"
            )

    def admit_rollback(
        self,
        request: PromotionRollbackRequestDTO,
        policy: PromotionRollbackPolicyDTO | None = None,
    ) -> PromotionRollbackAdmissionDTO:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT payload FROM rollback_bundles WHERE rollback_plan_id = ?", (request.rollback_plan_id,)).fetchone()
            if row is not None:
                raise PerceptionPromotionActivationError("rollback plan already executed")
            resolved = policy or PromotionRollbackPolicyDTO.create()
            current = _state_from_dict(_loads(conn.execute("SELECT payload FROM active_promotion_state WHERE scope = 'default'").fetchone()[0]))
            plan_row = conn.execute("SELECT payload FROM activation_bundles WHERE rollback_plan_id = ?", (request.rollback_plan_id,)).fetchone()
            if plan_row is None:
                plan = None
            else:
                activation = _activation_bundle_from_dict(_loads(plan_row[0]))
                self._validate_operation_rows(conn, activation)
                plan = activation.rollback_plan
            report = audit_promotion_rollback(current, plan, request, resolved)
            if plan is None:
                raise PerceptionPromotionActivationError(
                    "promotion rollback audit is not admissible"
                )
            admission, _ = authorize_promotion_rollback(report, current, plan, request, resolved)
            existing = conn.execute("SELECT payload FROM rollback_admissions WHERE admission_id = ?", (admission.admission_id,)).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO rollback_admissions(admission_id, rollback_plan_id, payload) VALUES(?, ?, ?)",
                    (admission.admission_id, admission.rollback_plan_id, _dumps(admission)),
                )
            elif _rollback_admission_from_dict(_loads(existing[0])) != admission:
                raise PerceptionPromotionActivationError(
                    "rollback admission identity collision"
                )
            conn.commit()
            return admission

    def _after_rollback_operation_applied(self, operation: PromotionMaterializationOperationDTO) -> None:
        """Extension hook used by fault-injection tests before transaction commit."""

    def _before_rollback_atomic_swap(self, bundle: PromotionRollbackBundleDTO) -> None:
        """Extension hook used by fault-injection tests before transaction commit."""

    def commit_rollback(
        self,
        admission: PromotionRollbackAdmissionDTO,
    ) -> PromotionRollbackBundleDTO:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT payload FROM rollback_bundles WHERE rollback_plan_id = ?", (admission.rollback_plan_id,)).fetchone()
            if row is not None:
                existing = _rollback_bundle_from_dict(_loads(row[0]))
                if existing.receipt.admission_id != admission.admission_id:
                    raise PerceptionPromotionActivationError(
                        "rollback plan already executed by a different admission"
                    )
                conn.commit()
                return existing
            row = conn.execute("SELECT payload FROM rollback_admissions WHERE admission_id = ?", (admission.admission_id,)).fetchone()
            if row is None or _rollback_admission_from_dict(_loads(row[0])) != admission:
                raise PerceptionPromotionActivationError("rollback admission was not durably admitted")
            current = _state_from_dict(_loads(conn.execute("SELECT payload FROM active_promotion_state WHERE scope = 'default'").fetchone()[0]))
            if (
                current.state_id != admission.expected_state_id
                or current.revision != admission.expected_revision
                or current.baseline().baseline_id != admission.expected_baseline_id
                or current.baseline_version_id != admission.expected_baseline_version_id
            ):
                raise PerceptionPromotionActivationError(
                    "active state changed before atomic rollback commit"
                )
            activation_row = conn.execute("SELECT payload FROM activation_bundles WHERE rollback_plan_id = ?", (admission.rollback_plan_id,)).fetchone()
            if activation_row is None:
                raise PerceptionPromotionActivationError("rollback plan has no activation owner")
            activation = _activation_bundle_from_dict(_loads(activation_row[0]))
            self._validate_operation_rows(conn, activation)
            restored = _apply_inverse_operations(current, activation.rollback_plan, operation_hook=self)
            if restored.state_id != admission.predicted_restore_state_id:
                raise PerceptionPromotionActivationError(
                    "rollback result differs from admitted restored state"
                )
            receipt_values: dict[str, object] = {
                "status": PROMOTION_ROLLBACK_RECEIPT_STATUS,
                "admission_id": admission.admission_id,
                "rollback_plan_id": activation.rollback_plan.rollback_plan_id,
                "activation_receipt_id": activation.receipt.receipt_id,
                "prior_state_id": current.state_id,
                "restored_state_id": restored.state_id,
                "execution_revision": current.revision + 1,
                "inverse_operation_ids": activation.rollback_plan.inverse_operation_ids,
                "semantics": PROMOTION_ACTIVATION_SEMANTICS,
                "version": PROMOTION_ROLLBACK_RECEIPT_VERSION,
            }
            receipt = PromotionRollbackReceiptDTO(
                receipt_id=_digest(receipt_values),
                **receipt_values,  # type: ignore[arg-type]
            )
            bundle_values: dict[str, object] = {
                "receipt": receipt,
                "restored_state": restored,
                "rollback_plan": activation.rollback_plan,
                "activation_receipt": activation.receipt,
                "semantics": PROMOTION_ACTIVATION_SEMANTICS,
                "version": PROMOTION_ROLLBACK_BUNDLE_VERSION,
            }
            identity_values = dict(bundle_values)
            for name in ("receipt", "restored_state", "rollback_plan", "activation_receipt"):
                identity_values[name] = asdict(bundle_values[name])  # type: ignore[arg-type]
            bundle = PromotionRollbackBundleDTO(
                bundle_id=_digest(identity_values),
                **bundle_values,  # type: ignore[arg-type]
            )
            self._before_rollback_atomic_swap(bundle)
            conn.execute(
                "INSERT INTO rollback_bundles(rollback_plan_id, admission_id, receipt_id, payload) VALUES(?, ?, ?, ?)",
                (activation.rollback_plan.rollback_plan_id, admission.admission_id, receipt.receipt_id, _dumps(bundle)),
            )
            conn.execute(
                "UPDATE active_promotion_state SET state_id = ?, revision = ?, baseline_version_id = ?, payload = ? WHERE scope = 'default'",
                (restored.state_id, restored.revision, restored.baseline_version_id, _dumps(restored)),
            )
            conn.commit()
            return bundle


def audit_promotion_activation(
    current: ActivePromotionStateDTO,
    change_set: PromotionMaterializationChangeSetDTO,
    policy: PromotionActivationPolicyDTO | None = None,
) -> PromotionActivationAuditReportDTO:
    """Audit a staged P18F change set against one exact active-state snapshot."""

    resolved = policy or PromotionActivationPolicyDTO.create()
    observed_baseline = current.baseline()
    findings: list[PromotionActivationAuditFindingDTO] = []
    resulting: ActivePromotionStateDTO | None = None
    if change_set.status == "no_approved_changes":
        findings.append(
            _finding(
                severity="info",
                code="no_approved_changes",
                subject_id=change_set.change_set_id,
                detail="The fully reviewed materialization contains no changes to activate.",
            )
        )
        status = "not_applicable"
    else:
        if change_set.status != "staged_inactive" or not change_set.changes:
            findings.append(
                _finding(
                    severity="error",
                    code="change_set_not_staged",
                    subject_id=change_set.change_set_id,
                    detail="Activation requires a non-empty staged_inactive P18F change set.",
                )
            )
        if len(change_set.changes) > resolved.maximum_change_count:
            findings.append(
                _finding(
                    severity="error",
                    code="change_count_exceeds_policy",
                    subject_id=change_set.change_set_id,
                    detail="The staged change count exceeds the activation policy limit.",
                )
            )
        disallowed = {
            item.target_kind for item in change_set.changes
        } - set(resolved.allowed_target_kinds)
        if disallowed:
            findings.append(
                _finding(
                    severity="error",
                    code="target_kind_disallowed",
                    subject_id=change_set.change_set_id,
                    detail=f"Activation policy disallows target kinds: {sorted(disallowed)}",
                )
            )
        if current.field_schema_id != change_set.field_schema_id:
            findings.append(
                _finding(
                    severity="error",
                    code="field_schema_mismatch",
                    subject_id=current.state_id,
                    detail="The active field schema differs from the staged change set.",
                )
            )
        if observed_baseline.baseline_version_id != change_set.baseline_version_id:
            findings.append(
                _finding(
                    severity="error",
                    code="baseline_version_mismatch",
                    subject_id=current.state_id,
                    detail="The active baseline version changed after materialization.",
                )
            )
        if observed_baseline.baseline_id != change_set.baseline_id:
            findings.append(
                _finding(
                    severity="error",
                    code="baseline_identity_mismatch",
                    subject_id=current.state_id,
                    detail="The active annotation/relation/expectation identity baseline changed.",
                )
            )
        if not any(item.severity == "error" for item in findings):
            try:
                resulting = _apply_forward_operations(current, change_set)
            except Exception as exc:
                findings.append(
                    _finding(
                        severity="error",
                        code="forward_plan_invalid",
                        subject_id=change_set.change_set_id,
                        detail=str(exc),
                    )
                )
        if any(item.severity == "error" for item in findings):
            status = "blocked"
            resulting = None
        else:
            findings.append(
                _finding(
                    severity="info",
                    code="activation_admissible",
                    subject_id=change_set.change_set_id,
                    detail="Exact baseline and forward-plan audits authorize activation.",
                )
            )
            status = "admissible"
    ordered_findings = tuple(sorted(findings, key=lambda item: item.finding_id))
    resulting_baseline = resulting.baseline() if resulting is not None else None
    values: dict[str, object] = {
        "status": status,
        "change_set_id": change_set.change_set_id,
        "policy_id": resolved.policy_id,
        "observed_state_id": current.state_id,
        "observed_baseline_id": observed_baseline.baseline_id,
        "observed_baseline_version_id": current.baseline_version_id,
        "expected_baseline_id": change_set.baseline_id,
        "expected_baseline_version_id": change_set.baseline_version_id,
        "resulting_state_id": None if resulting is None else resulting.state_id,
        "resulting_baseline_id": (
            None if resulting_baseline is None else resulting_baseline.baseline_id
        ),
        "resulting_baseline_version_id": (
            None if resulting is None else resulting.baseline_version_id
        ),
        "finding_ids": tuple(item.finding_id for item in ordered_findings),
        "findings": ordered_findings,
        "semantics": PROMOTION_ACTIVATION_SEMANTICS,
        "version": PROMOTION_ACTIVATION_AUDIT_REPORT_VERSION,
    }
    identity_values = dict(values)
    identity_values["findings"] = tuple(asdict(item) for item in ordered_findings)
    return PromotionActivationAuditReportDTO(
        report_id=_digest(identity_values),
        **values,  # type: ignore[arg-type]
    )


def authorize_promotion_activation(
    report: PromotionActivationAuditReportDTO,
    current: ActivePromotionStateDTO,
    change_set: PromotionMaterializationChangeSetDTO,
    policy: PromotionActivationPolicyDTO,
) -> tuple[PromotionActivationAdmissionDTO, ActivePromotionStateDTO]:
    """Bind an admissible audit to its exact expected and resulting states."""

    if report.status != "admissible":
        raise PerceptionPromotionActivationError(
            "promotion activation audit is not admissible"
        )
    current_baseline = current.baseline()
    if (
        report.change_set_id != change_set.change_set_id
        or report.policy_id != policy.policy_id
        or report.observed_state_id != current.state_id
        or report.observed_baseline_id != current_baseline.baseline_id
        or report.observed_baseline_version_id != current.baseline_version_id
    ):
        raise PerceptionPromotionActivationError(
            "promotion activation audit no longer matches the supplied pre-state"
        )
    resulting = _apply_forward_operations(current, change_set)
    resulting_baseline = resulting.baseline()
    if (
        report.resulting_state_id != resulting.state_id
        or report.resulting_baseline_id != resulting_baseline.baseline_id
        or report.resulting_baseline_version_id != resulting.baseline_version_id
    ):
        raise PerceptionPromotionActivationError(
            "promotion activation audit resulting-state prediction disagrees"
        )
    values: dict[str, object] = {
        "status": PROMOTION_ACTIVATION_ADMISSION_STATUS,
        "change_set_id": change_set.change_set_id,
        "policy_id": policy.policy_id,
        "audit_report_id": report.report_id,
        "expected_state_id": current.state_id,
        "expected_baseline_id": current_baseline.baseline_id,
        "expected_baseline_version_id": current.baseline_version_id,
        "resulting_state_id": resulting.state_id,
        "resulting_baseline_id": resulting_baseline.baseline_id,
        "resulting_baseline_version_id": resulting.baseline_version_id,
        "forward_operation_ids": change_set.forward_operation_ids,
        "inverse_operation_ids": change_set.inverse_operation_ids,
        "semantics": PROMOTION_ACTIVATION_SEMANTICS,
        "version": PROMOTION_ACTIVATION_ADMISSION_VERSION,
    }
    admission = PromotionActivationAdmissionDTO(
        admission_id=_digest(values),
        **values,  # type: ignore[arg-type]
    )
    return admission, resulting


def build_promotion_activation_bundle(
    current: ActivePromotionStateDTO,
    change_set: PromotionMaterializationChangeSetDTO,
    policy: PromotionActivationPolicyDTO | None = None,
) -> PromotionActivationBundleDTO:
    """Audit and prepare all immutable artifacts required for one atomic commit."""

    resolved = policy or PromotionActivationPolicyDTO.create()
    report = audit_promotion_activation(current, change_set, resolved)
    admission, resulting = authorize_promotion_activation(
        report,
        current,
        change_set,
        resolved,
    )
    current_baseline = current.baseline()
    resulting_baseline = resulting.baseline()
    inverse_operations = change_set.operations("inverse")
    rollback_values: dict[str, object] = {
        "status": PROMOTION_ROLLBACK_PLAN_STATUS,
        "change_set_id": change_set.change_set_id,
        "admission_id": admission.admission_id,
        "activated_state_id": resulting.state_id,
        "activated_baseline_id": resulting_baseline.baseline_id,
        "activated_baseline_version_id": resulting.baseline_version_id,
        "restore_state": current,
        "restore_baseline_id": current_baseline.baseline_id,
        "restore_baseline_version_id": current.baseline_version_id,
        "inverse_operation_ids": change_set.inverse_operation_ids,
        "inverse_operations": inverse_operations,
        "semantics": PROMOTION_ACTIVATION_SEMANTICS,
        "version": PROMOTION_ROLLBACK_PLAN_VERSION,
    }
    rollback_identity = dict(rollback_values)
    rollback_identity["restore_state"] = asdict(current)
    rollback_identity["inverse_operations"] = tuple(
        asdict(item) for item in inverse_operations
    )
    rollback_plan = PromotionRollbackPlanDTO(
        rollback_plan_id=_digest(rollback_identity),
        **rollback_values,  # type: ignore[arg-type]
    )
    receipt_values: dict[str, object] = {
        "status": PROMOTION_ACTIVATION_RECEIPT_STATUS,
        "change_set_id": change_set.change_set_id,
        "admission_id": admission.admission_id,
        "audit_report_id": report.report_id,
        "previous_state_id": current.state_id,
        "resulting_state_id": resulting.state_id,
        "previous_baseline_id": current_baseline.baseline_id,
        "resulting_baseline_id": resulting_baseline.baseline_id,
        "previous_baseline_version_id": current.baseline_version_id,
        "resulting_baseline_version_id": resulting.baseline_version_id,
        "resulting_revision": resulting.revision,
        "forward_operation_ids": change_set.forward_operation_ids,
        "rollback_plan_id": rollback_plan.rollback_plan_id,
        "semantics": PROMOTION_ACTIVATION_SEMANTICS,
        "version": PROMOTION_ACTIVATION_RECEIPT_VERSION,
    }
    receipt = PromotionActivationReceiptDTO(
        receipt_id=_digest(receipt_values),
        **receipt_values,  # type: ignore[arg-type]
    )
    bundle_values: dict[str, object] = {
        "change_set_id": change_set.change_set_id,
        "audit_report": report,
        "admission": admission,
        "rollback_plan": rollback_plan,
        "receipt": receipt,
        "resulting_state": resulting,
        "semantics": PROMOTION_ACTIVATION_SEMANTICS,
        "version": PROMOTION_ACTIVATION_BUNDLE_VERSION,
    }
    bundle_identity = dict(bundle_values)
    for name in (
        "audit_report",
        "admission",
        "rollback_plan",
        "receipt",
        "resulting_state",
    ):
        bundle_identity[name] = asdict(bundle_values[name])  # type: ignore[arg-type]
    return PromotionActivationBundleDTO(
        bundle_id=_digest(bundle_identity),
        **bundle_values,  # type: ignore[arg-type]
    )


class InMemoryPromotionActivationStore:
    """Atomic copy-on-write reference implementation of the P18G store protocol."""

    version: Final = PROMOTION_ACTIVATION_STORE_VERSION

    def __init__(self, initial_state: ActivePromotionStateDTO) -> None:
        self._state = initial_state
        self._bundles: dict[str, PromotionActivationBundleDTO] = {}
        self._rollback_admissions: dict[str, PromotionRollbackAdmissionDTO] = {}
        self._rollback_bundles: dict[str, PromotionRollbackBundleDTO] = {}
        self._lock = threading.RLock()

    def get_active_state(self) -> ActivePromotionStateDTO:
        with self._lock:
            return self._state

    def _after_operation_applied(
        self,
        operation: PromotionMaterializationOperationDTO,
    ) -> None:
        """Extension hook used by fault-injection tests before any atomic swap."""

    def _before_atomic_swap(self, bundle: PromotionActivationBundleDTO) -> None:
        """Extension hook used by store implementations before the final swap."""

    def commit_activation(
        self,
        expected_state: ActivePromotionStateDTO,
        change_set: PromotionMaterializationChangeSetDTO,
        bundle: PromotionActivationBundleDTO,
    ) -> None:
        with self._lock:
            current = self._state
            if current != expected_state or current.state_id != expected_state.state_id:
                raise PerceptionPromotionActivationError(
                    "active state changed before atomic activation commit"
                )
            current_baseline = current.baseline()
            admission = bundle.admission
            if (
                admission.expected_state_id != current.state_id
                or admission.expected_baseline_id != current_baseline.baseline_id
                or admission.expected_baseline_version_id
                != current.baseline_version_id
            ):
                raise PerceptionPromotionActivationError(
                    "activation admission does not authorize the current active baseline"
                )
            if bundle.change_set_id != change_set.change_set_id:
                raise PerceptionPromotionActivationError(
                    "activation bundle does not reference the supplied change set"
                )
            if change_set.change_set_id in self._bundles:
                raise PerceptionPromotionActivationError(
                    "promotion change set was already activated"
                )
            actual_result = _apply_forward_operations(
                current,
                change_set,
                operation_hook=self,
            )
            if actual_result != bundle.resulting_state:
                raise PerceptionPromotionActivationError(
                    "atomic store result differs from admitted resulting state"
                )
            if (
                bundle.rollback_plan.inverse_operation_ids
                != change_set.inverse_operation_ids
                or bundle.receipt.forward_operation_ids
                != change_set.forward_operation_ids
            ):
                raise PerceptionPromotionActivationError(
                    "activation bundle operation lineage disagrees with change set"
                )
            next_bundles = dict(self._bundles)
            next_bundles[change_set.change_set_id] = bundle
            self._before_atomic_swap(bundle)
            self._state = actual_result
            self._bundles = next_bundles

    def get_activation_bundle(self, change_set_id: str) -> PromotionActivationBundleDTO:
        with self._lock:
            try:
                return self._bundles[change_set_id]
            except KeyError as exc:
                raise PerceptionPromotionActivationError(
                    f"unknown activated change set: {change_set_id}"
                ) from exc

    def list_activation_bundles(self) -> tuple[PromotionActivationBundleDTO, ...]:
        with self._lock:
            return tuple(
                self._bundles[key] for key in sorted(self._bundles)
            )

    def get_rollback_plan(self, rollback_plan_id: str) -> PromotionRollbackPlanDTO:
        with self._lock:
            for bundle in self._bundles.values():
                if bundle.rollback_plan.rollback_plan_id == rollback_plan_id:
                    executed = self._rollback_bundles.get(rollback_plan_id)
                    return bundle.rollback_plan if executed is None else executed.rollback_plan
            raise PerceptionPromotionActivationError(
                f"unknown rollback plan: {rollback_plan_id}"
            )

    def _get_activation_bundle_by_plan(
        self,
        rollback_plan_id: str,
    ) -> PromotionActivationBundleDTO:
        for bundle in self._bundles.values():
            if bundle.rollback_plan.rollback_plan_id == rollback_plan_id:
                return bundle
        raise PerceptionPromotionActivationError(
            f"unknown rollback plan: {rollback_plan_id}"
        )

    def admit_rollback(
        self,
        request: PromotionRollbackRequestDTO,
        policy: PromotionRollbackPolicyDTO | None = None,
    ) -> PromotionRollbackAdmissionDTO:
        with self._lock:
            if request.rollback_plan_id in self._rollback_bundles:
                raise PerceptionPromotionActivationError(
                    "rollback plan already executed"
                )
            resolved = policy or PromotionRollbackPolicyDTO.create()
            plan = self.get_rollback_plan(request.rollback_plan_id)
            report = audit_promotion_rollback(self._state, plan, request, resolved)
            admission, _ = authorize_promotion_rollback(
                report,
                self._state,
                plan,
                request,
                resolved,
            )
            self._rollback_admissions[admission.admission_id] = admission
            return admission

    def _after_rollback_operation_applied(
        self,
        operation: PromotionMaterializationOperationDTO,
    ) -> None:
        """Extension hook used by fault-injection tests before any atomic swap."""

    def _before_rollback_atomic_swap(self, bundle: PromotionRollbackBundleDTO) -> None:
        """Extension hook used by store implementations before the final swap."""

    def commit_rollback(
        self,
        admission: PromotionRollbackAdmissionDTO,
    ) -> PromotionRollbackBundleDTO:
        with self._lock:
            existing = self._rollback_bundles.get(admission.rollback_plan_id)
            if existing is not None:
                if existing.receipt.admission_id != admission.admission_id:
                    raise PerceptionPromotionActivationError(
                        "rollback plan already executed by a different admission"
                    )
                return existing
            persisted_admission = self._rollback_admissions.get(admission.admission_id)
            if persisted_admission != admission:
                raise PerceptionPromotionActivationError(
                    "rollback admission was not durably admitted"
                )
            current = self._state
            if (
                current.state_id != admission.expected_state_id
                or current.revision != admission.expected_revision
                or current.baseline().baseline_id != admission.expected_baseline_id
                or current.baseline_version_id != admission.expected_baseline_version_id
            ):
                raise PerceptionPromotionActivationError(
                    "active state changed before atomic rollback commit"
                )
            activation = self._get_activation_bundle_by_plan(admission.rollback_plan_id)
            plan = activation.rollback_plan
            restored = _apply_inverse_operations(current, plan, operation_hook=self)
            if (
                restored.state_id != admission.predicted_restore_state_id
                or restored.baseline().baseline_id
                != admission.predicted_restore_baseline_id
                or restored.baseline_version_id
                != admission.predicted_restore_baseline_version_id
            ):
                raise PerceptionPromotionActivationError(
                    "rollback result differs from admitted restored state"
                )
            receipt_values: dict[str, object] = {
                "status": PROMOTION_ROLLBACK_RECEIPT_STATUS,
                "admission_id": admission.admission_id,
                "rollback_plan_id": plan.rollback_plan_id,
                "activation_receipt_id": activation.receipt.receipt_id,
                "prior_state_id": current.state_id,
                "restored_state_id": restored.state_id,
                "execution_revision": current.revision + 1,
                "inverse_operation_ids": plan.inverse_operation_ids,
                "semantics": PROMOTION_ACTIVATION_SEMANTICS,
                "version": PROMOTION_ROLLBACK_RECEIPT_VERSION,
            }
            receipt = PromotionRollbackReceiptDTO(
                receipt_id=_digest(receipt_values),
                **receipt_values,  # type: ignore[arg-type]
            )
            bundle_values: dict[str, object] = {
                "receipt": receipt,
                "restored_state": restored,
                "rollback_plan": plan,
                "activation_receipt": activation.receipt,
                "semantics": PROMOTION_ACTIVATION_SEMANTICS,
                "version": PROMOTION_ROLLBACK_BUNDLE_VERSION,
            }
            identity_values = dict(bundle_values)
            for name in ("receipt", "restored_state", "rollback_plan", "activation_receipt"):
                identity_values[name] = asdict(bundle_values[name])  # type: ignore[arg-type]
            bundle = PromotionRollbackBundleDTO(
                bundle_id=_digest(identity_values),
                **bundle_values,  # type: ignore[arg-type]
            )
            next_rollbacks = dict(self._rollback_bundles)
            next_rollbacks[plan.rollback_plan_id] = bundle
            self._before_rollback_atomic_swap(bundle)
            self._state = restored
            self._rollback_bundles = next_rollbacks
            return bundle


def execute_promotion_activation(
    store: PromotionActivationStore,
    change_set: PromotionMaterializationChangeSetDTO,
    policy: PromotionActivationPolicyDTO | None = None,
) -> PromotionActivationBundleDTO:
    """Audit, admit, and atomically activate one P18F change set or change nothing."""

    current = store.get_active_state()
    bundle = build_promotion_activation_bundle(current, change_set, policy)
    store.commit_activation(current, change_set, bundle)
    resulting = store.get_active_state()
    if resulting != bundle.resulting_state:
        raise PerceptionPromotionActivationError(
            "activation store committed a state that differs from the receipt"
        )
    persisted = store.get_activation_bundle(change_set.change_set_id)
    if persisted != bundle:
        raise PerceptionPromotionActivationError(
            "activation store did not persist the exact activation bundle"
        )
    return bundle


def execute_promotion_rollback(
    store: PromotionActivationStore,
    request: PromotionRollbackRequestDTO,
    policy: PromotionRollbackPolicyDTO | None = None,
) -> PromotionRollbackBundleDTO:
    """Audit, admit, and atomically execute one exact stored inverse rollback plan."""

    admission = store.admit_rollback(request, policy)
    bundle = store.commit_rollback(admission)
    if store.get_active_state() != bundle.restored_state:
        raise PerceptionPromotionActivationError(
            "rollback store committed a state that differs from the receipt"
        )
    return bundle

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
import threading
from dataclasses import asdict, dataclass
from typing import Final, Mapping, Protocol

from .expectations import PerceptionRegionAnnotationDTO, RelationAnnotationDTO
from .promotion_materialization import (
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
        if self.status != PROMOTION_ROLLBACK_PLAN_STATUS:
            raise PerceptionPromotionActivationError(
                "promotion rollback plans must remain stored_inactive"
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

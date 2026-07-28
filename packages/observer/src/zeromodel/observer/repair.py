"""Canonical DTOs for bounded Observer repair proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id

OBSERVER_REPAIR_CONSTRAINT_VERSION: Final = "observer-repair-constraint/1"
OBSERVER_PROPOSED_CHANGE_VERSION: Final = "observer-proposed-change/1"
OBSERVER_REPAIR_PROPOSAL_VERSION: Final = "observer-repair-proposal/2"

REPAIR_DISPOSITION_REPAIRABLE: Final = "repairable"
REPAIR_DISPOSITION_REQUIRES_SCHEMA_EXTENSION: Final = "requires_schema_extension"
REPAIR_DISPOSITION_INSUFFICIENT_EVIDENCE: Final = "insufficient_evidence"
REPAIR_DISPOSITION_UNSUPPORTED: Final = "unsupported"
REPAIR_DISPOSITIONS: Final = frozenset(
    {
        REPAIR_DISPOSITION_REPAIRABLE,
        REPAIR_DISPOSITION_REQUIRES_SCHEMA_EXTENSION,
        REPAIR_DISPOSITION_INSUFFICIENT_EVIDENCE,
        REPAIR_DISPOSITION_UNSUPPORTED,
    }
)

CHANGE_TARGET_KINDS: Final = frozenset(
    {
        "policy_row",
        "policy_cell",
        "transition_prediction",
        "context_precondition",
    }
)
CHANGE_OPERATIONS: Final = frozenset(
    {"replace", "add_precondition", "remove", "suspend"}
)
REPAIR_RATIONALE_CODES: Final = frozenset(
    {
        "transition_prediction_mismatch",
        "action_effect_mismatch",
        "observable_state_mismatch",
        "policy_consequence_mismatch",
        "hidden_state_exhausted",
        "missing_required_context",
        "affected_row_localised",
        "repair_scope_bounded",
        "unsupported_repair_type",
    }
)


class ObserverRepairProposalError(ValueError):
    """Raised when a bounded Observer repair proposal is invalid."""


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverRepairProposalError(f"{field_name} must be unique and sorted")


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverRepairProposalError(f"{field_name} must be non-empty")


def _ensure_namespaced(keys: tuple[str, ...], field_name: str) -> None:
    for key in keys:
        if not key.startswith(("visible.", "history.", "hidden.")):
            raise ObserverRepairProposalError(
                f"{field_name} contains non-namespaced context key: {key!r}"
            )


@dataclass(frozen=True)
class ObserverRepairConstraintDTO:
    """Declared authority boundary for one repair proposal."""

    repair_constraint_id: str
    allowed_row_ids: tuple[str, ...]
    allowed_cell_ids: tuple[str, ...]
    allowed_context_keys: tuple[str, ...]
    forbidden_row_ids: tuple[str, ...]
    max_changed_rows: int
    max_changed_cells: int
    allow_action_value_change: bool
    allow_transition_prediction_change: bool
    allow_new_context_precondition: bool
    allow_schema_extension_request: bool
    version: str = OBSERVER_REPAIR_CONSTRAINT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_REPAIR_CONSTRAINT_VERSION:
            raise ObserverRepairProposalError("unsupported repair constraint version")
        for field_name in (
            "allowed_row_ids",
            "allowed_cell_ids",
            "allowed_context_keys",
            "forbidden_row_ids",
        ):
            _ensure_sorted_unique(getattr(self, field_name), field_name)
        _ensure_namespaced(self.allowed_context_keys, "allowed_context_keys")
        if self.max_changed_rows < 0:
            raise ObserverRepairProposalError("max_changed_rows must be non-negative")
        if self.max_changed_cells < 0:
            raise ObserverRepairProposalError("max_changed_cells must be non-negative")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.repair_constraint_id != expected_id:
            raise ObserverRepairProposalError(
                "repair_constraint_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "allow_action_value_change": self.allow_action_value_change,
            "allow_new_context_precondition": self.allow_new_context_precondition,
            "allow_schema_extension_request": self.allow_schema_extension_request,
            "allow_transition_prediction_change": (
                self.allow_transition_prediction_change
            ),
            "allowed_cell_ids": list(self.allowed_cell_ids),
            "allowed_context_keys": list(self.allowed_context_keys),
            "allowed_row_ids": list(self.allowed_row_ids),
            "forbidden_row_ids": list(self.forbidden_row_ids),
            "max_changed_cells": self.max_changed_cells,
            "max_changed_rows": self.max_changed_rows,
            "version": self.version,
        }
        if include_id:
            payload["repair_constraint_id"] = self.repair_constraint_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        allowed_row_ids: tuple[str, ...],
        allowed_cell_ids: tuple[str, ...],
        allowed_context_keys: tuple[str, ...] = (),
        forbidden_row_ids: tuple[str, ...] = (),
        max_changed_rows: int = 0,
        max_changed_cells: int = 0,
        allow_action_value_change: bool = False,
        allow_transition_prediction_change: bool = False,
        allow_new_context_precondition: bool = False,
        allow_schema_extension_request: bool = False,
    ) -> "ObserverRepairConstraintDTO":
        payload = {
            "allow_action_value_change": allow_action_value_change,
            "allow_new_context_precondition": allow_new_context_precondition,
            "allow_schema_extension_request": allow_schema_extension_request,
            "allow_transition_prediction_change": allow_transition_prediction_change,
            "allowed_cell_ids": list(allowed_cell_ids),
            "allowed_context_keys": list(allowed_context_keys),
            "allowed_row_ids": list(allowed_row_ids),
            "forbidden_row_ids": list(forbidden_row_ids),
            "max_changed_cells": max_changed_cells,
            "max_changed_rows": max_changed_rows,
            "version": OBSERVER_REPAIR_CONSTRAINT_VERSION,
        }
        return cls(
            repair_constraint_id=canonical_id(payload),
            allowed_row_ids=allowed_row_ids,
            allowed_cell_ids=allowed_cell_ids,
            allowed_context_keys=allowed_context_keys,
            forbidden_row_ids=forbidden_row_ids,
            max_changed_rows=max_changed_rows,
            max_changed_cells=max_changed_cells,
            allow_action_value_change=allow_action_value_change,
            allow_transition_prediction_change=allow_transition_prediction_change,
            allow_new_context_precondition=allow_new_context_precondition,
            allow_schema_extension_request=allow_schema_extension_request,
        )


@dataclass(frozen=True)
class ObserverProposedChangeDTO:
    """One explicit requested change within a bounded repair proposal."""

    change_id: str
    target_kind: str
    target_id: str
    operation: str
    field_name: str
    old_value: object | None
    proposed_value: object | None
    condition_keys: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    version: str = OBSERVER_PROPOSED_CHANGE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_PROPOSED_CHANGE_VERSION:
            raise ObserverRepairProposalError("unsupported proposed change version")
        if self.target_kind not in CHANGE_TARGET_KINDS:
            raise ObserverRepairProposalError(
                f"unsupported proposed change target_kind: {self.target_kind!r}"
            )
        if self.operation not in CHANGE_OPERATIONS:
            raise ObserverRepairProposalError(
                f"unsupported proposed change operation: {self.operation!r}"
            )
        for field_name in ("target_id", "field_name"):
            _require_non_empty(getattr(self, field_name), field_name)
        _ensure_sorted_unique(self.condition_keys, "condition_keys")
        _ensure_sorted_unique(self.evidence_ids, "evidence_ids")
        _ensure_namespaced(self.condition_keys, "condition_keys")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.change_id != expected_id:
            raise ObserverRepairProposalError(
                "change_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "condition_keys": list(self.condition_keys),
            "evidence_ids": list(self.evidence_ids),
            "field_name": self.field_name,
            "old_value": self.old_value,
            "operation": self.operation,
            "proposed_value": self.proposed_value,
            "target_id": self.target_id,
            "target_kind": self.target_kind,
            "version": self.version,
        }
        if include_id:
            payload["change_id"] = self.change_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        target_kind: str,
        target_id: str,
        operation: str,
        field_name: str,
        old_value: object | None = None,
        proposed_value: object | None = None,
        condition_keys: tuple[str, ...] = (),
        evidence_ids: tuple[str, ...] = (),
    ) -> "ObserverProposedChangeDTO":
        payload = {
            "condition_keys": list(condition_keys),
            "evidence_ids": list(evidence_ids),
            "field_name": field_name,
            "old_value": old_value,
            "operation": operation,
            "proposed_value": proposed_value,
            "target_id": target_id,
            "target_kind": target_kind,
            "version": OBSERVER_PROPOSED_CHANGE_VERSION,
        }
        return cls(
            change_id=canonical_id(payload),
            target_kind=target_kind,
            target_id=target_id,
            operation=operation,
            field_name=field_name,
            old_value=old_value,
            proposed_value=proposed_value,
            condition_keys=condition_keys,
            evidence_ids=evidence_ids,
        )


@dataclass(frozen=True)
class ObserverRepairProposalDTO:
    """Canonical bounded repair proposal derived from a verified contradiction."""

    repair_proposal_id: str
    source_policy_artifact_id: str
    transition_verification_id: str
    contradiction_artifact_id: str
    repair_constraint_id: str
    disposition: str
    affected_row_ids: tuple[str, ...]
    affected_cell_ids: tuple[str, ...]
    required_context_keys: tuple[str, ...]
    missing_schema_keys: tuple[str, ...]
    requested_changes: tuple[ObserverProposedChangeDTO, ...]
    proposed_changes: tuple[ObserverProposedChangeDTO, ...]
    rationale_codes: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    version: str = OBSERVER_REPAIR_PROPOSAL_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_REPAIR_PROPOSAL_VERSION:
            raise ObserverRepairProposalError("unsupported repair proposal version")
        if self.disposition not in REPAIR_DISPOSITIONS:
            raise ObserverRepairProposalError(
                f"unsupported repair disposition: {self.disposition!r}"
            )
        for field_name in (
            "source_policy_artifact_id",
            "transition_verification_id",
            "contradiction_artifact_id",
            "repair_constraint_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        for field_name in (
            "affected_row_ids",
            "affected_cell_ids",
            "required_context_keys",
            "missing_schema_keys",
            "rationale_codes",
            "evidence_ids",
        ):
            _ensure_sorted_unique(getattr(self, field_name), field_name)
        _ensure_namespaced(self.required_context_keys, "required_context_keys")
        _ensure_namespaced(self.missing_schema_keys, "missing_schema_keys")
        unknown_rationales = set(self.rationale_codes) - REPAIR_RATIONALE_CODES
        if unknown_rationales:
            raise ObserverRepairProposalError(
                f"unsupported rationale_codes: {sorted(unknown_rationales)}"
            )
        for field_name in ("requested_changes", "proposed_changes"):
            change_ids = tuple(change.change_id for change in getattr(self, field_name))
            if change_ids != tuple(sorted(set(change_ids))):
                raise ObserverRepairProposalError(
                    f"{field_name} must have unique change IDs in sorted order"
                )
        if (
            self.disposition == REPAIR_DISPOSITION_REPAIRABLE
            and self.missing_schema_keys
        ):
            raise ObserverRepairProposalError(
                "repairable proposals cannot have missing_schema_keys"
            )
        if (
            self.disposition == REPAIR_DISPOSITION_REQUIRES_SCHEMA_EXTENSION
            and not self.missing_schema_keys
        ):
            raise ObserverRepairProposalError(
                "requires_schema_extension proposals require missing_schema_keys"
            )
        if self.disposition != REPAIR_DISPOSITION_REPAIRABLE and self.proposed_changes:
            raise ObserverRepairProposalError(
                f"{self.disposition} proposals cannot contain executable changes"
            )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.repair_proposal_id != expected_id:
            raise ObserverRepairProposalError(
                "repair_proposal_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "affected_cell_ids": list(self.affected_cell_ids),
            "affected_row_ids": list(self.affected_row_ids),
            "contradiction_artifact_id": self.contradiction_artifact_id,
            "disposition": self.disposition,
            "evidence_ids": list(self.evidence_ids),
            "missing_schema_keys": list(self.missing_schema_keys),
            "proposed_changes": [
                change.canonical_payload() for change in self.proposed_changes
            ],
            "rationale_codes": list(self.rationale_codes),
            "repair_constraint_id": self.repair_constraint_id,
            "requested_changes": [
                change.canonical_payload() for change in self.requested_changes
            ],
            "required_context_keys": list(self.required_context_keys),
            "source_policy_artifact_id": self.source_policy_artifact_id,
            "transition_verification_id": self.transition_verification_id,
            "version": self.version,
        }
        if include_id:
            payload["repair_proposal_id"] = self.repair_proposal_id
        return payload

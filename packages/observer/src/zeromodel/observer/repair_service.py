"""Application service for bounded Observer repair proposals."""

from __future__ import annotations

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.repair import (
    REPAIR_DISPOSITION_INSUFFICIENT_EVIDENCE,
    REPAIR_DISPOSITION_REPAIRABLE,
    REPAIR_DISPOSITION_REQUIRES_SCHEMA_EXTENSION,
    REPAIR_DISPOSITION_UNSUPPORTED,
    ObserverProposedChangeDTO,
    ObserverRepairConstraintDTO,
    ObserverRepairProposalDTO,
    ObserverRepairProposalError,
)
from zeromodel.observer.transition_service import (
    OBSERVER_VERIFICATION_CONTRADICTED,
    ObserverTransitionVerificationDTO,
)


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverRepairProposalError(f"{field_name} must be unique and sorted")


def _ensure_namespaced(keys: tuple[str, ...], field_name: str) -> None:
    for key in keys:
        if not key.startswith(("visible.", "history.", "hidden.")):
            raise ObserverRepairProposalError(
                f"{field_name} contains non-namespaced context key: {key!r}"
            )


def _row_for_cell(cell_id: str, available_policy_row_ids: tuple[str, ...]) -> str:
    matches = tuple(
        row_id
        for row_id in available_policy_row_ids
        if cell_id == row_id or cell_id.startswith(f"{row_id}/")
    )
    if not matches:
        raise ObserverRepairProposalError(
            f"policy cell {cell_id!r} does not identify an available policy row"
        )
    return max(matches, key=len)


def _change_row_ids(
    changes: tuple[ObserverProposedChangeDTO, ...],
    available_policy_row_ids: tuple[str, ...],
) -> tuple[str, ...]:
    rows: set[str] = set()
    for change in changes:
        if change.target_kind == "policy_row":
            rows.add(change.target_id)
        elif change.target_kind == "context_precondition":
            rows.add(change.target_id)
        elif change.target_kind in {"policy_cell", "transition_prediction"}:
            rows.add(_row_for_cell(change.target_id, available_policy_row_ids))
    return tuple(sorted(rows))


def _change_cell_ids(changes: tuple[ObserverProposedChangeDTO, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                change.target_id
                for change in changes
                if change.target_kind in {"policy_cell", "transition_prediction"}
            }
        )
    )


def _required_context_keys(
    changes: tuple[ObserverProposedChangeDTO, ...],
) -> tuple[str, ...]:
    return tuple(sorted({key for change in changes for key in change.condition_keys}))


def _validate_preconditions(verification: ObserverTransitionVerificationDTO) -> None:
    if verification.verification_status != OBSERVER_VERIFICATION_CONTRADICTED:
        raise ObserverRepairProposalError(
            "repair proposals require a contradicted transition verification"
        )
    if verification.contradiction_artifact is None:
        raise ObserverRepairProposalError(
            "contradicted transition verification is missing a contradiction artifact"
        )
    contradiction = verification.contradiction_artifact
    if (
        contradiction.transition_record_id
        != verification.transition_record.transition_record_id
    ):
        raise ObserverRepairProposalError(
            "contradiction artifact does not reference the verification transition record"
        )
    if (
        contradiction.comparison_result_id
        != verification.comparison_result.comparison_result_id
    ):
        raise ObserverRepairProposalError(
            "contradiction artifact does not reference the verification comparison result"
        )
    if (
        contradiction.source_policy_artifact_id
        != verification.transition_record.policy_artifact_id
    ):
        raise ObserverRepairProposalError(
            "contradiction source policy does not match transition record policy"
        )


def _validate_change_scope(
    *,
    changes: tuple[ObserverProposedChangeDTO, ...],
    constraint: ObserverRepairConstraintDTO,
    available_policy_row_ids: tuple[str, ...],
    available_policy_cell_ids: tuple[str, ...],
    affected_row_ids: tuple[str, ...],
    affected_cell_ids: tuple[str, ...],
) -> None:
    allowed_rows = set(constraint.allowed_row_ids)
    allowed_cells = set(constraint.allowed_cell_ids)
    available_rows = set(available_policy_row_ids)
    available_cells = set(available_policy_cell_ids)
    forbidden_rows = set(constraint.forbidden_row_ids)

    unknown_rows = set(affected_row_ids) - available_rows
    if unknown_rows:
        raise ObserverRepairProposalError(
            f"proposed target rows are not available: {sorted(unknown_rows)}"
        )
    unknown_cells = set(affected_cell_ids) - available_cells
    if unknown_cells:
        raise ObserverRepairProposalError(
            f"proposed target cells are not available: {sorted(unknown_cells)}"
        )
    outside_rows = set(affected_row_ids) - allowed_rows
    if outside_rows:
        raise ObserverRepairProposalError(
            f"proposed target rows are outside allowed_row_ids: {sorted(outside_rows)}"
        )
    outside_cells = set(affected_cell_ids) - allowed_cells
    if outside_cells:
        raise ObserverRepairProposalError(
            f"proposed target cells are outside allowed_cell_ids: {sorted(outside_cells)}"
        )
    forbidden_hits = set(affected_row_ids) & forbidden_rows
    if forbidden_hits:
        raise ObserverRepairProposalError(
            f"proposed target rows are forbidden: {sorted(forbidden_hits)}"
        )
    if len(affected_row_ids) > constraint.max_changed_rows:
        raise ObserverRepairProposalError(
            "proposed change exceeds max_changed_rows "
            f"({len(affected_row_ids)} > {constraint.max_changed_rows})"
        )
    if len(affected_cell_ids) > constraint.max_changed_cells:
        raise ObserverRepairProposalError(
            "proposed change exceeds max_changed_cells "
            f"({len(affected_cell_ids)} > {constraint.max_changed_cells})"
        )
    for change in changes:
        if (
            change.field_name == "action_value"
            and not constraint.allow_action_value_change
        ):
            raise ObserverRepairProposalError(
                "action-value changes are not permitted by the repair constraint"
            )
        if (
            change.target_kind == "transition_prediction"
            or change.field_name == "transition_prediction"
        ) and not constraint.allow_transition_prediction_change:
            raise ObserverRepairProposalError(
                "transition-prediction changes are not permitted by the repair constraint"
            )
        if (
            change.target_kind == "context_precondition"
            or change.operation == "add_precondition"
            or change.condition_keys
        ) and not constraint.allow_new_context_precondition:
            raise ObserverRepairProposalError(
                "new context preconditions are not permitted by the repair constraint"
            )


def _derive_disposition(
    *,
    requested_changes: tuple[ObserverProposedChangeDTO, ...],
    missing_schema_keys: tuple[str, ...],
    rationale_codes: tuple[str, ...],
) -> str:
    if missing_schema_keys:
        return REPAIR_DISPOSITION_REQUIRES_SCHEMA_EXTENSION
    if "unsupported_repair_type" in rationale_codes:
        return REPAIR_DISPOSITION_UNSUPPORTED
    if not requested_changes:
        return REPAIR_DISPOSITION_INSUFFICIENT_EVIDENCE
    return REPAIR_DISPOSITION_REPAIRABLE


def propose_observer_repair(
    *,
    verification: ObserverTransitionVerificationDTO,
    constraint: ObserverRepairConstraintDTO,
    available_policy_row_ids: tuple[str, ...],
    available_policy_cell_ids: tuple[str, ...],
    represented_context_keys: tuple[str, ...],
    requested_changes: tuple[ObserverProposedChangeDTO, ...] = (),
    missing_schema_keys: tuple[str, ...] = (),
    rationale_codes: tuple[str, ...] = (),
    evidence_ids: tuple[str, ...] = (),
) -> ObserverRepairProposalDTO:
    """Validate and canonicalize a bounded repair proposal.

    `repairable` means the proposal is well-formed and eligible for later
    candidate generation. It does not mean the policy has been repaired.
    """

    _ensure_sorted_unique(available_policy_row_ids, "available_policy_row_ids")
    _ensure_sorted_unique(available_policy_cell_ids, "available_policy_cell_ids")
    _ensure_sorted_unique(represented_context_keys, "represented_context_keys")
    _ensure_sorted_unique(missing_schema_keys, "missing_schema_keys")
    _ensure_sorted_unique(rationale_codes, "rationale_codes")
    _ensure_sorted_unique(evidence_ids, "evidence_ids")
    _ensure_namespaced(represented_context_keys, "represented_context_keys")
    _ensure_namespaced(missing_schema_keys, "missing_schema_keys")
    _validate_preconditions(verification)
    requested_changes = tuple(
        sorted(requested_changes, key=lambda item: item.change_id)
    )

    contradiction = verification.contradiction_artifact
    assert contradiction is not None
    source_policy_artifact_id = contradiction.source_policy_artifact_id
    if source_policy_artifact_id != verification.transition_record.policy_artifact_id:
        raise ObserverRepairProposalError(
            "source policy identity does not match verification transition record"
        )

    required_context_keys = _required_context_keys(requested_changes)
    _ensure_namespaced(required_context_keys, "required_context_keys")
    missing_from_schema = tuple(
        sorted(set(required_context_keys) - set(represented_context_keys))
    )
    effective_missing_schema_keys = tuple(
        sorted(set(missing_schema_keys) | set(missing_from_schema))
    )
    if effective_missing_schema_keys and not constraint.allow_schema_extension_request:
        raise ObserverRepairProposalError(
            "repair requires schema extension but constraint does not permit schema-extension requests"
        )

    affected_row_ids = _change_row_ids(requested_changes, available_policy_row_ids)
    affected_cell_ids = _change_cell_ids(requested_changes)
    if (
        requested_changes
        and contradiction.affected_policy_row_id not in affected_row_ids
    ):
        raise ObserverRepairProposalError(
            "requested changes must stay within the contradiction's affected policy row"
        )
    _validate_change_scope(
        changes=requested_changes,
        constraint=constraint,
        available_policy_row_ids=available_policy_row_ids,
        available_policy_cell_ids=available_policy_cell_ids,
        affected_row_ids=affected_row_ids,
        affected_cell_ids=affected_cell_ids,
    )

    disposition = _derive_disposition(
        requested_changes=requested_changes,
        missing_schema_keys=effective_missing_schema_keys,
        rationale_codes=rationale_codes,
    )
    executable_changes = (
        requested_changes if disposition == REPAIR_DISPOSITION_REPAIRABLE else ()
    )

    mandatory_evidence_ids = (
        verification.verification_id,
        verification.comparison_result.comparison_result_id,
        verification.transition_record.transition_record_id,
        contradiction.contradiction_artifact_id,
    )
    proposal_evidence_ids = tuple(
        sorted(set(evidence_ids) | set(mandatory_evidence_ids))
    )
    payload = {
        "affected_cell_ids": list(affected_cell_ids),
        "affected_row_ids": list(affected_row_ids),
        "contradiction_artifact_id": contradiction.contradiction_artifact_id,
        "disposition": disposition,
        "evidence_ids": list(proposal_evidence_ids),
        "missing_schema_keys": list(effective_missing_schema_keys),
        "proposed_changes": [
            change.canonical_payload() for change in executable_changes
        ],
        "rationale_codes": list(rationale_codes),
        "repair_constraint_id": constraint.repair_constraint_id,
        "requested_changes": [
            change.canonical_payload() for change in requested_changes
        ],
        "required_context_keys": list(required_context_keys),
        "source_policy_artifact_id": source_policy_artifact_id,
        "transition_verification_id": verification.verification_id,
        "version": "observer-repair-proposal/2",
    }
    return ObserverRepairProposalDTO(
        repair_proposal_id=canonical_id(payload),
        source_policy_artifact_id=source_policy_artifact_id,
        transition_verification_id=verification.verification_id,
        contradiction_artifact_id=contradiction.contradiction_artifact_id,
        repair_constraint_id=constraint.repair_constraint_id,
        disposition=disposition,
        affected_row_ids=affected_row_ids,
        affected_cell_ids=affected_cell_ids,
        required_context_keys=required_context_keys,
        missing_schema_keys=effective_missing_schema_keys,
        requested_changes=requested_changes,
        proposed_changes=executable_changes,
        rationale_codes=rationale_codes,
        evidence_ids=proposal_evidence_ids,
    )

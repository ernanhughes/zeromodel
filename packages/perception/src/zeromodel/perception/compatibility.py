"""Deployment and rollback compatibility contracts for Stage P16F.

Historical activation is necessary but not sufficient for rollback. A candidate must also
match the current action, representation, temporal, inference, and deployment contracts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .lifecycle import (
    ActiveModelPointerDTO,
    ModelLifecycleTransitionDTO,
    PerceptionModelLifecycleError,
    PerceptionModelLifecycleStore,
    rollback_active_model,
)
from .promotion import PromotedPerceptionModelDTO

MODEL_COMPATIBILITY_VERSION: Final = "perception-model-compatibility/1"
ROLLBACK_COMPATIBILITY_VERSION: Final = "perception-rollback-compatibility/1"
MODEL_COMPATIBILITY_SEMANTICS: Final = (
    "exact_action_representation_temporal_inference_and_deployment_contract"
)
ROLLBACK_COMPATIBILITY_STATUSES: Final = {"compatible", "incompatible"}


class PerceptionModelCompatibilityError(ValueError):
    """Raised when deployment or rollback compatibility cannot be established."""


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
class ModelCompatibilityContractDTO:
    contract_id: str
    promoted_model_id: str
    model_kind: str
    action_schema_id: str
    source_encoder_spec_id: str
    field_schema_id: str
    temporal_window_spec_id: str | None
    inference_semantics_version: str
    deployment_slot: str
    semantics: str = MODEL_COMPATIBILITY_SEMANTICS
    version: str = MODEL_COMPATIBILITY_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.contract_id,
                self.promoted_model_id,
                self.action_schema_id,
                self.source_encoder_spec_id,
                self.field_schema_id,
                self.inference_semantics_version,
                self.deployment_slot,
            )
        ):
            raise PerceptionModelCompatibilityError("compatibility identities must be non-empty")
        if self.model_kind not in {"single_frame", "temporal"}:
            raise PerceptionModelCompatibilityError("unsupported compatibility model kind")
        if self.model_kind == "temporal" and not self.temporal_window_spec_id:
            raise PerceptionModelCompatibilityError("temporal compatibility requires window identity")
        if self.model_kind == "single_frame" and self.temporal_window_spec_id is not None:
            raise PerceptionModelCompatibilityError(
                "single-frame compatibility cannot carry temporal window identity"
            )
        if self.semantics != MODEL_COMPATIBILITY_SEMANTICS:
            raise PerceptionModelCompatibilityError("unsupported compatibility semantics")
        if self.version != MODEL_COMPATIBILITY_VERSION:
            raise PerceptionModelCompatibilityError("unsupported compatibility version")

    def runtime_signature(self) -> tuple[str, ...]:
        return (
            self.model_kind,
            self.action_schema_id,
            self.source_encoder_spec_id,
            self.field_schema_id,
            self.temporal_window_spec_id or "",
            self.inference_semantics_version,
            self.deployment_slot,
        )


@dataclass(frozen=True)
class RollbackCompatibilityAssessmentDTO:
    assessment_id: str
    current_contract_id: str
    target_contract_id: str
    current_promoted_model_id: str
    target_promoted_model_id: str
    status: str
    mismatched_fields: tuple[str, ...]
    version: str = ROLLBACK_COMPATIBILITY_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.assessment_id,
                self.current_contract_id,
                self.target_contract_id,
                self.current_promoted_model_id,
                self.target_promoted_model_id,
            )
        ):
            raise PerceptionModelCompatibilityError("rollback assessment identities required")
        if self.status not in ROLLBACK_COMPATIBILITY_STATUSES:
            raise PerceptionModelCompatibilityError("unsupported rollback compatibility status")
        if self.mismatched_fields != tuple(sorted(set(self.mismatched_fields))):
            raise PerceptionModelCompatibilityError("mismatched fields must be unique and sorted")
        if (self.status == "compatible") != (not self.mismatched_fields):
            raise PerceptionModelCompatibilityError("assessment status must match mismatches")


def build_model_compatibility_contract(
    promoted: PromotedPerceptionModelDTO,
    *,
    action_schema_id: str,
    source_encoder_spec_id: str,
    field_schema_id: str,
    inference_semantics_version: str,
    deployment_slot: str,
) -> ModelCompatibilityContractDTO:
    payload: Mapping[str, object] = {
        "action_schema_id": action_schema_id,
        "deployment_slot": deployment_slot,
        "field_schema_id": field_schema_id,
        "inference_semantics_version": inference_semantics_version,
        "model_kind": promoted.model_kind,
        "promoted_model_id": promoted.promoted_model_id,
        "semantics": MODEL_COMPATIBILITY_SEMANTICS,
        "source_encoder_spec_id": source_encoder_spec_id,
        "temporal_window_spec_id": promoted.temporal_window_spec_id,
        "version": MODEL_COMPATIBILITY_VERSION,
    }
    return ModelCompatibilityContractDTO(contract_id=_digest(payload), **payload)  # type: ignore[arg-type]


def assess_rollback_compatibility(
    current: ModelCompatibilityContractDTO,
    target: ModelCompatibilityContractDTO,
) -> RollbackCompatibilityAssessmentDTO:
    fields = (
        "model_kind",
        "action_schema_id",
        "source_encoder_spec_id",
        "field_schema_id",
        "temporal_window_spec_id",
        "inference_semantics_version",
        "deployment_slot",
    )
    mismatches = tuple(
        sorted(name for name in fields if getattr(current, name) != getattr(target, name))
    )
    status = "compatible" if not mismatches else "incompatible"
    payload: Mapping[str, object] = {
        "current_contract_id": current.contract_id,
        "current_promoted_model_id": current.promoted_model_id,
        "mismatched_fields": list(mismatches),
        "status": status,
        "target_contract_id": target.contract_id,
        "target_promoted_model_id": target.promoted_model_id,
        "version": ROLLBACK_COMPATIBILITY_VERSION,
    }
    return RollbackCompatibilityAssessmentDTO(
        assessment_id=_digest(payload),
        current_contract_id=current.contract_id,
        target_contract_id=target.contract_id,
        current_promoted_model_id=current.promoted_model_id,
        target_promoted_model_id=target.promoted_model_id,
        status=status,
        mismatched_fields=mismatches,
    )


def rollback_compatible_model(
    store: PerceptionModelLifecycleStore,
    target_promoted_model_id: str,
    *,
    current_contract: ModelCompatibilityContractDTO,
    target_contract: ModelCompatibilityContractDTO,
    actor: str,
    reason: str,
) -> tuple[
    RollbackCompatibilityAssessmentDTO,
    ModelLifecycleTransitionDTO,
    ActiveModelPointerDTO,
]:
    """Rollback only when lifecycle history and exact runtime compatibility both hold."""

    active_id = store.get_active_pointer().active_promoted_model_id
    if active_id != current_contract.promoted_model_id:
        raise PerceptionModelCompatibilityError(
            "current compatibility contract does not describe the active model"
        )
    if target_promoted_model_id != target_contract.promoted_model_id:
        raise PerceptionModelCompatibilityError(
            "target compatibility contract does not describe rollback target"
        )
    assessment = assess_rollback_compatibility(current_contract, target_contract)
    if assessment.status != "compatible":
        raise PerceptionModelCompatibilityError(
            f"rollback target is incompatible: {list(assessment.mismatched_fields)}"
        )
    try:
        transition, pointer = rollback_active_model(
            store,
            target_promoted_model_id,
            actor=actor,
            reason=reason,
        )
    except PerceptionModelLifecycleError:
        raise
    return assessment, transition, pointer

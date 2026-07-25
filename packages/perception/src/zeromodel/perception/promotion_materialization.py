"""Reversible, inactive promotion materialization for Stage P18F.

P18F converts only fully reviewed P18E approvals into versioned annotation or
relation additions plus transition expectations. The resulting change set is
content-addressed, carries exact inverse operations, and remains inactive until
a later admission stage authorizes execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Final, Mapping

from .candidate_promotion import (
    CANDIDATE_PROMOTION_MATERIALIZATION_STATUS,
    CandidatePromotionDecisionDTO,
    CandidatePromotionProposalDTO,
    CandidatePromotionProposalSetDTO,
    CandidatePromotionReviewDTO,
)
from .expectations import (
    RELATION_ANNOTATION_VERSION,
    PerceptionRegionAnnotationDTO,
    RelationAnnotationDTO,
)
from .fields import VPMFieldSchemaDTO
from .transition_conformance import TransitionExpectationDTO

PROMOTION_MATERIALIZATION_DIRECTIVE_VERSION: Final = (
    "perception-promotion-materialization-directive/1"
)
PROMOTION_MATERIALIZATION_BASELINE_VERSION: Final = (
    "perception-promotion-materialization-baseline/1"
)
PROMOTION_MATERIALIZATION_OPERATION_VERSION: Final = (
    "perception-promotion-materialization-operation/1"
)
MATERIALIZED_PROMOTION_CHANGE_VERSION: Final = (
    "perception-materialized-promotion-change/1"
)
PROMOTION_MATERIALIZATION_CHANGE_SET_VERSION: Final = (
    "perception-promotion-materialization-change-set/1"
)
PROMOTION_MATERIALIZATION_SEMANTICS: Final = (
    "reversible_inactive_materialization_of_fully_reviewed_p18e_approvals"
)
PROMOTION_MATERIALIZATION_TARGET_KINDS: Final = {
    "region_annotation",
    "relation_annotation",
}
PROMOTION_MATERIALIZATION_OBJECT_KINDS: Final = {
    "annotation",
    "relation",
    "transition_expectation",
}
PROMOTION_MATERIALIZATION_OPERATION_ACTIONS: Final = {
    "add_annotation",
    "remove_annotation",
    "add_relation",
    "remove_relation",
    "add_transition_expectation",
    "remove_transition_expectation",
}
PROMOTION_MATERIALIZATION_DIRECTIONS: Final = {"forward", "inverse"}
PROMOTION_MATERIALIZATION_CHANGE_SET_STATUSES: Final = {
    "staged_inactive",
    "no_approved_changes",
}
PROMOTION_MATERIALIZATION_ACTIVATION_STATUS: Final = "not_admitted"
PROMOTION_MATERIALIZATION_ITEM_STATUS: Final = "staged_inactive"
_RESERVED_ANNOTATION_PROPERTY_KEYS: Final = {
    "semantic_type",
    "promotion_proposal_id",
    "promotion_decision_id",
}


class PerceptionPromotionMaterializationError(ValueError):
    """Raised when P18F materialization contracts are invalid."""


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
        raise PerceptionPromotionMaterializationError(f"{name} must be non-empty")
    if values != tuple(sorted(set(values))):
        raise PerceptionPromotionMaterializationError(
            f"{name} must be unique and sorted"
        )


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PerceptionPromotionMaterializationError(
            f"{name} must be a positive integer"
        )


def _object_digest(kind: str, value: object) -> str:
    return _digest({"object_kind": kind, "payload": asdict(value)})  # type: ignore[arg-type]


@dataclass(frozen=True)
class PromotionMaterializationDirectiveDTO:
    """Explicit ontology choice for one approved P18E proposal."""

    directive_id: str
    proposal_id: str
    target_kind: str
    relation_member_annotation_ids: tuple[str, ...] = ()
    annotation_properties: tuple[tuple[str, str], ...] = ()
    version: str = PROMOTION_MATERIALIZATION_DIRECTIVE_VERSION

    def __post_init__(self) -> None:
        if not self.directive_id or not self.proposal_id:
            raise PerceptionPromotionMaterializationError(
                "materialization directive identities must be non-empty"
            )
        if self.target_kind not in PROMOTION_MATERIALIZATION_TARGET_KINDS:
            raise PerceptionPromotionMaterializationError(
                f"unsupported materialization target_kind: {self.target_kind}"
            )
        _ordered_unique(
            "directive relation_member_annotation_ids",
            self.relation_member_annotation_ids,
        )
        if self.annotation_properties != tuple(sorted(set(self.annotation_properties))):
            raise PerceptionPromotionMaterializationError(
                "directive annotation_properties must be unique and sorted"
            )
        if any(not key or not value for key, value in self.annotation_properties):
            raise PerceptionPromotionMaterializationError(
                "directive annotation properties require non-empty keys and values"
            )
        property_keys = {key for key, _ in self.annotation_properties}
        reserved = property_keys & _RESERVED_ANNOTATION_PROPERTY_KEYS
        if reserved:
            raise PerceptionPromotionMaterializationError(
                f"directive annotation properties use reserved keys: {sorted(reserved)}"
            )
        if self.target_kind == "region_annotation":
            if self.relation_member_annotation_ids:
                raise PerceptionPromotionMaterializationError(
                    "region directives cannot declare relation members"
                )
        else:
            if len(self.relation_member_annotation_ids) < 2:
                raise PerceptionPromotionMaterializationError(
                    "relation directives require at least two annotation members"
                )
            if self.annotation_properties:
                raise PerceptionPromotionMaterializationError(
                    "relation directives cannot carry annotation properties"
                )
        if self.version != PROMOTION_MATERIALIZATION_DIRECTIVE_VERSION:
            raise PerceptionPromotionMaterializationError(
                "unsupported materialization directive version"
            )
        if self.directive_id != _digest(_payload(self, "directive_id")):
            raise PerceptionPromotionMaterializationError(
                "materialization directive identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        proposal: CandidatePromotionProposalDTO,
        *,
        target_kind: str,
        relation_member_annotation_ids: tuple[str, ...] = (),
        annotation_properties: tuple[tuple[str, str], ...] = (),
    ) -> "PromotionMaterializationDirectiveDTO":
        values: dict[str, object] = {
            "proposal_id": proposal.proposal_id,
            "target_kind": target_kind,
            "relation_member_annotation_ids": tuple(
                sorted(set(relation_member_annotation_ids))
            ),
            "annotation_properties": tuple(sorted(set(annotation_properties))),
            "version": PROMOTION_MATERIALIZATION_DIRECTIVE_VERSION,
        }
        return cls(directive_id=_digest(values), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PromotionMaterializationBaselineDTO:
    """Identity snapshot used to prevent non-reversible additive collisions."""

    baseline_id: str
    baseline_version_id: str
    field_schema_id: str
    existing_annotation_ids: tuple[str, ...] = ()
    existing_relation_ids: tuple[str, ...] = ()
    existing_transition_expectation_ids: tuple[str, ...] = ()
    version: str = PROMOTION_MATERIALIZATION_BASELINE_VERSION

    def __post_init__(self) -> None:
        if not self.baseline_id or not self.baseline_version_id or not self.field_schema_id:
            raise PerceptionPromotionMaterializationError(
                "materialization baseline identities must be non-empty"
            )
        for name in (
            "existing_annotation_ids",
            "existing_relation_ids",
            "existing_transition_expectation_ids",
        ):
            _ordered_unique(name, getattr(self, name))
        if self.version != PROMOTION_MATERIALIZATION_BASELINE_VERSION:
            raise PerceptionPromotionMaterializationError(
                "unsupported materialization baseline version"
            )
        if self.baseline_id != _digest(_payload(self, "baseline_id")):
            raise PerceptionPromotionMaterializationError(
                "materialization baseline identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        *,
        baseline_version_id: str,
        field_schema_id: str,
        existing_annotation_ids: tuple[str, ...] = (),
        existing_relation_ids: tuple[str, ...] = (),
        existing_transition_expectation_ids: tuple[str, ...] = (),
    ) -> "PromotionMaterializationBaselineDTO":
        values: dict[str, object] = {
            "baseline_version_id": baseline_version_id,
            "field_schema_id": field_schema_id,
            "existing_annotation_ids": tuple(sorted(set(existing_annotation_ids))),
            "existing_relation_ids": tuple(sorted(set(existing_relation_ids))),
            "existing_transition_expectation_ids": tuple(
                sorted(set(existing_transition_expectation_ids))
            ),
            "version": PROMOTION_MATERIALIZATION_BASELINE_VERSION,
        }
        return cls(baseline_id=_digest(values), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PromotionMaterializationOperationDTO:
    operation_id: str
    pair_id: str
    direction: str
    action: str
    object_kind: str
    object_id: str
    payload_digest: str
    proposal_id: str
    decision_id: str
    sequence: int
    version: str = PROMOTION_MATERIALIZATION_OPERATION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.operation_id,
                self.pair_id,
                self.object_id,
                self.payload_digest,
                self.proposal_id,
                self.decision_id,
            )
        ):
            raise PerceptionPromotionMaterializationError(
                "materialization operation identities must be non-empty"
            )
        if self.direction not in PROMOTION_MATERIALIZATION_DIRECTIONS:
            raise PerceptionPromotionMaterializationError(
                f"unsupported operation direction: {self.direction}"
            )
        if self.action not in PROMOTION_MATERIALIZATION_OPERATION_ACTIONS:
            raise PerceptionPromotionMaterializationError(
                f"unsupported materialization operation: {self.action}"
            )
        if self.object_kind not in PROMOTION_MATERIALIZATION_OBJECT_KINDS:
            raise PerceptionPromotionMaterializationError(
                f"unsupported materialization object_kind: {self.object_kind}"
            )
        _positive_int("materialization operation sequence", self.sequence)
        expected_action = {
            ("forward", "annotation"): "add_annotation",
            ("inverse", "annotation"): "remove_annotation",
            ("forward", "relation"): "add_relation",
            ("inverse", "relation"): "remove_relation",
            ("forward", "transition_expectation"): "add_transition_expectation",
            ("inverse", "transition_expectation"): "remove_transition_expectation",
        }[(self.direction, self.object_kind)]
        if self.action != expected_action:
            raise PerceptionPromotionMaterializationError(
                "materialization action disagrees with direction and object kind"
            )
        expected_pair = _digest(
            {
                "object_kind": self.object_kind,
                "object_id": self.object_id,
                "payload_digest": self.payload_digest,
                "proposal_id": self.proposal_id,
                "decision_id": self.decision_id,
            }
        )
        if self.pair_id != expected_pair:
            raise PerceptionPromotionMaterializationError(
                "materialization operation pair identity disagrees with object payload"
            )
        if self.version != PROMOTION_MATERIALIZATION_OPERATION_VERSION:
            raise PerceptionPromotionMaterializationError(
                "unsupported materialization operation version"
            )
        if self.operation_id != _digest(_payload(self, "operation_id")):
            raise PerceptionPromotionMaterializationError(
                "materialization operation identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class MaterializedPromotionChangeDTO:
    change_id: str
    proposal_id: str
    decision_id: str
    directive_id: str
    target_kind: str
    semantic_name: str
    semantic_type: str
    semantic_role: str | None
    annotation: PerceptionRegionAnnotationDTO | None
    relation: RelationAnnotationDTO | None
    transition_expectation: TransitionExpectationDTO
    forward_operations: tuple[PromotionMaterializationOperationDTO, ...]
    inverse_operations: tuple[PromotionMaterializationOperationDTO, ...]
    materialization_status: str = PROMOTION_MATERIALIZATION_ITEM_STATUS
    version: str = MATERIALIZED_PROMOTION_CHANGE_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.change_id,
                self.proposal_id,
                self.decision_id,
                self.directive_id,
                self.semantic_name,
                self.semantic_type,
            )
        ):
            raise PerceptionPromotionMaterializationError(
                "materialized promotion change identities and semantics must be non-empty"
            )
        if self.target_kind not in PROMOTION_MATERIALIZATION_TARGET_KINDS:
            raise PerceptionPromotionMaterializationError(
                f"unsupported materialized target_kind: {self.target_kind}"
            )
        if self.target_kind == "region_annotation":
            if self.annotation is None or self.relation is not None:
                raise PerceptionPromotionMaterializationError(
                    "region changes require annotation and forbid relation"
                )
            target_kind = "annotation"
            target_id = self.annotation.annotation_id
            if self.transition_expectation.annotation_ids != (target_id,):
                raise PerceptionPromotionMaterializationError(
                    "region transition expectation must target materialized annotation"
                )
            if self.transition_expectation.relation_ids:
                raise PerceptionPromotionMaterializationError(
                    "region transition expectation cannot target relations"
                )
        else:
            if self.relation is None or self.annotation is not None:
                raise PerceptionPromotionMaterializationError(
                    "relation changes require relation and forbid annotation"
                )
            target_kind = "relation"
            target_id = self.relation.relation_id
            if self.transition_expectation.relation_ids != (target_id,):
                raise PerceptionPromotionMaterializationError(
                    "relation transition expectation must target materialized relation"
                )
            if self.transition_expectation.annotation_ids:
                raise PerceptionPromotionMaterializationError(
                    "relation transition expectation cannot target annotations"
                )
        if self.annotation is not None:
            expected_annotation_id = _digest(
                {
                    "field_schema_id": self.annotation.field_schema_id,
                    "field_ids": list(self.annotation.field_ids),
                    "label": self.annotation.label,
                    "properties": [list(item) for item in self.annotation.properties],
                    "provenance_ref": self.annotation.provenance_ref,
                    "role": self.annotation.role,
                    "version": self.annotation.version,
                }
            )
            if self.annotation.annotation_id != expected_annotation_id:
                raise PerceptionPromotionMaterializationError(
                    "materialized annotation identity disagrees with payload"
                )
            properties = dict(self.annotation.properties)
            if (
                self.annotation.label != self.semantic_name
                or self.annotation.role != self.semantic_role
                or properties.get("semantic_type") != self.semantic_type
                or self.annotation.provenance_ref != self.decision_id
            ):
                raise PerceptionPromotionMaterializationError(
                    "materialized annotation disagrees with approved semantics"
                )
        if self.relation is not None:
            expected_relation_id = _digest(
                {
                    "relation_type": self.relation.relation_type,
                    "member_annotation_ids": self.relation.member_annotation_ids,
                    "derived_field_ids": self.relation.derived_field_ids,
                    "value": self.relation.value,
                    "version": self.relation.version,
                }
            )
            if self.relation.relation_id != expected_relation_id:
                raise PerceptionPromotionMaterializationError(
                    "materialized relation identity disagrees with payload"
                )
            if (
                self.relation.relation_type != self.semantic_type
                or self.relation.value != self.semantic_name
            ):
                raise PerceptionPromotionMaterializationError(
                    "materialized relation disagrees with approved semantics"
                )
        if len(self.forward_operations) != 2 or len(self.inverse_operations) != 2:
            raise PerceptionPromotionMaterializationError(
                "each materialized change requires two forward and two inverse operations"
            )
        for operations, direction in (
            (self.forward_operations, "forward"),
            (self.inverse_operations, "inverse"),
        ):
            if any(item.direction != direction for item in operations):
                raise PerceptionPromotionMaterializationError(
                    "change operation direction disagrees with operation collection"
                )
            if any(
                item.proposal_id != self.proposal_id
                or item.decision_id != self.decision_id
                for item in operations
            ):
                raise PerceptionPromotionMaterializationError(
                    "change operations disagree with proposal or decision lineage"
                )
        forward_pairs = {item.pair_id: item for item in self.forward_operations}
        inverse_pairs = {item.pair_id: item for item in self.inverse_operations}
        if set(forward_pairs) != set(inverse_pairs):
            raise PerceptionPromotionMaterializationError(
                "forward and inverse operations do not form exact pairs"
            )
        object_pairs = {
            (item.object_kind, item.object_id, item.payload_digest)
            for item in self.forward_operations
        }
        expected_objects = {
            (
                target_kind,
                target_id,
                _object_digest(target_kind, self.annotation or self.relation),
            ),
            (
                "transition_expectation",
                self.transition_expectation.expectation_id,
                _object_digest("transition_expectation", self.transition_expectation),
            ),
        }
        if object_pairs != expected_objects:
            raise PerceptionPromotionMaterializationError(
                "materialization operations disagree with materialized objects"
            )
        if self.materialization_status != PROMOTION_MATERIALIZATION_ITEM_STATUS:
            raise PerceptionPromotionMaterializationError(
                "materialized promotion changes must remain staged_inactive"
            )
        if self.version != MATERIALIZED_PROMOTION_CHANGE_VERSION:
            raise PerceptionPromotionMaterializationError(
                "unsupported materialized promotion change version"
            )
        if self.change_id != _digest(_payload(self, "change_id")):
            raise PerceptionPromotionMaterializationError(
                "materialized promotion change identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class PromotionMaterializationChangeSetDTO:
    change_set_id: str
    status: str
    proposal_set_id: str
    review_id: str
    baseline_id: str
    baseline_version_id: str
    field_schema_id: str
    approved_proposal_ids: tuple[str, ...]
    decision_ids: tuple[str, ...]
    directive_ids: tuple[str, ...]
    change_ids: tuple[str, ...]
    changes: tuple[MaterializedPromotionChangeDTO, ...]
    forward_operation_ids: tuple[str, ...]
    inverse_operation_ids: tuple[str, ...]
    activation_status: str = PROMOTION_MATERIALIZATION_ACTIVATION_STATUS
    semantics: str = PROMOTION_MATERIALIZATION_SEMANTICS
    version: str = PROMOTION_MATERIALIZATION_CHANGE_SET_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.change_set_id,
                self.proposal_set_id,
                self.review_id,
                self.baseline_id,
                self.baseline_version_id,
                self.field_schema_id,
            )
        ):
            raise PerceptionPromotionMaterializationError(
                "materialization change-set identities must be non-empty"
            )
        if self.status not in PROMOTION_MATERIALIZATION_CHANGE_SET_STATUSES:
            raise PerceptionPromotionMaterializationError(
                f"unsupported materialization change-set status: {self.status}"
            )
        for name in (
            "approved_proposal_ids",
            "decision_ids",
            "directive_ids",
            "change_ids",
        ):
            _ordered_unique(name, getattr(self, name))
        actual_change_ids = tuple(sorted(item.change_id for item in self.changes))
        if actual_change_ids != self.change_ids:
            raise PerceptionPromotionMaterializationError(
                "materialization change identities disagree with changes"
            )
        if tuple(sorted(item.proposal_id for item in self.changes)) != self.approved_proposal_ids:
            raise PerceptionPromotionMaterializationError(
                "approved proposal identities disagree with materialized changes"
            )
        if tuple(sorted(item.decision_id for item in self.changes)) != self.decision_ids:
            raise PerceptionPromotionMaterializationError(
                "decision identities disagree with materialized changes"
            )
        if tuple(sorted(item.directive_id for item in self.changes)) != self.directive_ids:
            raise PerceptionPromotionMaterializationError(
                "directive identities disagree with materialized changes"
            )
        if not (
            len(self.approved_proposal_ids)
            == len(self.decision_ids)
            == len(self.directive_ids)
            == len(self.change_ids)
            == len(self.changes)
        ):
            raise PerceptionPromotionMaterializationError(
                "materialization change-set lineage counts disagree"
            )
        if self.status == "staged_inactive" and not self.changes:
            raise PerceptionPromotionMaterializationError(
                "staged_inactive change sets require materialized changes"
            )
        if self.status == "no_approved_changes" and self.changes:
            raise PerceptionPromotionMaterializationError(
                "no_approved_changes change sets cannot contain changes"
            )
        forward = tuple(
            sorted(
                (
                    operation
                    for change in self.changes
                    for operation in change.forward_operations
                ),
                key=lambda item: item.sequence,
            )
        )
        inverse = tuple(
            sorted(
                (
                    operation
                    for change in self.changes
                    for operation in change.inverse_operations
                ),
                key=lambda item: item.sequence,
            )
        )
        if len(self.forward_operation_ids) != len(set(self.forward_operation_ids)):
            raise PerceptionPromotionMaterializationError(
                "forward operation identities must be unique"
            )
        if len(self.inverse_operation_ids) != len(set(self.inverse_operation_ids)):
            raise PerceptionPromotionMaterializationError(
                "inverse operation identities must be unique"
            )
        if tuple(item.operation_id for item in forward) != self.forward_operation_ids:
            raise PerceptionPromotionMaterializationError(
                "forward operation identities disagree with execution order"
            )
        if tuple(item.operation_id for item in inverse) != self.inverse_operation_ids:
            raise PerceptionPromotionMaterializationError(
                "inverse operation identities disagree with execution order"
            )
        if tuple(item.sequence for item in forward) != tuple(range(1, len(forward) + 1)):
            raise PerceptionPromotionMaterializationError(
                "forward operation sequence must be contiguous"
            )
        if tuple(item.sequence for item in inverse) != tuple(range(1, len(inverse) + 1)):
            raise PerceptionPromotionMaterializationError(
                "inverse operation sequence must be contiguous"
            )
        if {item.pair_id for item in forward} != {item.pair_id for item in inverse}:
            raise PerceptionPromotionMaterializationError(
                "change-set forward and inverse operations do not pair exactly"
            )
        if self.activation_status != PROMOTION_MATERIALIZATION_ACTIVATION_STATUS:
            raise PerceptionPromotionMaterializationError(
                "materialization change sets must remain not_admitted"
            )
        if self.semantics != PROMOTION_MATERIALIZATION_SEMANTICS:
            raise PerceptionPromotionMaterializationError(
                "unsupported promotion materialization semantics"
            )
        if self.version != PROMOTION_MATERIALIZATION_CHANGE_SET_VERSION:
            raise PerceptionPromotionMaterializationError(
                "unsupported promotion materialization change-set version"
            )
        if self.change_set_id != _digest(_payload(self, "change_set_id")):
            raise PerceptionPromotionMaterializationError(
                "promotion materialization change-set identity disagrees with canonical payload"
            )

    def operations(
        self,
        direction: str,
    ) -> tuple[PromotionMaterializationOperationDTO, ...]:
        if direction not in PROMOTION_MATERIALIZATION_DIRECTIONS:
            raise PerceptionPromotionMaterializationError(
                f"unsupported operation direction: {direction}"
            )
        operations = tuple(
            operation
            for change in self.changes
            for operation in (
                change.forward_operations
                if direction == "forward"
                else change.inverse_operations
            )
        )
        return tuple(sorted(operations, key=lambda item: item.sequence))


def _operation(
    *,
    direction: str,
    object_kind: str,
    object_id: str,
    payload_digest: str,
    proposal_id: str,
    decision_id: str,
    sequence: int,
) -> PromotionMaterializationOperationDTO:
    action = {
        ("forward", "annotation"): "add_annotation",
        ("inverse", "annotation"): "remove_annotation",
        ("forward", "relation"): "add_relation",
        ("inverse", "relation"): "remove_relation",
        ("forward", "transition_expectation"): "add_transition_expectation",
        ("inverse", "transition_expectation"): "remove_transition_expectation",
    }[(direction, object_kind)]
    pair_id = _digest(
        {
            "object_kind": object_kind,
            "object_id": object_id,
            "payload_digest": payload_digest,
            "proposal_id": proposal_id,
            "decision_id": decision_id,
        }
    )
    values: dict[str, object] = {
        "pair_id": pair_id,
        "direction": direction,
        "action": action,
        "object_kind": object_kind,
        "object_id": object_id,
        "payload_digest": payload_digest,
        "proposal_id": proposal_id,
        "decision_id": decision_id,
        "sequence": sequence,
        "version": PROMOTION_MATERIALIZATION_OPERATION_VERSION,
    }
    return PromotionMaterializationOperationDTO(
        operation_id=_digest(values),
        **values,  # type: ignore[arg-type]
    )


def _relation(
    *,
    proposal: CandidatePromotionProposalDTO,
    decision: CandidatePromotionDecisionDTO,
    directive: PromotionMaterializationDirectiveDTO,
) -> RelationAnnotationDTO:
    payload: dict[str, object] = {
        "relation_type": decision.semantic_type,
        "member_annotation_ids": directive.relation_member_annotation_ids,
        "derived_field_ids": proposal.field_ids,
        "value": decision.semantic_name,
        "version": RELATION_ANNOTATION_VERSION,
    }
    return RelationAnnotationDTO(
        relation_id=_digest(payload),
        relation_type=decision.semantic_type or "",
        member_annotation_ids=directive.relation_member_annotation_ids,
        derived_field_ids=proposal.field_ids,
        value=decision.semantic_name,
    )


def _objects_for_approval(
    *,
    proposal: CandidatePromotionProposalDTO,
    decision: CandidatePromotionDecisionDTO,
    directive: PromotionMaterializationDirectiveDTO,
    field_schema: VPMFieldSchemaDTO,
) -> tuple[
    PerceptionRegionAnnotationDTO | None,
    RelationAnnotationDTO | None,
    TransitionExpectationDTO,
]:
    if decision.decision != "approved" or not decision.semantic_name or not decision.semantic_type:
        raise PerceptionPromotionMaterializationError(
            "materialization requires an approved decision with semantic identity"
        )
    if directive.target_kind == "region_annotation":
        properties = tuple(
            sorted(
                set(directive.annotation_properties)
                | {
                    ("semantic_type", decision.semantic_type),
                    ("promotion_proposal_id", proposal.proposal_id),
                    ("promotion_decision_id", decision.decision_id),
                }
            )
        )
        annotation = PerceptionRegionAnnotationDTO.create(
            field_schema,
            proposal.field_ids,
            label=decision.semantic_name,
            role=decision.semantic_role,
            properties=properties,
            provenance_ref=decision.decision_id,
        )
        relation = None
        expectation = TransitionExpectationDTO.create(
            field_schema_id=proposal.field_schema_id,
            annotation_ids=(annotation.annotation_id,),
            expected_change=proposal.proposed_expected_change,
            minimum_mean_absolute_change=proposal.minimum_mean_absolute_change,
            minimum_changed_fraction=proposal.minimum_changed_fraction,
            minimum_signed_change_magnitude=proposal.minimum_signed_change_magnitude,
        )
    else:
        annotation = None
        relation = _relation(
            proposal=proposal,
            decision=decision,
            directive=directive,
        )
        expectation = TransitionExpectationDTO.create(
            field_schema_id=proposal.field_schema_id,
            relation_ids=(relation.relation_id,),
            expected_change=proposal.proposed_expected_change,
            minimum_mean_absolute_change=proposal.minimum_mean_absolute_change,
            minimum_changed_fraction=proposal.minimum_changed_fraction,
            minimum_signed_change_magnitude=proposal.minimum_signed_change_magnitude,
        )
    return annotation, relation, expectation


def materialize_approved_candidate_promotions(
    proposal_set: CandidatePromotionProposalSetDTO,
    review: CandidatePromotionReviewDTO,
    field_schema: VPMFieldSchemaDTO,
    baseline: PromotionMaterializationBaselineDTO,
    directives: tuple[PromotionMaterializationDirectiveDTO, ...] = (),
) -> PromotionMaterializationChangeSetDTO:
    """Create an inactive reversible change set from fully reviewed approvals."""

    if review.proposal_set_id != proposal_set.proposal_set_id:
        raise PerceptionPromotionMaterializationError(
            "promotion review does not reference the supplied proposal set"
        )
    if review.proposal_ids != proposal_set.proposal_ids:
        raise PerceptionPromotionMaterializationError(
            "promotion review proposal identities disagree with proposal set"
        )
    if review.status != "review_complete":
        raise PerceptionPromotionMaterializationError(
            "promotion materialization requires a fully reviewed ledger"
        )
    if proposal_set.field_schema_id != field_schema.field_schema_id:
        raise PerceptionPromotionMaterializationError(
            "field schema does not match promotion proposal set"
        )
    if baseline.field_schema_id != field_schema.field_schema_id:
        raise PerceptionPromotionMaterializationError(
            "materialization baseline field schema mismatch"
        )
    proposal_map = {item.proposal_id: item for item in proposal_set.proposals}
    decision_map = {item.proposal_id: item for item in review.decisions}
    approved_ids = review.approved_proposal_ids
    approved_decisions = tuple(decision_map[item] for item in approved_ids)
    if any(item.decision != "approved" for item in approved_decisions):
        raise PerceptionPromotionMaterializationError(
            "approved review category contains a non-approved decision"
        )
    directive_map = {item.proposal_id: item for item in directives}
    if len(directive_map) != len(directives):
        raise PerceptionPromotionMaterializationError(
            "materialization directives require one unique directive per proposal"
        )
    if set(directive_map) != set(approved_ids):
        raise PerceptionPromotionMaterializationError(
            "materialization directives must exactly cover approved proposals"
        )
    known_fields = {item.field_id for item in field_schema.fields}
    for proposal_id in approved_ids:
        proposal = proposal_map[proposal_id]
        decision = decision_map[proposal_id]
        directive = directive_map[proposal_id]
        if not set(proposal.field_ids) <= known_fields:
            raise PerceptionPromotionMaterializationError(
                "approved proposal references fields outside the supplied schema"
            )
        if proposal.materialization_status != CANDIDATE_PROMOTION_MATERIALIZATION_STATUS:
            raise PerceptionPromotionMaterializationError(
                "approved proposal was already materialized"
            )
        if decision.materialization_status != CANDIDATE_PROMOTION_MATERIALIZATION_STATUS:
            raise PerceptionPromotionMaterializationError(
                "approved decision was already materialized"
            )
        if directive.target_kind == "relation_annotation":
            missing_members = set(directive.relation_member_annotation_ids) - set(
                baseline.existing_annotation_ids
            )
            if missing_members:
                raise PerceptionPromotionMaterializationError(
                    "relation directive references annotations outside the baseline: "
                    f"{sorted(missing_members)}"
                )

    ordered_ids = tuple(sorted(approved_ids))
    object_rows: list[
        tuple[
            CandidatePromotionProposalDTO,
            CandidatePromotionDecisionDTO,
            PromotionMaterializationDirectiveDTO,
            PerceptionRegionAnnotationDTO | None,
            RelationAnnotationDTO | None,
            TransitionExpectationDTO,
        ]
    ] = []
    new_annotation_ids: set[str] = set()
    new_relation_ids: set[str] = set()
    new_expectation_ids: set[str] = set()
    for proposal_id in ordered_ids:
        proposal = proposal_map[proposal_id]
        decision = decision_map[proposal_id]
        directive = directive_map[proposal_id]
        annotation, relation, expectation = _objects_for_approval(
            proposal=proposal,
            decision=decision,
            directive=directive,
            field_schema=field_schema,
        )
        if annotation is not None:
            if (
                annotation.annotation_id in baseline.existing_annotation_ids
                or annotation.annotation_id in new_annotation_ids
            ):
                raise PerceptionPromotionMaterializationError(
                    "materialized annotation collides with baseline or another change"
                )
            new_annotation_ids.add(annotation.annotation_id)
        if relation is not None:
            if (
                relation.relation_id in baseline.existing_relation_ids
                or relation.relation_id in new_relation_ids
            ):
                raise PerceptionPromotionMaterializationError(
                    "materialized relation collides with baseline or another change"
                )
            new_relation_ids.add(relation.relation_id)
        if (
            expectation.expectation_id in baseline.existing_transition_expectation_ids
            or expectation.expectation_id in new_expectation_ids
        ):
            raise PerceptionPromotionMaterializationError(
                "materialized transition expectation collides with baseline or another change"
            )
        new_expectation_ids.add(expectation.expectation_id)
        object_rows.append(
            (proposal, decision, directive, annotation, relation, expectation)
        )

    count = len(object_rows)
    changes: list[MaterializedPromotionChangeDTO] = []
    for index, row in enumerate(object_rows):
        proposal, decision, directive, annotation, relation, expectation = row
        target = annotation or relation
        target_kind = "annotation" if annotation is not None else "relation"
        target_id = (
            annotation.annotation_id if annotation is not None else relation.relation_id  # type: ignore[union-attr]
        )
        target_digest = _object_digest(target_kind, target)
        expectation_digest = _object_digest("transition_expectation", expectation)
        forward_target_sequence = index * 2 + 1
        forward_expectation_sequence = index * 2 + 2
        inverse_expectation_sequence = (count - index - 1) * 2 + 1
        inverse_target_sequence = (count - index - 1) * 2 + 2
        forward_operations = (
            _operation(
                direction="forward",
                object_kind=target_kind,
                object_id=target_id,
                payload_digest=target_digest,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                sequence=forward_target_sequence,
            ),
            _operation(
                direction="forward",
                object_kind="transition_expectation",
                object_id=expectation.expectation_id,
                payload_digest=expectation_digest,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                sequence=forward_expectation_sequence,
            ),
        )
        inverse_operations = (
            _operation(
                direction="inverse",
                object_kind="transition_expectation",
                object_id=expectation.expectation_id,
                payload_digest=expectation_digest,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                sequence=inverse_expectation_sequence,
            ),
            _operation(
                direction="inverse",
                object_kind=target_kind,
                object_id=target_id,
                payload_digest=target_digest,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                sequence=inverse_target_sequence,
            ),
        )
        values: dict[str, object] = {
            "proposal_id": proposal.proposal_id,
            "decision_id": decision.decision_id,
            "directive_id": directive.directive_id,
            "target_kind": directive.target_kind,
            "semantic_name": decision.semantic_name,
            "semantic_type": decision.semantic_type,
            "semantic_role": decision.semantic_role,
            "annotation": annotation,
            "relation": relation,
            "transition_expectation": expectation,
            "forward_operations": forward_operations,
            "inverse_operations": inverse_operations,
            "materialization_status": PROMOTION_MATERIALIZATION_ITEM_STATUS,
            "version": MATERIALIZED_PROMOTION_CHANGE_VERSION,
        }
        identity_values = dict(values)
        identity_values["annotation"] = None if annotation is None else asdict(annotation)
        identity_values["relation"] = None if relation is None else asdict(relation)
        identity_values["transition_expectation"] = asdict(expectation)
        identity_values["forward_operations"] = tuple(
            asdict(item) for item in forward_operations
        )
        identity_values["inverse_operations"] = tuple(
            asdict(item) for item in inverse_operations
        )
        changes.append(
            MaterializedPromotionChangeDTO(
                change_id=_digest(identity_values),
                **values,  # type: ignore[arg-type]
            )
        )

    ordered_changes = tuple(sorted(changes, key=lambda item: item.change_id))
    forward = tuple(
        sorted(
            (
                operation
                for change in ordered_changes
                for operation in change.forward_operations
            ),
            key=lambda item: item.sequence,
        )
    )
    inverse = tuple(
        sorted(
            (
                operation
                for change in ordered_changes
                for operation in change.inverse_operations
            ),
            key=lambda item: item.sequence,
        )
    )
    status = "staged_inactive" if ordered_changes else "no_approved_changes"
    values = {
        "status": status,
        "proposal_set_id": proposal_set.proposal_set_id,
        "review_id": review.review_id,
        "baseline_id": baseline.baseline_id,
        "baseline_version_id": baseline.baseline_version_id,
        "field_schema_id": field_schema.field_schema_id,
        "approved_proposal_ids": ordered_ids,
        "decision_ids": tuple(
            sorted(decision_map[item].decision_id for item in ordered_ids)
        ),
        "directive_ids": tuple(
            sorted(directive_map[item].directive_id for item in ordered_ids)
        ),
        "change_ids": tuple(sorted(item.change_id for item in ordered_changes)),
        "changes": ordered_changes,
        "forward_operation_ids": tuple(item.operation_id for item in forward),
        "inverse_operation_ids": tuple(item.operation_id for item in inverse),
        "activation_status": PROMOTION_MATERIALIZATION_ACTIVATION_STATUS,
        "semantics": PROMOTION_MATERIALIZATION_SEMANTICS,
        "version": PROMOTION_MATERIALIZATION_CHANGE_SET_VERSION,
    }
    identity_values = dict(values)
    identity_values["changes"] = tuple(asdict(item) for item in ordered_changes)
    return PromotionMaterializationChangeSetDTO(
        change_set_id=_digest(identity_values),
        **values,  # type: ignore[arg-type]
    )

"""Optional semantic annotations and deterministic evidence conformance for Stage P6.

Annotations are declared hypotheses over P4A source fields. Observed registration is
computed from the absolute P5 translator coefficient surface and remains separate
from annotation labels. Conformance findings are diagnostic, not causal proof.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from .fields import VPMFieldSchemaDTO
from .translator import SourceTargetTranslatorDTO

ANNOTATION_VERSION: Final = "perception-region-annotation/1"
RELATION_ANNOTATION_VERSION: Final = "perception-relation-annotation/1"
EXPECTATION_VERSION: Final = "perception-evidence-expectation/1"
OBSERVED_REGISTRATION_VERSION: Final = "perception-observed-registration/1"
CONFORMANCE_REPORT_VERSION: Final = "perception-evidence-conformance/1"
REGISTRATION_SEMANTICS: Final = (
    "share_of_absolute_translator_coefficient_mass_in_declared_source_fields"
)
UNEXPLAINED_REGISTRATION_SEMANTICS: Final = (
    "share_of_absolute_translator_coefficient_mass_outside_all_annotations"
)

CONFORMANCE_STATUSES: Final = {
    "confirmed",
    "confirmed_with_unexpected_evidence",
    "missing_expected_evidence",
    "forbidden_evidence_present",
    "wrong_target_placement",
    "unstable_evidence",
    "inconclusive",
}


class PerceptionConformanceError(ValueError):
    """Raised when annotation or evidence-conformance contracts are invalid."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True)
class PerceptionRegionAnnotationDTO:
    """Optional label/role attached to one or more exact source fields."""

    annotation_id: str
    field_schema_id: str
    field_ids: tuple[str, ...]
    label: str | None = None
    role: str | None = None
    properties: tuple[tuple[str, str], ...] = ()
    provenance_ref: str | None = None
    version: str = ANNOTATION_VERSION

    def __post_init__(self) -> None:
        if not self.annotation_id or not self.field_schema_id:
            raise PerceptionConformanceError("annotation identities must be non-empty")
        if not self.field_ids:
            raise PerceptionConformanceError("annotation requires at least one field")
        if self.field_ids != tuple(sorted(set(self.field_ids))):
            raise PerceptionConformanceError("annotation field_ids must be unique and sorted")
        if self.properties != tuple(sorted(set(self.properties))):
            raise PerceptionConformanceError("annotation properties must be unique and sorted")

    @classmethod
    def create(
        cls,
        field_schema: VPMFieldSchemaDTO,
        field_ids: tuple[str, ...],
        *,
        label: str | None = None,
        role: str | None = None,
        properties: tuple[tuple[str, str], ...] = (),
        provenance_ref: str | None = None,
    ) -> "PerceptionRegionAnnotationDTO":
        known = {field.field_id for field in field_schema.fields}
        ordered_fields = tuple(sorted(set(field_ids)))
        unknown = set(ordered_fields) - known
        if unknown:
            raise PerceptionConformanceError(f"annotation contains unknown fields: {sorted(unknown)}")
        ordered_properties = tuple(sorted(set(properties)))
        payload: Mapping[str, object] = {
            "field_schema_id": field_schema.field_schema_id,
            "field_ids": list(ordered_fields),
            "label": label,
            "properties": [list(item) for item in ordered_properties],
            "provenance_ref": provenance_ref,
            "role": role,
            "version": ANNOTATION_VERSION,
        }
        return cls(
            annotation_id=_digest(_canonical_json(payload)),
            field_schema_id=field_schema.field_schema_id,
            field_ids=ordered_fields,
            label=label,
            role=role,
            properties=ordered_properties,
            provenance_ref=provenance_ref,
        )


@dataclass(frozen=True)
class RelationAnnotationDTO:
    relation_id: str
    relation_type: str
    member_annotation_ids: tuple[str, ...]
    derived_field_ids: tuple[str, ...] = ()
    value: float | str | None = None
    version: str = RELATION_ANNOTATION_VERSION

    def __post_init__(self) -> None:
        if not self.relation_id or not self.relation_type:
            raise PerceptionConformanceError("relation identity/type must be non-empty")
        if len(self.member_annotation_ids) < 2:
            raise PerceptionConformanceError("relation requires at least two annotation members")
        if self.member_annotation_ids != tuple(sorted(set(self.member_annotation_ids))):
            raise PerceptionConformanceError("relation members must be unique and sorted")
        if self.derived_field_ids != tuple(sorted(set(self.derived_field_ids))):
            raise PerceptionConformanceError("derived fields must be unique and sorted")


@dataclass(frozen=True)
class EvidenceExpectationDTO:
    expectation_id: str
    field_schema_id: str
    source_annotation_ids: tuple[str, ...]
    expected_action_labels: tuple[str, ...]
    required_relation_ids: tuple[str, ...] = ()
    forbidden_annotation_ids: tuple[str, ...] = ()
    minimum_registration: float | None = None
    maximum_unexplained_registration: float | None = None
    minimum_stability: float | None = None
    version: str = EXPECTATION_VERSION

    def __post_init__(self) -> None:
        if not self.expectation_id or not self.field_schema_id:
            raise PerceptionConformanceError("expectation identities must be non-empty")
        if not self.source_annotation_ids or not self.expected_action_labels:
            raise PerceptionConformanceError("expectation requires source annotations and actions")
        for name, values in (
            ("source_annotation_ids", self.source_annotation_ids),
            ("expected_action_labels", self.expected_action_labels),
            ("required_relation_ids", self.required_relation_ids),
            ("forbidden_annotation_ids", self.forbidden_annotation_ids),
        ):
            if values != tuple(sorted(set(values))):
                raise PerceptionConformanceError(f"{name} must be unique and sorted")
        for name, value in (
            ("minimum_registration", self.minimum_registration),
            ("maximum_unexplained_registration", self.maximum_unexplained_registration),
            ("minimum_stability", self.minimum_stability),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise PerceptionConformanceError(f"{name} must be in [0, 1]")

    @classmethod
    def create(
        cls,
        *,
        field_schema_id: str,
        source_annotation_ids: tuple[str, ...],
        expected_action_labels: tuple[str, ...],
        required_relation_ids: tuple[str, ...] = (),
        forbidden_annotation_ids: tuple[str, ...] = (),
        minimum_registration: float | None = None,
        maximum_unexplained_registration: float | None = None,
        minimum_stability: float | None = None,
    ) -> "EvidenceExpectationDTO":
        payload: Mapping[str, object] = {
            "expected_action_labels": sorted(set(expected_action_labels)),
            "field_schema_id": field_schema_id,
            "forbidden_annotation_ids": sorted(set(forbidden_annotation_ids)),
            "maximum_unexplained_registration": maximum_unexplained_registration,
            "minimum_registration": minimum_registration,
            "minimum_stability": minimum_stability,
            "required_relation_ids": sorted(set(required_relation_ids)),
            "source_annotation_ids": sorted(set(source_annotation_ids)),
            "version": EXPECTATION_VERSION,
        }
        return cls(
            expectation_id=_digest(_canonical_json(payload)),
            field_schema_id=field_schema_id,
            source_annotation_ids=tuple(payload["source_annotation_ids"]),  # type: ignore[arg-type]
            expected_action_labels=tuple(payload["expected_action_labels"]),  # type: ignore[arg-type]
            required_relation_ids=tuple(payload["required_relation_ids"]),  # type: ignore[arg-type]
            forbidden_annotation_ids=tuple(payload["forbidden_annotation_ids"]),  # type: ignore[arg-type]
            minimum_registration=minimum_registration,
            maximum_unexplained_registration=maximum_unexplained_registration,
            minimum_stability=minimum_stability,
        )


@dataclass(frozen=True)
class ObservedAnnotationRegistrationDTO:
    annotation_id: str
    action_label: str
    registration: float
    registration_semantics: str = REGISTRATION_SEMANTICS
    stability: float | None = None
    version: str = OBSERVED_REGISTRATION_VERSION

    def __post_init__(self) -> None:
        if not self.annotation_id or not self.action_label:
            raise PerceptionConformanceError("observed registration identities must be non-empty")
        if not 0.0 <= self.registration <= 1.0:
            raise PerceptionConformanceError("registration must be in [0, 1]")
        if self.stability is not None and not 0.0 <= self.stability <= 1.0:
            raise PerceptionConformanceError("stability must be in [0, 1]")
        if self.registration_semantics != REGISTRATION_SEMANTICS:
            raise PerceptionConformanceError("unsupported registration semantics")


@dataclass(frozen=True)
class EvidenceConformanceFindingDTO:
    finding_id: str
    status: str
    expectation_id: str
    action_labels: tuple[str, ...]
    annotation_ids: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        if self.status not in CONFORMANCE_STATUSES:
            raise PerceptionConformanceError("unsupported conformance status")


@dataclass(frozen=True)
class EvidenceConformanceReportDTO:
    report_id: str
    translator_id: str
    field_schema_id: str
    expectation_id: str
    registrations: tuple[ObservedAnnotationRegistrationDTO, ...]
    unexplained_registration_by_action: tuple[tuple[str, float], ...]
    findings: tuple[EvidenceConformanceFindingDTO, ...]
    overall_status: str
    registration_semantics: str = REGISTRATION_SEMANTICS
    unexplained_semantics: str = UNEXPLAINED_REGISTRATION_SEMANTICS
    version: str = CONFORMANCE_REPORT_VERSION

    def __post_init__(self) -> None:
        if self.overall_status not in CONFORMANCE_STATUSES:
            raise PerceptionConformanceError("unsupported overall conformance status")
        if not self.findings:
            raise PerceptionConformanceError("conformance report requires at least one finding")


def _registration_surface(
    translator: SourceTargetTranslatorDTO,
    annotations: tuple[PerceptionRegionAnnotationDTO, ...],
    stability_by_annotation_action: Mapping[tuple[str, str], float] | None,
) -> tuple[tuple[ObservedAnnotationRegistrationDTO, ...], tuple[tuple[str, float], ...]]:
    index_by_field = {field_id: index for index, field_id in enumerate(translator.source_field_ids)}
    annotated_fields = {field_id for annotation in annotations for field_id in annotation.field_ids}
    registrations: list[ObservedAnnotationRegistrationDTO] = []
    unexplained: list[tuple[str, float]] = []
    for action_index, action_label in enumerate(translator.action_labels):
        weights = np.abs(np.asarray(translator.coefficients[action_index], dtype=np.float64))
        total = float(np.sum(weights))
        denominator = total if total > 0.0 else 1.0
        for annotation in annotations:
            mass = sum(float(weights[index_by_field[field_id]]) for field_id in annotation.field_ids)
            registrations.append(
                ObservedAnnotationRegistrationDTO(
                    annotation_id=annotation.annotation_id,
                    action_label=action_label,
                    registration=mass / denominator,
                    stability=None
                    if stability_by_annotation_action is None
                    else stability_by_annotation_action.get((annotation.annotation_id, action_label)),
                )
            )
        unexplained_mass = sum(
            float(weights[index])
            for field_id, index in index_by_field.items()
            if field_id not in annotated_fields
        )
        unexplained.append((action_label, unexplained_mass / denominator))
    return (
        tuple(sorted(registrations, key=lambda item: (item.annotation_id, item.action_label))),
        tuple(sorted(unexplained)),
    )


def evaluate_evidence_conformance(
    translator: SourceTargetTranslatorDTO,
    field_schema: VPMFieldSchemaDTO,
    annotations: tuple[PerceptionRegionAnnotationDTO, ...],
    relations: tuple[RelationAnnotationDTO, ...],
    expectation: EvidenceExpectationDTO,
    *,
    stability_by_annotation_action: Mapping[tuple[str, str], float] | None = None,
) -> EvidenceConformanceReportDTO:
    """Compare optional expectations with action-specific translator evidence."""

    if translator.source_field_schema_id != field_schema.field_schema_id:
        raise PerceptionConformanceError("field schema does not match translator")
    if expectation.field_schema_id != field_schema.field_schema_id:
        raise PerceptionConformanceError("expectation field schema mismatch")
    annotation_by_id = {item.annotation_id: item for item in annotations}
    if len(annotation_by_id) != len(annotations):
        raise PerceptionConformanceError("annotations must have unique identities")
    relation_by_id = {item.relation_id: item for item in relations}
    known_fields = set(translator.source_field_ids)
    for annotation in annotations:
        if annotation.field_schema_id != field_schema.field_schema_id:
            raise PerceptionConformanceError("annotation field schema mismatch")
        if not set(annotation.field_ids).issubset(known_fields):
            raise PerceptionConformanceError("annotation contains field outside translator")
    missing_annotations = (
        set(expectation.source_annotation_ids) | set(expectation.forbidden_annotation_ids)
    ) - set(annotation_by_id)
    if missing_annotations:
        raise PerceptionConformanceError(
            f"expectation references unknown annotations: {sorted(missing_annotations)}"
        )
    missing_relations = set(expectation.required_relation_ids) - set(relation_by_id)
    if missing_relations:
        raise PerceptionConformanceError(
            f"expectation references unknown relations: {sorted(missing_relations)}"
        )
    if not set(expectation.expected_action_labels).issubset(set(translator.action_labels)):
        raise PerceptionConformanceError("expectation contains action outside translator")

    registrations, unexplained = _registration_surface(
        translator, annotations, stability_by_annotation_action
    )
    registration_by_key = {
        (item.annotation_id, item.action_label): item for item in registrations
    }
    unexplained_by_action = dict(unexplained)
    threshold = expectation.minimum_registration
    findings: list[EvidenceConformanceFindingDTO] = []

    def add(status: str, actions: tuple[str, ...], annotation_ids: tuple[str, ...], detail: str) -> None:
        payload: Mapping[str, object] = {
            "action_labels": list(actions),
            "annotation_ids": list(annotation_ids),
            "detail": detail,
            "expectation_id": expectation.expectation_id,
            "status": status,
        }
        findings.append(
            EvidenceConformanceFindingDTO(
                finding_id=_digest(_canonical_json(payload)),
                status=status,
                expectation_id=expectation.expectation_id,
                action_labels=actions,
                annotation_ids=annotation_ids,
                detail=detail,
            )
        )

    if threshold is None:
        add("inconclusive", expectation.expected_action_labels, expectation.source_annotation_ids,
            "minimum_registration is not declared")
    else:
        missing: list[tuple[str, str]] = []
        misplaced: list[tuple[str, str, str]] = []
        unstable: list[tuple[str, str]] = []
        for annotation_id in expectation.source_annotation_ids:
            for action_label in expectation.expected_action_labels:
                observed = registration_by_key[(annotation_id, action_label)]
                if observed.registration < threshold:
                    best_action = max(
                        translator.action_labels,
                        key=lambda label: (
                            registration_by_key[(annotation_id, label)].registration,
                            label,
                        ),
                    )
                    if (
                        best_action not in expectation.expected_action_labels
                        and registration_by_key[(annotation_id, best_action)].registration >= threshold
                    ):
                        misplaced.append((annotation_id, action_label, best_action))
                    else:
                        missing.append((annotation_id, action_label))
                if (
                    expectation.minimum_stability is not None
                    and observed.stability is not None
                    and observed.stability < expectation.minimum_stability
                ):
                    unstable.append((annotation_id, action_label))
        if misplaced:
            add(
                "wrong_target_placement",
                tuple(sorted({item[1] for item in misplaced} | {item[2] for item in misplaced})),
                tuple(sorted({item[0] for item in misplaced})),
                "expected source evidence registers above threshold for another target action",
            )
        if missing:
            add(
                "missing_expected_evidence",
                tuple(sorted({item[1] for item in missing})),
                tuple(sorted({item[0] for item in missing})),
                "expected annotation registration is below the declared minimum",
            )
        if unstable:
            add(
                "unstable_evidence",
                tuple(sorted({item[1] for item in unstable})),
                tuple(sorted({item[0] for item in unstable})),
                "observed stability is below the declared minimum",
            )

        forbidden_present = [
            (annotation_id, action_label)
            for annotation_id in expectation.forbidden_annotation_ids
            for action_label in expectation.expected_action_labels
            if registration_by_key[(annotation_id, action_label)].registration >= threshold
        ]
        if forbidden_present:
            add(
                "forbidden_evidence_present",
                tuple(sorted({item[1] for item in forbidden_present})),
                tuple(sorted({item[0] for item in forbidden_present})),
                "forbidden annotation registration meets or exceeds the declared minimum",
            )

        blocking = {item.status for item in findings} & {
            "wrong_target_placement",
            "missing_expected_evidence",
            "unstable_evidence",
            "forbidden_evidence_present",
        }
        if not blocking:
            unexpected = False
            if expectation.maximum_unexplained_registration is not None:
                unexpected = any(
                    unexplained_by_action[action] > expectation.maximum_unexplained_registration
                    for action in expectation.expected_action_labels
                )
            add(
                "confirmed_with_unexpected_evidence" if unexpected else "confirmed",
                expectation.expected_action_labels,
                expectation.source_annotation_ids,
                "declared evidence expectation is met"
                + (" but unexplained coefficient mass exceeds the declared maximum" if unexpected else ""),
            )

    priority = (
        "forbidden_evidence_present",
        "wrong_target_placement",
        "missing_expected_evidence",
        "unstable_evidence",
        "inconclusive",
        "confirmed_with_unexpected_evidence",
        "confirmed",
    )
    statuses = {item.status for item in findings}
    overall = next(status for status in priority if status in statuses)
    ordered_findings = tuple(sorted(findings, key=lambda item: (priority.index(item.status), item.finding_id)))
    payload = {
        "expectation_id": expectation.expectation_id,
        "field_schema_id": field_schema.field_schema_id,
        "findings": [item.finding_id for item in ordered_findings],
        "overall_status": overall,
        "registrations": [
            [item.annotation_id, item.action_label, item.registration, item.stability]
            for item in registrations
        ],
        "translator_id": translator.translator_id,
        "unexplained_registration_by_action": [list(item) for item in unexplained],
        "version": CONFORMANCE_REPORT_VERSION,
    }
    return EvidenceConformanceReportDTO(
        report_id=_digest(_canonical_json(payload)),
        translator_id=translator.translator_id,
        field_schema_id=field_schema.field_schema_id,
        expectation_id=expectation.expectation_id,
        registrations=registrations,
        unexplained_registration_by_action=unexplained,
        findings=ordered_findings,
        overall_status=overall,
    )

"""Annotation-aware transition conformance for Stage P18B.

P18B compares declarations about marked objects, relations, and stable controls
with one immutable P18A transition artifact. Findings are deterministic tests of
the declarations, not causal conclusions and not edits to the observed evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Final, Mapping

from .expectations import PerceptionRegionAnnotationDTO, RelationAnnotationDTO
from .transition_evidence import TransitionEvidenceVPMDTO, TransitionFieldEvidenceDTO

TRANSITION_EXPECTATION_VERSION: Final = "perception-transition-expectation/1"
TRANSITION_CONFORMANCE_FINDING_VERSION: Final = (
    "perception-transition-conformance-finding/1"
)
TRANSITION_CONFORMANCE_REPORT_VERSION: Final = (
    "perception-transition-conformance-report/1"
)
TRANSITION_CONFORMANCE_SEMANTICS: Final = "weighted_field_transition_measurements_compared_with_declared_annotation_or_relation_thresholds"
TRANSITION_EXPECTED_CHANGE_KINDS: Final = {
    "stable",
    "change",
    "increase",
    "decrease",
}
TRANSITION_CONFORMANCE_STATUSES: Final = {
    "confirmed",
    "missing_expected_change",
    "unexpected_change",
    "excessive_change",
    "insufficient_change",
    "wrong_change_direction",
    "unexplained_change",
    "inconclusive",
}
TRANSITION_CONFORMANCE_REPORT_STATUSES: Final = {
    "conformant",
    "attention_required",
    "nonconformant",
}
_NONCONFORMANT: Final = {
    "missing_expected_change",
    "unexpected_change",
    "excessive_change",
    "insufficient_change",
    "wrong_change_direction",
}
_ATTENTION: Final = {"unexplained_change", "inconclusive"}


class PerceptionTransitionConformanceError(ValueError):
    """Raised when transition-conformance contracts are invalid."""


def _json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, object]) -> str:
    value = _json(payload)
    hasher = hashlib.sha256()
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)
    return f"sha256:{hasher.hexdigest()}"


def _payload(value: object, identity_field: str) -> dict[str, object]:
    payload = asdict(value)  # type: ignore[arg-type]
    payload.pop(identity_field)
    return payload


def _unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise PerceptionTransitionConformanceError(f"{name} must be in [0, 1]")


def _ordered(name: str, values: tuple[str, ...]) -> None:
    if values != tuple(sorted(set(values))):
        raise PerceptionTransitionConformanceError(f"{name} must be unique and sorted")


@dataclass(frozen=True)
class TransitionExpectationDTO:
    """Declared transition behaviour for annotation or relation identities."""

    expectation_id: str
    field_schema_id: str
    annotation_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    expected_change: str
    minimum_mean_absolute_change: float = 0.0
    maximum_mean_absolute_change: float = 1.0
    minimum_changed_fraction: float = 0.0
    maximum_changed_fraction: float = 1.0
    minimum_signed_change_magnitude: float = 0.0
    version: str = TRANSITION_EXPECTATION_VERSION

    def __post_init__(self) -> None:
        if not self.expectation_id or not self.field_schema_id:
            raise PerceptionTransitionConformanceError(
                "transition expectation identities must be non-empty"
            )
        if not self.annotation_ids and not self.relation_ids:
            raise PerceptionTransitionConformanceError(
                "transition expectation requires annotations or relations"
            )
        _ordered("annotation_ids", self.annotation_ids)
        _ordered("relation_ids", self.relation_ids)
        if self.expected_change not in TRANSITION_EXPECTED_CHANGE_KINDS:
            raise PerceptionTransitionConformanceError(
                f"unsupported expected_change: {self.expected_change}"
            )
        for name in (
            "minimum_mean_absolute_change",
            "maximum_mean_absolute_change",
            "minimum_changed_fraction",
            "maximum_changed_fraction",
            "minimum_signed_change_magnitude",
        ):
            _unit(name, getattr(self, name))
        if self.minimum_mean_absolute_change > self.maximum_mean_absolute_change:
            raise PerceptionTransitionConformanceError(
                "minimum_mean_absolute_change exceeds maximum"
            )
        if self.minimum_changed_fraction > self.maximum_changed_fraction:
            raise PerceptionTransitionConformanceError(
                "minimum_changed_fraction exceeds maximum"
            )
        if (
            self.expected_change not in {"increase", "decrease"}
            and self.minimum_signed_change_magnitude != 0.0
        ):
            raise PerceptionTransitionConformanceError(
                "minimum_signed_change_magnitude requires increase or decrease"
            )
        if self.expected_change == "stable" and (
            self.minimum_mean_absolute_change != 0.0
            or self.minimum_changed_fraction != 0.0
        ):
            raise PerceptionTransitionConformanceError(
                "stable expectations use maximum tolerances"
            )
        if self.version != TRANSITION_EXPECTATION_VERSION:
            raise PerceptionTransitionConformanceError(
                "unsupported transition expectation version"
            )
        if self.expectation_id != _digest(_payload(self, "expectation_id")):
            raise PerceptionTransitionConformanceError(
                "transition expectation identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        *,
        field_schema_id: str,
        annotation_ids: tuple[str, ...] = (),
        relation_ids: tuple[str, ...] = (),
        expected_change: str,
        minimum_mean_absolute_change: float = 0.0,
        maximum_mean_absolute_change: float = 1.0,
        minimum_changed_fraction: float = 0.0,
        maximum_changed_fraction: float = 1.0,
        minimum_signed_change_magnitude: float = 0.0,
    ) -> "TransitionExpectationDTO":
        values: dict[str, object] = {
            "field_schema_id": field_schema_id,
            "annotation_ids": tuple(sorted(set(annotation_ids))),
            "relation_ids": tuple(sorted(set(relation_ids))),
            "expected_change": expected_change,
            "minimum_mean_absolute_change": minimum_mean_absolute_change,
            "maximum_mean_absolute_change": maximum_mean_absolute_change,
            "minimum_changed_fraction": minimum_changed_fraction,
            "maximum_changed_fraction": maximum_changed_fraction,
            "minimum_signed_change_magnitude": minimum_signed_change_magnitude,
            "version": TRANSITION_EXPECTATION_VERSION,
        }
        return cls(expectation_id=_digest(values), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TransitionConformanceFindingDTO:
    finding_id: str
    status: str
    expectation_id: str | None
    annotation_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    field_ids: tuple[str, ...]
    observed_mean_absolute_change: float
    observed_mean_signed_change: float
    observed_changed_fraction: float
    detail: str
    version: str = TRANSITION_CONFORMANCE_FINDING_VERSION

    def __post_init__(self) -> None:
        if not self.finding_id or not self.detail:
            raise PerceptionTransitionConformanceError(
                "transition finding identity and detail must be non-empty"
            )
        if self.status not in TRANSITION_CONFORMANCE_STATUSES:
            raise PerceptionTransitionConformanceError(
                f"unsupported transition finding status: {self.status}"
            )
        for name in ("annotation_ids", "relation_ids", "field_ids"):
            _ordered(name, getattr(self, name))
        if not self.field_ids:
            raise PerceptionTransitionConformanceError(
                "transition finding requires at least one field"
            )
        _unit("observed_mean_absolute_change", self.observed_mean_absolute_change)
        _unit("observed_changed_fraction", self.observed_changed_fraction)
        if not -1.0 <= self.observed_mean_signed_change <= 1.0:
            raise PerceptionTransitionConformanceError(
                "observed_mean_signed_change must be in [-1, 1]"
            )
        if self.status == "unexplained_change":
            if (
                self.expectation_id is not None
                or self.annotation_ids
                or self.relation_ids
            ):
                raise PerceptionTransitionConformanceError(
                    "unexplained findings cannot reference declared targets"
                )
        elif self.expectation_id is None:
            raise PerceptionTransitionConformanceError(
                "declared findings require expectation_id"
            )
        if self.version != TRANSITION_CONFORMANCE_FINDING_VERSION:
            raise PerceptionTransitionConformanceError(
                "unsupported transition finding version"
            )
        if self.finding_id != _digest(_payload(self, "finding_id")):
            raise PerceptionTransitionConformanceError(
                "transition finding identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class TransitionConformanceReportDTO:
    report_id: str
    status: str
    transition_evidence_id: str
    field_schema_id: str
    expectation_ids: tuple[str, ...]
    annotation_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    minimum_unexplained_mean_absolute_change: float
    minimum_unexplained_changed_fraction: float
    findings: tuple[TransitionConformanceFindingDTO, ...]
    semantics: str = TRANSITION_CONFORMANCE_SEMANTICS
    version: str = TRANSITION_CONFORMANCE_REPORT_VERSION

    def __post_init__(self) -> None:
        if (
            not self.report_id
            or not self.transition_evidence_id
            or not self.field_schema_id
        ):
            raise PerceptionTransitionConformanceError(
                "transition report identities must be non-empty"
            )
        if not self.expectation_ids:
            raise PerceptionTransitionConformanceError(
                "transition report requires at least one expectation"
            )
        if self.status not in TRANSITION_CONFORMANCE_REPORT_STATUSES:
            raise PerceptionTransitionConformanceError(
                f"unsupported transition report status: {self.status}"
            )
        for name in ("expectation_ids", "annotation_ids", "relation_ids"):
            _ordered(name, getattr(self, name))
        finding_ids = tuple(item.finding_id for item in self.findings)
        _ordered("transition finding identities", finding_ids)
        _unit(
            "minimum_unexplained_mean_absolute_change",
            self.minimum_unexplained_mean_absolute_change,
        )
        _unit(
            "minimum_unexplained_changed_fraction",
            self.minimum_unexplained_changed_fraction,
        )
        declared = {
            item.expectation_id
            for item in self.findings
            if item.expectation_id is not None
        }
        if declared != set(self.expectation_ids):
            raise PerceptionTransitionConformanceError(
                "report expectation identities must match declared findings"
            )
        if self.status != _report_status(self.findings):
            raise PerceptionTransitionConformanceError(
                "transition report status disagrees with findings"
            )
        if self.semantics != TRANSITION_CONFORMANCE_SEMANTICS:
            raise PerceptionTransitionConformanceError(
                "unsupported transition conformance semantics"
            )
        if self.version != TRANSITION_CONFORMANCE_REPORT_VERSION:
            raise PerceptionTransitionConformanceError(
                "unsupported transition conformance report version"
            )
        if self.report_id != _digest(_payload(self, "report_id")):
            raise PerceptionTransitionConformanceError(
                "transition report identity disagrees with canonical payload"
            )

    def findings_for_status(
        self, status: str
    ) -> tuple[TransitionConformanceFindingDTO, ...]:
        if status not in TRANSITION_CONFORMANCE_STATUSES:
            raise PerceptionTransitionConformanceError(
                f"unsupported transition finding status: {status}"
            )
        return tuple(item for item in self.findings if item.status == status)


def _report_status(findings: tuple[TransitionConformanceFindingDTO, ...]) -> str:
    statuses = {item.status for item in findings}
    if statuses & _NONCONFORMANT:
        return "nonconformant"
    if statuses & _ATTENTION:
        return "attention_required"
    return "conformant"


def _finding(
    *,
    status: str,
    expectation: TransitionExpectationDTO | None,
    field_ids: tuple[str, ...],
    mean_absolute: float,
    mean_signed: float,
    changed_fraction: float,
    detail: str,
) -> TransitionConformanceFindingDTO:
    values: dict[str, object] = {
        "status": status,
        "expectation_id": None if expectation is None else expectation.expectation_id,
        "annotation_ids": () if expectation is None else expectation.annotation_ids,
        "relation_ids": () if expectation is None else expectation.relation_ids,
        "field_ids": field_ids,
        "observed_mean_absolute_change": mean_absolute,
        "observed_mean_signed_change": mean_signed,
        "observed_changed_fraction": changed_fraction,
        "detail": detail,
        "version": TRANSITION_CONFORMANCE_FINDING_VERSION,
    }
    return TransitionConformanceFindingDTO(
        finding_id=_digest(values),
        **values,  # type: ignore[arg-type]
    )


def _aggregate(
    fields: tuple[TransitionFieldEvidenceDTO, ...],
) -> tuple[float, float, float, int]:
    total = sum(item.total_value_count for item in fields)
    if total <= 0:
        raise PerceptionTransitionConformanceError(
            "transition target has no measurable values"
        )
    absolute = (
        sum(item.mean_absolute_change * item.total_value_count for item in fields)
        / total
    )
    signed = (
        sum(item.mean_signed_change * item.total_value_count for item in fields) / total
    )
    changed = sum(item.changed_value_count for item in fields)
    return absolute, signed, changed / total, changed


def _classify(
    expectation: TransitionExpectationDTO,
    fields: tuple[TransitionFieldEvidenceDTO, ...],
) -> TransitionConformanceFindingDTO:
    absolute, signed, fraction, changed = _aggregate(fields)
    field_ids = tuple(sorted(item.field_id for item in fields))

    if expectation.expected_change == "stable":
        if (
            absolute > expectation.maximum_mean_absolute_change
            or fraction > expectation.maximum_changed_fraction
        ):
            status = "unexpected_change"
            detail = "declared stable target exceeded its change tolerance"
        else:
            status = "confirmed"
            detail = "declared stable target remained within tolerance"
    elif changed == 0:
        status = "missing_expected_change"
        detail = "declared changing target had no P18A threshold crossings"
    elif (
        absolute < expectation.minimum_mean_absolute_change
        or fraction < expectation.minimum_changed_fraction
    ):
        status = "insufficient_change"
        detail = "observed transition did not reach the declared minimum"
    elif expectation.expected_change in {"increase", "decrease"}:
        required = expectation.minimum_signed_change_magnitude
        wrong = (
            signed < -required
            if expectation.expected_change == "increase"
            else signed > required
        )
        inconclusive = (
            signed <= required
            if expectation.expected_change == "increase"
            else signed >= -required
        )
        if wrong:
            status = "wrong_change_direction"
            detail = "observed signed change was opposite the declaration"
        elif inconclusive:
            status = "inconclusive"
            detail = "absolute change was present but net direction was too small"
        elif (
            absolute > expectation.maximum_mean_absolute_change
            or fraction > expectation.maximum_changed_fraction
        ):
            status = "excessive_change"
            detail = "observed directional change exceeded the declared maximum"
        else:
            status = "confirmed"
            detail = f"observed transition confirmed the declared {expectation.expected_change}"
    elif (
        absolute > expectation.maximum_mean_absolute_change
        or fraction > expectation.maximum_changed_fraction
    ):
        status = "excessive_change"
        detail = "observed change exceeded the declared maximum"
    else:
        status = "confirmed"
        detail = "observed transition confirmed the declared change"

    return _finding(
        status=status,
        expectation=expectation,
        field_ids=field_ids,
        mean_absolute=absolute,
        mean_signed=signed,
        changed_fraction=fraction,
        detail=detail,
    )


def _relation_fields(
    relation: RelationAnnotationDTO,
    annotations: Mapping[str, PerceptionRegionAnnotationDTO],
    known_fields: set[str],
) -> tuple[str, ...]:
    unknown_members = set(relation.member_annotation_ids) - set(annotations)
    if unknown_members:
        raise PerceptionTransitionConformanceError(
            f"relation references unknown annotations: {sorted(unknown_members)}"
        )
    if relation.derived_field_ids:
        unknown = set(relation.derived_field_ids) - known_fields
        if unknown:
            raise PerceptionTransitionConformanceError(
                f"relation references unknown derived fields: {sorted(unknown)}"
            )
        return relation.derived_field_ids
    return tuple(
        sorted(
            {
                field_id
                for annotation_id in relation.member_annotation_ids
                for field_id in annotations[annotation_id].field_ids
            }
        )
    )


def evaluate_transition_conformance(
    transition: TransitionEvidenceVPMDTO,
    expectations: tuple[TransitionExpectationDTO, ...],
    annotations: tuple[PerceptionRegionAnnotationDTO, ...],
    relations: tuple[RelationAnnotationDTO, ...] = (),
    *,
    minimum_unexplained_mean_absolute_change: float = 0.0,
    minimum_unexplained_changed_fraction: float = 0.0,
) -> TransitionConformanceReportDTO:
    """Evaluate declarations while preserving the exact P18A measurements."""

    if not expectations:
        raise PerceptionTransitionConformanceError(
            "transition conformance requires at least one expectation"
        )
    _unit(
        "minimum_unexplained_mean_absolute_change",
        minimum_unexplained_mean_absolute_change,
    )
    _unit(
        "minimum_unexplained_changed_fraction",
        minimum_unexplained_changed_fraction,
    )
    expectation_map = {item.expectation_id: item for item in expectations}
    annotation_map = {item.annotation_id: item for item in annotations}
    relation_map = {item.relation_id: item for item in relations}
    if len(expectation_map) != len(expectations):
        raise PerceptionTransitionConformanceError(
            "transition expectations must have unique identities"
        )
    if len(annotation_map) != len(annotations):
        raise PerceptionTransitionConformanceError(
            "transition annotations must have unique identities"
        )
    if len(relation_map) != len(relations):
        raise PerceptionTransitionConformanceError(
            "transition relations must have unique identities"
        )
    if tuple(sorted(annotation_map)) != transition.annotation_ids:
        raise PerceptionTransitionConformanceError(
            "supplied annotations must exactly match P18A annotation identities"
        )

    field_map = {item.field_id: item for item in transition.fields}
    known_fields = set(field_map)
    expected_bindings: dict[str, list[str]] = {
        field_id: [] for field_id in known_fields
    }
    for annotation in annotations:
        if annotation.field_schema_id != transition.field_schema_id:
            raise PerceptionTransitionConformanceError(
                "annotation field schema does not match transition evidence"
            )
        unknown = set(annotation.field_ids) - known_fields
        if unknown:
            raise PerceptionTransitionConformanceError(
                f"annotation references unknown fields: {sorted(unknown)}"
            )
        for field_id in annotation.field_ids:
            expected_bindings[field_id].append(annotation.annotation_id)
    for field_id, field in field_map.items():
        if field.annotation_ids != tuple(sorted(expected_bindings[field_id])):
            raise PerceptionTransitionConformanceError(
                "supplied annotations disagree with P18A field bindings"
            )

    relation_field_map = {
        identity: _relation_fields(relation, annotation_map, known_fields)
        for identity, relation in relation_map.items()
    }
    findings: list[TransitionConformanceFindingDTO] = []
    covered: set[str] = set()
    for expectation in sorted(expectations, key=lambda item: item.expectation_id):
        if expectation.field_schema_id != transition.field_schema_id:
            raise PerceptionTransitionConformanceError(
                "expectation field schema does not match transition evidence"
            )
        unknown_annotations = set(expectation.annotation_ids) - set(annotation_map)
        unknown_relations = set(expectation.relation_ids) - set(relation_map)
        if unknown_annotations:
            raise PerceptionTransitionConformanceError(
                f"expectation references unknown annotations: {sorted(unknown_annotations)}"
            )
        if unknown_relations:
            raise PerceptionTransitionConformanceError(
                f"expectation references unknown relations: {sorted(unknown_relations)}"
            )
        target_fields = {
            field_id
            for annotation_id in expectation.annotation_ids
            for field_id in annotation_map[annotation_id].field_ids
        }
        for relation_id in expectation.relation_ids:
            target_fields.update(relation_field_map[relation_id])
        if not target_fields:
            raise PerceptionTransitionConformanceError(
                "transition expectation resolves to no fields"
            )
        covered.update(target_fields)
        findings.append(
            _classify(
                expectation,
                tuple(field_map[field_id] for field_id in sorted(target_fields)),
            )
        )

    for field_id in sorted(known_fields - covered):
        field = field_map[field_id]
        if (
            field.changed_value_count > 0
            and field.mean_absolute_change >= minimum_unexplained_mean_absolute_change
            and field.changed_fraction >= minimum_unexplained_changed_fraction
        ):
            findings.append(
                _finding(
                    status="unexplained_change",
                    expectation=None,
                    field_ids=(field_id,),
                    mean_absolute=field.mean_absolute_change,
                    mean_signed=field.mean_signed_change,
                    changed_fraction=field.changed_fraction,
                    detail=(
                        "field changed above the unexplained thresholds without "
                        "a declared transition expectation"
                    ),
                )
            )

    ordered_findings = tuple(sorted(findings, key=lambda item: item.finding_id))
    values: dict[str, object] = {
        "status": _report_status(ordered_findings),
        "transition_evidence_id": transition.transition_evidence_id,
        "field_schema_id": transition.field_schema_id,
        "expectation_ids": tuple(sorted(expectation_map)),
        "annotation_ids": tuple(sorted(annotation_map)),
        "relation_ids": tuple(sorted(relation_map)),
        "minimum_unexplained_mean_absolute_change": (
            minimum_unexplained_mean_absolute_change
        ),
        "minimum_unexplained_changed_fraction": minimum_unexplained_changed_fraction,
        "findings": ordered_findings,
        "semantics": TRANSITION_CONFORMANCE_SEMANTICS,
        "version": TRANSITION_CONFORMANCE_REPORT_VERSION,
    }
    identity_values = dict(values)
    identity_values["findings"] = tuple(asdict(item) for item in ordered_findings)
    return TransitionConformanceReportDTO(
        report_id=_digest(identity_values),
        **values,  # type: ignore[arg-type]
    )

"""Neutral, domain-agnostic report DTOs.

ZeroModel does not understand what "hallucination energy" or "AI-artifact
phrasing" mean. An external application's `ReportAdapter` translates its own
typed domain report into these neutral shapes; `compile_report` (in
`report_compiler.py`) only ever sees `AdaptedReportDTO` and below.

`adapted_report_id` follows the same self-validating pattern used
throughout this workspace (`ArtifactRef.artifact_id`,
`ArtifactAuthorizationDTO.authorization_id`, `NavigationTileDTO.tile_id`):
recomputed from the object's own canonical content in `__post_init__`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from zeromodel.artifacts.canonicalization import canonical_json_bytes, sha256_digest
from zeromodel.artifacts.report_errors import ReportAdaptationError
from zeromodel.artifacts.score_semantics import ScoreSemantics

SPEC_VERSION = "zeromodel-artifacts-report/v1"

_MISSING_VALUE_SEMANTICS = ("error", "absent")
_DUPLICATE_VALUE_SEMANTICS = ("reject",)


def _require_nonempty_str(value: object, message: str) -> None:
    if not isinstance(value, str) or not value:
        raise ReportAdaptationError(message)


def _require_finite(value: float, message: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ReportAdaptationError(message)


def _pairs_to_dict(pairs: Tuple[Tuple[str, str], ...], context: str) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ReportAdaptationError(f"{context} has a duplicate key: {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ReportAdapterContractDTO:
    """The stable contract governing one adapter's translation.

    `contract_id` is a self-validating content digest over every other
    field - two contracts with the same `contract_id` are guaranteed to
    declare the same adapter identity, report/subject kind, dimension
    namespace, compatibility class, and missing/duplicate-value policy.
    """

    contract_id: str
    adapter_id: str
    adapter_version: str
    report_kind: str
    subject_kind: str
    dimension_namespace: str
    compatibility_id: str
    missing_value_semantics: str = "error"
    duplicate_value_semantics: str = "reject"
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.adapter_id, "ReportAdapterContractDTO.adapter_id must be non-empty"
        )
        _require_nonempty_str(
            self.adapter_version,
            "ReportAdapterContractDTO.adapter_version must be non-empty",
        )
        _require_nonempty_str(
            self.report_kind, "ReportAdapterContractDTO.report_kind must be non-empty"
        )
        _require_nonempty_str(
            self.subject_kind, "ReportAdapterContractDTO.subject_kind must be non-empty"
        )
        _require_nonempty_str(
            self.dimension_namespace,
            "ReportAdapterContractDTO.dimension_namespace must be non-empty",
        )
        _require_nonempty_str(
            self.compatibility_id,
            "ReportAdapterContractDTO.compatibility_id must be non-empty",
        )
        if self.missing_value_semantics not in _MISSING_VALUE_SEMANTICS:
            raise ReportAdaptationError(
                f"ReportAdapterContractDTO.missing_value_semantics must be one of {_MISSING_VALUE_SEMANTICS}"
            )
        if self.duplicate_value_semantics not in _DUPLICATE_VALUE_SEMANTICS:
            raise ReportAdaptationError(
                f"ReportAdapterContractDTO.duplicate_value_semantics must be one of {_DUPLICATE_VALUE_SEMANTICS}"
            )
        expected_id = sha256_digest(
            canonical_json_bytes(report_adapter_contract_payload(self))
        )
        if self.contract_id != expected_id:
            raise ReportAdaptationError(
                "ReportAdapterContractDTO.contract_id does not match its own canonical content"
            )


def report_adapter_contract_payload(contract: ReportAdapterContractDTO) -> dict:
    return {
        "spec_version": contract.spec_version,
        "adapter_id": contract.adapter_id,
        "adapter_version": contract.adapter_version,
        "report_kind": contract.report_kind,
        "subject_kind": contract.subject_kind,
        "dimension_namespace": contract.dimension_namespace,
        "compatibility_id": contract.compatibility_id,
        "missing_value_semantics": contract.missing_value_semantics,
        "duplicate_value_semantics": contract.duplicate_value_semantics,
    }


def compute_report_adapter_contract_id(
    *,
    adapter_id: str,
    adapter_version: str,
    report_kind: str,
    subject_kind: str,
    dimension_namespace: str,
    compatibility_id: str,
    missing_value_semantics: str = "error",
    duplicate_value_semantics: str = "reject",
    spec_version: str = SPEC_VERSION,
) -> str:
    """Compute the contract_id for a not-yet-constructed contract."""
    payload = {
        "spec_version": spec_version,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "report_kind": report_kind,
        "subject_kind": subject_kind,
        "dimension_namespace": dimension_namespace,
        "compatibility_id": compatibility_id,
        "missing_value_semantics": missing_value_semantics,
        "duplicate_value_semantics": duplicate_value_semantics,
    }
    return sha256_digest(canonical_json_bytes(payload))


@dataclass(frozen=True, slots=True)
class AdaptedSubjectDTO:
    """One report subject (a sentence, a claim, a frame, ...)."""

    subject_id: str
    label: Optional[str] = None
    ordinal: Optional[int] = None
    source_ref: Optional[str] = None
    attributes: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.subject_id, "AdaptedSubjectDTO.subject_id must be non-empty"
        )

    def payload(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "label": self.label,
            "ordinal": self.ordinal,
            "source_ref": self.source_ref,
            "attributes": _pairs_to_dict(
                self.attributes, "AdaptedSubjectDTO.attributes"
            ),
        }


@dataclass(frozen=True, slots=True)
class AdaptedDimensionDTO:
    """One report dimension (metric) with an explicitly declared polarity."""

    dimension_id: str
    label: str
    score_semantics: ScoreSemantics
    family: Optional[str] = None
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    default_importance: float = 1.0
    attributes: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.dimension_id, "AdaptedDimensionDTO.dimension_id must be non-empty"
        )
        _require_nonempty_str(self.label, "AdaptedDimensionDTO.label must be non-empty")
        if not isinstance(self.score_semantics, ScoreSemantics):
            raise ReportAdaptationError(
                "AdaptedDimensionDTO.score_semantics must be a ScoreSemantics value"
            )
        if (
            self.value_min is not None
            and self.value_max is not None
            and self.value_min > self.value_max
        ):
            raise ReportAdaptationError(
                "AdaptedDimensionDTO.value_min must not exceed value_max"
            )
        if self.score_semantics == ScoreSemantics.TARGET_RANGE:
            if self.target_min is None or self.target_max is None:
                raise ReportAdaptationError(
                    "AdaptedDimensionDTO with score_semantics=target_range requires target_min and target_max"
                )
            if self.target_min > self.target_max:
                raise ReportAdaptationError(
                    "AdaptedDimensionDTO.target_min must not exceed target_max"
                )
        _require_finite(
            self.default_importance,
            "AdaptedDimensionDTO.default_importance must be a finite number",
        )
        if self.default_importance < 0:
            raise ReportAdaptationError(
                "AdaptedDimensionDTO.default_importance must be >= 0"
            )

    def payload(self) -> dict:
        return {
            "dimension_id": self.dimension_id,
            "label": self.label,
            "score_semantics": self.score_semantics.value,
            "family": self.family,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "default_importance": self.default_importance,
            "attributes": _pairs_to_dict(
                self.attributes, "AdaptedDimensionDTO.attributes"
            ),
        }


@dataclass(frozen=True, slots=True)
class ReportFindingRefDTO:
    """Identifies one finding in the source report - never an arbitrary
    mutable Python object."""

    report_id: str
    finding_id: str
    finding_kind: Optional[str] = None

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.report_id, "ReportFindingRefDTO.report_id must be non-empty"
        )
        _require_nonempty_str(
            self.finding_id, "ReportFindingRefDTO.finding_id must be non-empty"
        )

    def payload(self) -> dict:
        return {
            "report_id": self.report_id,
            "finding_id": self.finding_id,
            "finding_kind": self.finding_kind,
        }


@dataclass(frozen=True, slots=True)
class SourceBindingDTO:
    """Binds one adapted value back to its exact origin in the source report."""

    subject_id: str
    dimension_id: str
    finding_ref: ReportFindingRefDTO
    source_uri: Optional[str] = None
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    attributes: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.subject_id, "SourceBindingDTO.subject_id must be non-empty"
        )
        _require_nonempty_str(
            self.dimension_id, "SourceBindingDTO.dimension_id must be non-empty"
        )
        if self.source_start is not None and self.source_end is not None:
            if self.source_start < 0 or self.source_end < self.source_start:
                raise ReportAdaptationError(
                    "SourceBindingDTO.source_start/source_end must satisfy 0 <= source_start <= source_end"
                )

    def payload(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "dimension_id": self.dimension_id,
            "finding_ref": self.finding_ref.payload(),
            "source_uri": self.source_uri,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "attributes": _pairs_to_dict(
                self.attributes, "SourceBindingDTO.attributes"
            ),
        }


@dataclass(frozen=True, slots=True)
class AdaptedValueDTO:
    """One (subject, dimension) raw score, with its exact source binding.

    `raw_value` must be a finite number: this stage's compiler requires
    numeric dimensions (see `report_compiler.py`'s claims boundary) and
    never substitutes a missing value with zero.
    """

    subject_id: str
    dimension_id: str
    raw_value: float
    source_binding: SourceBindingDTO
    confidence: Optional[float] = None
    importance: Optional[float] = None

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.subject_id, "AdaptedValueDTO.subject_id must be non-empty"
        )
        _require_nonempty_str(
            self.dimension_id, "AdaptedValueDTO.dimension_id must be non-empty"
        )
        _require_finite(
            self.raw_value, "AdaptedValueDTO.raw_value must be a finite number"
        )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ReportAdaptationError(
                "AdaptedValueDTO.confidence must be within [0, 1]"
            )
        if self.importance is not None:
            _require_finite(
                self.importance, "AdaptedValueDTO.importance must be a finite number"
            )
            if self.importance < 0:
                raise ReportAdaptationError("AdaptedValueDTO.importance must be >= 0")
        if self.source_binding.subject_id != self.subject_id:
            raise ReportAdaptationError(
                "AdaptedValueDTO.source_binding.subject_id does not match this value's subject_id"
            )
        if self.source_binding.dimension_id != self.dimension_id:
            raise ReportAdaptationError(
                "AdaptedValueDTO.source_binding.dimension_id does not match this value's dimension_id"
            )

    def payload(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "dimension_id": self.dimension_id,
            "raw_value": self.raw_value,
            "confidence": self.confidence,
            "importance": self.importance,
            "source_binding": self.source_binding.payload(),
        }


@dataclass(frozen=True, slots=True)
class AdaptedReportDTO:
    """The neutral, adapter-produced report form `compile_report` consumes.

    `adapted_report_id` is a self-validating content digest covering every
    field below except itself - changing a raw value, a score semantic, a
    source binding, confidence, importance, or a parent relationship all
    change this id.
    """

    adapted_report_id: str
    report_id: str
    report_kind: str
    adapter_contract_id: str
    compatibility_id: str
    subjects: Tuple[AdaptedSubjectDTO, ...]
    dimensions: Tuple[AdaptedDimensionDTO, ...]
    values: Tuple[AdaptedValueDTO, ...]
    parent_report_ids: Tuple[str, ...] = field(default_factory=tuple)
    attributes: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.report_id, "AdaptedReportDTO.report_id must be non-empty"
        )
        _require_nonempty_str(
            self.report_kind, "AdaptedReportDTO.report_kind must be non-empty"
        )
        _require_nonempty_str(
            self.adapter_contract_id,
            "AdaptedReportDTO.adapter_contract_id must be non-empty",
        )
        _require_nonempty_str(
            self.compatibility_id, "AdaptedReportDTO.compatibility_id must be non-empty"
        )
        if not self.subjects:
            raise ReportAdaptationError("AdaptedReportDTO.subjects must not be empty")
        if not self.dimensions:
            raise ReportAdaptationError("AdaptedReportDTO.dimensions must not be empty")

        subject_ids = set()
        for subject in self.subjects:
            if subject.subject_id in subject_ids:
                raise ReportAdaptationError(
                    f"AdaptedReportDTO has a duplicate subject_id: {subject.subject_id!r}"
                )
            subject_ids.add(subject.subject_id)

        dimension_ids = set()
        for dimension in self.dimensions:
            if dimension.dimension_id in dimension_ids:
                raise ReportAdaptationError(
                    f"AdaptedReportDTO has a duplicate dimension_id: {dimension.dimension_id!r}"
                )
            dimension_ids.add(dimension.dimension_id)

        seen_pairs = set()
        for value in self.values:
            if value.subject_id not in subject_ids:
                raise ReportAdaptationError(
                    f"AdaptedReportDTO value references unknown subject_id: {value.subject_id!r}"
                )
            if value.dimension_id not in dimension_ids:
                raise ReportAdaptationError(
                    f"AdaptedReportDTO value references unknown dimension_id: {value.dimension_id!r}"
                )
            pair = (value.subject_id, value.dimension_id)
            if pair in seen_pairs:
                raise ReportAdaptationError(
                    f"AdaptedReportDTO has a duplicate value for subject={value.subject_id!r} "
                    f"dimension={value.dimension_id!r} - duplicate_value_semantics is always 'reject'"
                )
            seen_pairs.add(pair)

        expected_id = sha256_digest(
            canonical_json_bytes(adapted_report_signing_payload(self))
        )
        if self.adapted_report_id != expected_id:
            raise ReportAdaptationError(
                "AdaptedReportDTO.adapted_report_id does not match its own canonical content"
            )


def adapted_report_signing_payload(report: AdaptedReportDTO) -> dict:
    return {
        "spec_version": report.spec_version,
        "report_id": report.report_id,
        "report_kind": report.report_kind,
        "adapter_contract_id": report.adapter_contract_id,
        "compatibility_id": report.compatibility_id,
        "subjects": [subject.payload() for subject in report.subjects],
        "dimensions": [dimension.payload() for dimension in report.dimensions],
        "values": [value.payload() for value in report.values],
        "parent_report_ids": list(report.parent_report_ids),
        "attributes": _pairs_to_dict(report.attributes, "AdaptedReportDTO.attributes"),
    }


def compute_adapted_report_id(
    *,
    report_id: str,
    report_kind: str,
    adapter_contract_id: str,
    compatibility_id: str,
    subjects: Tuple[AdaptedSubjectDTO, ...],
    dimensions: Tuple[AdaptedDimensionDTO, ...],
    values: Tuple[AdaptedValueDTO, ...],
    parent_report_ids: Tuple[str, ...] = (),
    attributes: Tuple[Tuple[str, str], ...] = (),
    spec_version: str = SPEC_VERSION,
) -> str:
    """Compute the adapted_report_id for a not-yet-constructed AdaptedReportDTO."""
    payload = {
        "spec_version": spec_version,
        "report_id": report_id,
        "report_kind": report_kind,
        "adapter_contract_id": adapter_contract_id,
        "compatibility_id": compatibility_id,
        "subjects": [subject.payload() for subject in subjects],
        "dimensions": [dimension.payload() for dimension in dimensions],
        "values": [value.payload() for value in values],
        "parent_report_ids": list(parent_report_ids),
        "attributes": _pairs_to_dict(attributes, "AdaptedReportDTO.attributes"),
    }
    return sha256_digest(canonical_json_bytes(payload))

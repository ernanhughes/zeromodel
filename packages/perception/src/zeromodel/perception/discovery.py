"""Deterministic evidence discrepancy artifacts and unexpected-evidence discovery.

P7 materializes four inspectable evidence surfaces from the P6 expectation/conformance
contracts and preserves unexpected source fields as addressable, content-identified DTOs.
No semantic label is inferred automatically.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np
from PIL import Image

from .expectations import (
    EvidenceConformanceReportDTO,
    EvidenceExpectationDTO,
    PerceptionRegionAnnotationDTO,
)
from .fields import VPMFieldSchemaDTO
from .translator import SourceTargetTranslatorDTO

DISCOVERY_VERSION: Final = "perception-unexpected-evidence-discovery/1"
DISCREPANCY_VPM_VERSION: Final = "perception-evidence-discrepancy-vpm/1"
UNEXPLAINED_EVIDENCE_VERSION: Final = "perception-unexplained-evidence/1"

OBSERVED_SURFACE_SEMANTICS: Final = (
    "absolute_translator_coefficient_mass_normalized_per_action"
)
EXPECTED_SURFACE_SEMANTICS: Final = (
    "declared_expected_annotation_membership_by_action"
)
DIFFERENCE_SURFACE_SEMANTICS: Final = (
    "signed_observed_minus_expected_registration"
)
UNEXPLAINED_SURFACE_SEMANTICS: Final = (
    "observed_registration_outside_declared_expected_annotations"
)

_ALLOWED_SURFACES: Final = {"observed", "expected", "difference", "unexplained"}


class PerceptionDiscoveryError(ValueError):
    """Raised when discrepancy or unexpected-evidence contracts are invalid."""


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


def _png_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array, mode="L").save(
        output,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return output.getvalue()


@dataclass(frozen=True)
class EvidenceDiscrepancyVPMDTO:
    discrepancy_vpm_id: str
    translator_id: str
    field_schema_id: str
    expectation_id: str
    conformance_report_id: str
    action_label: str
    surface_kind: str
    surface_semantics: str
    width: int
    height: int
    field_values: tuple[tuple[str, float], ...]
    png_digest: str
    png_bytes: bytes
    version: str = DISCREPANCY_VPM_VERSION

    def __post_init__(self) -> None:
        if self.surface_kind not in _ALLOWED_SURFACES:
            raise PerceptionDiscoveryError("unsupported discrepancy surface kind")
        if not all(
            (
                self.discrepancy_vpm_id,
                self.translator_id,
                self.field_schema_id,
                self.expectation_id,
                self.conformance_report_id,
                self.action_label,
                self.surface_semantics,
            )
        ):
            raise PerceptionDiscoveryError("discrepancy identities must be non-empty")
        if self.width <= 0 or self.height <= 0:
            raise PerceptionDiscoveryError("discrepancy dimensions must be positive")
        ids = tuple(item[0] for item in self.field_values)
        if ids != tuple(sorted(set(ids))):
            raise PerceptionDiscoveryError("field_values must be unique and sorted")
        if any(not -1.0 <= value <= 1.0 for _, value in self.field_values):
            raise PerceptionDiscoveryError("discrepancy values must be in [-1, 1]")
        if _digest(self.png_bytes) != self.png_digest:
            raise PerceptionDiscoveryError("discrepancy PNG digest mismatch")

    def to_array(self) -> np.ndarray:
        with Image.open(io.BytesIO(self.png_bytes)) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
        if array.shape != (self.height, self.width):
            raise PerceptionDiscoveryError("discrepancy PNG shape mismatch")
        return array.copy()


@dataclass(frozen=True)
class UnexplainedEvidenceDTO:
    unexplained_id: str
    translator_id: str
    field_schema_id: str
    action_label: str
    field_ids: tuple[str, ...]
    contribution_score: float
    recurrence_count: int
    stability: float
    intervention_effect: float | None = None
    prototype_ref: str | None = None
    suggested_labels: tuple[str, ...] = ()
    version: str = UNEXPLAINED_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if not all((self.unexplained_id, self.translator_id, self.field_schema_id, self.action_label)):
            raise PerceptionDiscoveryError("unexplained evidence identities must be non-empty")
        if not self.field_ids or self.field_ids != tuple(sorted(set(self.field_ids))):
            raise PerceptionDiscoveryError("unexplained field_ids must be non-empty, unique, sorted")
        if not 0.0 <= self.contribution_score <= 1.0:
            raise PerceptionDiscoveryError("contribution_score must be in [0, 1]")
        if self.recurrence_count <= 0:
            raise PerceptionDiscoveryError("recurrence_count must be positive")
        if not 0.0 <= self.stability <= 1.0:
            raise PerceptionDiscoveryError("stability must be in [0, 1]")
        if self.intervention_effect is not None and not -1.0 <= self.intervention_effect <= 1.0:
            raise PerceptionDiscoveryError("intervention_effect must be in [-1, 1]")
        if self.suggested_labels != tuple(sorted(set(self.suggested_labels))):
            raise PerceptionDiscoveryError("suggested_labels must be unique and sorted")


@dataclass(frozen=True)
class EvidenceDiscoveryReportDTO:
    discovery_id: str
    translator_id: str
    field_schema_id: str
    expectation_id: str
    conformance_report_id: str
    surfaces: tuple[EvidenceDiscrepancyVPMDTO, ...]
    unexplained_evidence: tuple[UnexplainedEvidenceDTO, ...]
    contribution_threshold: float
    version: str = DISCOVERY_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.discovery_id,
                self.translator_id,
                self.field_schema_id,
                self.expectation_id,
                self.conformance_report_id,
            )
        ):
            raise PerceptionDiscoveryError("discovery identities must be non-empty")
        if not 0.0 <= self.contribution_threshold <= 1.0:
            raise PerceptionDiscoveryError("contribution_threshold must be in [0, 1]")
        keys = tuple((item.action_label, item.surface_kind) for item in self.surfaces)
        if keys != tuple(sorted(set(keys))):
            raise PerceptionDiscoveryError("surfaces must be unique and sorted")
        unexplained_ids = tuple(item.unexplained_id for item in self.unexplained_evidence)
        if unexplained_ids != tuple(sorted(set(unexplained_ids))):
            raise PerceptionDiscoveryError("unexplained evidence must be unique and sorted")


def _coefficient_surface(
    translator: SourceTargetTranslatorDTO,
    action_label: str,
) -> dict[str, float]:
    action_index = translator.action_labels.index(action_label)
    weights = np.abs(np.asarray(translator.coefficients[action_index], dtype=np.float64))
    total = float(np.sum(weights))
    if total <= 0.0:
        return {field_id: 0.0 for field_id in translator.source_field_ids}
    return {
        field_id: float(weights[index] / total)
        for index, field_id in enumerate(translator.source_field_ids)
    }


def _render_surface(
    field_schema: VPMFieldSchemaDTO,
    values: Mapping[str, float],
    *,
    signed: bool,
) -> tuple[bytes, tuple[tuple[str, float], ...]]:
    array = np.zeros((field_schema.height, field_schema.width), dtype=np.uint8)
    ordered = tuple(sorted((field_id, float(value)) for field_id, value in values.items()))
    for field_id, value in ordered:
        field = field_schema.field(field_id)
        if signed:
            rendered = np.uint8(round((value + 1.0) * 127.5))
        else:
            rendered = np.uint8(round(max(0.0, min(1.0, value)) * 255.0))
        array[field.y0 : field.y1, field.x0 : field.x1] = rendered
    return _png_bytes(array), ordered


def _surface_dto(
    translator: SourceTargetTranslatorDTO,
    field_schema: VPMFieldSchemaDTO,
    expectation: EvidenceExpectationDTO,
    conformance: EvidenceConformanceReportDTO,
    action_label: str,
    kind: str,
    semantics: str,
    values: Mapping[str, float],
    *,
    signed: bool = False,
) -> EvidenceDiscrepancyVPMDTO:
    png, ordered = _render_surface(field_schema, values, signed=signed)
    png_digest = _digest(png)
    payload: Mapping[str, object] = {
        "action_label": action_label,
        "conformance_report_id": conformance.report_id,
        "expectation_id": expectation.expectation_id,
        "field_schema_id": field_schema.field_schema_id,
        "field_values": [list(item) for item in ordered],
        "png_digest": png_digest,
        "surface_kind": kind,
        "surface_semantics": semantics,
        "translator_id": translator.translator_id,
        "version": DISCREPANCY_VPM_VERSION,
    }
    return EvidenceDiscrepancyVPMDTO(
        discrepancy_vpm_id=_digest(_canonical_json(payload)),
        translator_id=translator.translator_id,
        field_schema_id=field_schema.field_schema_id,
        expectation_id=expectation.expectation_id,
        conformance_report_id=conformance.report_id,
        action_label=action_label,
        surface_kind=kind,
        surface_semantics=semantics,
        width=field_schema.width,
        height=field_schema.height,
        field_values=ordered,
        png_digest=png_digest,
        png_bytes=png,
    )


def discover_unexpected_evidence(
    translator: SourceTargetTranslatorDTO,
    field_schema: VPMFieldSchemaDTO,
    annotations: tuple[PerceptionRegionAnnotationDTO, ...],
    expectation: EvidenceExpectationDTO,
    conformance: EvidenceConformanceReportDTO,
    *,
    contribution_threshold: float = 0.05,
    recurrence_by_action_field: Mapping[tuple[str, str], int] | None = None,
    stability_by_action_field: Mapping[tuple[str, str], float] | None = None,
    intervention_effect_by_action_field: Mapping[tuple[str, str], float] | None = None,
) -> EvidenceDiscoveryReportDTO:
    """Materialize discrepancy VPMs and addressable unexpected evidence fields."""

    if translator.source_field_schema_id != field_schema.field_schema_id:
        raise PerceptionDiscoveryError("field schema does not match translator")
    if expectation.field_schema_id != field_schema.field_schema_id:
        raise PerceptionDiscoveryError("expectation field schema mismatch")
    if conformance.translator_id != translator.translator_id:
        raise PerceptionDiscoveryError("conformance report translator mismatch")
    if conformance.expectation_id != expectation.expectation_id:
        raise PerceptionDiscoveryError("conformance report expectation mismatch")
    if not 0.0 <= contribution_threshold <= 1.0:
        raise PerceptionDiscoveryError("contribution_threshold must be in [0, 1]")

    annotation_by_id = {item.annotation_id: item for item in annotations}
    expected_annotation_ids = set(expectation.source_annotation_ids)
    unknown = expected_annotation_ids - set(annotation_by_id)
    if unknown:
        raise PerceptionDiscoveryError(f"expectation references unknown annotations: {sorted(unknown)}")
    expected_fields = {
        field_id
        for annotation_id in expected_annotation_ids
        for field_id in annotation_by_id[annotation_id].field_ids
    }

    surfaces: list[EvidenceDiscrepancyVPMDTO] = []
    unexplained_items: list[UnexplainedEvidenceDTO] = []

    for action_label in translator.action_labels:
        observed = _coefficient_surface(translator, action_label)
        action_expected = action_label in expectation.expected_action_labels
        expected = {
            field_id: (1.0 if action_expected and field_id in expected_fields else 0.0)
            for field_id in translator.source_field_ids
        }
        difference = {
            field_id: observed[field_id] - expected[field_id]
            for field_id in translator.source_field_ids
        }
        unexplained = {
            field_id: (observed[field_id] if field_id not in expected_fields else 0.0)
            for field_id in translator.source_field_ids
        }

        surfaces.extend(
            (
                _surface_dto(
                    translator,
                    field_schema,
                    expectation,
                    conformance,
                    action_label,
                    "observed",
                    OBSERVED_SURFACE_SEMANTICS,
                    observed,
                ),
                _surface_dto(
                    translator,
                    field_schema,
                    expectation,
                    conformance,
                    action_label,
                    "expected",
                    EXPECTED_SURFACE_SEMANTICS,
                    expected,
                ),
                _surface_dto(
                    translator,
                    field_schema,
                    expectation,
                    conformance,
                    action_label,
                    "difference",
                    DIFFERENCE_SURFACE_SEMANTICS,
                    difference,
                    signed=True,
                ),
                _surface_dto(
                    translator,
                    field_schema,
                    expectation,
                    conformance,
                    action_label,
                    "unexplained",
                    UNEXPLAINED_SURFACE_SEMANTICS,
                    unexplained,
                ),
            )
        )

        for field_id, contribution in sorted(unexplained.items()):
            if contribution < contribution_threshold:
                continue
            key = (action_label, field_id)
            recurrence = 1 if recurrence_by_action_field is None else recurrence_by_action_field.get(key, 1)
            stability = 1.0 if stability_by_action_field is None else stability_by_action_field.get(key, 1.0)
            effect = None
            if intervention_effect_by_action_field is not None:
                effect = intervention_effect_by_action_field.get(key)
            payload: Mapping[str, object] = {
                "action_label": action_label,
                "contribution_score": contribution,
                "field_ids": [field_id],
                "field_schema_id": field_schema.field_schema_id,
                "intervention_effect": effect,
                "recurrence_count": recurrence,
                "stability": stability,
                "translator_id": translator.translator_id,
                "version": UNEXPLAINED_EVIDENCE_VERSION,
            }
            unexplained_items.append(
                UnexplainedEvidenceDTO(
                    unexplained_id=_digest(_canonical_json(payload)),
                    translator_id=translator.translator_id,
                    field_schema_id=field_schema.field_schema_id,
                    action_label=action_label,
                    field_ids=(field_id,),
                    contribution_score=contribution,
                    recurrence_count=recurrence,
                    stability=stability,
                    intervention_effect=effect,
                )
            )

    ordered_surfaces = tuple(sorted(surfaces, key=lambda item: (item.action_label, item.surface_kind)))
    ordered_unexplained = tuple(sorted(unexplained_items, key=lambda item: item.unexplained_id))
    payload = {
        "conformance_report_id": conformance.report_id,
        "contribution_threshold": contribution_threshold,
        "expectation_id": expectation.expectation_id,
        "field_schema_id": field_schema.field_schema_id,
        "surface_ids": [item.discrepancy_vpm_id for item in ordered_surfaces],
        "translator_id": translator.translator_id,
        "unexplained_ids": [item.unexplained_id for item in ordered_unexplained],
        "version": DISCOVERY_VERSION,
    }
    return EvidenceDiscoveryReportDTO(
        discovery_id=_digest(_canonical_json(payload)),
        translator_id=translator.translator_id,
        field_schema_id=field_schema.field_schema_id,
        expectation_id=expectation.expectation_id,
        conformance_report_id=conformance.report_id,
        surfaces=ordered_surfaces,
        unexplained_evidence=ordered_unexplained,
        contribution_threshold=contribution_threshold,
    )

"""Deterministic before/after transition evidence maps for Stage P18A.

P18A compares two exact Source VPMs under one P4A field schema and materializes
an addressable fieldwise change surface. Optional P6 region annotations are bound
to the measured fields, but labels do not alter the measurements and no causal
claim is inferred automatically.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np
from PIL import Image

from .expectations import PerceptionRegionAnnotationDTO
from .fields import PerceptionFieldError, VPMFieldSchemaDTO, validate_source_for_schema
from .representation import SourceVPMDTO

TRANSITION_FIELD_EVIDENCE_VERSION: Final = "perception-transition-field-evidence/1"
TRANSITION_EVIDENCE_VPM_VERSION: Final = "perception-transition-evidence-vpm/1"
TRANSITION_CHANGE_SEMANTICS: Final = "mean_absolute_uint8_delta_divided_by_255"
TRANSITION_SIGNED_CHANGE_SEMANTICS: Final = "mean_signed_int16_delta_divided_by_255"
TRANSITION_CHANGED_FRACTION_SEMANTICS: Final = (
    "fraction_of_field_values_with_absolute_delta_at_least_change_threshold"
)
TRANSITION_RENDER_SEMANTICS: Final = (
    "rounded_uint8_field_mean_absolute_change_max_over_channels"
)


class PerceptionTransitionEvidenceError(ValueError):
    """Raised when transition evidence cannot be measured canonically."""


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


def _canonical_source_array(source: SourceVPMDTO) -> np.ndarray:
    array = source.to_array()
    if source.channels == 1:
        return array.reshape(source.height, source.width, 1)
    return array.reshape(source.height, source.width, source.channels)


@dataclass(frozen=True)
class TransitionFieldEvidenceDTO:
    """Measured change for one exact field between two Source VPMs."""

    field_id: str
    before_mean: float
    after_mean: float
    mean_absolute_change: float
    mean_signed_change: float
    changed_fraction: float
    changed_value_count: int
    total_value_count: int
    annotation_ids: tuple[str, ...] = ()
    change_semantics: str = TRANSITION_CHANGE_SEMANTICS
    signed_change_semantics: str = TRANSITION_SIGNED_CHANGE_SEMANTICS
    changed_fraction_semantics: str = TRANSITION_CHANGED_FRACTION_SEMANTICS
    version: str = TRANSITION_FIELD_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if not self.field_id:
            raise PerceptionTransitionEvidenceError("field_id must be non-empty")
        for name, value in (
            ("before_mean", self.before_mean),
            ("after_mean", self.after_mean),
            ("mean_absolute_change", self.mean_absolute_change),
            ("changed_fraction", self.changed_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise PerceptionTransitionEvidenceError(f"{name} must be in [0, 1]")
        if not -1.0 <= self.mean_signed_change <= 1.0:
            raise PerceptionTransitionEvidenceError(
                "mean_signed_change must be in [-1, 1]"
            )
        if self.total_value_count <= 0:
            raise PerceptionTransitionEvidenceError(
                "total_value_count must be positive"
            )
        if not 0 <= self.changed_value_count <= self.total_value_count:
            raise PerceptionTransitionEvidenceError(
                "changed_value_count must be within total_value_count"
            )
        expected_fraction = self.changed_value_count / self.total_value_count
        if abs(self.changed_fraction - expected_fraction) > 1e-12:
            raise PerceptionTransitionEvidenceError(
                "changed_fraction disagrees with measured counts"
            )
        if self.annotation_ids != tuple(sorted(set(self.annotation_ids))):
            raise PerceptionTransitionEvidenceError(
                "annotation_ids must be unique and sorted"
            )
        if self.change_semantics != TRANSITION_CHANGE_SEMANTICS:
            raise PerceptionTransitionEvidenceError(
                "unsupported transition change semantics"
            )
        if self.signed_change_semantics != TRANSITION_SIGNED_CHANGE_SEMANTICS:
            raise PerceptionTransitionEvidenceError(
                "unsupported signed transition semantics"
            )
        if self.changed_fraction_semantics != TRANSITION_CHANGED_FRACTION_SEMANTICS:
            raise PerceptionTransitionEvidenceError(
                "unsupported changed-fraction semantics"
            )
        if self.version != TRANSITION_FIELD_EVIDENCE_VERSION:
            raise PerceptionTransitionEvidenceError(
                "unsupported transition field evidence version"
            )

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "after_mean": self.after_mean,
            "annotation_ids": list(self.annotation_ids),
            "before_mean": self.before_mean,
            "changed_fraction": self.changed_fraction,
            "changed_fraction_semantics": self.changed_fraction_semantics,
            "changed_value_count": self.changed_value_count,
            "change_semantics": self.change_semantics,
            "field_id": self.field_id,
            "mean_absolute_change": self.mean_absolute_change,
            "mean_signed_change": self.mean_signed_change,
            "signed_change_semantics": self.signed_change_semantics,
            "total_value_count": self.total_value_count,
            "version": self.version,
        }


@dataclass(frozen=True)
class TransitionEvidenceVPMDTO:
    """Content-addressed fieldwise change surface between exact observations."""

    transition_evidence_id: str
    before_source_vpm_id: str
    after_source_vpm_id: str
    before_pixel_digest: str
    after_pixel_digest: str
    field_schema_id: str
    source_encoder_spec_id: str
    change_threshold: int
    width: int
    height: int
    fields: tuple[TransitionFieldEvidenceDTO, ...]
    annotation_ids: tuple[str, ...]
    png_digest: str
    png_bytes: bytes
    change_semantics: str = TRANSITION_CHANGE_SEMANTICS
    signed_change_semantics: str = TRANSITION_SIGNED_CHANGE_SEMANTICS
    changed_fraction_semantics: str = TRANSITION_CHANGED_FRACTION_SEMANTICS
    render_semantics: str = TRANSITION_RENDER_SEMANTICS
    version: str = TRANSITION_EVIDENCE_VPM_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.transition_evidence_id,
                self.before_source_vpm_id,
                self.after_source_vpm_id,
                self.before_pixel_digest,
                self.after_pixel_digest,
                self.field_schema_id,
                self.source_encoder_spec_id,
            )
        ):
            raise PerceptionTransitionEvidenceError(
                "transition evidence identities must be non-empty"
            )
        if (
            isinstance(self.change_threshold, bool)
            or not isinstance(self.change_threshold, int)
            or not 1 <= self.change_threshold <= 255
        ):
            raise PerceptionTransitionEvidenceError(
                "change_threshold must be an integer in [1, 255]"
            )
        if self.width <= 0 or self.height <= 0:
            raise PerceptionTransitionEvidenceError(
                "transition evidence dimensions must be positive"
            )
        field_ids = tuple(item.field_id for item in self.fields)
        if not field_ids or field_ids != tuple(sorted(set(field_ids))):
            raise PerceptionTransitionEvidenceError(
                "transition fields must be non-empty, unique, and sorted"
            )
        if self.annotation_ids != tuple(sorted(set(self.annotation_ids))):
            raise PerceptionTransitionEvidenceError(
                "annotation_ids must be unique and sorted"
            )
        bound_annotation_ids = {
            annotation_id
            for item in self.fields
            for annotation_id in item.annotation_ids
        }
        if bound_annotation_ids != set(self.annotation_ids):
            raise PerceptionTransitionEvidenceError(
                "top-level annotation identities must match field bindings"
            )
        if self.change_semantics != TRANSITION_CHANGE_SEMANTICS:
            raise PerceptionTransitionEvidenceError(
                "unsupported transition change semantics"
            )
        if self.signed_change_semantics != TRANSITION_SIGNED_CHANGE_SEMANTICS:
            raise PerceptionTransitionEvidenceError(
                "unsupported signed transition semantics"
            )
        if self.changed_fraction_semantics != TRANSITION_CHANGED_FRACTION_SEMANTICS:
            raise PerceptionTransitionEvidenceError(
                "unsupported changed-fraction semantics"
            )
        if self.render_semantics != TRANSITION_RENDER_SEMANTICS:
            raise PerceptionTransitionEvidenceError(
                "unsupported transition render semantics"
            )
        if self.version != TRANSITION_EVIDENCE_VPM_VERSION:
            raise PerceptionTransitionEvidenceError(
                "unsupported transition evidence version"
            )
        if _digest(self.png_bytes) != self.png_digest:
            raise PerceptionTransitionEvidenceError(
                "transition evidence PNG digest mismatch"
            )
        expected_id = _digest(_canonical_json(self.canonical_payload()))
        if self.transition_evidence_id != expected_id:
            raise PerceptionTransitionEvidenceError(
                "transition evidence identity disagrees with canonical payload"
            )

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "after_pixel_digest": self.after_pixel_digest,
            "after_source_vpm_id": self.after_source_vpm_id,
            "annotation_ids": list(self.annotation_ids),
            "before_pixel_digest": self.before_pixel_digest,
            "before_source_vpm_id": self.before_source_vpm_id,
            "change_semantics": self.change_semantics,
            "change_threshold": self.change_threshold,
            "changed_fraction_semantics": self.changed_fraction_semantics,
            "field_schema_id": self.field_schema_id,
            "fields": [item.canonical_payload() for item in self.fields],
            "height": self.height,
            "png_digest": self.png_digest,
            "render_semantics": self.render_semantics,
            "signed_change_semantics": self.signed_change_semantics,
            "source_encoder_spec_id": self.source_encoder_spec_id,
            "version": self.version,
            "width": self.width,
        }

    def to_array(self) -> np.ndarray:
        with Image.open(io.BytesIO(self.png_bytes)) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
        if array.shape != (self.height, self.width):
            raise PerceptionTransitionEvidenceError(
                "transition evidence PNG shape mismatch"
            )
        return array.copy()

    def field_evidence(self, field_id: str) -> TransitionFieldEvidenceDTO:
        for item in self.fields:
            if item.field_id == field_id:
                return item
        raise KeyError(field_id)

    def changed_field_ids(
        self,
        *,
        minimum_mean_absolute_change: float = 0.0,
        minimum_changed_fraction: float = 0.0,
    ) -> tuple[str, ...]:
        if not 0.0 <= minimum_mean_absolute_change <= 1.0:
            raise PerceptionTransitionEvidenceError(
                "minimum_mean_absolute_change must be in [0, 1]"
            )
        if not 0.0 <= minimum_changed_fraction <= 1.0:
            raise PerceptionTransitionEvidenceError(
                "minimum_changed_fraction must be in [0, 1]"
            )
        return tuple(
            item.field_id
            for item in self.fields
            if item.mean_absolute_change >= minimum_mean_absolute_change
            and item.changed_fraction >= minimum_changed_fraction
            and item.changed_value_count > 0
        )


def _validate_annotations(
    annotations: tuple[PerceptionRegionAnnotationDTO, ...],
    field_schema: VPMFieldSchemaDTO,
) -> dict[str, tuple[str, ...]]:
    annotation_by_id = {item.annotation_id: item for item in annotations}
    if len(annotation_by_id) != len(annotations):
        raise PerceptionTransitionEvidenceError(
            "annotations must have unique identities"
        )
    known_fields = {item.field_id for item in field_schema.fields}
    annotation_ids_by_field: dict[str, list[str]] = {
        field_id: [] for field_id in known_fields
    }
    for annotation in annotations:
        if annotation.field_schema_id != field_schema.field_schema_id:
            raise PerceptionTransitionEvidenceError(
                "annotation field schema does not match transition schema"
            )
        unknown = set(annotation.field_ids) - known_fields
        if unknown:
            raise PerceptionTransitionEvidenceError(
                f"annotation contains unknown fields: {sorted(unknown)}"
            )
        for field_id in annotation.field_ids:
            annotation_ids_by_field[field_id].append(annotation.annotation_id)
    return {
        field_id: tuple(sorted(annotation_ids))
        for field_id, annotation_ids in annotation_ids_by_field.items()
    }


def build_transition_evidence_vpm(
    before: SourceVPMDTO,
    after: SourceVPMDTO,
    field_schema: VPMFieldSchemaDTO,
    *,
    annotations: tuple[PerceptionRegionAnnotationDTO, ...] = (),
    change_threshold: int = 1,
) -> TransitionEvidenceVPMDTO:
    """Measure and render exact fieldwise change between two Source VPMs."""

    if (
        isinstance(change_threshold, bool)
        or not isinstance(change_threshold, int)
        or not 1 <= change_threshold <= 255
    ):
        raise PerceptionTransitionEvidenceError(
            "change_threshold must be an integer in [1, 255]"
        )
    try:
        validate_source_for_schema(before, field_schema)
        validate_source_for_schema(after, field_schema)
    except PerceptionFieldError as exc:
        raise PerceptionTransitionEvidenceError(str(exc)) from exc
    annotation_ids_by_field = _validate_annotations(annotations, field_schema)

    before_array = _canonical_source_array(before).astype(np.int16)
    after_array = _canonical_source_array(after).astype(np.int16)
    rendered = np.zeros((field_schema.height, field_schema.width), dtype=np.uint8)
    evidence: list[TransitionFieldEvidenceDTO] = []

    for field in field_schema.fields:
        region = np.s_[
            field.y0 : field.y1,
            field.x0 : field.x1,
            field.channel_start : field.channel_end,
        ]
        before_values = before_array[region]
        after_values = after_array[region]
        delta = after_values - before_values
        absolute_delta = np.abs(delta)
        total_value_count = int(delta.size)
        changed_value_count = int(np.count_nonzero(absolute_delta >= change_threshold))
        mean_absolute_change = float(np.mean(absolute_delta)) / 255.0
        item = TransitionFieldEvidenceDTO(
            field_id=field.field_id,
            before_mean=float(np.mean(before_values)) / 255.0,
            after_mean=float(np.mean(after_values)) / 255.0,
            mean_absolute_change=mean_absolute_change,
            mean_signed_change=float(np.mean(delta)) / 255.0,
            changed_fraction=changed_value_count / total_value_count,
            changed_value_count=changed_value_count,
            total_value_count=total_value_count,
            annotation_ids=annotation_ids_by_field[field.field_id],
        )
        evidence.append(item)
        rendered_value = np.uint8(round(mean_absolute_change * 255.0))
        rendered_region = rendered[field.y0 : field.y1, field.x0 : field.x1]
        np.maximum(rendered_region, rendered_value, out=rendered_region)

    ordered_fields = tuple(sorted(evidence, key=lambda item: item.field_id))
    annotation_ids = tuple(sorted(item.annotation_id for item in annotations))
    png = _png_bytes(rendered)
    png_digest = _digest(png)
    payload: Mapping[str, object] = {
        "after_pixel_digest": after.pixel_digest,
        "after_source_vpm_id": after.source_vpm_id,
        "annotation_ids": list(annotation_ids),
        "before_pixel_digest": before.pixel_digest,
        "before_source_vpm_id": before.source_vpm_id,
        "change_semantics": TRANSITION_CHANGE_SEMANTICS,
        "change_threshold": change_threshold,
        "changed_fraction_semantics": TRANSITION_CHANGED_FRACTION_SEMANTICS,
        "field_schema_id": field_schema.field_schema_id,
        "fields": [item.canonical_payload() for item in ordered_fields],
        "height": field_schema.height,
        "png_digest": png_digest,
        "render_semantics": TRANSITION_RENDER_SEMANTICS,
        "signed_change_semantics": TRANSITION_SIGNED_CHANGE_SEMANTICS,
        "source_encoder_spec_id": field_schema.source_encoder_spec_id,
        "version": TRANSITION_EVIDENCE_VPM_VERSION,
        "width": field_schema.width,
    }
    return TransitionEvidenceVPMDTO(
        transition_evidence_id=_digest(_canonical_json(payload)),
        before_source_vpm_id=before.source_vpm_id,
        after_source_vpm_id=after.source_vpm_id,
        before_pixel_digest=before.pixel_digest,
        after_pixel_digest=after.pixel_digest,
        field_schema_id=field_schema.field_schema_id,
        source_encoder_spec_id=field_schema.source_encoder_spec_id,
        change_threshold=change_threshold,
        width=field_schema.width,
        height=field_schema.height,
        fields=ordered_fields,
        annotation_ids=annotation_ids,
        png_digest=png_digest,
        png_bytes=png,
    )

"""Measured field relevance and deterministic Evidence VPMs for Stage P4B.

P4B estimates predictive association only. Scores are not causal claims; keep/remove
intervention evaluation and weighted inference belong to P4C.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np
from PIL import Image

from .dataset import PerceptionDatasetManifestDTO
from .fields import VPMFieldSchemaDTO, extract_source_fields, validate_source_for_schema
from .representation import SourceVPMDTO

FIELD_RELEVANCE_VERSION: Final = "perception-field-relevance/1"
EVIDENCE_VPM_VERSION: Final = "perception-evidence-vpm/1"
FIELD_RELEVANCE_SEMANTICS: Final = "eta_squared_of_field_mean_by_action"
EVIDENCE_RENDER_SEMANTICS: Final = "rounded_uint8_field_relevance_max_over_channels"


class PerceptionEvidenceError(ValueError):
    """Raised when field relevance or Evidence VPM materialization is invalid."""


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
class FieldRelevanceDTO:
    field_id: str
    score: float
    score_semantics: str
    support_count: int
    action_count: int
    within_action_variance: float
    between_action_variance: float
    version: str = FIELD_RELEVANCE_VERSION

    def __post_init__(self) -> None:
        if not self.field_id:
            raise PerceptionEvidenceError("field_id must be non-empty")
        if self.score_semantics != FIELD_RELEVANCE_SEMANTICS:
            raise PerceptionEvidenceError("unsupported field relevance semantics")
        if not 0.0 <= self.score <= 1.0:
            raise PerceptionEvidenceError("field relevance score must be in [0, 1]")
        if self.support_count <= 0 or self.action_count <= 0:
            raise PerceptionEvidenceError("relevance support counts must be positive")
        if self.within_action_variance < 0.0 or self.between_action_variance < 0.0:
            raise PerceptionEvidenceError("variance components must be non-negative")


@dataclass(frozen=True)
class EvidenceVPMDTO:
    evidence_vpm_id: str
    dataset_id: str
    field_schema_id: str
    source_encoder_spec_id: str
    training_split: str
    width: int
    height: int
    score_semantics: str
    render_semantics: str
    relevances: tuple[FieldRelevanceDTO, ...]
    png_digest: str
    png_bytes: bytes
    version: str = EVIDENCE_VPM_VERSION

    def __post_init__(self) -> None:
        if not all((self.evidence_vpm_id, self.dataset_id, self.field_schema_id)):
            raise PerceptionEvidenceError("Evidence VPM identities must be non-empty")
        if self.width <= 0 or self.height <= 0:
            raise PerceptionEvidenceError("Evidence VPM dimensions must be positive")
        ids = tuple(item.field_id for item in self.relevances)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise PerceptionEvidenceError("relevances must be unique and sorted")
        if _digest(self.png_bytes) != self.png_digest:
            raise PerceptionEvidenceError("Evidence VPM PNG digest mismatch")

    def to_array(self) -> np.ndarray:
        with Image.open(io.BytesIO(self.png_bytes)) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
        if array.shape != (self.height, self.width):
            raise PerceptionEvidenceError("Evidence VPM PNG shape mismatch")
        return array.copy()

    def relevance_for(self, field_id: str) -> FieldRelevanceDTO:
        for item in self.relevances:
            if item.field_id == field_id:
                return item
        raise KeyError(field_id)


def _eta_squared(values: np.ndarray, labels: tuple[str, ...]) -> tuple[float, float, float]:
    grand_mean = float(np.mean(values))
    total = float(np.sum((values - grand_mean) ** 2))
    within = 0.0
    between = 0.0
    for label in sorted(set(labels)):
        group = values[np.asarray([item == label for item in labels], dtype=bool)]
        group_mean = float(np.mean(group))
        within += float(np.sum((group - group_mean) ** 2))
        between += float(group.size * ((group_mean - grand_mean) ** 2))
    if total <= 0.0:
        return 0.0, within, between
    score = min(1.0, max(0.0, between / total))
    return score, within, between


def estimate_field_relevance(
    manifest: PerceptionDatasetManifestDTO,
    source_vpms: Mapping[str, SourceVPMDTO],
    field_schema: VPMFieldSchemaDTO,
    *,
    training_split: str = "train",
) -> EvidenceVPMDTO:
    """Estimate action association from per-field mean intensity and render evidence."""

    if training_split not in {"train", "validation", "test", "all"}:
        raise PerceptionEvidenceError(
            "training_split must be train, validation, test, or all"
        )
    selected_ids = {
        item.interaction_id
        for item in manifest.split_assignments
        if training_split == "all" or item.split == training_split
    }
    interactions = tuple(
        item for item in manifest.interactions if item.interaction_id in selected_ids
    )
    if not interactions:
        raise PerceptionEvidenceError(
            f"dataset contains no {training_split!r} interactions"
        )
    labels = tuple(item.action_label for item in interactions)
    action_count = len(set(labels))
    if action_count < 2:
        raise PerceptionEvidenceError("field relevance requires at least two actions")

    values_by_field: dict[str, list[float]] = {
        field.field_id: [] for field in field_schema.fields
    }
    for interaction in interactions:
        try:
            source = source_vpms[interaction.source_vpm_id]
        except KeyError as exc:
            raise PerceptionEvidenceError(
                f"missing SourceVPMDTO for {interaction.source_vpm_id}"
            ) from exc
        if source.pixel_digest != interaction.source_pixel_digest:
            raise PerceptionEvidenceError("source pixel identity disagrees with interaction")
        validate_source_for_schema(source, field_schema)
        for sample in extract_source_fields(source, field_schema):
            values_by_field[sample.field_id].append(float(np.mean(sample.to_array())) / 255.0)

    relevances: list[FieldRelevanceDTO] = []
    for field in field_schema.fields:
        values = np.asarray(values_by_field[field.field_id], dtype=np.float64)
        score, within, between = _eta_squared(values, labels)
        relevances.append(
            FieldRelevanceDTO(
                field_id=field.field_id,
                score=score,
                score_semantics=FIELD_RELEVANCE_SEMANTICS,
                support_count=len(interactions),
                action_count=action_count,
                within_action_variance=within,
                between_action_variance=between,
            )
        )
    ordered = tuple(sorted(relevances, key=lambda item: item.field_id))
    score_by_id = {item.field_id: item.score for item in ordered}
    evidence = np.zeros((field_schema.height, field_schema.width), dtype=np.uint8)
    for field in field_schema.fields:
        rendered = np.uint8(round(score_by_id[field.field_id] * 255.0))
        region = evidence[field.y0 : field.y1, field.x0 : field.x1]
        np.maximum(region, rendered, out=region)
    png_bytes = _png_bytes(evidence)
    png_digest = _digest(png_bytes)
    payload: Mapping[str, object] = {
        "dataset_id": manifest.dataset_id,
        "field_schema_id": field_schema.field_schema_id,
        "png_digest": png_digest,
        "relevances": [
            {
                "action_count": item.action_count,
                "between_action_variance": item.between_action_variance,
                "field_id": item.field_id,
                "score": item.score,
                "score_semantics": item.score_semantics,
                "support_count": item.support_count,
                "version": item.version,
                "within_action_variance": item.within_action_variance,
            }
            for item in ordered
        ],
        "render_semantics": EVIDENCE_RENDER_SEMANTICS,
        "score_semantics": FIELD_RELEVANCE_SEMANTICS,
        "source_encoder_spec_id": field_schema.source_encoder_spec_id,
        "training_split": training_split,
        "version": EVIDENCE_VPM_VERSION,
    }
    return EvidenceVPMDTO(
        evidence_vpm_id=_digest(_canonical_json(payload)),
        dataset_id=manifest.dataset_id,
        field_schema_id=field_schema.field_schema_id,
        source_encoder_spec_id=field_schema.source_encoder_spec_id,
        training_split=training_split,
        width=field_schema.width,
        height=field_schema.height,
        score_semantics=FIELD_RELEVANCE_SEMANTICS,
        render_semantics=EVIDENCE_RENDER_SEMANTICS,
        relevances=ordered,
        png_digest=png_digest,
        png_bytes=png_bytes,
    )

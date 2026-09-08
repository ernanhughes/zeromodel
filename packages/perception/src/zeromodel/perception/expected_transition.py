"""Structured expected-transition memory for action-conditioned futures.

An ExpectedTransitionVPM is a unit of compiled memory: what a candidate action
should make happen, expressed per field as what changes, where, in what
direction, by approximately how much, how stably, and with how much support.
It is a visualization of structured change, never a synthesized future frame.

Language is deliberately non-causal: projected / predicted / expected /
supported / contradicted / out-of-distribution. No causal understanding,
general world modelling, or safety is claimed.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np
from PIL import Image

from .transition_evidence import (
    TRANSITION_CHANGED_FRACTION_SEMANTICS,
    TRANSITION_CHANGE_SEMANTICS,
    TRANSITION_SIGNED_CHANGE_SEMANTICS,
)

EXPECTED_TRANSITION_FIELD_VERSION: Final = "perception-expected-transition-field/1"
EXPECTED_TRANSITION_VPM_VERSION: Final = "perception-expected-transition-vpm/1"
EXPECTED_TRANSITION_RENDER_SEMANTICS: Final = (
    "rounded_uint8_expected_mean_absolute_change_max_over_channels"
)
EXPECTED_TRANSITION_STATUSES: Final = frozenset(
    {
        "supported",
        "insufficient_examples",
        "out_of_distribution",
        "ambiguous_future",
        "unsupported_action",
    }
)


class PerceptionExpectedTransitionError(ValueError):
    """Raised when an expected transition cannot be represented canonically."""


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
class ExpectedTransitionFieldDTO:
    """Projected per-field change for one candidate action from one source."""

    field_id: str
    expected_after_mean: float
    expected_mean_signed_change: float
    expected_mean_absolute_change: float
    expected_changed_fraction: float
    signed_change_dispersion: float
    absolute_change_dispersion: float
    support_count: int
    change_semantics: str = TRANSITION_CHANGE_SEMANTICS
    signed_change_semantics: str = TRANSITION_SIGNED_CHANGE_SEMANTICS
    changed_fraction_semantics: str = TRANSITION_CHANGED_FRACTION_SEMANTICS
    version: str = EXPECTED_TRANSITION_FIELD_VERSION

    def __post_init__(self) -> None:
        if not self.field_id:
            raise PerceptionExpectedTransitionError("field_id must be non-empty")
        for name, value in (
            ("expected_after_mean", self.expected_after_mean),
            ("expected_mean_absolute_change", self.expected_mean_absolute_change),
            ("expected_changed_fraction", self.expected_changed_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise PerceptionExpectedTransitionError(f"{name} must be in [0, 1]")
        if not -1.0 <= self.expected_mean_signed_change <= 1.0:
            raise PerceptionExpectedTransitionError(
                "expected_mean_signed_change must be in [-1, 1]"
            )
        for name, value in (
            ("signed_change_dispersion", self.signed_change_dispersion),
            ("absolute_change_dispersion", self.absolute_change_dispersion),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise PerceptionExpectedTransitionError(f"{name} must be finite >= 0")
        if self.support_count < 0:
            raise PerceptionExpectedTransitionError("support_count must be >= 0")
        if self.change_semantics != TRANSITION_CHANGE_SEMANTICS:
            raise PerceptionExpectedTransitionError(
                "unsupported expected change semantics"
            )
        if self.signed_change_semantics != TRANSITION_SIGNED_CHANGE_SEMANTICS:
            raise PerceptionExpectedTransitionError(
                "unsupported expected signed change semantics"
            )
        if self.changed_fraction_semantics != TRANSITION_CHANGED_FRACTION_SEMANTICS:
            raise PerceptionExpectedTransitionError(
                "unsupported expected changed-fraction semantics"
            )
        if self.version != EXPECTED_TRANSITION_FIELD_VERSION:
            raise PerceptionExpectedTransitionError(
                "unsupported expected transition field version"
            )

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "absolute_change_dispersion": self.absolute_change_dispersion,
            "changed_fraction_semantics": self.changed_fraction_semantics,
            "change_semantics": self.change_semantics,
            "expected_after_mean": self.expected_after_mean,
            "expected_changed_fraction": self.expected_changed_fraction,
            "expected_mean_absolute_change": self.expected_mean_absolute_change,
            "expected_mean_signed_change": self.expected_mean_signed_change,
            "field_id": self.field_id,
            "signed_change_dispersion": self.signed_change_dispersion,
            "signed_change_semantics": self.signed_change_semantics,
            "support_count": self.support_count,
            "version": self.version,
        }


@dataclass(frozen=True)
class ExpectedTransitionVPMDTO:
    """Content-addressed structured future for one source plus one action."""

    expected_transition_id: str
    source_vpm_id: str
    action_label: str
    field_schema_id: str
    source_encoder_spec_id: str
    fields: tuple[ExpectedTransitionFieldDTO, ...]
    model_id: str
    training_dataset_id: str
    support_count: int
    confidence: float
    status: str
    png_digest: str
    png_bytes: bytes
    render_semantics: str = EXPECTED_TRANSITION_RENDER_SEMANTICS
    version: str = EXPECTED_TRANSITION_VPM_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.expected_transition_id,
                self.source_vpm_id,
                self.action_label,
                self.field_schema_id,
                self.source_encoder_spec_id,
                self.model_id,
                self.training_dataset_id,
            )
        ):
            raise PerceptionExpectedTransitionError(
                "expected transition identities must be non-empty"
            )
        field_ids = tuple(item.field_id for item in self.fields)
        if not field_ids or field_ids != tuple(sorted(set(field_ids))):
            raise PerceptionExpectedTransitionError(
                "expected fields must be non-empty, unique, and sorted"
            )
        if self.status not in EXPECTED_TRANSITION_STATUSES:
            raise PerceptionExpectedTransitionError(
                f"unsupported expected transition status: {self.status!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise PerceptionExpectedTransitionError("confidence must be in [0, 1]")
        if self.support_count < 0:
            raise PerceptionExpectedTransitionError("support_count must be >= 0")
        if self.render_semantics != EXPECTED_TRANSITION_RENDER_SEMANTICS:
            raise PerceptionExpectedTransitionError(
                "unsupported expected transition render semantics"
            )
        if self.version != EXPECTED_TRANSITION_VPM_VERSION:
            raise PerceptionExpectedTransitionError(
                "unsupported expected transition version"
            )
        if _digest(self.png_bytes) != self.png_digest:
            raise PerceptionExpectedTransitionError(
                "expected transition PNG digest mismatch"
            )
        expected_id = _digest(_canonical_json(self.canonical_payload()))
        if self.expected_transition_id != expected_id:
            raise PerceptionExpectedTransitionError(
                "expected transition identity disagrees with canonical payload"
            )

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "action_label": self.action_label,
            "confidence": self.confidence,
            "fields": [item.canonical_payload() for item in self.fields],
            "field_schema_id": self.field_schema_id,
            "model_id": self.model_id,
            "png_digest": self.png_digest,
            "render_semantics": self.render_semantics,
            "source_encoder_spec_id": self.source_encoder_spec_id,
            "source_vpm_id": self.source_vpm_id,
            "status": self.status,
            "support_count": self.support_count,
            "training_dataset_id": self.training_dataset_id,
            "version": self.version,
        }

    def to_array(self) -> np.ndarray:
        with Image.open(io.BytesIO(self.png_bytes)) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
        return array.copy()

    def field_evidence(self, field_id: str) -> ExpectedTransitionFieldDTO:
        for item in self.fields:
            if item.field_id == field_id:
                return item
        raise KeyError(field_id)

    def to_dict(self) -> dict[str, object]:
        payload = dict(self.canonical_payload())
        payload["expected_transition_id"] = self.expected_transition_id
        payload["png_bytes_hex"] = self.png_bytes.hex()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ExpectedTransitionVPMDTO":
        raw_fields = data["fields"]
        assert isinstance(raw_fields, list)
        fields = tuple(
            ExpectedTransitionFieldDTO(
                field_id=str(item["field_id"]),
                expected_after_mean=float(item["expected_after_mean"]),
                expected_mean_signed_change=float(item["expected_mean_signed_change"]),
                expected_mean_absolute_change=float(
                    item["expected_mean_absolute_change"]
                ),
                expected_changed_fraction=float(item["expected_changed_fraction"]),
                signed_change_dispersion=float(item["signed_change_dispersion"]),
                absolute_change_dispersion=float(item["absolute_change_dispersion"]),
                support_count=int(item["support_count"]),
            )
            for item in raw_fields
        )
        png_bytes = bytes.fromhex(str(data["png_bytes_hex"]))
        return cls(
            expected_transition_id=str(data["expected_transition_id"]),
            source_vpm_id=str(data["source_vpm_id"]),
            action_label=str(data["action_label"]),
            field_schema_id=str(data["field_schema_id"]),
            source_encoder_spec_id=str(data["source_encoder_spec_id"]),
            fields=fields,
            model_id=str(data["model_id"]),
            training_dataset_id=str(data["training_dataset_id"]),
            support_count=int(data["support_count"]),  # type: ignore[arg-type]
            confidence=float(data["confidence"]),  # type: ignore[arg-type]
            status=str(data["status"]),
            png_digest=str(data["png_digest"]),
            png_bytes=png_bytes,
        )


def render_expected_transition_png(
    fields: tuple[ExpectedTransitionFieldDTO, ...],
    field_schema: object,
    width: int,
    height: int,
) -> tuple[bytes, str]:
    """Render expected absolute change per field (inspectability surface)."""
    from .fields import VPMFieldSchemaDTO

    assert isinstance(field_schema, VPMFieldSchemaDTO)
    rendered = np.zeros((height, width), dtype=np.uint8)
    for item in fields:
        region = next(
            field for field in field_schema.fields if field.field_id == item.field_id
        )
        value = np.uint8(round(item.expected_mean_absolute_change * 255.0))
        canvas = rendered[region.y0 : region.y1, region.x0 : region.x1]
        np.maximum(canvas, value, out=canvas)
    png = _png_bytes(rendered)
    return png, _digest(png)

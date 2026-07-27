"""System C: a narrow ZeroModel adapter over the existing perception package.

Reused, unmodified, from ``zeromodel.perception``:
  - representation.py  (P1)  : SourceVPMDTO / encode_source_array
  - fields.py          (P4A) : VPMFieldSchemaDTO / build_grid_field_schema
  - expectations.py    (P6)  : PerceptionRegionAnnotationDTO
  - transition_evidence.py    (P18A) : build_transition_evidence_vpm
  - transition_conformance.py (P18B) : TransitionExpectationDTO / evaluate_transition_conformance
  - transition_discovery.py   (P18C) : recurrent unexplained-field discovery (used
    only for the cohort-level demonstration in report.py, not per-transition).

Explicitly bypassed: the certification/governance/promotion/lifecycle machinery
(P17*, P18D-P18G). This benchmark asks whether the *representation* carries a
useful transition-debugging signal; it is not exercising deployment machinery.

Non-privileged input contract
------------------------------
``analyze()`` receives only ``frame_before``, ``frame_after``, ``action``, and a
``TransitionMetadata`` that carries no ground-truth state (no tank_x/target_x/
cooldown, no fault label). The four named regions ("tank", "alien", "cooldown",
"background") are declared **once**, statically, as fixed row/column bands taken
directly from the environment's rendering contract
(``zeromodel.video.arcade_policy.rendering.render_state_frame``):

  tank      -> rows 11-13, any column       (the cannon sprite never appears elsewhere)
  alien     -> rows 2-4,  any column        (the target sprite never appears elsewhere)
  cooldown  -> rows 7-8,  rightmost column   (the indicator is drawn at a fixed corner)
  background-> every remaining field         (never legitimately touched)

These bands do not depend on tank_x/target_x for any specific transition, so
declaring them does not leak per-transition ground truth. Because the adapter
cannot see cooldown/target state, only two claims are asserted with full
confidence per action:
  - LEFT/RIGHT  : tank band is expected to change; background must stay exactly
                  stable; cooldown/alien are left uncovered (monitored only).
  - STAY        : tank band is expected to stay exactly stable; background must
                  stay exactly stable; cooldown/alien uncovered.
  - FIRE        : tank band is expected to stay exactly stable; cooldown band is
                  expected to change (true whenever fire is issued from a ready
                  cooldown -- see README's "known limitations" for the case where
                  FIRE is issued while already on cooldown); background must stay
                  exactly stable; alien uncovered (hit/miss cannot be known from
                  pixels + action alone).

Uncovered bands are not ignored: any field change there above
``UNEXPLAINED_MIN_*`` is reported through P18B's own "unexplained_change" bucket
(a soft, attention-worthy signal, never a hard pass/fail claim).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Protocol, Tuple

import numpy as np

from zeromodel.perception.expectations import PerceptionRegionAnnotationDTO
from zeromodel.perception.fields import VPMFieldSchemaDTO, build_grid_field_schema
from zeromodel.perception.representation import (
    SourceImageEncoderSpecDTO,
    SourceVPMDTO,
    encode_source_array,
)
from zeromodel.perception.transition_conformance import (
    TransitionConformanceReportDTO,
    TransitionExpectationDTO,
    evaluate_transition_conformance,
)
from zeromodel.perception.transition_evidence import (
    TransitionEvidenceVPMDTO,
    build_transition_evidence_vpm,
)

from visual_transition_benchmark.dataset import CELL_PIXELS, COMPONENT_NAMES, WIDTH_PX

FRAME_HEIGHT = 16

# Detection thresholds (System C's own, documented, fixed parameters).
CHANGE_THRESHOLD = 8  # per-pixel uint8 delta counted as "changed" (P18A)
FIELD_MIN_MEAN_ABS = 0.05
FIELD_MIN_CHANGED_FRACTION = 0.05

TANK_CHANGE_MIN_MEAN_ABS = 0.05
TANK_CHANGE_MIN_FRACTION = 0.02
STABLE_MAX_MEAN_ABS = 0.0
STABLE_MAX_FRACTION = 0.0
COOLDOWN_CHANGE_MIN_MEAN_ABS = 0.05
COOLDOWN_CHANGE_MIN_FRACTION = 0.3
UNEXPLAINED_MIN_MEAN_ABS = 0.05
UNEXPLAINED_MIN_FRACTION = 0.05

_SPEC = SourceImageEncoderSpecDTO(color_space="L")


def _build_field_schema() -> VPMFieldSchemaDTO:
    dummy = encode_source_array(
        np.zeros((FRAME_HEIGHT, WIDTH_PX), dtype=np.uint8), _SPEC
    )
    return build_grid_field_schema(
        dummy, tile_width=CELL_PIXELS, tile_height=1, channel_mode="joint"
    )


FIELD_SCHEMA: VPMFieldSchemaDTO = _build_field_schema()


def _band_for_field(y0: int, x0: int) -> str:
    if 11 <= y0 <= 13:
        return "tank"
    if 2 <= y0 <= 4:
        return "alien"
    if 7 <= y0 <= 8 and x0 == WIDTH_PX - CELL_PIXELS:
        return "cooldown"
    return "background"


BAND_FIELD_IDS: Dict[str, Tuple[str, ...]] = {name: () for name in COMPONENT_NAMES}
FIELD_ID_TO_BAND: Dict[str, str] = {}
_by_band: Dict[str, list] = {name: [] for name in COMPONENT_NAMES}
for _field in FIELD_SCHEMA.fields:
    _band = _band_for_field(_field.y0, _field.x0)
    _by_band[_band].append(_field.field_id)
    FIELD_ID_TO_BAND[_field.field_id] = _band
BAND_FIELD_IDS = {name: tuple(sorted(ids)) for name, ids in _by_band.items()}
assert sum(len(v) for v in BAND_FIELD_IDS.values()) == len(FIELD_SCHEMA.fields)


def _make_annotation(name: str) -> PerceptionRegionAnnotationDTO:
    return PerceptionRegionAnnotationDTO.create(
        FIELD_SCHEMA, BAND_FIELD_IDS[name], label=name, role="declared_static_band"
    )


ANNOTATIONS: Dict[str, PerceptionRegionAnnotationDTO] = {
    name: _make_annotation(name) for name in COMPONENT_NAMES
}
ANNOTATIONS_TUPLE: Tuple[PerceptionRegionAnnotationDTO, ...] = tuple(
    ANNOTATIONS.values()
)
ANNOTATION_ID_TO_NAME: Dict[str, str] = {
    ann.annotation_id: name for name, ann in ANNOTATIONS.items()
}


def _expectation(name: str, expected_change: str, **bounds) -> TransitionExpectationDTO:
    return TransitionExpectationDTO.create(
        field_schema_id=FIELD_SCHEMA.field_schema_id,
        annotation_ids=(ANNOTATIONS[name].annotation_id,),
        expected_change=expected_change,
        **bounds,
    )


_TANK_CHANGE = _expectation(
    "tank",
    "change",
    minimum_mean_absolute_change=TANK_CHANGE_MIN_MEAN_ABS,
    minimum_changed_fraction=TANK_CHANGE_MIN_FRACTION,
)
_TANK_STABLE = _expectation(
    "tank",
    "stable",
    maximum_mean_absolute_change=STABLE_MAX_MEAN_ABS,
    maximum_changed_fraction=STABLE_MAX_FRACTION,
)
_BACKGROUND_STABLE = _expectation(
    "background",
    "stable",
    maximum_mean_absolute_change=STABLE_MAX_MEAN_ABS,
    maximum_changed_fraction=STABLE_MAX_FRACTION,
)
_COOLDOWN_CHANGE = _expectation(
    "cooldown",
    "change",
    minimum_mean_absolute_change=COOLDOWN_CHANGE_MIN_MEAN_ABS,
    minimum_changed_fraction=COOLDOWN_CHANGE_MIN_FRACTION,
)

EXPECTATIONS_BY_ACTION: Dict[str, Tuple[TransitionExpectationDTO, ...]] = {
    "LEFT": (_TANK_CHANGE, _BACKGROUND_STABLE),
    "RIGHT": (_TANK_CHANGE, _BACKGROUND_STABLE),
    "STAY": (_TANK_STABLE, _BACKGROUND_STABLE),
    "FIRE": (_TANK_STABLE, _BACKGROUND_STABLE, _COOLDOWN_CHANGE),
}

EXPECTED_CHANGE_BAND_BY_ACTION: Dict[str, Tuple[str, ...]] = {
    "LEFT": ("tank",),
    "RIGHT": ("tank",),
    "STAY": (),
    "FIRE": ("cooldown",),
}

_MISSING_STATUSES = {"missing_expected_change", "insufficient_change"}
_VIOLATION_STATUSES = {
    "unexpected_change",
    "excessive_change",
    "wrong_change_direction",
}


@dataclass(frozen=True)
class TransitionMetadata:
    """Non-privileged, per-transition bookkeeping (no ground-truth state)."""

    transition_id: str
    step_number: int


@dataclass(frozen=True)
class TransitionAnalysis:
    predicted_region_mask: np.ndarray
    predicted_fields: Tuple[str, ...]
    predicted_components: Tuple[str, ...]
    expected_components: Tuple[str, ...]
    unexpected_components: Tuple[str, ...]
    missing_components: Tuple[str, ...]
    evidence_scores: Dict[str, float]
    diagnostics: Mapping[str, object]
    transition_evidence: TransitionEvidenceVPMDTO
    conformance_report: TransitionConformanceReportDTO


class ZeroModelTransitionAnalyzer(Protocol):
    def analyze(
        self,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        action: str,
        metadata: TransitionMetadata,
    ) -> TransitionAnalysis: ...


def _region_mask(field_ids: Tuple[str, ...]) -> np.ndarray:
    mask = np.zeros((FRAME_HEIGHT, WIDTH_PX), dtype=bool)
    by_id = {field.field_id: field for field in FIELD_SCHEMA.fields}
    for field_id in field_ids:
        field = by_id[field_id]
        mask[field.y0 : field.y1, field.x0 : field.x1] = True
    return mask


# Static declared-band pixel masks, shared with baselines.py so every system is
# scored against (and, for System A's component attribution, evaluated using)
# exactly the same named regions.
BAND_MASKS: Dict[str, np.ndarray] = {
    name: _region_mask(field_ids) for name, field_ids in BAND_FIELD_IDS.items()
}


class ArcadeBandZeroModelAnalyzer:
    """System C: P4A field partitioning + P18A evidence + P18B conformance."""

    def analyze(
        self,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        action: str,
        metadata: TransitionMetadata,
    ) -> TransitionAnalysis:
        if action not in EXPECTATIONS_BY_ACTION:
            raise ValueError(f"unsupported action: {action}")
        before_vpm: SourceVPMDTO = encode_source_array(
            np.ascontiguousarray(frame_before, dtype=np.uint8), _SPEC
        )
        after_vpm: SourceVPMDTO = encode_source_array(
            np.ascontiguousarray(frame_after, dtype=np.uint8), _SPEC
        )
        transition_evidence = build_transition_evidence_vpm(
            before_vpm,
            after_vpm,
            FIELD_SCHEMA,
            annotations=ANNOTATIONS_TUPLE,
            change_threshold=CHANGE_THRESHOLD,
        )
        predicted_fields = transition_evidence.changed_field_ids(
            minimum_mean_absolute_change=FIELD_MIN_MEAN_ABS,
            minimum_changed_fraction=FIELD_MIN_CHANGED_FRACTION,
        )
        predicted_components = tuple(
            sorted(
                name
                for name in COMPONENT_NAMES
                if set(BAND_FIELD_IDS[name]) & set(predicted_fields)
            )
        )
        evidence_scores = {
            name: max(
                (
                    transition_evidence.field_evidence(fid).mean_absolute_change
                    for fid in fids
                ),
                default=0.0,
            )
            for name, fids in BAND_FIELD_IDS.items()
        }

        expectations = EXPECTATIONS_BY_ACTION[action]
        report = evaluate_transition_conformance(
            transition_evidence,
            expectations,
            ANNOTATIONS_TUPLE,
            relations=(),
            minimum_unexplained_mean_absolute_change=UNEXPLAINED_MIN_MEAN_ABS,
            minimum_unexplained_changed_fraction=UNEXPLAINED_MIN_FRACTION,
        )

        missing = set()
        unexpected = set()
        unexplained = set()
        for finding in report.findings:
            if finding.status == "unexplained_change":
                for field_id in finding.field_ids:
                    unexplained.add(FIELD_ID_TO_BAND[field_id])
                continue
            names = {ANNOTATION_ID_TO_NAME[aid] for aid in finding.annotation_ids}
            if finding.status in _MISSING_STATUSES:
                missing |= names
            elif finding.status in _VIOLATION_STATUSES:
                unexpected |= names

        diagnostics = {
            "conformance_status": report.status,
            "field_schema_id": FIELD_SCHEMA.field_schema_id,
            "transition_evidence_id": transition_evidence.transition_evidence_id,
            "conformance_report_id": report.report_id,
            "unexplained_components": tuple(sorted(unexplained)),
            "finding_statuses": tuple(sorted({f.status for f in report.findings})),
            "thresholds": {
                "field_min_mean_absolute_change": FIELD_MIN_MEAN_ABS,
                "field_min_changed_fraction": FIELD_MIN_CHANGED_FRACTION,
                "change_threshold_pixels": CHANGE_THRESHOLD,
                "unexplained_min_mean_absolute_change": UNEXPLAINED_MIN_MEAN_ABS,
                "unexplained_min_changed_fraction": UNEXPLAINED_MIN_FRACTION,
            },
            "output_level": (
                "component: predicted_components/expected_components/missing_components/"
                "unexpected_components are declared-band labels; predicted_fields/"
                "predicted_region_mask are exact P4A field tiles and pixel rectangles; "
                "evidence_scores are raw P18A per-band mean_absolute_change"
            ),
        }

        return TransitionAnalysis(
            predicted_region_mask=_region_mask(predicted_fields),
            predicted_fields=predicted_fields,
            predicted_components=predicted_components,
            expected_components=EXPECTED_CHANGE_BAND_BY_ACTION[action],
            unexpected_components=tuple(sorted(unexpected)),
            missing_components=tuple(sorted(missing)),
            evidence_scores=evidence_scores,
            diagnostics=diagnostics,
            transition_evidence=transition_evidence,
            conformance_report=report,
        )

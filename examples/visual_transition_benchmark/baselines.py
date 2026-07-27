"""System A (raw pixel difference) and System B (privileged ground truth).

Both baselines expose the same shape of output as
``zeromodel_adapter.TransitionAnalysis`` (duck-typed, see ``SystemOutput``) so
``metrics.py`` can score all three systems identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

from visual_transition_benchmark import zeromodel_adapter as zm
from visual_transition_benchmark.dataset import COMPONENT_NAMES, TransitionRecord

# System A fixed parameters (documented, per benchmark section 3).
PIXEL_THRESHOLD = 8  # identical sensitivity to CHANGE_THRESHOLD used by System C
MIN_COMPONENT_SIZE = 2  # pixels; components smaller than this are dropped as noise
CONNECTIVITY = 4  # 4-connected neighbourhood
MORPHOLOGY = "none"  # no dilation/erosion/smoothing is applied


@dataclass(frozen=True)
class SystemOutput:
    system: str
    predicted_region_mask: np.ndarray
    predicted_fields: Tuple[str, ...]
    predicted_components: Tuple[str, ...]
    expected_components: Tuple[str, ...]
    unexpected_components: Tuple[str, ...]
    missing_components: Tuple[str, ...]
    evidence_scores: Dict[str, float]
    diagnostics: Mapping[str, object]


def _fields_touched(mask: np.ndarray) -> Tuple[str, ...]:
    touched = []
    for field in zm.FIELD_SCHEMA.fields:
        if mask[field.y0 : field.y1, field.x0 : field.x1].any():
            touched.append(field.field_id)
    return tuple(sorted(touched))


def _components_touching(mask: np.ndarray) -> Tuple[str, ...]:
    return tuple(
        sorted(name for name in COMPONENT_NAMES if mask[zm.BAND_MASKS[name]].any())
    )


def _connected_components(changed_mask: np.ndarray) -> np.ndarray:
    """Label 4-connected components with a plain BFS (no scipy dependency)."""

    labels = np.zeros(changed_mask.shape, dtype=np.int32)
    next_label = 0
    height, width = changed_mask.shape
    for start_r in range(height):
        for start_c in range(width):
            if not changed_mask[start_r, start_c] or labels[start_r, start_c] != 0:
                continue
            next_label += 1
            stack = [(start_r, start_c)]
            labels[start_r, start_c] = next_label
            while stack:
                r, c = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and changed_mask[nr, nc]
                        and labels[nr, nc] == 0
                    ):
                        labels[nr, nc] = next_label
                        stack.append((nr, nc))
    return labels


def pixel_diff_baseline(
    frame_before: np.ndarray,
    frame_after: np.ndarray,
    *,
    pixel_threshold: int = PIXEL_THRESHOLD,
    min_component_size: int = MIN_COMPONENT_SIZE,
) -> SystemOutput:
    """System A: absolute pixel difference + connected-component aggregation.

    No expectation model exists here: ``missing_components`` is always empty
    (pixel differencing cannot represent "an expected change failed to
    occur") and ``unexpected_components`` is always empty (there is no
    declared "must stay stable" claim to violate). This is intentional and is
    exactly the capability gap the benchmark measures.
    """

    diff = np.abs(frame_after.astype(np.int16) - frame_before.astype(np.int16))
    changed_mask = diff >= pixel_threshold
    labels = _connected_components(changed_mask)
    counts = np.bincount(labels.reshape(-1))
    keep_mask = np.zeros_like(changed_mask)
    for label_id in range(1, len(counts)):
        if counts[label_id] >= min_component_size:
            keep_mask |= labels == label_id

    predicted_components = _components_touching(keep_mask)
    evidence_scores = {
        name: float(np.mean(diff[zm.BAND_MASKS[name]]) / 255.0)
        for name in COMPONENT_NAMES
    }
    diagnostics = {
        "pixel_threshold": pixel_threshold,
        "min_component_size": min_component_size,
        "connectivity": CONNECTIVITY,
        "morphology": MORPHOLOGY,
        "region_aggregation_method": "4-connected component labeling, components < min_component_size dropped",
        "no_expectation_model": True,
        "note": (
            "System A has no notion of an expected transition outcome; "
            "missing_components and unexpected_components are always empty by "
            "construction, not because nothing was missed"
        ),
    }
    return SystemOutput(
        system="pixel_diff",
        predicted_region_mask=keep_mask,
        predicted_fields=_fields_touched(keep_mask),
        predicted_components=predicted_components,
        expected_components=(),
        unexpected_components=(),
        missing_components=(),
        evidence_scores=evidence_scores,
        diagnostics=diagnostics,
    )


def privileged_baseline(record: TransitionRecord) -> SystemOutput:
    """System B: exact ground-truth component masks (upper reference, not deployable)."""

    observed = set(record.observed_changed_components)
    expected = set(record.expected_changed_components)
    region_mask = np.zeros_like(record.component_annotations["tank"])
    for name in observed:
        region_mask |= record.component_annotations[name]
    evidence_scores = {
        name: (1.0 if name in observed else 0.0) for name in COMPONENT_NAMES
    }
    diagnostics = {
        "source": "exact ground-truth component masks (tank/alien/cooldown/background)",
        "privileged": True,
        "deployable": False,
    }
    return SystemOutput(
        system="privileged",
        predicted_region_mask=region_mask,
        predicted_fields=_fields_touched(region_mask),
        predicted_components=tuple(sorted(observed)),
        expected_components=tuple(sorted(expected)),
        unexpected_components=tuple(sorted(observed - expected)),
        missing_components=tuple(sorted(expected - observed)),
        evidence_scores=evidence_scores,
        diagnostics=diagnostics,
    )

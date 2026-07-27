"""Compile a P4A field schema from declared evidence requirements.

This is the direct generalization of what ``zeromodel_adapter.py`` (4x1px
tiles) and ``value_contracts.py`` (1x1px tiles) each hand-built separately in
stages 1 and 2: both are special cases of "pick a tile size fine enough for
every declared requirement, then group tiles by declared region." The
compiler picks ONE uniform tile size -- the finest (smallest) requested by any
requirement -- because a finer grid can always answer a coarser question
(mean/max over more, smaller tiles), but a coarser grid cannot answer a finer
one (this is exactly stage 2's cooldown-dilution bug).

Everything in this module is domain-neutral: it never mentions a component
name specific to arcade or warehouse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

from zeromodel.perception.fields import VPMFieldSchemaDTO, build_grid_field_schema
from zeromodel.perception.representation import (
    SourceImageEncoderSpecDTO,
    encode_source_array,
)
from zeromodel.perception.transition_evidence import (
    TransitionEvidenceVPMDTO,
    build_transition_evidence_vpm,
)

from visual_transition_benchmark.compilation.evidence_requirements import (
    VisualEvidenceRequirement,
)

_SPEC = SourceImageEncoderSpecDTO(color_space="L")


class FieldSchemaCompilationError(ValueError):
    pass


@dataclass(frozen=True)
class CompiledFieldSchema:
    canvas_shape: Tuple[int, int]
    tile_height: int
    tile_width: int
    field_schema: VPMFieldSchemaDTO
    fields_for: Mapping[
        str, Tuple[str, ...]
    ]  # "component.property" -> field ids inside its declared region
    requirement_by_key: Mapping[str, VisualEvidenceRequirement]

    def field_ids(self, component: str, property_name: str) -> Tuple[str, ...]:
        return self.fields_for[f"{component}.{property_name}"]

    def build_transition_evidence(
        self,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        *,
        change_threshold: int = 8,
    ) -> TransitionEvidenceVPMDTO:
        before_vpm = encode_source_array(
            np.ascontiguousarray(frame_before, dtype=np.uint8), _SPEC
        )
        after_vpm = encode_source_array(
            np.ascontiguousarray(frame_after, dtype=np.uint8), _SPEC
        )
        return build_transition_evidence_vpm(
            before_vpm,
            after_vpm,
            self.field_schema,
            annotations=(),
            change_threshold=change_threshold,
        )


def compile_field_schema(
    canvas_shape: Tuple[int, int], requirements: Sequence[VisualEvidenceRequirement]
) -> CompiledFieldSchema:
    if not requirements:
        raise FieldSchemaCompilationError(
            "at least one evidence requirement is required"
        )
    height, width = canvas_shape
    tile_height = min(req.required_resolution[0] for req in requirements)
    tile_width = min(req.required_resolution[1] for req in requirements)
    if height % tile_height != 0 and tile_height != 1:
        tile_height = (
            1  # fall back to exact resolution rather than silently misaligning
        )
    if width % tile_width != 0 and tile_width != 1:
        tile_width = 1

    dummy = encode_source_array(np.zeros((height, width), dtype=np.uint8), _SPEC)
    schema = build_grid_field_schema(
        dummy, tile_width=tile_width, tile_height=tile_height, channel_mode="joint"
    )

    fields_for: Dict[str, Tuple[str, ...]] = {}
    requirement_by_key: Dict[str, VisualEvidenceRequirement] = {}
    for req in requirements:
        y0, y1, x0, x1 = req.region
        ids = tuple(
            sorted(
                field.field_id
                for field in schema.fields
                if field.y0 >= y0
                and field.y1 <= y1
                and field.x0 >= x0
                and field.x1 <= x1
            )
        )
        if not ids:
            raise FieldSchemaCompilationError(
                f"requirement {req.key!r} resolves to zero fields at tile size "
                f"({tile_height}, {tile_width}) -- region {req.region} does not align"
            )
        fields_for[req.key] = ids
        requirement_by_key[req.key] = req

    return CompiledFieldSchema(
        canvas_shape=canvas_shape,
        tile_height=tile_height,
        tile_width=tile_width,
        field_schema=schema,
        fields_for=fields_for,
        requirement_by_key=requirement_by_key,
    )


def aggregate_by_group(
    transition_evidence: TransitionEvidenceVPMDTO,
    field_ids: Sequence[str],
    which: str,
    group_of: Callable[[str], Hashable],
    aggregation: str,
    field_by_id: Mapping[str, object],
) -> Dict[Hashable, float]:
    """Group field ids by an arbitrary domain-supplied key and aggregate.

    ``group_of`` maps a field_id to whatever grouping key the caller wants
    (a column index, a (row, col) grid cell, ...) -- this function never
    interprets that key, only groups by it, so it stays domain-neutral.
    """

    totals: Dict[Hashable, float] = {}
    counts: Dict[Hashable, int] = {}
    maxima: Dict[Hashable, float] = {}
    for field_id in field_ids:
        value = getattr(transition_evidence.field_evidence(field_id), which)
        group = group_of(field_id)
        totals[group] = totals.get(group, 0.0) + value
        counts[group] = counts.get(group, 0) + 1
        maxima[group] = max(maxima.get(group, 0.0), value)
    if aggregation == "mean":
        return {group: totals[group] / counts[group] for group in totals}
    if aggregation == "max":
        return dict(maxima)
    raise ValueError(
        f"aggregate_by_group does not support aggregation={aggregation!r} (use exact_pattern for that)"
    )


def argmax_group(
    groups: Mapping[Hashable, float], *, threshold: float
) -> Optional[Hashable]:
    if not groups:
        return None
    best = max(groups, key=lambda key: groups[key])
    if groups[best] < threshold:
        return None
    return best


def exact_pattern(
    transition_evidence: TransitionEvidenceVPMDTO,
    field_ids: Sequence[str],
    which: str,
    *,
    on_threshold: float,
) -> Tuple[bool, ...]:
    """A deterministic on/off signature over an exact region -- for
    categorical_shape / visible_marker_pattern evidence, where the caller
    (a domain-specific decoder) interprets the resulting pattern."""

    return tuple(
        getattr(transition_evidence.field_evidence(fid), which) >= on_threshold
        for fid in sorted(field_ids)
    )

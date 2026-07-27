"""Declared evidence requirements: the input to the field-schema compiler.

A domain declares, once, per (component, property): where on the canvas it
can ever appear, how fine a grid is needed to read it without ambiguity, and
how to aggregate a tile group into one scalar. The compiler
(``field_schema_compiler.py``) turns a list of these into one P4A field
schema. This is the direct generalization of the two representation bugs
found in stage 2: the cooldown-dilution bug was an under-specified
``required_resolution``, and the tank-bleed bug was a wrong ``aggregation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

EVIDENCE_KINDS = (
    "presence",
    "spatial_position",
    "numeric_intensity",
    "categorical_shape",
    "visible_marker_pattern",
)
AGGREGATIONS = ("mean", "max", "exact")


@dataclass(frozen=True)
class VisualEvidenceRequirement:
    component: str
    property_name: str
    evidence_kind: str
    region: Tuple[int, int, int, int]  # (y0, y1, x0, x1), half-open, static and declared once
    required_resolution: Tuple[int, int]  # (tile_height, tile_width)
    aggregation: str

    def __post_init__(self) -> None:
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(f"unsupported evidence_kind: {self.evidence_kind}")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"unsupported aggregation: {self.aggregation}")
        y0, y1, x0, x1 = self.region
        if not (0 <= y0 < y1 and 0 <= x0 < x1):
            raise ValueError(f"invalid region: {self.region}")
        th, tw = self.required_resolution
        if th <= 0 or tw <= 0:
            raise ValueError(f"invalid required_resolution: {self.required_resolution}")

    @property
    def key(self) -> str:
        return f"{self.component}.{self.property_name}"

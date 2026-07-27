"""Bounded, deterministic candidate representation generation.

Each evidence kind gets a small, hand-bounded set of candidates that vary
only the parameters meaningful for that kind (per the task spec: "use only
operators meaningful for the evidence kind", "do not create an unbounded
combinatorial search"). Nothing here is arcade- or warehouse-specific; the
region geometry (bounds, logical cell size, canonical levels, identity
sub-patch offsets) is supplied by the caller (a domain adapter), not looked
up by name.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple

from visual_transition_benchmark.compiler.contracts import (
    AggregationKind,
    ComparisonKind,
    VisualEvidenceRequirement,
)

DECODER_WEIGHT = {
    "presence_threshold": 1.0,
    "nearest_permitted_value": 2.0,
    "exact_lookup": 2.0,
    "dominant_field_value": 2.5,
    "argmax_field": 2.0,
    "categorical_template": 3.0,
    "local_marker_pattern": 3.0,
    "signed_delta_over_position": 3.0,
    "exact_delta_over_position": 3.0,
    "relation_over_decoded": 4.0,
}


def _canonical_json(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(payload) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()[:24]}"


@dataclass(frozen=True)
class RepresentationCandidate:
    requirement_id: str
    region_id: str
    field_height: int
    field_width: int
    aggregation: AggregationKind
    decoder_kind: str
    comparison: ComparisonKind
    complexity_cost: float
    assumptions: Tuple[str, ...] = ()

    candidate_id: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "requirement_id": self.requirement_id,
            "region_id": self.region_id,
            "field_height": self.field_height,
            "field_width": self.field_width,
            "aggregation": self.aggregation,
            "decoder_kind": self.decoder_kind,
            "comparison": self.comparison,
        }
        object.__setattr__(self, "candidate_id", _digest(payload))


@dataclass(frozen=True)
class RegionGeometry:
    """Domain-declared geometry for one candidate region. Only the fields a
    given evidence kind actually needs are read. Bounds are absolute
    coordinates within the domain's full canvas (needed to reuse
    ``compilation.field_schema_compiler.compile_field_schema`` unchanged)."""

    region_id: str
    canvas_shape: Tuple[int, int]
    y0: int
    y1: int
    x0: int
    x1: int
    cell_height: int = 1
    cell_width: int = 1
    sub_patch_size: int = 2  # for visible_identity marker patches

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


def _complexity(region: RegionGeometry, field_height: int, field_width: int, decoder_kind: str) -> float:
    area = region.height * region.width
    tile_area = max(1, field_height * field_width)
    field_count_estimate = max(1, -(-area // tile_area))  # ceil
    return field_count_estimate * DECODER_WEIGHT.get(decoder_kind, 2.0)


def generate_candidates(
    requirement: VisualEvidenceRequirement, region: RegionGeometry
) -> Tuple[RepresentationCandidate, ...]:
    kind = requirement.evidence_kind
    generator = _GENERATORS.get(kind)
    if generator is None:
        raise ValueError(f"no candidate generator registered for evidence_kind={kind!r}")
    candidates = generator(requirement, region)
    return tuple(sorted(candidates, key=lambda c: c.candidate_id))


def _presence_candidates(req, region):
    out = []
    for (fh, fw), note in ((region.cell_height, region.cell_width), "cell-resolution"), ((1, 1), "pixel-resolution"):
        out.append(
            RepresentationCandidate(
                requirement_id=req.requirement_id,
                region_id=region.region_id,
                field_height=fh,
                field_width=fw,
                aggregation="mean",
                decoder_kind="presence_threshold",
                comparison=req.comparison,
                complexity_cost=_complexity(region, fh, fw, "presence_threshold"),
                assumptions=(note,),
            )
        )
    return out


def _numeric_value_candidates(req, region):
    out = []
    resolutions = [
        (region.cell_height, region.cell_width, "cell-resolution (may dilute a sub-cell signal)"),
        (1, 1, "pixel-resolution"),
    ]
    # "nearest_permitted_value"/"exact_lookup" average every field the
    # declared region contains, naively -- this is what reproduces the
    # cooldown dilution bug when the region is wider than the true signal.
    # "dominant_field_value" is the auto-narrowing repair: it is fit once on
    # development samples (pixel variance only, no labels) to find which
    # fields ever carry non-background signal, then averages only those.
    for fh, fw, note in resolutions:
        for decoder in ("nearest_permitted_value", "exact_lookup", "dominant_field_value"):
            out.append(
                RepresentationCandidate(
                    requirement_id=req.requirement_id,
                    region_id=region.region_id,
                    field_height=fh,
                    field_width=fw,
                    aggregation="mean",
                    decoder_kind=decoder,
                    comparison=req.comparison,
                    complexity_cost=_complexity(region, fh, fw, decoder),
                    assumptions=(note,),
                )
            )
    return out


def _categorical_state_candidates(req, region):
    out = []
    resolutions = [
        (region.height, region.width, "whole-region (may dilute a thin glyph)"),
        (1, 1, "pixel-resolution"),
    ]
    for fh, fw, note in resolutions:
        for decoder in ("nearest_permitted_value", "categorical_template", "dominant_field_value"):
            out.append(
                RepresentationCandidate(
                    requirement_id=req.requirement_id,
                    region_id=region.region_id,
                    field_height=fh,
                    field_width=fw,
                    aggregation="mean",
                    decoder_kind=decoder,
                    comparison=req.comparison,
                    complexity_cost=_complexity(region, fh, fw, decoder),
                    assumptions=(note,),
                )
            )
    return out


def _spatial_position_candidates(req, region):
    out = []
    resolutions = [(region.cell_height, region.cell_width, "cell-resolution"), (1, 1, "pixel-resolution")]
    for fh, fw, note in resolutions:
        for aggregation in ("mean", "max", "centroid"):
            out.append(
                RepresentationCandidate(
                    requirement_id=req.requirement_id,
                    region_id=region.region_id,
                    field_height=fh,
                    field_width=fw,
                    aggregation=aggregation,
                    decoder_kind="argmax_field",
                    comparison=req.comparison,
                    complexity_cost=_complexity(region, fh, fw, "argmax_field"),
                    assumptions=(note,),
                )
            )
    return out


def _delta_candidates(req, region):
    decoder = "signed_delta_over_position" if req.evidence_kind == "signed_delta" else "exact_delta_over_position"
    out = []
    resolutions = [(region.cell_height, region.cell_width, "cell-resolution"), (1, 1, "pixel-resolution")]
    for fh, fw, note in resolutions:
        for aggregation in ("mean", "max"):
            out.append(
                RepresentationCandidate(
                    requirement_id=req.requirement_id,
                    region_id=region.region_id,
                    field_height=fh,
                    field_width=fw,
                    aggregation=aggregation,
                    decoder_kind=decoder,
                    comparison=req.comparison,
                    complexity_cost=_complexity(region, fh, fw, decoder),
                    assumptions=(note, "requires a before/after frame pair"),
                )
            )
    return out


def _relation_candidates(req, region):
    out = []
    for threshold, note in ((1, "adjacency<=1"), (0, "adjacency==0 (coincidence only)")):
        out.append(
            RepresentationCandidate(
                requirement_id=req.requirement_id,
                region_id=region.region_id,
                field_height=1,
                field_width=1,
                aggregation="mean",
                decoder_kind="relation_over_decoded",
                comparison=req.comparison,
                complexity_cost=_complexity(region, 1, 1, "relation_over_decoded") + threshold,
                assumptions=(note,),
            )
        )
    return out


def _visible_identity_candidates(req, region):
    out = []
    out.append(
        RepresentationCandidate(
            requirement_id=req.requirement_id,
            region_id=region.region_id,
            field_height=region.cell_height,
            field_width=region.cell_width,
            aggregation="mean",
            decoder_kind="nearest_permitted_value",
            comparison=req.comparison,
            complexity_cost=_complexity(region, region.cell_height, region.cell_width, "nearest_permitted_value"),
            assumptions=("cell-resolution mean (may wash out a small sub-cell marker)",),
        )
    )
    out.append(
        RepresentationCandidate(
            requirement_id=req.requirement_id,
            region_id=region.region_id,
            field_height=1,
            field_width=1,
            aggregation="exact_pattern",
            decoder_kind="local_marker_pattern",
            comparison=req.comparison,
            complexity_cost=_complexity(region, 1, 1, "local_marker_pattern"),
            assumptions=("pixel-resolution exact marker pattern",),
        )
    )
    return out


_GENERATORS = {
    "presence": _presence_candidates,
    "numeric_value": _numeric_value_candidates,
    "categorical_state": _categorical_state_candidates,
    "spatial_position": _spatial_position_candidates,
    "signed_delta": _delta_candidates,
    "exact_magnitude": _delta_candidates,
    "relation": _relation_candidates,
    "visible_identity": _visible_identity_candidates,
}

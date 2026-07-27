"""Candidate decoding and evidence-preservation evaluation.

Reuses ``compilation.field_schema_compiler`` (P4A/P18A, unchanged from the
cross-domain experiment) to build the actual field grid at a candidate's
declared resolution; everything here is the decode/scoring layer on top.

Evaluation runs only on development samples. Ground-truth ``true_before``/
``true_after`` values are privileged (real simulator state) -- used here only
as scoring labels, never fed into ``decode_candidate`` itself, which reads
only ``frame_before``/``frame_after`` pixels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from visual_transition_benchmark.compilation.evidence_requirements import (
    VisualEvidenceRequirement as LowLevelRequirement,
)
from visual_transition_benchmark.compilation.field_schema_compiler import (
    aggregate_by_group,
    argmax_group,
    compile_field_schema,
    exact_pattern,
)
from visual_transition_benchmark.compiler.candidates import RegionGeometry, RepresentationCandidate
from visual_transition_benchmark.compiler.contracts import VisualEvidenceRequirement


@dataclass(frozen=True)
class DevelopmentSample:
    sample_id: str
    frame_before: np.ndarray
    frame_after: np.ndarray
    true_before: object  # privileged; scoring only
    true_after: object
    is_unrelated: bool = False  # this sample's only real change is unrelated to the property under test
    fault_present: bool = False


@dataclass(frozen=True)
class DecodedProperty:
    before: object
    after: object
    tie_before: bool = False
    tie_after: bool = False


def _sign(value) -> object:
    if isinstance(value, tuple):
        return tuple(_sign(v) for v in value)
    return -1 if value < 0 else (1 if value > 0 else 0)


def _subtract(a, b):
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(x - y for x, y in zip(a, b))
    return a - b


def _nearest_level(value: float, levels: Sequence[float], tolerance: float) -> object:
    if not levels:
        # No canonical vocabulary was declared to snap to -- passing the raw
        # continuous intensity through as if it were a decode would silently
        # fabricate precision this decoder never earned, and would prevent a
        # genuinely-unobservable property (e.g. an identity with no visible
        # marker at all) from ever being classified as degenerate: the raw
        # intensity varies with irrelevant scene content even though it
        # carries no information about the declared property.
        return "no_canonical_levels_declared"
    best, best_distance = None, None
    for level in levels:
        distance = abs(value - level)
        if best_distance is None or distance < best_distance:
            best, best_distance = level, distance
    if best_distance <= tolerance:
        return best
    return "out_of_domain"


class _FastFieldEvidence:
    """Wraps a ``TransitionEvidenceVPMDTO`` so ``.field_evidence(field_id)``
    is an O(1) dict lookup instead of P18A's by-design O(n) linear scan.
    Same performance fix as ``domains.warehouse.contracts.DecodedGrids``, just
    shaped to stay a drop-in replacement for the DTO (both
    ``aggregate_by_group`` and this module call ``.field_evidence(fid)``
    without otherwise touching the DTO), so no shared code needs editing."""

    __slots__ = ("_by_id",)

    def __init__(self, evidence) -> None:
        self._by_id = {item.field_id: item for item in evidence.fields}

    def field_evidence(self, field_id: str):
        return self._by_id[field_id]


@lru_cache(maxsize=None)
def _build_schema(candidate: RepresentationCandidate, region: RegionGeometry):
    # A candidate's schema depends only on (field_height, field_width, region
    # geometry) -- never on the sample -- so caching by (candidate, region)
    # turns "rebuild the whole P4A grid per sample" into "build it once per
    # candidate", which is what actually made evaluation over more than a
    # handful of samples tractable (the same fix class as
    # domains.warehouse.contracts.DecodedGrids, applied one level up).
    low = LowLevelRequirement(
        component=region.region_id,
        property_name="candidate",
        evidence_kind="presence",  # not semantically used downstream; only region/resolution/aggregation matter
        region=(region.y0, region.y1, region.x0, region.x1),
        required_resolution=(candidate.field_height, candidate.field_width),
        aggregation="mean" if candidate.aggregation not in ("mean", "max") else candidate.aggregation,
    )
    return compile_field_schema(region.canvas_shape, (low,))


def _cell_group_of(compiled, field_id: str, region: RegionGeometry):
    field = compiled.requirement_by_key[f"{region.region_id}.candidate"]
    by_id = {f.field_id: f for f in compiled.field_schema.fields}
    f = by_id[field_id]
    row = (f.y0 - region.y0) // max(1, region.cell_height)
    col = (f.x0 - region.x0) // max(1, region.cell_width)
    return (row, col)


def decode_candidate(
    candidate: RepresentationCandidate,
    region: RegionGeometry,
    frame_before: np.ndarray,
    frame_after: np.ndarray,
    *,
    canonical_levels: Sequence[float] = (),
    level_tolerance: float = 0.02,
    sub_patch_offsets: Sequence[Tuple[int, int]] = (),
    dot_threshold: float = 0.5,
    alive_threshold: float = 0.05,
    active_field_ids: Optional[frozenset] = None,
) -> DecodedProperty:
    compiled = _build_schema(candidate, region)
    evidence = _FastFieldEvidence(compiled.build_transition_evidence(frame_before, frame_after))
    field_ids = compiled.field_ids(region.region_id, "candidate")
    by_id = {f.field_id: f for f in compiled.field_schema.fields}

    decoder = candidate.decoder_kind

    if decoder == "presence_threshold":
        # Per-field vector, not a single region-wide mean: a region-wide
        # average is blind to a change that relocates signal within the
        # region without changing its total (e.g. a sprite sliding sideways
        # conserves total lit-pixel mass across tiles), which would silently
        # decode as "unchanged" despite every pixel having moved.
        before_vec = tuple(evidence.field_evidence(fid).before_mean for fid in field_ids)
        after_vec = tuple(evidence.field_evidence(fid).after_mean for fid in field_ids)
        return DecodedProperty(before=before_vec, after=after_vec)

    if decoder in ("nearest_permitted_value", "exact_lookup"):
        before_mean = float(np.mean([evidence.field_evidence(fid).before_mean for fid in field_ids]))
        after_mean = float(np.mean([evidence.field_evidence(fid).after_mean for fid in field_ids]))
        tol = 1e-6 if decoder == "exact_lookup" else level_tolerance
        return DecodedProperty(
            before=_nearest_level(before_mean, canonical_levels, tol),
            after=_nearest_level(after_mean, canonical_levels, tol),
        )

    if decoder == "dominant_field_value":
        # Auto-narrowing repair for a declared region wider than the true
        # signal (the cooldown/door dilution fix): average only the fields
        # identified, from development samples alone, as ever carrying
        # non-background signal. Falls back to the full region if no
        # active-field fit was supplied (e.g. a single-field coarse tile).
        use_ids = active_field_ids if active_field_ids else field_ids
        before_mean = float(np.mean([evidence.field_evidence(fid).before_mean for fid in use_ids]))
        after_mean = float(np.mean([evidence.field_evidence(fid).after_mean for fid in use_ids]))
        return DecodedProperty(
            before=_nearest_level(before_mean, canonical_levels, level_tolerance),
            after=_nearest_level(after_mean, canonical_levels, level_tolerance),
        )

    if decoder == "categorical_template":
        before_mean = float(np.mean([evidence.field_evidence(fid).before_mean for fid in field_ids]))
        after_mean = float(np.mean([evidence.field_evidence(fid).after_mean for fid in field_ids]))
        return DecodedProperty(
            before=_nearest_level(before_mean, canonical_levels, level_tolerance),
            after=_nearest_level(after_mean, canonical_levels, level_tolerance),
        )

    if decoder == "argmax_field":
        group_of = lambda fid: _cell_group_of(compiled, fid, region)
        agg = candidate.aggregation if candidate.aggregation in ("mean", "max") else "mean"
        before_groups = aggregate_by_group(evidence, field_ids, "before_mean", group_of, agg, by_id)
        after_groups = aggregate_by_group(evidence, field_ids, "after_mean", group_of, agg, by_id)
        before_cell = argmax_group(before_groups, threshold=alive_threshold)
        after_cell = argmax_group(after_groups, threshold=alive_threshold)
        tie_before = _is_tie(before_groups, before_cell)
        tie_after = _is_tie(after_groups, after_cell)
        return DecodedProperty(before=before_cell, after=after_cell, tie_before=tie_before, tie_after=tie_after)

    if decoder in ("signed_delta_over_position", "exact_delta_over_position"):
        group_of = lambda fid: _cell_group_of(compiled, fid, region)
        agg = candidate.aggregation if candidate.aggregation in ("mean", "max") else "mean"
        before_groups = aggregate_by_group(evidence, field_ids, "before_mean", group_of, agg, by_id)
        after_groups = aggregate_by_group(evidence, field_ids, "after_mean", group_of, agg, by_id)
        before_cell = argmax_group(before_groups, threshold=alive_threshold)
        after_cell = argmax_group(after_groups, threshold=alive_threshold)
        tie_before = _is_tie(before_groups, before_cell)
        tie_after = _is_tie(after_groups, after_cell)
        delta = None if before_cell is None or after_cell is None else _subtract(after_cell, before_cell)
        return DecodedProperty(before=before_cell, after=delta, tie_before=tie_before, tie_after=tie_after)

    if decoder == "local_marker_pattern":
        lit_before = 0
        lit_after = 0
        for dy, dx in sub_patch_offsets:
            y0, x0 = region.y0 + dy, region.x0 + dx
            patch_ids = [
                fid
                for fid in field_ids
                if by_id[fid].y0 in (y0, y0 + 1) and by_id[fid].x0 in (x0, x0 + 1)
            ]
            if not patch_ids:
                continue
            before_vals = [evidence.field_evidence(fid).before_mean for fid in patch_ids]
            after_vals = [evidence.field_evidence(fid).after_mean for fid in patch_ids]
            if min(before_vals) >= dot_threshold:
                lit_before += 1
            elif lit_before == 0:
                pass
            if min(after_vals) >= dot_threshold:
                lit_after += 1
        before_id = (lit_before - 1) if lit_before > 0 else None
        after_id = (lit_after - 1) if lit_after > 0 else None
        return DecodedProperty(before=before_id, after=after_id)

    if decoder == "relation_over_decoded":
        # Relation candidates are evaluated with pre-decoded positions supplied
        # via true_before/true_after on the sample by the caller (see
        # compiler_adapters); this decoder only applies the adjacency test.
        return DecodedProperty(before=None, after=None)

    raise ValueError(f"unsupported decoder_kind: {decoder}")


def compute_dominant_fields(
    candidate: RepresentationCandidate,
    region: RegionGeometry,
    samples: Sequence[DevelopmentSample],
    *,
    signal_epsilon: float = 0.02,
) -> frozenset:
    """Development-only fit (pixel variance across dev samples; no labels,
    no privileged state): which fields in the declared region ever carry
    non-background signal. This is the auto-narrowing step that repairs a
    declared region wider than the true signal (e.g. a numeric-value tile
    that also spans always-zero background columns)."""

    compiled = _build_schema(candidate, region)
    field_ids = compiled.field_ids(region.region_id, "candidate")
    max_seen: Dict[str, float] = {fid: 0.0 for fid in field_ids}
    for sample in samples:
        evidence = _FastFieldEvidence(compiled.build_transition_evidence(sample.frame_before, sample.frame_after))
        for fid in field_ids:
            fe = evidence.field_evidence(fid)
            max_seen[fid] = max(max_seen[fid], fe.before_mean, fe.after_mean)
    active = frozenset(fid for fid, value in max_seen.items() if value > signal_epsilon)
    return active if active else frozenset(field_ids)


def _is_tie(groups: Dict, winner) -> bool:
    if winner is None or not groups:
        return False
    best_score = groups[winner]
    ties = [key for key, score in groups.items() if abs(score - best_score) < 1e-9]
    return len(ties) > 1


@dataclass(frozen=True)
class CandidateEvaluationResult:
    candidate_id: str
    requirement_id: str
    n_samples: int
    decoding_accuracy: float
    collision_rate: float
    ambiguous_sample_count: int
    ambiguous_value_pairs: Tuple[Tuple[object, object], ...]
    stability_false_change_rate: float
    complexity_cost: float
    rejection_reasons: Tuple[str, ...]
    distinct_decoded_values: int = 0
    distinct_true_values: int = 0

    @property
    def passed(self) -> bool:
        return not self.rejection_reasons

    @property
    def is_degenerate(self) -> bool:
        """True if the candidate's decoded output carries essentially no
        information (constant regardless of input) while ground truth
        genuinely varies -- the signature of evidence absent from the
        permitted frames, not merely poorly represented."""

        return self.distinct_decoded_values <= 1 and self.distinct_true_values > 1


def evaluate_candidate(
    candidate: RepresentationCandidate,
    requirement: VisualEvidenceRequirement,
    region: RegionGeometry,
    samples: Sequence[DevelopmentSample],
    *,
    canonical_levels: Sequence[float] = (),
    sub_patch_offsets: Sequence[Tuple[int, int]] = (),
    min_decoding_accuracy: float = 0.95,
    require_zero_collisions: bool = False,
) -> CandidateEvaluationResult:
    if not samples:
        raise ValueError("evaluate_candidate requires at least one development sample")

    active_field_ids = (
        compute_dominant_fields(candidate, region, samples)
        if candidate.decoder_kind == "dominant_field_value"
        else None
    )

    correct = 0
    ambiguous = 0
    false_changes = 0
    n_unrelated = 0
    decoded_to_true: Dict[object, set] = defaultdict(set)
    seen_decoded = set()
    seen_true = set()

    for sample in samples:
        decoded = decode_candidate(
            candidate,
            region,
            sample.frame_before,
            sample.frame_after,
            canonical_levels=canonical_levels,
            sub_patch_offsets=sub_patch_offsets,
            active_field_ids=active_field_ids,
        )
        if decoded.tie_before or decoded.tie_after:
            ambiguous += 1

        outcome = _score_sample(requirement.comparison, decoded, sample)
        if outcome:
            correct += 1

        decoded_to_true[decoded.after].add(_freeze(sample.true_after))
        seen_decoded.add(decoded.after)
        seen_true.add(_freeze(sample.true_after))

        if sample.is_unrelated:
            n_unrelated += 1
            if decoded.before != decoded.after:
                false_changes += 1

    n = len(samples)
    decoding_accuracy = correct / n
    ambiguous_pairs = tuple(
        (decoded_value, tuple(sorted(true_values, key=str)))
        for decoded_value, true_values in decoded_to_true.items()
        if len(true_values) > 1
    )
    collision_rate = len(ambiguous_pairs) / max(1, len(decoded_to_true))
    stability_rate = (false_changes / n_unrelated) if n_unrelated else 0.0

    reasons = []
    if decoding_accuracy < min_decoding_accuracy:
        reasons.append(f"decoding_accuracy {decoding_accuracy:.3f} < required {min_decoding_accuracy:.3f}")
    if require_zero_collisions and ambiguous_pairs:
        reasons.append(f"{len(ambiguous_pairs)} value collisions found but exactness is required")
    if ambiguous > 0 and candidate.aggregation == "max":
        reasons.append(f"{ambiguous} of {n} samples produced a tied (ambiguous) decode under max aggregation")

    return CandidateEvaluationResult(
        candidate_id=candidate.candidate_id,
        requirement_id=requirement.requirement_id,
        n_samples=n,
        decoding_accuracy=decoding_accuracy,
        collision_rate=collision_rate,
        ambiguous_sample_count=ambiguous,
        ambiguous_value_pairs=ambiguous_pairs,
        stability_false_change_rate=stability_rate,
        complexity_cost=candidate.complexity_cost,
        rejection_reasons=tuple(reasons),
        distinct_decoded_values=len(seen_decoded),
        distinct_true_values=len(seen_true),
    )


def _freeze(value):
    if isinstance(value, list):
        return tuple(value)
    return value


def _score_sample(comparison: str, decoded: DecodedProperty, sample: DevelopmentSample) -> bool:
    if comparison in ("equal", "categorical_transition", "identity_equal"):
        return decoded.after == _freeze(sample.true_after)
    if comparison == "not_equal":
        return decoded.after != _freeze(sample.true_after)
    if comparison in ("changed", "unchanged"):
        changed = decoded.before != decoded.after
        expected = bool(sample.true_after)
        return changed == expected
    if comparison == "signed_delta":
        if decoded.after is None:
            return False
        true_delta = _subtract(_freeze(sample.true_after), _freeze(sample.true_before))
        return _sign(decoded.after) == _sign(true_delta)
    if comparison == "exact_delta":
        if decoded.after is None:
            return False
        true_delta = _subtract(_freeze(sample.true_after), _freeze(sample.true_before))
        return decoded.after == true_delta
    if comparison == "relation_holds":
        return bool(decoded.after) == bool(sample.true_after)
    raise ValueError(f"unsupported comparison: {comparison}")

"""Deterministic selection policy and the three compiler outcomes.

Selection order (frozen before any evaluation-split run, per the task spec):
  1. reject candidates that fail hard validity constraints
  2. reject candidates below the minimum development evidence-preservation bar
  3. reject candidates with unresolved collisions where exactness is required
  4. rank remaining candidates by decoding accuracy, then stability, then
     lower ambiguity, then lower complexity
  5. break ties by candidate_id (deterministic, arbitrary but stable)

Selection never looks at evaluation-split data -- only the development
samples passed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from visual_transition_benchmark.compiler.candidates import RegionGeometry, RepresentationCandidate
from visual_transition_benchmark.compiler.contracts import VisualEvidenceRequirement
from visual_transition_benchmark.compiler.evaluate import (
    CandidateEvaluationResult,
    DevelopmentSample,
    evaluate_candidate,
)

CompileStatus = str  # "compiled" | "insufficient_representation" | "insufficient_observability"


@dataclass(frozen=True)
class CompiledEvidenceRepresentation:
    status: CompileStatus
    requirement_id: str
    selected_candidate: Optional[RepresentationCandidate]
    selected_evaluation: Optional[CandidateEvaluationResult]
    all_evaluations: Tuple[CandidateEvaluationResult, ...]
    selection_rationale: Tuple[str, ...]
    known_limitations: Tuple[str, ...]


def _rank_key(result: CandidateEvaluationResult):
    return (
        -result.decoding_accuracy,
        result.stability_false_change_rate,
        result.collision_rate,
        result.complexity_cost,
        result.candidate_id,
    )


def compile_requirement(
    requirement: VisualEvidenceRequirement,
    region: RegionGeometry,
    candidates: Sequence[RepresentationCandidate],
    samples: Sequence[DevelopmentSample],
    *,
    canonical_levels: Sequence[float] = (),
    sub_patch_offsets: Sequence[Tuple[int, int]] = (),
    min_decoding_accuracy: float = 0.95,
    require_zero_collisions: bool = False,
) -> CompiledEvidenceRepresentation:
    evaluations = tuple(
        evaluate_candidate(
            candidate,
            requirement,
            region,
            samples,
            canonical_levels=canonical_levels,
            sub_patch_offsets=sub_patch_offsets,
            min_decoding_accuracy=min_decoding_accuracy,
            require_zero_collisions=require_zero_collisions,
        )
        for candidate in candidates
    )
    by_id = {c.candidate_id: c for c in candidates}

    passing = [e for e in evaluations if e.passed]
    if passing:
        best = min(passing, key=_rank_key)
        rationale = [
            f"selected {best.candidate_id} (decoding_accuracy={best.decoding_accuracy:.3f}, "
            f"stability_false_change_rate={best.stability_false_change_rate:.3f}, "
            f"collision_rate={best.collision_rate:.3f}, complexity_cost={best.complexity_cost:.1f})"
        ]
        if len(passing) > 1:
            rationale.append(f"{len(passing) - 1} other candidate(s) also passed and were ranked lower")
        limitations = []
        if best.ambiguous_sample_count:
            limitations.append(f"{best.ambiguous_sample_count} development samples were ambiguous")
        return CompiledEvidenceRepresentation(
            status="compiled",
            requirement_id=requirement.requirement_id,
            selected_candidate=by_id[best.candidate_id],
            selected_evaluation=best,
            all_evaluations=evaluations,
            selection_rationale=tuple(rationale),
            known_limitations=tuple(limitations),
        )

    # Nothing passed. Distinguish "the representation lost evidence" from
    # "the permitted frames never contained the evidence at all".
    if evaluations and all(e.is_degenerate for e in evaluations):
        return CompiledEvidenceRepresentation(
            status="insufficient_observability",
            requirement_id=requirement.requirement_id,
            selected_candidate=None,
            selected_evaluation=None,
            all_evaluations=evaluations,
            selection_rationale=(
                "every candidate produced a decoded value with no variation across development "
                "samples whose true values genuinely vary -- the permitted region/frames do not "
                "contain evidence for this property, regardless of resolution or aggregation",
            ),
            known_limitations=("insufficient_observability: no representation can recover this property",),
        )

    best_attempt = min(evaluations, key=_rank_key) if evaluations else None
    rationale = ["no candidate met the acceptance bar"]
    if best_attempt is not None:
        rationale.append(
            f"closest candidate {best_attempt.candidate_id}: decoding_accuracy={best_attempt.decoding_accuracy:.3f}, "
            f"rejection_reasons={list(best_attempt.rejection_reasons)}"
        )
    return CompiledEvidenceRepresentation(
        status="insufficient_representation",
        requirement_id=requirement.requirement_id,
        selected_candidate=None,
        selected_evaluation=best_attempt,
        all_evaluations=evaluations,
        selection_rationale=tuple(rationale),
        known_limitations=("insufficient_representation: evidence may exist but no candidate in the bounded search preserved it",),
    )

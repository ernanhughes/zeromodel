from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from zeromodel.perception.transition_analysis import (
    TransitionActionDeclarationDTO,
    VisualTransitionReaderTraceDTO,
)
from zeromodel.perception.transition_evidence import TransitionEvidenceVPMDTO
from zeromodel.perception.representation import encode_source_array
from zeromodel.vision import VisualFeatureSpec, visual_input_digest, visual_raw_input_digest

from visual_transition_benchmark import zeromodel_adapter as component_zm
from visual_transition_benchmark.adjudication._json import digest
from visual_transition_benchmark.adjudication.candidate_universe import candidate_rows
from visual_transition_benchmark.adjudication.contracts import (
    build_candidate_contract,
    contract_matches_observation,
)


@dataclass(frozen=True)
class RuntimeAdjudicationInput:
    case_id: str
    visual_decision: object
    candidate_universe: str
    evidence_mode: str
    addressed_observation: np.ndarray
    addressed_observation_transform_id: str
    feature_spec: VisualFeatureSpec
    frame_before: np.ndarray
    frame_after: np.ndarray
    action: TransitionActionDeclarationDTO
    transition_evidence: TransitionEvidenceVPMDTO


@dataclass(frozen=True)
class CandidateAdjudicationResult:
    candidate_id: str
    row_id: str
    transition_consistent: bool
    reason_codes: tuple[str, ...]
    expected_transition_signature_id: str
    observed_transition_signature_id: str
    contract_id: str
    analysis_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "row_id": self.row_id,
            "transition_consistent": self.transition_consistent,
            "reason_codes": list(self.reason_codes),
            "expected_transition_signature_id": self.expected_transition_signature_id,
            "observed_transition_signature_id": self.observed_transition_signature_id,
            "contract_id": self.contract_id,
            "analysis_id": self.analysis_id,
        }


@dataclass(frozen=True)
class AddressTransitionAdjudicationResult:
    result_id: str
    case_id: str
    addressed_row_id: str | None
    addressed_action: str | None
    action_id: str
    transition_evidence_id: str
    addressed_observation_digest: str | None
    candidate_universe: str
    evidence_mode: str
    candidate_results: tuple[CandidateAdjudicationResult, ...]
    consistent_candidate_ids: tuple[str, ...]
    contradicted_candidate_ids: tuple[str, ...]
    addressed_candidate_status: str
    candidate_count_before: int
    candidate_count_after: int
    candidate_reduction: bool
    runtime_adjudication_status: str
    transition_signature_collision_ids: tuple[str, ...]
    observability_status: str
    reason_codes: tuple[str, ...]

    def runtime_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "addressed_row_id": self.addressed_row_id,
            "addressed_action": self.addressed_action,
            "action_id": self.action_id,
            "transition_evidence_id": self.transition_evidence_id,
            "addressed_observation_digest": self.addressed_observation_digest,
            "candidate_universe": self.candidate_universe,
            "evidence_mode": self.evidence_mode,
            "candidate_results": [item.to_dict() for item in self.candidate_results],
            "consistent_candidate_ids": list(self.consistent_candidate_ids),
            "contradicted_candidate_ids": list(self.contradicted_candidate_ids),
            "addressed_candidate_status": self.addressed_candidate_status,
            "candidate_count_before": self.candidate_count_before,
            "candidate_count_after": self.candidate_count_after,
            "candidate_reduction": self.candidate_reduction,
            "runtime_adjudication_status": self.runtime_adjudication_status,
            "transition_signature_collision_ids": list(
                self.transition_signature_collision_ids
            ),
            "observability_status": self.observability_status,
            "reason_codes": list(self.reason_codes),
        }

    def to_dict(self) -> dict[str, object]:
        return {"result_id": self.result_id, **self.runtime_dict()}


def _transition_evidence_matches_frames(
    transition: TransitionEvidenceVPMDTO,
    frame_before: np.ndarray,
    frame_after: np.ndarray,
) -> bool:
    before = encode_source_array(
        np.ascontiguousarray(frame_before, dtype=np.uint8), component_zm._SPEC
    )
    after = encode_source_array(
        np.ascontiguousarray(frame_after, dtype=np.uint8), component_zm._SPEC
    )
    return (
        transition.before_source_vpm_id == before.source_vpm_id
        and transition.after_source_vpm_id == after.source_vpm_id
    )


def _collision_ids(results: tuple[CandidateAdjudicationResult, ...]) -> tuple[str, ...]:
    by_signature: dict[str, list[str]] = {}
    for item in results:
        by_signature.setdefault(item.expected_transition_signature_id, []).append(
            item.row_id
        )
    groups = [
        digest({"transition_signature_collision": sorted(rows)})
        for rows in by_signature.values()
        if len(rows) > 1
    ]
    return tuple(sorted(groups))


def _status(
    *,
    decision: object,
    addressed_row: str | None,
    consistent: tuple[str, ...],
    before_count: int,
    collisions: tuple[str, ...],
    evidence_mode: str,
) -> tuple[str, str, tuple[str, ...]]:
    if not bool(getattr(decision, "accepted")):
        return "reader_rejected", "not_evaluated", ("reader_rejected",)
    if not bool(getattr(decision, "policy_executed")):
        return "policy_not_executed", "not_evaluated", ("policy_not_executed",)
    if not consistent:
        return "address_contradicted", "contradicted", ("all_candidates_contradicted",)
    addressed_retained = addressed_row in consistent
    if not addressed_retained:
        if len(consistent) == 1:
            return "unique_transition_consistent_candidate", "contradicted", ()
        return "candidate_set_narrowed", "contradicted", ()
    if len(consistent) == before_count:
        if collisions:
            return (
                "transition_signature_collision",
                "retained",
                ("transition_signature_collision",),
            )
        return "no_visual_effect_unresolved", "retained", ("no_effect_transition",)
    if collisions and addressed_retained:
        return (
            "transition_signature_collision",
            "retained",
            ("transition_signature_collision",),
        )
    if len(consistent) == 1:
        return "address_transition_consistent", "retained", ()
    if evidence_mode == "component":
        return "action_equivalent_unresolved", "retained", ()
    return "candidate_set_narrowed", "retained", ()


def adjudicate_address_transition(
    input_data: RuntimeAdjudicationInput,
) -> AddressTransitionAdjudicationResult:
    decision = input_data.visual_decision
    trace = VisualTransitionReaderTraceDTO.from_visual_decision(decision)
    addressed_row = trace.matched_row_id
    addressed_action = getattr(decision, "action", None)
    reasons: list[str] = []
    candidate_results: list[CandidateAdjudicationResult] = []

    if not _transition_evidence_matches_frames(
        input_data.transition_evidence, input_data.frame_before, input_data.frame_after
    ):
        reasons.append("invalid_transition_evidence")
        rows: tuple[str, ...] = ()
    else:
        rows = candidate_rows(input_data.candidate_universe, decision)

    addressed_raw_digest = visual_raw_input_digest(
        input_data.addressed_observation,
        input_data.feature_spec,
    )
    addressed_canonical_digest = visual_input_digest(
        input_data.addressed_observation,
        input_data.feature_spec,
    )
    if addressed_raw_digest != trace.raw_input_digest:
        reasons.append("reader_observation_mismatch")
    if addressed_canonical_digest != trace.canonical_input_digest:
        reasons.append("reader_observation_mismatch")
    before_canonical_digest = visual_input_digest(
        input_data.frame_before,
        input_data.feature_spec,
    )
    if (
        addressed_canonical_digest != before_canonical_digest
        and input_data.addressed_observation_transform_id != "noncanonical_exact_background_pixel"
    ):
        reasons.append("stale_before_observation")

    if len(rows) != len(set(rows)):
        reasons.append("duplicate_candidate_rows")
    rows = tuple(sorted(set(rows)))

    if trace.policy_executed:
        if input_data.action.action_type != addressed_action:
            reasons.append("wrong_action_identity")
        if input_data.action.payload.get("row_id") != addressed_row:
            reasons.append("wrong_action_payload")

    if reasons:
        rows = ()

    for row_id in rows:
        contract = build_candidate_contract(
            row_id,
            str(addressed_action),
            evidence_mode=input_data.evidence_mode,  # type: ignore[arg-type]
        )
        ok, result_reasons, expected_signature, observed_signature, analysis_id = contract_matches_observation(
            contract,
            input_data.frame_before,
            input_data.frame_after,
            transition_evidence=input_data.transition_evidence,
            action=input_data.action,
            reader_trace=trace,
        )
        candidate_results.append(
            CandidateAdjudicationResult(
                candidate_id=contract.candidate_id,
                row_id=row_id,
                transition_consistent=ok,
                reason_codes=result_reasons,
                expected_transition_signature_id=expected_signature,
                observed_transition_signature_id=observed_signature,
                contract_id=contract.candidate_id,
                analysis_id=analysis_id,
            )
        )

    ordered_results = tuple(sorted(candidate_results, key=lambda item: item.row_id))
    consistent = tuple(
        item.row_id for item in ordered_results if item.transition_consistent
    )
    contradicted = tuple(
        item.row_id for item in ordered_results if not item.transition_consistent
    )
    collisions = _collision_ids(
        tuple(item for item in ordered_results if item.transition_consistent)
    )
    runtime_status, addressed_status, status_reasons = _status(
        decision=decision,
        addressed_row=addressed_row,
        consistent=consistent,
        before_count=len(rows),
        collisions=collisions,
        evidence_mode=input_data.evidence_mode,
    )
    reasons.extend(status_reasons)
    observability = (
        "insufficient_observability"
        if runtime_status
        in {"transition_signature_collision", "no_visual_effect_unresolved"}
        else "observable"
    )
    if (
        input_data.evidence_mode == "component"
        and runtime_status == "action_equivalent_unresolved"
    ):
        observability = "insufficient_representation"
    payload: Mapping[str, object] = {
        "case_id": input_data.case_id,
        "addressed_row_id": addressed_row,
        "addressed_action": addressed_action,
        "action_id": input_data.action.action_id,
        "transition_evidence_id": input_data.transition_evidence.transition_evidence_id,
        "addressed_observation_digest": addressed_canonical_digest,
        "candidate_universe": input_data.candidate_universe,
        "evidence_mode": input_data.evidence_mode,
        "candidate_results": [item.to_dict() for item in ordered_results],
        "consistent_candidate_ids": consistent,
        "contradicted_candidate_ids": contradicted,
        "addressed_candidate_status": addressed_status,
        "candidate_count_before": len(rows),
        "candidate_count_after": len(consistent),
        "candidate_reduction": len(consistent) < len(rows),
        "runtime_adjudication_status": runtime_status
        if not reasons or reasons == list(status_reasons)
        else "invalid_transition_evidence",
        "transition_signature_collision_ids": collisions,
        "observability_status": observability,
        "reason_codes": tuple(sorted(set(reasons))),
    }
    return AddressTransitionAdjudicationResult(
        result_id=digest(payload),
        case_id=input_data.case_id,
        addressed_row_id=addressed_row,
        addressed_action=addressed_action,
        action_id=input_data.action.action_id,
        transition_evidence_id=input_data.transition_evidence.transition_evidence_id,
        addressed_observation_digest=addressed_canonical_digest,
        candidate_universe=input_data.candidate_universe,
        evidence_mode=input_data.evidence_mode,
        candidate_results=ordered_results,
        consistent_candidate_ids=consistent,
        contradicted_candidate_ids=contradicted,
        addressed_candidate_status=addressed_status,
        candidate_count_before=len(rows),
        candidate_count_after=len(consistent),
        candidate_reduction=len(consistent) < len(rows),
        runtime_adjudication_status=(
            runtime_status
            if not reasons or reasons == list(status_reasons)
            else "invalid_transition_evidence"
        ),
        transition_signature_collision_ids=collisions,
        observability_status=observability,
        reason_codes=tuple(sorted(set(reasons))),
    )

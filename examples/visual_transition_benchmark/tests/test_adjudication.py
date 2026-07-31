from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception.representation import encode_source_array
from zeromodel.perception.transition_analysis import TransitionActionDeclarationDTO
from zeromodel.perception.transition_evidence import build_transition_evidence_vpm

from visual_transition_benchmark import zeromodel_adapter as component_zm
from visual_transition_benchmark.adjudication.adjudicator import (
    RuntimeAdjudicationInput,
    adjudicate_address_transition,
)
from visual_transition_benchmark.adjudication.candidate_universe import candidate_rows
from visual_transition_benchmark.adjudication.contracts import build_candidate_contract
from visual_transition_benchmark.adjudication.corpus import build_case_corpus


def _transition(before, after):
    return build_transition_evidence_vpm(
        encode_source_array(
            np.ascontiguousarray(before, dtype=np.uint8), component_zm._SPEC
        ),
        encode_source_array(
            np.ascontiguousarray(after, dtype=np.uint8), component_zm._SPEC
        ),
        component_zm.FIELD_SCHEMA,
        annotations=component_zm.ANNOTATIONS_TUPLE,
        change_threshold=component_zm.CHANGE_THRESHOLD,
    )


def _run(case):
    action = TransitionActionDeclarationDTO.create(
        action_type=str(case.addressed_action or "NONE"),
        payload={"row_id": case.visual_decision.matched_row_id},
        provider_id="visual-sign-reader",
    )
    return adjudicate_address_transition(
        RuntimeAdjudicationInput(
            case_id=case.case_id,
            visual_decision=case.visual_decision,
            candidate_universe=case.candidate_universe,
            evidence_mode=case.evidence_mode,
            addressed_observation=case.observed_frame,
            addressed_observation_transform_id=case.observation_transform_id,
            feature_spec=case.feature_spec,
            frame_before=case.true_before_frame,
            frame_after=case.true_after_frame,
            action=action,
            transition_evidence=_transition(
                case.true_before_frame, case.true_after_frame
            ),
        )
    )


def _case(prefix: str, *, universe: str = "reader_local", mode: str = "component"):
    for case in build_case_corpus():
        if case.case_id == f"{prefix}-{universe}-{mode}":
            return case
    raise AssertionError(prefix)


def test_candidate_ordering_determinism_and_identity():
    case = _case("exact-codeword-canonical")
    first = candidate_rows(case.candidate_universe, case.visual_decision)
    second = candidate_rows(case.candidate_universe, case.visual_decision)
    assert first == tuple(sorted(first))
    assert first == second
    assert len(first) == len(set(first))


def test_contract_identity_is_deterministic():
    left = build_candidate_contract(
        "tank=0|target=0|cooldown=0", "FIRE", evidence_mode="component"
    )
    right = build_candidate_contract(
        "tank=0|target=0|cooldown=0", "FIRE", evidence_mode="component"
    )
    changed = build_candidate_contract(
        "tank=0|target=0|cooldown=0", "FIRE", evidence_mode="value"
    )
    assert left.candidate_id == right.candidate_id
    assert left.candidate_id != changed.candidate_id


def test_correct_canonical_and_noncanonical_address_retained():
    assert _run(_case("canonical-only-accepted")).addressed_candidate_status == "retained"
    assert (
        _run(_case("exact-codeword-noncanonical-correct")).addressed_candidate_status
        == "retained"
    )


def test_genuine_true_observation_perturbations_do_not_create_wrong_row_aliases():
    wrong = [
        case
        for case in build_case_corpus()
        if case.visual_decision.policy_executed
        and case.visual_decision.matched_row_id != case.true_row_id
    ]
    assert wrong == []


def test_action_equivalent_address_remains_unresolved():
    result = _run(_case("no-effect-unresolved", universe="policy_action"))
    assert result.runtime_adjudication_status in {
        "action_equivalent_unresolved",
        "transition_signature_collision",
        "no_visual_effect_unresolved",
    }


def test_transition_signature_collision_and_no_effect_reported():
    result = _run(_case("no-effect-unresolved", universe="policy_action"))
    assert result.runtime_adjudication_status in {
        "transition_signature_collision",
        "no_visual_effect_unresolved",
    }
    assert result.observability_status == "insufficient_observability"


def test_insufficient_observability_reported():
    result = _run(_case("insufficient-observability", universe="policy_action"))
    assert result.observability_status in {
        "insufficient_observability",
        "insufficient_representation",
    }


def test_reader_rejection_and_evidence_only_produce_no_execution_adjudication():
    rejected = next(
        case
        for case in build_case_corpus()
        if case.case_id == "reader-rejected-canonical-only"
    )
    evidence_only = next(
        case
        for case in build_case_corpus()
        if case.case_id == "evidence-only-no-execution"
    )
    assert _run(rejected).runtime_adjudication_status == "reader_rejected"
    assert _run(evidence_only).runtime_adjudication_status == "policy_not_executed"


def test_wrong_action_identity_and_payload_rejected():
    case = _case("canonical-only-accepted")
    wrong_action = TransitionActionDeclarationDTO.create(
        action_type="LEFT", payload={"row_id": "wrong"}
    )
    result = adjudicate_address_transition(
        RuntimeAdjudicationInput(
            case_id=case.case_id,
            visual_decision=case.visual_decision,
            candidate_universe=case.candidate_universe,
            evidence_mode=case.evidence_mode,
            addressed_observation=case.observed_frame,
            addressed_observation_transform_id=case.observation_transform_id,
            feature_spec=case.feature_spec,
            frame_before=case.true_before_frame,
            frame_after=case.true_after_frame,
            action=wrong_action,
            transition_evidence=_transition(
                case.true_before_frame, case.true_after_frame
            ),
        )
    )
    assert result.runtime_adjudication_status == "invalid_transition_evidence"
    assert {"wrong_action_identity", "wrong_action_payload"} <= set(result.reason_codes)


def test_wrong_transition_evidence_and_swap_detected():
    case = _case("canonical-only-accepted")
    result = adjudicate_address_transition(
        RuntimeAdjudicationInput(
            case_id=case.case_id,
            visual_decision=case.visual_decision,
            candidate_universe=case.candidate_universe,
            evidence_mode=case.evidence_mode,
            addressed_observation=case.observed_frame,
            addressed_observation_transform_id=case.observation_transform_id,
            feature_spec=case.feature_spec,
            frame_before=case.true_before_frame,
            frame_after=case.true_after_frame,
            action=TransitionActionDeclarationDTO.create(
                action_type=str(case.addressed_action),
                payload={"row_id": case.visual_decision.matched_row_id},
            ),
            transition_evidence=_transition(
                case.true_after_frame, case.true_before_frame
            ),
        )
    )
    assert result.runtime_adjudication_status == "invalid_transition_evidence"
    assert "invalid_transition_evidence" in result.reason_codes


def test_decision_bound_to_exact_addressed_observation():
    case = _case("canonical-only-accepted")
    result = adjudicate_address_transition(
        RuntimeAdjudicationInput(
            case_id=case.case_id,
            visual_decision=case.visual_decision,
            candidate_universe=case.candidate_universe,
            evidence_mode=case.evidence_mode,
            addressed_observation=np.array(case.true_after_frame, copy=True),
            addressed_observation_transform_id="canonical",
            feature_spec=case.feature_spec,
            frame_before=case.true_before_frame,
            frame_after=case.true_after_frame,
            action=TransitionActionDeclarationDTO.create(
                action_type=str(case.addressed_action),
                payload={"row_id": case.visual_decision.matched_row_id},
                provider_id="visual-sign-reader",
            ),
            transition_evidence=_transition(
                case.true_before_frame, case.true_after_frame
            ),
        )
    )
    assert result.runtime_adjudication_status == "invalid_transition_evidence"
    assert {"reader_observation_mismatch", "stale_before_observation"} <= set(
        result.reason_codes
    )


def test_serialization_round_trip_runtime_dict():
    result = _run(_case("canonical-only-accepted"))
    assert result.result_id
    assert result.runtime_dict()["case_id"] == result.case_id


def test_true_row_absent_from_runtime_input_and_label_mutation_noop():
    case = _case("exact-codeword-noncanonical-correct")
    result = _run(case)
    mutated_truth = replace(case, true_row_id="tank=6|target=6|cooldown=1")
    assert _run(mutated_truth).runtime_dict() == result.runtime_dict()


def test_after_frame_mutation_changes_runtime_output():
    case = _case("canonical-only-accepted")
    result = _run(case)
    mutated_after = np.array(case.true_after_frame, copy=True)
    mutated_after[6, 0] = 90
    mutated = replace(case, true_after_frame=mutated_after)
    assert _run(mutated).runtime_dict() != result.runtime_dict()


def test_wrong_reader_trace_rejected_by_production_trace_invariant():
    case = _case("canonical-only-accepted")
    with pytest.raises(Exception):
        replace(case.visual_decision, policy_executed=True, matched_row_id=None)

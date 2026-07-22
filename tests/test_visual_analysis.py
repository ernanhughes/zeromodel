from __future__ import annotations

from zeromodel.observation.visual_address import VisualAddressDecision
from research.visual.visual_analysis import (
    EXPLORATORY_TEST_REUSE_WARNING,
    GLOBAL_SCORE_THRESHOLD_CURVE_TYPE,
    analyze_trace_sets,
    global_score_threshold_curve,
    paired_top1_outcomes,
    trace_points,
)
from research.visual.visual_experiment import (
    EXPECTED_ACCEPT,
    EXPECTED_REJECT,
    IMPOSSIBILITY_CONTROL,
    VisualEvaluationTrace,
)


def _decision(*, accepted: bool, nearest_row_id: str | None, nearest_score: float | None) -> VisualAddressDecision:
    return VisualAddressDecision(
        accepted=accepted,
        reason="accepted" if accepted else "rejected",
        observation_digest="obs",
        representation_digest="repr",
        provider_kind="test",
        provider_version="v1",
        score_semantics="similarity",
        address_artifact_id="addr",
        calibration_artifact_id="cal",
        policy_artifact_id="policy",
        nearest_row_id=nearest_row_id,
        nearest_score=nearest_score,
        second_row_id=None,
        second_score=None,
        ambiguity_measure=None,
        matched_row_id=nearest_row_id if accepted else None,
    )


def _trace(observation_id, family_id, expected, row, action, top1_row, top1_action, score):
    accepted = top1_row is not None and expected == EXPECTED_ACCEPT
    decision = _decision(
        accepted=accepted,
        nearest_row_id=top1_row,
        nearest_score=score,
    )
    return VisualEvaluationTrace(
        observation_id=observation_id,
        family_id=family_id,
        split="final_evaluation",
        expected_disposition=expected,
        expected_accept=(
            True if expected == EXPECTED_ACCEPT else False if expected == EXPECTED_REJECT else None
        ),
        expected_row_id=row,
        expected_action_id=action,
        predicted_row_id=top1_row if accepted else None,
        predicted_action_id=top1_action if accepted else None,
        top1_row_id=top1_row,
        top1_action_id=top1_action,
        decision=decision,
    ).to_dict()


def test_trace_point_contract_uses_canonical_dispositions() -> None:
    accept = trace_points(
        (_trace("a", "fam", EXPECTED_ACCEPT, "r1", "left", "r1", "left", 0.9),)
    )[0]
    reject = trace_points(
        (_trace("b", "fam", EXPECTED_REJECT, None, None, "r1", "left", 0.4),)
    )[0]
    control = trace_points(
        (_trace("c", "fam", IMPOSSIBILITY_CONTROL, "r2", "right", "r2", "right", 0.8),)
    )[0]

    assert accept.expected_accept and not accept.expected_reject and not accept.information_theoretic_control
    assert reject.expected_reject and not reject.expected_accept and not reject.information_theoretic_control
    assert control.information_theoretic_control and not control.expected_accept and not control.expected_reject


def test_global_threshold_curve_separates_ranking_from_thresholding_and_excludes_controls() -> None:
    points = trace_points(
        (
            _trace("a", "benign", EXPECTED_ACCEPT, "r1", "left", "r1", "left", 0.9),
            _trace("b", "benign", EXPECTED_ACCEPT, "r2", "right", "r1", "left", 0.4),
            _trace("c", "ood", EXPECTED_REJECT, None, None, "r1", "left", 0.5),
            _trace("d", "control", IMPOSSIBILITY_CONTROL, "r3", "stay", "r3", "stay", 0.95),
        )
    )
    rows = global_score_threshold_curve(points, (0.0, 0.6))
    assert rows[0]["curve_type"] == GLOBAL_SCORE_THRESHOLD_CURVE_TYPE
    assert rows[0]["coverage"] == 1.0
    assert rows[0]["accepted_row_precision"] == 0.5
    assert rows[0]["false_acceptance_rate"] == 1.0
    assert rows[0]["benign_count"] == 2
    assert rows[0]["rejection_count"] == 1
    assert rows[1]["coverage"] == 0.5
    assert rows[1]["accepted_row_precision"] == 1.0
    assert rows[1]["false_acceptance_rate"] == 0.0


def test_paired_top1_counts_off_diagonal_outcomes() -> None:
    left = trace_points(
        (
            _trace("a", "x", EXPECTED_ACCEPT, "r1", "left", "r1", "left", 0.8),
            _trace("b", "x", EXPECTED_ACCEPT, "r2", "right", "r2", "right", 0.7),
        )
    )
    right = trace_points(
        (
            _trace("a", "x", EXPECTED_ACCEPT, "r1", "left", "r2", "left", 0.9),
            _trace("b", "x", EXPECTED_ACCEPT, "r2", "right", "r2", "right", 0.9),
        )
    )
    paired = paired_top1_outcomes(left, right)
    assert paired["row"] == {
        "both_correct": 1,
        "left_only": 1,
        "right_only": 0,
        "neither": 0,
    }
    assert paired["action"]["both_correct"] == 2


def test_analysis_marks_curves_as_exploratory() -> None:
    traces = {
        "B": (
            _trace("a", "x", EXPECTED_ACCEPT, "r1", "left", "r1", "left", 0.8),
        ),
        "D": (
            _trace("a", "x", EXPECTED_ACCEPT, "r1", "left", "r2", "left", 0.9),
        ),
    }
    report = analyze_trace_sets(traces)
    assert report["status"] == "exploratory"
    assert report["warning"] == EXPLORATORY_TEST_REUSE_WARNING
    assert report["paired_B_D"]["row"]["left_only"] == 1
    assert "global_score_threshold_curve" in report["systems"]["B"]

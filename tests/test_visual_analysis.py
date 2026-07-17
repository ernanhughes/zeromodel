from zeromodel.visual_analysis import (
    EXPLORATORY_TEST_REUSE_WARNING,
    analyze_trace_sets,
    operating_curve,
    paired_top1_outcomes,
    trace_points,
)


def _trace(observation_id, family_id, expected, row, action, predicted_row, predicted_action, score):
    return {
        "observation_id": observation_id,
        "family_id": family_id,
        "expected_disposition": expected,
        "expected_row_id": row,
        "expected_action_id": action,
        "top1_row_id": predicted_row,
        "top1_action_id": predicted_action,
        "predicted_row_id": predicted_row,
        "predicted_action_id": predicted_action,
        "decision": {
            "nearest_row_id": predicted_row,
            "nearest_score": score,
        },
    }


def test_operating_curve_separates_ranking_from_thresholding():
    points = trace_points(
        (
            _trace("a", "benign", "accept", "r1", "left", "r1", "left", 0.9),
            _trace("b", "benign", "accept", "r2", "right", "r1", "left", 0.4),
            _trace("c", "ood", "reject", None, None, "r1", "left", 0.5),
        )
    )
    rows = operating_curve(points, (0.0, 0.6))
    assert rows[0]["coverage"] == 1.0
    assert rows[0]["accepted_row_precision"] == 0.5
    assert rows[0]["false_acceptance_rate"] == 1.0
    assert rows[1]["coverage"] == 0.5
    assert rows[1]["accepted_row_precision"] == 1.0
    assert rows[1]["false_acceptance_rate"] == 0.0


def test_paired_top1_counts_off_diagonal_outcomes():
    left = trace_points(
        (
            _trace("a", "x", "accept", "r1", "left", "r1", "left", 0.8),
            _trace("b", "x", "accept", "r2", "right", "r2", "right", 0.7),
        )
    )
    right = trace_points(
        (
            _trace("a", "x", "accept", "r1", "left", "r2", "left", 0.9),
            _trace("b", "x", "accept", "r2", "right", "r2", "right", 0.9),
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


def test_analysis_marks_curves_as_exploratory():
    traces = {
        "B": (
            _trace("a", "x", "accept", "r1", "left", "r1", "left", 0.8),
        ),
        "D": (
            _trace("a", "x", "accept", "r1", "left", "r2", "left", 0.9),
        ),
    }
    report = analyze_trace_sets(traces)
    assert report["status"] == "exploratory"
    assert report["warning"] == EXPLORATORY_TEST_REUSE_WARNING
    assert report["paired_B_D"]["row"]["left_only"] == 1

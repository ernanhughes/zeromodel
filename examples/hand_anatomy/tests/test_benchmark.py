import numpy as np
import pytest

from examples.hand_anatomy import synthetic as sy
from examples.hand_anatomy.assessment import DecisionContract
from examples.hand_anatomy.benchmark import (
    baseline_preds,
    calibrate_baseline,
    evaluate_case,
    leakage_checks,
    method_preds,
    score,
    summarize,
    wilson,
)
from examples.hand_anatomy.pose_library import build_library


@pytest.fixture(scope="module")
def splits():
    return sy.build_cases()


def test_fixed_split_membership_and_no_leakage(splits):
    again = sy.build_cases()
    for k in splits:
        assert [c.case_id for c in splits[k]] == [c.case_id for c in again[k]]
    leak = leakage_checks(splits, build_library())
    assert leak["ok"] and leak["case_id_overlap"] == 0 and leak["mask_overlap_calibration_test"] == 0
    assert leak["mask_overlap_test_library"] == 0


def test_every_defect_class_and_label_present_in_both_splits(splits):
    for split in ("calibration", "test"):
        kinds = {(c.group, c.kind) for c in splits[split]}
        for k in sy.MALFORMED_KINDS:
            assert ("independent", k) in kinds
        for k in sy.VALID_KINDS:
            assert ("independent", k) in kinds
        assert any(c.ambiguous for c in splits["calibration"] + splits["test"])
    assert {c.group for c in splits["test"]} == {"independent", "same_renderer"}


def _rec(label, iou=0.9, ambiguous=False, group="independent", kind="k"):
    rows = [{"template_id": "t", "name": "t", "protrusions_template": 4,
             "metrics": dict(silhouette_agreement=iou, boundary_agreement=0.8, geometry_agreement=0.9,
                             observed_explainability=0.99, protrusion_explainability=1.0,
                             missing_explainability=1.0, structure_agreement=1.0)}]
    return {"label": label, "best_iou": iou, "rows": rows, "ambiguous": ambiguous, "group": group, "kind": kind,
            "obs_level": dict(connectivity=1.0, border_contact_fraction=0.0, protrusions_observed=4)}


def test_metrics_recompute_from_case_results_and_zero_denominators():
    recs = [_rec("malformed"), _rec("malformed"), _rec("valid"), _rec("valid")]
    s = score(recs, ["pos", "abstain", "neg", "pos"])
    assert (s["tp"], s["fp"], s["tn"], s["fn"]) == (1, 1, 1, 0)
    assert s["abstained_malformed"] == 1 and s["coverage"] == 0.75
    assert s["recall_overall"] == 0.5 and s["precision"] == 0.5
    assert score([], [])["precision"] is None and score(recs[:2], ["abstain"] * 2)["accuracy_on_accepted"] is None
    assert wilson(0, 0) is None and wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1]


def test_baseline_and_method_use_identical_cases_and_negatives_are_kept():
    recs = [_rec("malformed", 0.6, kind="six"), _rec("malformed", 0.95, kind="six"), _rec("valid", 0.9, kind="open")]
    a = baseline_preds(recs, 0.8)
    b = method_preds(recs, DecisionContract())
    assert len(a) == len(b) == len(recs)
    summary = summarize(recs, b)
    assert summary["all"]["n"] == 3 and "independent/six" in summary["per_kind"]  # hard cases stay in the table
    assert summary["per_kind"]["independent/six"]["n"] == 2
    assert a == ["pos", "neg", "neg"]  # the 0.95-IoU malformed hand is a retained miss for the baseline


def test_calibration_uses_only_supplied_split():
    cal = [_rec("malformed", 0.6), _rec("malformed", 0.65), _rec("valid", 0.9), _rec("valid", 0.92)]
    t, info = calibrate_baseline(cal)
    assert 0.65 < t <= 0.9 and info["fpr_cap"] == 0.1


def test_evaluate_case_is_deterministic_and_vpm_equals_direct_decision(splits):
    case = next(c for c in splits["test"] if c.kind == "open_palm")
    a, b = evaluate_case(case), evaluate_case(case)
    assert a["vpm_id"] == b["vpm_id"] and a["vpm_matches_direct"] is True
    assert a["rows"][0]["metrics"] == b["rows"][0]["metrics"]
    assert a["label"] == "valid" and np.isfinite(a["best_iou"])

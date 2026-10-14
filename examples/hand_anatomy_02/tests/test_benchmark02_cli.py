import json

import pytest

from examples.hand_anatomy_02 import benchmark02 as b2
from examples.hand_anatomy_02.cli import main
from examples.hand_anatomy_02.vpm02 import METRICS02


def test_threshold_at_fpr_and_sign_test():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    t = b2.threshold_at_fpr(vals, 0.10)
    assert sum(v > t for v in vals) <= 1 and t == 0.9
    assert b2.threshold_at_fpr([0.5], 0.10) == 0.5
    assert b2.sign_test_p(0, 0) == 1.0
    assert b2.sign_test_p(10, 0) == pytest.approx(2 / 1024)
    assert b2.sign_test_p(5, 5) == 1.0


def _rec(case_id, role, pair_id, label, a_iou, c_res, subtype="outer", split="x", indist=False, category="extra_finger"):
    metrics = {m: 0.9 for m in METRICS02}
    metrics.update(library_iou=1 - a_iou, silhouette_agreement=1 - a_iou,
                   finger_residual_explainability=1 - c_res, library_finger_residual_explainability=1 - c_res)
    return {"case_id": case_id, "split": split, "role": role, "label": label, "pair_id": pair_id,
            "category": category, "subtype": subtype, "indistinguishable": indist,
            "rows": [{"template_id": "t", "name": "t", "metrics": metrics}]}


def _synthetic(n_pairs, c_gain):
    cal, test = [], []
    for i in range(20):
        cal.append(_rec("cv%d" % i, "valid_pool", "", "valid", 0.1 + 0.01 * i, 0.1 + 0.01 * i, split="c"))
    for i in range(n_pairs):
        pid = "p%d" % i
        test.append(_rec("v%d" % i, "valid_twin", pid, "valid", 0.15, 0.15))
        # the baseline misses every defect; C catches the first c_gain fraction
        test.append(_rec("d%d" % i, "defect", pid, "malformed", 0.2, 0.9 if i < c_gain * n_pairs else 0.15))
    return cal, test


def test_rule_accepts_clear_gain_and_rejects_otherwise():
    cal, test = _synthetic(48, 0.6)
    ev = b2.evaluate(cal, test)
    assert ev["methods"]["C_residual"]["meets_preregistered_rule"] is True
    assert ev["methods"]["C_residual"]["primary"]["pairs"] == 48
    cal, test = _synthetic(48, 0.05)
    assert b2.evaluate(cal, test)["methods"]["C_residual"]["meets_preregistered_rule"] is False
    cal, test = _synthetic(30, 0.9)  # too few pairs: cannot accept
    assert b2.evaluate(cal, test)["methods"]["C_residual"]["meets_preregistered_rule"] is False


def test_indistinguishable_pairs_are_excluded_from_primary_and_not_credited():
    cal, test = _synthetic(45, 0.6)
    for i in range(10):
        pid = "h%d" % i
        test.append(_rec("hv%d" % i, "valid_twin", pid, "valid", 0.15, 0.15, subtype="hidden", indist=True))
        test.append(_rec("hd%d" % i, "defect", pid, "malformed", 0.2, 0.9, subtype="hidden", indist=True))
    ev = b2.evaluate(cal, test)
    assert ev["methods"]["C_residual"]["primary"]["pairs"] == 45
    assert "extra_finger/hidden[indistinguishable]" in ev["methods"]["C_residual"]["by_group"]


def test_frozen_rule_constants_match_preregistration():
    text = (b2.Path(b2.__file__).with_name("PREREGISTRATION.md")).read_text(encoding="utf-8")
    assert b2.RULE["min_recall_gain_vs_iou"] == 0.15 and ">= 0.15" in text
    assert b2.FPR_CAP == 0.10 and "<= 0.10" in text and b2.RULE["max_test_fpr"] == pytest.approx(0.15)
    assert "p < 0.05" in text and b2.RULE["min_primary_pairs"] == 40


def test_cli_help_and_bad_input_exit_codes(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0 and "pair-sheet" in capsys.readouterr().out
    assert main(["inspect", "--mask", str(tmp_path / "nope.png"), "--output", str(tmp_path / "o")]) == 2
    assert main(["validate-report", "--report", str(tmp_path / "nope.json")]) == 2
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({"schema_version": "hand-anatomy-report/2"}))
    assert main(["validate-report", "--report", str(bad)]) == 1

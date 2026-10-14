import copy
import json

import numpy as np
import pytest

from examples.hand_anatomy._common import HandAnatomyError, digest
from examples.hand_anatomy.alignment import align
from examples.hand_anatomy.geometry import pose_violations
from examples.hand_anatomy.observations import observation_from_arrays
from examples.hand_anatomy_02.inspect02 import (
    build_report02,
    inspect02,
    replay_report02,
    validate_report02,
    write_inspection02,
)
from examples.hand_anatomy_02.pairs import make_pair
from examples.hand_anatomy_02.refine import refine
from examples.hand_anatomy_02.vpm02 import (
    INSUFFICIENT,
    METRICS02,
    SUPPORTED,
    SUSPECT,
    Contract02,
    compile_vpm02,
    decide02,
    rows_from_vpm02,
    scores_from_rows,
)


@pytest.fixture(scope="module")
def pair():
    return make_pair("cal02", "extra_finger", "outer", 0)


@pytest.fixture(scope="module")
def obs(pair):
    return observation_from_arrays(pair.valid)


def test_refinement_is_constraint_respecting_deterministic_and_not_worse(small_lib, obs):
    a = align(obs, small_lib)[0]
    fit = refine(obs.work_mask, small_lib.get(a.template_id), a)
    assert pose_violations(fit.template.pose) == []
    assert fit.iou >= a.iou_post - 1e-9 and fit.evaluations > 50
    again = refine(obs.work_mask, small_lib.get(a.template_id), a)
    assert again.template.template_id == fit.template.template_id and again.iou == fit.iou


def _row(name, **over):
    base = {m: 0.9 for m in METRICS02}
    base["finger_residual_explainability"] = 1.0
    base.update(over)
    return {"template_id": name, "name": name, "metrics": base, "diagnostics": {"x": 1}}


def test_vpm_is_real_with_raw_values_and_rejects_incompatible_inputs():
    rows = [_row("t1"), _row("t2", silhouette_agreement=0.6, finger_residual_explainability=0.2)]
    level = {"border_contact_fraction": 0.0, "connectivity": 1.0, "branches_observed": 5, "area_px": 2600}
    vpm = compile_vpm02("obs", "lib", {"r": "x"}, rows, level)
    got = {r["template_id"]: r["metrics"] for r in rows_from_vpm02(vpm)}
    assert got["t2"]["silhouette_agreement"] == 0.6  # RAW value, not the normalised view
    assert vpm.artifact_id == compile_vpm02("obs", "lib", {"r": "x"}, rows, level).artifact_id
    assert vpm.artifact_id != compile_vpm02("obs2", "lib", {"r": "x"}, rows, level).artifact_id
    broken = copy.deepcopy(rows)
    del broken[0]["metrics"]["digit_coverage"]
    with pytest.raises(HandAnatomyError):
        compile_vpm02("obs", "lib", {"r": "x"}, broken, level)
    with pytest.raises(HandAnatomyError):
        compile_vpm02("obs", "lib", {"r": "x"}, [], level)


LEVEL = {"border_contact_fraction": 0.0, "connectivity": 1.0}


def _ranked(**over):
    return [{"template_id": "t", "name": "t", "metrics": {**{m: 0.9 for m in METRICS02},
                                                          "finger_residual_explainability": 1.0, **over}}]


def test_decision_contract_statuses():
    assert decide02(_ranked(), LEVEL)["status"] == SUPPORTED
    assert decide02(_ranked(finger_residual_explainability=0.1), LEVEL)["status"] == SUSPECT
    assert decide02(_ranked(silhouette_agreement=0.3), LEVEL)["status"] == INSUFFICIENT
    assert decide02(_ranked(silhouette_agreement=0.6), LEVEL)["status"] == INSUFFICIENT
    assert decide02(_ranked(), {**LEVEL, "border_contact_fraction": 0.5})["status"] == INSUFFICIENT
    assert decide02([], LEVEL)["status"] == INSUFFICIENT
    high_overlap = _ranked(silhouette_agreement=0.97, finger_residual_explainability=0.0)
    assert decide02(high_overlap, LEVEL)["status"] == SUSPECT  # never averaged away
    c = Contract02(residual_threshold=0.95)
    assert decide02(_ranked(finger_residual_explainability=0.1), LEVEL, c)["status"] == SUPPORTED


def test_scores_come_from_raw_cells_with_declared_polarity():
    rows = _ranked(library_iou=0.7, silhouette_agreement=0.8, finger_residual_explainability=0.4)
    s = scores_from_rows(rows)
    assert s["A_iou"] == pytest.approx(0.3) and s["A_refined_iou"] == pytest.approx(0.2)
    assert s["C_residual"] == pytest.approx(0.6)


def test_end_to_end_report_validates_replays_and_detects_tampering(small_lib, obs, pair, tmp_path):
    res = inspect02(obs, small_lib)
    rep = json.loads(json.dumps(build_report02(res)))
    assert validate_report02(rep) == [] and replay_report02(rep, obs, small_lib) == []
    other = observation_from_arrays(pair.defect)
    assert any("observation_id" in p for p in replay_report02(rep, other, small_lib))
    t = copy.deepcopy(rep)
    t["scores"]["C_residual"] = 0.0
    assert validate_report02(t)
    t = copy.deepcopy(rep)
    t["decision"] = "suspected_anomaly" if rep["decision"] != "suspected_anomaly" else "supported_plausible"
    body = {k: v for k, v in t.items() if k not in ("report_digest", "artifact_references")}
    t["report_digest"] = digest(body)  # forged digest still fails: decision must follow from the VPM
    assert any("does not follow" in p for p in validate_report02(t))
    refs = write_inspection02(res, tmp_path)
    assert (tmp_path / "report.json").is_file() and "04_residual_overlay.png" in refs
    assert np.isfinite(res.vpm.source.values).all()

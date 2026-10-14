import copy
import json

import numpy as np
from PIL import Image, ImageDraw
import pytest

from examples.hand_anatomy.alignment import AlignmentRecipe
from examples.hand_anatomy.assessment import DecisionContract, UNSUPPORTED
from examples.hand_anatomy.comparison import ComparisonRecipe
from examples.hand_anatomy.evidence import (
    build_report,
    inspect_observation,
    replay_report,
    unsupported_report,
    validate_report,
    write_inspection,
)
from examples.hand_anatomy.pose_library import build_library

from .conftest import observation_from_template


def _extra(mask):
    img = Image.fromarray((mask * 255).astype(np.uint8), "L")
    ys, xs = np.nonzero(mask)
    ImageDraw.Draw(img).line([xs.min() + 6, ys.min() + 75, xs.min() - 34, ys.min() + 75], fill=255, width=9)
    return np.asarray(img) > 127


@pytest.fixture(scope="module")
def case(small_lib):
    tpl = next(t for t in small_lib.templates if t.name == "right/open/standard/yaw0")
    obs = observation_from_template(tpl)
    res = inspect_observation(obs, small_lib)
    return tpl, obs, res, json.loads(json.dumps(build_report(res)))


def test_plausible_fixture_is_supported_and_report_is_complete(case):
    _, _, res, rep = case
    assert rep["decision"] == "supported_plausible"
    for key in ("observation_id", "template_library_id", "comparison_recipe_id", "vpm_artifact_id",
                "best_template_id", "decision", "evidence", "limitations", "decision_contract_id"):
        assert key in rep
    assert rep["evidence"]["best_template"]["silhouette_agreement"] > 0.95
    assert any("uncalibrated" in item for item in rep["limitations"])


def test_malformed_fixture_is_suspected(small_lib, case):
    tpl = case[0]
    res = inspect_observation(observation_from_template(tpl, extra=_extra), small_lib)
    assert res.decision["status"] == "suspected_anomaly"
    assert "unexplained_protrusion" in res.decision["reasons"]


def test_saved_report_validates_and_replays(small_lib, case):
    _, obs, _, rep = case
    assert validate_report(rep) == []
    assert replay_report(rep, obs, small_lib) == []


def test_tampered_evidence_is_rejected(case):
    rep = copy.deepcopy(case[3])
    rep["evidence"]["best_template"]["silhouette_agreement"] = 0.0
    assert any("digest" in p for p in validate_report(rep))
    rep = copy.deepcopy(case[3])
    rep["vpm_artifact"]["source"]["values"][0][0] = 0.123
    assert validate_report(rep)
    rep = copy.deepcopy(case[3])
    rep["decision"] = "suspected_anomaly"
    assert validate_report(rep)


def test_forged_decision_with_recomputed_digest_still_rejected(case):
    from examples.hand_anatomy._common import digest

    rep = copy.deepcopy(case[3])
    rep["decision"] = "suspected_anomaly"
    body = {k: v for k, v in rep.items() if k not in ("report_digest", "artifact_references")}
    rep["report_digest"] = digest(body)
    assert any("does not follow" in p for p in validate_report(rep))


def test_changed_observation_invalidates_reuse(small_lib, case):
    tpl, _, _, rep = case
    other = observation_from_template(tpl, extra=_extra)
    assert any("observation_id" in p for p in replay_report(rep, other, small_lib))


def test_changed_templates_and_recipes_change_identity(small_lib, case):
    _, obs, res, rep = case
    other_lib = build_library(gestures=["open", "fist"], yaws=(0.0,), shapes=["standard"])
    assert other_lib.library_id != small_lib.library_id
    assert any("library" in p for p in replay_report(rep, obs, other_lib))
    res2 = inspect_observation(obs, small_lib, comparison_recipe=ComparisonRecipe(tolerance_px=3))
    assert res2.compiled.vpm.artifact_id != res.compiled.vpm.artifact_id
    res3 = inspect_observation(obs, small_lib, alignment_recipe=AlignmentRecipe(shortlist_size=6))
    assert res3.compiled.vpm.artifact_id != res.compiled.vpm.artifact_id
    res4 = inspect_observation(obs, small_lib, contract=DecisionContract(fit_ok=0.5))
    assert res4.compiled.vpm.artifact_id == res.compiled.vpm.artifact_id  # contract is not VPM content
    assert res4.contract.contract_id != res.contract.contract_id


def test_replay_reproduces_decision_under_same_contract(small_lib, case):
    _, obs, _, _ = case
    contract = DecisionContract(fit_ok=0.99)
    res = inspect_observation(obs, small_lib, contract)
    rep = json.loads(json.dumps(build_report(res)))
    assert rep["decision_contract"]["fit_ok"] == 0.99
    assert replay_report(rep, obs, small_lib) == []


def test_unsupported_report_validates():
    rep = unsupported_report("empty mask")
    assert rep["decision"] == UNSUPPORTED and validate_report(rep) == []
    rep["decision_detail"]["reasons"] = ["x"]
    assert validate_report(rep)


def test_write_inspection_outputs_images_and_report(small_lib, case, tmp_path):
    _, _, res, _ = case
    refs = write_inspection(res, tmp_path)
    for name in ("02_normalized_mask.png", "03_best_template.png", "04_overlay_best.png",
                 "05_unexplained_observed.png", "06_missing_expected.png",
                 "07_candidate_contact_sheet.png", "08_vpm_annotated.png", "08_vpm_core_render.png", "report.json"):
        assert (tmp_path / name).is_file(), name
    saved = json.loads((tmp_path / "report.json").read_text())
    assert validate_report(saved) == [] and set(saved["artifact_references"]) <= set(refs)

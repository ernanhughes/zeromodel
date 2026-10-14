import json

from examples.hand_anatomy.assessment import (
    INSUFFICIENT,
    SUPPORTED,
    SUSPECT,
    UNSUPPORTED,
    DecisionContract,
    decide,
)
from examples.hand_anatomy.evidence import inspect_files

GOOD = dict(
    silhouette_agreement=0.9, boundary_agreement=0.8, geometry_agreement=0.9,
    observed_explainability=0.99, protrusion_explainability=1.0,
    missing_explainability=0.98, structure_agreement=1.0,
)
OBS = dict(connectivity=1.0, border_contact_fraction=0.0, protrusions_observed=4, component_count=1)


def row(name="t0", protrusions=4, **over):
    return {"template_id": name, "name": name, "protrusions_template": protrusions, "metrics": {**GOOD, **over}}


def test_supported_when_all_declared_tests_pass():
    d = decide([row()], OBS)
    assert d["status"] == SUPPORTED and d["contradictions"] == []


def test_known_malformed_unexplained_protrusion_is_suspected():
    d = decide([row(protrusion_explainability=0.1)], OBS)
    assert d["status"] == SUSPECT and d["reasons"] == ["unexplained_protrusion"]


def test_high_overlap_does_not_average_away_a_hard_contradiction():
    d = decide([row(silhouette_agreement=0.97, boundary_agreement=0.97, protrusion_explainability=0.0)], OBS)
    assert d["status"] == SUSPECT


def test_best_case_explainer_is_used_across_permitted_candidates():
    rows = [row("a", protrusion_explainability=0.0), row("b", protrusion_explainability=0.9, silhouette_agreement=0.7)]
    d = decide(rows, OBS)
    assert d["explainer_template_id"] == "b" and d["status"] == SUPPORTED


def test_candidates_below_fit_floor_do_not_count_as_explanations():
    d = decide([row(silhouette_agreement=0.3)], OBS)
    assert d["status"] == INSUFFICIENT and d["reasons"] == ["no_permitted_configuration_reaches_fit_floor"]


def test_insufficient_never_promoted_to_pass():
    assert decide([row(silhouette_agreement=0.6)], OBS)["status"] == INSUFFICIENT
    assert decide([row(missing_explainability=0.4)], OBS)["status"] == INSUFFICIENT
    assert decide([], OBS)["status"] == INSUFFICIENT
    assert decide([row()], {**OBS, "border_contact_fraction": 0.5})["status"] == INSUFFICIENT


def test_disconnected_structure_is_a_contradiction():
    assert decide([row()], {**OBS, "connectivity": 0.8})["status"] == SUSPECT


def test_structure_heuristic_is_opt_in():
    obs = {**OBS, "protrusions_observed": 6}
    assert decide([row()], obs)["status"] == SUPPORTED
    assert decide([row()], obs, DecisionContract(use_structure=True))["status"] == SUSPECT
    assert decide([row(protrusions=2)], obs, DecisionContract(use_structure=True))["status"] != SUSPECT


def test_contract_identity_and_calibration_flag():
    base = DecisionContract()
    assert base.contract_id != base.with_values(fit_ok=0.9).contract_id
    assert base.to_dict()["calibrated"] is False
    cal = base.with_values(calibration={"split_id": "x"})
    assert cal.to_dict()["calibrated"] is True and decide([row()], OBS, cal)["calibrated"] is True
    assert DecisionContract.from_dict(json.loads(json.dumps(cal.to_dict()))).contract_id == cal.contract_id


def test_unsupported_input_yields_unsupported_report(tmp_path, small_lib):
    report = inspect_files(small_lib, mask=str(tmp_path / "missing.png"), out_dir=str(tmp_path / "out"))
    assert report["decision"] == UNSUPPORTED
    assert (tmp_path / "out" / "report.json").exists()

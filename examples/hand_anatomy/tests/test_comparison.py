import math

import numpy as np
from PIL import Image, ImageDraw
import pytest

from examples.hand_anatomy._common import HandAnatomyError
from examples.hand_anatomy.alignment import align
from examples.hand_anatomy.comparison import (
    CORE_METRICS,
    ComparisonRecipe,
    compare_candidate,
    connected_components,
    observation_features,
    symmetric_chamfer,
)

from .conftest import observation_from_template

R = ComparisonRecipe()


def _evaluate(lib, tpl, **kw):
    obs = observation_from_template(tpl, **kw)
    feats = observation_features(obs, R)
    cands = [compare_candidate(obs, feats, lib.get(a.template_id), a, R) for a in align(obs, lib)]
    return obs, feats, cands


@pytest.fixture(scope="module")
def tpl(small_lib):
    return next(t for t in small_lib.templates if t.name == "right/open/standard/yaw0")


def _add_finger(mask):
    img = Image.fromarray((mask * 255).astype(np.uint8), "L")
    ys, xs = np.nonzero(mask)
    top = int(ys.min())
    ImageDraw.Draw(img).line([xs.min() - 4, top + 70, xs.min() - 30, top + 5], fill=255, width=9)
    return np.asarray(img) > 127


def _erase_finger(mask):
    out = mask.copy()
    ys, xs = np.nonzero(mask)
    out[: int(ys.min()) + 40, : int(np.median(xs)) - 6] = False  # lops off the left-hand fingers' tips
    return out


def test_identical_silhouette_scores_highly_and_metrics_are_bounded(small_lib, tpl):
    _, _, cands = _evaluate(small_lib, tpl)
    best = cands[0]
    assert best.template_id == tpl.template_id
    assert best.metrics["silhouette_agreement"] > 0.95
    assert best.metrics["observed_explainability"] > 0.99
    assert best.metrics["protrusion_explainability"] == 1.0
    for c in cands:
        assert set(c.metrics) == set(CORE_METRICS)
        assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in c.metrics.values())


def test_extra_protrusion_yields_unexplained_region_evidence(small_lib, tpl):
    _, _, base = _evaluate(small_lib, tpl)
    _, _, extra = _evaluate(small_lib, tpl, extra=_add_finger)
    exp = max(c.metrics["protrusion_explainability"] for c in extra if c.metrics["silhouette_agreement"] > 0.55)
    assert exp < 0.5 < base[0].metrics["protrusion_explainability"]
    top = extra[0]
    assert top.diagnostics["unexplained_components"] and top.diagnostics["largest_unexplained"]["bbox_yxyx"]
    # strong silhouette match coexists with a contradiction: it is NOT averaged away
    assert top.metrics["silhouette_agreement"] > 0.8 and top.metrics["protrusion_explainability"] < 0.5


def test_missing_structure_is_missing_evidence_not_unexplained(small_lib, tpl):
    _, _, cands = _evaluate(small_lib, tpl, extra=_erase_finger)
    top = max(cands, key=lambda c: c.metrics["silhouette_agreement"])
    assert top.diagnostics["missing_pixels"] > 0
    assert top.metrics["observed_explainability"] > 0.97  # nothing extra was observed


def test_occlusion_is_not_labelled_an_anomaly(small_lib, tpl):
    from examples.hand_anatomy.evidence import inspect_observation

    obs = observation_from_template(tpl, extra=_erase_finger)
    res = inspect_observation(obs, small_lib)
    assert res.decision["status"] != "suspected_anomaly"


def test_metric_polarity_more_damage_lowers_compatibility(small_lib, tpl):
    values = []
    for length in (0, 15, 40):
        def grow(mask, length=length):
            if not length:
                return mask
            img = Image.fromarray((mask * 255).astype(np.uint8), "L")
            ys, xs = np.nonzero(mask)
            x0, y0 = int(xs.min()) + 6, int(ys.min()) + 75
            ImageDraw.Draw(img).line([x0, y0, x0 - length, y0], fill=255, width=9)
            return np.asarray(img) > 127

        _, _, cands = _evaluate(small_lib, tpl, extra=grow)
        values.append(next(c for c in cands if c.template_id == tpl.template_id).metrics["observed_explainability"])
    assert values[0] >= values[1] >= values[2] and values[0] > values[2]


def test_components_and_chamfer_edge_cases():
    m = np.zeros((20, 20), bool)
    m[2:5, 2:5] = True
    m[10:14, 10:14] = True
    labels, areas = connected_components(m)
    assert sorted(areas) == [9, 16]
    _, areas_min = connected_components(m, min_area=10)
    assert areas_min == [16]
    assert symmetric_chamfer(m, m) == 0.0
    assert symmetric_chamfer(m, np.zeros_like(m)) == float("inf")


def test_invalid_recipe_rejected():
    with pytest.raises(HandAnatomyError):
        ComparisonRecipe(tolerance_px=-1).validate()
    with pytest.raises(HandAnatomyError):
        ComparisonRecipe(protrusion_ref_fraction=0.0).validate()
    assert ComparisonRecipe().recipe_id != ComparisonRecipe(tolerance_px=3).recipe_id

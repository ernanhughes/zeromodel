import dataclasses

import numpy as np
import pytest

from zeromodel.core import ScoreTable, VPMArtifact

from examples.hand_anatomy._common import HandAnatomyError
from examples.hand_anatomy.alignment import AlignmentRecipe, align
from examples.hand_anatomy.comparison import (
    CORE_METRICS,
    ComparisonRecipe,
    compare_candidate,
    observation_features,
)
from examples.hand_anatomy.vpm_compiler import (
    compile_candidate_vpm,
    evidence_rows_from_vpm,
    layout_recipe,
)

from .conftest import observation_from_template

R = ComparisonRecipe()
AR = AlignmentRecipe()


def _compile(lib, obs, mutate=None):
    feats = observation_features(obs, R)
    cands = [compare_candidate(obs, feats, lib.get(a.template_id), a, R) for a in align(obs, lib)]
    if mutate:
        cands = mutate(cands)
    return cands, compile_candidate_vpm(obs, feats, cands, lib.library_id, R, AR.recipe_id)


@pytest.fixture(scope="module")
def built(small_lib):
    tpl = next(t for t in small_lib.templates if t.name == "right/open/standard/yaw0")
    obs = observation_from_template(tpl)
    return obs, _compile(small_lib, obs)


def test_real_score_table_and_artifact(built):
    _, (cands, compiled) = built
    assert isinstance(compiled.table, ScoreTable) and isinstance(compiled.vpm, VPMArtifact)
    assert compiled.table.metric_ids == CORE_METRICS
    assert compiled.vpm.artifact_id == compiled.vpm.compute_artifact_id()
    assert VPMArtifact.from_dict(compiled.vpm.to_dict()).artifact_id == compiled.vpm.artifact_id


def test_row_and_metric_mapping_and_raw_values(built):
    _, (cands, compiled) = built
    by_id = {c.template_id: c for c in cands}
    assert set(compiled.table.row_ids) == set(by_id)
    for r, rid in enumerate(compiled.table.row_ids):
        for m, mid in enumerate(compiled.table.metric_ids):
            assert compiled.table.values[r, m] == by_id[rid].metrics[mid]
    for row in evidence_rows_from_vpm(compiled.vpm):  # RAW values round-trip through the view
        assert row["metrics"] == by_id[row["template_id"]].metrics


def test_view_is_normalized_but_raw_evidence_is_retained(built):
    _, (_, compiled) = built
    norm = compiled.vpm.normalized_values
    assert norm.min() >= 0 and norm.max() <= 1
    top = compiled.vpm.cell(0, 0)
    assert top.raw_value == pytest.approx(compiled.table.values[top.source_row_index, top.source_metric_index])
    # per-column min-max stretches a poor column: raw and normalized can differ a lot
    assert any(abs(compiled.vpm.cell(r, 1).raw_value - compiled.vpm.cell(r, 1).normalized_value) > 0.2
               for r in range(len(compiled.vpm.row_order)))


def test_deterministic_identity_and_input_sensitivity(small_lib, built):
    obs, (_, compiled) = built
    _, again = _compile(small_lib, obs)
    assert again.vpm.artifact_id == compiled.vpm.artifact_id
    tpl = next(t for t in small_lib.templates if t.name == "right/spread/standard/yaw0")
    _, other = _compile(small_lib, observation_from_template(tpl))
    assert other.vpm.artifact_id != compiled.vpm.artifact_id
    assert compiled.vpm.provenance["observation_id"] == obs.observation_id
    assert compiled.table.metadata["library_id"] == small_lib.library_id
    assert "higher_is_more_compatible" in compiled.table.metadata["metrics"]["silhouette_agreement"]["direction"]


def test_layout_recipe_is_valid_and_declared():
    rec = layout_recipe(CORE_METRICS)
    assert rec.data["row_order"]["kind"] == "weighted_score"
    assert rec.data["normalization"]["kind"] == "per_metric_minmax"


def test_incompatible_comparison_inputs_rejected(small_lib, built):
    obs, _ = built

    def drop_metric(cands):
        broken = dataclasses.replace(cands[0], metrics={k: v for k, v in cands[0].metrics.items() if k != "structure_agreement"})
        return [broken] + cands[1:]

    with pytest.raises(HandAnatomyError):
        _compile(small_lib, obs, drop_metric)
    with pytest.raises(HandAnatomyError):
        compile_candidate_vpm(obs, observation_features(obs, R), [], small_lib.library_id, R, AR.recipe_id)
    assert np.isfinite(built[1][1].table.values).all()

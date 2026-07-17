"""Tests for the Bertin-inspired pattern detector.

The three kill conditions from the design review are asserted directly:

1. Planted structure recovery: the detector must recover known block structure
   from a shuffled matrix and report significance.
2. Null calibration: on structureless noise the detector must not report
   significance (deterministic seeds; the selection-corrected null makes this
   the load-bearing test).
3. Frozen-outcome determinism: identical inputs yield identical report digests
   and identical discovered-view identities, and the view's identity derives
   from the explicit ordering, not from re-running discovery.
"""

from __future__ import annotations

import numpy as np
import pytest

from zeromodel import LayoutRecipe, ScoreTable, build_vpm
from zeromodel.artifact import VPMValidationError
from zeromodel.patterns import (
    MatrixPatternDetector,
    PatternAnalysisSpec,
    build_discovered_view,
    detect_patterns,
)


def _artifact_from_matrix(matrix: np.ndarray, prefix: str = "row"):
    table = ScoreTable(
        values=matrix.tolist(),
        row_ids=["%s:%03d" % (prefix, index) for index in range(matrix.shape[0])],
        metric_ids=["m%d" % index for index in range(matrix.shape[1])],
        metadata={"kind": "pattern-test"},
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "pattern-test-source",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(table, recipe, provenance={"kind": "pattern-test"})


def _planted_matrix(seed: int = 7, noise: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Three planted row blocks with distinct column signatures, shuffled."""

    rng = np.random.default_rng(seed)
    signatures = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )
    labels = np.repeat(np.arange(3), 8)
    matrix = signatures[labels] + rng.normal(0.0, noise, size=(24, 6))
    shuffle = rng.permutation(24)
    return matrix[shuffle], labels[shuffle]


def _same_block_adjacency(order_row_ids, row_ids, labels) -> float:
    label_by_row_id = dict(zip(row_ids, labels))
    ordered_labels = [label_by_row_id[row_id] for row_id in order_row_ids]
    adjacent = sum(
        1 for a, b in zip(ordered_labels, ordered_labels[1:]) if a == b
    )
    return adjacent / (len(ordered_labels) - 1)


def test_planted_structure_is_recovered_and_significant() -> None:
    matrix, labels = _planted_matrix()
    artifact = _artifact_from_matrix(matrix)
    spec = PatternAnalysisSpec(null_samples=99, seed=11)
    report = detect_patterns(artifact, spec)

    adjacency = _same_block_adjacency(
        report.row_order, artifact.source.row_ids, labels
    )
    # Perfect recovery groups all three blocks: 21/23 same-block adjacencies.
    assert adjacency > 0.85
    assert report.family_p_value <= spec.alpha
    assert report.significant
    assert not report.degenerate


def test_pure_noise_is_not_reported_as_structure() -> None:
    rng = np.random.default_rng(3)
    for seed in (0, 1, 2):
        matrix = rng.uniform(size=(20, 6))
        artifact = _artifact_from_matrix(matrix, prefix="noise%d" % seed)
        spec = PatternAnalysisSpec(null_samples=99, seed=seed)
        report = detect_patterns(artifact, spec)
        assert report.family_p_value > spec.alpha, (
            "detector reported structure in noise (seed=%d, p=%.4f)"
            % (seed, report.family_p_value)
        )
        assert not report.significant


def test_alpha_is_part_of_the_specification_contract() -> None:
    default = PatternAnalysisSpec(null_samples=25, seed=5, alpha=0.05)
    stricter = PatternAnalysisSpec(null_samples=25, seed=5, alpha=0.01)
    assert default.digest != stricter.digest
    assert default.to_dict()["alpha"] == 0.05

    matrix, _ = _planted_matrix()
    artifact = _artifact_from_matrix(matrix)
    default_report = detect_patterns(artifact, default)
    strict_report = detect_patterns(artifact, stricter)
    assert default_report.row_order == strict_report.row_order
    assert default_report.family_p_value == strict_report.family_p_value
    assert default_report.digest != strict_report.digest
    assert default_report.significant == (
        default_report.family_p_value <= default.alpha
    )
    assert strict_report.significant == (
        strict_report.family_p_value <= stricter.alpha
    )


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_alpha_is_rejected(alpha: float) -> None:
    with pytest.raises(VPMValidationError):
        PatternAnalysisSpec(alpha=alpha).validate()


def test_report_and_view_are_deterministic_and_frozen() -> None:
    matrix, _ = _planted_matrix()
    artifact = _artifact_from_matrix(matrix)
    spec = PatternAnalysisSpec(null_samples=25, seed=5)

    first = detect_patterns(artifact, spec)
    second = detect_patterns(artifact, spec)
    first_artifact = first.to_vpm()
    second_artifact = second.to_vpm()
    assert first.digest == second.digest
    assert first_artifact.artifact_id == second_artifact.artifact_id

    view_a = build_discovered_view(artifact, first, first_artifact)
    view_b = build_discovered_view(artifact, second, second_artifact)
    assert view_a.artifact_id == view_b.artifact_id

    # The discovered ordering is calibration-independent: a different null seed
    # changes p-values but not the order, the recipe, or the rendered content.
    # The view's record identity legitimately differs because its provenance
    # cites a different report lineage; content invariants must still match.
    other_seed = detect_patterns(artifact, PatternAnalysisSpec(null_samples=25, seed=99))
    other_artifact = other_seed.to_vpm()
    assert other_seed.row_order == first.row_order
    assert other_seed.digest != first.digest
    view_c = build_discovered_view(artifact, other_seed, other_artifact)
    assert view_c.recipe.digest == view_a.recipe.digest
    assert np.array_equal(view_c.normalized_values, view_a.normalized_values)
    assert view_c.artifact_id != view_a.artifact_id


def test_view_lineage_links_materialized_report() -> None:
    matrix, _ = _planted_matrix()
    artifact = _artifact_from_matrix(matrix)
    report = detect_patterns(artifact, PatternAnalysisSpec(null_samples=25, seed=5))
    report_artifact = report.to_vpm()
    view = build_discovered_view(artifact, report, report_artifact)

    relations = {
        parent["relation"]: parent["artifact_id"]
        for parent in view.provenance["parents"]
    }
    assert relations["derived_from"] == artifact.artifact_id
    assert relations["ordered_by"] == report_artifact.artifact_id
    report_parents = report_artifact.provenance["parents"]
    assert report_parents[0]["relation"] == "analyzes"
    assert report_parents[0]["artifact_id"] == artifact.artifact_id
    assert view.source.digest == artifact.source.digest


def test_materialize_returns_complete_lineage_set() -> None:
    matrix, _ = _planted_matrix()
    artifact = _artifact_from_matrix(matrix)
    detector = MatrixPatternDetector(PatternAnalysisSpec(null_samples=25, seed=5))
    result = detector.materialize(artifact)

    assert result.report_artifact.artifact_id == result.report.to_vpm().artifact_id
    relations = {
        parent["relation"]: parent["artifact_id"]
        for parent in result.view_artifact.provenance["parents"]
    }
    assert relations["ordered_by"] == result.report_artifact.artifact_id
    assert relations["derived_from"] == artifact.artifact_id


def test_degenerate_constant_matrix_falls_back_without_significance() -> None:
    artifact = _artifact_from_matrix(np.full((8, 4), 0.5), prefix="const")
    report = detect_patterns(artifact, PatternAnalysisSpec(null_samples=25, seed=1))
    assert report.degenerate
    assert report.row_order == tuple(artifact.source.row_ids)
    assert not report.significant


def test_explicit_row_order_rejects_incomplete_permutations() -> None:
    matrix, _ = _planted_matrix()
    artifact = _artifact_from_matrix(matrix)
    incomplete = {
        "version": "vpm-layout/0",
        "name": "bad-explicit",
        "row_order": {
            "kind": "explicit",
            "row_ids": list(artifact.source.row_ids[:-1]),
            "tie_break": "row_id",
        },
        "column_order": {"kind": "source"},
        "normalization": {"kind": "per_metric_minmax", "clip": True},
    }
    recipe = LayoutRecipe.from_dict(incomplete)
    with pytest.raises(VPMValidationError):
        build_vpm(artifact.source, recipe)


def test_report_rejects_mismatched_artifact() -> None:
    matrix, _ = _planted_matrix()
    artifact = _artifact_from_matrix(matrix)
    other = _artifact_from_matrix(matrix + 0.5, prefix="other")
    report = detect_patterns(artifact, PatternAnalysisSpec(null_samples=10, seed=2))
    with pytest.raises(VPMValidationError):
        build_discovered_view(other, report, report.to_vpm())


def test_view_rejects_unrelated_report_artifact() -> None:
    matrix, _ = _planted_matrix()
    artifact = _artifact_from_matrix(matrix)
    report = detect_patterns(artifact, PatternAnalysisSpec(null_samples=10, seed=2))
    different_report = detect_patterns(
        artifact, PatternAnalysisSpec(null_samples=10, seed=3)
    )
    with pytest.raises(VPMValidationError):
        build_discovered_view(artifact, report, different_report.to_vpm())


def test_detector_wrapper_and_bundle_round_trip(tmp_path) -> None:
    from zeromodel.bundle import from_bundle, to_bundle

    matrix, _ = _planted_matrix()
    artifact = _artifact_from_matrix(matrix)
    detector = MatrixPatternDetector(PatternAnalysisSpec(null_samples=25, seed=5))
    result = detector.materialize(artifact)

    assert result.report.significant
    assert result.report.primary_objective in {
        "adjacent_coherence",
        "anti_robinson",
    }
    report_path = to_bundle(
        result.report_artifact, tmp_path / "pattern-report.vpm"
    )
    view_path = to_bundle(result.view_artifact, tmp_path / "pattern-view.vpm")
    assert from_bundle(report_path).artifact_id == result.report_artifact.artifact_id
    assert from_bundle(view_path).artifact_id == result.view_artifact.artifact_id
    assert "family_p_value" in result.report_artifact.source.metric_ids

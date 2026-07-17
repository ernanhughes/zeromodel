from __future__ import annotations

import numpy as np
import pytest

from zeromodel.artifact import VPMValidationError
from zeromodel.visual_retrieval import (
    LinearProbeIndex,
    VectorAddressIndex,
    build_linear_probe,
    build_vector_address,
)


PROTOTYPES = np.asarray(
    [
        [1.0, 0.0],
        [0.95, 0.05],
        [0.0, 1.0],
        [0.05, 0.95],
    ],
    dtype=np.float32,
)
PROTOTYPE_ROWS = ("left", "left", "right", "right")
PROTOTYPE_ACTIONS = ("A", "A", "B", "B")
PROTOTYPE_IDS = ("p0", "p1", "p2", "p3")
CALIBRATION = np.asarray([[0.98, 0.02], [0.02, 0.98]], dtype=np.float32)
CALIBRATION_ROWS = ("left", "right")
CALIBRATION_ACTIONS = ("A", "B")
CALIBRATION_IDS = ("c0", "c1")


def _build(strategy: str = "medoid"):
    return build_vector_address(
        prototype_vectors=PROTOTYPES,
        prototype_row_ids=PROTOTYPE_ROWS,
        prototype_action_ids=PROTOTYPE_ACTIONS,
        prototype_observation_ids=PROTOTYPE_IDS,
        calibration_vectors=CALIBRATION,
        calibration_row_ids=CALIBRATION_ROWS,
        calibration_action_ids=CALIBRATION_ACTIONS,
        calibration_observation_ids=CALIBRATION_IDS,
        policy_artifact_id="policy",
        source_scope="fixture",
        representation_spec_digest="representation",
        encoder_manifest_id="encoder",
        strategy=strategy,
        calibration_quantile=0.0,
    )


def test_medoid_build_stores_one_prototype_per_policy_row() -> None:
    build = _build("medoid")

    assert build.matrix_blob.shape == (2, 2)
    assert tuple(
        binding.policy_row_id for binding in build.manifest.prototype_bindings
    ) == ("left", "right")
    assert build.manifest.matrix_blob_id == build.matrix_blob.blob_id
    assert build.manifest.calibration_artifact_id == build.calibration.digest


def test_all_example_build_keeps_multiple_prototypes_per_row() -> None:
    build = _build("all")

    assert build.matrix_blob.shape == (4, 2)
    assert [
        binding.policy_row_id for binding in build.manifest.prototype_bindings
    ].count("left") == 2


def test_vector_index_accepts_known_direction_and_rejects_ambiguous_query() -> None:
    index = VectorAddressIndex(_build("medoid"))

    accepted = index.match_vector(
        np.asarray([1.0, 0.0], dtype=np.float32),
        observation_digest="obs:left",
    )
    assert accepted.accepted
    assert accepted.matched_row_id == "left"
    assert accepted.nearest_score > accepted.second_score
    assert "conflicting_action_margin" in accepted.accepted_by

    ambiguous = index.match_vector(
        np.asarray([1.0, 1.0], dtype=np.float32),
        observation_digest="obs:ambiguous",
    )
    assert not ambiguous.accepted
    assert ambiguous.reason == "ambiguous_visual_address"
    assert ambiguous.matched_row_id is None

    zero = index.match_vector(
        np.asarray([0.0, 0.0], dtype=np.float32),
        observation_digest="obs:zero",
    )
    assert not zero.accepted
    assert zero.reason == "zero_visual_representation"


def test_vector_build_requires_independent_row_coverage() -> None:
    with pytest.raises(VPMValidationError, match="rows must match"):
        build_vector_address(
            prototype_vectors=PROTOTYPES,
            prototype_row_ids=PROTOTYPE_ROWS,
            prototype_action_ids=PROTOTYPE_ACTIONS,
            prototype_observation_ids=PROTOTYPE_IDS,
            calibration_vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            calibration_row_ids=("left",),
            calibration_action_ids=("A",),
            calibration_observation_ids=("c0",),
            policy_artifact_id="policy",
            source_scope="fixture",
            representation_spec_digest="representation",
            encoder_manifest_id="encoder",
        )


def test_linear_probe_fits_rows_and_rejects_midpoint() -> None:
    build = build_linear_probe(
        prototype_vectors=PROTOTYPES,
        prototype_row_ids=PROTOTYPE_ROWS,
        prototype_action_ids=PROTOTYPE_ACTIONS,
        calibration_vectors=CALIBRATION,
        calibration_row_ids=CALIBRATION_ROWS,
        calibration_action_ids=CALIBRATION_ACTIONS,
        policy_artifact_id="policy",
        source_scope="fixture",
        representation_spec_digest="representation",
        encoder_manifest_id="encoder",
    )
    index = LinearProbeIndex(build)

    accepted = index.match_vector(
        np.asarray([1.0, 0.0], dtype=np.float32),
        observation_digest="obs:left",
    )
    assert accepted.accepted
    assert accepted.matched_row_id == "left"

    midpoint = index.match_vector(
        np.asarray([1.0, 1.0], dtype=np.float32),
        observation_digest="obs:midpoint",
    )
    assert not midpoint.accepted
    assert midpoint.reason == "ambiguous_visual_address"

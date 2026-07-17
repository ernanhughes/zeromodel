from __future__ import annotations

import numpy as np
import pytest

from zeromodel.artifact import VPMValidationError
from zeromodel.visual_address import ImageObservation
from zeromodel.visual_precomputed import PrecomputedVectorAddressProvider
from zeromodel.visual_retrieval import VectorAddressIndex, build_vector_address


def _index() -> VectorAddressIndex:
    build = build_vector_address(
        prototype_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        prototype_row_ids=("left", "right"),
        prototype_action_ids=("A", "B"),
        prototype_observation_ids=("p0", "p1"),
        calibration_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        calibration_row_ids=("left", "right"),
        calibration_action_ids=("A", "B"),
        calibration_observation_ids=("c0", "c1"),
        policy_artifact_id="policy",
        source_scope="fixture",
        representation_spec_digest="representation",
        encoder_manifest_id="encoder",
    )
    return VectorAddressIndex(build)


def test_precomputed_provider_reuses_digest_bound_representation() -> None:
    observation = ImageObservation(
        pixels=np.asarray([[1, 2], [3, 4]], dtype=np.uint8),
        source_id="fixture",
    )
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    provider = PrecomputedVectorAddressProvider(
        _index(),
        {observation.raw_digest: vector},
    )

    decision = provider.read(observation)

    assert decision.accepted
    assert decision.matched_row_id == "left"
    assert decision.observation_digest == observation.raw_digest
    assert vector.flags.writeable


def test_precomputed_provider_rejects_missing_or_inconsistent_vectors() -> None:
    observation = ImageObservation(
        pixels=np.asarray([[1, 2], [3, 4]], dtype=np.uint8),
        source_id="fixture",
    )
    provider = PrecomputedVectorAddressProvider(
        _index(),
        {"sha256:other": np.asarray([1.0, 0.0], dtype=np.float32)},
    )
    with pytest.raises(VPMValidationError, match="no precomputed representation"):
        provider.read(observation)

    with pytest.raises(VPMValidationError, match="share one dimension"):
        PrecomputedVectorAddressProvider(
            _index(),
            {
                "a": np.asarray([1.0, 0.0], dtype=np.float32),
                "b": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            },
        )

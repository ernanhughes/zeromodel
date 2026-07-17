from __future__ import annotations

import json

import numpy as np
import pytest

from zeromodel import LayoutRecipe, ScoreTable, VPMPolicyLookup, build_vpm
from zeromodel.artifact import VPMValidationError
from zeromodel.visual import VisualFeatureSpec, VisualSignReader, build_visual_index
from zeromodel.visual_address import (
    ImageObservation,
    VisualAddressContract,
    VisualAddressDecision,
)
from zeromodel.visual_address_manifest import PrototypeBinding, VisualAddressManifest
from zeromodel.visual_policy import (
    DeterministicVisualAddressProvider,
    VisualPolicyReader,
)


def _policy():
    table = ScoreTable(
        values=[[1.0, 0.0], [0.0, 1.0]],
        row_ids=["left", "right"],
        metric_ids=["A", "B"],
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "visual-address-test",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(table, recipe, provenance={"kind": "test-policy"})


def _provider():
    policy = _policy()
    spec = VisualFeatureSpec(
        input_height=1,
        input_width=1,
        target_height=1,
        target_width=1,
        quantization_levels=256,
    )
    index = build_visual_index(
        policy,
        {
            "left": np.array([[0]], dtype=np.uint8),
            "right": np.array([[4]], dtype=np.uint8),
        },
        spec,
        threshold_fraction=0.25,
        margin_fraction=0.75,
    )
    reader = VisualSignReader(
        index.artifact,
        policy,
        action_metric_ids=("A", "B"),
    )
    return policy, DeterministicVisualAddressProvider(
        reader,
        source_scope="fixture:two-pixel-states",
    )


def test_image_observation_owns_memory_and_has_stable_digest() -> None:
    caller = np.zeros((2, 2), dtype=np.uint8)
    observation = ImageObservation(
        caller,
        source_id="camera-1",
        metadata={"frame": 1},
    )
    before = observation.raw_digest

    caller[0, 0] = 255

    assert observation.pixels.flags.writeable is False
    assert observation.pixels[0, 0] == 0
    assert observation.raw_digest == before
    assert json.loads(json.dumps(observation.to_descriptor())) == observation.to_descriptor()


def test_deterministic_provider_exposes_contract_before_read() -> None:
    policy, provider = _provider()
    contract = provider.contract()

    assert contract.provider_kind == "deterministic_codebook"
    assert contract.score_semantics == "distance"
    assert contract.policy_artifact_id == policy.artifact_id
    assert contract.source_scope == "fixture:two-pixel-states"
    assert contract.replay_contract == "exact_bytes"
    assert contract.digest == VisualAddressContract.from_dict(
        contract.to_dict()
    ).digest


def test_deterministic_provider_maps_acceptance_and_rejection() -> None:
    _, provider = _provider()

    accepted = provider.read(ImageObservation(np.array([[0]], dtype=np.uint8)))
    ambiguous = provider.read(ImageObservation(np.array([[1]], dtype=np.uint8)))

    assert accepted.accepted
    assert accepted.matched_row_id == "left"
    assert accepted.exact_match
    assert accepted.nearest_score == 0.0
    assert "distance_threshold" in accepted.accepted_by
    assert "absolute_gap" in accepted.accepted_by

    assert not ambiguous.accepted
    assert ambiguous.reason == "ambiguous_visual_address"
    assert ambiguous.matched_row_id is None
    assert ambiguous.nearest_row_id == "left"
    assert ambiguous.ambiguity_measure == pytest.approx(2.0)
    assert json.loads(json.dumps(ambiguous.to_dict())) == ambiguous.to_dict()


def test_visual_policy_reader_performs_independent_policy_lookup() -> None:
    policy, provider = _provider()
    lookup = VPMPolicyLookup(policy, action_metric_ids=("A", "B"))
    reader = VisualPolicyReader(provider, lookup)

    accepted = reader.read(ImageObservation(np.array([[4]], dtype=np.uint8)))
    rejected = reader.read(ImageObservation(np.array([[1]], dtype=np.uint8)))

    assert accepted.accepted
    assert accepted.address.matched_row_id == "right"
    assert accepted.policy is not None
    assert accepted.policy.row_id == "right"
    assert accepted.action == "B"

    assert not rejected.accepted
    assert rejected.policy is None
    assert rejected.action is None


def test_visual_policy_reader_rejects_policy_contract_mismatch() -> None:
    _, provider = _provider()
    other = _policy()
    other = build_vpm(
        other.source,
        other.recipe,
        provenance={"kind": "other-policy"},
    )
    lookup = VPMPolicyLookup(other, action_metric_ids=("A", "B"))

    with pytest.raises(VPMValidationError, match="targets policy"):
        VisualPolicyReader(provider, lookup)


def test_visual_address_manifest_binds_matrix_rows_without_fake_policy_rows() -> None:
    manifest = VisualAddressManifest(
        address_kind="embedding_prototype",
        policy_artifact_id="policy-1",
        matrix_blob_id="blob-1",
        matrix_row_count=3,
        representation_spec_digest="rep-1",
        calibration_artifact_id="cal-1",
        score_semantics="similarity",
        source_scope="fixture:arcade",
        prototype_bindings=[
            PrototypeBinding(
                prototype_id="left#0",
                vector_index=0,
                policy_row_id="left",
            ),
            PrototypeBinding(
                prototype_id="left#1",
                vector_index=1,
                policy_row_id="left",
            ),
            PrototypeBinding(
                prototype_id="right#0",
                vector_index=2,
                policy_row_id="right",
            ),
        ],
    )
    loaded = VisualAddressManifest.from_dict(manifest.to_dict())

    assert loaded.manifest_id == manifest.manifest_id
    assert [item.policy_row_id for item in loaded.prototype_bindings] == [
        "left",
        "left",
        "right",
    ]
    assert not loaded.deployment_permitted


def test_visual_address_manifest_requires_complete_vector_binding() -> None:
    with pytest.raises(VPMValidationError, match="cover every matrix row"):
        VisualAddressManifest(
            address_kind="embedding_prototype",
            policy_artifact_id="policy-1",
            matrix_blob_id="blob-1",
            matrix_row_count=2,
            representation_spec_digest="rep-1",
            calibration_artifact_id="cal-1",
            score_semantics="similarity",
            source_scope="fixture:arcade",
            prototype_bindings=[
                PrototypeBinding(
                    prototype_id="left#0",
                    vector_index=0,
                    policy_row_id="left",
                )
            ],
        )


def test_visual_address_trace_must_be_canonical_json() -> None:
    with pytest.raises(VPMValidationError, match="JSON-serializable"):
        VisualAddressDecision(
            accepted=False,
            reason="rejected",
            observation_digest="obs",
            representation_digest="rep",
            provider_kind="test",
            provider_version="v1",
            score_semantics="distance",
            address_artifact_id="address",
            calibration_artifact_id="calibration",
            policy_artifact_id="policy",
            nearest_row_id=None,
            nearest_score=None,
            second_row_id=None,
            second_score=None,
            ambiguity_measure=None,
            trace={"bad": {1, 2, 3}},
        )

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from zeromodel.core import MatrixBlob, VPMValidationError
from zeromodel.observation import (
    IMAGE_OBSERVATION_VERSION,
    VISUAL_ADDRESS_CONTRACT_VERSION,
    VISUAL_ADDRESS_DECISION_VERSION,
    VISUAL_ADDRESS_MANIFEST_VERSION,
    ImageObservation,
    PrototypeBinding,
    VisualAddressContract,
    VisualAddressDecision,
    VisualAddressManifest,
    VisualAddressProvider,
)


def _contract(**overrides: object) -> VisualAddressContract:
    values = {
        "provider_kind": "contract-test",
        "provider_version": "v1",
        "score_semantics": "distance",
        "observation_spec_digest": "obs-spec",
        "representation_spec_digest": "rep-spec",
        "address_artifact_id": "address-1",
        "calibration_artifact_id": "calibration-1",
        "policy_artifact_id": "policy-1",
        "source_scope": "fixture:scope",
        "replay_contract": "exact_bytes",
        "metadata": {"calibration": {"threshold": 0.25}},
    }
    values.update(overrides)
    return VisualAddressContract(**values)


def _accepted_decision(**overrides: object) -> VisualAddressDecision:
    values = {
        "accepted": True,
        "reason": "accepted",
        "observation_digest": "sha256:obs",
        "representation_digest": "sha256:rep",
        "provider_kind": "contract-test",
        "provider_version": "v1",
        "score_semantics": "distance",
        "address_artifact_id": "address-1",
        "calibration_artifact_id": "calibration-1",
        "policy_artifact_id": "policy-1",
        "nearest_row_id": "row-a",
        "nearest_score": 0.0,
        "second_row_id": "row-b",
        "second_score": 3.0,
        "ambiguity_measure": 3.0,
        "matched_row_id": "row-a",
        "exact_match": True,
        "accepted_by": ("distance_threshold", "margin"),
        "trace": {"rank": [1, 2]},
    }
    values.update(overrides)
    return VisualAddressDecision(**values)


def test_image_observation_owns_memory_and_has_stable_digest() -> None:
    caller = np.zeros((2, 2), dtype=np.uint8)
    observation = ImageObservation(caller, source_id="camera-1", metadata={"frame": 1})
    before = observation.raw_digest

    caller[0, 0] = 255

    assert observation.version == IMAGE_OBSERVATION_VERSION
    assert observation.pixels.flags.writeable is False
    assert observation.pixels[0, 0] == 0
    assert observation.raw_digest == before
    assert (
        json.loads(json.dumps(observation.to_descriptor()))
        == observation.to_descriptor()
    )


def test_image_observation_digest_is_shape_sensitive() -> None:
    flat = ImageObservation(np.array([[1, 2, 3, 4]], dtype=np.uint8))
    square = ImageObservation(np.array([[1, 2], [3, 4]], dtype=np.uint8))

    assert flat.raw_digest != square.raw_digest


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((1,), dtype=np.uint8),
        np.zeros((1, 1, 2), dtype=np.uint8),
        np.zeros((0, 1), dtype=np.uint8),
        np.zeros((1, 1), dtype=np.float32),
    ],
)
def test_image_observation_rejects_malformed_pixels(pixels: np.ndarray) -> None:
    with pytest.raises(VPMValidationError):
        ImageObservation(pixels)


def test_visual_address_contract_identity_and_round_trip_are_canonical() -> None:
    first = _contract(metadata={"b": 2, "a": {"x": 1}})
    second = VisualAddressContract.from_dict(first.to_dict())

    assert first.version == VISUAL_ADDRESS_CONTRACT_VERSION
    assert first.digest == second.digest
    assert first.to_dict() == second.to_dict()
    assert first.digest != _contract(source_scope="other").digest
    assert first.digest != _contract(replay_contract="exact_decision").digest
    assert first.digest != _contract(provider_version="v2").digest


def test_visual_address_contract_rejects_malformed_values() -> None:
    with pytest.raises(VPMValidationError, match="score_semantics"):
        _contract(score_semantics="confidence")
    with pytest.raises(VPMValidationError, match="unsupported replay_contract"):
        _contract(replay_contract="best_effort")
    with pytest.raises(VPMValidationError, match="unsupported visual address contract"):
        _contract(version="old")
    with pytest.raises(VPMValidationError, match="plain scalar"):
        _contract(metadata={"bad": np.float64(1.0)})


def test_visual_address_decision_round_trip_and_identity() -> None:
    accepted = _accepted_decision()
    loaded = VisualAddressDecision.from_dict(accepted.to_dict())
    rejected = VisualAddressDecision(
        accepted=False,
        reason="ambiguous_visual_address",
        observation_digest="sha256:obs",
        representation_digest="sha256:rep",
        provider_kind="contract-test",
        provider_version="v1",
        score_semantics="distance",
        address_artifact_id="address-1",
        calibration_artifact_id="calibration-1",
        policy_artifact_id="policy-1",
        nearest_row_id="row-a",
        nearest_score=1.0,
        second_row_id="row-b",
        second_score=1.2,
        ambiguity_measure=0.2,
    )

    assert accepted.version == VISUAL_ADDRESS_DECISION_VERSION
    assert loaded.digest == accepted.digest
    assert rejected.digest != accepted.digest
    assert json.loads(json.dumps(rejected.to_dict())) == rejected.to_dict()


@pytest.mark.parametrize(
    "overrides",
    [
        {"accepted": True, "matched_row_id": None},
        {"accepted": True, "nearest_row_id": None},
        {"accepted": True, "reason": "ambiguous"},
        {"accepted": False, "reason": "accepted", "matched_row_id": None},
        {"accepted": False, "matched_row_id": None, "accepted_by": ("x",)},
        {"accepted_by": ("x", "x")},
        {"visible_evidence_fraction": 1.1},
        {"nearest_score": float("nan")},
        {"score_semantics": "confidence"},
        {"version": "old"},
    ],
)
def test_visual_address_decision_rejects_contradictory_states(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(VPMValidationError):
        _accepted_decision(**overrides)


def test_visual_address_provider_protocol_accepts_minimal_provider() -> None:
    class MinimalProvider:
        def contract(self) -> VisualAddressContract:
            return _contract()

        def read(self, observation: ImageObservation) -> VisualAddressDecision:
            assert isinstance(observation, ImageObservation)
            return _accepted_decision(observation_digest=observation.raw_digest)

    provider = MinimalProvider()

    assert isinstance(provider, VisualAddressProvider)
    decision = provider.read(ImageObservation(np.array([[0]], dtype=np.uint8)))
    assert decision.accepted


def test_visual_address_manifest_identity_and_matrix_blob_linkage() -> None:
    blob = MatrixBlob.from_array(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        metadata={"kind": "address-prototypes"},
    )
    manifest = VisualAddressManifest(
        address_kind="embedding_prototype",
        policy_artifact_id="policy-1",
        matrix_blob_id=blob.blob_id,
        matrix_row_count=2,
        representation_spec_digest="rep-1",
        calibration_artifact_id="cal-1",
        score_semantics="similarity",
        source_scope="fixture:arcade",
        prototype_bindings=(
            PrototypeBinding("left#0", 0, "left"),
            PrototypeBinding("right#0", 1, "right"),
        ),
        deployment_status="validated",
    )
    loaded = VisualAddressManifest.from_dict(manifest.to_dict())

    assert manifest.version == VISUAL_ADDRESS_MANIFEST_VERSION
    assert loaded.manifest_id == manifest.manifest_id
    assert loaded.deployment_permitted
    assert [item.policy_row_id for item in loaded.prototype_bindings] == [
        "left",
        "right",
    ]


@pytest.mark.parametrize(
    "bindings, row_count, message",
    [
        (
            (PrototypeBinding("left", 0, "row"), PrototypeBinding("left", 1, "row")),
            2,
            "unique",
        ),
        ((PrototypeBinding("left", 1, "row"),), 1, "cover matrix rows"),
        ((PrototypeBinding("left", 0, "row"),), 2, "cover every matrix row"),
    ],
)
def test_visual_address_manifest_rejects_malformed_bindings(
    bindings: tuple[PrototypeBinding, ...], row_count: int, message: str
) -> None:
    with pytest.raises(VPMValidationError, match=message):
        VisualAddressManifest(
            address_kind="embedding_prototype",
            policy_artifact_id="policy-1",
            matrix_blob_id="blob-1",
            matrix_row_count=row_count,
            representation_spec_digest="rep-1",
            calibration_artifact_id="cal-1",
            score_semantics="similarity",
            source_scope="fixture:arcade",
            prototype_bindings=bindings,
        )


def test_visual_address_manifest_rejects_identity_tamper() -> None:
    manifest = VisualAddressManifest(
        address_kind="embedding_prototype",
        policy_artifact_id="policy-1",
        matrix_blob_id="blob-1",
        matrix_row_count=1,
        representation_spec_digest="rep-1",
        calibration_artifact_id="cal-1",
        score_semantics="similarity",
        source_scope="fixture:arcade",
        prototype_bindings=(PrototypeBinding("left", 0, "row"),),
    )
    payload = manifest.to_dict()
    payload["source_scope"] = "changed"

    with pytest.raises(VPMValidationError, match="id mismatch"):
        VisualAddressManifest.from_dict(payload)


def test_wheel_content_when_path_is_provided() -> None:
    import os

    wheel_path = os.environ.get("OBSERVATION_WHEEL_PATH")
    if not wheel_path:
        return

    with zipfile.ZipFile(Path(wheel_path)) as wheel:
        names = set(wheel.namelist())

    expected = {
        "zeromodel/observation/__init__.py",
        "zeromodel/observation/deployment_binding.py",
        "zeromodel/observation/visual_address.py",
        "zeromodel/observation/visual_address_manifest.py",
    }
    assert expected <= names
    assert "zeromodel/__init__.py" not in names
    forbidden_prefixes = (
        "zeromodel/core/",
        "zeromodel/analysis/",
        "zeromodel/vision/",
        "zeromodel/video/",
        "zeromodel/persistence/",
        "tests/",
        "research/",
        "examples/",
        "docs/",
        "scripts/",
    )
    assert not [
        name
        for name in names
        if any(name.startswith(prefix) for prefix in forbidden_prefixes)
    ]

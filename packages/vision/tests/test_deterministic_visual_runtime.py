from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from zeromodel.core import LayoutRecipe, ScoreTable, VPMValidationError, build_vpm
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.observation import (
    ImageObservation,
    VisualAddressDecision,
    VisualAddressProvider,
)
from zeromodel.vision import (
    VISUAL_FEATURE_VERSION,
    VISUAL_INDEX_VERSION,
    VISUAL_POLICY_DECISION_VERSION,
    DeterministicVisualAddressProvider,
    VisualFeatureSpec,
    VisualIndexCalibration,
    VisualPolicyDecision,
    VisualPolicyReader,
    VisualSignReader,
    build_visual_index,
    extract_visual_features,
    visual_feature_digest,
    visual_input_digest,
)


ACTIONS = ("A", "B")


def _policy():
    table = ScoreTable(
        values=[[1.0, 0.0], [0.0, 1.0], [0.4, 0.8]],
        row_ids=("left", "right", "stay"),
        metric_ids=ACTIONS,
        metadata={"kind": "vision-test-policy"},
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "vision-source",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(table, recipe, provenance={"kind": "vision-test-policy"})


def _spec() -> VisualFeatureSpec:
    return VisualFeatureSpec(
        input_height=1,
        input_width=1,
        target_height=1,
        target_width=1,
        quantization_levels=256,
    )


def _frames() -> dict[str, np.ndarray]:
    return {
        "left": np.array([[0]], dtype=np.uint8),
        "right": np.array([[4]], dtype=np.uint8),
        "stay": np.array([[8]], dtype=np.uint8),
    }


def _reader() -> tuple[object, object, VisualSignReader]:
    policy = _policy()
    index = build_visual_index(
        policy,
        _frames(),
        _spec(),
        threshold_fraction=0.25,
        margin_fraction=0.75,
    )
    return (
        policy,
        index,
        VisualSignReader(index.artifact, policy, action_metric_ids=ACTIONS),
    )


def test_feature_spec_round_trip_and_validation() -> None:
    spec = _spec()
    restored = VisualFeatureSpec.from_dict(spec.to_dict())

    assert spec.version == VISUAL_FEATURE_VERSION
    assert restored.digest == spec.digest
    assert spec.feature_count == 1
    with pytest.raises(VPMValidationError):
        VisualFeatureSpec.from_dict({**spec.to_dict(), "version": "old"})
    with pytest.raises(VPMValidationError):
        VisualFeatureSpec(
            input_height=3,
            input_width=1,
            target_height=2,
            target_width=1,
        ).validate()
    with pytest.raises(VPMValidationError):
        VisualFeatureSpec(
            input_height=True,
            input_width=1,
            target_height=1,
            target_width=1,
        ).validate()


def test_feature_extraction_is_deterministic_and_immutable() -> None:
    spec = VisualFeatureSpec(2, 2, 1, 1, quantization_levels=256)
    gray = np.array([[0, 2], [4, 6]], dtype=np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)

    first = extract_visual_features(gray, spec)
    second = extract_visual_features(gray.copy(), spec)

    assert np.array_equal(first, second)
    assert np.array_equal(first, extract_visual_features(rgb, spec))
    assert first.flags.writeable is False
    assert visual_feature_digest(first, spec) == visual_feature_digest(second, spec)
    assert visual_input_digest(gray, spec) == visual_input_digest(gray.copy(), spec)

    with pytest.raises(VPMValidationError):
        extract_visual_features(np.zeros((1,), dtype=np.uint8), spec)
    with pytest.raises(VPMValidationError):
        extract_visual_features(gray.astype(np.float32), spec)


def test_visual_index_builds_deterministic_identity_and_calibration() -> None:
    policy = _policy()
    first = build_visual_index(policy, _frames(), _spec())
    second = build_visual_index(
        policy, dict(reversed(list(_frames().items()))), _spec()
    )

    assert first.artifact.artifact_id == second.artifact.artifact_id
    assert (
        first.artifact.source.metadata["visual_index_version"] == VISUAL_INDEX_VERSION
    )
    assert first.calibration.state_count == 3
    assert first.calibration.feature_count == 1
    assert first.calibration.closest_pair_row_ids == ("left", "right")
    assert first.artifact.provenance["parents"][0]["artifact_id"] == policy.artifact_id

    restored = VisualIndexCalibration.from_dict(first.calibration.to_dict())
    assert restored.digest == first.calibration.digest


def test_visual_index_rejects_incomplete_or_colliding_codebooks() -> None:
    policy = _policy()
    frames = _frames()
    frames.pop("stay")
    with pytest.raises(VPMValidationError, match="cover policy rows exactly"):
        build_visual_index(policy, frames, _spec())

    colliding = _frames()
    colliding["right"] = colliding["left"]
    with pytest.raises(VPMValidationError, match="not separable"):
        build_visual_index(policy, colliding, _spec())


def test_calibration_rejects_fixed_shape_and_threshold_malformed_values() -> None:
    with pytest.raises(VPMValidationError, match="two row ids"):
        VisualIndexCalibration(
            state_count=2,
            feature_count=1,
            min_between_distance=4.0,
            closest_pair_row_ids=("left",),
            threshold_fraction=0.25,
            acceptance_threshold=1.0,
            margin_fraction=0.75,
            required_margin=3.0,
        ).validate()

    with pytest.raises(VPMValidationError, match="vacuous"):
        VisualIndexCalibration(
            state_count=2,
            feature_count=1,
            min_between_distance=4.0,
            closest_pair_row_ids=("left", "right"),
            threshold_fraction=0.25,
            acceptance_threshold=1.0,
            margin_fraction=0.5,
            required_margin=2.0,
        ).validate()


def test_visual_reader_recovers_canonical_rows_and_rejects_ambiguous_inputs() -> None:
    policy, index, reader = _reader()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    recovered = rejected = wrong = 0

    for row_id, frame in _frames().items():
        decision = reader.read(frame)
        recovered += int(decision.accepted)
        wrong += int(decision.matched_row_id != row_id)
        assert decision.reason == "accepted"
        assert decision.action == lookup.choose(row_id)
        assert decision.policy_artifact_id == policy.artifact_id
        assert decision.visual_index_artifact_id == index.artifact.artifact_id
        assert decision.exact_feature_match

    ambiguous = reader.read(np.array([[1]], dtype=np.uint8))
    rejected += int(not ambiguous.accepted)
    assert ambiguous.reason == "ambiguous_visual_address"
    assert ambiguous.action is None
    assert ambiguous.nearest_row_id == "left"

    far = reader.read(np.array([[255]], dtype=np.uint8))
    rejected += int(not far.accepted)
    assert far.reason == "visual_distance_above_threshold"

    assert recovered == 3
    assert rejected == 2
    assert wrong == 0


def test_deterministic_provider_and_visual_policy_reader_smoke() -> None:
    policy, _index, reader = _reader()
    provider = DeterministicVisualAddressProvider(reader, source_scope="fixture")
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    visual_policy = VisualPolicyReader(provider, lookup)

    assert isinstance(provider, VisualAddressProvider)
    address = provider.read(ImageObservation(np.array([[4]], dtype=np.uint8)))
    assert isinstance(address, VisualAddressDecision)
    assert address.accepted
    assert address.matched_row_id == "right"
    assert address.observation_digest
    assert provider.contract().policy_artifact_id == policy.artifact_id

    accepted = visual_policy.read(ImageObservation(np.array([[4]], dtype=np.uint8)))
    rejected = visual_policy.read(ImageObservation(np.array([[1]], dtype=np.uint8)))
    restored = VisualPolicyDecision.from_dict(accepted.to_dict())

    assert accepted.version == VISUAL_POLICY_DECISION_VERSION
    assert accepted.accepted
    assert accepted.action == "B"
    assert accepted.policy is not None
    assert accepted.policy.row_id == "right"
    assert restored.digest == accepted.digest
    assert not rejected.accepted
    assert rejected.action is None
    assert rejected.policy is None


def test_visual_policy_reader_rejects_contract_mismatch() -> None:
    policy, _index, reader = _reader()
    provider = DeterministicVisualAddressProvider(reader, source_scope="fixture")
    other = build_vpm(policy.source, policy.recipe, provenance={"kind": "other"})
    lookup = VPMPolicyLookup(other, action_metric_ids=ACTIONS)

    with pytest.raises(VPMValidationError, match="targets policy"):
        VisualPolicyReader(provider, lookup)


def test_vision_import_and_full_smoke_avoid_forbidden_heavy_dependencies() -> None:
    script = r"""
import json
import sys
import numpy as np
from zeromodel.core import LayoutRecipe, ScoreTable, build_vpm
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.observation import ImageObservation
from zeromodel.vision import (
    DeterministicVisualAddressProvider,
    VisualFeatureSpec,
    VisualPolicyReader,
    VisualSignReader,
    build_visual_index,
)

table = ScoreTable([[1.0, 0.0], [0.0, 1.0]], ["left", "right"], ["A", "B"])
recipe = LayoutRecipe.from_dict({
    "version": "vpm-layout/0",
    "name": "smoke",
    "row_order": {"kind": "source", "tie_break": "row_id"},
    "column_order": {"kind": "source"},
    "normalization": {"kind": "per_metric_minmax", "clip": True},
})
policy = build_vpm(table, recipe)
spec = VisualFeatureSpec(1, 1, 1, 1, quantization_levels=256)
index = build_visual_index(
    policy,
    {"left": np.array([[0]], dtype=np.uint8), "right": np.array([[4]], dtype=np.uint8)},
    spec,
    threshold_fraction=0.25,
    margin_fraction=0.75,
)
reader = VisualSignReader(index.artifact, policy, action_metric_ids=("A", "B"))
provider = DeterministicVisualAddressProvider(reader, source_scope="fixture")
decision = VisualPolicyReader(provider, VPMPolicyLookup(policy, action_metric_ids=("A", "B"))).read(
    ImageObservation(np.array([[4]], dtype=np.uint8))
)
forbidden = [
    "zeromodel.analysis",
    "zeromodel.video",
    "zeromodel.persistence",
    "sqlalchemy",
    "torch",
    "torchvision",
    "transformers",
    "huggingface_hub",
    "PIL",
]
loaded_forbidden = [
    name
    for name in forbidden
    if name in sys.modules or any(module.startswith(name + ".") for module in sys.modules)
]
required = {
    name: name in sys.modules or any(module.startswith(name + ".") for module in sys.modules)
    for name in ("zeromodel.core", "zeromodel.observation")
}
print(json.dumps({"action": decision.action, "required": required, "forbidden": loaded_forbidden}))
raise SystemExit(0 if decision.action == "B" and all(required.values()) and not loaded_forbidden else 1)
"""
    args = [sys.executable, "-c", script]
    if os.environ.get("VISION_WHEEL_PATH"):
        args.insert(1, "-I")
    result = subprocess.run(args, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_vision_wheel_contains_only_vision_namespace_when_path_is_provided() -> None:
    wheel_path = os.environ.get("VISION_WHEEL_PATH")
    if not wheel_path:
        return

    with zipfile.ZipFile(Path(wheel_path)) as wheel:
        names = set(wheel.namelist())

    expected = {
        "zeromodel/vision/__init__.py",
        "zeromodel/vision/visual.py",
        "zeromodel/vision/visual_policy.py",
    }
    assert expected <= names
    assert "zeromodel/__init__.py" not in names
    forbidden_prefixes = (
        "zeromodel/core/",
        "zeromodel/analysis/",
        "zeromodel/observation/",
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

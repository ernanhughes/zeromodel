"""Exhaustive validation for observation-addressed arcade policy lookup."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from zeromodel import (
    LayoutRecipe,
    ScoreTable,
    VPMPolicyLookup,
    build_vpm,
    from_bundle,
    to_bundle,
)
from zeromodel.artifact import VPMValidationError
from zeromodel.visual import (
    MARGIN_RULE,
    VISUAL_INDEX_VERSION,
    VisualFeatureSpec,
    VisualIndexCalibration,
    VisualSignReader,
    build_visual_index,
    extract_visual_features,
)


def _load_demo():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "arcade_visual_sign_reader.py"
    )
    spec = importlib.util.spec_from_file_location(
        "arcade_visual_sign_reader_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _two_row_policy():
    table = ScoreTable(
        values=[[1.0, 0.0], [0.0, 1.0]],
        row_ids=["left", "right"],
        metric_ids=["A", "B"],
        metadata={"kind": "two-row-visual-test"},
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "two-row-source",
            "row_order": {
                "kind": "source",
                "tie_break": "row_id",
            },
            "column_order": {"kind": "source"},
            "normalization": {
                "kind": "per_metric_minmax",
                "clip": True,
            },
        }
    )
    return build_vpm(
        table,
        recipe,
        provenance={"kind": "two-row-visual-test"},
    )


def test_integer_feature_contract_is_deterministic_for_gray_and_rgb() -> None:
    demo = _load_demo()
    spec = demo.arcade_visual_feature_spec()
    frame = demo.render_state_frame(3, 5, 1)
    rgb = np.repeat(frame[:, :, None], 3, axis=2)

    first = extract_visual_features(frame, spec)
    second = extract_visual_features(frame.copy(), spec)
    rgb_features = extract_visual_features(rgb, spec)

    assert np.array_equal(first, second)
    assert np.array_equal(first, rgb_features)
    assert first.shape == (112,)
    assert first.dtype == np.uint8
    assert first.flags.writeable is False
    assert spec.digest == VisualFeatureSpec.from_dict(
        spec.to_dict()
    ).digest

    with pytest.raises(VPMValidationError):
        extract_visual_features(
            np.zeros((8, 8), dtype=np.uint8), spec
        )
    with pytest.raises(VPMValidationError):
        extract_visual_features(frame.astype(np.float64), spec)


def test_visual_read_never_freezes_or_mutates_caller_frame() -> None:
    demo = _load_demo()
    policy = demo.compile_policy_artifact()
    visual = demo.compile_visual_index_artifact(
        policy_artifact=policy
    )
    reader = demo.make_visual_reader(policy, visual)
    frame = demo.render_state_frame(3, 3, 0).copy()
    before = frame.copy()

    assert frame.flags.writeable
    decision = reader.read(frame)

    assert decision.accepted
    assert frame.flags.writeable
    assert np.array_equal(frame, before)

    frame[0, 0] = 1
    assert frame[0, 0] == 1


def test_separation_audit_compiles_identity_bearing_visual_index(
    tmp_path: Path,
) -> None:
    demo = _load_demo()
    policy = demo.compile_policy_artifact()
    visual = demo.compile_visual_index_artifact(
        policy_artifact=policy
    )

    assert (
        visual.artifact.source.metadata["visual_index_version"]
        == VISUAL_INDEX_VERSION
    )
    assert len(visual.artifact.source.row_ids) == 112
    assert len(visual.artifact.source.metric_ids) == 112
    assert visual.calibration.state_count == 112
    assert visual.calibration.feature_count == 112
    assert visual.calibration.min_between_distance == pytest.approx(2.0)
    assert visual.calibration.acceptance_threshold == pytest.approx(0.5)
    assert visual.calibration.required_margin == pytest.approx(1.5)
    assert visual.calibration.margin_rule == MARGIN_RULE
    assert (
        visual.calibration.guaranteed_margin_fraction
        == pytest.approx(0.5)
    )
    assert visual.calibration.closest_pair_row_ids == (
        "tank=0|target=none|cooldown=0",
        "tank=0|target=none|cooldown=1",
    )
    assert (
        visual.artifact.source.metadata["feature_spec_digest"]
        == visual.feature_spec.digest
    )
    assert (
        visual.artifact.source.metadata["calibration_digest"]
        == visual.calibration.digest
    )
    assert (
        visual.artifact.provenance["parents"][0]["relation"]
        == "addresses"
    )
    assert (
        visual.artifact.provenance["parents"][0]["artifact_id"]
        == policy.artifact_id
    )

    bundle = to_bundle(
        visual.artifact,
        tmp_path / "arcade-visual-index.vpm",
    )
    loaded = from_bundle(bundle)
    assert loaded.artifact_id == visual.artifact.artifact_id
    reader = VisualSignReader(
        loaded,
        policy,
        action_metric_ids=demo.ACTIONS,
    )
    decision = reader.read(
        demo.render_state_frame(3, 3, 0)
    )
    assert decision.accepted
    assert decision.action == "FIRE"


def test_vacuous_absolute_margin_calibration_is_rejected() -> None:
    with pytest.raises(
        VPMValidationError, match="margin_fraction is vacuous"
    ):
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


def test_ambiguous_visual_address_is_reachable() -> None:
    policy = _two_row_policy()
    spec = VisualFeatureSpec(
        input_height=1,
        input_width=1,
        target_height=1,
        target_width=1,
        quantization_levels=256,
    )
    visual = build_visual_index(
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
        visual.artifact,
        policy,
        action_metric_ids=("A", "B"),
    )

    # Query 1 is inside the distance radius around 0:
    # nearest=1, threshold=1. Its second distance is 3, so the
    # absolute gap is 2, below the required non-vacuous margin of 3.
    query = np.array([[1]], dtype=np.uint8)
    decision = reader.read(query)

    assert query.flags.writeable
    assert not decision.accepted
    assert decision.reason == "ambiguous_visual_address"
    assert not decision.exact_feature_match
    assert decision.nearest_row_id == "left"
    assert decision.nearest_distance == pytest.approx(1.0)
    assert decision.second_nearest_distance == pytest.approx(3.0)
    assert decision.distance_margin == pytest.approx(2.0)
    assert decision.acceptance_threshold == pytest.approx(1.0)
    assert decision.required_margin == pytest.approx(3.0)
    assert decision.action is None


def test_all_112_canonical_frames_recover_exact_policy_rows_and_actions() -> None:
    demo = _load_demo()
    policy = demo.compile_policy_artifact()
    visual = demo.compile_visual_index_artifact(
        policy_artifact=policy
    )
    reader = demo.make_visual_reader(policy, visual)
    symbolic = VPMPolicyLookup(
        policy, action_metric_ids=demo.ACTIONS
    )

    checked = 0
    for row_id, frame in demo.enumerate_visual_frames().items():
        decision = reader.read(frame)
        assert decision.accepted, row_id
        assert decision.reason == "accepted", row_id
        assert decision.exact_feature_match, row_id
        assert decision.nearest_distance == 0.0, row_id
        assert decision.matched_row_id == row_id
        assert decision.action == symbolic.choose(row_id)
        assert decision.policy_artifact_id == policy.artifact_id
        assert (
            decision.visual_index_artifact_id
            == visual.artifact.artifact_id
        )
        assert (
            decision.distance_margin
            >= visual.calibration.min_between_distance
        )
        assert (
            json.loads(json.dumps(decision.to_dict()))
            == decision.to_dict()
        )
        checked += 1

    assert checked == 112


def test_sub_quantization_change_has_new_input_digest_but_same_exact_codeword() -> None:
    demo = _load_demo()
    policy = demo.compile_policy_artifact()
    visual = demo.compile_visual_index_artifact(
        policy_artifact=policy
    )
    reader = demo.make_visual_reader(policy, visual)

    original = demo.render_state_frame(3, 3, 0).copy()
    changed = original.copy()
    changed[0, 0] = 1

    original_decision = reader.read(original)
    changed_decision = reader.read(changed)

    assert original_decision.accepted
    assert changed_decision.accepted
    assert (
        changed_decision.input_digest
        != original_decision.input_digest
    )
    assert (
        changed_decision.feature_digest
        == original_decision.feature_digest
    )
    assert changed_decision.exact_feature_match
    assert (
        changed_decision.matched_row_id
        == original_decision.matched_row_id
    )
    assert changed_decision.action == original_decision.action


def test_unknown_and_feature_changing_frames_are_rejected_with_evidence() -> None:
    demo = _load_demo()
    policy = demo.compile_policy_artifact()
    visual = demo.compile_visual_index_artifact(
        policy_artifact=policy
    )
    reader = demo.make_visual_reader(policy, visual)

    blank = np.zeros(
        (
            visual.feature_spec.input_height,
            visual.feature_spec.input_width,
        ),
        dtype=np.uint8,
    )
    blank_decision = reader.read(blank)
    assert not blank_decision.accepted
    assert blank_decision.action is None
    assert blank_decision.matched_row_id is None
    assert (
        blank_decision.reason
        == "visual_distance_above_threshold"
    )
    assert (
        blank_decision.nearest_distance
        > blank_decision.acceptance_threshold
    )
    assert blank_decision.nearest_row_id
    assert blank_decision.second_nearest_row_id
    assert not blank_decision.exact_feature_match

    original = demo.render_state_frame(3, 3, 0)
    original_decision = reader.read(original)
    corrupted = original.copy()
    alien_centre = (
        4 * demo.CELL_PIXELS + demo.CELL_PIXELS // 2
    )
    corrupted[
        2:4, alien_centre - 1 : alien_centre + 2
    ] = demo.TARGET_VALUE
    corrupted[4, alien_centre] = demo.TARGET_VALUE
    corrupted_decision = reader.read(corrupted)

    assert not corrupted_decision.accepted
    assert corrupted_decision.action is None
    assert not corrupted_decision.exact_feature_match
    assert (
        corrupted_decision.feature_digest
        != original_decision.feature_digest
    )
    assert (
        corrupted_decision.nearest_distance
        > corrupted_decision.acceptance_threshold
    )


def test_visual_index_build_fails_when_decision_states_collide() -> None:
    demo = _load_demo()
    policy = demo.compile_policy_artifact()
    frames = dict(demo.enumerate_visual_frames())
    first, second = tuple(policy.source.row_ids[:2])
    frames[second] = frames[first]

    with pytest.raises(
        VPMValidationError, match="not separable"
    ):
        build_visual_index(
            policy,
            frames,
            demo.arcade_visual_feature_spec(),
        )


def test_reader_rejects_visual_index_policy_identity_mismatch() -> None:
    demo = _load_demo()
    policy = demo.compile_policy_artifact()
    visual = demo.compile_visual_index_artifact(
        policy_artifact=policy
    )

    # Same source and recipe, different lineage -> different identity.
    other_policy = build_vpm(
        policy.source,
        policy.recipe,
        provenance={"kind": "different-policy-record"},
    )

    assert other_policy.artifact_id != policy.artifact_id
    with pytest.raises(
        VPMValidationError, match="addresses policy"
    ):
        VisualSignReader(
            visual.artifact,
            other_policy,
            action_metric_ids=demo.ACTIONS,
        )


def test_all_2401_waves_are_action_equivalent_through_visual_addressing() -> None:
    demo = _load_demo()
    result = demo.exhaustive_visual_equivalence()

    assert result["waves_evaluated"] == 2401
    assert result["waves_cleared"] == 2401
    assert result["visual_decisions_compared"] == 31213
    assert (
        result["visual_symbolic_action_equivalence_percent"]
        == 100.0
    )

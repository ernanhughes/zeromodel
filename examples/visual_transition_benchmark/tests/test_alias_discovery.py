from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from zeromodel.vision import (
    VisualAcceptanceProfile,
    extract_visual_features,
    visual_feature_digest,
    visual_input_digest,
    visual_raw_input_digest,
)

from visual_transition_benchmark.alias_discovery._json import digest
from visual_transition_benchmark.alias_discovery.corpus import (
    build_context,
    build_case,
    generate_cases,
    split_for_row,
)
from visual_transition_benchmark.alias_discovery.deduplication import deduplicate
from visual_transition_benchmark.alias_discovery.registry import (
    REGISTRY_FILE,
    default_registry,
    registry_id,
    registry_payload,
    write_default_registry,
)
from visual_transition_benchmark.alias_discovery.run import (
    _assert_confirmation_registry_clean,
    _save_observation,
)
from visual_transition_benchmark.alias_discovery.transforms import transform_frame


def test_transform_registry_identity_determinism_and_parameter_canonicalization():
    first = registry_id(default_registry())
    second = registry_id(tuple(reversed(tuple(reversed(default_registry())))))
    assert first == second
    assert registry_payload(default_registry())["transforms"][0]["parameters"] == {}


def test_seeded_transform_determinism_and_chain_stability():
    context = build_context()
    transform = next(item for item in default_registry() if item.transform_id == "uniform_noise")
    source = next(iter(context.frames_by_row_id.values()))
    first = transform_frame(source, transform, seed=17)
    second = transform_frame(source, transform, seed=17)
    different = transform_frame(source, transform, seed=18)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_source_only_transform_interface_and_target_argument_prohibited():
    signature = inspect.signature(transform_frame)
    assert "source_observation" in signature.parameters
    assert "target_row" not in signature.parameters
    assert "target" not in signature.parameters
    with pytest.raises(TypeError):
        transform_frame(np.zeros((16, 28), dtype=np.uint8), default_registry()[0], target_row="x")  # type: ignore[call-arg]


def test_canonical_source_and_transformed_observation_provenance_replay():
    context = build_context()
    row_id = next(iter(context.frames_by_row_id))
    transform = next(item for item in default_registry() if item.transform_id == "grayscale_to_rgb")
    case, transformed = build_case(
        context=context,
        split=split_for_row(row_id),
        source_row_id=row_id,
        transform=transform,
        transform_registry_id=registry_id(default_registry()),
        seed=None,
        severity_rank=1,
        profile=VisualAcceptanceProfile.EXACT_CODEWORD,
    )
    source = context.frames_by_row_id[row_id]
    assert case.source_observation_raw_digest == visual_raw_input_digest(source, context.feature_spec)
    assert case.source_observation_canonical_digest == visual_input_digest(source, context.feature_spec)
    assert case.transformed_observation_raw_digest == visual_raw_input_digest(transformed, context.feature_spec)
    assert case.transformed_observation_canonical_digest == visual_input_digest(transformed, context.feature_spec)
    assert case.transformed_feature_digest == visual_feature_digest(
        extract_visual_features(transformed, context.feature_spec), context.feature_spec
    )
    replay = context.reader.read(transformed, acceptance_profile=case.acceptance_profile)
    assert replay.matched_row_id == case.matched_row_id
    assert replay.action == case.matched_action


def test_profile_specific_invariants_enforced():
    cases, _, _ = generate_cases(mode="smoke")
    for case in cases:
        if case.acceptance_profile == VisualAcceptanceProfile.EXACT_CODEWORD and case.policy_executed:
            assert case.exact_feature_match
        if case.acceptance_profile == VisualAcceptanceProfile.CALIBRATED_NEAREST and case.policy_executed:
            assert case.nearest_distance <= case.acceptance_threshold + 1e-12
            assert case.distance_margin + 1e-12 >= case.required_margin
        if case.acceptance_profile == VisualAcceptanceProfile.EVIDENCE_ONLY:
            assert not case.policy_executed


def test_loaded_registry_identity_propagates_to_cases():
    specs = default_registry()
    rid = registry_id(specs)
    cases, _, _ = generate_cases(mode="smoke", registry=specs, transform_registry_id=rid)
    assert {case.transform_registry_id for case in cases} == {rid}


def test_deduplication_determinism_and_duplicate_provenance_retention():
    cases, _, _ = generate_cases(mode="smoke")
    first = deduplicate(cases)
    second = deduplicate(list(reversed(cases)))
    assert first["generated_case_count"] == second["generated_case_count"]
    assert first["duplicate_count"] == second["duplicate_count"]
    if first["duplicate_groups"]:
        group = first["duplicate_groups"][0]
        assert group["representative_case_id"]
        assert group["all_transform_chain_ids"]


def test_case_identity_determinism_and_duplicate_identity_detection():
    cases, _, _ = generate_cases(mode="smoke")
    first = cases[0]
    assert first.identity == digest(first.identity_payload())
    assert len({case.identity for case in cases[:20]}) == len(cases[:20])


def test_corpus_membership_independent_of_transition_data():
    cases, _, _ = generate_cases(mode="smoke")
    baseline_ids = [case.case_id for case in cases]
    mutated_transition_payload = {"after_frame_removed": True, "transition_result": "mutated"}
    assert mutated_transition_payload
    assert [case.case_id for case in cases] == baseline_ids


def test_discovery_and_confirmation_split_integrity():
    context = build_context()
    rows = tuple(context.frames_by_row_id)
    discovery = {row for row in rows if split_for_row(row) == "discovery"}
    confirmation = {row for row in rows if split_for_row(row) == "confirmation"}
    assert discovery
    assert confirmation
    assert not (discovery & confirmation)


def test_modified_registry_rejected_in_confirmation_mode():
    registry_path = REGISTRY_FILE
    write_default_registry(registry_path)
    original = registry_path.read_text(encoding="utf-8")
    try:
        registry_path.write_text(original + "\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            _assert_confirmation_registry_clean(registry_path)
    finally:
        registry_path.write_text(original, encoding="utf-8")


def test_binary_artifact_digest_verification(tmp_path: Path):
    cases, observations, context = generate_cases(mode="smoke")
    case = cases[0]
    artifact = _save_observation(tmp_path / "observation.npz", observations[case.case_id])
    loaded = np.load(artifact["path"])["observation"]
    assert visual_raw_input_digest(loaded, context.feature_spec) == case.transformed_observation_raw_digest
    with open(artifact["path"], "ab") as handle:
        handle.write(b"x")
    assert artifact["file_digest"] != __import__("visual_transition_benchmark.alias_discovery._json", fromlist=["file_digest"]).file_digest(Path(artifact["path"]))

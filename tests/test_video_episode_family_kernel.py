from __future__ import annotations

import pytest

import research.benchmarks.video_action_set_benchmark as benchmark
from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set import episode_families as families
from zeromodel.video.domains.video_action_set.contracts import (
    CONFLICTING_ACTION_SPLICE_VERSION,
    CRITICAL_EVIDENCE_CORRUPTION_VERSION,
    DECLARED_GAP_OR_UNKNOWN_VERSION,
    EPISODE_FAMILY_REGISTRY_VERSION,
    FAMILY_CLOSURE_VERSION,
    FAMILY_INTERVENTION_VERSION,
    FRAME_INVALID_CLOSURE_VERSION,
    IMPOSSIBLE_TRANSITION_VERSION,
    INFORMATION_CONTROL_VERSION,
    REORDERED_FRAMES_VERSION,
    SPLICE_MASK_VERSION,
    STALE_REPEATED_FRAME_VERSION,
    TRANSFORMATION_FAMILY_VERSION,
    VALID_FAMILY_VERSION,
)


EXPECTED_FAMILY_IDS = [
    "valid",
    "conflicting_action_splice",
    "critical_evidence_corruption",
    "reordered_frames",
    "stale_repeated_frame",
    "impossible_transition",
    "declared_gap_or_unknown_action",
    "information_control",
]
EXPECTED_FAMILY_VERSIONS = [
    "zeromodel-video-action-set-family-valid/v1",
    "zeromodel-video-action-set-family-conflicting-action-splice/v3",
    "zeromodel-video-action-set-family-critical-evidence-corruption/v1",
    "zeromodel-video-action-set-family-reordered-frames/v1",
    "zeromodel-video-action-set-family-stale-repeated-frame/v1",
    "zeromodel-video-action-set-family-impossible-transition/v1",
    "zeromodel-video-action-set-family-declared-gap-or-unknown/v1",
    "zeromodel-video-action-set-family-information-control/v3",
]
EXPECTED_REGISTRY_DIGEST = (
    "sha256:2d5777cbbe6839dfa866c5e95f4c62a70a402ee4dfe7191e099b8b5258681917"
)


def test_episode_family_registry_is_frozen_and_fresh() -> None:
    registry = families.episode_family_registry()

    assert registry["version"] == EPISODE_FAMILY_REGISTRY_VERSION
    assert registry["transformation_contract_version"] == TRANSFORMATION_FAMILY_VERSION
    assert registry["registry_digest"] == EXPECTED_REGISTRY_DIGEST
    assert [item["family_id"] for item in registry["families"]] == EXPECTED_FAMILY_IDS
    assert [item["family_version"] for item in registry["families"]] == (
        EXPECTED_FAMILY_VERSIONS
    )

    registry["families"][0]["family_id"] = "mutated"
    assert families.episode_family_registry()["families"][0]["family_id"] == "valid"


def test_family_contract_lookup_returns_copies() -> None:
    valid = families.family_contract("valid", None)
    conflicting = families.family_contract("frame_invalid", "conflicting_action_splice")
    control = families.family_contract("information_control", None)

    assert valid["family_version"] == VALID_FAMILY_VERSION
    assert conflicting["family_version"] == CONFLICTING_ACTION_SPLICE_VERSION
    assert conflicting["classification"] == "invalid"
    assert control["family_version"] == INFORMATION_CONTROL_VERSION
    assert control["source_action_required"] is False

    conflicting["family_id"] = "mutated"
    assert (
        families.family_contract("frame_invalid", "conflicting_action_splice")[
            "family_id"
        ]
        == "conflicting_action_splice"
    )
    with pytest.raises(VPMValidationError, match="unknown episode family"):
        families.family_contract("frame_invalid", "unsupported")


def test_family_schedule_and_disposition_contracts_are_frozen() -> None:
    assert families.family_schedule() == (
        "exact",
        "bounded_photometric",
        "bounded_translation",
        "bounded_translation_photometric",
        "bounded_translation_occlusion",
        "compound_bounded",
    )
    assert families.episode_disposition("valid") == "valid"
    assert (
        families.episode_disposition("frame_invalid") == "distinguishable_invalid_input"
    )
    assert (
        families.episode_disposition("temporal_negative")
        == "distinguishable_temporal_invalid"
    )
    assert (
        families.episode_disposition("information_control")
        == "information_theoretic_control"
    )
    assert families.episode_disposition("custom", "x") == "x"
    assert families.denominator_class("valid") == "valid_denominator"
    assert (
        families.denominator_class("frame_invalid")
        == "distinguishable_invalid_denominator"
    )
    assert (
        families.denominator_class("temporal_negative")
        == "temporal_negative_denominator"
    )
    assert (
        families.denominator_class("information_control")
        == "excluded_information_control"
    )
    assert families.frame_disposition_for_episode("temporal_negative") == (
        "valid_frame_payload"
    )


def test_expected_frame_disposition_temporal_closure() -> None:
    intervention = {
        "stale_repeat": {"destination_frame_index": 1},
        "impossible_transition": {"destination_frame_index": 2},
        "gap_event": {"position": 3},
    }

    assert (
        families.expected_frame_disposition(
            "temporal_negative", "reordered_frames", 0, intervention
        )
        == "temporally_reordered_frame_payload"
    )
    assert (
        families.expected_frame_disposition(
            "temporal_negative", "stale_repeated_frame", 1, intervention
        )
        == "stale_repeated_frame_payload"
    )
    assert (
        families.expected_frame_disposition(
            "temporal_negative", "stale_repeated_frame", 0, intervention
        )
        == "valid_frame_payload"
    )
    assert (
        families.expected_frame_disposition(
            "temporal_negative", "impossible_transition", 2, intervention
        )
        == "unreachable_destination_frame_payload"
    )
    assert (
        families.expected_frame_disposition(
            "temporal_negative",
            "declared_gap_or_unknown_action",
            3,
            intervention,
        )
        == "declared_gap_or_unknown_action"
    )
    assert (
        families.expected_frame_disposition("information_control", None, 0)
        == "information_theoretic_control"
    )


def test_legacy_benchmark_exposes_direct_family_aliases_and_contracts() -> None:
    assert benchmark._episode_family_registry is families.episode_family_registry
    assert benchmark._family_contract is families.family_contract
    assert benchmark._family_schedule is families.family_schedule
    assert benchmark._episode_disposition is families.episode_disposition
    assert benchmark._denominator_class is families.denominator_class
    assert (
        benchmark._frame_disposition_for_episode
        is families.frame_disposition_for_episode
    )
    assert benchmark.expected_frame_disposition is families.expected_frame_disposition

    assert benchmark.EPISODE_FAMILY_REGISTRY_VERSION == EPISODE_FAMILY_REGISTRY_VERSION
    assert benchmark.FAMILY_INTERVENTION_VERSION == FAMILY_INTERVENTION_VERSION
    assert benchmark.CONFLICTING_ACTION_SPLICE_VERSION == (
        CONFLICTING_ACTION_SPLICE_VERSION
    )
    assert benchmark.CRITICAL_EVIDENCE_CORRUPTION_VERSION == (
        CRITICAL_EVIDENCE_CORRUPTION_VERSION
    )
    assert benchmark.REORDERED_FRAMES_VERSION == REORDERED_FRAMES_VERSION
    assert benchmark.STALE_REPEATED_FRAME_VERSION == STALE_REPEATED_FRAME_VERSION
    assert benchmark.IMPOSSIBLE_TRANSITION_VERSION == IMPOSSIBLE_TRANSITION_VERSION
    assert benchmark.DECLARED_GAP_OR_UNKNOWN_VERSION == DECLARED_GAP_OR_UNKNOWN_VERSION
    assert benchmark.INFORMATION_CONTROL_VERSION == INFORMATION_CONTROL_VERSION
    assert benchmark.SPLICE_MASK_VERSION == SPLICE_MASK_VERSION
    assert benchmark.FRAME_INVALID_CLOSURE_VERSION == FRAME_INVALID_CLOSURE_VERSION
    assert benchmark.FAMILY_CLOSURE_VERSION == FAMILY_CLOSURE_VERSION

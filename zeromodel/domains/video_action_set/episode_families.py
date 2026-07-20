from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...artifact import VPMValidationError
from .canonical_json import canonical_sha256
from .contracts import (
    CONFLICTING_ACTION_SPLICE_VERSION,
    CRITICAL_EVIDENCE_CORRUPTION_VERSION,
    DECLARED_GAP_OR_UNKNOWN_VERSION,
    EPISODE_FAMILY_REGISTRY_VERSION,
    IMPOSSIBLE_TRANSITION_VERSION,
    INFORMATION_CONTROL_VERSION,
    REORDERED_FRAMES_VERSION,
    STALE_REPEATED_FRAME_VERSION,
    TRANSFORMATION_FAMILY_VERSION,
    VALID_FAMILY_VERSION,
)


def _valid_family_entry() -> dict[str, Any]:
    return {
        "family_id": "valid",
        "family_version": VALID_FAMILY_VERSION,
        "classification": "valid",
        "source_row_required": True,
        "source_action_required": True,
        "pixel_intervention": "bounded valid transformation only",
        "sequence_intervention": "none",
        "expected_semantic_effect": "row/action should remain admissible under complete evidence",
        "distinguishability_status": "distinguishable_valid",
        "denominator_treatment": "valid denominator",
        "regeneration_requirements": [
            "source row",
            "sealed seed lineage",
            "transformation parameters",
            "pixel digest",
        ],
        "implementation_status": "implemented_bounded_scaffold",
    }


def _conflicting_splice_family_entry() -> dict[str, Any]:
    return {
        "family_id": "conflicting_action_splice",
        "family_version": CONFLICTING_ACTION_SPLICE_VERSION,
        "classification": "invalid",
        "source_row_required": True,
        "source_action_required": True,
        "pixel_intervention": "primary target evidence plus additive secondary target evidence with valid-state collision closure",
        "sequence_intervention": "none",
        "expected_semantic_effect": "constructed multi-target visual evidence that cannot decode to a canonical valid state",
        "distinguishability_status": "distinguishable_invalid_input",
        "denominator_treatment": "distinguishable-invalid denominator",
        "regeneration_requirements": [
            "primary source",
            "secondary source",
            "target-evidence composition mask",
            "source contribution counts",
            "canonical byte-universe noncollision",
            "output digest",
        ],
        "implementation_status": "implemented_bounded_scaffold",
    }


def _critical_corruption_family_entry() -> dict[str, Any]:
    return {
        "family_id": "critical_evidence_corruption",
        "family_version": CRITICAL_EVIDENCE_CORRUPTION_VERSION,
        "classification": "invalid",
        "source_row_required": True,
        "source_action_required": True,
        "pixel_intervention": "frozen critical coordinate replacement",
        "sequence_intervention": "none",
        "expected_semantic_effect": "critical evidence is no longer faithful to the source row",
        "distinguishability_status": "distinguishable_invalid_input",
        "denominator_treatment": "distinguishable-invalid denominator",
        "regeneration_requirements": [
            "critical coordinate set",
            "original values",
            "replacement values",
            "output digest",
        ],
        "implementation_status": "implemented_bounded_scaffold",
    }


def _reordered_frames_family_entry() -> dict[str, Any]:
    return {
        "family_id": "reordered_frames",
        "family_version": REORDERED_FRAMES_VERSION,
        "classification": "temporal-negative",
        "source_row_required": True,
        "source_action_required": True,
        "pixel_intervention": "none",
        "sequence_intervention": "non-identity permutation of frame payload order",
        "expected_semantic_effect": "sequence order evidence is invalid",
        "distinguishability_status": "distinguishable_temporal_invalid",
        "denominator_treatment": "temporal-negative denominator",
        "regeneration_requirements": [
            "original order",
            "mutated order",
            "sequence digest",
        ],
        "implementation_status": "implemented_bounded_scaffold",
    }


def _stale_repeat_family_entry() -> dict[str, Any]:
    return {
        "family_id": "stale_repeated_frame",
        "family_version": STALE_REPEATED_FRAME_VERSION,
        "classification": "temporal-negative",
        "source_row_required": True,
        "source_action_required": True,
        "pixel_intervention": "later frame payload replaced by earlier payload",
        "sequence_intervention": "explicit stale repeat horizon",
        "expected_semantic_effect": "current frame evidence is stale",
        "distinguishability_status": "distinguishable_temporal_invalid",
        "denominator_treatment": "temporal-negative denominator",
        "regeneration_requirements": [
            "source frame index",
            "destination frame index",
            "original destination digest",
            "replacement digest",
        ],
        "implementation_status": "implemented_bounded_scaffold",
    }


def _impossible_transition_family_entry() -> dict[str, Any]:
    return {
        "family_id": "impossible_transition",
        "family_version": IMPOSSIBLE_TRANSITION_VERSION,
        "classification": "temporal-negative",
        "source_row_required": True,
        "source_action_required": True,
        "pixel_intervention": "destination frame set to a nonreachable policy row",
        "sequence_intervention": "violates frozen reachability relation",
        "expected_semantic_effect": "no admissible source-to-destination transition",
        "distinguishability_status": "distinguishable_temporal_invalid",
        "denominator_treatment": "temporal-negative denominator",
        "regeneration_requirements": [
            "transition relation identity",
            "pairwise reachability audit",
            "endpoint frame digests",
        ],
        "implementation_status": "implemented_bounded_scaffold",
    }


def _declared_gap_family_entry() -> dict[str, Any]:
    return {
        "family_id": "declared_gap_or_unknown_action",
        "family_version": DECLARED_GAP_OR_UNKNOWN_VERSION,
        "classification": "temporal-negative",
        "source_row_required": True,
        "source_action_required": False,
        "pixel_intervention": "typed gap event with no ordinary pixels",
        "sequence_intervention": "explicit gap/unknown event",
        "expected_semantic_effect": "reader sees a deterministic non-frame event",
        "distinguishability_status": "typed_temporal_event",
        "denominator_treatment": "temporal-negative denominator",
        "regeneration_requirements": [
            "event identity",
            "position",
            "duration",
            "sequence digest",
        ],
        "implementation_status": "implemented_bounded_scaffold",
    }


def _information_control_family_entry() -> dict[str, Any]:
    return {
        "family_id": "information_control",
        "family_version": INFORMATION_CONTROL_VERSION,
        "classification": "information-theoretic-control",
        "source_row_required": True,
        "source_action_required": False,
        "pixel_intervention": "byte-identical control observations with multiple hidden source-history labels",
        "sequence_intervention": "control-only grouping",
        "expected_semantic_effect": "hidden distinction unavailable to providers",
        "distinguishability_status": "information_theoretic_control",
        "denominator_treatment": "excluded from distinguishable-invalid denominators",
        "regeneration_requirements": [
            "control group",
            "byte identity digest",
            "hidden history labels",
            "hidden-label cardinality",
            "provider-visible field closure",
        ],
        "implementation_status": "implemented_bounded_scaffold",
    }


def episode_family_registry() -> dict[str, Any]:
    entries = [
        _valid_family_entry(),
        _conflicting_splice_family_entry(),
        _critical_corruption_family_entry(),
        _reordered_frames_family_entry(),
        _stale_repeat_family_entry(),
        _impossible_transition_family_entry(),
        _declared_gap_family_entry(),
        _information_control_family_entry(),
    ]
    return {
        "version": EPISODE_FAMILY_REGISTRY_VERSION,
        "transformation_contract_version": TRANSFORMATION_FAMILY_VERSION,
        "families": entries,
        "registry_digest": canonical_sha256(
            {"version": EPISODE_FAMILY_REGISTRY_VERSION, "families": entries}
        ),
    }


def family_contract(
    family_label: str,
    mutation_kind: str | None,
) -> dict[str, Any]:
    family_id = (
        "valid" if family_label == "valid" else str(mutation_kind or family_label)
    )
    for entry in episode_family_registry()["families"]:
        if entry["family_id"] == family_id:
            return dict(entry)
    raise VPMValidationError("unknown episode family")


def family_schedule() -> tuple[str, ...]:
    return (
        "exact",
        "bounded_photometric",
        "bounded_translation",
        "bounded_translation_photometric",
        "bounded_translation_occlusion",
        "compound_bounded",
    )


def episode_disposition(
    family_label: str,
    mutation_kind: str | None = None,
) -> str:
    if family_label == "valid":
        return "valid"
    if family_label == "frame_invalid":
        return "distinguishable_invalid_input"
    if family_label == "temporal_negative":
        return "distinguishable_temporal_invalid"
    if family_label == "information_control":
        return "information_theoretic_control"
    return str(mutation_kind or family_label)


def denominator_class(
    family_label: str,
    mutation_kind: str | None = None,
) -> str:
    if family_label == "valid":
        return "valid_denominator"
    if family_label == "frame_invalid":
        return "distinguishable_invalid_denominator"
    if family_label == "temporal_negative":
        return "temporal_negative_denominator"
    if family_label == "information_control":
        return "excluded_information_control"
    return str(mutation_kind or family_label)


def frame_disposition_for_episode(
    family_label: str,
    mutation_kind: str | None = None,
) -> str:
    if family_label == "valid":
        return "valid"
    if family_label == "frame_invalid":
        return "distinguishable_invalid_input"
    if family_label == "information_control":
        return "information_theoretic_control"
    if family_label == "temporal_negative":
        return "valid_frame_payload"
    return str(mutation_kind or family_label)


def expected_frame_disposition(
    episode_family: str,
    mutation_kind: str | None,
    frame_index: int,
    intervention_plan: Mapping[str, Any] | None = None,
) -> str:
    family = str(episode_family)
    kind = None if mutation_kind is None else str(mutation_kind)
    index = int(frame_index)
    intervention = dict(intervention_plan or {})
    if family == "valid":
        return "valid"
    if family == "frame_invalid":
        return "distinguishable_invalid_input"
    if family == "information_control":
        return "information_theoretic_control"
    if family != "temporal_negative":
        return str(kind or family)
    if kind == "reordered_frames":
        return "temporally_reordered_frame_payload"
    if kind == "stale_repeated_frame":
        repeat = intervention.get("stale_repeat", {})
        return (
            "stale_repeated_frame_payload"
            if index == int(repeat.get("destination_frame_index", -1))
            else "valid_frame_payload"
        )
    if kind == "impossible_transition":
        transition = intervention.get("impossible_transition", {})
        return (
            "unreachable_destination_frame_payload"
            if index == int(transition.get("destination_frame_index", -1))
            else "valid_frame_payload"
        )
    if kind == "declared_gap_or_unknown_action":
        gap = intervention.get("gap_event", {})
        return (
            "declared_gap_or_unknown_action"
            if index == int(gap.get("position", -1))
            else "valid_frame_payload"
        )
    return "valid_frame_payload"


__all__ = [
    "denominator_class",
    "episode_disposition",
    "episode_family_registry",
    "expected_frame_disposition",
    "family_contract",
    "family_schedule",
    "frame_disposition_for_episode",
]

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...arcade_policy import ShooterConfig, next_rows, parse_state_row_id
from ...artifact import VPMValidationError
from .canonical_json import canonical_sha256
from .contracts import (
    FAMILY_INTERVENTION_VERSION,
    GAP_EVENT_VERSION,
    INFORMATION_CONTROL_AMBIGUITY_VERSION,
    PROVIDER_OBSERVATION_BOUNDARY_VERSION,
    REACHABILITY_TILE_DIGEST,
    SEED_DERIVATION_VERSION,
)
from .control_histories import select_grounded_control_histories
from .dto import BenchmarkIdentityDTO
from .frame_family_kernels import (
    critical_coordinate_manifest,
    splice_mask_manifest,
    splice_pair_has_final_visible_action_conflict,
)


def seed_int_from_digest(digest: str) -> int:
    if not digest.startswith("sha256:"):
        raise VPMValidationError("seed identity must be a sha256 digest")
    return int(digest.removeprefix("sha256:")[:16], 16)


def derived_seed(
    identity: BenchmarkIdentityDTO,
    *,
    split: str,
    ordinal: int,
    namespace: str,
    parent_identities: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    parents = tuple((str(name), str(value)) for name, value in parent_identities)
    if len({name for name, _value in parents}) != len(parents):
        raise VPMValidationError("derived seed parent names must be unique")
    payload = {
        "version": SEED_DERIVATION_VERSION,
        "root_seed_digest": identity.seed_digest,
        "split": split,
        "episode_ordinal": int(ordinal),
        "namespace": namespace,
        "parent_identities": [
            {"name": name, "identity": value} for name, value in parents
        ],
    }
    digest = canonical_sha256(payload)
    return payload | {"seed_digest": digest, "seed_int64": seed_int_from_digest(digest)}


def frame_count_for_plan(split: str, family_label: str) -> int:
    if split == "development":
        return 1
    return 4


def state_row_values(row_id: str) -> tuple[int, int | None, int]:
    return parse_state_row_id(str(row_id))


def secondary_row_for_splice(
    row_ids: list[str],
    row_actions: Mapping[str, str],
    source_row_id: str,
) -> str:
    source_action = row_actions[source_row_id]
    _source_tank, source_target, _source_cooldown = state_row_values(source_row_id)
    if source_target is None:
        raise VPMValidationError(
            "conflicting splice source row must contain visible target evidence"
        )
    for row_id in row_ids:
        if row_id == source_row_id:
            continue
        if row_actions[row_id] == source_action:
            continue
        _tank, target, _cooldown = state_row_values(row_id)
        if target is None or target == source_target:
            continue
        if not splice_pair_has_final_visible_action_conflict(
            source_row_id, row_id, row_actions
        ):
            continue
        return row_id
    raise VPMValidationError(
        "frame splice requires a secondary row with conflicting action and distinct visible target evidence"
    )


def conflicting_splice_source_rows(
    row_ids: list[str],
    row_actions: Mapping[str, str],
    count: int,
) -> list[str]:
    selected: list[str] = []
    for row_id in row_ids:
        _tank, target, _cooldown = state_row_values(row_id)
        if target is None:
            continue
        try:
            secondary_row_for_splice(row_ids, row_actions, row_id)
        except VPMValidationError:
            continue
        selected.append(row_id)
        if len(selected) == int(count):
            return selected
    raise VPMValidationError(
        "unable to select enough conflicting splice source rows with visible target evidence"
    )


def impossible_destination_row(
    row_ids: list[str],
    row_actions: Mapping[str, str],
    source_row_id: str,
) -> str:
    action = row_actions[source_row_id]
    tank, target, cooldown = state_row_values(source_row_id)
    reachable = set(
        next_rows(tank, target, cooldown, action, width=ShooterConfig().width)
    )
    for row_id in reversed(row_ids):
        if row_id not in reachable and row_id != source_row_id:
            return row_id
    raise VPMValidationError("unable to select impossible transition destination")


def _family_id(family_label: str, mutation_kind: str | None) -> str:
    return "valid" if family_label == "valid" else str(mutation_kind or family_label)


def _family_intervention_seed(
    *,
    identity: BenchmarkIdentityDTO,
    split: str,
    ordinal: int,
    family_id: str,
    source_row_id: str,
    secondary_row_id: str | None,
) -> dict[str, Any]:
    return derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="family_intervention",
        parent_identities=(
            ("family_id", family_id),
            ("source_row_id", source_row_id),
            ("secondary_row_id", secondary_row_id or "none"),
        ),
    )


def _base_payload(
    family_id: str,
    seed: Mapping[str, Any],
    original_order: list[int],
) -> dict[str, Any]:
    return {
        "version": FAMILY_INTERVENTION_VERSION,
        "family_id": family_id,
        "intervention_seed_identity": seed["seed_digest"],
        "original_order": original_order,
        "materialized_order": list(original_order),
        "event_type": "frame_sequence",
    }


def _impossible_transition_payload(
    row_ids: Sequence[str],
    row_actions: Mapping[str, str],
    source_row_id: str,
) -> dict[str, Any]:
    destination = impossible_destination_row(list(row_ids), row_actions, source_row_id)
    return {
        "transition_relation_identity": REACHABILITY_TILE_DIGEST,
        "impossible_transition": {
            "source_frame_index": 0,
            "destination_frame_index": 1,
            "source_row_id": source_row_id,
            "source_action_id": row_actions[source_row_id],
            "destination_row_id": destination,
            "destination_action_id": row_actions[destination],
        },
    }


def _gap_event_payload(
    *,
    identity: BenchmarkIdentityDTO,
    split: str,
    ordinal: int,
    seed: Mapping[str, Any],
) -> dict[str, Any]:
    gap_seed = derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="gap_event_identity",
        parent_identities=(
            ("family_intervention", seed["seed_digest"]),
            ("gap_position", "2"),
        ),
    )
    return {
        "event_type": "typed_gap_sequence",
        "gap_event": {
            "version": GAP_EVENT_VERSION,
            "position": 2,
            "duration_frames": 1,
            "reason": "declared_gap_or_unknown_action",
            "event_id": gap_seed["seed_digest"],
        },
    }


def _control_group_payload(
    *,
    identity: BenchmarkIdentityDTO,
    split: str,
    ordinal: int,
    seed: Mapping[str, Any],
    source_row_id: str,
    row_ids: Sequence[str],
    frame_count: int,
) -> dict[str, Any]:
    control_seed = derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="information_control_identity",
        parent_identities=(
            ("family_intervention", seed["seed_digest"]),
            ("current_row_id", source_row_id),
        ),
    )
    hidden_histories = select_grounded_control_histories(
        source_row_id,
        row_ids,
        control_group_id=control_seed["seed_digest"],
        frame_count=frame_count,
    )
    hidden_label_digests = [
        history["hidden_source_label_digest"] for history in hidden_histories
    ]
    return {
        "control_group": {
            "version": INFORMATION_CONTROL_AMBIGUITY_VERSION,
            "control_group_id": control_seed["seed_digest"],
            "current_row_id": source_row_id,
            "byte_identity_required": True,
            "minimum_grounded_causal_history_cardinality": 2,
            "grounded_causal_history_count": len(
                {
                    history["normalized_causal_tuple_digest"]
                    for history in hidden_histories
                }
            ),
            "hidden_source_history_count": len(hidden_histories),
            "hidden_source_histories": hidden_histories,
            "hidden_source_label_digests": hidden_label_digests,
            "hidden_source_label_digest": canonical_sha256(
                {
                    "control_group_id": control_seed["seed_digest"],
                    "hidden_source_label_digests": hidden_label_digests,
                }
            ),
            "provider_observation_boundary_version": PROVIDER_OBSERVATION_BOUNDARY_VERSION,
            "provider_visible_fields": [
                "pixels",
                "shape",
                "raw_digest",
                "timestamp",
                "source_id",
                "metadata",
                "version",
            ],
            "provider_hidden_fields": [
                "frame_id",
                "source_row_id",
                "source_action_id",
                "grounded_causal_history",
                "hidden_source_history_id",
                "hidden_source_label_digest",
            ],
        }
    }


def _with_intervention_digest(
    payload: dict[str, Any], family_id: str
) -> dict[str, Any]:
    payload["sequence_digest"] = canonical_sha256(
        {
            "order": payload["materialized_order"],
            "event_type": payload["event_type"],
            "family_id": family_id,
        }
    )
    payload["intervention_digest"] = canonical_sha256(payload)
    return payload


def family_intervention_plan(
    *,
    identity: BenchmarkIdentityDTO,
    split: str,
    ordinal: int,
    family_label: str,
    mutation_kind: str | None,
    source_row_id: str,
    secondary_row_id: str | None,
    row_ids: Sequence[str],
    row_actions: Mapping[str, str],
) -> dict[str, Any]:
    family_id = _family_id(family_label, mutation_kind)
    seed = _family_intervention_seed(
        identity=identity,
        split=split,
        ordinal=ordinal,
        family_id=family_id,
        source_row_id=source_row_id,
        secondary_row_id=secondary_row_id,
    )
    original_order = list(range(frame_count_for_plan(split, family_label)))
    payload = _base_payload(family_id, seed, original_order)
    if family_id == "conflicting_action_splice":
        payload |= {
            "primary_source_row_id": source_row_id,
            "primary_source_action_id": row_actions[source_row_id],
            "secondary_source_row_id": secondary_row_id,
            "secondary_source_action_id": None
            if secondary_row_id is None
            else row_actions[secondary_row_id],
            "splice_mask": splice_mask_manifest(),
        }
    elif family_id == "critical_evidence_corruption":
        payload |= {"critical_coordinates": critical_coordinate_manifest()}
    elif family_id == "reordered_frames":
        mutated = [1, 0, 2, 3]
        if mutated == original_order:
            raise VPMValidationError("reordered family requires non-identity order")
        payload |= {
            "materialized_order": mutated,
            "sequence_rule": "non_identity_permutation",
        }
    elif family_id == "stale_repeated_frame":
        payload |= {
            "stale_repeat": {
                "source_frame_index": 0,
                "destination_frame_index": 1,
                "maximum_stale_horizon": 1,
            }
        }
    elif family_id == "impossible_transition":
        payload |= _impossible_transition_payload(row_ids, row_actions, source_row_id)
    elif family_id == "declared_gap_or_unknown_action":
        payload |= _gap_event_payload(
            identity=identity,
            split=split,
            ordinal=ordinal,
            seed=seed,
        )
    elif family_id == "information_control":
        payload |= _control_group_payload(
            identity=identity,
            split=split,
            ordinal=ordinal,
            seed=seed,
            source_row_id=source_row_id,
            row_ids=row_ids,
            frame_count=len(original_order),
        )
    return _with_intervention_digest(payload, family_id)


__all__ = [
    "conflicting_splice_source_rows",
    "derived_seed",
    "family_intervention_plan",
    "frame_count_for_plan",
    "impossible_destination_row",
    "secondary_row_for_splice",
    "seed_int_from_digest",
    "state_row_values",
]

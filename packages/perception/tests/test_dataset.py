from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception import (
    DATASET_MANIFEST_VERSION,
    SPLIT_ASSIGNMENT_VERSION,
    DiscreteActionSchemaDTO,
    InMemoryPerceptionDatasetStore,
    PerceptionDatasetError,
    RecordedInteractionDTO,
    SourceImageEncoderSpecDTO,
    build_dataset_manifest,
    encode_discrete_action,
    encode_source_array,
)


def _interaction(
    value: int,
    action: str,
    *,
    sequence_id: str = "episode-1",
    step_index: int = 0,
    observed_at_ns: int | None = None,
) -> tuple[RecordedInteractionDTO, str]:
    spec = SourceImageEncoderSpecDTO(color_space="L")
    source = encode_source_array(np.full((2, 2), value, dtype=np.uint8), spec)
    schema = DiscreteActionSchemaDTO.from_labels(["FIRE", "LEFT", "RIGHT"])
    target = encode_discrete_action(action, schema)
    return (
        RecordedInteractionDTO.from_vpms(
            sequence_id=sequence_id,
            step_index=step_index,
            observed_at_ns=observed_at_ns,
            source=source,
            target=target,
        ),
        spec.encoder_spec_id,
    )


def test_interaction_identity_is_content_derived() -> None:
    first, _ = _interaction(1, "LEFT", step_index=2)
    second, _ = _interaction(1, "LEFT", step_index=2)
    changed, _ = _interaction(1, "RIGHT", step_index=2)

    assert first == second
    assert first.interaction_id != changed.interaction_id


def test_manifest_identity_and_splits_are_order_independent() -> None:
    first, encoder_id = _interaction(1, "LEFT", step_index=0)
    second, _ = _interaction(2, "RIGHT", step_index=1)

    forward = build_dataset_manifest(
        [first, second], source_encoder_spec_ids=[encoder_id]
    )
    reverse = build_dataset_manifest(
        [second, first], source_encoder_spec_ids=[encoder_id]
    )

    assert forward == reverse
    assert forward.dataset_id == reverse.dataset_id
    assert forward.version == DATASET_MANIFEST_VERSION
    assert {item.version for item in forward.split_assignments} == {
        SPLIT_ASSIGNMENT_VERSION
    }
    assert {item.split for item in forward.split_assignments} <= {
        "train",
        "validation",
        "test",
    }


def test_all_interactions_in_one_sequence_share_one_split() -> None:
    interactions = []
    encoder_id = ""
    for step_index in range(20):
        interaction, encoder_id = _interaction(
            step_index,
            "LEFT" if step_index % 2 == 0 else "RIGHT",
            sequence_id="episode-owned-as-one-unit",
            step_index=step_index,
        )
        interactions.append(interaction)

    manifest = build_dataset_manifest(
        interactions,
        source_encoder_spec_ids=[encoder_id],
    )

    assert len({item.split for item in manifest.split_assignments}) == 1
    assert not any(item.code == "sequence_crosses_splits" for item in manifest.findings)


def test_identical_pixels_with_conflicting_actions_are_rejected() -> None:
    left, encoder_id = _interaction(7, "LEFT", step_index=0)
    right, _ = _interaction(7, "RIGHT", step_index=1)

    with pytest.raises(PerceptionDatasetError, match="conflicting_actions"):
        build_dataset_manifest([left, right], source_encoder_spec_ids=[encoder_id])

    manifest = build_dataset_manifest(
        [left, right],
        source_encoder_spec_ids=[encoder_id],
        reject_errors=False,
    )
    assert manifest.findings[0].code == "conflicting_actions_for_identical_source"


def test_identical_pixels_with_same_action_cannot_cross_splits() -> None:
    interactions = []
    encoder_id = ""
    for index in range(64):
        interaction, encoder_id = _interaction(
            7,
            "LEFT",
            sequence_id=f"independent-episode-{index}",
            step_index=0,
        )
        interactions.append(interaction)

    manifest = build_dataset_manifest(
        interactions,
        source_encoder_spec_ids=[encoder_id],
        reject_errors=False,
    )

    assert len({item.split for item in manifest.split_assignments}) > 1
    finding = next(
        item for item in manifest.findings if item.code == "identical_source_across_splits"
    )
    assert finding.severity == "error"
    assert set(finding.interaction_ids) == {
        item.interaction_id for item in interactions
    }

    with pytest.raises(PerceptionDatasetError, match="identical_source_across_splits"):
        build_dataset_manifest(
            interactions,
            source_encoder_spec_ids=[encoder_id],
        )


def test_duplicate_sequence_steps_are_rejected() -> None:
    first, encoder_id = _interaction(1, "LEFT", step_index=0)
    second, _ = _interaction(2, "RIGHT", step_index=0)

    with pytest.raises(PerceptionDatasetError, match="duplicate_sequence_step"):
        build_dataset_manifest([first, second], source_encoder_spec_ids=[encoder_id])


def test_non_monotonic_timestamps_are_rejected() -> None:
    first, encoder_id = _interaction(1, "LEFT", step_index=0, observed_at_ns=20)
    second, _ = _interaction(2, "RIGHT", step_index=1, observed_at_ns=10)

    with pytest.raises(PerceptionDatasetError, match="non_monotonic_sequence_time"):
        build_dataset_manifest([first, second], source_encoder_spec_ids=[encoder_id])


def test_manifest_rejects_mixed_action_schemas() -> None:
    interaction, encoder_id = _interaction(1, "LEFT")
    changed = replace(
        interaction,
        interaction_id="sha256:changed-interaction",
        action_schema_id="sha256:other",
    )

    with pytest.raises(PerceptionDatasetError, match="one action schema"):
        build_dataset_manifest(
            [interaction, changed], source_encoder_spec_ids=[encoder_id]
        )


def test_in_memory_store_is_idempotent_and_rejects_identity_collision() -> None:
    interaction, encoder_id = _interaction(1, "LEFT")
    manifest = build_dataset_manifest(
        [interaction], source_encoder_spec_ids=[encoder_id]
    )
    store = InMemoryPerceptionDatasetStore()

    store.put(manifest)
    store.put(manifest)

    assert store.get(manifest.dataset_id) == manifest
    assert store.list_ids() == (manifest.dataset_id,)

    conflicting = replace(
        manifest,
        source_encoder_spec_ids=("sha256:different",),
    )
    with pytest.raises(PerceptionDatasetError, match="different content"):
        store.put(conflicting)


def test_manifest_requires_interactions_encoder_identity_and_seed() -> None:
    with pytest.raises(PerceptionDatasetError, match="at least one interaction"):
        build_dataset_manifest([], source_encoder_spec_ids=["sha256:x"])

    interaction, _ = _interaction(1, "LEFT")
    with pytest.raises(PerceptionDatasetError, match="source_encoder_spec_ids"):
        build_dataset_manifest([interaction], source_encoder_spec_ids=[])
    with pytest.raises(PerceptionDatasetError, match="split_seed"):
        build_dataset_manifest(
            [interaction],
            source_encoder_spec_ids=["sha256:x"],
            split_seed="",
        )

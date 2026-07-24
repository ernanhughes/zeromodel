from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception import (
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
    assert {item.split for item in forward.split_assignments} <= {
        "train",
        "validation",
        "test",
    }


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


def test_duplicate_sequence_steps_are_rejected() -> None:
    first, encoder_id = _interaction(1, "LEFT", step_index=0)
    second, _ = _interaction(2, "RIGHT", step_index=0)

    with pytest.raises(PerceptionDatasetError, match="duplicate_sequence_step"):
        build_dataset_manifest([first, second], source_encoder_spec_ids=[encoder_id])


def test_non_monotonic_timestamps_are_rejected() -> None:
    first, encoder_id = _interaction(
        1, "LEFT", step_index=0, observed_at_ns=20
    )
    second, _ = _interaction(
        2, "RIGHT", step_index=1, observed_at_ns=10
    )

    with pytest.raises(PerceptionDatasetError, match="non_monotonic_sequence_time"):
        build_dataset_manifest([first, second], source_encoder_spec_ids=[encoder_id])


def test_manifest_rejects_mixed_action_schemas() -> None:
    interaction, encoder_id = _interaction(1, "LEFT")
    changed = replace(interaction, action_schema_id="sha256:other")

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

    conflicting = replace(manifest, findings=(
        replace(
            manifest.findings[0], detail="changed"
        ),
    )) if manifest.findings else replace(
        manifest,
        source_encoder_spec_ids=("sha256:different",),
    )
    with pytest.raises(PerceptionDatasetError, match="different content"):
        store.put(conflicting)


def test_manifest_requires_interactions_and_encoder_identity() -> None:
    with pytest.raises(PerceptionDatasetError, match="at least one interaction"):
        build_dataset_manifest([], source_encoder_spec_ids=["sha256:x"])

    interaction, _ = _interaction(1, "LEFT")
    with pytest.raises(PerceptionDatasetError, match="source_encoder_spec_ids"):
        build_dataset_manifest([interaction], source_encoder_spec_ids=[])

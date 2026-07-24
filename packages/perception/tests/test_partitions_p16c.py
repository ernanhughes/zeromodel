from __future__ import annotations

import numpy as np
import pytest

from zeromodel.perception import (
    DiscreteActionSchemaDTO,
    RecordedInteractionDTO,
    SourceImageEncoderSpecDTO,
    build_dataset_manifest,
    encode_discrete_action,
    encode_source_array,
)
from zeromodel.perception.partitions import (
    DATASET_PARTITION_SEMANTICS,
    DATASET_PARTITION_VERSION,
    PerceptionDatasetPartitionError,
    build_dataset_partition,
)


def _manifest():
    encoder = SourceImageEncoderSpecDTO(color_space="L")
    schema = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    interactions = []
    for index in range(120):
        source = encode_source_array(
            np.asarray([[index, 255 - index]], dtype=np.uint8),
            encoder,
        )
        interactions.append(
            RecordedInteractionDTO.from_vpms(
                sequence_id=f"episode-{index}",
                step_index=0,
                source=source,
                target=encode_discrete_action(
                    "LEFT" if index % 2 == 0 else "RIGHT",
                    schema,
                ),
            )
        )
    return build_dataset_manifest(
        interactions,
        source_encoder_spec_ids=[encoder.encoder_spec_id],
        split_seed="p16c-partition-test",
    )


def test_partition_is_manifest_derived_and_deterministic() -> None:
    manifest = _manifest()
    first = build_dataset_partition(manifest, "validation")
    second = build_dataset_partition(manifest, "validation")

    assert first == second
    assert first.dataset_id == manifest.dataset_id
    assert first.action_schema_id == manifest.action_schema_id
    assert first.split == "validation"
    assert first.interaction_ids == tuple(
        item.interaction_id
        for item in manifest.interactions
        if manifest.split_for(item.interaction_id) == "validation"
    )
    assert all(first.owns_interaction(value) for value in first.interaction_ids)
    assert first.semantics == DATASET_PARTITION_SEMANTICS
    assert first.version == DATASET_PARTITION_VERSION


def test_partition_identity_changes_with_owned_evidence() -> None:
    manifest = _manifest()
    validation = build_dataset_partition(manifest, "validation")
    test = build_dataset_partition(manifest, "test")

    assert validation.partition_id != test.partition_id
    assert set(validation.interaction_ids).isdisjoint(test.interaction_ids)
    assert set(validation.sequence_ids).isdisjoint(test.sequence_ids)
    assert set(validation.source_pixel_digests).isdisjoint(test.source_pixel_digests)


def test_partition_rejects_unsupported_or_empty_split() -> None:
    manifest = _manifest()
    with pytest.raises(PerceptionDatasetPartitionError, match="unsupported"):
        build_dataset_partition(manifest, "shadow")

    train_only = build_dataset_manifest(
        [manifest.interactions[0]],
        source_encoder_spec_ids=manifest.source_encoder_spec_ids,
        split_seed="find-train-only",
    )
    empty_split = next(
        split
        for split in ("train", "validation", "test")
        if not any(
            train_only.split_for(item.interaction_id) == split
            for item in train_only.interactions
        )
    )
    with pytest.raises(PerceptionDatasetPartitionError, match="has no interactions"):
        build_dataset_partition(train_only, empty_split)

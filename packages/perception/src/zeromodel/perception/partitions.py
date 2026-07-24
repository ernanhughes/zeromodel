"""Manifest-owned dataset partition provenance for Stage P16C.

A split name alone is not evidence of split ownership. This module materializes one
content-addressed partition from an immutable P16B manifest, preserving the exact
interaction, sequence, and source-pixel identities owned by that partition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .dataset import PerceptionDatasetManifestDTO

DATASET_PARTITION_VERSION: Final = "perception-dataset-partition/1"
DATASET_PARTITION_SEMANTICS: Final = (
    "manifest_derived_exact_interaction_sequence_and_source_identity_partition"
)
_ALLOWED_PARTITIONS: Final = {"train", "validation", "test"}


class PerceptionDatasetPartitionError(ValueError):
    """Raised when manifest-owned partition provenance is invalid."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, object]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


@dataclass(frozen=True)
class DatasetPartitionDTO:
    partition_id: str
    dataset_id: str
    split: str
    action_schema_id: str
    interaction_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    source_pixel_digests: tuple[str, ...]
    semantics: str = DATASET_PARTITION_SEMANTICS
    version: str = DATASET_PARTITION_VERSION

    def __post_init__(self) -> None:
        if not all((self.partition_id, self.dataset_id, self.action_schema_id)):
            raise PerceptionDatasetPartitionError("partition identities must be non-empty")
        if self.split not in _ALLOWED_PARTITIONS:
            raise PerceptionDatasetPartitionError("unsupported dataset partition")
        for name, values in (
            ("interaction_ids", self.interaction_ids),
            ("sequence_ids", self.sequence_ids),
            ("source_pixel_digests", self.source_pixel_digests),
        ):
            if not values or values != tuple(sorted(set(values))):
                raise PerceptionDatasetPartitionError(
                    f"{name} must be non-empty, unique, and sorted"
                )
        if self.semantics != DATASET_PARTITION_SEMANTICS:
            raise PerceptionDatasetPartitionError("unsupported partition semantics")
        if self.version != DATASET_PARTITION_VERSION:
            raise PerceptionDatasetPartitionError("unsupported partition version")

    def owns_interaction(self, interaction_id: str) -> bool:
        return interaction_id in self.interaction_ids


def build_dataset_partition(
    manifest: PerceptionDatasetManifestDTO,
    split: str,
) -> DatasetPartitionDTO:
    """Materialize one exact partition from the manifest's authoritative assignments."""

    if split not in _ALLOWED_PARTITIONS:
        raise PerceptionDatasetPartitionError("unsupported dataset partition")
    selected = tuple(
        interaction
        for interaction in manifest.interactions
        if manifest.split_for(interaction.interaction_id) == split
    )
    if not selected:
        raise PerceptionDatasetPartitionError(
            f"manifest {manifest.dataset_id} has no interactions in {split!r}"
        )
    interaction_ids = tuple(sorted(item.interaction_id for item in selected))
    sequence_ids = tuple(sorted({item.sequence_id for item in selected}))
    source_pixel_digests = tuple(sorted({item.source_pixel_digest for item in selected}))
    payload: Mapping[str, object] = {
        "action_schema_id": manifest.action_schema_id,
        "dataset_id": manifest.dataset_id,
        "interaction_ids": list(interaction_ids),
        "semantics": DATASET_PARTITION_SEMANTICS,
        "sequence_ids": list(sequence_ids),
        "source_pixel_digests": list(source_pixel_digests),
        "split": split,
        "version": DATASET_PARTITION_VERSION,
    }
    return DatasetPartitionDTO(
        partition_id=_digest(payload),
        dataset_id=manifest.dataset_id,
        split=split,
        action_schema_id=manifest.action_schema_id,
        interaction_ids=interaction_ids,
        sequence_ids=sequence_ids,
        source_pixel_digests=source_pixel_digests,
    )

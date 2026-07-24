"""ZeroModel perception public API."""

from __future__ import annotations

from .dataset import (
    DATASET_MANIFEST_VERSION,
    INTERACTION_VERSION,
    SPLIT_ASSIGNMENT_VERSION,
    DatasetFindingDTO,
    InMemoryPerceptionDatasetStore,
    PerceptionDatasetError,
    PerceptionDatasetManifestDTO,
    PerceptionDatasetStore,
    RecordedInteractionDTO,
    SplitAssignmentDTO,
    build_dataset_manifest,
)
from .representation import (
    ACTION_SCHEMA_VERSION,
    SOURCE_ENCODER_VERSION,
    TARGET_ENCODER_VERSION,
    DiscreteActionSchemaDTO,
    PerceptionRepresentationError,
    SourceImageEncoderSpecDTO,
    SourceVPMDTO,
    TargetVPMDTO,
    decode_discrete_action,
    encode_discrete_action,
    encode_source_array,
    encode_source_image_bytes,
)

PERCEPTION_PACKAGE_VERSION = "1.0.13"
PERCEPTION_STAGE = "P2"

__all__ = [
    "ACTION_SCHEMA_VERSION",
    "DATASET_MANIFEST_VERSION",
    "INTERACTION_VERSION",
    "PERCEPTION_PACKAGE_VERSION",
    "PERCEPTION_STAGE",
    "SOURCE_ENCODER_VERSION",
    "SPLIT_ASSIGNMENT_VERSION",
    "TARGET_ENCODER_VERSION",
    "DatasetFindingDTO",
    "DiscreteActionSchemaDTO",
    "InMemoryPerceptionDatasetStore",
    "PerceptionDatasetError",
    "PerceptionDatasetManifestDTO",
    "PerceptionDatasetStore",
    "PerceptionRepresentationError",
    "RecordedInteractionDTO",
    "SourceImageEncoderSpecDTO",
    "SourceVPMDTO",
    "SplitAssignmentDTO",
    "TargetVPMDTO",
    "build_dataset_manifest",
    "decode_discrete_action",
    "encode_discrete_action",
    "encode_source_array",
    "encode_source_image_bytes",
]

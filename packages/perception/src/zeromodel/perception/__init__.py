"""ZeroModel perception public API."""

from __future__ import annotations

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
PERCEPTION_STAGE = "P1"

__all__ = [
    "ACTION_SCHEMA_VERSION",
    "PERCEPTION_PACKAGE_VERSION",
    "PERCEPTION_STAGE",
    "SOURCE_ENCODER_VERSION",
    "TARGET_ENCODER_VERSION",
    "DiscreteActionSchemaDTO",
    "PerceptionRepresentationError",
    "SourceImageEncoderSpecDTO",
    "SourceVPMDTO",
    "TargetVPMDTO",
    "decode_discrete_action",
    "encode_discrete_action",
    "encode_source_array",
    "encode_source_image_bytes",
]

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import cast

import numpy as np

from ...artifact import VPMValidationError
from ...matrix_blob import MatrixBlob
from ...visual_address import IMAGE_OBSERVATION_VERSION
from .canonical_json import canonical_json_bytes, canonical_sha256
from .contracts import (
    BENCHMARK_VERSION,
    FRAME_SHAPE,
    GENERATOR_VERSION,
    OBSERVATION_OPERATION_CHAIN_VERSION,
    PROVIDER_OBSERVATION_BOUNDARY_VERSION,
)
from .dto import CanonicalJsonDTO, SPLITS

OBSERVATION_RECORD_KEYS = (
    "benchmark_version",
    "generator_version",
    "split",
    "episode_id",
    "clip_id",
    "frame_id",
    "sequence_number",
    "event_type",
    "family",
    "expected_disposition",
    "episode_family",
    "episode_disposition",
    "frame_disposition",
    "denominator_class",
    "expected_row",
    "expected_action",
    "actual_executed_action",
    "action_known",
    "gap_declaration",
    "observation_pixel_digest",
    "metadata",
)
STRUCTURED_METADATA_KEYS = (
    "observation_operation_chain",
    "provider_observation_descriptor",
    "provider_observation_digest",
)
PROVIDER_DESCRIPTOR_KEYS = (
    "version",
    "raw_digest",
    "shape",
    "timestamp",
    "source_id",
    "metadata",
)
OPERATION_KEYS = (
    "index",
    "operation",
    "operation_version",
    "input_digests",
    "parameters",
    "parameter_digest",
    "output_digest",
    "operation_digest",
)
OPERATION_CHAIN_KEYS = (
    "version",
    "operations",
    "final_emitted_digest",
    "operation_chain_digest",
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(item in "0123456789abcdef" for item in value[7:])
    )


def _sha256(value: object, message: str) -> str:
    if not _is_sha256(value):
        raise VPMValidationError(message)
    return str(value)


def _optional_sha256(value: object, message: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, message)


def _require_keys(
    payload: Mapping[str, object],
    keys: tuple[str, ...],
    message: str,
) -> None:
    if set(payload) != set(keys):
        raise VPMValidationError(message)


def _record_keys(payload: Mapping[str, object]) -> None:
    allowed = {
        frozenset(OBSERVATION_RECORD_KEYS),
        frozenset((*OBSERVATION_RECORD_KEYS, "pixels")),
    }
    if frozenset(payload) not in allowed:
        raise VPMValidationError("observation record keys mismatch")


def _mapping(value: object, message: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise VPMValidationError(message)
    return cast(Mapping[str, object], value)


def _sequence(value: object, message: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise VPMValidationError(message)
    return cast(Sequence[object], value)


def _string(payload: Mapping[str, object], key: str, message: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise VPMValidationError(message)
    return value


def _optional_string(
    payload: Mapping[str, object],
    key: str,
    message: str,
) -> str | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise VPMValidationError(message)
    return value


def _integer(payload: Mapping[str, object], key: str, message: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise VPMValidationError(message)
    return value


def _boolean(payload: Mapping[str, object], key: str, message: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise VPMValidationError(message)
    return value


def _string_tuple(value: object, message: str) -> tuple[str | None, ...]:
    items = _sequence(value, message)
    result: list[str | None] = []
    for item in items:
        result.append(None if item is None else _sha256(item, message))
    return tuple(result)


def _json_mapping(dto: CanonicalJsonDTO, message: str) -> Mapping[str, object]:
    return _mapping(dto.to_value(), message)


def _pixel_digest_from_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _pixel_digest_from_array(pixels: object) -> str:
    return _pixel_digest_from_bytes(
        np.ascontiguousarray(pixels, dtype=np.uint8).tobytes(order="C")
    )


def _provider_raw_digest(blob: MatrixBlob) -> str:
    payload = (
        b"zeromodel.image-observation.raw.v1\0"
        + canonical_json_bytes(list(blob.shape))
        + blob.data
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ProviderObservationDescriptorDTO:
    version: str
    raw_digest: str
    shape: tuple[int, ...]
    timestamp: str | None
    source_id: str
    metadata: CanonicalJsonDTO

    def __post_init__(self) -> None:
        if self.version != IMAGE_OBSERVATION_VERSION:
            raise VPMValidationError("unsupported provider observation version")
        _sha256(self.raw_digest, "provider observation raw digest is not sha256")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise VPMValidationError("provider observation shape mismatch")
        if not self.source_id:
            raise VPMValidationError("provider observation source mismatch")
        _json_mapping(self.metadata, "provider observation metadata mismatch")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> ProviderObservationDescriptorDTO:
        _require_keys(
            payload,
            PROVIDER_DESCRIPTOR_KEYS,
            "provider observation descriptor keys mismatch",
        )
        return cls(
            version=_string(
                payload, "version", "unsupported provider observation version"
            ),
            raw_digest=_sha256(
                payload["raw_digest"],
                "provider observation raw digest is not sha256",
            ),
            shape=tuple(
                int(str(item))
                for item in _sequence(
                    payload["shape"], "provider observation shape mismatch"
                )
            ),
            timestamp=_optional_string(
                payload,
                "timestamp",
                "provider observation timestamp mismatch",
            ),
            source_id=_string(
                payload,
                "source_id",
                "provider observation source mismatch",
            ),
            metadata=CanonicalJsonDTO.from_value(payload["metadata"]),
        )

    @property
    def descriptor_digest(self) -> str:
        return canonical_sha256(
            {
                "version": PROVIDER_OBSERVATION_BOUNDARY_VERSION,
                "descriptor": self.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "raw_digest": self.raw_digest,
            "shape": list(self.shape),
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "metadata": self.metadata.to_value(),
        }


@dataclass(frozen=True, slots=True)
class ObservationOperationDTO:
    index: int
    operation: str
    operation_version: str
    input_digests: tuple[str | None, ...]
    parameters: CanonicalJsonDTO
    parameter_digest: str
    output_digest: str | None
    operation_digest: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise VPMValidationError("observation operation index cannot be negative")
        if not self.operation or not self.operation_version:
            raise VPMValidationError("observation operation payload keys mismatch")
        if self.parameter_digest != canonical_sha256(self.parameters.to_value()):
            raise VPMValidationError("observation operation parameter digest mismatch")
        for digest in (*self.input_digests, self.output_digest):
            _optional_sha256(digest, "observation operation digest is not sha256")
        _sha256(self.operation_digest, "observation operation digest is not sha256")
        if canonical_sha256(self._payload_without_digest()) != self.operation_digest:
            raise VPMValidationError("observation operation digest mismatch")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ObservationOperationDTO:
        _require_keys(
            payload,
            OPERATION_KEYS,
            "observation operation payload keys mismatch",
        )
        return cls(
            index=_integer(
                payload,
                "index",
                "observation operation index cannot be negative",
            ),
            operation=_string(
                payload,
                "operation",
                "observation operation payload keys mismatch",
            ),
            operation_version=_string(
                payload,
                "operation_version",
                "observation operation payload keys mismatch",
            ),
            input_digests=_string_tuple(
                payload["input_digests"],
                "observation operation digest is not sha256",
            ),
            parameters=CanonicalJsonDTO.from_value(payload["parameters"]),
            parameter_digest=_sha256(
                payload["parameter_digest"],
                "observation operation parameter digest mismatch",
            ),
            output_digest=_optional_sha256(
                payload["output_digest"],
                "observation operation digest is not sha256",
            ),
            operation_digest=_sha256(
                payload["operation_digest"],
                "observation operation digest is not sha256",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_digest() | {
            "operation_digest": self.operation_digest
        }

    def _payload_without_digest(self) -> dict[str, object]:
        return {
            "index": self.index,
            "operation": self.operation,
            "operation_version": self.operation_version,
            "input_digests": list(self.input_digests),
            "parameters": self.parameters.to_value(),
            "parameter_digest": self.parameter_digest,
            "output_digest": self.output_digest,
        }


@dataclass(frozen=True, slots=True)
class ObservationOperationChainDTO:
    version: str
    operations: tuple[ObservationOperationDTO, ...]
    final_emitted_digest: str | None
    operation_chain_digest: str

    def __post_init__(self) -> None:
        if self.version != OBSERVATION_OPERATION_CHAIN_VERSION:
            raise VPMValidationError("unsupported observation operation chain version")
        if not self.operations:
            raise VPMValidationError("observation operation indexes are not contiguous")
        for expected, operation in enumerate(self.operations):
            if operation.index != expected:
                raise VPMValidationError(
                    "observation operation indexes are not contiguous"
                )
        _optional_sha256(
            self.final_emitted_digest,
            "observation operation chain final digest mismatch",
        )
        if self.operations[-1].output_digest != self.final_emitted_digest:
            raise VPMValidationError(
                "observation operation chain final digest mismatch"
            )
        if (
            canonical_sha256(self._payload_without_digest())
            != self.operation_chain_digest
        ):
            raise VPMValidationError("observation operation chain digest mismatch")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> ObservationOperationChainDTO:
        _require_keys(payload, OPERATION_CHAIN_KEYS, "operation chain keys mismatch")
        return cls(
            version=_string(
                payload,
                "version",
                "unsupported observation operation chain version",
            ),
            operations=tuple(
                ObservationOperationDTO.from_dict(
                    _mapping(item, "observation operation payload keys mismatch")
                )
                for item in _sequence(
                    payload["operations"],
                    "observation operation payload keys mismatch",
                )
            ),
            final_emitted_digest=_optional_sha256(
                payload["final_emitted_digest"],
                "observation operation chain final digest mismatch",
            ),
            operation_chain_digest=_sha256(
                payload["operation_chain_digest"],
                "observation operation chain digest mismatch",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_digest() | {
            "operation_chain_digest": self.operation_chain_digest
        }

    def _payload_without_digest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "operations": [operation.to_dict() for operation in self.operations],
            "final_emitted_digest": self.final_emitted_digest,
        }


@dataclass(frozen=True, slots=True)
class ObservationDTO:
    benchmark_version: str
    generator_version: str
    benchmark_seed_digest: str
    episode_plan_digest: str
    split: str
    episode_id: str
    clip_id: str
    frame_id: str
    sequence_number: int
    event_type: str
    family: str
    expected_disposition: str
    episode_family: str
    episode_disposition: str
    frame_disposition: str
    denominator_class: str
    expected_row: str | None
    expected_action: str | None
    actual_executed_action: str | None
    action_known: bool
    gap_declaration: CanonicalJsonDTO | None
    observation_pixel_digest: str | None
    matrix_blob_id: str | None
    provider_observation_descriptor: ProviderObservationDescriptorDTO | None
    provider_observation_digest: str | None
    operation_chain: ObservationOperationChainDTO
    metadata: CanonicalJsonDTO

    def __post_init__(self) -> None:
        self._validate_ids()
        if self.action_known != (self.actual_executed_action is not None):
            raise VPMValidationError("observation action_known mismatch")
        _sha256(self.benchmark_seed_digest, "observation seed digest mismatch")
        _sha256(self.episode_plan_digest, "observation episode plan digest mismatch")
        _optional_sha256(
            self.observation_pixel_digest,
            "observation pixel digest mismatch",
        )
        self._validate_materialization_contract()
        self._validate_provider_descriptor()
        _json_mapping(self.metadata, "observation metadata mismatch")

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
    ) -> MaterializedObservationDTO:
        _record_keys(record)
        metadata = dict(_mapping(record["metadata"], "observation metadata mismatch"))
        chain = ObservationOperationChainDTO.from_dict(
            _mapping(
                metadata.pop("observation_operation_chain", None),
                "observation operation chain mismatch",
            )
        )
        descriptor = _provider_descriptor_from_metadata(metadata)
        provider_digest = _optional_sha256(
            metadata.pop("provider_observation_digest", None),
            "provider observation digest mismatch",
        )
        pixel_digest = _optional_sha256(
            record["observation_pixel_digest"],
            "observation pixel digest mismatch",
        )
        blob = _blob_from_record(record, pixel_digest)
        dto = cls(
            benchmark_version=_string(
                record,
                "benchmark_version",
                "unsupported observation benchmark version",
            ),
            generator_version=_string(
                record,
                "generator_version",
                "unsupported observation generator version",
            ),
            benchmark_seed_digest=_sha256(
                metadata.get("seed_digest"),
                "observation seed digest mismatch",
            ),
            episode_plan_digest=_sha256(
                metadata.get("episode_plan_digest"),
                "observation episode plan digest mismatch",
            ),
            split=_string(record, "split", "observation split mismatch"),
            episode_id=_string(record, "episode_id", "observation split mismatch"),
            clip_id=_string(record, "clip_id", "observation clip id mismatch"),
            frame_id=_string(record, "frame_id", "observation frame id mismatch"),
            sequence_number=_integer(
                record,
                "sequence_number",
                "observation sequence number cannot be negative",
            ),
            event_type=_string(
                record, "event_type", "observation record keys mismatch"
            ),
            family=_string(record, "family", "observation record keys mismatch"),
            expected_disposition=_string(
                record,
                "expected_disposition",
                "observation record keys mismatch",
            ),
            episode_family=_string(
                record, "episode_family", "observation record keys mismatch"
            ),
            episode_disposition=_string(
                record, "episode_disposition", "observation record keys mismatch"
            ),
            frame_disposition=_string(
                record, "frame_disposition", "observation record keys mismatch"
            ),
            denominator_class=_string(
                record, "denominator_class", "observation record keys mismatch"
            ),
            expected_row=_optional_string(
                record, "expected_row", "observation record keys mismatch"
            ),
            expected_action=_optional_string(
                record, "expected_action", "observation record keys mismatch"
            ),
            actual_executed_action=_optional_string(
                record, "actual_executed_action", "observation record keys mismatch"
            ),
            action_known=_boolean(
                record, "action_known", "observation action_known mismatch"
            ),
            gap_declaration=_canonical_optional(record["gap_declaration"]),
            observation_pixel_digest=pixel_digest,
            matrix_blob_id=None if blob is None else blob.blob_id,
            provider_observation_descriptor=descriptor,
            provider_observation_digest=provider_digest,
            operation_chain=chain,
            metadata=CanonicalJsonDTO.from_value(metadata),
        )
        return MaterializedObservationDTO(observation=dto, matrix_blob=blob)

    def to_record(
        self,
        *,
        matrix_blob: MatrixBlob | None = None,
        include_pixels: bool = True,
    ) -> dict[str, object]:
        if include_pixels:
            self._validate_record_blob(matrix_blob)
        metadata = dict(_json_mapping(self.metadata, "observation metadata mismatch"))
        self._restore_structured_metadata(metadata)
        payload = {
            "benchmark_version": self.benchmark_version,
            "generator_version": self.generator_version,
            "split": self.split,
            "episode_id": self.episode_id,
            "clip_id": self.clip_id,
            "frame_id": self.frame_id,
            "sequence_number": self.sequence_number,
            "event_type": self.event_type,
            "family": self.family,
            "expected_disposition": self.expected_disposition,
            "episode_family": self.episode_family,
            "episode_disposition": self.episode_disposition,
            "frame_disposition": self.frame_disposition,
            "denominator_class": self.denominator_class,
            "expected_row": self.expected_row,
            "expected_action": self.expected_action,
            "actual_executed_action": self.actual_executed_action,
            "action_known": self.action_known,
            "gap_declaration": (
                None
                if self.gap_declaration is None
                else self.gap_declaration.to_value()
            ),
            "observation_pixel_digest": self.observation_pixel_digest,
            "metadata": metadata,
        }
        if include_pixels:
            payload["pixels"] = None if matrix_blob is None else matrix_blob.to_array()
        return payload

    @property
    def has_pixels(self) -> bool:
        return self.matrix_blob_id is not None

    def _validate_ids(self) -> None:
        if self.benchmark_version != BENCHMARK_VERSION:
            raise VPMValidationError("unsupported observation benchmark version")
        if self.generator_version != GENERATOR_VERSION:
            raise VPMValidationError("unsupported observation generator version")
        if self.split not in SPLITS:
            raise VPMValidationError("observation split mismatch")
        if not self.episode_id.startswith(f"{self.split}:"):
            raise VPMValidationError("observation split mismatch")
        if self.clip_id != f"{self.split}:{self.episode_id}:clip":
            raise VPMValidationError("observation clip id mismatch")
        frame_prefix = f"{self.split}:{self.episode_id}:frame-"
        if not self.frame_id.startswith(frame_prefix):
            raise VPMValidationError("observation frame id mismatch")
        if self.sequence_number < 0:
            raise VPMValidationError("observation sequence number cannot be negative")
        suffix = self.frame_id.removeprefix(frame_prefix)
        if not suffix.isdecimal():
            raise VPMValidationError("observation frame id mismatch")
        if int(suffix) != self.sequence_number and not self._is_reordered_frame_id(
            int(suffix)
        ):
            raise VPMValidationError("observation frame id mismatch")

    def _is_reordered_frame_id(self, frame_index: int) -> bool:
        metadata = _json_mapping(self.metadata, "observation metadata mismatch")
        return (
            metadata.get("original_frame_index") == frame_index
            and "materialized_order" in metadata
        )

    def _validate_materialization_contract(self) -> None:
        if self.split == "final" and (
            self.matrix_blob_id is not None or self.observation_pixel_digest is not None
        ):
            raise VPMValidationError(
                "final split observation materialization is prohibited"
            )
        if self.event_type == "gap_unknown":
            if (
                self.matrix_blob_id is not None
                or self.observation_pixel_digest is not None
                or self.provider_observation_descriptor is not None
                or self.provider_observation_digest is not None
            ):
                raise VPMValidationError("observation typed gap mismatch")
        elif self.matrix_blob_id is None or self.observation_pixel_digest is None:
            raise VPMValidationError("observation materialized pixel mismatch")
        if self.operation_chain.final_emitted_digest != self.observation_pixel_digest:
            raise VPMValidationError("observation operation chain mismatch")

    def _validate_provider_descriptor(self) -> None:
        if self.provider_observation_descriptor is None:
            if self.provider_observation_digest is not None:
                raise VPMValidationError("provider observation digest mismatch")
            return
        if (
            self.provider_observation_descriptor.descriptor_digest
            != self.provider_observation_digest
        ):
            raise VPMValidationError("provider observation digest mismatch")

    def _validate_record_blob(self, matrix_blob: MatrixBlob | None) -> None:
        if self.matrix_blob_id is None:
            if matrix_blob is not None:
                raise VPMValidationError("observation matrix blob mismatch")
            return
        if matrix_blob is None or matrix_blob.blob_id != self.matrix_blob_id:
            raise VPMValidationError("observation matrix blob mismatch")
        validate_observation_matrix_blob(self, matrix_blob)

    def _restore_structured_metadata(self, metadata: dict[str, object]) -> None:
        metadata["observation_operation_chain"] = self.operation_chain.to_dict()
        if self.provider_observation_descriptor is not None:
            metadata["provider_observation_descriptor"] = (
                self.provider_observation_descriptor.to_dict()
            )
            metadata["provider_observation_digest"] = self.provider_observation_digest


@dataclass(frozen=True, slots=True)
class MaterializedObservationDTO:
    observation: ObservationDTO
    matrix_blob: MatrixBlob | None

    def __post_init__(self) -> None:
        if self.observation.split == "final":
            raise VPMValidationError(
                "final split observation materialization is prohibited"
            )
        if self.matrix_blob is None:
            if self.observation.matrix_blob_id is not None:
                raise VPMValidationError("observation matrix blob mismatch")
            return
        if self.observation.matrix_blob_id != self.matrix_blob.blob_id:
            raise VPMValidationError("observation matrix blob mismatch")
        validate_observation_matrix_blob(self.observation, self.matrix_blob)

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
    ) -> MaterializedObservationDTO:
        return ObservationDTO.from_record(record)

    def to_record(self, *, include_pixels: bool = True) -> dict[str, object]:
        return self.observation.to_record(
            matrix_blob=self.matrix_blob,
            include_pixels=include_pixels,
        )


def validate_observation_matrix_blob(
    observation: ObservationDTO,
    blob: MatrixBlob,
) -> None:
    if blob.dtype != "uint8" or tuple(blob.shape) != FRAME_SHAPE:
        raise VPMValidationError(
            "observation matrix blob does not match declared pixels"
        )
    if _pixel_digest_from_bytes(blob.data) != observation.observation_pixel_digest:
        raise VPMValidationError(
            "observation matrix blob does not match declared pixels"
        )
    metadata = dict(blob.metadata)
    if metadata != {
        "kind": "video_action_set_frame_pixels",
        "pixel_digest": observation.observation_pixel_digest,
    }:
        raise VPMValidationError(
            "observation matrix blob does not match declared pixels"
        )
    descriptor = observation.provider_observation_descriptor
    if descriptor is None:
        raise VPMValidationError("provider observation descriptor mismatch")
    if descriptor.shape != tuple(
        blob.shape
    ) or descriptor.raw_digest != _provider_raw_digest(blob):
        raise VPMValidationError("provider observation descriptor mismatch")


def _provider_descriptor_from_metadata(
    metadata: dict[str, object],
) -> ProviderObservationDescriptorDTO | None:
    value = metadata.pop("provider_observation_descriptor", None)
    if value is None:
        return None
    return ProviderObservationDescriptorDTO.from_dict(
        _mapping(value, "provider observation descriptor keys mismatch")
    )


def _blob_from_record(
    record: Mapping[str, object],
    pixel_digest: str | None,
) -> MatrixBlob | None:
    pixels = record.get("pixels")
    if pixels is None:
        return None
    if pixel_digest is None or _pixel_digest_from_array(pixels) != pixel_digest:
        raise VPMValidationError("observation pixel digest mismatch")
    return MatrixBlob.from_array(
        np.ascontiguousarray(pixels, dtype=np.uint8),
        dtype="uint8",
        metadata={
            "kind": "video_action_set_frame_pixels",
            "pixel_digest": pixel_digest,
        },
    )


def _canonical_optional(value: object) -> CanonicalJsonDTO | None:
    if value is None:
        return None
    return CanonicalJsonDTO.from_value(value)


__all__ = [
    "MaterializedObservationDTO",
    "ObservationDTO",
    "ObservationOperationChainDTO",
    "ObservationOperationDTO",
    "ProviderObservationDescriptorDTO",
    "validate_observation_matrix_blob",
]

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.video.domains.video_action_set.contracts import (
    BENCHMARK_VERSION,
    GENERATOR_VERSION,
)
from zeromodel.video.domains.video_action_set.dto import CanonicalJsonDTO, SPLITS
from zeromodel.video.domains.video_action_set.observation_common import (
    boolean,
    integer,
    json_mapping,
    mapping,
    optional_sha256,
    optional_string,
    sha256,
    string,
)
from zeromodel.video.domains.video_action_set.observation_materialization import (
    MaterializedObservationDTO,
    blob_from_record,
    register_materialized_observation_loader,
    validate_observation_matrix_blob,
)
from zeromodel.video.domains.video_action_set.observation_provenance_dto import (
    ObservationOperationChainDTO,
    ObservationOperationDTO,
)
from zeromodel.video.domains.video_action_set.provider_observation_dto import (
    ProviderObservationDescriptorDTO,
)


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
    final_access_id: str | None = None

    def __post_init__(self) -> None:
        self._validate_ids()
        if self.action_known != (self.actual_executed_action is not None):
            raise VPMValidationError("observation action_known mismatch")
        sha256(self.benchmark_seed_digest, "observation seed digest mismatch")
        sha256(self.episode_plan_digest, "observation episode plan digest mismatch")
        optional_sha256(
            self.observation_pixel_digest,
            "observation pixel digest mismatch",
        )
        self._validate_materialization_contract()
        self._validate_provider_descriptor()
        json_mapping(self.metadata, "observation metadata mismatch")
        if self.final_access_id is not None and self.split != "final":
            raise VPMValidationError("final access id is only valid for final")

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        final_access_id: str | None = None,
    ) -> MaterializedObservationDTO:
        _record_keys(record)
        metadata = dict(mapping(record["metadata"], "observation metadata mismatch"))
        chain = ObservationOperationChainDTO.from_dict(
            mapping(
                metadata.pop("observation_operation_chain", None),
                "observation operation chain mismatch",
            )
        )
        descriptor = _provider_descriptor_from_metadata(metadata)
        provider_digest = optional_sha256(
            metadata.pop("provider_observation_digest", None),
            "provider observation digest mismatch",
        )
        pixel_digest = optional_sha256(
            record["observation_pixel_digest"],
            "observation pixel digest mismatch",
        )
        blob = blob_from_record(record, pixel_digest)
        dto = cls(
            benchmark_version=string(
                record,
                "benchmark_version",
                "unsupported observation benchmark version",
            ),
            generator_version=string(
                record,
                "generator_version",
                "unsupported observation generator version",
            ),
            benchmark_seed_digest=sha256(
                metadata.get("seed_digest"),
                "observation seed digest mismatch",
            ),
            episode_plan_digest=sha256(
                metadata.get("episode_plan_digest"),
                "observation episode plan digest mismatch",
            ),
            split=string(record, "split", "observation split mismatch"),
            episode_id=string(record, "episode_id", "observation split mismatch"),
            clip_id=string(record, "clip_id", "observation clip id mismatch"),
            frame_id=string(record, "frame_id", "observation frame id mismatch"),
            sequence_number=integer(
                record,
                "sequence_number",
                "observation sequence number cannot be negative",
            ),
            event_type=string(record, "event_type", "observation record keys mismatch"),
            family=string(record, "family", "observation record keys mismatch"),
            expected_disposition=string(
                record,
                "expected_disposition",
                "observation record keys mismatch",
            ),
            episode_family=string(
                record, "episode_family", "observation record keys mismatch"
            ),
            episode_disposition=string(
                record, "episode_disposition", "observation record keys mismatch"
            ),
            frame_disposition=string(
                record, "frame_disposition", "observation record keys mismatch"
            ),
            denominator_class=string(
                record, "denominator_class", "observation record keys mismatch"
            ),
            expected_row=optional_string(
                record, "expected_row", "observation record keys mismatch"
            ),
            expected_action=optional_string(
                record, "expected_action", "observation record keys mismatch"
            ),
            actual_executed_action=optional_string(
                record, "actual_executed_action", "observation record keys mismatch"
            ),
            action_known=boolean(
                record, "action_known", "observation action_known mismatch"
            ),
            gap_declaration=_canonical_optional(record["gap_declaration"]),
            observation_pixel_digest=pixel_digest,
            matrix_blob_id=None if blob is None else blob.blob_id,
            provider_observation_descriptor=descriptor,
            provider_observation_digest=provider_digest,
            operation_chain=chain,
            metadata=CanonicalJsonDTO.from_value(metadata),
            final_access_id=final_access_id,
        )
        return MaterializedObservationDTO(
            observation=dto,
            matrix_blob=blob,
            final_access_id=final_access_id,
        )

    def to_record(
        self,
        *,
        matrix_blob: MatrixBlob | None = None,
        include_pixels: bool = True,
    ) -> dict[str, object]:
        if include_pixels:
            self._validate_record_blob(matrix_blob)
        metadata = dict(json_mapping(self.metadata, "observation metadata mismatch"))
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
        metadata = json_mapping(self.metadata, "observation metadata mismatch")
        return (
            metadata.get("original_frame_index") == frame_index
            and "materialized_order" in metadata
        )

    def _validate_materialization_contract(self) -> None:
        if (
            self.split == "final"
            and self.final_access_id is None
            and (
                self.matrix_blob_id is not None
                or self.observation_pixel_digest is not None
            )
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


def _record_keys(payload: Mapping[str, object]) -> None:
    allowed = {
        frozenset(OBSERVATION_RECORD_KEYS),
        frozenset((*OBSERVATION_RECORD_KEYS, "pixels")),
    }
    if frozenset(payload) not in allowed:
        raise VPMValidationError("observation record keys mismatch")


def _provider_descriptor_from_metadata(
    metadata: dict[str, object],
) -> ProviderObservationDescriptorDTO | None:
    value = metadata.pop("provider_observation_descriptor", None)
    if value is None:
        return None
    return ProviderObservationDescriptorDTO.from_dict(
        mapping(value, "provider observation descriptor keys mismatch")
    )


def _canonical_optional(value: object) -> CanonicalJsonDTO | None:
    if value is None:
        return None
    return CanonicalJsonDTO.from_value(value)


register_materialized_observation_loader(ObservationDTO.from_record)


__all__ = [
    "MaterializedObservationDTO",
    "ObservationDTO",
    "ObservationOperationChainDTO",
    "ObservationOperationDTO",
    "ProviderObservationDescriptorDTO",
    "validate_observation_matrix_blob",
]

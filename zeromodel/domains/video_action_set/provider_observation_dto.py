from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ...artifact import VPMValidationError
from ...visual_address import IMAGE_OBSERVATION_VERSION
from .canonical_json import canonical_sha256
from .contracts import PROVIDER_OBSERVATION_BOUNDARY_VERSION
from .dto import CanonicalJsonDTO
from .observation_common import (
    json_mapping,
    optional_string,
    require_keys,
    sequence,
    sha256,
    string,
)


PROVIDER_DESCRIPTOR_KEYS = (
    "version",
    "raw_digest",
    "shape",
    "timestamp",
    "source_id",
    "metadata",
)


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
        sha256(self.raw_digest, "provider observation raw digest is not sha256")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise VPMValidationError("provider observation shape mismatch")
        if not self.source_id:
            raise VPMValidationError("provider observation source mismatch")
        json_mapping(self.metadata, "provider observation metadata mismatch")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> ProviderObservationDescriptorDTO:
        require_keys(
            payload,
            PROVIDER_DESCRIPTOR_KEYS,
            "provider observation descriptor keys mismatch",
        )
        return cls(
            version=string(
                payload, "version", "unsupported provider observation version"
            ),
            raw_digest=sha256(
                payload["raw_digest"],
                "provider observation raw digest is not sha256",
            ),
            shape=tuple(
                int(str(item))
                for item in sequence(
                    payload["shape"], "provider observation shape mismatch"
                )
            ),
            timestamp=optional_string(
                payload,
                "timestamp",
                "provider observation timestamp mismatch",
            ),
            source_id=string(
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


__all__ = ["ProviderObservationDescriptorDTO"]

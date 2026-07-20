from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ...artifact import VPMValidationError
from .canonical_json import canonical_sha256
from .contracts import OBSERVATION_OPERATION_CHAIN_VERSION
from .dto import CanonicalJsonDTO
from .observation_common import (
    integer,
    mapping,
    optional_sha256,
    require_keys,
    sequence,
    sha256,
    string,
    string_tuple,
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
            optional_sha256(digest, "observation operation digest is not sha256")
        sha256(self.operation_digest, "observation operation digest is not sha256")
        if canonical_sha256(self._payload_without_digest()) != self.operation_digest:
            raise VPMValidationError("observation operation digest mismatch")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ObservationOperationDTO:
        require_keys(
            payload,
            OPERATION_KEYS,
            "observation operation payload keys mismatch",
        )
        return cls(
            index=integer(
                payload,
                "index",
                "observation operation index cannot be negative",
            ),
            operation=string(
                payload,
                "operation",
                "observation operation payload keys mismatch",
            ),
            operation_version=string(
                payload,
                "operation_version",
                "observation operation payload keys mismatch",
            ),
            input_digests=string_tuple(
                payload["input_digests"],
                "observation operation digest is not sha256",
            ),
            parameters=CanonicalJsonDTO.from_value(payload["parameters"]),
            parameter_digest=sha256(
                payload["parameter_digest"],
                "observation operation parameter digest mismatch",
            ),
            output_digest=optional_sha256(
                payload["output_digest"],
                "observation operation digest is not sha256",
            ),
            operation_digest=sha256(
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
        optional_sha256(
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
        require_keys(payload, OPERATION_CHAIN_KEYS, "operation chain keys mismatch")
        return cls(
            version=string(
                payload,
                "version",
                "unsupported observation operation chain version",
            ),
            operations=tuple(
                ObservationOperationDTO.from_dict(
                    mapping(item, "observation operation payload keys mismatch")
                )
                for item in sequence(
                    payload["operations"],
                    "observation operation payload keys mismatch",
                )
            ),
            final_emitted_digest=optional_sha256(
                payload["final_emitted_digest"],
                "observation operation chain final digest mismatch",
            ),
            operation_chain_digest=sha256(
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


__all__ = ["ObservationOperationChainDTO", "ObservationOperationDTO"]
